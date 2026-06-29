import math
import mlx.core as mx
import mlx.nn as nn

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = mx.linspace(0, 3, knots)
        dt = 3.0 / (knots - 1)
        weights_data = [2.0 * dt] * knots
        weights_data[0] = dt
        weights_data[-1] = dt
        weights = mx.array(weights_data)
        window = mx.exp(-mx.square(t) / 2.0)
        self.t = t
        self.phi = window
        self.weights = weights * window

    def __call__(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = mx.random.normal(shape=(proj.shape[-1], self.num_proj))
        A = A / mx.linalg.norm(A, ord=2, axis=0, keepdims=True)
        # compute the epps-pulley statistic
        x_t = mx.expand_dims(mx.matmul(proj, A), -1) * self.t
        
        # mean(-3) is over the B dimension (axis 1)
        mean_cos = mx.mean(mx.cos(x_t), axis=1) # (T, num_proj, knots)
        mean_sin = mx.mean(mx.sin(x_t), axis=1) # (T, num_proj, knots)
        
        err = mx.square(mean_cos - self.phi) + mx.square(mean_sin) # (T, num_proj, knots)
        
        # err @ weights
        stat_sum = mx.sum(err * self.weights, axis=-1) # (T, num_proj)
        statistic = stat_sum * proj.shape[1] # B is proj.shape[1]
        
        return mx.mean(statistic) # average over projections and time

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def __call__(self, x):
        return self.net(x)

class Identity(nn.Module):
    def __call__(self, x):
        return x

class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout_rate = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else Identity()
        )

    def __call__(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        qkv = self.to_qkv(x)
        
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        B, T, _ = q.shape
        q = q.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        if causal:
            mask = mx.triu(mx.full((T, T), -float("inf")), k=1)
            scores = scores + mask
            
        attn = mx.softmax(scores, axis=-1)
        
        if self.training and self.dropout_rate > 0.0:
            attn = nn.Dropout(self.dropout_rate)(attn)
            
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out(out)

class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        # Zero init the modulation weights and biases
        linear_layer = self.adaLN_modulation.layers[-1]
        linear_layer.weight = mx.zeros_like(linear_layer.weight)
        linear_layer.bias = mx.zeros_like(linear_layer.bias)

    def __call__(self, x, c):
        chunks = mx.split(self.adaLN_modulation(c), 6, axis=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else Identity()
        )

        self.layers = [
            block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ]

    def __call__(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            if isinstance(block, Block):
                x = block(x)
            else:
                x = block(x, c)
                
        x = self.norm(x)
        x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def __call__(self, x):
        """
        x: (B, T, D)
        """
        x = x.astype(mx.float32)
        x = self.patch_embed(x)
        x = self.embed(x)
        return x

class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=None,
        act_fn=nn.GELU,
    ):
        super().__init__()
        
        norm_layer = Identity()
        if norm_fn is not None:
            if "BatchNorm" in str(norm_fn):
                norm_layer = nn.BatchNorm(hidden_dim)
            elif "LayerNorm" in str(norm_fn):
                norm_layer = nn.LayerNorm(hidden_dim)
            else:
                try:
                    norm_layer = norm_fn(hidden_dim)
                except:
                    norm_layer = Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def __call__(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)

class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = mx.random.normal(shape=(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def __call__(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.shape[1]
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
