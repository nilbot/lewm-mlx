import math
import mlx.core as mx
import mlx.nn as nn

def interpolate_pos_encoding_grid(pos_embed, H_new, W_new):
    """Bilinear interpolation of position embeddings to new grid size."""
    # pos_embed: (1, N_old + 1, D)
    D = pos_embed.shape[-1]
    cls_token = pos_embed[:, :1, :]  # (1, 1, D)
    patch_pos = pos_embed[0, 1:, :]  # (N_old, D)
    
    N_old = patch_pos.shape[0]
    H_old = int(math.isqrt(N_old))
    W_old = H_old
    
    if H_old * W_old != N_old:
        return pos_embed
        
    grid = patch_pos.reshape(H_old, W_old, D)
    
    # Generate coordinates
    ys = mx.linspace(0, H_old - 1, H_new)
    xs = mx.linspace(0, W_old - 1, W_new)
    
    x_grid, y_grid = mx.meshgrid(xs, ys)  # (H_new, W_new)
    
    # Compute bounds
    y0 = mx.floor(y_grid).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H_old - 1)
    x0 = mx.floor(x_grid).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, W_old - 1)
    
    # Compute weights
    wa = mx.expand_dims((x1 - x_grid) * (y1 - y_grid), -1)
    wb = mx.expand_dims((x1 - x_grid) * (y_grid - y0), -1)
    wc = mx.expand_dims((x_grid - x0) * (y1 - y_grid), -1)
    wd = mx.expand_dims((x_grid - x0) * (y_grid - y0), -1)
    
    # Gather values
    Ia = grid[y0, x0]
    Ib = grid[y1, x0]
    Ic = grid[y0, x1]
    Id = grid[y1, x1]
    
    interpolated = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (H_new, W_new, D)
    interpolated = interpolated.reshape(1, H_new * W_new, D)
    
    return mx.concatenate([cls_token, interpolated], axis=1)

class ViTAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = self.attention_head_size**-0.5

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size)
        self.o_proj = nn.Linear(self.all_head_size, hidden_size)

    def __call__(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, T, _ = q.shape
        q = q.reshape(B, T, self.num_attention_heads, self.attention_head_size).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_attention_heads, self.attention_head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_attention_heads, self.attention_head_size).transpose(0, 2, 1, 3)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)

class ViTMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))

class ViTLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = ViTAttention(hidden_size, num_attention_heads)
        self.mlp = ViTMLP(hidden_size, intermediate_size)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=1e-12)

    def __call__(self, x):
        x = x + self.attention(self.layernorm_before(x))
        x = x + self.mlp(self.layernorm_after(x))
        return x

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def __call__(self, x):
        x = self.projection(x)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)
        return x

class ViTEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()
        self.patch_embeddings = ViTPatchEmbeddings(image_size, patch_size, num_channels, hidden_size)
        
        self.num_patches = (image_size // patch_size) ** 2
        self.cls_token = mx.random.normal(shape=(1, 1, hidden_size))
        self.position_embeddings = mx.random.normal(shape=(1, self.num_patches + 1, hidden_size))

    def __call__(self, x, interpolate_pos_encoding=True):
        B, H, W, C = x.shape
        patch_size = self.patch_embeddings.patch_size
        H_new = H // patch_size
        W_new = W // patch_size
        num_patches_new = H_new * W_new

        x = self.patch_embeddings(x)
        
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.cls_token.shape[-1]))
        x = mx.concatenate([cls_tokens, x], axis=1)
        
        if interpolate_pos_encoding and num_patches_new != self.num_patches:
            pos_embed = interpolate_pos_encoding_grid(self.position_embeddings, H_new, W_new)
        else:
            pos_embed = self.position_embeddings
            
        T = x.shape[1]
        x = x + pos_embed[:, :T]
        return x

class ViTModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state

class ViTModel(nn.Module):
    """Vision Transformer in MLX matching HF architecture (>= v4.57)."""

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        **kwargs
    ):
        super().__init__()
        self.embeddings = ViTEmbeddings(image_size, patch_size, num_channels, hidden_size)
        self.layers = [
            ViTLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ]
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def __call__(self, x, interpolate_pos_encoding=True):
        if x.ndim == 4 and x.shape[1] == 3:
            x = x.transpose(0, 2, 3, 1)

        x = self.embeddings(x, interpolate_pos_encoding=interpolate_pos_encoding)
        for layer in self.layers:
            x = layer(x)
        x = self.layernorm(x)
        return ViTModelOutput(x)
