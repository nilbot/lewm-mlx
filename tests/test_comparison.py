import sys
import os
import math
from pathlib import Path
import numpy as np
import pytest

# Add reference repo to python path
ref_path = str(Path(__file__).parent.parent / "le-wm-ref")
if ref_path not in sys.path:
    sys.path.insert(0, ref_path)

import torch
import mlx.core as mx
import mlx.nn as nn

# Import PyTorch classes
import module as pt_module
import jepa as pt_jepa
from transformers import ViTConfig, ViTModel as PT_ViTModel

# Import MLX classes
import lewm_mlx.module as mx_module
import lewm_mlx.vit as mx_vit
import lewm_mlx.jepa as mx_jepa
from lewm_mlx.utils import load_pytorch_weights, resolve_mlx_path, set_nested_value

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, mx.array):
        return np.array(x)
    return x

def test_attention_equivalence():
    B, T, D = 2, 4, 128
    heads = 4
    dim_head = 32

    # Instantiate PyTorch Attention
    # Make sure we set training=False to avoid dropout during evaluation comparison
    pt_attn = pt_module.Attention(dim=D, heads=heads, dim_head=dim_head, dropout=0.0)
    pt_attn.eval()

    # Instantiate MLX Attention
    mx_attn = mx_module.Attention(dim=D, heads=heads, dim_head=dim_head, dropout=0.0)
    mx_attn.eval()

    # Sync weights
    # 1. Norm layer
    mx_attn.norm.weight = mx.array(to_np(pt_attn.norm.weight))
    mx_attn.norm.bias = mx.array(to_np(pt_attn.norm.bias))
    # 2. to_qkv projection
    mx_attn.to_qkv.weight = mx.array(to_np(pt_attn.to_qkv.weight))
    if not (heads == 1 and dim_head == D):
        mx_attn.to_out.layers[0].weight = mx.array(to_np(pt_attn.to_out[0].weight))
        mx_attn.to_out.layers[0].bias = mx.array(to_np(pt_attn.to_out[0].bias))

    # Inputs
    x_np = np.random.randn(B, T, D).astype(np.float32)
    x_pt = torch.tensor(x_np)
    x_mx = mx.array(x_np)

    # Forward
    with torch.no_grad():
        out_pt = pt_attn(x_pt, causal=True)
    out_mx = mx_attn(x_mx, causal=True)

    assert np.allclose(to_np(out_pt), to_np(out_mx), atol=1e-5)

def test_conditional_block_equivalence():
    B, T, D = 2, 4, 64
    heads = 2
    dim_head = 32
    mlp_dim = 128

    pt_block = pt_module.ConditionalBlock(dim=D, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.0)
    pt_block.eval()

    mx_block = mx_module.ConditionalBlock(dim=D, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.0)
    mx_block.eval()

    # Sync weights helper (using resolve_mlx_path for simplicity)
    mlx_params = mx_block.parameters()
    new_params = {}
    for k, val in pt_block.state_dict().items():
        val_np = val.detach().cpu().numpy()
        mx_val = mx.array(val_np)
        path = resolve_mlx_path(k, mlx_params)
        set_nested_value(new_params, path, mx_val)
    mx_block.update(new_params)

    # Inputs
    x_np = np.random.randn(B, T, D).astype(np.float32)
    c_np = np.random.randn(B, T, D).astype(np.float32)
    
    x_pt = torch.tensor(x_np)
    c_pt = torch.tensor(c_np)
    
    x_mx = mx.array(x_np)
    c_mx = mx.array(c_np)

    with torch.no_grad():
        out_pt = pt_block(x_pt, c_pt)
    out_mx = mx_block(x_mx, c_mx)

    assert np.allclose(to_np(out_pt), to_np(out_mx), atol=1e-5)

def test_sigreg_equivalence():
    T, B, D = 5, 3, 16
    knots = 17
    num_proj = 64

    # Force identical projection matrix A
    A_np = np.random.randn(D, num_proj).astype(np.float32)
    A_np = A_np / np.linalg.norm(A_np, axis=0, keepdims=True)

    # Instantiate PyTorch SIGReg and patch forward
    pt_sig = pt_module.SIGReg(knots=knots, num_proj=num_proj)
    def patched_pt_forward(proj):
        A = torch.tensor(A_np, device=proj.device)
        x_t = (proj @ A).unsqueeze(-1) * pt_sig.t
        err = (x_t.cos().mean(-3) - pt_sig.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ pt_sig.weights) * proj.size(-2)
        return statistic.mean()
    pt_sig.forward = patched_pt_forward

    # Subclass MLX SIGReg to inject fixed A
    class PatchedMXSIGReg(mx_module.SIGReg):
        def __call__(self, proj):
            A = mx.array(A_np)
            x_t = mx.expand_dims(mx.matmul(proj, A), -1) * self.t
            mean_cos = mx.mean(mx.cos(x_t), axis=1)
            mean_sin = mx.mean(mx.sin(x_t), axis=1)
            err = mx.square(mean_cos - self.phi) + mx.square(mean_sin)
            stat_sum = mx.sum(err * self.weights, axis=-1)
            statistic = stat_sum * proj.shape[1]
            return mx.mean(statistic)

    mx_sig = PatchedMXSIGReg(knots=knots, num_proj=num_proj)

    # Inputs
    proj_np = np.random.randn(T, B, D).astype(np.float32)
    proj_pt = torch.tensor(proj_np)
    proj_mx = mx.array(proj_np)

    out_pt = pt_sig(proj_pt)
    out_mx = mx_sig(proj_mx)

    assert np.allclose(to_np(out_pt), to_np(out_mx), atol=1e-5)

def test_embedder_equivalence():
    B, T, D_in = 2, 4, 10
    smoothed_dim = 16
    emb_dim = 32

    pt_embed = pt_module.Embedder(input_dim=D_in, smoothed_dim=smoothed_dim, emb_dim=emb_dim)
    pt_embed.eval()

    mx_embed = mx_module.Embedder(input_dim=D_in, smoothed_dim=smoothed_dim, emb_dim=emb_dim)
    mx_embed.eval()

    # Sync weights
    # PyTorch Conv1d weight: (smoothed_dim, D_in, 1) -> MLX weight: (smoothed_dim, 1, D_in)
    pt_conv_w = pt_embed.patch_embed.weight.detach().numpy()
    mx_embed.patch_embed.weight = mx.array(pt_conv_w.transpose(0, 2, 1))
    mx_embed.patch_embed.bias = mx.array(pt_embed.patch_embed.bias.detach().numpy())

    # Sequentials
    mlx_params = mx_embed.embed.parameters()
    new_params = {}
    for k, val in pt_embed.embed.state_dict().items():
        val_np = val.detach().cpu().numpy()
        mx_val = mx.array(val_np)
        path = resolve_mlx_path(k, mlx_params)
        set_nested_value(new_params, path, mx_val)
    mx_embed.embed.update(new_params)

    # Inputs
    x_np = np.random.randn(B, T, D_in).astype(np.float32)
    x_pt = torch.tensor(x_np)
    x_mx = mx.array(x_np)

    with torch.no_grad():
        out_pt = pt_embed(x_pt)
    out_mx = mx_embed(x_mx)

    assert np.allclose(to_np(out_pt), to_np(out_mx), atol=1e-5)

def test_vit_equivalence():
    # Instantiate configuration
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=192,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=256
    )
    pt_vit = PT_ViTModel(config, add_pooling_layer=False)
    pt_vit.eval()

    mx_vit_model = mx_vit.ViTModel(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=192,
        num_hidden_layers=2,
        num_attention_heads=3,
        intermediate_size=256
    )
    mx_vit_model.eval()

    # Load weights using utils
    # Save PyTorch weights temporarily
    temp_path = "/tmp/pt_vit_temp.pt"
    torch.save(pt_vit.state_dict(), temp_path)
    
    try:
        load_pytorch_weights(mx_vit_model, temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Image input NHWC (MLX convention) or NCHW (PyTorch convention)
    # Both architectures should accept NCHW (via auto-transpose)
    img_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    img_pt = torch.tensor(img_np)
    img_mx = mx.array(img_np)

    with torch.no_grad():
        out_pt = pt_vit(img_pt)
    out_mx = mx_vit_model(img_mx)

    assert np.allclose(to_np(out_pt.last_hidden_state), to_np(out_mx.last_hidden_state), atol=1e-4)

def test_jepa_rollout():
    # Setup miniature architecture configs
    hidden_size = 64
    num_frames = 3
    action_dim = 4

    # MLX sub-components
    mx_encoder = mx_vit.ViTModel(image_size=224, patch_size=16, hidden_size=hidden_size, num_hidden_layers=1, num_attention_heads=2, intermediate_size=128)
    mx_predictor = mx_module.ARPredictor(num_frames=num_frames, depth=1, heads=2, mlp_dim=128, input_dim=hidden_size, hidden_dim=hidden_size)
    mx_action_encoder = mx_module.Embedder(input_dim=action_dim, smoothed_dim=hidden_size, emb_dim=hidden_size)
    mx_projector = mx_module.MLP(input_dim=hidden_size, hidden_dim=128, output_dim=hidden_size)
    mx_pred_proj = mx_module.MLP(input_dim=hidden_size, hidden_dim=128, output_dim=hidden_size)

    mx_jepa_model = mx_jepa.JEPA(
        encoder=mx_encoder,
        predictor=mx_predictor,
        action_encoder=mx_action_encoder,
        projector=mx_projector,
        pred_proj=mx_pred_proj
    )
    mx_jepa_model.eval()

    # Construct PyTorch equivalent models
    # Patch ViTConfig
    config = ViTConfig(image_size=224, patch_size=16, hidden_size=hidden_size, num_hidden_layers=1, num_attention_heads=2, intermediate_size=128)
    pt_encoder = PT_ViTModel(config, add_pooling_layer=False)
    pt_predictor = pt_module.ARPredictor(num_frames=num_frames, depth=1, heads=2, mlp_dim=128, input_dim=hidden_size, hidden_dim=hidden_size)
    pt_action_encoder = pt_module.Embedder(input_dim=action_dim, smoothed_dim=hidden_size, emb_dim=hidden_size)
    pt_projector = pt_module.MLP(input_dim=hidden_size, hidden_dim=128, output_dim=hidden_size, norm_fn=None)
    pt_pred_proj = pt_module.MLP(input_dim=hidden_size, hidden_dim=128, output_dim=hidden_size, norm_fn=None)

    pt_jepa_model = pt_jepa.JEPA(
        encoder=pt_encoder,
        predictor=pt_predictor,
        action_encoder=pt_action_encoder,
        projector=pt_projector,
        pred_proj=pt_pred_proj
    )
    pt_jepa_model.eval()

    # Sync weights
    temp_path = "/tmp/pt_jepa_temp.pt"
    torch.save(pt_jepa_model.state_dict(), temp_path)
    try:
        load_pytorch_weights(mx_jepa_model, temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Rollout Inputs
    # B=2, S=3, H=3, C=3, H_img=224, W_img=224
    pixels_np = np.random.randn(2, 3, 3, 3, 224, 224).astype(np.float32)
    actions_np = np.random.randn(2, 3, 5, action_dim).astype(np.float32) # T = 5, n_steps = T - H = 2

    # In PyTorch:
    info_pt = {"pixels": torch.tensor(pixels_np)}
    actions_pt = torch.tensor(actions_np)

    with torch.no_grad():
        out_info_pt = pt_jepa_model.rollout(info_pt, actions_pt, history_size=num_frames)

    # In MLX:
    info_mx = {"pixels": mx.array(pixels_np)}
    actions_mx = mx.array(actions_np)
    out_info_mx = mx_jepa_model.rollout(info_mx, actions_mx, history_size=num_frames)

    assert np.allclose(to_np(out_info_pt["predicted_emb"]), to_np(out_info_mx["predicted_emb"]), atol=1e-4)
