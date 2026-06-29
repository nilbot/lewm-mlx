import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

def resolve_mlx_path(pt_key, mlx_params):
    """Resolve a flat PyTorch key to a nested path in the MLX parameters dict."""
    parts = pt_key.split('.')
    path = []
    curr = mlx_params
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # Check if it is a list index following a Sequential or List
        if isinstance(curr, dict) and 'layers' in curr and part.isdigit():
            path.append('layers')
            idx = int(part)
            path.append(idx)
            curr = curr['layers'][idx]
            i += 1
            continue
            
        # General list handling (like 'layer' inside 'encoder', or 'layers' in 'transformer')
        if isinstance(curr, dict) and part in curr:
            val = curr[part]
            if isinstance(val, list):
                path.append(part)
                idx = int(parts[i+1])
                path.append(idx)
                curr = val[idx]
                i += 2
                continue
            else:
                path.append(part)
                curr = val
                i += 1
                continue
        
        # Check if it is inside a Sequential that we didn't specify 'layers' for in PyTorch key
        # e.g., PyTorch key was 'projector.net.0.weight' -> part is 'net', next is '0'.
        if isinstance(curr, dict) and part in curr and isinstance(curr[part], dict) and 'layers' in curr[part] and i+1 < len(parts) and parts[i+1].isdigit():
            path.append(part)
            path.append('layers')
            idx = int(parts[i+1])
            path.append(idx)
            curr = curr[part]['layers'][idx]
            i += 2
            continue
            
        # Fallback
        path.append(part)
        i += 1
        
    return path

def set_nested_value(d, path, val):
    """Sets a value in a nested dictionary/list structure based on path."""
    curr = d
    for i, key in enumerate(path[:-1]):
        next_key = path[i + 1]
        if isinstance(next_key, int):
            if key not in curr:
                curr[key] = []
            while len(curr[key]) <= next_key:
                curr[key].append({})
            curr = curr[key][next_key]
        else:
            if isinstance(key, int):
                pass
            else:
                if key not in curr:
                    curr[key] = {}
                curr = curr[key]
                
    curr[path[-1]] = val

def load_pytorch_weights(model: nn.Module, torch_ckpt_path: str):
    """Loads PyTorch state_dict or Lightning checkpoint weights into an MLX model."""
    import torch
    
    # Load state dict
    ckpt = torch.load(torch_ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
        
    # Get target MLX model parameters structure
    mlx_params = model.parameters()
    
    # New parameter dictionary to construct
    new_params = {}
    
    for pt_key, val in state_dict.items():
        # Clean PyTorch key (strip model. prefix if it exists)
        clean_key = pt_key
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]
            
        # Skip keys that are not in our MLX model
        # e.g., sigreg or lightning internal keys
        if clean_key.startswith("sigreg") or clean_key.startswith("criterion"):
            continue
            
        # Convert PyTorch tensor to numpy
        val_np = val.detach().cpu().numpy()
        
        # Transpose weights based on layer types
        if clean_key.endswith(".weight"):
            if val_np.ndim == 4:
                # Conv2d: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
                val_np = val_np.transpose(0, 2, 3, 1)
            elif val_np.ndim == 3:
                # Conv1d: PyTorch [out, in, L] -> MLX [out, L, in]
                val_np = val_np.transpose(0, 2, 1)
                
        # Convert to MLX array
        mx_val = mx.array(val_np)
        
        # Resolve nested path
        try:
            path = resolve_mlx_path(clean_key, mlx_params)
            set_nested_value(new_params, path, mx_val)
        except Exception as e:
            # Output warning but continue
            print(f"Warning: could not map key {pt_key} (clean: {clean_key}): {e}")
            
    # Load parameters into the MLX model
    model.update(new_params)
    return model
