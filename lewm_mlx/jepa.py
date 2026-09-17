from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from lewm_mlx.module import Identity


class JEPA(nn.Module):
    """Joint-Embedding Predictive Architecture (JEPA) in MLX."""

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector if projector is not None else Identity()
        self.pred_proj = pred_proj if pred_proj is not None else Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info["pixels"]
        if not isinstance(pixels, mx.array):
            pixels = mx.array(pixels)
        pixels = pixels.astype(mx.float32)

        B = pixels.shape[0]
        T = pixels.shape[1]

        if pixels.ndim == 5:
            # Handle NCHW -> NHWC transposition
            if pixels.shape[2] == 3:  # C is index 2
                pixels = pixels.transpose(0, 1, 3, 4, 2)
            pixels_flat = pixels.reshape(B * T, *pixels.shape[2:])
        else:
            pixels_flat = pixels

        output = self.encoder(pixels_flat, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0, :]

        emb = self.projector(pixels_emb)
        info["emb"] = emb.reshape(B, T, -1)

        if "action" in info:
            action = info["action"]
            if not isinstance(action, mx.array):
                action = mx.array(action)
            info["act_emb"] = self.action_encoder(action)

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        B, T, D = emb.shape
        preds = self.predictor(emb, act_emb)
        preds_flat = preds.reshape(B * T, -1)
        preds_proj = self.pred_proj(preds_flat)
        return preds_proj.reshape(B, T, -1)

    def rollout(self, info, action_sequence, history_size: Optional[int] = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W) or (B, S, T, H, W, C)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """
        assert "pixels" in info, "pixels not in info_dict"

        pixels = info["pixels"]
        if not isinstance(pixels, mx.array):
            pixels = mx.array(pixels)

        # Context size H is the size of pixels' 3rd dimension
        H = pixels.shape[2]
        if history_size is None:
            history_size = H
        B, S, T = action_sequence.shape[:3]

        # Split action sequence: act_0 is history actions, act_future is future actions
        # act_0: (B, S, H, action_dim), act_future: (B, S, T - H, action_dim)
        act_0 = action_sequence[:, :, :H, :]
        act_future = action_sequence[:, :, H:, :]
        n_steps = T - H

        # copy and encode initial info dict
        _init = {}
        for k, v in info.items():
            if isinstance(v, mx.array):
                _init[k] = v[:, 0]
            elif hasattr(v, "__len__") and not isinstance(v, str):
                try:
                    arr = mx.array(v)
                    _init[k] = arr[:, 0]
                except Exception:
                    pass

        _init = self.encode(_init)

        # Expand _init["emb"] (B, H, D) to (B, S, H, D)
        init_emb = _init["emb"]
        emb = mx.broadcast_to(
            mx.expand_dims(init_emb, 1), (B, S, H, init_emb.shape[-1])
        )
        info["emb"] = emb

        # Flatten batch and sample dimensions: (B * S, H, D)
        emb = emb.reshape(B * S, H, -1)
        act = act_0.reshape(B * S, H, -1)
        act_future = act_future.reshape(B * S, n_steps, -1)

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:, :]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:, :]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:, :]  # (BS, 1, D)
            emb = mx.concatenate([emb, pred_emb], axis=1)  # (BS, H+t+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = mx.concatenate([act, next_act], axis=1)  # (BS, H+t+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:, :]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:, :]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:, :]  # (BS, 1, D)
        emb = mx.concatenate([emb, pred_emb], axis=1)

        # unflatten batch and sample dimensions
        pred_rollout = emb.reshape(B, S, -1, emb.shape[-1])
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, T_pred, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T_goal, dim) or (B, T_goal, dim)

        if goal_emb.ndim == 3:
            goal_emb = mx.expand_dims(goal_emb, 1)  # (B, 1, T_goal, dim)

        # Compute MSE loss on the last step: sum of squared differences
        last_goal = goal_emb[:, :, -1:, :]  # (B, 1 or S, 1, dim)
        last_pred = pred_emb[:, :, -1:, :]  # (B, S, 1, dim)

        diff_sq = mx.square(last_pred - last_goal)
        cost = mx.sum(diff_sq, axis=(2, 3))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: mx.array):
        """Compute the cost of action candidates given an info dict with goal and initial state."""
        assert "goal" in info_dict, "goal not in info_dict"

        # Convert everything in info_dict to mx.array if possible
        for k in list(info_dict.keys()):
            if not isinstance(info_dict[k], mx.array):
                try:
                    info_dict[k] = mx.array(info_dict[k])
                except Exception:
                    pass

        # extract goal dict
        goal = {}
        for k, v in info_dict.items():
            if isinstance(v, mx.array):
                goal[k] = v[:, 0]

        goal["pixels"] = goal["goal"]

        # Rename goal_xxx keys to xxx
        for k in list(info_dict.keys()):
            if k.startswith("goal_"):
                new_key = k[len("goal_") :]
                if k in goal:
                    goal[new_key] = goal.pop(k)

        if "action" in goal:
            goal.pop("action")

        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        return cost
