from typing import List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtrl.agent.components import base as base_component
from mtrl.utils.types import TensorType

from .pa_math import DRegCfg, soft_ce, two_hot_inv, symlog

try:
    Mish = nn.Mish  # Use PyTorch built-in if available
except AttributeError:
    class Mish(nn.Module):
        def __init__(self, inplace: bool = False):
            super().__init__()
            self.inplace = inplace  # Keep interface for compatibility
        def forward(self, x):
            # Mish(x) = x * tanh(softplus(x))
            return x * torch.tanh(F.softplus(x))


# ----------------- small blocks -----------------
class SimNorm(nn.Module):
    """Simplicial normalization: reshape [..., K*G] -> [..., K, G], softmax over G."""
    def __init__(self, simnorm_dim: int):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Module):
    """Linear (+ optional Dropout) + LayerNorm + Activation (default Mish)."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0., act=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        if act is None:
            act = Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout and dropout > 1e-8 else None

    def forward(self, x):
        x = self.linear(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.ln(x)
        return self.act(x)

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.linear.in_features}, " \
               f"out_features={self.linear.out_features}, " \
               f"bias={self.linear.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"


def mlp(in_dim: int, mlp_dims: List[int], out_dim: int, act=None, dropout: float = 0.):
    """
    TD-MPC2 style MLP block.
    [SERGEY NOTE]: Modified to apply Dropout on all hidden layers to prevent overfitting during offline training.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        # [Modified]: removed the `if i == 0 else 0.0` restriction
        layers.append(NormedLinear(dims[i], dims[i+1], dropout=dropout))
    
    # Last layer typically has no LayerNorm or Activation (if it's a linear projection)
    # But depends on the act parameter
    layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# ----------------- submodules -----------------
class PredictiveAdapterEncoder(base_component.Component):
    """z = h([s, e])"""
    def __init__(self, state_dim: int, task_encoding_dim: int, latent_dim: int,
                 num_enc_layers: int = 2, enc_dim: int = 256, simnorm_dim: int = 8, dropout: float = 0.):
        super().__init__()
        assert latent_dim % simnorm_dim == 0, \
            f"latent_dim ({latent_dim}) must be divisible by simnorm_dim ({simnorm_dim})"
        input_dim = state_dim + task_encoding_dim
        hidden_dims = [enc_dim] * max(num_enc_layers - 1, 1)
        self.encoder = mlp(input_dim, hidden_dims, latent_dim, act=SimNorm(simnorm_dim), dropout=dropout)
        self.expected_in = state_dim + task_encoding_dim

    def forward(self, state: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([state, task_encoding], dim=-1)
        if not hasattr(self, "_dbg_printed"):
            print(f"[PA-ENC/DEBUG] state{tuple(state.shape)} + task{tuple(task_encoding.shape)} -> x{tuple(x.shape)}")
            assert x.shape[-1] == self.expected_in, \
                f"encoder input {x.shape[-1]} != expected_in {self.expected_in}"
            self._dbg_printed = True
        return self.encoder(x)


class PredictiveAdapterDynamics(base_component.Component):
    """z' = d([z, a, e])"""
    def __init__(self, latent_dim: int, action_dim: int, task_encoding_dim: int,
                 mlp_dim: int = 512, simnorm_dim: int = 8, dropout: float = 0.):
        super().__init__()
        assert latent_dim % simnorm_dim == 0, \
            f"latent_dim ({latent_dim}) must be divisible by simnorm_dim ({simnorm_dim})"
        input_dim = latent_dim + action_dim + task_encoding_dim
        self.dynamics = mlp(input_dim, [mlp_dim, mlp_dim], latent_dim, act=SimNorm(simnorm_dim), dropout=dropout)

    def forward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([latent, action, task_encoding], dim=-1)
        return self.dynamics(x)


class PredictiveAdapterReward(base_component.Component):
    """r̂ = R([z, a, e]) → logits (two-hot) or scalar (reward_bins=1)"""
    def __init__(self, latent_dim: int, action_dim: int, task_encoding_dim: int,
                 mlp_dim: int = 512, reward_bins: int = 101, dropout: float = 0.):
        super().__init__()
        input_dim = latent_dim + action_dim + task_encoding_dim
        self.reward_bins = reward_bins
        self.reward_net = mlp(input_dim, [mlp_dim, mlp_dim], max(reward_bins, 1), dropout=dropout)

    def forward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([latent, action, task_encoding], dim=-1)
        return self.reward_net(x)


# ----------------- predictive adapter -----------------
class PredictiveAdapter(base_component.Component):
    """
    TD-MPC2 style implicit world model (slightly modified version):
      - z = h([s,e])
      - z' = d([z,a,e])
      - r̂ = R([z,a,e]) (two-hot discrete regression; degenerates to scalar regression when reward_bins=1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        task_encoding_dim: int,
        # Read hyperparameters from cfg; default values given below
        latent_dim: int = 512,
        num_enc_layers: int = 2,
        enc_dim: int = 256,
        mlp_dim: int = 512,
        reward_bins: int = 101,
        simnorm_dim: int = 8,
        reward_bounds: Tuple[float, float] = (-10.0, 10.0),
        dropout: float = 0.0,
        rollout_horizon: int = 1,
        rollout_discount: float = 1.0,
        dynamics_loss_weight: float = 200.0,
        reward_loss_weight: float = 1.0,
        delta_state_loss_weight: float = 0.0,
        delta_state_similarity_temperature: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.reward_bins = reward_bins
        self.task_encoding_dim = task_encoding_dim
        self.rollout_horizon = max(int(rollout_horizon), 1)
        self.rollout_discount = float(rollout_discount)
        self.dynamics_loss_weight = float(dynamics_loss_weight)
        self.reward_loss_weight = float(reward_loss_weight)
        self.delta_state_loss_weight = float(delta_state_loss_weight)
        self.delta_state_similarity_temperature = float(delta_state_similarity_temperature)

        # modules
        self.encoder = PredictiveAdapterEncoder(
            state_dim=state_dim,
            task_encoding_dim=task_encoding_dim,
            latent_dim=latent_dim,
            num_enc_layers=num_enc_layers,
            enc_dim=enc_dim,
            simnorm_dim=simnorm_dim,
            dropout=dropout,
        )
        self.dynamics = PredictiveAdapterDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            task_encoding_dim=task_encoding_dim,
            mlp_dim=mlp_dim,
            simnorm_dim=simnorm_dim,
            dropout=dropout,
        )
        self.reward_head = PredictiveAdapterReward(
            latent_dim=latent_dim,
            action_dim=action_dim,
            task_encoding_dim=task_encoding_dim,
            mlp_dim=mlp_dim,
            reward_bins=reward_bins,
            dropout=dropout,
        )

        self.expected_in = state_dim + task_encoding_dim  # Record expected input dimension
        # Find the first Linear layer in the encoder (whether nn.Linear or wrapped in NormedLinear)
        first_linear_in = None
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                first_linear_in = m.in_features
                break
        print(f"[PA-ENC/DEBUG] expect_in={self.expected_in}, first_linear_in={first_linear_in}, latent_dim={latent_dim}")

        # dreg (two-hot discrete regression config)
        if self.reward_bins > 1:
            rmin, rmax = reward_bounds
            vmin = symlog(torch.tensor([rmin])).item()
            vmax = symlog(torch.tensor([rmax])).item()
            self.dreg_cfg = DRegCfg(num_bins=self.reward_bins, vmin=vmin, vmax=vmax)
        else:
            self.dreg_cfg = None

    # ---------- public API ----------
    def encode(self, state: TensorType, task_encoding: TensorType) -> TensorType:
        return self.encoder(state, task_encoding)

    def predict_next_latent(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        return self.dynamics(latent, action, task_encoding)

    def reward_logits(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        return self.reward_head(latent, action, task_encoding)

    def predict_reward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        """Return continuous scalar reward ([B,1]), with two-hot inverse transform."""
        logits = self.reward_logits(latent, action, task_encoding)
        if self.reward_bins <= 1:
            # Scalar regression (MSE)
            return logits if logits.ndim == 2 else logits.unsqueeze(-1)
        return two_hot_inv(
            logits,
            self.dreg_cfg.num_bins,
            self.dreg_cfg.vmin,
            self.dreg_cfg.vmax,
        )  # [B,1]

    def forward(self, state: TensorType, action: TensorType, task_encoding: TensorType):
        z = self.encode(state, task_encoding)
        z_next = self.predict_next_latent(z, action, task_encoding)
        r_hat = self.predict_reward(z, action, task_encoding)
        return z, z_next, r_hat

    @torch.no_grad()
    def latent_rollout(self, z0: TensorType, actions: TensorType, task_encoding: TensorType) -> TensorType:
        """Multi-step latent rollout (for diagnostics)"""
        if actions.dim() == 3:
            actions = actions.transpose(0, 1)
        T = actions.shape[0]
        B = z0.shape[0]
        z = z0
        traj = [z0]
        for t in range(T):
            z = self.predict_next_latent(z, actions[t].view(B, -1), task_encoding)
            traj.append(z)
        return torch.stack(traj, dim=0)  # [T+1, B, latent_dim]

    # ---------- loss ----------
    def compute_loss(self, state: TensorType, action: TensorType, next_state: TensorType,
                     reward: TensorType, task_encoding: TensorType, valid_mask: Optional[TensorType] = None):
        """Return (dyn_loss, rew_loss, total_loss)"""
        if action.dim() == 2:
            # 1) encode
            z = self.encode(state, task_encoding)
            with torch.no_grad():
                z_next_tgt = self.encode(next_state, task_encoding)

            # 2) Dynamics consistency
            z_next_pred = self.predict_next_latent(z, action, task_encoding)
            dyn_loss = F.mse_loss(z_next_pred, z_next_tgt)

            # 3) Reward two-hot (or MSE when reward_bins=1)
            logits = self.reward_logits(z, action, task_encoding)
            if self.reward_bins > 1:
                r = reward if reward.ndim == 2 else reward.unsqueeze(-1)
                rew_loss = soft_ce(logits, r, self.dreg_cfg).mean()
            else:
                r = reward if reward.ndim == 2 else reward.unsqueeze(-1)
                rew_loss = F.mse_loss(logits, r)
        else:
            dyn_loss, rew_loss = self._compute_multi_step_loss(
                state=state,
                actions=action,
                next_states=next_state,
                rewards=reward,
                task_encoding=task_encoding,
                valid_mask=valid_mask,
            )

        total = self.dynamics_loss_weight * dyn_loss + self.reward_loss_weight * rew_loss
        return dyn_loss, rew_loss, total

    def _compute_multi_step_loss(
        self,
        state: TensorType,
        actions: TensorType,
        next_states: TensorType,
        rewards: TensorType,
        task_encoding: TensorType,
        valid_mask: Optional[TensorType] = None,
    ) -> Tuple[TensorType, TensorType]:
        """Compute a horizon-H rollout loss in latent space."""
        batch_size, horizon = actions.shape[0], actions.shape[1]
        device = state.device

        if valid_mask is None:
            valid_mask = torch.ones(batch_size, horizon, 1, device=device, dtype=state.dtype)
        else:
            valid_mask = valid_mask.to(device=device, dtype=state.dtype)
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(-1)

        discount = torch.pow(
            torch.full((horizon,), self.rollout_discount, device=device, dtype=state.dtype),
            torch.arange(horizon, device=device, dtype=state.dtype),
        ).view(1, horizon, 1)
        weights = valid_mask * discount
        normalizer = weights.sum().clamp_min(1.0)

        z_roll = self.encode(state, task_encoding)
        task_encoding_seq = task_encoding.unsqueeze(1).expand(-1, horizon, -1)

        with torch.no_grad():
            next_states_flat = next_states.reshape(batch_size * horizon, -1)
            task_flat = task_encoding_seq.reshape(batch_size * horizon, -1)
            z_targets = self.encode(next_states_flat, task_flat).view(batch_size, horizon, -1)

        pred_latents = []
        reward_logits = []
        z_curr = z_roll
        for step in range(horizon):
            action_step = actions[:, step]
            reward_logits.append(self.reward_logits(z_curr, action_step, task_encoding))
            z_curr = self.predict_next_latent(z_curr, action_step, task_encoding)
            pred_latents.append(z_curr)

        pred_latents = torch.stack(pred_latents, dim=1)
        reward_logits = torch.stack(reward_logits, dim=1)

        dyn_per_step = F.mse_loss(pred_latents, z_targets, reduction="none").mean(dim=-1, keepdim=True)
        dyn_loss = (dyn_per_step * weights).sum() / normalizer

        if self.reward_bins > 1:
            logits_flat = reward_logits.reshape(batch_size * horizon, -1)
            rewards_flat = rewards.reshape(batch_size * horizon, 1)
            rew_per_step = soft_ce(logits_flat, rewards_flat, self.dreg_cfg).view(batch_size, horizon, 1)
        else:
            rew_per_step = F.mse_loss(reward_logits, rewards, reduction="none").mean(dim=-1, keepdim=True)

        rew_loss = (rew_per_step * weights).sum() / normalizer
        return dyn_loss, rew_loss

    @staticmethod
    def _ensure_sequence_shape(
        state: TensorType,
        action: TensorType,
        next_state: TensorType,
        valid_mask: Optional[TensorType],
    ) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if next_state.dim() == 2:
            next_state = next_state.unsqueeze(1)
        if valid_mask is None:
            valid_mask = torch.ones(
                next_state.shape[0], next_state.shape[1], 1,
                device=next_state.device, dtype=next_state.dtype
            )
        elif valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        prev_state = torch.cat([state.unsqueeze(1), next_state[:, :-1]], dim=1)
        return prev_state, action, next_state, valid_mask

    def compute_delta_state_similarity_loss(
        self,
        state_a: TensorType,
        action_a: TensorType,
        next_state_a: TensorType,
        state_b: TensorType,
        action_b: TensorType,
        next_state_b: TensorType,
        valid_mask_a: Optional[TensorType] = None,
        valid_mask_b: Optional[TensorType] = None,
    ) -> TensorType:
        """
        Align state-change sequences across different robots for the same task.
        Samples are matched by action-sequence cosine similarity, then their
        delta-state sequences are encouraged to be close in cosine space.
        """
        prev_a, action_a, next_state_a, valid_mask_a = self._ensure_sequence_shape(
            state_a, action_a, next_state_a, valid_mask_a
        )
        prev_b, action_b, next_state_b, valid_mask_b = self._ensure_sequence_shape(
            state_b, action_b, next_state_b, valid_mask_b
        )

        delta_a = (next_state_a - prev_a) * valid_mask_a
        delta_b = (next_state_b - prev_b) * valid_mask_b
        action_a = action_a * valid_mask_a
        action_b = action_b * valid_mask_b

        delta_feat_a = delta_a.reshape(delta_a.shape[0], -1)
        delta_feat_b = delta_b.reshape(delta_b.shape[0], -1)
        action_feat_a = action_a.reshape(action_a.shape[0], -1)
        action_feat_b = action_b.reshape(action_b.shape[0], -1)

        valid_rows_a = valid_mask_a.sum(dim=(1, 2)) > 0
        valid_rows_b = valid_mask_b.sum(dim=(1, 2)) > 0
        if valid_rows_a.sum() == 0 or valid_rows_b.sum() == 0:
            return torch.zeros((), device=state_a.device, dtype=state_a.dtype)

        delta_feat_a = delta_feat_a[valid_rows_a]
        delta_feat_b = delta_feat_b[valid_rows_b]
        action_feat_a = action_feat_a[valid_rows_a]
        action_feat_b = action_feat_b[valid_rows_b]

        action_feat_a = F.normalize(action_feat_a, p=2, dim=-1, eps=1e-8)
        action_feat_b = F.normalize(action_feat_b, p=2, dim=-1, eps=1e-8)
        delta_feat_a = F.normalize(delta_feat_a, p=2, dim=-1, eps=1e-8)
        delta_feat_b = F.normalize(delta_feat_b, p=2, dim=-1, eps=1e-8)

        temperature = max(self.delta_state_similarity_temperature, 1e-6)
        action_sim = (action_feat_a @ action_feat_b.t()) / temperature

        # Keep correspondence soft so temperature actually controls the
        # sharpness of cross-robot matching.
        weights_ab = F.softmax(action_sim, dim=1)
        weights_ba = F.softmax(action_sim.t(), dim=1)

        aligned_delta_b = weights_ab @ delta_feat_b
        aligned_delta_a = weights_ba @ delta_feat_a
        aligned_delta_b = F.normalize(aligned_delta_b, p=2, dim=-1, eps=1e-8)
        aligned_delta_a = F.normalize(aligned_delta_a, p=2, dim=-1, eps=1e-8)

        loss_ab = 1.0 - F.cosine_similarity(delta_feat_a, aligned_delta_b, dim=-1).mean()
        loss_ba = 1.0 - F.cosine_similarity(delta_feat_b, aligned_delta_a, dim=-1).mean()
        return 0.5 * (loss_ab + loss_ba)

    # ---------- reward-bounds utilities ----------
    def set_reward_bounds(self, vmin: float, vmax: float):
        """Update two-hot symlog(bins) bounds online; only effective when reward_bins>1."""
        if self.reward_bins <= 1:
            self.dreg_cfg = None
            return
        # Avoid equal bounds
        if math.isclose(vmin, vmax):
            eps = 1e-3
            vmin, vmax = vmin - eps, vmax + eps
        self.dreg_cfg = DRegCfg(num_bins=self.reward_bins, vmin=vmin, vmax=vmax)

    @staticmethod
    def estimate_reward_bounds_from_buffer(
        replay_buffer,
        max_samples: int = 100_000,
        q_low: float = 0.01,
        q_high: float = 0.99,
        print_stats: bool = True,
    ):
        """
        Adapted for TransformerReplayBuffer:
        - Reads directly from replay_buffer.rewards (shape [E, T, 1])
        - Only uses the valid range (full ? capacity : idx)
        - Applies symlog first, then computes quantiles on the symlog axis as (vmin, vmax)
        Returns: vmin, vmax (float, on the symlog axis)
        """
        # Number of valid samples
        N_valid = replay_buffer.capacity if getattr(replay_buffer, "full", False) \
                else getattr(replay_buffer, "idx", 0)
        if N_valid <= 0:
            raise RuntimeError("Replay buffer has no valid reward samples yet.")

        # Flatten and take the first N_valid entries (avoid including unwritten trailing zeros)
        r_np = replay_buffer.rewards.reshape(-1)[:N_valid].astype(np.float32)
        if r_np.size == 0:
            raise RuntimeError("Rewards retrieved from replay_buffer.rewards are empty.")

        # Subsample if too many samples (fixed random seed for reproducibility)
        if r_np.size > max_samples:
            rng = np.random.RandomState(0)
            idx = rng.choice(r_np.size, size=max_samples, replace=False)
            r_np = r_np[idx]

        # Convert to symlog axis
        r_symlog = symlog(torch.from_numpy(r_np)).numpy()

        # Use quantiles as bounds (more robust to outliers)
        vmin = float(np.quantile(r_symlog, q_low))
        vmax = float(np.quantile(r_symlog, q_high))

        # Prevent bounds from being equal
        if np.isclose(vmin, vmax):
            eps = 1e-3
            vmin, vmax = vmin - eps, vmax + eps

        if print_stats:
            print(
                "[Reward bounds] "
                f"raw_min={r_np.min():.4f}, raw_max={r_np.max():.4f}, "
                f"symlog_q{int(q_low*100)}={vmin:.4f}, symlog_q{int(q_high*100)}={vmax:.4f}, "
                f"N={r_np.size}"
            )

        return vmin, vmax
