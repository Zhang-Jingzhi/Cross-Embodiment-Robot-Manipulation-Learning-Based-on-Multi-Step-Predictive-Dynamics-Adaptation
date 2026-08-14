#!/usr/bin/env python3

import argparse
import contextlib
import io
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mtrl.agent.components.predictive_adapter import PredictiveAdapter
from mtrl.agent.transformer_agent import _compress_meta_state
from mtrl.transformer_replay_buffer import TransformerReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predictive-adapter k-step rollout error on the validation buffer."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: label=/abs/save_dir:seed . Example: h1=/path/to/run:5",
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--eval-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--buffer-save-dir",
        type=Path,
        help="Optional save_dir to reuse a single validation buffer for all runs.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_pa_cfg(project_root: Path) -> Dict:
    cfg = OmegaConf.load(
        project_root / "config/transformer_collective_network/components/metaworld_predictive_adapter.yaml"
    )
    for key in ["pretrained_dir", "pretrained_step", "freeze_after_load", "load_on_init"]:
        if key in cfg:
            del cfg[key]
    return OmegaConf.to_container(cfg, resolve=False)


def latest_checkpoint(model_dir: Path) -> Path:
    checkpoints = sorted(model_dir.glob("predictive_adapter_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No predictive_adapter_*.pt found in {model_dir}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def parse_run_spec(spec: str) -> Tuple[str, Path, int]:
    label, raw = spec.split("=", 1)
    save_dir_raw, seed_raw = raw.rsplit(":", 1)
    return label, Path(save_dir_raw), int(seed_raw)


def make_buffer(buffer_root: Path, batch_size: int, device: str) -> TransformerReplayBuffer:
    subdirs = sorted(path for path in buffer_root.iterdir() if path.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No validation sub-buffers found under {buffer_root}")
    buffer = TransformerReplayBuffer(
        env_obs_shape=[39],
        task_obs_shape=(1,),
        action_shape=(4,),
        capacity=2_000_000,
        batch_size=batch_size,
        device=torch.device(device),
        normalize_rewards=False,
        seq_len=20,
        task_encoding_shape=6,
        compressed_state=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        buffer.load_multiple_buffer([str(path) for path in subdirs])
    return buffer


def sample_multi_step_fallback(
    buffer: TransformerReplayBuffer,
    horizon: int,
    batch_idxs,
    device: str,
):
    idxs = np.asarray(batch_idxs)
    horizon = max(int(horizon), 1)
    ep_len = 400

    episode_ids = idxs // ep_len
    timesteps = idxs % ep_len
    future_offsets = np.arange(horizon)[None, :]
    future_timesteps = timesteps[:, None] + future_offsets
    within_episode = future_timesteps < ep_len
    future_timesteps = np.clip(future_timesteps, 0, ep_len - 1)

    current_states = torch.as_tensor(
        buffer.env_obses[episode_ids, timesteps], device=device
    ).float()
    action_seq = torch.as_tensor(
        buffer.actions[episode_ids[:, None], future_timesteps], device=device
    ).float()
    reward_seq = torch.as_tensor(
        buffer.rewards[episode_ids[:, None], future_timesteps], device=device
    ).float()
    next_state_seq = torch.as_tensor(
        buffer.next_env_obses[episode_ids[:, None], future_timesteps], device=device
    ).float()
    env_indices = torch.as_tensor(
        buffer.task_obs[episode_ids, timesteps], device=device
    )
    task_encoding = torch.as_tensor(
        buffer.task_encodings[episode_ids, timesteps], device=device
    ).float()
    not_done_seq = torch.as_tensor(
        buffer.not_dones[episode_ids[:, None], future_timesteps], device=device
    ).float()
    within_episode = torch.as_tensor(
        within_episode, device=device, dtype=torch.float32
    ).unsqueeze(-1)
    carry_mask = torch.cat(
        [
            torch.ones((len(idxs), 1, 1), device=device, dtype=torch.float32),
            torch.cumprod(not_done_seq[:, :-1], dim=1),
        ],
        dim=1,
    )
    valid_mask = within_episode * carry_mask
    return current_states, action_seq, reward_seq, next_state_seq, valid_mask, env_indices, task_encoding


def build_model(project_root: Path, checkpoint_dir: Path, device: str) -> PredictiveAdapter:
    cfg = load_pa_cfg(project_root)
    accepted = set(inspect.signature(PredictiveAdapter.__init__).parameters.keys())
    cfg = {key: value for key, value in cfg.items() if key in accepted}
    model = PredictiveAdapter(
        state_dim=21,
        action_dim=4,
        task_encoding_dim=6,
        **cfg,
    ).to(device)
    ckpt = latest_checkpoint(checkpoint_dir)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_one_run(
    label: str,
    save_dir: Path,
    seed: int,
    args,
    buffer_cache: Dict[str, TransformerReplayBuffer],
) -> Dict:
    experiment_name = save_dir.name
    checkpoint_dir = save_dir / "model_dir" / experiment_name / f"model_predictive_adapter_seed_{seed}"
    buffer_save_dir = args.buffer_save_dir if args.buffer_save_dir else save_dir
    buffer_root = buffer_save_dir / "buffer" / "collective_buffer" / "validation"
    if not buffer_root.exists():
        raise FileNotFoundError(f"Validation buffer not found: {buffer_root}")

    model = build_model(args.project_root, checkpoint_dir, args.device)
    buffer_key = str(buffer_root.resolve())
    if buffer_key not in buffer_cache:
        buffer_cache[buffer_key] = make_buffer(buffer_root, args.batch_size, args.device)
    buffer = buffer_cache[buffer_key]

    valid_count = buffer.capacity if buffer.full else buffer.idx
    sample_count = min(valid_count, args.num_samples)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(valid_count, size=sample_count, replace=False)

    latent_num = torch.zeros(args.eval_horizon, device=args.device)
    reward_mae_num = torch.zeros(args.eval_horizon, device=args.device)
    reward_mse_num = torch.zeros(args.eval_horizon, device=args.device)
    denom = torch.zeros(args.eval_horizon, device=args.device)

    for start in range(0, sample_count, args.batch_size):
        batch_idxs = indices[start : start + args.batch_size]
        if hasattr(buffer, "sample_multi_step"):
            states, actions, rewards, next_states, valid_mask, _, task_encoding = buffer.sample_multi_step(
                args.eval_horizon, index=batch_idxs, device=torch.device(args.device)
            )
        else:
            states, actions, rewards, next_states, valid_mask, _, task_encoding = sample_multi_step_fallback(
                buffer, args.eval_horizon, batch_idxs, args.device
            )
        next_states = _compress_meta_state(next_states)
        mask = valid_mask.squeeze(-1)

        z0 = model.encode(states, task_encoding)
        expanded_task = task_encoding.unsqueeze(1).expand(-1, args.eval_horizon, -1)
        z_targets = model.encode(
            next_states.reshape(-1, next_states.shape[-1]),
            expanded_task.reshape(-1, expanded_task.shape[-1]),
        ).view(states.shape[0], args.eval_horizon, -1)

        rollout = model.latent_rollout(z0, actions, task_encoding)[1:].transpose(0, 1)
        latent_mse = (rollout - z_targets).pow(2).mean(dim=-1)

        reward_preds: List[torch.Tensor] = []
        z_curr = z0
        for step in range(args.eval_horizon):
            reward_preds.append(model.predict_reward(z_curr, actions[:, step], task_encoding))
            z_curr = model.predict_next_latent(z_curr, actions[:, step], task_encoding)
        reward_preds = torch.stack(reward_preds, dim=1).squeeze(-1)
        reward_target = rewards.squeeze(-1)

        latent_num += (latent_mse * mask).sum(dim=0)
        reward_mae_num += ((reward_preds - reward_target).abs() * mask).sum(dim=0)
        reward_mse_num += (((reward_preds - reward_target) ** 2) * mask).sum(dim=0)
        denom += mask.sum(dim=0)

    denom = denom.clamp_min(1.0)
    result = {
        "label": label,
        "save_dir": str(save_dir),
        "seed": seed,
        "sample_count": int(sample_count),
        "eval_horizon": args.eval_horizon,
        "latent_mse": [round(x, 6) for x in (latent_num / denom).detach().cpu().tolist()],
        "reward_mae": [round(x, 6) for x in (reward_mae_num / denom).detach().cpu().tolist()],
        "reward_mse": [round(x, 6) for x in (reward_mse_num / denom).detach().cpu().tolist()],
        "valid_counts": [int(x) for x in denom.detach().cpu().tolist()],
    }
    return result


def main():
    args = parse_args()
    runs = [parse_run_spec(spec) for spec in args.run]
    buffer_cache: Dict[str, TransformerReplayBuffer] = {}
    payload = {
        "metric": "predictive_adapter_k_step_rollout_error",
        "results": [evaluate_one_run(label, save_dir, seed, args, buffer_cache) for label, save_dir, seed in runs],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
