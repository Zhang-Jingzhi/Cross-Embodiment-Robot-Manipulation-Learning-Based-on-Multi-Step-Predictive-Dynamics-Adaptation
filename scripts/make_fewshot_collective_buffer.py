#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


TASK_NAMES = [
    "button-press-topdown-v2",
    "door-open-v2",
    "drawer-open-v2",
    "faucet-open-v2",
    "peg-insert-side-v2",
    "pick-place-v2",
    "push-v2",
    "reach-v2",
    "window-close-v2",
    "window-open-v2",
]

DEFAULT_FEWSHOT_ROBOTS = ["panda", "kuka", "ur5e", "viperx"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a K-shot collective buffer by keeping only K full episodes for selected robots."
    )
    parser.add_argument("--src", required=True, type=Path, help="Source collective_buffer directory")
    parser.add_argument("--dst", required=True, type=Path, help="Destination collective_buffer directory")
    parser.add_argument("--shots", required=True, type=int, help="Number of train episodes to keep per robot-task for few-shot robots")
    parser.add_argument(
        "--fewshot-robots",
        default=",".join(DEFAULT_FEWSHOT_ROBOTS),
        help="Comma-separated robot names treated as unseen/few-shot robots",
    )
    parser.add_argument(
        "--fewshot-val-shots",
        type=int,
        default=0,
        help="Validation episodes to keep for few-shot robots. Default 0 to avoid unseen validation leakage.",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=400,
        help="Transitions per episode inside the stored replay chunks",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["first", "random"],
        default="random",
        help="How to choose the kept episodes for few-shot robots",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Overwrite dst if it already exists")
    return parser.parse_args()


def parse_buffer_name(name: str) -> Optional[Tuple[str, str]]:
    prefix = "online_buffer_"
    if not name.startswith(prefix):
        return None
    core = name[len(prefix) :]
    seed_marker = core.rfind("_seed_")
    if seed_marker == -1:
        return None
    core = core[:seed_marker]
    for task in sorted(TASK_NAMES, key=len, reverse=True):
        marker = f"_{task}"
        if core.endswith(marker):
            robot = core[: -len(marker)]
            return robot, task
    return None


def list_chunk_paths(buffer_dir: Path) -> List[Path]:
    return sorted(buffer_dir.glob("*.pt"), key=lambda path: int(path.stem.split("_")[0]))


def load_episodes(buffer_dir: Path, episode_len: int) -> List[List[torch.Tensor]]:
    episodes: List[List[torch.Tensor]] = []
    for chunk_path in list_chunk_paths(buffer_dir):
        payload = torch.load(chunk_path, map_location="cpu")
        if not isinstance(payload, list):
            raise TypeError(f"Unexpected chunk payload type at {chunk_path}: {type(payload)!r}")
        num_rows = payload[0].shape[0]
        if num_rows % episode_len != 0:
            raise ValueError(
                f"Chunk {chunk_path} has {num_rows} rows, not divisible by episode_len={episode_len}"
            )
        for start in range(0, num_rows, episode_len):
            end = start + episode_len
            episodes.append([tensor[start:end].clone() for tensor in payload])
    return episodes


def choose_episode_indices(
    total_episodes: int,
    keep_episodes: int,
    sampling_mode: str,
    seed: int,
    key: str,
) -> List[int]:
    keep_episodes = max(0, min(keep_episodes, total_episodes))
    if keep_episodes == 0:
        return []
    if keep_episodes == total_episodes or sampling_mode == "first":
        return list(range(keep_episodes))

    import random

    hashed = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    rng = random.Random(int(hashed[:16], 16))
    return sorted(rng.sample(range(total_episodes), keep_episodes))


def write_episode_subset(
    src_dir: Path,
    dst_dir: Path,
    keep_episodes: int,
    episode_len: int,
    sampling_mode: str,
    seed: int,
    key: str,
) -> dict:
    episodes = load_episodes(src_dir, episode_len=episode_len)
    selected = choose_episode_indices(
        total_episodes=len(episodes),
        keep_episodes=keep_episodes,
        sampling_mode=sampling_mode,
        seed=seed,
        key=key,
    )

    dst_dir.mkdir(parents=True, exist_ok=True)
    if not selected:
        return {
            "mode": "subset",
            "source_dir": str(src_dir),
            "kept_episodes": 0,
            "total_episodes": len(episodes),
            "selected_episode_indices": [],
            "chunk_files": [],
        }

    selected_episodes = [episodes[idx] for idx in selected]
    flat_payloads = [
        torch.cat([episode[field_idx] for episode in selected_episodes], dim=0)
        for field_idx in range(len(selected_episodes[0]))
    ]
    total_rows = flat_payloads[0].shape[0]
    chunk_size = 16000
    chunk_files = []
    start = 0
    while start < total_rows:
        end = min(start + chunk_size, total_rows)
        chunk_payload = [tensor[start:end].clone() for tensor in flat_payloads]
        chunk_path = dst_dir / f"{start}_{end - 1}.pt"
        torch.save(chunk_payload, chunk_path)
        chunk_files.append(chunk_path.name)
        start = end

    return {
        "mode": "subset",
        "source_dir": str(src_dir),
        "kept_episodes": len(selected),
        "total_episodes": len(episodes),
        "selected_episode_indices": selected,
        "chunk_files": chunk_files,
    }


def symlink_dir(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)
    return {
        "mode": "symlink",
        "source_dir": str(src),
    }


def build_split(
    src_split: Path,
    dst_split: Path,
    fewshot_robots: Sequence[str],
    episode_len: int,
    train_shots: int,
    val_shots: int,
    sampling_mode: str,
    seed: int,
    split_name: str,
) -> dict:
    split_summary = {}
    if not src_split.exists():
        return split_summary

    dst_split.mkdir(parents=True, exist_ok=True)
    for src_dir in sorted(path for path in src_split.iterdir() if path.is_dir()):
        parsed = parse_buffer_name(src_dir.name)
        if parsed is None:
            continue
        robot, task = parsed
        key = f"{split_name}:{robot}:{task}"
        dst_dir = dst_split / src_dir.name
        if dst_dir.exists() or dst_dir.is_symlink():
            raise FileExistsError(f"Destination already exists: {dst_dir}")

        if robot not in fewshot_robots:
            split_summary[src_dir.name] = symlink_dir(src_dir, dst_dir)
            continue

        keep_shots = train_shots if split_name == "train" else val_shots
        if keep_shots <= 0:
            split_summary[src_dir.name] = {
                "mode": "omitted",
                "source_dir": str(src_dir),
                "kept_episodes": 0,
            }
            continue

        split_summary[src_dir.name] = write_episode_subset(
            src_dir=src_dir,
            dst_dir=dst_dir,
            keep_episodes=keep_shots,
            episode_len=episode_len,
            sampling_mode=sampling_mode,
            seed=seed,
            key=key,
        )
    return split_summary


def count_summary(split_summary: dict) -> dict:
    kept_dirs = 0
    kept_episodes = 0
    for info in split_summary.values():
        if info["mode"] in {"symlink", "subset"}:
            kept_dirs += 1
        kept_episodes += int(info.get("kept_episodes", 0))
    return {"kept_dirs": kept_dirs, "kept_episodes": kept_episodes}


def main():
    args = parse_args()
    fewshot_robots = [robot.strip() for robot in args.fewshot_robots.split(",") if robot.strip()]
    if args.shots < 0:
        raise ValueError("--shots must be >= 0")
    if args.fewshot_val_shots < 0:
        raise ValueError("--fewshot-val-shots must be >= 0")

    if args.dst.exists():
        if not args.force:
            raise FileExistsError(f"Destination already exists: {args.dst}. Use --force to overwrite.")
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    train_summary = build_split(
        src_split=args.src / "train",
        dst_split=args.dst / "train",
        fewshot_robots=fewshot_robots,
        episode_len=args.episode_len,
        train_shots=args.shots,
        val_shots=args.fewshot_val_shots,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
        split_name="train",
    )
    val_summary = build_split(
        src_split=args.src / "validation",
        dst_split=args.dst / "validation",
        fewshot_robots=fewshot_robots,
        episode_len=args.episode_len,
        train_shots=args.shots,
        val_shots=args.fewshot_val_shots,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
        split_name="validation",
    )

    metadata = {
        "src": str(args.src),
        "dst": str(args.dst),
        "shots": args.shots,
        "fewshot_val_shots": args.fewshot_val_shots,
        "fewshot_robots": fewshot_robots,
        "episode_len": args.episode_len,
        "sampling_mode": args.sampling_mode,
        "seed": args.seed,
        "train": train_summary,
        "validation": val_summary,
        "train_counts": count_summary(train_summary),
        "validation_counts": count_summary(val_summary),
    }
    metadata_path = args.dst / "fewshot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata["train_counts"], indent=2))
    print(json.dumps(metadata["validation_counts"], indent=2))
    print(f"Wrote few-shot collective buffer to {args.dst}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
