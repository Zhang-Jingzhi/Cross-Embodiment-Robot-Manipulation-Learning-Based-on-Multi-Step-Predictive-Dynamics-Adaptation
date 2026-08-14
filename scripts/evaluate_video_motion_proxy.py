#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from statistics import mean

import imageio
import numpy as np


VIDEO_RE = re.compile(
    r"(?P<robot>[^_/]+)_transformer_env\d+_(?P<task>.+?)_success_(?P<success>[0-9.]+)_reward_(?P<reward>-?\d+)_sample_\d+\.gif$"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute a lightweight motion-smoothness proxy from GIFs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: label=/abs/path/to/video_dir",
    )
    parser.add_argument("--robots", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def frame_motion_metrics(path: Path):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        frames.append(arr)
    reader.close()
    if len(frames) < 2:
        return None
    stack = np.stack(frames, axis=0)
    diffs = np.abs(np.diff(stack, axis=0))
    motion_energy = diffs.mean(axis=(1, 2, 3))
    jerk = np.abs(np.diff(motion_energy)) if len(motion_energy) > 1 else np.zeros(1, dtype=np.float32)
    return {
        "mean_motion_energy": float(motion_energy.mean()),
        "std_motion_energy": float(motion_energy.std()),
        "motion_jerk": float(jerk.mean()),
        "normalized_jerk": float(jerk.mean() / (motion_energy.mean() + 1e-8)),
    }


def main():
    args = parse_args()
    robots = {item.strip() for item in args.robots.split(",") if item.strip()}
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    payload = {"metric": "gif_motion_proxy", "results": []}

    for spec in args.run:
        label, raw_dir = spec.split("=", 1)
        video_dir = Path(raw_dir)
        rows = []
        for path in sorted(video_dir.glob("*.gif")):
            match = VIDEO_RE.match(path.name)
            if not match:
                continue
            robot = match.group("robot")
            task = match.group("task")
            if robots and robot not in robots:
                continue
            if tasks and task not in tasks:
                continue
            metrics = frame_motion_metrics(path)
            if metrics is None:
                continue
            rows.append(
                {
                    "robot": robot,
                    "task": task,
                    "success": float(match.group("success")),
                    "reward": float(match.group("reward")),
                    "path": str(path),
                    **metrics,
                }
            )

        aggregate = {
            "label": label,
            "video_dir": str(video_dir),
            "num_cases": len(rows),
        }
        if rows:
            aggregate["mean_motion_energy"] = round(mean(row["mean_motion_energy"] for row in rows), 6)
            aggregate["mean_motion_jerk"] = round(mean(row["motion_jerk"] for row in rows), 6)
            aggregate["mean_normalized_jerk"] = round(mean(row["normalized_jerk"] for row in rows), 6)
            aggregate["per_task_normalized_jerk"] = {
                task: round(mean(row["normalized_jerk"] for row in rows if row["task"] == task), 6)
                for task in sorted({row["task"] for row in rows})
            }
        payload["results"].append(aggregate)

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
