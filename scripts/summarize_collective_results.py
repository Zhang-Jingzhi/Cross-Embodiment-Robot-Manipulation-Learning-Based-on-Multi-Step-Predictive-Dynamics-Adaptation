#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from statistics import mean


SEEN_ROBOTS = {"sawyer", "gen3", "xarm7", "ur10e", "unitree_z1"}
UNSEEN_ROBOTS = {"panda", "kuka", "ur5e", "viperx"}
EVAL_RE = re.compile(r"Evaluation: robot\s+([^\s]+)\s+\[([0-9.]+)\]\s+from\s+([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output")
    return parser.parse_args()


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    match = EVAL_RE.search(text)
    if not match:
      return None
    robot = match.group(1)
    success_count = float(match.group(2))
    total = float(match.group(3))
    if total == 0:
      return None
    task = path.parent.name
    task = task[: -(len("_" + args.experiment))]
    success_rate = success_count / total * 100.0
    return {
        "robot": robot,
        "task": task,
        "success_count": success_count,
        "total": total,
        "success_rate": success_rate,
        "path": str(path),
    }


def round_map(values):
    return {k: round(v, 4) for k, v in sorted(values.items())}


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(args.project_root)
    results_root = project_root / "logs" / "results" / "col"
    pattern = f"*_{{exp}}/eval_*_seed_{args.seed}.log".format(exp=args.experiment)
    rows = []
    for path in sorted(results_root.glob(pattern)):
        parsed = parse_log(path)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        raise SystemExit(f"No completed result logs found for experiment={args.experiment} seed={args.seed}")

    overall = mean(row["success_rate"] for row in rows)
    seen_rows = [row for row in rows if row["robot"] in SEEN_ROBOTS]
    unseen_rows = [row for row in rows if row["robot"] in UNSEEN_ROBOTS]
    by_robot = {}
    by_task = {}
    for row in rows:
        by_robot.setdefault(row["robot"], []).append(row["success_rate"])
        by_task.setdefault(row["task"], []).append(row["success_rate"])

    summary = {
        "experiment": args.experiment,
        "seed": args.seed,
        "num_cases": len(rows),
        "overall": round(overall, 4),
        "seen": round(mean(r["success_rate"] for r in seen_rows), 4) if seen_rows else None,
        "unseen": round(mean(r["success_rate"] for r in unseen_rows), 4) if unseen_rows else None,
        "per_robot": round_map({robot: mean(vals) for robot, vals in by_robot.items()}),
        "per_task": round_map({task: mean(vals) for task, vals in by_task.items()}),
        "worst_cases": sorted(
            [
                {
                    "robot": row["robot"],
                    "task": row["task"],
                    "success_rate": round(row["success_rate"], 4),
                    "path": row["path"],
                }
                for row in rows
            ],
            key=lambda item: item["success_rate"],
        )[:10],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
