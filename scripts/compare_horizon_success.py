#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(description="Compare success summaries across horizons.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: label=/abs/path/to/evaluation_summary.json OR label=experiment:seed",
    )
    parser.add_argument(
        "--hard-tasks",
        default="peg-insert-side-v2,door-open-v2,window-open-v2,window-close-v2,faucet-open-v2",
    )
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_summary(project_root: Path, spec: str):
    label, raw = spec.split("=", 1)
    maybe_path = Path(raw)
    if maybe_path.exists():
        return label, json.loads(maybe_path.read_text())

    experiment, seed = raw.rsplit(":", 1)
    proc = subprocess.run(
        ["python", "scripts/summarize_collective_results.py", "--experiment", experiment, "--seed", seed],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return label, json.loads(proc.stdout)


def round_map(values):
    return {k: round(v, 4) for k, v in sorted(values.items(), key=lambda item: item[0])}


def main():
    args = parse_args()
    hard_tasks = [task.strip() for task in args.hard_tasks.split(",") if task.strip()]
    summaries = [load_summary(args.project_root, spec) for spec in args.run]
    baseline_label, baseline = summaries[0]

    table = {}
    for label, summary in summaries:
        hard_score = mean(summary["per_task"][task] for task in hard_tasks)
        table[label] = {
            "overall": round(summary["overall"], 4),
            "seen": round(summary["seen"], 4),
            "unseen": round(summary["unseen"], 4),
            "hard_subset": round(hard_score, 4),
        }

    deltas = {}
    for label, summary in summaries[1:]:
        deltas[label] = {
            "overall_delta_vs_" + baseline_label: round(summary["overall"] - baseline["overall"], 4),
            "hard_subset_delta_vs_" + baseline_label: round(
                mean(summary["per_task"][task] for task in hard_tasks)
                - mean(baseline["per_task"][task] for task in hard_tasks),
                4,
            ),
            "per_task_delta": round_map(
                {
                    task: summary["per_task"][task] - baseline["per_task"][task]
                    for task in baseline["per_task"]
                }
            ),
            "per_robot_delta": round_map(
                {
                    robot: summary["per_robot"][robot] - baseline["per_robot"][robot]
                    for robot in baseline["per_robot"]
                }
            ),
        }

    payload = {
        "baseline": baseline_label,
        "hard_tasks": hard_tasks,
        "table": table,
        "deltas_vs_baseline": deltas,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
