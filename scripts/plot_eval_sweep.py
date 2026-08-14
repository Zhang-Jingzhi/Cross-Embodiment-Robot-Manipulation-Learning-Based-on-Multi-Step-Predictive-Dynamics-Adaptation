#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot one or more evaluation sweeps.")
    parser.add_argument(
        "--point",
        action="append",
        required=True,
        help="Format: label:x=/abs/path/to/evaluation_summary.json",
    )
    parser.add_argument(
        "--metric",
        default="unseen",
        help="overall | seen | unseen | hard_subset | per_task:<task> | per_robot:<robot>",
    )
    parser.add_argument(
        "--hard-tasks",
        default="peg-insert-side-v2,door-open-v2,window-open-v2,window-close-v2,faucet-open-v2",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--xlabel", default="Sweep Value")
    parser.add_argument("--ylabel", default="Success Rate (%)")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_summary(path: Path):
    return json.loads(path.read_text())


def metric_value(summary, metric: str, hard_tasks):
    if metric == "overall":
        return summary["overall"]
    if metric == "seen":
        return summary["seen"]
    if metric == "unseen":
        return summary["unseen"]
    if metric == "hard_subset":
        return mean(summary["per_task"][task] for task in hard_tasks)
    if metric.startswith("per_task:"):
        task = metric.split(":", 1)[1]
        return summary["per_task"][task]
    if metric.startswith("per_robot:"):
        robot = metric.split(":", 1)[1]
        return summary["per_robot"][robot]
    raise ValueError(f"Unsupported metric={metric}")


def main():
    args = parse_args()
    hard_tasks = [task.strip() for task in args.hard_tasks.split(",") if task.strip()]

    series = {}
    for raw in args.point:
        lhs, rhs = raw.split("=", 1)
        label, x_raw = lhs.split(":", 1)
        x_val = float(x_raw)
        summary = load_summary(Path(rhs))
        y_val = metric_value(summary, args.metric, hard_tasks)
        series.setdefault(label, []).append((x_val, y_val))

    plt.figure(figsize=(8, 5))
    for label, pairs in sorted(series.items()):
        pairs = sorted(pairs, key=lambda item: item[0])
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        plt.plot(xs, ys, marker="o", linewidth=2, label=label)

    if args.title:
        plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
