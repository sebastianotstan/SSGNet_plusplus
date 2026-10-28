#!/usr/bin/env python3
"""Summarize completed held-out test results without changing run files."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


CONFIG_TO_ARM = {
    "mb": "mb",
    "multihead": "mh",
    "mbmh": "mbmh",
    "mbmh_refine": "mbmh-both",
    "mbmh_full": "mbmh-both-tk",
    "mbmh_confidence": "mbmh-both-tk-confsel",
    "single": "single-control",
}
DATASET_ORDER = {name: i for i, name in enumerate(("cbis", "isic2017", "isic2018", "polyps"))}
ARM_ORDER = {name: i for i, name in enumerate((
    "mb", "mh", "mbmh", "mbmh-both", "mbmh-both-tk",
    "mbmh-both-tk-confsel", "single-control",
))}
METRICS = ("dice", "foreground_iou", "sensitivity", "specificity", "accuracy")


def read_runs(root):
    runs = []
    for path in sorted(root.glob("*/*/seed_*/test/summary.json")):
        config, dataset = path.parents[3].name, path.parents[2].name
        arm = CONFIG_TO_ARM.get(config)
        if arm is None:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = data["selected_test"]
            row = {
                "arm": arm,
                "config": config,
                "dataset": data.get("dataset", dataset),
                "seed": int(data["seed"]),
                "selected_round": int(data["selected_round_by_validation"]),
                "summary_path": str(path),
            }
            for metric in METRICS:
                value = float(metrics[metric])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {metric}")
                row[metric] = value
            runs.append(row)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read completed summary {path}: {exc}") from exc
    return runs


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs):
    groups = defaultdict(list)
    for row in runs:
        groups[(row["arm"], row["dataset"])].append(row)
    output = []
    for (arm, dataset), items in sorted(
        groups.items(), key=lambda item: (ARM_ORDER.get(item[0][0], 99), DATASET_ORDER.get(item[0][1], 99))
    ):
        row = {
            "arm": arm,
            "dataset": dataset,
            "n_seeds": len(items),
            "seeds": ";".join(str(item["seed"]) for item in sorted(items, key=lambda x: x["seed"])),
        }
        rounds = Counter(item["selected_round"] for item in items)
        for round_index in range(3):
            row[f"selected_r{round_index}"] = rounds.get(round_index, 0)
        for metric in METRICS:
            values = [item[metric] for item in items]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("revision/runs"))
    parser.add_argument("--summary-out", type=Path, default=Path("revision/progress_summary.csv"))
    parser.add_argument("--runs-out", type=Path, default=Path("revision/progress_runs.csv"))
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error(f"Run folder not found: {args.runs}")
    runs = read_runs(args.runs)
    if not runs:
        parser.error(f"No completed test summaries found under {args.runs}")
    groups = summarize(runs)
    write_csv(args.summary_out, groups, list(groups[0]))
    run_fields = ["arm", "config", "dataset", "seed", "selected_round", *METRICS, "summary_path"]
    write_csv(args.runs_out, sorted(runs, key=lambda r: (
        ARM_ORDER.get(r["arm"], 99), DATASET_ORDER.get(r["dataset"], 99), r["seed"]
    )), run_fields)
    print(f"Read {len(runs)} completed test summaries across {len(groups)} arm/dataset groups.")
    print(f"Group summary: {args.summary_out}")
    print(f"Per-run details: {args.runs_out}")
    print("Group means are descriptive; seed counts are incomplete and may differ by group.")


if __name__ == "__main__":
    main()
