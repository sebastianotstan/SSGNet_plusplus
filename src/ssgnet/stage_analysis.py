#!/usr/bin/env python3
"""Export round-by-round and paired ablation data from revision run folders.

The script never reads test results to choose a round. It uses the selected round
recorded by training (validation selection), then reports that round and all
predeclared rounds separately.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("dice", "foreground_iou", "sensitivity", "specificity", "accuracy")
EXPECTED_CONFIGS = (
    "mb", "multihead", "mbmh", "mbmh_refine", "mbmh_full",
    "mbmh_confidence", "single",
)
EXPECTED_DATASETS = ("cbis", "isic2017", "isic2018", "polyps")
EXPECTED_SEEDS = (42, 123, 2024, 3407)
CONFIG_ARMS = {
    "mb": "MB",
    "multihead": "MH",
    "mbmh": "MBMH",
    "mbmh_refine": "MBMH+both",
    "mbmh_full": "MBMH+both+TK",
    "mbmh_confidence": "MBMH+both+TK+confsel",
    "single": "single-control",
    "real_only": "real-only",
    "mh_real_only": "MH-real-only",
    "matched_mb": "matched-MB",
}
CONTRASTS = (
    ("single", "multihead", "Single U-Net to multihead U-Net"),
    ("mb", "mbmh", "Three FPN bases: add multihead"),
    ("mbmh", "mbmh_refine", "Add JBF+CRF package"),
    ("mbmh_refine", "mbmh_full", "Add TK"),
    ("mbmh_full", "mbmh_confidence", "Random to confidence selection"),
    ("single", "mb", "Overall U-Net to three-FPN comparison; not an isolated MB effect"),
    ("mb", "mbmh_confidence", "MB to full option stack"),
)
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def number(value):
    if value is None or isinstance(value, bool):
        return ""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return result if math.isfinite(result) else ""


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        # Preserve columns that occur only in later history/manifest rows.
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_columns(prefix, metrics):
    result = {}
    for name in METRICS:
        result[prefix + name] = number(metrics.get(name))
    return result


def predictor_summaries(path):
    data = read_json(path)
    if not isinstance(data, dict):
        return {}
    result = {}
    for predictor, item in data.items():
        if not isinstance(item, dict) or not isinstance(item.get("summary"), dict):
            continue
        summary = item["summary"]
        result[predictor] = {
            "n_images": summary.get("n_images", ""),
            "n_nonempty_targets": summary.get("n_nonempty_targets", ""),
            "macro_image": summary.get("macro_image", {}),
            "macro_nonempty_target": summary.get("macro_nonempty_target", {}),
            "micro_pixels": summary.get("micro_pixels", {}),
        }
    return result


def run_folders(root):
    for config_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(dataset_dir.glob("seed_*")):
                if seed_dir.is_dir():
                    yield config_dir.name, dataset_dir.name, seed_dir


def export_run(root, config_name, dataset, folder, include_images):
    run_data = read_json(folder / "run.json") or {}
    config = run_data.get("config", {})
    complete = read_json(folder / "complete.json") or {}
    test_summary = read_json(folder / "test" / "summary.json") or {}
    split = read_json(folder / "split.json") or {}
    rows = split.get("rows", []) if isinstance(split, dict) else []
    split_counts = defaultdict(int)
    for row in rows:
        split_counts[row.get("split", "unknown")] += 1

    selected = complete.get("selected_round_by_validation",
                            test_summary.get("selected_round_by_validation", ""))
    status = "test_complete" if test_summary else (
        "training_complete_test_missing" if complete else (
            "run_metadata_only" if run_data else "partial_or_empty"))
    run_row = {
        "config": config_name,
        "arm": CONFIG_ARMS.get(config_name, config_name),
        "dataset": run_data.get("dataset", dataset),
        "seed": run_data.get("seed", folder.name[5:] if folder.name.startswith("seed_") else folder.name),
        "status": status,
        "run_dir": str(folder),
        "models": ";".join(config.get("models", [])),
        "multihead": config.get("multihead", ""),
        "refine": config.get("refine", ""),
        "final_refine": config.get("final_refine", ""),
        "tk": config.get("tk", ""),
        "selection": config.get("selection", ""),
        "pseudo_ratios": ";".join(map(str, config.get("pseudo_ratios", []))),
        "configured_rounds": config.get("rounds", ""),
        "eval_threshold": config.get("eval_threshold", ""),
        "synthetic_files": len(run_data.get("synthetic_files", []) or []),
        "n_train": split_counts.get("train", ""),
        "n_val": split_counts.get("val", ""),
        "n_test": split_counts.get("test", ""),
        "split_sha256": complete.get("split_sha256", run_data.get("split_sha256", "")),
        "code_sha256": run_data.get("code_sha256", ""),
        "selected_round_by_validation": selected,
        "provenance": test_summary.get("provenance", ""),
        "patient_independence_verified": test_summary.get("patient_independence_verified", ""),
    }
    # Keep nested provenance values readable and valid in CSV rather than using
    # Python's implementation-specific dict representation.
    if isinstance(run_row["code_sha256"], (dict, list)):
        run_row["code_sha256"] = json.dumps(run_row["code_sha256"], sort_keys=True)
    val_rounds = {int(item["round"]): item for item in complete.get("rounds", [])
                  if isinstance(item, dict) and "round" in item}
    test_rounds = {int(item["round"]): item for item in test_summary.get("all_predeclared_rounds", [])
                   if isinstance(item, dict) and "round" in item}
    max_round = max([int(config.get("rounds", -1))] + list(val_rounds) + list(test_rounds))
    for r in range(max_round + 1):
        val = val_rounds.get(r, {}).get("ensemble_validation", {})
        tst = test_rounds.get(r, {})
        run_row[f"r{r}_pseudo_count"] = val_rounds.get(r, {}).get("pseudo_count", "")
        for metric in METRICS:
            run_row[f"r{r}_val_{metric}"] = number(val.get(metric))
            run_row[f"r{r}_test_{metric}"] = number(tst.get(metric))

    selected_test = test_summary.get("selected_test", {})
    for metric in METRICS:
        run_row[f"selected_test_{metric}"] = number(selected_test.get(metric))

    round_rows, image_rows, history_rows, pseudo_rows = [], [], [], []
    known_rounds = sorted(set(range(max_round + 1)) | set(val_rounds) | set(test_rounds))
    for r in known_rounds:
        for evaluation_split, subdir in (("validation", "validation"), ("test", "test")):
            # Validation metrics are saved with each training round, whereas
            # held-out test metrics are grouped beneath test/round_n/.
            metrics_path = (folder / f"round_{r}" / "validation" / "metrics.json"
                            if evaluation_split == "validation" else
                            folder / "test" / f"round_{r}" / "metrics.json")
            predictors = predictor_summaries(metrics_path)
            for predictor, details in predictors.items():
                base = {
                    "config": config_name,
                    "arm": CONFIG_ARMS.get(config_name, config_name),
                    "dataset": dataset,
                    "seed": run_row["seed"],
                    "round": r,
                    "evaluation_split": evaluation_split,
                    "predictor": predictor,
                    "selected_by_validation": bool(selected != "" and int(selected) == r),
                    "n_images": details["n_images"],
                    "n_nonempty_targets": details["n_nonempty_targets"],
                    "run_dir": str(folder),
                }
                for averaging in ("macro_image", "macro_nonempty_target", "micro_pixels"):
                    base.update(metric_columns(averaging + "_", details.get(averaging, {})))
                round_rows.append(base)

            if include_images and evaluation_split == "test":
                metric_data = read_json(metrics_path)
                if isinstance(metric_data, dict):
                    for predictor, info in metric_data.items():
                        if not isinstance(info, dict):
                            continue
                        for item in info.get("per_image", []):
                            image_row = {
                                "config": config_name,
                                "arm": CONFIG_ARMS.get(config_name, config_name),
                                "dataset": dataset,
                                "seed": run_row["seed"],
                                "round": r,
                                "predictor": predictor,
                                "selected_by_validation": bool(selected != "" and int(selected) == r),
                                "image_id": item.get("id", ""),
                                "group": item.get("group", ""),
                                "tp": item.get("tp", ""),
                                "fp": item.get("fp", ""),
                                "fn": item.get("fn", ""),
                                "tn": item.get("tn", ""),
                                "run_dir": str(folder),
                            }
                            image_row.update(metric_columns("", item))
                            image_rows.append(image_row)

        for history_path in sorted((folder / f"round_{r}").glob("*_history.json")):
            history = read_json(history_path)
            if not isinstance(history, dict):
                continue
            for epoch_row in history.get("history", []):
                history_rows.append({
                    "config": config_name, "arm": CONFIG_ARMS.get(config_name, config_name),
                    "dataset": dataset, "seed": run_row["seed"], "round": r,
                    "model": (history_path.name[:-len("_history.json")]
                              if history_path.name.endswith("_history.json") else history_path.name),
                    "best_epoch": history.get("best_epoch", ""),
                    "best_dice": history.get("best_dice", ""),
                    **epoch_row,
                })

        selection_path = folder / f"round_{r}" / "pseudo" / "selection.csv"
        if selection_path.is_file():
            with selection_path.open(newline="", encoding="utf-8") as handle:
                for item in csv.DictReader(handle):
                    pseudo_rows.append({
                        "config": config_name, "arm": CONFIG_ARMS.get(config_name, config_name),
                        "dataset": dataset, "seed": run_row["seed"], "round": r,
                        **item,
                    })
    split_rows = []
    split_meta = {key: value for key, value in split.items() if key != "rows"} if isinstance(split, dict) else {}
    for item in rows:
        split_rows.append({
            "config": config_name,
            "dataset": dataset,
            "seed": run_row["seed"],
            "run_dir": str(folder),
            **split_meta,
            **item,
        })
    return run_row, round_rows, image_rows, history_rows, pseudo_rows, split_rows


def t_critical(n):
    df = n - 1
    return T95.get(df, 1.96 if df > 30 else "")


def make_pairs(run_rows, round_rows):
    run_index = {(r["config"], r["dataset"], str(r["seed"])): r for r in run_rows}
    round_index = {}
    for r in round_rows:
        if r["evaluation_split"] == "test" and r["predictor"] == "ensemble":
            round_index[(r["config"], r["dataset"], str(r["seed"]), int(r["round"]))] = r
    paired_rows = []
    modes = [(f"round_{r}", r, r) for r in range(3)] + [("validation_selected", None, None)]
    for config_a, config_b, label in CONTRASTS:
        datasets = sorted({key[1] for key in run_index if key[0] == config_a} |
                          {key[1] for key in run_index if key[0] == config_b})
        for dataset in datasets:
            seeds_a = {key[2] for key in run_index if key[0] == config_a and key[1] == dataset}
            seeds_b = {key[2] for key in run_index if key[0] == config_b and key[1] == dataset}
            for seed in sorted(seeds_a & seeds_b, key=lambda x: int(x)):
                meta_a = run_index[(config_a, dataset, seed)]
                meta_b = run_index[(config_b, dataset, seed)]
                if meta_a.get("split_sha256") and meta_b.get("split_sha256") and meta_a["split_sha256"] != meta_b["split_sha256"]:
                    continue
                for mode, fixed_a, fixed_b in modes:
                    if mode == "validation_selected":
                        ra = meta_a.get("selected_round_by_validation", "")
                        rb = meta_b.get("selected_round_by_validation", "")
                        if ra == "" or rb == "":
                            continue
                        round_a, round_b = int(ra), int(rb)
                    else:
                        round_a, round_b = fixed_a, fixed_b
                    row_a = round_index.get((config_a, dataset, seed, round_a))
                    row_b = round_index.get((config_b, dataset, seed, round_b))
                    if not row_a or not row_b:
                        continue
                    pair = {
                        "contrast": label,
                        "config_a": config_a,
                        "config_b": config_b,
                        "dataset": dataset,
                        "seed": seed,
                        "evaluation": mode,
                        "round_a": round_a,
                        "round_b": round_b,
                    }
                    for metric in METRICS:
                        a = number(row_a.get("macro_image_" + metric))
                        b = number(row_b.get("macro_image_" + metric))
                        pair[metric + "_a"] = a
                        pair[metric + "_b"] = b
                        pair[metric + "_delta_b_minus_a"] = (b - a) if a != "" and b != "" else ""
                    paired_rows.append(pair)
    return paired_rows


def summarize_pairs(pairs):
    grouped = defaultdict(list)
    for row in pairs:
        for metric in METRICS:
            value = row.get(metric + "_delta_b_minus_a", "")
            if value != "":
                grouped[(row["contrast"], row["config_a"], row["config_b"], row["dataset"],
                         row["evaluation"], metric)].append(float(value))
    out = []
    for key, values in sorted(grouped.items()):
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else ""
        half = ""
        lo = hi = ""
        if len(values) > 1:
            critical = t_critical(len(values))
            if critical != "":
                half = critical * float(sd) / math.sqrt(len(values))
                lo, hi = mean - half, mean + half
        contrast, config_a, config_b, dataset, evaluation, metric = key
        out.append({
            "contrast": contrast, "config_a": config_a, "config_b": config_b,
            "dataset": dataset, "evaluation": evaluation, "metric": metric,
            "n_paired_seeds": len(values), "mean_delta_b_minus_a": mean,
            "mean_delta_percentage_points": 100 * mean,
            "sd_delta": sd, "ci95_low": lo, "ci95_high": hi,
            "ci95_low_percentage_points": 100 * lo if lo != "" else "",
            "ci95_high_percentage_points": 100 * hi if hi != "" else "",
        })
    return out


def summarize_rounds(round_rows):
    groups = defaultdict(list)
    for row in round_rows:
        for metric in METRICS:
            value = row.get("macro_image_" + metric, "")
            if value != "":
                key = (row["config"], row["arm"], row["dataset"], row["evaluation_split"],
                       row["round"], row["predictor"], metric)
                groups[key].append(float(value))
    out = []
    for key, values in sorted(groups.items()):
        config, arm, dataset, split, r, predictor, metric = key
        out.append({
            "config": config, "arm": arm, "dataset": dataset,
            "evaluation_split": split, "round": r, "predictor": predictor,
            "metric": metric, "n_seeds": len(values),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else "",
        })
    return out


def make_coverage(run_rows):
    index = defaultdict(dict)
    for row in run_rows:
        if row["config"] in EXPECTED_CONFIGS and row["dataset"] in EXPECTED_DATASETS:
            index[(row["config"], row["dataset"])][int(row["seed"])] = row["status"]
    out = []
    for config in EXPECTED_CONFIGS:
        for dataset in EXPECTED_DATASETS:
            found = index.get((config, dataset), {})
            done = sorted(seed for seed, status in found.items() if status == "test_complete")
            missing = sorted(set(EXPECTED_SEEDS) - set(done))
            out.append({
                "config": config, "arm": CONFIG_ARMS[config], "dataset": dataset,
                "expected_seeds": ";".join(map(str, EXPECTED_SEEDS)),
                "test_complete_n": len(done), "test_complete_seeds": ";".join(map(str, done)),
                "missing_or_incomplete_seeds": ";".join(map(str, missing)),
            })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("revision/runs"))
    parser.add_argument("--out", type=Path, default=Path("revision/stage_analysis"))
    parser.add_argument("--images", action="store_true",
                        help="also export per-image test metrics; can create a large CSV")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error(f"Run folder not found: {args.runs}")

    inventories, all_rounds, all_images, all_histories, all_pseudo, all_splits = [], [], [], [], [], []
    for config, dataset, folder in run_folders(args.runs):
        result = export_run(args.runs, config, dataset, folder, args.images)
        run_row, rounds, images, histories, pseudo, splits = result
        inventories.append(run_row)
        all_rounds.extend(rounds)
        all_images.extend(images)
        all_histories.extend(histories)
        all_pseudo.extend(pseudo)
        all_splits.extend(splits)

    if not inventories:
        parser.error(f"No run folders found under {args.runs}")
    pairs = make_pairs(inventories, all_rounds)
    write_csv(args.out / "run_inventory.csv", inventories)
    write_csv(args.out / "round_metrics.csv", all_rounds)
    write_csv(args.out / "round_summary.csv", summarize_rounds(all_rounds))
    write_csv(args.out / "paired_differences.csv", pairs)
    write_csv(args.out / "paired_summary.csv", summarize_pairs(pairs))
    write_csv(args.out / "coverage.csv", make_coverage(inventories))
    write_csv(args.out / "training_history.csv", all_histories)
    write_csv(args.out / "pseudo_selection.csv", all_pseudo)
    write_csv(args.out / "split_membership.csv", all_splits)
    if args.images:
        write_csv(args.out / "test_image_metrics.csv", all_images)
    print(f"Scanned {len(inventories)} run folders; {sum(r['status'] == 'test_complete' for r in inventories)} have test summaries.")
    print(f"Exported round metrics, paired comparisons, coverage, split membership, training histories, and pseudo-selection records to {args.out}.")
    if args.images:
        print(f"Exported {len(all_images)} per-image test rows.")
    print("Selected-round comparisons use each run's validation-selected round; test scores never select rounds.")


if __name__ == "__main__":
    main()
