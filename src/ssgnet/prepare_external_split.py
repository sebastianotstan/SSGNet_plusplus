"""Build a fresh train/full-validation/external-test manifest.

Expected layout under REAL by default:
    REAL/<dataset>/train/{images,masks}
    REAL/<dataset>/val/{images,masks}
    REAL/<dataset>/external/{images,masks}

The original train split is preserved. All non-duplicate original validation
cases remain validation. Every external case is assigned to test.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from common import collect_pairs, validate_manifest, verify_files, write_json


def read_groups(path):
    mapping = {}
    if path is None:
        return mapping
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "group"}.issubset(reader.fieldnames):
            raise ValueError("groups CSV must have image and group columns")
        for row in reader:
            image = row["image"].strip()
            group = row["group"].strip()
            if not image or not group or image in mapping:
                raise ValueError("groups CSV has a blank image/group or duplicate image")
            mapping[image] = row
    return mapping


def apply_groups(rows, mapping):
    for row in rows:
        info = mapping.get(row["image"])
        if mapping and info is None:
            raise ValueError("Missing group mapping for " + row["image"])
        row["group"] = info["group"].strip() if info else row["pixel_sha256"]
        row["stratum"] = (info.get("class", "").strip() or "unknown") if info else "unknown"


def make_manifest(root, dataset, external_split, mapping):
    train_rows = collect_pairs(root, dataset, "train")
    old_val_rows = collect_pairs(root, dataset, "val")
    external_rows = collect_pairs(root, dataset, external_split)
    all_rows = train_rows + old_val_rows + external_rows
    apply_groups(all_rows, mapping)

    train_groups = {row["group"] for row in train_rows}
    train_pixels = {row["pixel_sha256"] for row in train_rows}
    excluded_val = [row for row in old_val_rows
                    if row["group"] in train_groups or row["pixel_sha256"] in train_pixels]
    excluded_ids = {row["id"] for row in excluded_val}

    rows = []
    for row in train_rows:
        row["split"] = "train"
        rows.append(row)
    for row in old_val_rows:
        if row["id"] not in excluded_ids:
            row["split"] = "val"
            rows.append(row)
    for row in external_rows:
        row["split"] = "test"
        rows.append(row)

    manifest = {
        "schema": 1,
        "dataset": dataset,
        "split_seed": None,
        "external_split_name": external_split,
        "counts": dict(Counter(row["split"] for row in rows)),
        "grouping": "provided_group_ids" if mapping else "image_content_only",
        "patient_independence_verified": False,
        "patient_independence_note": (
            "Verify group IDs are patient IDs and disjoint across train, validation, and external test."
            if mapping else "No case IDs provided; independence is checked by image content only."
        ),
        "provenance": "Original validation restored; external cohort assigned exclusively to test.",
        "gan_training_provenance": "StyleGAN was trained on original training data only (user confirmed).",
        "excluded_old_val_count": len(excluded_val),
        "excluded_old_val": excluded_val,
        "rows": rows,
    }
    validate_manifest(manifest)
    verify_files(rows, root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_root", type=Path, default=Path("REAL"))
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset folders that have an external split")
    parser.add_argument("--external_split", default="external",
                        help="Folder name under each dataset containing external/images and masks")
    parser.add_argument("--groups_csv", type=Path,
                        help="Optional CSV with image,group,class columns; image paths relative to REAL")
    parser.add_argument("--out", type=Path, default=Path("revision/external_splits"))
    args = parser.parse_args()

    root = args.real_root.resolve()
    mapping = read_groups(args.groups_csv)
    output_paths = [args.out / (dataset + ".json") for dataset in args.datasets]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing manifests: " + ", ".join(map(str, existing)))

    manifests = [make_manifest(root, dataset, args.external_split, mapping)
                 for dataset in args.datasets]
    for manifest, path in zip(manifests, output_paths):
        write_json(path, manifest)
        print(manifest["dataset"], manifest["counts"],
              "excluded original-validation duplicates:", manifest["excluded_old_val_count"])
    print("External cases are test-only. Image-content checks do not establish patient independence.")


if __name__ == "__main__":
    main()
