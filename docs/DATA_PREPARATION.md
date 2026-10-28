# Data preparation and manifests

The released manifests in `data/manifests/internal/` and `data/manifests/external/` are frozen specifications for the reported cohorts. They contain image-level relative paths and hashes, but no image pixels. Match the local files to the manifests with `preflight.py` before training.

To create a new post-hoc internal split from a local REAL dataset tree, use a fresh output directory because the utility will not overwrite manifests:

```bash
python src/ssgnet/prepare_splits.py --real_root REAL \
  --datasets cbis isic2017 isic2018 polyps \
  --out data/manifests/internal_fresh --test_fraction 0.5 --seed 2026
```

This split uses the former validation pool and is therefore not a historically untouched test cohort. If supplied case/patient grouping metadata exists, pass `--groups_csv`; absent identifiers, the tool falls back to image-content grouping, which cannot establish patient independence.

For external-data preparation, see `EXTERNAL_DATA.md`. When using manifests in a non-default directory, pass `SPLIT_ROOT=data/manifests/<your-folder>` to `scripts/run_job.sh` and `scripts/submit_experiments.sh`.
