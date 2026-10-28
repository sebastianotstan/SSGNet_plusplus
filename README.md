# SSGNet++ reproducibility package

This repository contains the segmentation training/evaluation pipeline and reproducibility utilities used for the reported SSGNet++ experiments. It is arranged as a small research software project: executable Python code is in `src/ssgnet/`, experiment configurations in `configs/`, image-level split manifests in `data/manifests/`, launchers in `scripts/`, and protocol/setup notes in `docs/`.

The modeling, training, evaluation, refinement, and metric utilities in `src/ssgnet/` are copied from the locally tested implementation. The smoke-test harness and launch scripts use this repository layout; training/model behavior is not altered. The package does not include image datasets, synthetic images, model checkpoints, caches, per-image predictions, or intermediate run folders. It does not include StyleGAN3 training code; it documents the data-generation boundary and provides the Inception Score utility for already-generated image pools.

## Contents

- `src/ssgnet/`: training, validation-based checkpoint/round selection, pseudo-mask generation and refresh, held-out evaluation, optional refinements, metric auditing, split preparation and verification, seed/stage analysis, and Inception Score utility.
- `configs/`: the seven reported segmentation configurations and a small smoke-test configuration.
- `data/manifests/internal/` and `data/manifests/external/`: frozen image-level manifests. No source images or masks are included. Patient-level independence is not verified because identifiers were unavailable.
- `scripts/`: local and PBS launchers. HPC resources and environment setup need site-specific configuration.
- `StyleGAN3/README.md`: upstream StyleGAN3 links, train-only data layout, training command template, and runtime-planning notes; no upstream code or data are bundled.
- `docs/`: environment, data protocol, external cohort preparation, results interpretation, verification scope, and publication checklist.

## Data layout

Obtain the datasets independently and place data under these roots at the repository top level:

```text
REAL/<dataset>/{train,val,test}/{images,masks}/
SYNTH/<dataset>/train/images/
```

Dataset keys are `cbis`, `isic2017`, `isic2018`, and `polyps`. For the internal protocol, the manifest's test subset was carved post hoc from a previously used validation pool; it is not a historically untouched test set. External manifests use the separately assembled cohorts described in `docs/EXTERNAL_DATA.md`. Review `docs/DATA_PROTOCOL.md` before running; the manifest paths and SHA-256 checks must match your local files.

## Install and check

Use Python 3.10 or newer. Install a PyTorch/torchvision build appropriate for your CUDA system first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python src/ssgnet/smoke_test.py --out outputs/checks/smoke
```

Check one dataset/configuration and a GPU forward pass before training:

```bash
python src/ssgnet/preflight.py --config configs/mb.json \
  --split data/manifests/internal/cbis.json \
  --real_root REAL --synth_root SYNTH --device cuda:0 --forward
```

## One complete run

The following performs preflight, fresh Round 0/1/2 training, validation-only checkpoint/round selection, pseudo-mask refresh, and final held-out evaluation:

```bash
bash scripts/run_local.sh full mb cbis 42 internal
```

Change `mb` to `multihead`, `mbmh`, `mbmh_refine`, `mbmh_full`, `mbmh_confidence`, or `single`; change `cbis` to one of the other dataset keys. Use `external` as the last argument for the external-cohort protocol. Each output path is new-only: choose a fresh `OUT_ROOT`/`OUT_ROOT_BASE` for reruns.

## HPC batch runs

Copy `scripts/hpc_env.example.sh` to the ignored `scripts/hpc_env.sh`, then set your cluster's module/Conda activation, Python path, and resource requirements. Review the PBS directives in `scripts/run_hpc.pbs` for your HPC site. From the repository root:

```bash
bash scripts/submit_experiments.sh --dry-run --protocol internal
bash scripts/submit_experiments.sh --check --protocol internal
bash scripts/submit_experiments.sh --submit --protocol internal
```

The full matrix is 7 arms × 4 datasets × 4 seeds (112 jobs). `--dry-run` prints commands; `--check` submits preflight jobs for seed 42 only; `--submit` submits the full matrix. The scheduler controls when jobs start. For external cohorts, replace `internal` with `external`. Set `OUT_ROOT_BASE` to a new path for fresh reruns. These scripts were adapted from the local/HPC launchers; validate resource directives and environment setup on your site before a large batch.

## Analysis and synthetic-image score

See `docs/RESULTS.md` for the scope and limitations of the reported follow-up results. The analysis tools are `src/ssgnet/stage_analysis.py`, `compare_seeds.py`, `summarize_results.py`, and `audit_metrics.py`. Run `python src/ssgnet/inception_score.py --help` to score generated image folders. IS uses ImageNet Inception features and is a supplementary diagnostic, not a measure of clinical realism or downstream task utility.

## Reuse and citation

The original SSGNet++ code in this repository is released under the MIT License; see `LICENSE`. This licence does not cover datasets, model weights, StyleGAN3, or other third-party components, which remain subject to their own terms. Cite the article using `CITATION.cff`. See `docs/RELEASE_CHECKLIST.md` for remaining publication checks.
