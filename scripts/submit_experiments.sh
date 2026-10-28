#!/usr/bin/env bash
set -euo pipefail
ACTION=--dry-run; PROTOCOL=internal; ARM=all
while (($#)); do case "$1" in
 --dry-run|--check|--submit) ACTION="$1" ;;
 --protocol) shift; PROTOCOL="${1:?--protocol requires internal or external}" ;;
 --arm) shift; ARM="${1:?--arm requires an arm name}" ;;
 *) echo "Usage: bash scripts/submit_experiments.sh [--dry-run|--check|--submit] [--protocol internal|external] [--arm all|mb|mh|mbmh|both|tk|confsel|single]" >&2; exit 2 ;;
esac; shift; done
DATASETS=(cbis isic2017 isic2018 polyps); SEEDS=(42 123 2024 3407)
case "$ARM" in all) CONFIGS=(mb multihead mbmh mbmh_refine mbmh_full mbmh_confidence single);; mb) CONFIGS=(mb);; mh) CONFIGS=(multihead);; mbmh) CONFIGS=(mbmh);; both) CONFIGS=(mbmh_refine);; tk) CONFIGS=(mbmh_full);; confsel) CONFIGS=(mbmh_confidence);; single) CONFIGS=(single);; *) echo "Unknown arm: $ARM" >&2; exit 2;; esac
case "$PROTOCOL" in internal) SPLIT_ROOT="${SPLIT_ROOT:-data/manifests/internal}"; OUT_BASE="${OUT_ROOT_BASE:-outputs/internal}";; external) SPLIT_ROOT="${SPLIT_ROOT:-data/manifests/external}"; OUT_BASE="${OUT_ROOT_BASE:-outputs/external}";; *) echo "Protocol must be internal or external" >&2; exit 2;; esac
MODE=full; SELECTED_SEEDS=("${SEEDS[@]}"); if [[ "$ACTION" == --check ]]; then MODE=check; SELECTED_SEEDS=(42); fi
for c in "${CONFIGS[@]}"; do [[ -f "configs/$c.json" ]] || { echo "Missing config $c" >&2; exit 2; }; for d in "${DATASETS[@]}"; do [[ -f "$SPLIT_ROOT/$d.json" ]] || { echo "Missing manifest $d" >&2; exit 2; }; done; done
if [[ "$MODE" == full ]]; then for c in "${CONFIGS[@]}"; do for d in "${DATASETS[@]}"; do for s in "${SELECTED_SEEDS[@]}"; do [[ ! -e "$OUT_BASE/$c/$d/seed_$s" ]] || { echo "Output exists: $OUT_BASE/$c/$d/seed_$s; set OUT_ROOT_BASE to a fresh root." >&2; exit 2; }; done; done; done; fi
if [[ "$ACTION" != --dry-run ]] && ! command -v qsub >/dev/null 2>&1; then echo "qsub unavailable; use --dry-run locally or submit on HPC." >&2; exit 2; fi
n=0; for c in "${CONFIGS[@]}"; do for d in "${DATASETS[@]}"; do for s in "${SELECTED_SEEDS[@]}"; do n=$((n+1)); name="${PROTOCOL:0:3}_${c}_${d}_s${s}"; vars="MODE=$MODE,PROTOCOL=$PROTOCOL,CONFIG=$c,DATASET=$d,SEED=$s,OUT_ROOT_BASE=$OUT_BASE,SPLIT_ROOT=$SPLIT_ROOT"; if [[ "$ACTION" == --dry-run ]]; then printf 'qsub -N %q -v %q scripts/run_hpc.pbs\n' "$name" "$vars"; else id=$(qsub -N "$name" -v "$vars" scripts/run_hpc.pbs); echo "$id protocol=$PROTOCOL config=$c dataset=$d seed=$s mode=$MODE"; fi; done; done; done
if [[ "$ACTION" == --dry-run ]]; then echo "$n PBS commands printed; none submitted."; elif [[ "$ACTION" == --check ]]; then echo "$n preflight jobs submitted."; else echo "$n experiment jobs submitted."; fi
