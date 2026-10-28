# SSGNet++ reproducibility package
set -euo pipefail
MODE="${1:-full}"; CONFIG="${2:-mb}"; DATASET="${3:-cbis}"; SEED="${4:-42}"; PROTOCOL="${5:-internal}"
PYTHON="${PYTHON:-python3}"; DEVICE="${DEVICE:-cuda:0}"; WORKERS="${WORKERS:-4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
case "$PROTOCOL" in
 internal) SPLIT_ROOT="${SPLIT_ROOT:-data/manifests/internal}"; OUT_ROOT="${OUT_ROOT:-${OUT_ROOT_BASE:-outputs/internal}/$CONFIG}" ;;
 external) SPLIT_ROOT="${SPLIT_ROOT:-data/manifests/external}"; OUT_ROOT="${OUT_ROOT:-${OUT_ROOT_BASE:-outputs/external}/$CONFIG}" ;;
 *) echo "Protocol must be internal or external" >&2; exit 2 ;;
esac
CONFIG_FILE="configs/$CONFIG.json"; SPLIT_FILE="$SPLIT_ROOT/$DATASET.json"
[[ -f "$CONFIG_FILE" && -f "$SPLIT_FILE" ]] || { echo "Missing config or split manifest" >&2; exit 2; }
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8 NO_ALBUMENTATIONS_UPDATE=1
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
case "$MODE" in
 check) "$PYTHON" src/ssgnet/preflight.py --config "$CONFIG_FILE" --split "$SPLIT_FILE" --real_root "${REAL_ROOT:-REAL}" --synth_root "${SYNTH_ROOT:-SYNTH}" --device "$DEVICE" --forward ;;
 full|all) "$PYTHON" src/ssgnet/run_experiment.py --config "$CONFIG_FILE" --split "$SPLIT_FILE" --real_root "${REAL_ROOT:-REAL}" --synth_root "${SYNTH_ROOT:-SYNTH}" --out_root "$OUT_ROOT" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS" ;;
 train) "$PYTHON" src/ssgnet/train.py --config "$CONFIG_FILE" --split "$SPLIT_FILE" --real_root "${REAL_ROOT:-REAL}" --synth_root "${SYNTH_ROOT:-SYNTH}" --out_root "$OUT_ROOT" --seeds "$SEED" --device "$DEVICE" --workers "$WORKERS" ;;
 test) "$PYTHON" src/ssgnet/evaluate.py --run "$OUT_ROOT/$DATASET/seed_$SEED" --real_root "${REAL_ROOT:-REAL}" --device "$DEVICE" --workers "$WORKERS" ;;
 is) mkdir -p results/inception_score; "$PYTHON" src/ssgnet/inception_score.py --images "${SYNTH_ROOT:-SYNTH}/$DATASET/train/images" --out "results/inception_score/${DATASET}_seed${SEED}.json" --device "$DEVICE" ;;
 *) echo "Usage: bash scripts/run_job.sh {check|full|train|test|is} CONFIG DATASET SEED [internal|external]" >&2; exit 2 ;;
esac
