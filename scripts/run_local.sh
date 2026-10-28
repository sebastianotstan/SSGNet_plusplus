#!/usr/bin/env bash
# Local single-run wrapper; activate your environment first.
set -euo pipefail
exec bash scripts/run_job.sh "${1:-full}" "${2:-mb}" "${3:-cbis}" "${4:-42}" "${5:-internal}"
