#!/usr/bin/env bash
# Run the monthly TheScriptGap data refresh locally or in CI.
#
# Usage:
#   ./scripts/run_monthly.sh              # light refresh (support, exposure, viz)
#   ./scripts/run_monthly.sh --full       # include diversity + complexity
#   ./scripts/run_monthly.sh --force      # pass --force to pipeline.py
#
# Environment (see README):
#   GOOGLE_FONTS_API, GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT
#   GOOGLE_FONTS_DIR (required for --full)
#   SKIP_BIGQUERY=1 to reuse existing BigQuery CSVs

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORCE=false
FULL=false

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    --full) FULL=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

PIPE_ARGS=()
if $FORCE; then
  PIPE_ARGS+=(--force)
fi

if $FULL; then
  STAGES=(all complexity)
else
  STAGES=(support exposure viz)
fi

echo "=== TheScriptGap monthly run (stages: ${STAGES[*]}) ==="

if $FULL; then
  if [[ -z "${GOOGLE_FONTS_DIR:-}" ]]; then
    echo "GOOGLE_FONTS_DIR must be set for --full (diversity + complexity)."
    exit 1
  fi
  if [[ ! -d "$GOOGLE_FONTS_DIR" ]]; then
    echo "Google Fonts directory not found: $GOOGLE_FONTS_DIR"
    exit 1
  fi
fi

python pipeline.py --stages "${STAGES[@]}" "${PIPE_ARGS[@]}"
bash "$ROOT/scripts/publish_site_assets.sh"

echo "=== Monthly run complete ==="
