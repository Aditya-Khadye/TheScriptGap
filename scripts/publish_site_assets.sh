#!/usr/bin/env bash
# Copy generated visualization assets into docs/ for GitHub Pages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/data/viz/heatmap.html"
DEST_DIR="$ROOT/docs/viz"
DEST="$DEST_DIR/heatmap.html"

if [[ ! -f "$SRC" ]]; then
  echo "Missing $SRC — run the viz stage first (python pipeline.py --stages viz)."
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"
echo "Published $SRC → $DEST"

if [[ -f "$ROOT/data/viz/heatmap_data.csv" ]]; then
  cp "$ROOT/data/viz/heatmap_data.csv" "$DEST_DIR/heatmap_data.csv"
  echo "Published heatmap_data.csv"
fi
