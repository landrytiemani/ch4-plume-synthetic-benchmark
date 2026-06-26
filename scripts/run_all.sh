#!/usr/bin/env bash
# run_all.sh — Steps 2–4: build manifest → train all models → evaluate
#
# Prerequisites: raw S2 L1C chips must already be in data/raw/sentinel2_l1c/exports/
# If you haven't downloaded chips yet, first run:
#   GEE_PROJECT=your-gcp-project bash scripts/acquire_s2_data.sh
#
# USAGE
#   bash scripts/run_all.sh
#
# Full pipeline from scratch:
#   GEE_PROJECT=your-gcp-project bash scripts/acquire_s2_data.sh
#   bash scripts/run_all.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Step 2: Build training manifest ==="
bash scripts/build_training_manifest.sh

echo "=== Step 3: Train benchmark models ==="
bash scripts/run_synthetic_publication_benchmark.sh
