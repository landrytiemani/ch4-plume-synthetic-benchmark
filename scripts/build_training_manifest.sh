#!/usr/bin/env bash
# build_training_manifest.sh — align plume masks to S2 chip grids and curate
#
# Run after acquire_s2_data.sh once all GeoTIFF chips are downloaded locally.
#
# USAGE
#   bash scripts/build_training_manifest.sh
#
# ENVIRONMENT VARIABLES
#   S2_DIR            directory containing downloaded S2 L1C chips
#   SPLIT_CATALOG     path to the L1C reference match catalog
#   OVERWRITE         set to 1 to overwrite existing aligned masks

set -euo pipefail

S2_DIR="${S2_DIR:-data/raw/sentinel2_l1c/exports}"
SPLIT_CATALOG="${SPLIT_CATALOG:-data/raw/sentinel2_l1c/sentinel2_l1c_reference_match_catalog.csv}"
OVERWRITE="${OVERWRITE:-0}"

OVERWRITE_ARG=""
if [ "$OVERWRITE" = "1" ]; then
  OVERWRITE_ARG="--overwrite"
fi

mkdir -p data/training_l1c/aligned_masks

# ── Step 1: align plume masks to S2 chip grids ──────────────────────────────
echo "[1/3] Building aligned training dataset..."
python -m ch4l1c.cli build-training-dataset \
  --split-catalog "$SPLIT_CATALOG" \
  --s2-dir "$S2_DIR" \
  --out-dir data/training_l1c/aligned_masks \
  --manifest-path data/training_l1c/segmentation_training_manifest.csv \
  --file-prefix-kind s2l1c_pair \
  $OVERWRITE_ARG

echo "  → wrote data/training_l1c/segmentation_training_manifest.csv"

# ── Step 2: audit aligned masks ──────────────────────────────────────────────
echo "[2/3] Auditing aligned training dataset..."
python -m ch4l1c.cli audit-training-dataset \
  --manifest data/training_l1c/segmentation_training_manifest.csv

# ── Step 3: curate (add quality flags and sample weights) ───────────────────
echo "[3/3] Curating training manifest (quality flags, sample weights)..."
python -m ch4l1c.cli curate-training-manifest \
  --manifest data/training_l1c/segmentation_training_manifest.csv \
  --out data/training_l1c/segmentation_training_curated.csv

echo ""
echo "Training manifest ready: data/training_l1c/segmentation_training_curated.csv"
echo "Next: run scripts/run_synthetic_publication_benchmark.sh"
