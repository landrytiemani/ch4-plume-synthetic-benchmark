#!/usr/bin/env bash
# acquire_s2_data.sh — full Sentinel-2 L1C data acquisition pipeline
#
# PREREQUISITES
#   1. Google Earth Engine account with API access
#      pip install earthengine-api && earthengine authenticate
#   2. rclone configured with a Google Drive remote named "gdrive"
#      rclone config  →  name it "gdrive", type Google Drive
#   3. Plume catalog built by Step 0 (scripts/build_plume_catalog.sh)
#      Generates: data/raw/splits/segmentation_split_catalog.csv
#
# USAGE
#   bash scripts/acquire_s2_data.sh                   # full run
#   LIMIT=20 bash scripts/acquire_s2_data.sh          # smoke test
#
# ENVIRONMENT VARIABLES
#   GEE_PROJECT       Google Cloud project for Earth Engine  (default: none)
#   DRIVE_FOLDER      Google Drive folder name               (default: CH4_Plume_L1C_S2_pairs)
#   OUT_DIR           local directory for downloaded chips   (default: data/raw/sentinel2_l1c/exports)
#   SPLIT_CATALOG     path to the split catalog CSV          (default: data/splits/segmentation_split_catalog.csv)
#   LIMIT             max rows to process (for smoke tests)  (default: none = all)
#   SKIP_MATCH        set to 1 to skip the GEE matching step (use existing catalog)
#   SKIP_EXPORT       set to 1 to skip GEE export queueing
#   SKIP_PULL         set to 1 to skip rclone download

set -euo pipefail

GEE_PROJECT="${GEE_PROJECT:-}"
DRIVE_FOLDER="${DRIVE_FOLDER:-CH4_Plume_L1C_S2_pairs}"
OUT_DIR="${OUT_DIR:-data/raw/sentinel2_l1c/exports}"
SPLIT_CATALOG="${SPLIT_CATALOG:-data/raw/splits/segmentation_split_catalog.csv}"
LIMIT="${LIMIT:-}"
SKIP_MATCH="${SKIP_MATCH:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PULL="${SKIP_PULL:-0}"

MATCH_CATALOG="data/raw/sentinel2_l1c/sentinel2_l1c_reference_match_catalog.csv"
EXPORT_MANIFEST="data/raw/sentinel2_l1c/sentinel2_l1c_pair_export_manifest.csv"

mkdir -p data/raw/sentinel2_l1c/exports data/raw/splits audit_logs

STAMP=$(date +%Y%m%d_%H%M%S)
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi
PROJECT_ARG=""
if [ -n "$GEE_PROJECT" ]; then
  PROJECT_ARG="--project $GEE_PROJECT"
fi

# ── Step 1: match each plume event to an L1C event/reference pair ──────────
if [ "$SKIP_MATCH" != "1" ]; then
  echo "[1/4] Matching plume events to Sentinel-2 L1C event/reference pairs..."
  python -m ch4l1c.cli s2-l1c-reference-match \
    --split-catalog "$SPLIT_CATALOG" \
    --out "$MATCH_CATALOG" \
    $PROJECT_ARG \
    $LIMIT_ARG \
    2>&1 | tee "audit_logs/s2_l1c_match_${STAMP}.log"
  echo "  → wrote $MATCH_CATALOG"
else
  echo "[1/4] Skipping GEE match (SKIP_MATCH=1), using existing $MATCH_CATALOG"
fi

# ── Step 2: queue exports to Google Drive ───────────────────────────────────
if [ "$SKIP_EXPORT" != "1" ]; then
  echo "[2/4] Queuing Sentinel-2 L1C pair chip exports to Google Drive..."
  python -m ch4l1c.cli s2-l1c-export-pairs \
    --match-catalog "$MATCH_CATALOG" \
    --out-manifest "$EXPORT_MANIFEST" \
    --drive-folder "$DRIVE_FOLDER" \
    $PROJECT_ARG \
    $LIMIT_ARG \
    2>&1 | tee "audit_logs/s2_l1c_export_${STAMP}.log"
  echo "  → queued exports; monitor at https://code.earthengine.google.com/tasks"
  echo "  → wrote manifest $EXPORT_MANIFEST"
  echo ""
  echo "  Waiting for GEE exports to complete..."
  echo "  Check task status at https://code.earthengine.google.com/tasks"
  echo "  Re-run with SKIP_MATCH=1 SKIP_EXPORT=1 when all tasks show COMPLETED."
  echo ""
  read -p "  Press Enter once all GEE export tasks are COMPLETED..."
else
  echo "[2/4] Skipping GEE export queueing (SKIP_EXPORT=1)"
fi

# ── Step 3: download from Google Drive ─────────────────────────────────────
if [ "$SKIP_PULL" != "1" ]; then
  echo "[3/4] Downloading exported chips from Google Drive (via rclone)..."
  mkdir -p "$OUT_DIR" audit_logs
  rclone copy "gdrive:${DRIVE_FOLDER}" "$OUT_DIR" \
    --progress \
    --fast-list \
    --transfers 8 \
    --checkers 16 \
    --drive-chunk-size 256M \
    --log-file "audit_logs/rclone_pull_${STAMP}.log" \
    --log-level INFO
  echo "  → downloaded chips to $OUT_DIR"
else
  echo "[3/4] Skipping rclone download (SKIP_PULL=1)"
fi

# ── Step 4: audit downloads ──────────────────────────────────────────────────
echo "[4/4] Auditing downloaded GeoTIFF files..."
python -m ch4l1c.cli audit-s2-exports \
  --manifest "$EXPORT_MANIFEST" \
  --download-dir "$OUT_DIR" \
  2>&1 | tee "audit_logs/s2_l1c_audit_${STAMP}.log"

echo ""
echo "Data acquisition complete. Next: run scripts/build_training_manifest.sh"
