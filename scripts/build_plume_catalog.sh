#!/usr/bin/env bash
# build_plume_catalog.sh — Step 0: build the plume event catalog from public sources
#
# Downloads Carbon Mapper and EMIT methane plume data from their public APIs,
# converts rasters to standardised binary masks, matches each plume event to a
# Sentinel-2 L1C scene pair via Google Earth Engine, then assigns
# TRAIN/VAL/TEST splits using spatial blocking.
#
# Output: data/raw/splits/segmentation_split_catalog.csv
#
# PREREQUISITES
#   pip install -e ".[acquire,labels]"
#   earthaccess login (interactive, first run only):
#     python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
#
# USAGE
#   GEE_PROJECT=your-gcp-project bash scripts/build_plume_catalog.sh
#
# ENVIRONMENT VARIABLES
#   GEE_PROJECT       Google Cloud project for GEE authentication  (required)
#   CM_START_YEAR     Carbon Mapper start year                     (default: 2019)
#   CM_END_YEAR       Carbon Mapper end year                       (default: 2025)
#   EMIT_START        EMIT search start date YYYY-MM-DD            (default: 2022-08-01)
#   EMIT_END          EMIT search end date   YYYY-MM-DD            (default: 2026-01-01)
#   SKIP_CM           set to 1 to skip Carbon Mapper download
#   SKIP_EMIT         set to 1 to skip EMIT download
#   SKIP_MASKS        set to 1 to skip mask generation
#   SKIP_S2_MATCH     set to 1 to skip GEE scene matching
#   SKIP_SPLIT        set to 1 to skip split assignment

set -euo pipefail

GEE_PROJECT="${GEE_PROJECT:-}"
CM_START_YEAR="${CM_START_YEAR:-2019}"
CM_END_YEAR="${CM_END_YEAR:-2025}"
EMIT_START="${EMIT_START:-2022-08-01}"
EMIT_END="${EMIT_END:-2026-01-01}"
SKIP_CM="${SKIP_CM:-0}"
SKIP_EMIT="${SKIP_EMIT:-0}"
SKIP_MASKS="${SKIP_MASKS:-0}"
SKIP_S2_MATCH="${SKIP_S2_MATCH:-0}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"

if [ -z "$GEE_PROJECT" ] && [ "$SKIP_S2_MATCH" != "1" ]; then
  echo "ERROR: GEE_PROJECT is required for S2 scene matching."
  echo "       Set it or pass SKIP_S2_MATCH=1 to skip that step."
  exit 1
fi

mkdir -p data/carbon_mapper data/emit data/labels/source_masks data/raw/splits

# ── Step 0a: Carbon Mapper catalog (public API, no auth) ─────────────────────
if [ "$SKIP_CM" != "1" ]; then
  echo "[0a] Downloading Carbon Mapper CH4 plume catalog..."
  ch4l1c download-cm-catalog \
    --start-year "$CM_START_YEAR" \
    --end-year   "$CM_END_YEAR"

  echo "[0b] Downloading Carbon Mapper plume rasters..."
  ch4l1c download-cm-rasters
else
  echo "[0a/0b] Skipping Carbon Mapper download."
fi

# ── Step 0c: EMIT plume products (free NASA Earthdata account) ───────────────
if [ "$SKIP_EMIT" != "1" ]; then
  echo "[0c] Downloading EMIT CH4 plume products..."
  ch4l1c download-emit \
    --start-date "$EMIT_START" \
    --end-date   "$EMIT_END"
else
  echo "[0c] Skipping EMIT download."
fi

# ── Step 0d: Unified label catalog + binary masks ────────────────────────────
if [ "$SKIP_MASKS" != "1" ]; then
  echo "[0d] Building unified label catalog..."
  ch4l1c build-label-catalog

  echo "[0e] Converting plume rasters to binary masks..."
  ch4l1c build-source-masks
else
  echo "[0d/0e] Skipping mask generation."
fi

# ── Step 0f: GEE scene matching ──────────────────────────────────────────────
if [ "$SKIP_S2_MATCH" != "1" ]; then
  echo "[0f] Matching plume events to Sentinel-2 L1C scene pairs via GEE..."
  ch4l1c s2-l1c-reference-match \
    --project "$GEE_PROJECT" \
    --split-catalog data/labels/plume_label_catalog.csv
else
  echo "[0f] Skipping GEE scene matching."
fi

# ── Step 0g: Spatial-block TRAIN/VAL/TEST split assignment ──────────────────
if [ "$SKIP_SPLIT" != "1" ]; then
  echo "[0g] Assigning TRAIN/VAL/TEST splits (spatial blocking, 70/15/15)..."
  ch4l1c build-split-catalog
  ch4l1c audit-split-catalog
else
  echo "[0g] Skipping split assignment."
fi

echo ""
echo "Plume catalog built: data/raw/splits/segmentation_split_catalog.csv"
echo "Next: bash scripts/acquire_s2_data.sh"
