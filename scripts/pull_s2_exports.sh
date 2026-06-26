#!/usr/bin/env bash
set -euo pipefail

DRIVE_FOLDER="${1:-CH4_Plume_L1C_S2_pairs}"
OUT_DIR="${2:-data/sentinel2_l1c/exports}"

mkdir -p "$OUT_DIR" audit_logs
STAMP=$(date +%Y%m%d_%H%M%S)

rclone copy "gdrive:${DRIVE_FOLDER}" "$OUT_DIR" \
  --progress \
  --fast-list \
  --transfers 8 \
  --checkers 16 \
  --drive-chunk-size 256M \
  --log-file "audit_logs/l1c_pair_pull_${STAMP}.log" \
  --log-level INFO
