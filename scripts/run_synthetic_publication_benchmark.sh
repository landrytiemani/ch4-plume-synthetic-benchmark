#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="${MANIFEST:-data/training_l1c/segmentation_training_curated.csv}"
OUT_DIR="${OUT_DIR:-data/models/synthetic_publication_benchmark}"
MODELS="${MODELS:-unet attn_unet unet_pp deeplabv3p phys_tau_net}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATCH_SIZE="${PATCH_SIZE:-128}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-8192}"
VALIDATION_PATCHES="${VALIDATION_PATCHES:-1024}"
TEST_PATCHES="${TEST_PATCHES:-2048}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-7}"

MIN_PLUME_FRACTION="${MIN_PLUME_FRACTION:-0.01}"
MAX_PLUME_FRACTION="${MAX_PLUME_FRACTION:-0.20}"
MIN_TAU="${MIN_TAU:-0.004}"
MAX_TAU="${MAX_TAU:-0.055}"

mkdir -p "$OUT_DIR" data/outputs/tables data/outputs/publication_figures reports

echo "manifest=$MANIFEST"
echo "out_dir=$OUT_DIR"
echo "models=$MODELS"
echo "epochs=$EPOCHS patch_size=$PATCH_SIZE patches_per_epoch=$PATCHES_PER_EPOCH"
echo "validation_patches=$VALIDATION_PATCHES test_patches=$TEST_PATCHES"
echo "plume_fraction=$MIN_PLUME_FRACTION-$MAX_PLUME_FRACTION tau=$MIN_TAU-$MAX_TAU"

for MODEL in $MODELS; do
  echo
  echo "=== Synthetic training: $MODEL ==="
  python scripts/train_synthetic_model.py \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-epoch "$PATCHES_PER_EPOCH" \
    --validation-patches "$VALIDATION_PATCHES" \
    --min-plume-fraction "$MIN_PLUME_FRACTION" \
    --max-plume-fraction "$MAX_PLUME_FRACTION" \
    --min-tau "$MIN_TAU" \
    --max-tau "$MAX_TAU" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --device "$DEVICE"

  echo "=== Synthetic test evaluation: $MODEL ==="
  python scripts/evaluate_synthetic_model.py \
    --manifest "$MANIFEST" \
    --checkpoint "$OUT_DIR/$MODEL/best.pt" \
    --out-dir "$OUT_DIR/$MODEL" \
    --split TEST \
    --patch-size "$PATCH_SIZE" \
    --evaluation-patches "$TEST_PATCHES" \
    --batch-size "$BATCH_SIZE" \
    --min-plume-fraction "$MIN_PLUME_FRACTION" \
    --max-plume-fraction "$MAX_PLUME_FRACTION" \
    --min-tau "$MIN_TAU" \
    --max-tau "$MAX_TAU" \
    --num-workers "$NUM_WORKERS" \
    --seed "$((SEED + 20000))" \
    --device "$DEVICE"
done

python scripts/summarize_synthetic_publication_benchmark.py \
  --benchmark-dir "$OUT_DIR" \
  --out data/outputs/tables/synthetic_publication_benchmark_summary.csv \
  --report reports/synthetic_publication_benchmark_report.md \
  --models $MODELS

python scripts/plot_synthetic_publication_figures.py \
  --manifest "$MANIFEST" \
  --benchmark-dir "$OUT_DIR" \
  --out-dir data/outputs/publication_figures \
  --models $MODELS \
  --examples "${EXAMPLES:-6}" \
  --device "$DEVICE"
