#!/bin/bash

# Fingering Correction Model Training Script
# Uses GPU 0, epoch 10000, early stopping 30
# Only uses R1-checked data (101 pieces)

set -e

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate piannotate

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Fingering Correction Model Training"
echo "========================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Epochs: 10000 (with early stopping 30)"
echo "Using only R1-checked data (101 pieces)"
echo "========================================"

# Backup existing checkpoints
CHECKPOINT_DIR="model/checkpoints"
if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CHECKPOINT_DIR/best.pt" ]; then
    BACKUP_DIR="${CHECKPOINT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing checkpoints to $BACKUP_DIR"
    cp -r "$CHECKPOINT_DIR" "$BACKUP_DIR"
fi

python model/run_training.py \
    --rebuild-samples \
    --epochs 10000 \
    --batch-size 8 \
    --lr 1e-4 \
    --d-model 256 \
    --nhead 8 \
    --num-layers 4 \
    --dropout 0.1 \
    --max-seq-len 128 \
    --patience 30 \
    --output-dir model/checkpoints

echo "========================================"
echo "Training Complete!"
echo "========================================"
