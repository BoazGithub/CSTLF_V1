#!/bin/bash
# train.sh — CSTLF training launcher
# Usage: bash train.sh [DATASET]
# Example: bash train.sh sKwandaSCD_V1

DATASET=${1:-sKwandaSCD_V1}
CONFIG="configs/${DATASET}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG"
    echo "Available: sKwandaSCD_V1 | SECOND | LsSCD_Ex"
    exit 1
fi

echo "Training CSTLF on ${DATASET} ..."
python main.py --config "$CONFIG" --mode train

echo "Done. Checkpoints saved to checkpoints/${DATASET}/"
