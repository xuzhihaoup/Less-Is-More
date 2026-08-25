#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 {ph2|brats2020|covid19ct} DATASET_ROOT" >&2
    exit 2
fi

DATASET="$1"
DATASET_ROOT="$2"
EPOCHS="${EPOCHS:-320}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
AMP="${AMP:-true}"

case "${DATASET}" in
    ph2|brats2020|covid19ct) ;;
    *)
        echo "Unsupported dataset: ${DATASET}" >&2
        exit 2
        ;;
esac

MODELS=(unet cmunext mk_unet dag_unet ege_unet)

for MODEL in "${MODELS[@]}"; do
    COMMAND=(
        python -m less_is_more.cli
        --model "${MODEL}"
        --dataset "${DATASET}"
        --dataset-root "${DATASET_ROOT}"
        --run-name "${DATASET}_${MODEL}"
        --output-dir "${OUTPUT_DIR}"
        --epochs "${EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --learning-rate "${LEARNING_RATE}"
        --num-workers "${NUM_WORKERS}"
    )
    if [[ "${AMP}" == "true" ]]; then
        COMMAND+=(--amp)
    fi

    echo "Training ${MODEL} on ${DATASET}"
    "${COMMAND[@]}"
done
