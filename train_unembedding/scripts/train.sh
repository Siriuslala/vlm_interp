#!/bin/bash

set -e

MODEL_NAME="qwen2_5_vl"  # "llava1_5_7b"  "qwen2_5_vl"
WORK_DIR="..."
SAVE_DIR="${WORK_DIR}/checkpoints_vision_decoder"
DEVICE="cuda:0"

EPOCHS=1
BATCH_SIZE=8
GRAD_ACCUMU_STEPS=4
LEARNING_RATE=2e-4
WARMUP_STEPS=1000
WEIGHT_DECAY=0.01
PATIENCE=10

ALPHA=0.8
TEMPERATURE=4.0

LOG_STEPS=10
EVAL_STEPS=1000
SAVE_STEPS=1000

REAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUMU_STEPS))

RUN_NAME="${MODEL_NAME}-epoch${EPOCHS}-bsz${REAL_BATCH_SIZE}-lr${LEARNING_RATE}-warmup${WARMUP_STEPS}-alpha${ALPHA}-temp${TEMPERATURE}-patience${PATIENCE}"


echo "========================================================"
echo "Starting Training Run"
echo "========================================================"
echo "Model Name:     ${MODEL_NAME}"
echo "Run Name:       ${RUN_NAME}"
echo "Save Directory: ${SAVE_DIR}"
echo "Learning Rate:  ${LEARNING_RATE}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Patience:       ${PATIENCE}"
echo "Alpha:          ${ALPHA}"
echo "Temperature:    ${TEMPERATURE}"
echo "========================================================"

ROOT_DIR="..."
cd ${ROOT_DIR}/train_unembedding
python train.py \
    --model_name ${MODEL_NAME} \
    --run_name ${RUN_NAME} \
    --save_dir ${SAVE_DIR} \
    --device ${DEVICE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUMU_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --patience ${PATIENCE} \
    --alpha ${ALPHA} \
    --temperature ${TEMPERATURE} \
    --log_steps ${LOG_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS}

echo "========================================================"
echo "Training run ${RUN_NAME} finished."
echo "========================================================"