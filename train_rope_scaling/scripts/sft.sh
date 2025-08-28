#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts

export CUDA_VISIBLE_DEVICES=3
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
NPROC_PER_NODE=1                            # Number of GPUs per node (set manually for single GPU)

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"  # [ModelArguments] Pretrained model path
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models
DATASETS="gqa_spatial_60000"                  # [DataArguments] Dataset with sampling rate

# ======================
# Basic Params
# ======================
MODEL_NAME="qwen2_vl_2b"
LR=3e-6
EPOCHS=3
BATCH_SIZE_PER_GPU=8                     # [TrainingArguments] Batch size per GPU
GRAD_ACC_STEPS=4
BATCH_SIZE=$(($BATCH_SIZE_PER_GPU * $GRAD_ACC_STEPS))  # Total batch size across all GPUs

# ======================
# LoRA Params
# ======================
USE_LORA="false"
LORA_RANK=16
LORA_ALPHA=32

# ======================
# RoPE Params
# ======================
ROPE_SCALING="true"
SCALING_TYPE="poly"  # options: sigmoid, poly
POLY_ALPHA=99
POLY_P=10
SIG_ALPHA=99
SIG_MID_POINT=0.6
SIG_K=40

WORK_DIR="..."  # Working directory for saving outputs
# set output dir                     # [ModelArguments] Whether to use RoPE scaling
if [ "$ROPE_SCALING" = "true" ]; then
    OUTPUT_DIR="${WORK_DIR}/checkpoints/rope_scaling"
else
    OUTPUT_DIR="${WORK_DIR}/checkpoints/normal"
fi

# set run name
if [ "$USE_LORA" = "true" ]; then
    run_name="lora-${MODEL_NAME}-data_${DATASETS}-epoch_${EPOCHS}-bsz_${BATCH_SIZE}-lr_${LR}-rank_${LORA_RANK}-alpha_${LORA_ALPHA}"
else
    run_name="sft-${MODEL_NAME}-data_${DATASETS}-epoch_${EPOCHS}-bsz_${BATCH_SIZE}-lr_${LR}"
fi

if [ "$ROPE_SCALING" = "true" ]; then
    run_name="${run_name}-scaling_type_${SCALING_TYPE}"
    if [ "$SCALING_TYPE" = "poly" ]; then
        run_name="${run_name}-poly_alpha_${POLY_ALPHA}-poly_p_${POLY_P}"
    elif [ "$SCALING_TYPE" = "sigmoid" ]; then
        run_name="${run_name}-sig_alpha_${SIG_ALPHA}-sig_mid_point_${SIG_MID_POINT}-sig_k_${SIG_K}"
    fi
fi

# Create output directory
OUTPUT_DIR="${OUTPUT_DIR}/${run_name}"  # Output directory for this run
mkdir -p $OUTPUT_DIR

# torchrun --nproc_per_node=$NPROC_PER_NODE \
#          --master_addr=$MASTER_ADDR \
#          --master_port=$MASTER_PORT \

ROOT_DIR="..."  # Root directory of the project
python -u ${ROOT_DIR}/train/qwen2_vl_sft_single_gpu.py \
        --model_name_or_path $MODEL_PATH \
        --dataset_use $DATASETS \
        --output_dir $OUTPUT_DIR \
        --run_name ${run_name} \
        --cache_dir $CACHE_DIR \
        --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
        --learning_rate ${LR} \
        --rope_scaling ${ROPE_SCALING} \
        --scaling_type ${SCALING_TYPE} \
        --poly_alpha ${POLY_ALPHA} \
        --poly_p ${POLY_P} \
        --sig_alpha ${SIG_ALPHA} \
        --sig_mid_point ${SIG_MID_POINT} \
        --sig_k ${SIG_K} \
        --use_lora ${USE_LORA} \
        --lora_r ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
        --lora_dropout 0.05 \
        --bf16 \
        --model_max_length 4096 \
        --data_flatten False \
        --data_packing False \
        --dataloader_num_workers 4 \
        --max_pixels 50176 \
        --min_pixels 784 \
        --num_train_epochs ${EPOCHS} \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --weight_decay 0.01 \
        --log_level info \
        --logging_steps 10 \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 15 \
        --gradient_checkpointing False \
        --report_to wandb \
        2>&1 | tee ${OUTPUT_DIR}/train.log
