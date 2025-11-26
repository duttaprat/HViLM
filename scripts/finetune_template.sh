#!/usr/bin/env bash

################################################################################
# HViLM Fine-tuning Script Template
# 
# This script fine-tunes HViLM on your custom viral genomics classification task.
# Customize the variables below for your specific dataset and task.
################################################################################

###########################
# GPU CONFIGURATION
###########################
# Specify which GPUs to use (comma-separated, e.g., "0,1,2,3")
export CUDA_VISIBLE_DEVICES=0

###########################
# MODEL CONFIGURATION
###########################
# HViLM base model from Hugging Face
export MODEL_NAME="duttaprat/HViLM-base"

# Task name (for organizing outputs and W&B logging)
export TASK_NAME="my_viral_classification_task"

###########################
# DATA CONFIGURATION
###########################
# Path to your data folder containing train.csv, dev.csv, test.csv
# Each CSV should have columns: sequence,label
export DATA_PATH="./data/my_dataset"

# Model max length (~25% of your sequence length due to BPE tokenization)
# For 1000bp sequences, use 250
# For 500bp sequences, use 125
export MAX_LENGTH=250

###########################
# TRAINING HYPERPARAMETERS
###########################
export LR=3e-5                    # Learning rate
export NUM_EPOCHS=10              # Number of training epochs
export TRAIN_BATCH=8              # Training batch size per GPU
export EVAL_BATCH=16              # Evaluation batch size per GPU
export GRAD_ACC=1                 # Gradient accumulation steps
export WARMUP_STEPS=50            # Warmup steps
export SAVE_STEPS=200             # Save checkpoint every N steps
export EVAL_STEPS=200             # Evaluate every N steps
export LOG_STEPS=100              # Log metrics every N steps

###########################
# OPTUNA HYPERPARAMETER SEARCH (Optional)
###########################
# Set to 0 to skip hyperparameter search and use values above
# Set to >0 to run Optuna hyperparameter search
export NUM_TRIALS=0

###########################
# OUTPUT CONFIGURATION
###########################
export OUTPUT_DIR="./output/${TASK_NAME}"

###########################
# WANDB CONFIGURATION (Optional)
###########################
# Set these if you want to track experiments with Weights & Biases
# Leave empty to disable W&B logging
export WANDB_PROJECT="HViLM-Finetune"          # Your W&B project name
export WANDB_ENTITY=""                          # Your W&B username/team (optional)
export WANDB_TAGS="${TASK_NAME}"               # Tags for this run
export RUN_NAME="HViLM_${TASK_NAME}"           # Run name

###########################
# LORA CONFIGURATION
###########################
# LoRA enables parameter-efficient fine-tuning
export USE_LORA=True
export LORA_R=8                   # LoRA rank
export LORA_ALPHA=16              # LoRA alpha
export LORA_DROPOUT=0.1           # LoRA dropout

################################################################################
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
################################################################################

echo "========================================="
echo "HViLM Fine-tuning Configuration"
echo "========================================="
echo "Model:           ${MODEL_NAME}"
echo "Task:            ${TASK_NAME}"
echo "Data path:       ${DATA_PATH}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Max length:      ${MAX_LENGTH}"
echo "Learning rate:   ${LR}"
echo "Epochs:          ${NUM_EPOCHS}"
echo "Batch size:      ${TRAIN_BATCH}"
echo "Optuna trials:   ${NUM_TRIALS}"
if [ -n "$WANDB_PROJECT" ]; then
    echo "W&B project:     ${WANDB_PROJECT}"
fi
echo "========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Determine which training script to use
if [ "$NUM_TRIALS" -gt 0 ]; then
    TRAIN_SCRIPT="train_optuna.py"
    echo "Using Optuna hyperparameter search (${NUM_TRIALS} trials)"
else
    TRAIN_SCRIPT="train.py"
    echo "Using standard training (no hyperparameter search)"
fi

# Build command
CMD="python finetune/${TRAIN_SCRIPT} \
  --model_name_or_path ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --kmer -1 \
  --run_name ${RUN_NAME} \
  --model_max_length ${MAX_LENGTH} \
  --per_device_train_batch_size ${TRAIN_BATCH} \
  --per_device_eval_batch_size ${EVAL_BATCH} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --learning_rate ${LR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --fp16 \
  --save_steps ${SAVE_STEPS} \
  --output_dir ${OUTPUT_DIR} \
  --evaluation_strategy steps \
  --eval_steps ${EVAL_STEPS} \
  --warmup_steps ${WARMUP_STEPS} \
  --logging_steps ${LOG_STEPS} \
  --overwrite_output_dir True \
  --log_level info \
  --find_unused_parameters False \
  --load_best_model_at_end True \
  --save_total_limit 3"

# Add LoRA parameters if enabled
if [ "$USE_LORA" = "True" ]; then
    CMD="${CMD} \
  --use_lora \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT}"
fi

# Add W&B parameters if configured
if [ -n "$WANDB_PROJECT" ]; then
    CMD="${CMD} --report_to wandb"
fi

# Add Optuna trials if doing hyperparameter search
if [ "$NUM_TRIALS" -gt 0 ]; then
    CMD="${CMD} --n_trials ${NUM_TRIALS}"
fi

# Run training
echo ""
echo "Starting training..."
echo ""
eval $CMD

echo ""
echo "========================================="
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
if [ -n "$WANDB_PROJECT" ] && [ -n "$WANDB_ENTITY" ]; then
    echo "View results: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
elif [ -n "$WANDB_PROJECT" ]; then
    echo "View results on your W&B dashboard: ${WANDB_PROJECT}"
fi
echo "========================================="
