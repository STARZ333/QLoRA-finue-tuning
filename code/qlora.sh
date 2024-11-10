#!/bin/bash

# 训练脚本的路径
TRAIN_SCRIPT="qlora.py"

# 模型参数（保持不变或根据需要修改）
# MODEL_NAME_OR_PATH="zake7749/gemma-2-2b-it-chinese-kyara-dpo"
MODEL_NAME_OR_PATH="yentinglin/Llama-3-Taiwan-8B-Instruct"
TRUST_REMOTE_CODE="False"
USE_AUTH_TOKEN="False"

# 数据参数
DATASET="data/train.json"  # 将此替换为您的数据集文件路径
EVAL_DATASET="data/public_test.json"
DATASET_FORMAT="custom"  # 使用您在代码中定义的新数据集格式
EVAL_DATASET_SIZE=1024
MAX_TRAIN_SAMPLES=10000  # 根据需要调整
MAX_EVAL_SAMPLES=1000    # 根据需要调整
SOURCE_MAX_LEN=1024
TARGET_MAX_LEN=256

# 训练参数（保持不变或根据需要修改）
OUTPUT_DIR="./output_bonus/qlora_1"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
MAX_STEPS=1250
LEARNING_RATE=0.0001
FP16="True"
BF16="False"
FULL_FINETUNE="False"
BITS=4
LORA_R=32
LORA_ALPHA=16
LORA_DROPOUT=0.1
SAVE_STEPS=250
SAVE_TOTAL_LIMIT=40
EVALUATION_STRATEGY="steps"
EVAL_STEPS=250
LOGGING_STEPS=10
REPORT_TO="none"

# 生成参数
MAX_NEW_TOKENS=256
DO_SAMPLE="False"
NUM_BEAMS=1

LR_SCHEDULER_TYPE="constant"  # 可选：linear, cosine, polynomial, constant, constant_with_warmup 等
WARMUP_RATIO=0.03           # warmup 阶段占总训练步数的比例
NUM_WARMUP_STEPS=50          # 或者您可以指定具体的 warmup 步数

# 开始运行训练脚本
python $TRAIN_SCRIPT \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --eval_dataset $EVAL_DATASET \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --warmup_ratio $WARMUP_RATIO \
    --num_warmup_steps $NUM_WARMUP_STEPS \
    --trust_remote_code $TRUST_REMOTE_CODE \
    --use_auth_token $USE_AUTH_TOKEN \
    --dataset $DATASET \
    --dataset_format $DATASET_FORMAT \
    --eval_dataset_size $EVAL_DATASET_SIZE \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --source_max_len $SOURCE_MAX_LEN \
    --target_max_len $TARGET_MAX_LEN \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --fp16 $FP16 \
    --bf16 $BF16 \
    --full_finetune $FULL_FINETUNE \
    --bits $BITS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --evaluation_strategy $EVALUATION_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --report_to $REPORT_TO \
    --max_new_tokens $MAX_NEW_TOKENS \
    --do_sample $DO_SAMPLE \
    --num_beams $NUM_BEAMS \
    --do_train \
    --do_eval \
    --overwrite_output_dir