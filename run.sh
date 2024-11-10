#!/bin/bash

# 確保傳入的參數有4個
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_path> <adapter_checkpoint> <input_data> <output_data>"
    exit 1
fi

# 將參數賦值給變量
MODEL_PATH=$1
ADAPTER_CHECKPOINT=$2
INPUT_DATA=$3
OUTPUT_DATA=$4

# 執行 Python 腳本並傳入參數
python infer.py "$MODEL_PATH" "$ADAPTER_CHECKPOINT" "$INPUT_DATA" "$OUTPUT_DATA"
