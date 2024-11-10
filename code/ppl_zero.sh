# MODEL_NAME_OR_PATH="zake7749/gemma-2-2b-it-chinese-kyara-dpo"
# MODEL_NAME_OR_PATH="output/qlora_3/checkpoint-1250"
# MODEL_NAME_OR_PATH="yentinglin/Llama-3-Taiwan-8B-Instruct"
MODEL_NAME_OR_PATH="output_bonus/qlora_1/checkpoint-1250"
TEST_DATA_PATH="data/public_test.json"
PEFT_CHECKPOINT="output/qlora_3/checkpoint-1250"



python ppl_zero.py \
    --base_model_path $MODEL_NAME_OR_PATH \
    --test_data_path $TEST_DATA_PATH \
    # --peft_path $PEFT_CHECKPOINT