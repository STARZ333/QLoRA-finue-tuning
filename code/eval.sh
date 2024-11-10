BASE_MODEL_PATH="zake7749/gemma-2-2b-it-chinese-kyara-dpo"
TEST_DATA_PATH='data/public_test.json'
PEFT_CHECKPOINT='adapter_model'

python ppl.py --base_model_path $BASE_MODEL_PATH \
  --peft_path $PEFT_CHECKPOINT \
  --test_data_path $TEST_DATA_PATH