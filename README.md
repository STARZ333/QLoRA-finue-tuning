# QLoRA-finue-tuning
Dataset & evaluation script for ADL 2024 homework 3


## How to start?
1. Create conda environment.
    ```bash
    conda create -n adlhw3 python=3.8
    conda activate adlhw3
    ```
2. Install library
    ```
    pip install torch transformers bitsandbytes peft datasets accelerate matplotlib
    ```
    Or
    ```
    pip install --no-deps -r requirement.txt
    ```

## Train
You can run the following command to finetune the model:
```bash
bash qlora.sh
```

## Evaluation
You can run the following command to evaluate:
```bash
bash eval.sh
```

## Prediction
You can run the following command to predict:
```bash
python infer.py "zake7749/gemma-2-2b-it-chinese-kyara-dpo" adapter_model/ data/private_test.json output_test.json
```
Or
```bash
bash run.sh "zake7749/gemma-2-2b-it-chinese-kyara-dpo" adapter_model/ data/private_test.json  output_test.json
```
### arguments
${1}: path to the model checkpoint folder

${2}: path to the adapter_checkpoint downloaded under your folder

${3}: path to the input file (.json)

${4}: path to the output file (.json)
