import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from utils import get_prompt, get_bnb_config
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('base_model_path', type=str, help='Path to the model checkpoint folder')
    parser.add_argument('peft_path', type=str, help='Path to the adapter checkpoint')
    parser.add_argument('input_file', type=str, help='Path to the input file (.json)')
    parser.add_argument('output_file', type=str, help='Path to the output file (.json)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    # Handle padding token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load PEFT adapter
    model = PeftModel.from_pretrained(model, args.peft_path)
    model.to(device)
    model.eval()

    # Load input data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare output list
    outputs = []

    # Generate outputs
    for item in tqdm(data):
        # Assuming each item has 'id' and 'instruction' fields
        if 'instruction' in item:
            input_text = get_prompt(item['instruction'])
        elif 'input' in item:
            input_text = get_prompt(item['input'])
        else:
            print(f"Item with id {item.get('id', 'unknown')} has no 'instruction' or 'input' field.")
            continue

        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove the prompt from the output to get only the generated response
        generated_output = output_text[len(input_text):].strip()
        outputs.append({
            'id': item['id'],
            'output': generated_output
        })

    # Save the outputs
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
