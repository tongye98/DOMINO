import json 
import os
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm 

def final_prompt(synthetic_text):
    code = synthetic_text
    return f"""You are provided with a Python function and a valid input for that function. Given the valid input, try to run the function. You can list the intermediate steps, and finally provide the result. Present the final result in the form of an assert statement, enclosed within [ANSWER] and [/ANSWER] tags.
### Python Function and Valid Input:
{code}

### Response:
    """

def process_response_batch(batch, llm, params, tokenizer):
    synthetic_instructions = [item['synthetic_text'] for item in batch]

    prompts = []
    for synthetic_instruction in synthetic_instructions:
        format = final_prompt(synthetic_instruction)
        chat = [{"role": "user", "content": format}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)
    
    outputs = llm.generate(prompts, params)
    
    for i, item in enumerate(batch):
        item["original_response"] = outputs[i].outputs[0].text.strip()

    return batch

def extract_instruct_quality(instruction_quality_string):
    valid_quality = {'unknown', 'very poor', 'poor', 'average', 'good', 'excellent'}
    try:
        cleaned_instruction_quality_string = instruction_quality_string.replace("```json", "").replace("```", "").strip()
        instruction_quality_json = json.loads(cleaned_instruction_quality_string)
        input_quality = instruction_quality_json['input_quality']
    except:
        match = re.search(r'"input_quality"\s*:\s*"([^"]+)"', instruction_quality_string)
        if match:
            input_quality = match.group(1)
        else:
            input_quality = "unknown"
    
    input_quality = input_quality.lower()
    if input_quality not in valid_quality:
        input_quality = 'unknown'
        
    return input_quality

def get_vllm_configuration(inference_model, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    tokenizer = AutoTokenizer.from_pretrained(inference_model, trust_remote_code=True)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        
    print("Start Local vllm engine...")
    llm = LLM(model=inference_model, 
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=len(devices.split(',')),
        gpu_memory_utilization=0.95)

    params = SamplingParams(
        max_tokens=4096,
        temperature=0,
        top_p=1,
        repetition_penalty=1,
        stop_token_ids=stop_token_ids,
        )

    return llm, params, tokenizer

def generate_response(synthetic_instruct_path, inference_model, BATCH_SIZE, devices):
    synthetic_instruct_datasets = []
    with open(synthetic_instruct_path, 'r') as f:
        for line in f:
            synthetic_instruct_datasets.append(json.loads(line))
            
    print(f"len of synthetic instruct dataset = {len(synthetic_instruct_datasets)}")
    
    threshold_instruct_quality = {'excellent'}
    excellent_instruct_datasets = []
    for item in synthetic_instruct_datasets:
        instruction_quality_string = item['instruction_quality']
        input_quality = extract_instruct_quality(instruction_quality_string)
        
        if input_quality in threshold_instruct_quality:
            excellent_instruct_datasets.append(item)
            
    print(f"above threshold instruct count = {len(excellent_instruct_datasets)}")
    
    llm, params, tokenizer = get_vllm_configuration(inference_model, devices)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    num_batches = (len(excellent_instruct_datasets) + BATCH_SIZE -1) // BATCH_SIZE
    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = min((i+1) * BATCH_SIZE, len(excellent_instruct_datasets))
        batch = excellent_instruct_datasets[start_idx:end_idx]
        
        batch = process_response_batch(batch, llm, params, tokenizer)
        
        excellent_instruct_datasets[start_idx:end_idx] = batch
        
    saved_path = synthetic_instruct_path.replace(".jsonl", "_step_response.jsonl")
    with open(saved_path, 'w') as g:
        for item in excellent_instruct_datasets:
            g.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_stage", type=str, default='inference_response', choices=["inference_response"])
    parser.add_argument("--synthetic_instruct_path", type=str, required=True)
    parser.add_argument("--inference_model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--devices", type=str, default='0,1,2,3')
    args = parser.parse_args()
    
    if args.process_stage == "inference_response":
        generate_response(synthetic_instruct_path=args.synthetic_instruct_path,
                          inference_model=args.inference_model,
                          BATCH_SIZE=args.batch_size,
                          devices=args.devices)
    else:
        print("Check process stage.")
