import os 
import json 
import re
import argparse
from tqdm import tqdm 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def input_quality_rating(input):
    user_message = f'''
# Instruction

You need to assess the quality of the given python function and corresponding input based on its clarity, specificity, completeness, and challenge.

The rating scale is as follows:

- very poor: The function is unclear, vague, or disorganized. It contains many unreadable characters for humans or has large scale repetitive parts.
- poor: The function is somewhat unclear or lacks important details, contains irrelevant content and repetition. The corresponding are partially wrong. It also contains a small number of human unreadable characters and minor repetition.
- average: The function has moderate clarity and specificity. The difficulty is relatively easy. The corresponding input are right.
- good: The function is clear and specific, and it is generally well-articulated. The corresponding input are reasonable and right.
- excellent: The function is very clear, specific and well - articulated, and poses a certain level of difficulty. The corresponding inputs are reasonable and right, being challenging and taking some edge cases into account.

## Function and Corresponding Input
```
{input}
```

## Output Format
Given the function, you first need to give an assesement, highlighting the strengths and/or weaknesses of the function and input.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
{{   
    "explanation": "[...]",
    "input_quality": "[very poor/poor/average/good/excellent]"
}}
'''
    return user_message

def get_vllm_configuration(GRADE_MODEL, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    tokenizer = AutoTokenizer.from_pretrained(GRADE_MODEL, trust_remote_code=True)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        
    print("Start Local vllm engine...")
    llm = LLM(model=GRADE_MODEL, 
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


def quality_process_batch(batch, llm, params, tokenizer):
    synthetic_instructions = [item['synthetic_text'] for item in batch]
    prompts = []
    for instruction in synthetic_instructions:
        instruction = input_quality_rating(input=instruction)
        chat = [{"role": "user", "content": instruction}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        item["instruction_quality"] = outputs[i].outputs[0].text.strip()
    
    return batch

def instruct_quality_assessment(synthetic_instruct_path, GRADE_MODEL, BATCH_SIZE, devices):
    synthetic_instruct_dataset = []
    with open(synthetic_instruct_path, 'r') as f:
        for line in f:
            synthetic_instruct_dataset.append(json.loads(line))
    print(f"synthetic instruction dataset length = {len(synthetic_instruct_dataset)}")
    
    llm, params, tokenizer = get_vllm_configuration(GRADE_MODEL, devices)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    num_batches = (len(synthetic_instruct_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = min((i+1) * BATCH_SIZE, len(synthetic_instruct_dataset))
        batch = synthetic_instruct_dataset[start_idx:end_idx]
        
        batch = quality_process_batch(batch, llm, params, tokenizer)
        
        synthetic_instruct_dataset[start_idx:end_idx] = batch
    
    saved_path = synthetic_instruct_path.replace(".jsonl", "_instruct_quality.jsonl")
    with open(saved_path, 'w') as g:
        for item in synthetic_instruct_dataset:
            g.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_stage", type=str, default='quality_assessment', choices=["quality_assessment"])
    parser.add_argument("--synthetic_instruct_path", type=str, required=True)
    parser.add_argument("--GRADE_MODEL", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--devices", type=str, default='0,1,2,3')
    args = parser.parse_args()
    
    if args.process_stage == "quality_assessment":
        instruct_quality_assessment(
            synthetic_instruct_path=args.synthetic_instruct_path,
            GRADE_MODEL=args.GRADE_MODEL,
            BATCH_SIZE=args.batch_size,
            devices=args.devices
        )
    else:
        print("Check process stage.")
