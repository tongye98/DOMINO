import os 
import re
import json
import argparse
import tree_sitter
import tree_sitter_python as tspython
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def response_quality_rating(coding_question, coding_solution):
    user_message = f'''
# Instruction

Your task is to evaluate the quality of the code solution for the given problem, focusing on the following three key dimensions: code completeness, functional correctness, and the ability of the code to pass the problem's test cases.

The rating is as follows:

- poor: The code solution is incomplete or has major errors, fails to solve the problem correctly, and does not pass the problem's test cases or has significant logical flaws.
- average: The code is mostly complete but has minor issues, solves the problem partially or with some logical flaws, and passes some test cases but fails others or has edge case issues.
- good: The code is complete and well-structured, solves the problem correctly, passes all test cases, and handles edge cases appropriately.

## Coding Question
```
{coding_question}
```

## Coding Solution
{coding_solution}

## Output Format
Given the coding solution, you first need to give an assesement, highlighting the strengths and/or weaknesses of the coding solution.
Then, you need to output a rating from poor to good by filling in the placeholders in [...]:
{{   
    "explanation": "[...]",
    "solution_quality": "[poor/average/good]"
}}
'''
    return user_message

def get_vllm_configuration(GRADE_MODEL, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    with open(os.path.join(os.path.dirname(__file__), '..', 'llm_configs.json'), "r") as f:
        model_configs = json.load(f)
        model_config = model_configs[GRADE_MODEL.split('/')[-1]]
        stop_token_ids = model_config["stop_token_ids"]
        
    print("Start Local vllm engine...")
    llm = LLM(model=GRADE_MODEL, 
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,
        tensor_parallel_size=len(devices.split(',')),
        gpu_memory_utilization=0.95)

    params = SamplingParams(
        max_tokens=4096,
        temperature=0,
        top_p=1,
        repetition_penalty=1,
        stop_token_ids=stop_token_ids,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(GRADE_MODEL)

    return llm, params, tokenizer

def response_quality_process_batch(batch, llm, params, tokenizer):
    instruction_response_pairs = [(item['synthetic_text'], item['original_response']) for item in batch]
    prompts = []
    for instruction_response_pair in instruction_response_pairs:
        format = response_quality_rating(instruction_response_pair[0], instruction_response_pair[1])
        chat = [{"role": "user", "content": format}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        item['response_quality'] = outputs[i].outputs[0].text.strip()

    return batch 

def response_quality_assessment(instruct_response_path, llm_assess_model, BATCH_SIZE, devices):
    instruct_response_dataset = []
    with open(instruct_response_path, 'r') as f:
        for line in f:
            instruct_response_dataset.append(json.loads(line))
    print(f"instruct & response dataset length = {len(instruct_response_dataset)}")
    
    llm, params, tokenizer = get_vllm_configuration(llm_assess_model, devices)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_batches = (len(instruct_response_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = min((i+1) * BATCH_SIZE, len(instruct_response_dataset))
        batch = instruct_response_dataset[start_idx:end_idx]
        
        batch = response_quality_process_batch(batch, llm, params, tokenizer)
        
        instruct_response_dataset[start_idx:end_idx] = batch   
    
    saved_path = instruct_response_path.replace(".jsonl", "_llm_quality.jsonl")
    with open(saved_path, 'w') as g:
        for item in instruct_response_dataset:
            g.write(json.dumps(item) + '\n')

def extract_code(text):
    code_match = re.search(r'```\s*([^`]*)```', text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ''
    if code.startswith('python'):
        code = code.replace('python', '').strip()
    return code

def syntax_check(parser, code):
    tree = parser.parse(code.encode())
    if tree.root_node.has_error:
        return False
    else:
        return True

def response_tree_sitter_check(instruct_response_path):
    instruct_response_dataset = []
    with open(instruct_response_path, 'r') as f:
        for line in f:
            instruct_response_dataset.append(json.loads(line))
    print(f"instruct & response dataset length = {len(instruct_response_dataset)}")
    
    language = tree_sitter.Language(tspython.language())
    parser = tree_sitter.Parser(language)
    
    syntex_correct_count = 0
    syntex_wrong_count = 0
    
    for item in tqdm(instruct_response_dataset):
        original_response = item['original_response']
        code = extract_code(original_response)
        
        if code:
            is_syntex_correct = syntax_check(parser, code)
            if is_syntex_correct: 
                syntex_correct_count += 1
                item['response_syntex'] = 'right'
            else: 
                syntex_wrong_count += 1
                item['response_syntex'] = 'wrong'
        else:
            syntex_wrong_count += 1
            item['response_syntex'] = 'unknown'

    print(f"syntex correct count = {syntex_correct_count}")
    print(f"syntex wrong count = {syntex_wrong_count}")
    
    saved_path = instruct_response_path.replace(".jsonl", "_syntex_quality.jsonl")
    with open(saved_path, 'w') as g:
        for item in instruct_response_dataset:
            g.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_stage", type=str, default='tree-sitter', choices=["response_llm_assessment", "tree-sitter"])
    parser.add_argument("--instruct_response_path", type=str, required=True)
    parser.add_argument("--llm_assess_model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--devices", type=str, default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    
    if args.process_stage == 'response_llm_assessment':
        response_quality_assessment(
            instruct_response_path=args.instruct_response_path,
            llm_assess_model=args.llm_assess_model,
            BATCH_SIZE=args.batch_size,
            devices=args.devices
            )
    elif args.process_stage == "tree-sitter":
        response_tree_sitter_check(
            instruct_response_path=args.instruct_response_path,
        )
    else:
        print("Check process stage.")
