import os
import re
import json
import argparse

def extract_response_quality(response_llm_quality):
    valid_quality = {'poor', 'average', 'good'}
    try:
        cleaned_response_quality_string = response_llm_quality.replace("```json", "").replace("```", "").strip()
        response_quality_json = json.loads(cleaned_response_quality_string)
        response_quality = response_quality_json['solution_quality']
    except:
        match = re.search(r'"solution_quality"\s*:\s*"([^"]+)"', response_llm_quality)
        if match:
            response_quality = match.group(1)
        else:
            response_quality = "unknown"
    
    response_quality = response_quality.lower()
    if response_quality not in valid_quality:
        response_quality = 'unknown'
        
    return response_quality

def make_pairs(instruct_response_llm_check_path, instruct_response_syntex_check_path):
    instruct_response_llm_check_dataset = []
    with open(instruct_response_llm_check_path, 'r') as f:
        for line in f:
            instruct_response_llm_check_dataset.append(json.loads(line))
    print(f"instruct & response llm check dataset length = {len(instruct_response_llm_check_dataset)}")
    
    instruct_response_syntex_check_dataset = []
    with open(instruct_response_syntex_check_path, 'r') as g:
        for line in g:
           instruct_response_syntex_check_dataset.append(json.loads(line))
    print(f"instruct & response syntex check dataset length = {len(instruct_response_syntex_check_dataset)}")
    
    assert len(instruct_response_llm_check_dataset) == len(instruct_response_syntex_check_dataset)
    
    response_llm_check_not_good_count = 0
    response_syntex_check_not_right_count = 0
    filterd_dataset = []
    for llm_check_item, syntex_check_item in zip(instruct_response_llm_check_dataset, instruct_response_syntex_check_dataset):
        response_llm_quality = llm_check_item['response_quality']
        response_llm_check = extract_response_quality(response_llm_quality)
        if response_llm_check != 'good':
            response_llm_check_not_good_count += 1
        
        response_syntex_check = syntex_check_item['response_syntex']
        if response_syntex_check != 'right':
            response_syntex_check_not_right_count += 1
        
        if response_llm_check == 'good' and response_syntex_check == 'right':
            filterd_dataset.append(llm_check_item)
    
    print(f"response_llm_check_not_good_count = {response_llm_check_not_good_count}")
    print(f"response_syntex_check_not_right_count = {response_syntex_check_not_right_count}")
    print(f"final llm check good and syntex right datasets = {len(filterd_dataset)}")
    
    saved_path = instruct_response_llm_check_path.replace("_llm_quality.jsonl", "_quality_filtered.jsonl")
    with open(saved_path, 'w') as g:
        for item in filterd_dataset:
            g.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_stage", type=str, default='make-pairs', choices=["make-pairs"])
    parser.add_argument("--instruct_response_llm_check_path", type=str, required=True)
    parser.add_argument("--instruct_response_syntex_check_path", type=str, required=True)
    args = parser.parse_args()
    
    if args.process_stage == 'make-pairs':
        make_pairs(
            instruct_response_llm_check_path=args.instruct_response_llm_check_path,
            instruct_response_syntex_check_path=args.instruct_response_syntex_check_path
            )
    else:
        print("Check process stage.")
