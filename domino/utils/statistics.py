import json
from tqdm import tqdm


def reference_data_statistics(data_path):
    datasets = []
    with open(data_path, 'r') as f:
        for line in f:
            datasets.append(json.loads(line))

    print(f"length of datasets = {len(datasets)}")

    content_tokens = []
    for item in datasets:
        question_content = item["question_content"]
        content_token_count = len(question_content.split())
        content_tokens.append(content_token_count)

    print(f"max token = {max(content_tokens)}")
    print(f"min token = {min(content_tokens)}")
    print(f"avg token = {sum(content_tokens) / len(content_tokens)}")


def synthetic_data_count_statistics(synthetic_dir, synthetic_instruction_file):
    synthetic_instruction_path = f"{synthetic_dir}/{synthetic_instruction_file}"
    synthetic_instruction_filtered_path = synthetic_instruction_path.replace('.jsonl', '_instruct_quality_response.jsonl')
    instruction_response_filtered_path = synthetic_instruction_path.replace('.jsonl', '_instruct_quality_response_quality_filtered.jsonl')

    with open(synthetic_instruction_path, 'r') as f:
        synthetic_instruction_dataset = [json.loads(line) for line in f]
    with open(synthetic_instruction_filtered_path, 'r') as f:
        synthetic_instruction_filtered_dataset = [json.loads(line) for line in f]
    with open(instruction_response_filtered_path, 'r') as f:
        instruction_response_filtered_dataset = [json.loads(line) for line in f]

    print(f"Synthetic instructions: {len(synthetic_instruction_dataset)}")
    print(f"After quality filter: {len(synthetic_instruction_filtered_dataset)}")
    print(f"After response filter: {len(instruction_response_filtered_dataset)}")
