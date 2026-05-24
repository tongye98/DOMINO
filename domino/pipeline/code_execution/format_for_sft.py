import os 
import json


def format_code_execution_for_sft(path):
    datasets = []
    with open(path, 'r') as f:
        for line in f:
            datasets.append(json.loads(line))

    print(f"length of datasets = {len(datasets)}")

    formated_samples = []
    for item in datasets:
        system = "You are given a Python function and an assertion containing an input to the function.\
Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input,\
even if the function is incorrect or incomplete. Do NOT output any extra information. \
Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags."
        
        code = f"[PYTHON]\n{item['code']}\n\nassert {item['input']} == ??\n[/PYTHON]"
        output = f"[ANSWER]\nassert {item['input']} == {item['output']}\n[/ANSWER]"

        point = {
            "instruction": code,
            "input": "",
            "output": output,
            "system": system
        }

        formated_samples.append(point)
    
    output_path = path.replace(".jsonl", "_sft_format.json")
    with open(output_path, 'w') as g:
        json.dump(formated_samples, g, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        format_code_execution_for_sft(sys.argv[1])
