import re 
import json 


def extract_function_name(sample):
    pattern = r'def\s+(\w+)\s*\('
    result = re.findall(pattern, sample)
    return result


def filter_novel_functions(train_path, synthetic_path):
    """Filter synthetic samples whose function names don't appear in training set."""
    train_samples = []
    with open(train_path, 'r') as f:
        for line in f:
            train_samples.append(json.loads(line))

    print(f"length of train dataset = {len(train_samples)}")

    train_function_names = set()
    for sample in train_samples:
        function_name = sample.get("function_name")
        if function_name:
            train_function_names.add(function_name)

    print(f"trained function names = {train_function_names}")

    synthetic_samples = []
    with open(synthetic_path, 'r') as f:
        for line in f:
            synthetic_samples.append(json.loads(line))
    print(f"length of synthetic samples = {len(synthetic_samples)}")
    
    count = 0
    novel_samples = []
    for sample in synthetic_samples:
        flag = False
        extracted_synthetic_function_name = extract_function_name(sample.get("generated_text", ""))
        for name in extracted_synthetic_function_name:
            if name in train_function_names:
                flag = True
                break
        
        if not flag:
            count += 1
            novel_samples.append(sample)
    
    print(f"new functions = {count}")
    return novel_samples

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        filter_novel_functions(sys.argv[1], sys.argv[2])
