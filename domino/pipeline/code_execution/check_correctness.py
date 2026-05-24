import json
from .utils import check_correctness, BASE_IMPORTS
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor, as_completed

def check_synthetic_correctness():
    """Run correctness checks on synthetic execution data."""
    datasets = []
    with open(synthetic_path, 'r') as f:
        for line in f:
            datasets.append(json.loads(line))

    print(f"length of datasets = {len(datasets)}")

    execution_results = []
    for item in tqdm(datasets):
        generated_text = item["generated_text"]
        python_code = generated_text.split("[/PYTHON]")[0]
        python_code = python_code.split("assert")[0].split("[PYTHON]")[1]

        assert_answer = generated_text.split("[ANSWER]")[1].split("[/ANSWER]")[0]

        code_to_execute = f"{BASE_IMPORTS}\n{python_code}\n{assert_answer}"
        execution_results.append(check_correctness(code_to_execute, 3))
