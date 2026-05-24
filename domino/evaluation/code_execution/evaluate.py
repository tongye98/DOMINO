import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from .metrics import code_execution_metrics, pass_at_k
from .utils import BASE_IMPORTS, check_correctness


def make_cot_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def performOperation(s):
    s = s + s
    return "b" + s + "a"
assert performOperation(s = "hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function performOperation is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert performOperation(s = "hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[THOUGHT]
"""


def run_batch(prompts, model_path, tensor_parallel_size=4):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=8000,
        trust_remote_code=True,
        gpu_memory_utilization=0.98,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        n=1,
        max_tokens=2048,
        temperature=0,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
    )

    outputs = [None for _ in prompts]
    remaining_prompts = []
    remaining_indices = []
    for prompt_index, prompt in enumerate(prompts):
        remaining_prompts.append(prompt)
        remaining_indices.append(prompt_index)

    if remaining_prompts:
        vllm_outputs = llm.generate(remaining_prompts, sampling_params)
        for index, vllm_output in zip(remaining_indices, vllm_outputs):
            outputs[index] = [o.text for o in vllm_output.outputs]

    return outputs


def evaluate_code_execution(
    test_path,
    model_path,
    tensor_parallel_size=4,
    output_path=None,
):
    test_samples = []
    with open(test_path, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))

    print(f"test samples length = {len(test_samples)}")

    prompts = [make_cot_output_prompt((item['code'], item['input'])) for item in test_samples]
    outputs = run_batch(prompts, model_path, tensor_parallel_size)

    generations = []
    for model_output in outputs:
        model_output = model_output[0]
        if "==" in model_output:
            model_output = model_output.split("==")[1].strip()
        if "[/ANSWER]" in model_output:
            model_output = model_output.split("[/ANSWER]")[0].strip()
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
        else:
            model_output = model_output.split("\n")[0].strip()
        generations.append([model_output])

    result = code_execution_metrics(test_samples, generations)
    metrics = result[0]
    results = result[1]

    print(f"pass@1 = {metrics['pass@1']}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    evaluate_code_execution(
        test_path=args.test_path,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        output_path=args.output_path,
    )
