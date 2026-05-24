from .codegen_metrics import codegen_metrics
from .codeproblem import load_code_generation_dataset
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer


class PromptConstants:
    SYSTEM_MESSAGE_QWENCODER = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    SYSTEM_MESSAGE_OPENCODER = "<|im_start|>system\nYou are OpenCoder, created by OpenCoder Team.<|im_end|>\n<|im_start|>user"
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def get_qwencoder_question_template_answer(question):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n<|im_end|>\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def extract_code(model_output: str):
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def format_prompt_qwencoder(question):
    prompt = f"{PromptConstants.SYSTEM_MESSAGE_QWENCODER}\n\n"
    prompt += f"{get_qwencoder_question_template_answer(question)}"
    return prompt


def run_batch(prompts, model_path, tensor_parallel_size=4, n_samples=10, temperature=0.2):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=8000,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=4096,
        temperature=temperature,
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


def get_metrics(benchmark, combined_results):
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=12,
        timeout=6,
        debug=False,
    )

    print(f"pass@1 = {metrics[0]['pass@1']}")
    print(f"pass@5 = {metrics[0]['pass@5']}")
    print(f"pass@10 = {metrics[0]['pass@10']}")

    return metrics


def evaluate_code_generation(
    data_path,
    model_path,
    tensor_parallel_size=4,
    n_samples=10,
    temperature=0.2,
    output_path=None,
):
    benchmark = load_code_generation_dataset(data_path)
    benchmark = sorted(benchmark, key=lambda x: x.question_id)

    prompts = [format_prompt_qwencoder(problem) for problem in benchmark]
    results = run_batch(prompts, model_path, tensor_parallel_size, n_samples, temperature)

    combined_results = [(outputs_list, [extract_code(output) for output in outputs_list]) for outputs_list in results]

    save_results = [instance.insert_output(outputs_list, extracted_list) for instance, (outputs_list, extracted_list) in zip(benchmark, combined_results)]

    combined_results = [(save_result_instance["output_list"], save_result_instance["code_list"]) for save_result_instance in save_results]

    metrics = get_metrics(benchmark, combined_results)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    evaluate_code_generation(
        data_path=args.data_path,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        n_samples=args.n_samples,
        temperature=args.temperature,
        output_path=args.output_path,
    )
