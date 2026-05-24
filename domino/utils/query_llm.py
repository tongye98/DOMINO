import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def query_llm(prompt, model_path, temperature=0.2, max_tokens=4096, tensor_parallel_size=1):
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.95,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
    )

    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()
