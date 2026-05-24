import os 
import json 
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm 


def process_response_batch(batch, llm, params, tokenizer):
    synthetic_instructions = [item['question_content'] for item in batch]
    
    FORMATTING_WITHOUT_STARTER_CODE = "You will solve the code problem using Python. First generate the problem-solving idea, and then generate the code. Enclose your code within delimiters."

    prompts = []
    for synthetic_instruction in synthetic_instructions:
        format = f"### Instruction:\n{synthetic_instruction}\n{FORMATTING_WITHOUT_STARTER_CODE}\n\n### Response:\n"
        chat = [{"role": "user", "content": format}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)
    
    outputs = llm.generate(prompts, params)
    
    for i, item in enumerate(batch):
        item["original_response"] = outputs[i].outputs[0].text.strip()

    return batch

def get_vllm_configuration(inference_model, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    tokenizer = AutoTokenizer.from_pretrained(inference_model, trust_remote_code=True)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        
    print("Start Local vllm engine...")
    llm = LLM(model=inference_model, 
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=len(devices.split(',')),
        gpu_memory_utilization=0.95)

    params = SamplingParams(
        max_tokens=4096,
        temperature=0.2,
        top_p=1,
        repetition_penalty=1,
        stop_token_ids=stop_token_ids,
        )
    
    return llm, params, tokenizer
