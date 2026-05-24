#!/bin/bash
# Pipeline: Synthetic Code Execution Data via DOMINO soft tokens
# Customize paths for your environment.

LLM_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
SOFT_PROMPT_DIR="./outputs/soft_prompt"
SOFT_TOKEN_COUNT=256
TEMP=0.6
TARGET_COUNT=40000
BATCH_SIZE=200
DEVICES='0,1,2,3'

## Step 1: Generate synthetic code execution problems
python -m domino.soft_prompt.generate \
    --pretrained_model_name_or_path "${LLM_MODEL}" \
    --tokenizer_name_or_path "${LLM_MODEL}" \
    --soft_prompt_dir "${SOFT_PROMPT_DIR}" \
    --inference_engine vllm \
    --soft_token_count "${SOFT_TOKEN_COUNT}" \
    --temp "${TEMP}" \
    --target_count "${TARGET_COUNT}" \
    --tensor_parallel_size 4 \
    --device "cuda:0"
echo "======Synthetic Generation Done! $(date) ========"

## Step 2: Quality assessment
python -m domino.pipeline.code_execution.quality_assessment \
    --process_stage quality_assessment \
    --synthetic_instruct_path "${SOFT_PROMPT_DIR}/vllm_generated_${TARGET_COUNT}_samples_temp${TEMP}.jsonl" \
    --GRADE_MODEL "${LLM_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --devices "${DEVICES}"
echo "======Quality Assessment Done! $(date) ========"

## Step 3: Generate responses
python -m domino.pipeline.code_execution.generate_response \
    --process_stage inference_response \
    --synthetic_instruct_path "${SOFT_PROMPT_DIR}/vllm_generated_${TARGET_COUNT}_samples_temp${TEMP}_instruct_quality.jsonl" \
    --inference_model "${LLM_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --devices "${DEVICES}"
echo "======Response Generation Done! $(date) ========"
