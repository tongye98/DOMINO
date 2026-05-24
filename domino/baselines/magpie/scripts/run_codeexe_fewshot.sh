#!/bin/bash
# MAGPIE baseline: Code Execution (few-shot) using Qwen2.5-Coder-7B-Instruct
# Requires a cloned MAGPIE repo: https://github.com/magpie-align/magpie
# Usage: bash run_codeexe_fewshot.sh [model_path] [total_prompts]

MAGPIE_DIR="../magpie"
MODEL_PATH=${1:-"Qwen/Qwen2.5-Coder-7B-Instruct"}
TOTAL_PROMPTS=${2:-40000}
INS_TOP_P=1
INS_TEMP=0.8
RES_TOP_P=1
RES_TEMP=0
RES_REP=1
DEVICE="0,1,2,3"
TENSOR_PARALLEL=4
GPU_MEM_UTIL=0.95
N=200
BATCH_SIZE=200

timestamp=$(date +%Y%m%d%H%M)
timestamp=${timestamp//-/}
job_name="code_exe_few_shot_${MODEL_PATH##*/}_topp${INS_TOP_P}_temp${INS_TEMP}_${timestamp}"
OUTPUT_DIR="./outputs/magpie/code-exe"

# Setup logging
job_path="${OUTPUT_DIR}/logger/${job_name}"
mkdir -p "$job_path"
exec > >(tee -a "$job_path/${job_name}.log") 2>&1

echo "[MAGPIE] Model: $MODEL_PATH"
echo "[MAGPIE] Total Prompts: $TOTAL_PROMPTS"
echo "[MAGPIE] Instruction: temp=$INS_TEMP, top_p=$INS_TOP_P"
echo "[MAGPIE] Response: temp=$RES_TEMP, top_p=$RES_TOP_P"
echo "[MAGPIE] Control Task: code_exe_few_shot"

echo "[MAGPIE] Generating instructions..."
CUDA_VISIBLE_DEVICES=$DEVICE python "${MAGPIE_DIR}/exp/gen_ins.py" \
    --device "$DEVICE" \
    --model_path "$MODEL_PATH" \
    --total_prompts "$TOTAL_PROMPTS" \
    --top_p "$INS_TOP_P" \
    --temperature "$INS_TEMP" \
    --tensor_parallel "$TENSOR_PARALLEL" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --n "$N" \
    --job_name "$job_name" \
    --timestamp "$timestamp" \
    --control_tasks "code_exe_few_shot" \
    --output_folder "$OUTPUT_DIR" \
    --seed 42

echo "[MAGPIE] Generating responses..."
CUDA_VISIBLE_DEVICES=$DEVICE python "${MAGPIE_DIR}/exp/gen_res.py" \
    --device "$DEVICE" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --top_p "$RES_TOP_P" \
    --temperature "$RES_TEMP" \
    --repetition_penalty "$RES_REP" \
    --tensor_parallel "$TENSOR_PARALLEL" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --input_file "$OUTPUT_DIR/$job_name/Magpie_${MODEL_PATH##*/}_${TOTAL_PROMPTS}_${timestamp}_ins.json" \
    --tokenizer_template false \
    --offline

echo "[MAGPIE] Done!"
