# MAGPIE Baseline for DOMINO Comparison

This directory contains the configuration and scripts used to run **MAGPIE** (Xu et al., ICLR 2025) as a baseline for comparison with DOMINO.

MAGPIE is a **self-synthesis** method that uses an LLM to generate instruction-response pairs without requiring seed examples or manual annotation.

## Reference

- Paper: [MAGPIE: Aligning Large Language Models with Data from the Wild](https://arxiv.org/abs/2402.12017) (ICLR 2025)
- Official repo: [https://github.com/magpie-align/magpie](https://github.com/magpie-align/magpie)

## Setup

### Step 1: Clone the official MAGPIE repository

```bash
git clone https://github.com/magpie-align/magpie.git
```

### Step 2: Add custom code domain prompts

Copy the custom system prompt templates from `config/code_prompts.json` into MAGPIE's `configs/model_configs.json`. Add the `Qwen2.5-Coder-7B-Instruct` and `Qwen2.5-Coder-14B-Instruct` entries (with their `pre_query_template_code_*` fields) to the file.

### Step 3: Add `--control_tasks` support (required modification)

The DOMINO comparison requires a `--control_tasks` flag in `exp/gen_ins.py`. This flag selects domain-specific system prompt templates at generation time.

**Required modifications to `exp/gen_ins.py`:**

1. Add `--control_tasks` argument:
```python
parser.add_argument("--control_tasks", type=str, default=None,
                    choices=[None, "code_general", "code_gen_domain",
                             "code_gen_few_shot", "code_exe_few_shot", "code_exe"])
```

2. In the generation loop, when `--control_tasks` is specified, load the corresponding `pre_query_template_{control_task}` from the model config instead of the default `pre_query_template`.

3. Also add `--seed` and `--gpu_memory_utilization` arguments if not present.

### Step 4: Run experiments

```bash
# Code Generation (Few-Shot) - 80K samples
bash scripts/run_codegen_fewshot.sh

# Code Execution (Few-Shot) - 40K samples
bash scripts/run_codeexe_fewshot.sh

# Code Generation (Domain) - 80K samples
bash scripts/run_codegen_domain.sh
```

## Three Comparison Settings

| Script | Model | Control Task | Instruction Temp | Total Prompts |
|---|---|---|---|---|
| `run_codegen_fewshot.sh` | Qwen2.5-Coder-7B | `code_gen_few_shot` | 0.8 | 80,000 |
| `run_codeexe_fewshot.sh` | Qwen2.5-Coder-7B | `code_exe_few_shot` | 0.8 | 40,000 |
| `run_codegen_domain.sh` | Qwen2.5-Coder-14B | `code_gen_domain` | 1.0 | 80,000 |

## Files

| File | Description |
|---|---|
| `scripts/run_codegen_fewshot.sh` | Code gen few-shot experiment script |
| `scripts/run_codeexe_fewshot.sh` | Code exe few-shot experiment script |
| `scripts/run_codegen_domain.sh` | Code gen domain experiment script |
| `config/code_prompts.json` | Custom system prompt templates for code domains |
