# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

Python SLM classifier. Uses two Qwen2.5 models for classification via zero-shot and few-shot prompting (max 3 examples), comparing weak SLM output against strong LLM reference.

**Outputs per run:**
1. Zero-shot SLM output (weak model, no examples)
2. Few-shot SLM output (weak model, plain few-shot)
3. Optimized SLM output (weak model, few-shot + CoT)
4. LLM reference output (strong model, few-shot + CoT)
5. F1 / Accuracy — optimized SLM vs LLM ref, and optionally vs ground truth

## Environment

- Python managed via pyenv, environment named `SLM`
- Activate: `pyenv activate SLM`
- Install deps: `pip install -r requirements.txt`
- Run: `python main.py` (fully interactive — no CLI flags)
- Set `HF_TOKEN` in `.env` or export before running

## Architecture

| File | Role |
|------|------|
| `main.py` | Interactive menu, orchestration, output tables |
| `models.py` | `WeakSLM` (local transformers) + `StrongLLM` (HF Inference API) |
| `prompts.py` | `zero_shot`, `few_shot`, `optimized` builders + `extract_label` |
| `evaluate.py` | `score(predictions, references)` → F1 + accuracy via sklearn |

## Models

- **Weak SLM**: `Qwen/Qwen2.5-0.5B-Instruct` — loaded locally via `transformers` pipeline
- **Strong LLM**: `Qwen/Qwen2.5-7B-Instruct` — called via `huggingface_hub.InferenceClient`

## Prompt Strategies

Three strategies in `prompts.py`:
- `zero_shot` — baseline, no examples
- `few_shot` — up to 3 labeled examples
- `optimized` — few-shot + role prompting + chain-of-thought + output constraint (last line = label)

`extract_label()` strips CoT reasoning by taking the last non-empty line of model output.
