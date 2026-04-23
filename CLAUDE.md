# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python SLM classifier. Uses two Qwen2.5 models for classification via zero-shot and few-shot prompting (max 3 examples), comparing weak SLM output against strong LLM reference.

**Outputs per run:**
1. Raw SLM output (zero-shot, weak model)
2. Optimized SLM output (few-shot + CoT, weak model)
3. LLM reference output (few-shot + CoT, strong model)
4. Accuracy / F1 score (optimized SLM vs reference)

## Environment

- Python managed via pyenv, environment named `SLM`
- Activate: `pyenv activate SLM`
- Install deps: `pip install -r requirements.txt`
- Run: `python main.py --task "..." --text "..." --examples "label:text" [--hf-token TOKEN]`
- HF token can also be set via `HF_TOKEN` env var

## Architecture

| File | Role |
|------|------|
| `main.py` | CLI entry, orchestration, output table |
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
