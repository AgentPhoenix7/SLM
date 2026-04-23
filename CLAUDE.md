# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python SLM (Small Language Model) classifier. Performs classification via zero-shot and few-shot prompting (max 3 examples), optimizing SLM output to match stronger LLM quality.

**Outputs per run:**
1. Raw SLM output
2. Optimized SLM output
3. LLM reference output
4. Evaluation score (F1 or Accuracy)

## Environment

- Python managed via pyenv, environment named `SLM`
- Activate: `pyenv activate SLM`
- Run: `python main.py`

## Architecture Intent

When implemented, the system should have these logical components:
- **Prompt builder** — constructs zero-shot and few-shot prompts from input text + examples
- **SLM interface** — calls small model (e.g. local via `llama.cpp`, `ollama`, or HF `transformers`)
- **LLM reference interface** — calls stronger model (e.g. Claude API, OpenAI) for ground truth
- **Optimizer** — refines SLM prompt strategy to close gap with LLM reference output
- **Evaluator** — computes F1/accuracy between SLM output and LLM reference

## Dependencies

No `requirements.txt` yet. Add one when first dependencies are installed.
