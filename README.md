# 🧠 SLM Classifier

> **Can a tiny 0.5B model match a 7B LLM through prompt engineering alone?**  
> This project finds out — across 10 classification tasks, three prompting strategies, and a rigorous F1/Accuracy comparison.

---

## What This Is

A zero-training NLP classification system that pits a **weak Small Language Model** against a **strong Large Language Model** using only prompt engineering. No fine-tuning. No labelled training data. Just cleverly constructed prompts.

| Model | Size | Where | Role |
|-------|------|--------|------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Local GPU | Weak SLM — subject under test |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | HF Inference API | Strong LLM — reference target |

The goal: make the SLM's output **as close as possible** to the LLM's through prompting alone.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        run.py                               │
│              Interactive dataset selector                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ pipes stdin
┌──────────────────────────▼──────────────────────────────────┐
│                        main.py                              │
│   collect_inputs() → classify_all() → print results         │
└──────┬─────────────────────────────────────┬────────────────┘
       │                                     │
┌──────▼──────┐   prompts.py        ┌────────▼────────┐
│  WeakSLM    │◄──zero_shot()       │   StrongLLM     │
│  (local)    │◄──few_shot()        │   (HF API)      │
│  batch GPU  │◄──optimized()       │   sequential    │
└──────┬──────┘                     └────────┬────────┘
       │                                     │
       └────────────┬────────────────────────┘
                    │
           ┌────────▼────────┐
           │  evaluate.py    │
           │  F1 + Accuracy  │
           └─────────────────┘
```

---

## Prompting Strategies

Three strategies are compared on every run, applied to both the SLM and LLM:

### 1 · Zero-shot
No examples. The model must classify from task description alone.
```
Task: Classify sentiment as positive or negative
Classify the following text. Output only the label, nothing else.
Text: The movie was incredible!
Label:
```

### 2 · Few-shot
Up to 3 labeled examples shown before the target text.
```
Task: Classify sentiment as positive or negative

Examples:
Text: I loved every minute → positive
Text: Total waste of time  → negative
Text: Absolutely brilliant → positive

Now classify:
Text: The movie was incredible!
Label:
```

### 3 · Optimized ⭐
Role prompting + explicit label enumeration + strict output constraint.  
This is the "closest to LLM" strategy — the key contribution of this project.
```
You are an expert text classifier. Task: Classify sentiment as positive or negative

Examples:
Text: I loved every minute → positive
Text: Total waste of time  → negative
Text: Absolutely brilliant → positive

Valid labels: positive, negative
Text: The movie was incredible!

Output ONLY the label, exactly as shown in the examples. Nothing else.
Label:
```

> **Design choice:** Full chain-of-thought was dropped. A 0.5B model fills its 64-token budget with reasoning and never reaches the label. Explicit label enumeration is far more effective at this scale.

---

## Output

### Single input mode
```
┌──────────────────────────────┬──────────────────────────────┐
│ Zero-shot SLM                │ Negative                     │
│ Few-shot SLM                 │ positive                     │
│ Optimized SLM                │ positive                     │
│ LLM reference                │ positive                     │
│ Accuracy / F1                │ 1.00 / 1.00                  │
└──────────────────────────────┴──────────────────────────────┘
```

### Batch mode
```
  Strategy                    vs LLM ref     vs Ground Truth
  ────────────────────────  ────────────  ────────────────
  Zero-shot SLM             acc=0.60 f1=0.58  acc=0.60 f1=0.58
  Few-shot SLM              acc=0.80 f1=0.82  acc=0.80 f1=0.82
  Optimized SLM             acc=1.00 f1=1.00  acc=1.00 f1=1.00
```

Metrics are computed with `sklearn` using weighted F1 and standard accuracy.  
**vs LLM ref** = how closely SLM matches the strong model.  
**vs Ground Truth** = absolute correctness (requires `label:text` format in input).

---

## Project Structure

```
SLM/
├── main.py          — interactive menu, orchestration, result tables
├── models.py        — WeakSLM (local batch GPU) + StrongLLM (HF API)
├── prompts.py       — zero_shot, few_shot, optimized, extract_label
├── evaluate.py      — score() → F1 + accuracy via sklearn
├── run.py           — dataset selector (pick a set, pipe to main.py)
├── requirements.txt
└── inputs/
    ├── set01_sentiment_zeroshot_single.txt
    ├── set02_sentiment_fewshot_batch.txt
    ├── set03_topic_fewshot_batch.txt
    ├── set04_spam_fewshot_batch.txt
    ├── set05_emotion_fewshot_batch.txt
    ├── set06_news_category_fewshot_batch.txt
    ├── set07_intent_fewshot_batch.txt
    ├── set08_language_zeroshot_batch.txt
    ├── set09_toxicity_zeroshot_batch.txt
    └── set10_product_review_fewshot_batch.txt
```

---

## Setup

**Requirements:** Python 3.10+, a CUDA GPU for local inference, a HuggingFace account.

```bash
# 1. Clone and enter
git clone <repo-url> && cd SLM

# 2. Create environment (pyenv recommended)
pyenv virtualenv 3.11.x SLM
pyenv activate SLM

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env
```

> You need a HF token with access to `Qwen/Qwen2.5-7B-Instruct`. Request access at [huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) if needed.

---

## Usage

### Option A — Pick from the prebuilt dataset

```bash
python run.py
```

```
  ┌─────┬──────────────────────────────────────────────────────────┐
  │  #  │ Set                                                      │
  ├─────┼──────────────────────────────────────────────────────────┤
  │   1 │ Sentiment                    ZEROSHOT | SINGLE           │
  │   2 │ Sentiment                    FEWSHOT | BATCH             │
  │   3 │ Topic                        FEWSHOT | BATCH             │
  │   4 │ Spam                         FEWSHOT | BATCH             │
  │   5 │ Emotion                      FEWSHOT | BATCH             │
  │   6 │ News Category                FEWSHOT | BATCH             │
  │   7 │ Intent                       FEWSHOT | BATCH             │
  │   8 │ Language                     ZEROSHOT | BATCH            │
  │   9 │ Toxicity                     ZEROSHOT | BATCH            │
  │  10 │ Product Review               FEWSHOT | BATCH             │
  └─────┴──────────────────────────────────────────────────────────┘

  Select set number: _
```

### Option B — Pipe a specific input file

```bash
python main.py < inputs/set03_topic_fewshot_batch.txt
```

### Option C — Fully interactive

```bash
python main.py
```

Follow the prompts to enter your own task, examples, and texts.

---

## Input File Format

Input files are plain text, piped to stdin. Each line maps to an interactive prompt.

**Zero-shot, single input:**
```
Classify sentiment as positive or negative   ← task description
1                                            ← 1=zero-shot
1                                            ← 1=single input
The movie was absolutely incredible.         ← text to classify
```

**Few-shot, batch with ground truth:**
```
Classify sentiment as positive or negative   ← task description
2                                            ← 2=few-shot
positive:I loved every minute of it.         ← example 1 (label:text)
negative:Total waste of time and money.      ← example 2
positive:Wonderful from start to finish.     ← example 3
2                                            ← 2=batch mode
positive:The food here was incredible.       ← input 1 with ground truth
negative:Service was terrible and rude.      ← input 2
                                             ← blank line = done
```

> Ground truth labels are optional. Omit the `label:` prefix to classify without scoring vs truth.

---

## Prebuilt Input Sets

| # | Task | Labels | Mode |
|---|------|--------|------|
| 01 | Sentiment analysis | positive, negative | Zero-shot · Single |
| 02 | Sentiment analysis | positive, negative | Few-shot · Batch |
| 03 | Topic classification | sports, politics, technology | Few-shot · Batch |
| 04 | Spam detection | spam, not_spam | Few-shot · Batch |
| 05 | Emotion detection | joy, anger, sadness, fear | Few-shot · Batch |
| 06 | News category | business, science, health, entertainment | Few-shot · Batch |
| 07 | User intent | question, complaint, compliment, request | Few-shot · Batch |
| 08 | Language detection | English, French, Spanish, German | Zero-shot · Batch |
| 09 | Toxicity detection | toxic, non_toxic | Zero-shot · Batch |
| 10 | Product review | positive, neutral, negative | Few-shot · Batch |

---

## Key Design Decisions

**Why no CoT for the SLM?**  
Chain-of-thought requires the model to reason before outputting the label. With `max_new_tokens=64`, a 0.5B model fills the budget with incomplete reasoning and never produces a label. Explicit label enumeration (`Valid labels: X, Y, Z`) achieves far better output alignment at this scale.

**Why batch GPU inference?**  
Calling the pipeline once per text triggers a transformer warning about sequential GPU usage and is inefficient. `classify_batch()` sends all texts for a strategy in a single forward pass, using the GPU as intended.

**Why weighted F1?**  
Label distributions in small batches are rarely balanced. Weighted F1 gives a fair aggregate score without penalising the model for class imbalance in the test set.

**Why the same `optimized` prompt for SLM and LLM?**  
The comparison is fair only if both models see identical input. The gap in their outputs — despite the same prompt — is the measured signal.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | Local Qwen2.5-0.5B pipeline |
| `huggingface_hub` | HF Inference API for Qwen2.5-7B |
| `torch` + `accelerate` | GPU inference, `device_map="auto"` |
| `scikit-learn` | F1 score + accuracy |
| `python-dotenv` | Load `HF_TOKEN` from `.env` |

---

<div align="center">
  <sub>Built with Qwen2.5 · HuggingFace Transformers · scikit-learn</sub>
</div>
