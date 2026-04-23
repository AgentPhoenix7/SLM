from __future__ import annotations
import os
from dotenv import load_dotenv

from models import WeakSLM, StrongLLM
from prompts import zero_shot, optimized, extract_label
from evaluate import score


# ── display helpers ────────────────────────────────────────────────────────────

DIVIDER = "─" * 50

def header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def prompt_line(label: str) -> str:
    return input(f"  {label}: ").strip()

def choose(label: str, options: list[str]) -> str:
    print(f"\n  {label}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = input("  Choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice. Try again.")

def _row(label: str, value: str, w: int = 24) -> str:
    return f"│ {label:<{w}} │ {value:<20} │"

def print_table(raw_slm: str, opt_slm: str, llm_ref: str, metrics: dict[str, float]) -> None:
    border = "─" * 26
    sep    = "─" * 22
    print(f"\n┌{border}┬{sep}┐")
    print(_row("Raw SLM output",       raw_slm))
    print(_row("Optimized SLM output", opt_slm))
    print(_row("LLM reference output", llm_ref))
    print(_row("Accuracy / F1",        f"{metrics['accuracy']:.2f} / {metrics['f1']:.2f}"))
    print(f"└{border}┴{sep}┘")


# ── interactive input ──────────────────────────────────────────────────────────

def collect_inputs() -> tuple[str, str, list[tuple[str, str]]]:
    header("SLM Classifier — Setup")

    task = ""
    while not task:
        task = prompt_line("Classification task (e.g. 'Classify sentiment as positive or negative')")

    text = ""
    while not text:
        text = prompt_line("Text to classify")

    examples: list[tuple[str, str]] = []
    mode = choose("Prompting mode", ["Zero-shot (no examples)", "Few-shot (add up to 3 examples)"])

    if "Few-shot" in mode:
        print("\n  Enter examples (format: label : text). Leave blank to stop.")
        for i in range(1, 4):
            raw = input(f"  Example {i}: ").strip()
            if not raw:
                break
            if ":" not in raw:
                print("  Skipped — must be 'label:text'")
                continue
            label, ex_text = raw.split(":", 1)
            examples.append((label.strip(), ex_text.strip()))

    return task, text, examples


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set. Add it to .env or export HF_TOKEN=...")
        return

    task, text, examples = collect_inputs()

    header("Loading Models")
    print("  [1/2] Loading weak SLM (Qwen2.5-0.5B, local)...")
    slm = WeakSLM()
    print("        Done.")
    print("  [2/2] Connecting to strong LLM (Qwen2.5-7B, HF API)...")
    llm = StrongLLM(token=hf_token)
    print("        Done.")

    header("Running Classification")

    print("  [1/3] Zero-shot on weak SLM...")
    raw_output = extract_label(slm.classify(zero_shot(task, text)))
    print(f"        -> {raw_output!r}")

    print("  [2/3] Optimized few-shot on weak SLM...")
    opt_prompt = optimized(task, text, examples) if examples else zero_shot(task, text)
    opt_output = extract_label(slm.classify(opt_prompt))
    print(f"        -> {opt_output!r}")

    print("  [3/3] Optimized few-shot on strong LLM (reference)...")
    ref_prompt = optimized(task, text, examples) if examples else zero_shot(task, text)
    ref_output = extract_label(llm.classify(ref_prompt))
    print(f"        -> {ref_output!r}")

    metrics = score([opt_output], [ref_output])

    header("Results")
    print_table(raw_output, opt_output, ref_output, metrics)
    print()


if __name__ == "__main__":
    main()
