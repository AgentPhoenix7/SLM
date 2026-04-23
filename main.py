from __future__ import annotations
import argparse
import os

from models import WeakSLM, StrongLLM
from prompts import zero_shot, optimized, extract_label
from evaluate import score


def parse_examples(raw: list[str]) -> list[tuple[str, str]]:
    """Parse 'label:text' strings into (label, text) tuples."""
    pairs: list[tuple[str, str]] = []
    for item in raw or []:
        if ":" not in item:
            raise ValueError(f"Example must be 'label:text', got: {item!r}")
        label, text = item.split(":", 1)
        pairs.append((label.strip(), text.strip()))
    return pairs[:3]


def _row(label: str, value: str, width: int = 24) -> str:
    return f"│ {label:<{width}} │ {value:<20} │"


def print_table(
    raw_slm: str,
    opt_slm: str,
    llm_ref: str,
    metrics: dict[str, float],
) -> None:
    border = "─" * 26
    sep    = "─" * 22
    print(f"┌{border}┬{sep}┐")
    print(_row("Raw SLM output",       raw_slm))
    print(_row("Optimized SLM output", opt_slm))
    print(_row("LLM reference output", llm_ref))
    score_str = f"{metrics['accuracy']:.2f} / {metrics['f1']:.2f}"
    print(_row("Accuracy / F1",        score_str))
    print(f"└{border}┴{sep}┘")


def main() -> None:
    parser = argparse.ArgumentParser(description="SLM classifier with zero/few-shot prompting")
    parser.add_argument("--task",     required=True,       help="Classification task description")
    parser.add_argument("--text",     required=True,       help="Input text to classify")
    parser.add_argument("--examples", nargs="*", default=[], metavar="LABEL:TEXT",
                        help="Up to 3 few-shot examples as 'label:text'")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if not args.hf_token:
        parser.error("HuggingFace token required via --hf-token or HF_TOKEN env var")

    examples = parse_examples(args.examples)

    print("Loading weak SLM (Qwen2.5-0.5B, local)...")
    slm = WeakSLM()

    print("Connecting to strong LLM (Qwen2.5-7B, HF API)...")
    llm = StrongLLM(token=args.hf_token)

    # --- zero-shot on weak model ---
    raw_prompt = zero_shot(args.task, args.text)
    raw_output = extract_label(slm.classify(raw_prompt))

    # --- optimized few-shot on weak model ---
    if examples:
        opt_prompt = optimized(args.task, args.text, examples)
    else:
        opt_prompt = zero_shot(args.task, args.text)
    opt_output = extract_label(slm.classify(opt_prompt))

    # --- optimized few-shot on strong model (reference) ---
    ref_prompt = optimized(args.task, args.text, examples) if examples else zero_shot(args.task, args.text)
    ref_output = extract_label(llm.classify(ref_prompt))

    metrics = score([opt_output], [ref_output])

    print()
    print_table(raw_output, opt_output, ref_output, metrics)


if __name__ == "__main__":
    main()
