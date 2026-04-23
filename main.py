from __future__ import annotations
import os
from dotenv import load_dotenv

from models import WeakSLM, StrongLLM
from prompts import zero_shot, few_shot, optimized, extract_label
from evaluate import score


# ── display helpers ────────────────────────────────────────────────────────────

DIVIDER = "─" * 54

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
    return f"│ {label:<{w}} │ {value:<24} │"

def print_single_table(
    zero_out: str,
    fs_out: str,
    opt_out: str,
    ref_out: str,
    metrics: dict[str, float],
) -> None:
    border, sep = "─" * 26, "─" * 26
    print(f"\n┌{border}┬{sep}┐")
    print(_row("Zero-shot SLM",   zero_out))
    print(_row("Few-shot SLM",    fs_out))
    print(_row("Optimized SLM",   opt_out))
    print(_row("LLM reference",   ref_out))
    print(_row("Accuracy / F1",   f"{metrics['accuracy']:.2f} / {metrics['f1']:.2f}"))
    print(f"└{border}┴{sep}┘")

def print_batch_table(results: list[dict[str, str]], gt_labels: list[str | None]) -> None:
    """Print per-strategy aggregate metrics across all batch inputs."""
    strategies = ["zero", "few", "opt"]
    labels_map = {"zero": "Zero-shot SLM", "few": "Few-shot SLM", "opt": "Optimized SLM"}
    ref_labels = [r["ref"] for r in results]

    print(f"\n  {'Strategy':<24}  {'vs LLM ref':>12}  {'vs Ground Truth':>16}")
    print(f"  {'─'*24}  {'─'*12}  {'─'*16}")
    for key in strategies:
        preds   = [r[key] for r in results]
        vs_ref  = score(preds, ref_labels)
        gt_pairs = [(r[key], gt) for r, gt in zip(results, gt_labels) if gt is not None]
        if gt_pairs:
            gt_preds, gt_truths = zip(*gt_pairs)
            vs_gt  = score(list(gt_preds), list(gt_truths))
            gt_str = f"acc={vs_gt['accuracy']:.2f} f1={vs_gt['f1']:.2f}"
        else:
            gt_str = "n/a"
        ref_str = f"acc={vs_ref['accuracy']:.2f} f1={vs_ref['f1']:.2f}"
        print(f"  {labels_map[key]:<24}  {ref_str:>12}  {gt_str:>16}")


# ── interactive input ──────────────────────────────────────────────────────────

def collect_examples() -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    print("\n  Enter up to 3 examples (format: label:text). Leave blank to stop.")
    for i in range(1, 4):
        raw = input(f"  Example {i}: ").strip()
        if not raw:
            break
        if ":" not in raw:
            print("  Skipped — must be 'label:text'")
            continue
        label, ex_text = raw.split(":", 1)
        examples.append((label.strip(), ex_text.strip()))
    return examples

def collect_batch_inputs() -> list[tuple[str, str | None]]:
    """Collect multiple texts. Format: 'label:text' (with ground truth) or plain text."""
    inputs: list[tuple[str, str | None]] = []
    print("\n  Enter texts to classify. Format 'label:text' to include ground truth.")
    print("  Leave blank to finish (min 2 inputs for meaningful metrics).")
    i = 1
    while True:
        raw = input(f"  Input {i}: ").strip()
        if not raw:
            if len(inputs) < 2:
                print("  Need at least 2 inputs. Keep going.")
                continue
            break
        if ":" in raw:
            label, text = raw.split(":", 1)
            inputs.append((text.strip(), label.strip()))
        else:
            inputs.append((raw, None))
        i += 1
    return inputs

def collect_inputs() -> tuple[str, list[tuple[str, str]], str, list[tuple[str, str | None]]]:
    header("SLM Classifier — Setup")

    task = ""
    while not task:
        task = prompt_line("Classification task (e.g. 'Classify sentiment as positive or negative')")

    prompt_mode = choose("Prompting mode", ["Zero-shot (no examples)", "Few-shot (add up to 3 examples)"])
    examples = collect_examples() if "Few-shot" in prompt_mode else []

    eval_mode = choose("Evaluation mode", [
        "Single input  — classify one text, compare SLM vs LLM",
        "Batch inputs  — classify multiple texts, compute F1/Accuracy",
    ])

    if "Single" in eval_mode:
        text = ""
        while not text:
            text = prompt_line("Text to classify")
        batch: list[tuple[str, str | None]] = [(text, None)]
    else:
        batch = collect_batch_inputs()

    return task, examples, eval_mode, batch


# ── run one input through all strategies ──────────────────────────────────────

def classify_one(
    slm: WeakSLM,
    llm: StrongLLM,
    task: str,
    text: str,
    examples: list[tuple[str, str]],
) -> dict[str, str]:
    zero_out = extract_label(slm.classify(zero_shot(task, text)))
    fs_out   = extract_label(slm.classify(few_shot(task, text, examples) if examples else zero_shot(task, text)))
    opt_out  = extract_label(slm.classify(optimized(task, text, examples) if examples else zero_shot(task, text)))
    ref_out  = extract_label(llm.classify(optimized(task, text, examples) if examples else zero_shot(task, text)))
    return {"zero": zero_out, "few": fs_out, "opt": opt_out, "ref": ref_out}


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set. Add it to .env or export HF_TOKEN=...")
        return

    task, examples, eval_mode, batch = collect_inputs()

    header("Loading Models")
    print("  [1/2] Loading weak SLM (Qwen2.5-0.5B, local)...")
    slm = WeakSLM()
    print("        Done.")
    print("  [2/2] Connecting to strong LLM (Qwen2.5-7B, HF API)...")
    llm = StrongLLM(token=hf_token)
    print("        Done.")

    header("Running Classification")
    results: list[dict[str, str]] = []
    gt_labels: list[str | None] = []
    total = len(batch)
    for idx, (text, gt) in enumerate(batch, 1):
        print(f"  [{idx}/{total}] {text[:60]!r}")
        out = classify_one(slm, llm, task, text, examples)
        results.append(out)
        gt_labels.append(gt)
        print(f"          zero={out['zero']!r}  few={out['few']!r}  opt={out['opt']!r}  ref={out['ref']!r}")

    header("Results")
    if "Single" in eval_mode:
        r = results[0]
        metrics = score([r["opt"]], [r["ref"]])
        print_single_table(r["zero"], r["few"], r["opt"], r["ref"], metrics)
        print("\n  Note: metrics on single sample are binary (1.0 or 0.0).")
        print("  Use Batch mode for meaningful F1/Accuracy across multiple inputs.")
    else:
        print_batch_table(results, gt_labels)
    print()


if __name__ == "__main__":
    main()
