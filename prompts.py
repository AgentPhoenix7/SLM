from __future__ import annotations
import re


LABEL_PREFIX = re.compile(r"^\s*(?:final\s+)?label\s*:\s*", re.IGNORECASE)


def zero_shot(task: str, text: str) -> str:
    return (
        f"Task: {task}\n"
        "Classify the following text. Output only the label, nothing else.\n"
        f"Text: {text}\n"
        "Label:"
    )


def few_shot(task: str, text: str, examples: list[tuple[str, str]]) -> str:
    example_block = "\n".join(
        f"Text: {t} → {label}" for label, t in examples[:3]
    )
    return (
        f"Task: {task}\n\n"
        f"Examples:\n{example_block}\n\n"
        "Now classify:\n"
        f"Text: {text}\n"
        "Label:"
    )


def optimized(task: str, text: str, examples: list[tuple[str, str]]) -> str:
    """Few-shot + role prompting + chain-of-thought + output constraint."""
    example_block = "\n".join(
        f"Text: {t} → {label}" for label, t in examples[:3]
    )
    return (
        f"You are an expert text classifier. Your task: {task}\n\n"
        "Study the examples below, then think step by step before giving your answer. "
        "Output ONLY the final label on the very last line — no explanation after it.\n\n"
        f"Examples:\n{example_block}\n\n"
        f"Text to classify: {text}\n\n"
        "Step-by-step reasoning:"
    )


def extract_label(raw: str) -> str:
    """Pull a clean final label from model output."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    label = lines[-1] if lines else raw.strip()
    label = LABEL_PREFIX.sub("", label).strip()

    if "->" in label:
        label = label.rsplit("->", 1)[-1].strip()
    if "→" in label:
        label = label.rsplit("→", 1)[-1].strip()

    return label.strip(" .\"'")
