from __future__ import annotations
import re


LABEL_PREFIX = re.compile(
    r"^\s*(?:final\s+)?(?:label|topic|category|answer|class|sentiment)\s*:\s*",
    re.IGNORECASE,
)


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
    """Few-shot + role prompting + explicit label constraint. No CoT — 0.5B lacks capacity."""
    example_block = "\n".join(
        f"Text: {t} → {label}" for label, t in examples[:3]
    )
    valid = list(dict.fromkeys(label for label, _ in examples[:3]))
    label_constraint = f"Valid labels: {', '.join(valid)}\n" if valid else ""
    return (
        f"You are an expert text classifier. Task: {task}\n\n"
        f"Examples:\n{example_block}\n\n"
        f"{label_constraint}"
        f"Text: {text}\n\n"
        "Output ONLY the label, exactly as shown in the examples. Nothing else.\n"
        "Label:"
    )


def extract_label(raw: str) -> str:
    """Pull a clean final label from model output."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return raw.strip()

    # prefer last line that has an explicit label prefix
    label = lines[-1]
    for line in reversed(lines):
        stripped = LABEL_PREFIX.sub("", line)
        if stripped != line:
            label = stripped
            break
    else:
        label = LABEL_PREFIX.sub("", label)

    if "->" in label:
        label = label.rsplit("->", 1)[-1].strip()
    if "→" in label:
        label = label.rsplit("→", 1)[-1].strip()

    return label.strip(" .\"'")
