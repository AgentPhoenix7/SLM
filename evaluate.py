from __future__ import annotations
from sklearn.metrics import f1_score, accuracy_score
from prompts import extract_label


def normalize(label: str) -> str:
    return extract_label(label).lower().strip()


def score(predictions: list[str], references: list[str]) -> dict[str, float]:
    preds = [normalize(p) for p in predictions]
    refs = [normalize(r) for r in references]
    return {
        "accuracy": accuracy_score(refs, preds),
        "f1": f1_score(refs, preds, average="weighted", zero_division=0),
    }
