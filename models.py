from __future__ import annotations
import logging
from transformers import pipeline
from huggingface_hub import InferenceClient

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


class WeakSLM:
    """Qwen2.5-0.5B-Instruct running locally via transformers."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self) -> None:
        self.pipe = pipeline(
            "text-generation",
            model=self.MODEL,
            device_map="auto",
        )

    def classify_batch(self, prompts: list[str]) -> list[str]:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        results = self.pipe(conversations, max_new_tokens=64, batch_size=len(conversations))
        return [r[0]["generated_text"][-1]["content"].strip() for r in results]


class StrongLLM:
    """Qwen2.5-7B-Instruct via HuggingFace Inference API."""

    MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, token: str) -> None:
        self.client = InferenceClient(model=self.MODEL, token=token)

    def classify(self, prompt: str) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
        )
        return response.choices[0].message.content.strip()
