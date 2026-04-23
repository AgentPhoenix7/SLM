from __future__ import annotations
from transformers import pipeline
from huggingface_hub import InferenceClient


class WeakSLM:
    """Qwen2.5-0.5B-Instruct running locally via transformers."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self) -> None:
        self.pipe = pipeline(
            "text-generation",
            model=self.MODEL,
            device_map="auto",
        )
        # model's generation_config.json sets max_length=20 which conflicts
        # with max_new_tokens; clear it so only max_new_tokens applies
        self.pipe.model.generation_config.max_length = None

    def classify(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self.pipe(messages, max_new_tokens=64)
        return result[0]["generated_text"][-1]["content"].strip()


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
