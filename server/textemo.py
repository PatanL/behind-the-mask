"""What the *words* express, independently of the model's internal state: a GoEmotions classifier
(Demszky et al. 2020, SamLowe/roberta-base-go_emotions, MIT), grouped into the exhibit's emotions."""
from __future__ import annotations

GROUPS = {
    "joy": ["joy", "amusement", "excitement", "love", "optimism", "gratitude", "pride", "relief", "admiration"],
    "sadness": ["sadness", "grief", "disappointment", "remorse"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "curiosity": ["curiosity", "confusion", "realization"],
    "surprise": ["surprise"],
    "disgust": ["disgust", "embarrassment"],
    "calm": ["caring", "approval"],
    "neutral": ["neutral"],
}


class TextEmotion:
    def __init__(self, device: str = "cpu"):
        self.clf = None
        try:
            from transformers import pipeline
            self.clf = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)
        except Exception as e:  # noqa: BLE001
            print("[textemo] classifier unavailable:", e)

    def score(self, text: str) -> dict:
        if not self.clf or not text.strip():
            return {}
        raw = {s["label"]: s["score"] for s in self.clf(text[-512:])[0]}
        return {g: round(min(1.0, sum(raw.get(l, 0.0) for l in labels)), 3) for g, labels in GROUPS.items()}
