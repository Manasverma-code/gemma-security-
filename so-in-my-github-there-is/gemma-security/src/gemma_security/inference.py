"""Run predictions with a trained Gemma Security model."""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ID_TO_LABEL, OUTPUT_MODEL_DIR
from .data import build_security_prompt


def predict_security_label(text: str, model_dir=OUTPUT_MODEL_DIR) -> dict:
    """Predict the most likely security label for one input string."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    prompt = build_security_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    label_id = int(torch.argmax(probabilities).item())

    return {
        "label": ID_TO_LABEL[label_id],
        "confidence": float(probabilities[label_id].item()),
    }


def main():
    """Allow quick command-line testing."""

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.gemma_security.inference '<security text>'")

    result = predict_security_label(sys.argv[1])
    print(result)


if __name__ == "__main__":
    main()
