"""
Inference utilities and prediction wrappers.

Purpose:
    - Provide `predict_text(text)` high-level interface.

Responsibilities:
    - Tokenize single samples, run forward pass, apply softmax,
      and return label + confidence.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from bert_emotion_project.src.config import CONFIG
from bert_emotion_project.src.model import EmotionClassifier


def predict_text(text: str) -> dict:
    """
    Predict emotion for a single input text.

    Args:
        text (str): Raw input string.

    Returns:
        dict:
            {
                "label": predicted_class_name (string),
                "confidence": float_value
            }
    """

    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Load tokenizer
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model"]["pretrained_model_name"]
    )

    # -------------------------------
    # Load dataset to obtain class names (same ordering as training)
    # -------------------------------
    ds = load_dataset(CONFIG["dataset"]["hf_identifier"])
    full_dataset = ds["train"]

    text_col = "sentence" if "sentence" in full_dataset.column_names else "text"
    label_col = "emotion" if "emotion" in full_dataset.column_names else "label"

    class_names = sorted(list(set(full_dataset[label_col])))

    # -------------------------------
    # Initialize model
    # -------------------------------
    model = EmotionClassifier(
        num_labels=CONFIG["model"]["num_labels"],
        dropout_prob=CONFIG["model"]["dropout"],
        pretrained_model_name=CONFIG["model"]["pretrained_model_name"],
    )

    # -------------------------------
    # Load saved weights
    # -------------------------------
    checkpoint_path = Path(CONFIG["paths"]["models"]) / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # -------------------------------
    # Tokenize input
    # -------------------------------
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=CONFIG["tokenization"]["max_length"],
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # -------------------------------
    # Forward pass
    # -------------------------------
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = torch.max(probabilities).item()

    predicted_label = class_names[predicted_index]

    return {
        "label": predicted_label,
        "confidence": confidence
    }


if __name__ == "__main__":
    examples = [
        "I feel so lonely tonight.",
        "I just got promoted at work!",
        "That noise scared me.",
        "I'm embarrassed about what I said.",
        "I love spending time with my family."
    ]

    for text in examples:
        print(text)
        print(predict_text(text))
        print()
