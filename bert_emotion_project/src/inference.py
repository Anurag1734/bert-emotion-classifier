import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from bert_emotion_project.src.config import CONFIG
from bert_emotion_project.src.model import EmotionClassifier

_INFER_STATE = None


def _load_inference_state():
    global _INFER_STATE
    if _INFER_STATE is not None:
        return _INFER_STATE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["pretrained_model_name"])

    ds = load_dataset(CONFIG["dataset"]["hf_identifier"])
    full_dataset = ds["train"]
    label_col = "emotion" if "emotion" in full_dataset.column_names else "label"
    class_names = list(full_dataset.features[label_col].names)

    model = EmotionClassifier(
        num_labels=len(class_names),
        dropout_prob=CONFIG["model"]["dropout"],
        pretrained_model_name=CONFIG["model"]["pretrained_model_name"],
    )

    checkpoint_path = Path(CONFIG["paths"]["models"]) / "best_model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _INFER_STATE = {
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "class_names": class_names,
    }
    return _INFER_STATE


def predict_text(text: str) -> dict:
    state = _load_inference_state()
    device = state["device"]
    tokenizer = state["tokenizer"]
    model = state["model"]
    class_names = state["class_names"]

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=CONFIG["tokenization"]["max_length"],
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = torch.max(probabilities).item()

    predicted_label = class_names[predicted_index]

    return {"label": predicted_label, "confidence": confidence}


if __name__ == "__main__":
    examples = [
        "I feel so lonely tonight.",
        "I just got promoted at work!",
        "That noise scared me.",
        "I'm embarrassed about what I said.",
        "I love spending time with my family.",
    ]

    for text in examples:
        print(text)
        print(predict_text(text))
        print()
