from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG: Dict[str, Any] = {
    "seed": 42,
    "paths": {
        "root": str(ROOT_DIR),
        "data_raw": str(ROOT_DIR / "data" / "raw"),
        "data_processed": str(ROOT_DIR / "data" / "processed"),
        "models": str(ROOT_DIR / "models"),
        "reports": str(ROOT_DIR / "reports"),
    },
    "dataset": {
        "hf_identifier": "shreyaspullehf/emotion_dataset_100k",
    },
    "model": {
        "pretrained_model_name": "bert-base-uncased",
        "num_labels": 10,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 2,
        "validation_split": 0.1,
        "test_split": 0.1,
    },
    "tokenization": {
        "percentile_max_length": 95,
    },
    "data": {
        "text_col": "text",
        "label_col": "label",
    },
}


def get_config() -> Dict[str, Any]:
    return dict(CONFIG)


CONFIG["tokenization"]["max_length"] = 72
