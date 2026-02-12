import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, output_path, class_names=None):
    plt.figure(figsize=(12, 10))

    if class_names is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            square=True,
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
        )
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True, cbar=True)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
