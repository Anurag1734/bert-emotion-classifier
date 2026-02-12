"""
Evaluation utilities and metrics reporting.

Purpose:
    - Compute evaluation metrics (accuracy, macro & weighted F1, precision, recall, confusion matrix)
      and provide structured report artifacts.

Responsibilities:
    - Expose functions to compute metrics and plot confusion matrices.

Dependencies:
    - sklearn.metrics
    - matplotlib, seaborn
"""

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
    """
    Compute evaluation metrics.
    
    Args:
        preds: Array of predicted class indices.
        labels: Array of true class indices.
        
    Returns:
        Dict with accuracy, macro_f1, weighted_f1, precision, recall, and confusion matrix.
    """
    preds = np.array(preds)
    labels = np.array(labels)
    
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, preds)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
    }


def plot_confusion_matrix(cm, output_path, class_names=None):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix from sklearn.
        output_path: Path to save the figure (e.g., reports/figures/confusion_matrix.png).
        class_names: Optional list of class label names.
    """
    plt.figure(figsize=(12, 10))

    if class_names is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            cbar=True
        )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
