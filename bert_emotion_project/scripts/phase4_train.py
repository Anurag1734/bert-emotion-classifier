from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import ClassLabel

from bert_emotion_project.src.config import CONFIG
from bert_emotion_project.src.dataset import EmotionDataset
from bert_emotion_project.src.data_loader import get_dataloaders
from bert_emotion_project.src.model import EmotionClassifier
from bert_emotion_project.src.optimizer import build_optimizer, build_scheduler
from bert_emotion_project.src.trainer import Trainer
from bert_emotion_project.src.evaluator import plot_confusion_matrix
from bert_emotion_project.src.seed import set_seed


def create_splits(dataset, label_col, val_ratio=0.1, test_ratio=0.1, seed=42):
    split1 = dataset.train_test_split(
        test_size=test_ratio, stratify_by_column=label_col, seed=seed
    )

    val_size = val_ratio / (1.0 - test_ratio)

    split2 = split1["train"].train_test_split(
        test_size=val_size, stratify_by_column=label_col, seed=seed
    )

    return split2["train"], split2["test"], split1["test"]


def main():
    set_seed(CONFIG["seed"])

    print("=" * 60)
    print("PHASE 4: Training & Evaluation")
    print("=" * 60)
    print()

    print("Loading dataset...")
    ds = load_dataset(CONFIG["dataset"]["hf_identifier"])
    full_dataset = ds["train"]

    print(f"Dataset size: {len(full_dataset)} samples")
    print(f"Available columns: {full_dataset.column_names}")
    print()

    text_col = "sentence" if "sentence" in full_dataset.column_names else "text"
    label_col = "emotion" if "emotion" in full_dataset.column_names else "label"

    print(f"Using text column:  {text_col}")
    print(f"Using label column: {label_col}")
    print()

    if not isinstance(full_dataset.features[label_col], ClassLabel):
        sample_value = full_dataset[label_col][0]
        if isinstance(sample_value, str):
            unique_labels = sorted(list(set(full_dataset[label_col])))
            class_label = ClassLabel(names=unique_labels)
        else:
            unique_labels = sorted(list(set(full_dataset[label_col])))
            class_label = ClassLabel(names=[str(x) for x in unique_labels])
        full_dataset = full_dataset.cast_column(label_col, class_label)

    class_names = full_dataset.features[label_col].names

    print(f"Number of classes: {len(class_names)}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["pretrained_model_name"])
    max_length = CONFIG["tokenization"]["max_length"]
    print(f"Max length: {max_length}")
    print()

    print("Creating deterministic stratified 80/10/10 split...")
    train_data, val_data, test_data = create_splits(
        full_dataset,
        label_col=label_col,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=CONFIG["seed"],
    )

    print(f"Train: {len(train_data)}")
    print(f"Val:   {len(val_data)}")
    print(f"Test:  {len(test_data)}")
    print()

    train_texts = [str(x[text_col]) for x in train_data]
    train_labels = [int(x[label_col]) for x in train_data]

    val_texts = [str(x[text_col]) for x in val_data]
    val_labels = [int(x[label_col]) for x in val_data]

    test_texts = [str(x[text_col]) for x in test_data]
    test_labels = [int(x[label_col]) for x in test_data]

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length)

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, num_workers=0
    )
    print()

    print("Initializing model...")
    model = EmotionClassifier(
        num_labels=len(class_names),
        dropout_prob=CONFIG["model"]["dropout"],
        pretrained_model_name=CONFIG["model"]["pretrained_model_name"],
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("Building optimizer...")
    optimizer = build_optimizer(model, CONFIG)

    num_training_steps = CONFIG["training"]["num_epochs"] * len(train_loader)

    print("Building scheduler...")
    scheduler = build_scheduler(optimizer, num_training_steps, CONFIG)

    print(f"Total training steps: {num_training_steps}")
    print()

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer = Trainer(
        model, train_loader, val_loader, test_loader, optimizer, scheduler, CONFIG
    )

    test_metrics, elapsed_time = trainer.train()

    print("=" * 60)
    print("Training Summary")
    print("-" * 60)
    print(f"Best Epoch: {trainer.best_epoch}")
    print(f"Best Validation Macro F1: {trainer.best_val_macro_f1:.4f}")
    print(f"Total Training Time: {elapsed_time:.2f} seconds")
    print()

    print("Test Set Metrics")
    print("-" * 60)
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"Recall:      {test_metrics['recall']:.4f}")
    print()

    print("Plotting confusion matrix...")

    figures_dir = Path(CONFIG["paths"]["reports"]) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        figures_dir / "confusion_matrix.png",
        class_names=class_names,
    )

    print(f"Saved to: {figures_dir / 'confusion_matrix.png'}")
    print()

    print("=" * 60)
    print("Phase 4 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
