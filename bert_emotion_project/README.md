# BERT Emotion Classification

Fine-tuning `bert-base-uncased` for multiclass emotion classification using a custom PyTorch training loop (no Hugging Face `Trainer`).

## Dataset
- Hugging Face ID: `shreyaspullehf/emotion_dataset_100k`
- Task: 10-class emotion detection

## Project Structure
- `src/config.py`: central configuration
- `src/dataset.py`: PyTorch `Dataset`
- `src/model.py`: BERT + classifier head
- `src/optimizer.py`: AdamW + linear warmup/decay scheduler
- `src/trainer.py`: training/validation/test loop, early stopping, checkpointing
- `src/evaluator.py`: metrics + confusion matrix plotting
- `src/inference.py`: `predict_text(text)` inference API
- `scripts/phase1_data_prep.py`: EDA, MAX_LENGTH calculation, summary generation
- `scripts/phase4_train.py`: end-to-end training and evaluation

## Setup
```bash
uv sync
```

## Run
1. Generate EDA artifacts and update tokenization max length:
```bash
uv run python scripts/phase1_data_prep.py
```

2. Train and evaluate:
```bash
uv run python scripts/phase4_train.py
```

3. Run sample inference:
```bash
uv run python -m bert_emotion_project.src.inference
```

## Outputs
- Best model checkpoint: `models/best_model.pt`
- Training log: `reports/training_log.csv`
- Label/text-length plots: `reports/figures/`
- Confusion matrix: `reports/figures/confusion_matrix.png`
- Data summary: `docs/PHASE_1_DATA_SUMMARY.md`

## Metrics Reported
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 score (macro and weighted)
- Confusion matrix
