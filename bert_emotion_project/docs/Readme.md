# BERT Fine-Tuning for Emotion Classification

## Overview

This project implements transfer learning by fine-tuning a BERT-based model on the dataset:

`shreyaspullehf/emotion_dataset_100k`

The objective is to perform multi-class emotion classification (10 classes) using a fully custom PyTorch training pipeline without relying on HuggingFace's Trainer API.

All components — data processing, model architecture, optimizer, scheduler, training loop, evaluation, and inference — are implemented explicitly in PyTorch.

---

# Dataset

- **Dataset ID:** `shreyaspullehf/emotion_dataset_100k`
- **Total Samples:** 99,746
- **Number of Classes:** 10 emotions
- **Balanced Dataset:** Imbalance ratio = 1.01

Emotion Classes:
- disgust
- drive
- embarrassment
- excitement
- fear
- happiness
- loneliness
- love
- sadness
- surprise

---

# Exploratory Data Analysis (Phase 1)

## Token Length Statistics (BERT Tokenization)

| Metric | Value |
|--------|--------|
| Mean | 41.11 |
| Median | 39 |
| Max | 155 |
| 95th Percentile | 65 |

## MAX_LENGTH Decision

- Percentile Used: 95%
- Rounded to nearest multiple of 8: 72
- Final MAX_LENGTH: 72
- Truncation Ratio: 3.42%

This percentile-based truncation strategy ensures minimal information loss while maintaining efficient memory usage.

## Validation Split

- Stratified 90/10 split
- Seed = 42
- Max class proportion difference (train vs val): 0.0001

All splits are deterministic and reproducible.

---

# Model Architecture (Phase 3)

## Backbone

- `bert-base-uncased`
- 110M parameters

## Classification Head

- CLS token extraction (`last_hidden_state[:, 0, :]`)
- Dropout
- Linear layer (768 → 10)

All BERT layers are fine-tuned (no freezing).

Total Parameters: ~110M

---

# Optimization Strategy

- Optimizer: AdamW
- Learning Rate: 2e-5
- Weight Decay: 0.01 (excluding bias and LayerNorm weights)
- Scheduler: Linear Warmup + Linear Decay
- Warmup Ratio: 10%
- Gradient Clipping: max_norm = 1.0
- Batch Size: 32
- Epochs: 3
- Early Stopping Patience: 2 (based on Validation Macro F1)

---

# Training & Evaluation (Phase 4)

## Data Split

Deterministic stratified split:

- Train: 80%
- Validation: 10%
- Test: 10%

Test set is used **only once** after training completes.

---

## Final Test Metrics

| Metric | Score |
|--------|--------|
| Accuracy | 0.9676 |
| Macro F1 | 0.9677 |
| Weighted F1 | 0.9677 |
| Precision | 0.9679 |
| Recall | 0.9676 |

Confusion matrix saved at:

# BERT Fine-Tuning for Emotion Classification

## Overview

This project implements transfer learning by fine-tuning a BERT-based model on the dataset:

`shreyaspullehf/emotion_dataset_100k`

The objective is to perform multi-class emotion classification (10 classes) using a fully custom PyTorch training pipeline without relying on HuggingFace's Trainer API.

All components — data processing, model architecture, optimizer, scheduler, training loop, evaluation, and inference — are implemented explicitly in PyTorch.

---

# Dataset

- **Dataset ID:** `shreyaspullehf/emotion_dataset_100k`
- **Total Samples:** 99,746
- **Number of Classes:** 10 emotions
- **Balanced Dataset:** Imbalance ratio = 1.01

Emotion Classes:
- disgust
- drive
- embarrassment
- excitement
- fear
- happiness
- loneliness
- love
- sadness
- surprise

---

# Exploratory Data Analysis (Phase 1)

## Token Length Statistics (BERT Tokenization)

| Metric | Value |
|--------|--------|
| Mean | 41.11 |
| Median | 39 |
| Max | 155 |
| 95th Percentile | 65 |

## MAX_LENGTH Decision

- Percentile Used: 95%
- Rounded to nearest multiple of 8: 72
- Final MAX_LENGTH: 72
- Truncation Ratio: 3.42%

This percentile-based truncation strategy ensures minimal information loss while maintaining efficient memory usage.

## Validation Split

- Stratified 90/10 split
- Seed = 42
- Max class proportion difference (train vs val): 0.0001

All splits are deterministic and reproducible.

---

# Model Architecture (Phase 3)

## Backbone

- `bert-base-uncased`
- 110M parameters

## Classification Head

- CLS token extraction (`last_hidden_state[:, 0, :]`)
- Dropout
- Linear layer (768 → 10)

All BERT layers are fine-tuned (no freezing).

Total Parameters: ~110M

---

# Optimization Strategy

- Optimizer: AdamW
- Learning Rate: 2e-5
- Weight Decay: 0.01 (excluding bias and LayerNorm weights)
- Scheduler: Linear Warmup + Linear Decay
- Warmup Ratio: 10%
- Gradient Clipping: max_norm = 1.0
- Batch Size: 32
- Epochs: 3
- Early Stopping Patience: 2 (based on Validation Macro F1)

---

# Training & Evaluation (Phase 4)

## Data Split

Deterministic stratified split:

- Train: 80%
- Validation: 10%
- Test: 10%

Test set is used **only once** after training completes.

---

## Final Test Metrics

| Metric | Score |
|--------|--------|
| Accuracy | 0.9676 |
| Macro F1 | 0.9677 |
| Weighted F1 | 0.9677 |
| Precision | 0.9679 |
| Recall | 0.9676 |

Confusion matrix saved at:

reports/figures/confusion_matrix.png


---

# Inference Pipeline

A standalone inference module is implemented:

```bash
python -m bert_emotion_project.src.inference
Example outputs:

Input: "I feel so lonely tonight."
Output: {'label': 'sadness', 'confidence': 0.9995}

Input: "I just got promoted at work!"
Output: {'label': 'excitement', 'confidence': 0.9982}
```

bert_emotion_project/    
├── data/  
├── src/   
│   ├── config.py  
│   ├── dataset.py  
│   ├── data_loader.py  
│   ├── model.py    
│   ├── optimizer.py    
│   ├── trainer.py  
│   ├── evaluator.py    
│   └── inference.py    
│   
├── scripts/    
│   ├── phase1_data_prep.py 
│   └── phase4_train.py 
│   
├── docs/   
│   ├── PHASE_1_DATA_SUMMARY.md 
│   ├── PHASE_3_MODEL_OPTIMIZATION_SUMMARY.md      
│   └── PHASE_4_TRAINING_EVALUATION_SUMMARY.md  
│   
├── reports/    
│   └── figures/    
│   
├── models/ 
│   └── best_model.pt   
│   
└── README.md

## Key Design Decisions

- Used CLS token instead of pooler_output for transparency.

- Applied weight decay grouping (bias & LayerNorm excluded).

- Used percentile-based MAX_LENGTH for memory efficiency.

- Ensured deterministic splits via seed control.

- Avoided HuggingFace Trainer to maintain full PyTorch control.

## Reproducibility

- Global seed = 42

- Deterministic data splits

- Explicit optimizer and scheduler setup

- Saved best checkpoint (models/best_model.pt)

## Conclusion

The model achieves high performance (96.7% Macro F1) on a 10-class emotion classification task using transfer learning with BERT.

The pipeline strictly follows assignment constraints:

- Custom PyTorch training loop

- No HuggingFace Trainer

- Proper evaluation metrics

- Confusion matrix

- Inference function

- EDA with visualization

All required components have been implemented successfully.