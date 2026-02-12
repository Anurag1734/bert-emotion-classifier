# PHASE 0 — Architecture Summary

This document is the authoritative artifact for Phase 0: repository layout, module responsibilities,
high-level data and training flow design, validation & evaluation strategy, design principles, and
what is intentionally not implemented yet.

**Project root**: `bert_emotion_project/`

## 1. Folder structure (tree)

```
bert_emotion_project/
│
├── data/
│   ├── raw/                # Original (downloaded) dataset files
│   └── processed/          # Tokenized and split artifacts
│
├── src/                    # Application source code
│   ├── config.py           # Centralized configuration (placeholders)
│   ├── seed.py             # Reproducible seed utility
│   ├── data_loader.py      # DataLoader construction
│   ├── dataset.py          # Dataset wrappers
│   ├── model.py            # Model skeleton (BERT + head)
│   ├── optimizer.py        # Optimizer + scheduler builders
│   ├── trainer.py          # Training orchestration (skeleton)
│   ├── evaluator.py        # Metrics and reporting utilities
│   └── inference.py        # Single-sample inference helper
│
├── notebooks/              # Exploratory notebooks (kept lightweight)
│
├── reports/
│   ├── figures/            # Saved figures (confusion matrices, metrics curves)
│   └── metrics/            # Serialized metric reports (JSON/CSV)
│
├── docs/
│   └── PHASE_0_ARCHITECTURE_SUMMARY.md
│
├── models/                 # Saved checkpoints (gitignored in practice)
├── requirements.txt
├── README.md

```

## 2. Module responsibilities

- `src/config.py`
  - Purpose: Centralized configuration for hyperparameters, file paths and seed.
  - Contains: Structured placeholders for model, training and tokenization parameters.
  - Must NOT contain: Environment-specific secrets or run-time side-effects.
  - Dependencies: Standard library only; imported early by other modules.

- `src/seed.py`
  - Purpose: Single helper `set_seed(seed)` to set `random`, `numpy` and (if available) `torch` seeds
    and set cuDNN determinism options.
  - Contains: Deterministic seeding logic.
  - Must NOT contain: Training loops, dataset ops.
  - Dependencies: `numpy` and `torch` (torch imported lazily).

- `src/dataset.py`
  - Purpose: `EmotionDataset` wrapper around tokenized inputs and labels.
  - Contains: Dataset class skeleton and minimal transforms.
  - Must NOT contain: Data download or large-scale preprocessing pipelines.
  - Dependencies: Tokenizer provided externally; torch tensors created at integration.

- `src/data_loader.py`
  - Purpose: Build `DataLoader` objects for train/val/test with proper collate functions.
  - Contains: `get_dataloaders(...)` signature.
  - Must NOT contain: Hard-coded batch sizes or training loops (these come from config).
  - Dependencies: `torch.utils.data` at integration time.

- `src/model.py`
  - Purpose: Define `EmotionClassifier` that composes a `BertModel` backbone with a classification head.
  - Contains: Class skeleton and interface; no forward implementation in Phase 0.
  - Must NOT contain: Training-specific shortcuts or direct dataset references.
  - Dependencies: `transformers.BertModel` at integration.

- `src/optimizer.py`
  - Purpose: Build `AdamW` optimizer with proper parameter grouping (weight decay vs none) and expose
    scheduler factory for warmup + decay.
  - Contains: `build_optimizer` and `build_scheduler` signatures/placeholders.
  - Must NOT contain: Hard-coded parameter groups; must accept `model` and `config`.

- `src/trainer.py`
  - Purpose: High-level training orchestrator with early stopping, checkpointing, and logging.
  - Contains: `Trainer` class interface with `train()` and `evaluate()` placeholders.
  - Must NOT contain: Concrete training loop in Phase 0.

- `src/evaluator.py`
  - Purpose: Compute metrics and generate artifacts (confusion matrix, F1 reports).
  - Contains: `compute_metrics(preds, labels)` and `plot_confusion_matrix(...)` signatures.
  - Must NOT contain: Model forward-pass; training.

- `src/inference.py`
  - Purpose: Single-sample inference API: `predict_text(text, model, tokenizer, config)`.
  - Contains: Function signature and documentation; no runtime logic in Phase 0.

## 3. Planned data flow (high level)

- Raw data
  - Source: `shreyaspullehf/emotion_dataset_100k` (downloaded once into `data/raw/`).
  - Keep the raw files immutable in `data/raw/`.

- Tokenization
  - Use `transformers.BertTokenizer` to tokenize text. Tokenizer is instantiated using
    `config["model"]["pretrained_model_name"]`.
  - MAX_LENGTH is computed during preprocessing using a percentile-based rule (see below).

- Dataset
  - Tokenized inputs + labels are wrapped in `EmotionDataset` instances and serialized
    to `data/processed/` as small artifacts (optionally cached as torch tensors).

- Dataloader
  - `data_loader.get_dataloaders(...)` constructs `torch.utils.data.DataLoader` objects for
    train/val/test with appropriate `collate_fn`, batching, and `num_workers`.

### Train/Validation split logic

- Perform a stratified split on the labeled dataset to ensure label distribution parity
  between train and validation sets. Use `sklearn.model_selection.StratifiedShuffleSplit`.
- `config["training"]["validation_split"]` gives the target validation percentage.
- Save the train/val/test indices or processed files under `data/processed/` to ensure reproducibility.

### Percentile MAX_LENGTH decision logic

- Compute token lengths for the training corpus using the tokenizer and take the `p`-th
  percentile (e.g., p = `config["tokenization"]["percentile_max_length"]`) as `MAX_LENGTH`.
- Rationale: keeps memory use bounded while minimizing truncation of natural text. Document the chosen
  percentile and store it in `data/processed/metadata.json`.

## 4. Model architecture plan

- Backbone
  - Use `transformers.BertModel` (no `Trainer`). Load weights from `config["model"]["pretrained_model_name"]`.

- CLS extraction
  - Extract the representation of the `[CLS]` token (token index 0 after tokenization) from
    `last_hidden_state` and use it as the pooled representation for classification.
  - Explicitly avoid using `pooler_output` to keep behavior consistent across model variants.

- Classification head
  - Lightweight linear head: Dropout -> Linear(num_labels).
  - Rationale: a single linear layer is the standard fine-tuning head for BERT —
    BERT's CLS embedding is highly expressive, and a single linear classifier is
    easier to debug and defend academically. Keep deeper heads for later experimentation.
  - Apply logits directly; softmax applied only at inference/evaluation boundary.

## 5. Optimization strategy

- Optimizer
  - Use `AdamW` from `torch.optim`.

- Weight decay grouping
  - Group parameters into two groups: parameters with weight decay and parameters without (bias and LayerNorm weights).
  - Example rule: exclude parameter names that match `bias` or `LayerNorm.weight` from weight decay.

- Scheduler
  - Use a linear warmup + linear decay scheduler. Compute `num_warmup_steps` from
    `warmup_proportion * total_training_steps`.

- Gradient clipping
  - Clip gradients to `config["training"]["max_grad_norm"]` to stabilize training.

- Early stopping
 - Early stopping
  - Early stopping metric: **Macro F1** computed on the validation set. Model selection metric: **Macro F1**.
  - Weighted F1 is collected and reported (for stakeholder summaries) but is NOT used for early stopping or model selection.
  - Use patience defined by `config["training"]["early_stopping_patience"]` and restore the best checkpoint according to macro-F1.

## 6. Training flow (high level)

1. Initialize environment
   - `set_seed(config["seed"])` to fix randomness.
   - Instantiate tokenizer and compute `MAX_LENGTH` via percentile logic.

2. Prepare data
   - Tokenize, create `EmotionDataset` instances and build DataLoaders.

3. Build model & optimization
   - Load `BertModel` and wrap with classification head.
   - Build optimizer with proper weight decay grouping.
   - Build scheduler with warmup.

4. Epoch loop (outline, implemented later)
   - For epoch in 1..N:
       - Train on `train_loader` with gradient accumulation if configured.
       - After epoch (or step-based frequency), run validation on `val_loader`.
       - Compute metrics: macro-F1, weighted-F1, accuracy.
       - Save checkpoint when monitored metric improves.
       - Apply early stopping policy based on validation macro-F1.

5. Finalize
   - After early stopping / epochs, load best checkpoint and evaluate once on the test set.

### Why the test set is untouched during training

- The test set is reserved for a single final evaluation to estimate generalization and avoid
  leak-driven overfitting. Validation guides optimization and early stopping only.

## 7. Evaluation strategy

- Accuracy
  - Quick, interpretable macro-level correctness but sensitive to class imbalance.

- Weighted F1
  - Accounts for class imbalance by weighting per-class F1 by support. Useful when reporting
    overall system performance on imbalanced datasets.

- Macro F1
  - Average of per-class F1 (unweighted). Prioritized for model selection and early stopping
    when class-level fairness is important.

- Confusion matrix
  - Per-class error analysis and visualization to uncover systematic confusions.

- Why each is needed
  - Use Macro-F1 as the primary metric for early stopping and for model selection to ensure even performance across labels; report Weighted-F1 and Accuracy for stakeholder-facing summaries and to provide complementary perspectives.

## 8. Inference pipeline plan

- `predict_text(text, model, tokenizer, config)`
  - Tokenize text with `return_tensors='pt'`, apply necessary padding/truncation to `MAX_LENGTH`.
  - Move inputs to model device, run forward pass and extract logits.
  - Apply `softmax` to logits to obtain confidence distribution.
  - Return: `pred_label`, `confidence_score`, `topk` (optional) where `confidence_score` is the probability of `pred_label`.

- Softmax logic
  - Use `torch.nn.functional.softmax(logits, dim=-1)`; confidence is `max(softmax_probs)`.

## 9. Reproducibility plan

- Seed control
  - Use `src/seed.set_seed(seed)` early in the pipeline and persist the seed used into the
    experiment metadata.

- Determinism
  - Set `torch.backends.cudnn.deterministic = True` and `benchmark=False` when reproducibility
    is prioritized over peak throughput.

- Why important
  - Ensures experiments are comparable, reduces noise when tuning hyperparameters, and makes
    reported metrics auditable.

## 10. Risks & design tradeoffs

- Deliberately avoided complexity
  - No HuggingFace `Trainer`: We use native PyTorch loops to have full control over grouping,
    schedulers, and metrics.
  - No over-engineered experiment framework for Phase 0: keep modules simple and testable.

- Potential limitations
  - Deterministic cuDNN may reduce throughput; documented as a controllable config option.
  - Percentile-based `MAX_LENGTH` may still truncate rare long texts—tradeoff favors memory predictability.

- Where to improve later
  - Add experiment tracking (e.g., MLflow or Weights & Biases) in Phase 1.
  - Expand trainer to support mixed-precision (AMP) and distributed training.
  - Add a model registry and automated model evaluation pipelines.

## 11. Items intentionally NOT implemented in Phase 0

- No training loops or forward-pass implementations.
- No dataset download or preprocessing scripts that mutate `data/raw/`.
- No checkpointing or serialization logic.
- No plotting or heavy I/O utilities; placeholders exist for future implementation.

## 12. Next steps (Phase 1 plan summary)

1. Integrate dataset download and deterministic preprocessing; compute label mapping and update `config`.
2. Implement tokenization pipeline, compute percentile `MAX_LENGTH`, and cache processed artifacts.
3. Implement model forward pass using CLS extraction and small classification head.
4. Implement optimizer parameter grouping, scheduler, training loop, validation checks, and early stopping.
5. Integrate evaluation metrics and add reporting/figures generation.

---

This artifact is designed to be the single source of truth for reviewers during the architecture sign-off stage.
