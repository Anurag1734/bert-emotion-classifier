**Phase 3 — Model, Optimizer & Scheduler Summary**

---

## 1. Model Architecture

The `EmotionClassifier` in `src/model.py` implements a straightforward BERT-based classifier for 10-class emotion recognition:

**Components:**
- **BERT Backbone**: `BertModel.from_pretrained("bert-base-uncased")` — fixed pre-trained BERT weights at start (fine-tuning enabled during training).
- **CLS Token Extraction**: `outputs.last_hidden_state[:, 0, :]` — extracts the learned [CLS] token embedding (shape: `batch_size × 768`).
- **Dropout Layer**: `nn.Dropout(dropout_prob)` — applied to CLS embedding before classification, reducing overfitting.
- **Classification Head**: Single `nn.Linear(768 → num_labels=10)` layer projecting CLS embedding to logits.

**Forward Pass:**
```
input_ids + attention_mask 
  → BertModel layers 
  → [CLS] token (position 0) 
  → Dropout 
  → Linear projection 
  → logits (batch_size × 10)
```

**Output:** Raw logits only (no softmax or argmax). Loss function (e.g., CrossEntropyLoss) will apply softmax internally during training.

---

## 2. Why CLS Token Instead of Pooler Output

**BERT Architecture Detail:**
BERT provides two ways to extract a sequence representation:
- **pooler_output**: A learned linear projection of [CLS] applied *inside* the base model `(batch_size × 768)`.
- **last_hidden_state[:, 0, :]**: The raw [CLS] token embedding from the final transformer layer `(batch_size × 768)`.

**Choice: CLS Token (last_hidden_state[:, 0, :])**

Reasons:
1. **Transparency:** Raw embeddings are unmodified; no hidden nonlinear projection inside the model.
2. **Fine-tuning Clarity:** During fine-tuning, we control exactly how the representation is used (via our own Dropout + Linear).
3. **Standard Practice:** Most modern fine-tuning recipes use raw [CLS] embeddings and apply task-specific heads on top.
4. **Avoids Double-Processing:** The pooler is not part of the pre-trained language modeling objective; bypassing it keeps the pipeline cleaner.

---

## 3. Classification Head Design Reasoning

**Design: Single Linear Layer (Dropout → Linear)**

Rationale:
- **Shallow Head:** BERT's [CLS] embedding (768-dim) is already highly expressive due to pre-training on masked language modeling and next-sentence prediction. It encodes rich semantic information about the input.
- **No Hidden Layers:** Adding deep multi-layer MLP heads often overfits on smaller fine-tuning datasets and increases training time. A single linear layer is standard for classification with pre-trained LLMs.
- **Dropout Before Linear:** Regularization is applied to [CLS] to reduce overfitting, not within the head itself.
- **Minimal Parameters:** Only ~7.7K additional parameters (768×10 + 10 bias), keeping the fine-tuned model lean and efficient.

**Alternative (Not Used):**
A deeper head (e.g., Linear → ReLU → Linear) would be reserved for future experimentation phases (e.g., if the dataset is very large or the problem requires more complex intermediate reasoning).

---

## 4. Parameter Count

| Component | Parameters |
|-----------|-----------|
| BertModel (110M base weights) | ~110,086,912 |
| Classification Linear (768 → 10) | 7,690 |
| Dropout | 0 |
| **Total** | **~110,094,602** |

**Trainable Parameters:** All 110,094,602 are trainable by default during fine-tuning (no frozen layers in Phase 3).

**Note:** For very large models or limited GPU memory, selective layer freezing or LoRA-style techniques could be applied in future phases, but not in Phase 3.

---

## 5. Weight Decay Grouping Explanation

**Optimizer: AdamW with Selective Weight Decay**

Problem:
- Indiscriminately applying weight decay to all parameters can hurt convergence, especially for bias terms and normalization layer weights.

Solution (`build_optimizer` in `src/optimizer.py`):

**No Weight Decay (weight_decay = 0.0):**
- Parameters containing `'bias'` in their name.
- Parameters containing `'LayerNorm.weight'` in their name.

Reason: Bias terms and LayerNorm scales are non-linear components whose magnitude depends on the input distribution; decaying them treats them the same as learned feature weights, which is suboptimal.

**Standard Weight Decay (weight_decay = 0.01 from CONFIG):**
- All other parameters (transformer attention layers, feed-forward layers, embeddings, classifier weights).

Reason: Weight decay regularizes the learned feature representations, preventing excessively large weights and improving generalization.

**Parameter Groups Implementation:**
```python
optimizer_grouped_params = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_grouped_params, lr=learning_rate)
```

---

## 6. Warmup Schedule Explanation

**Scheduler: Linear Warmup + Linear Decay**

Purpose:
- **Warmup Phase:** Gradually increase learning rate from 0 to the target LR over the first `warmup_steps`. Prevents large gradient updates early in training when the model is far from a good solution.
- **Decay Phase:** After warmup, linearly decrease learning rate to 0 over the remaining training steps. Encourages convergence by reducing step size as training progresses.

**Calculation:**

From config:
- `num_epochs = 10`
- `warmup_ratio = 0.1` (10% of total steps)
- `learning_rate = 2e-5`

Given a hypothetical training set size that yields, say, 5000 steps per epoch:
- `num_training_steps = 10 × 5000 = 50,000`
- `warmup_steps = int(0.1 × 50,000) = 5,000`

**LR Schedule:**
- Steps 0–5,000: LR increases linearly from 0 to 2e-5.
- Steps 5,000–50,000: LR decreases linearly from 2e-5 to 0.

**Implementation:**
`build_scheduler` in `src/optimizer.py` uses `transformers.get_linear_schedule_with_warmup()`, which is the standard HuggingFace scheduler.

---

## 7. Dependency Flow

**Configuration → Model → Optimizer → Scheduler**

```
CONFIG (src/config.py)
  ├── model.pretrained_model_name → EmotionClassifier.__init__()
  ├── model.num_labels → EmotionClassifier.__init__()
  ├── model.dropout → EmotionClassifier.__init__()
  ├── training.learning_rate → build_optimizer()
  ├── training.weight_decay → build_optimizer()
  ├── training.num_epochs → (compute num_training_steps)
  └── training.warmup_ratio → build_scheduler()

EmotionClassifier (src/model.py)
  └── model.parameters() → build_optimizer()

build_optimizer() (src/optimizer.py)
  ├── Returns: optimizer instance
  └── optimizer → build_scheduler()

build_scheduler() (src/optimizer.py)
  ├── Inputs: optimizer, num_training_steps (= epochs × steps_per_epoch)
  └── Returns: learning rate scheduler
```

**Integration Flow (typical training loop, not implemented in Phase 3):**
1. Load CONFIG.
2. Instantiate EmotionClassifier with CONFIG['model'] values.
3. Call build_optimizer(model, CONFIG).
4. Compute num_training_steps = CONFIG['training']['num_epochs'] × len(train_loader).
5. Call build_scheduler(optimizer, num_training_steps, CONFIG).
6. Train loop: step optimizer, update LR schedule per batch or epoch.

---

## 8. What Is Intentionally NOT Implemented in Phase 3

**Deferred to Phase 4+ (Training Loop & Evaluation):**

1. **Training Loop**: Forward pass, backward pass, optimizer step, scheduler step.
2. **Validation & Metrics**: Accuracy, F1, precision, recall, confusion matrices.
3. **Evaluation on Test Set**: Final model assessment.
4. **Inference Pipeline**: Loading a trained model and predicting on new examples.
5. **Model Checkpointing**: Saving best model, resuming from checkpoint.
6. **Early Stopping**: Monitoring validation loss and halting training.
7. **Learning Rate Warmup** (applied during training step, not in Phase 3 code).
8. **Loss Function**: CrossEntropyLoss is not defined here; chosen at training time.
9. **Device Management** (CPU/GPU): Deferred to training integration.
10. **Distributed Training**: Single-GPU/CPU training only in Phase 3.

---

## 9. Risks & Limitations

### Training Risks
1. **No Early Stopping**: Model may overfit if training for all 10 epochs without validation monitoring.
2. **Fixed Learning Rate Schedule**: Warmup + decay is rigid; no dynamic adjustment based on validation loss plateau.
3. **Large Model (110M params)**: Requires significant GPU memory (~4–6 GB with batch_size=32). CPU training will be slow.
4. **10-Class Problem Complexity**: Multi-class emotion classification across 10 distinct classes is inherently harder than binary/multi-label. Risk of class confusion with similar emotions. Mitigated partially by BERT's semantic understanding.

### Data Risks
1. **Class Imbalance**: If emotion classes are unevenly distributed within the "balanced" dataset, standard CE loss may bias toward majority classes. Unaddressed in Phase 3.
2. **Tokenization Edge Cases**: max_length=72 may truncate very long texts; information loss is not quantified per sample.

### Optimization Risks
1. **Weight Decay Tuning**: Standard value 0.01 may not be optimal for this dataset/task. Manual tuning deferred.
2. **Warmup Ratio**: 10% warmup may be too short for some initialization schemes. No ablation in Phase 3.

### Integration Risks
1. **Hardcoded Model Name**: "bert-base-uncased" is hardcoded in EmotionClassifier. Switching to another BERT variant (e.g., bert-large-uncased) requires code change.
2. **Num Labels Assumption**: CONFIG['model']['num_labels'] must be set correctly; currently set to 10 to match dataset. Runtime check deferred.
3. **Hidden Size Assumption**: Code infers hidden_size from self.bert.config.hidden_size. Works for BERT but may break if swapped to a model with different architecture.

### Future Optimization Opportunities
1. **Layer Freezing**: Freeze early transformer layers for faster training if convergence is slow.
2. **LoRA Adapters**: Use parameter-efficient fine-tuning to reduce trainable params.
3. **Mixed Precision**: Use FP16 to reduce memory and speed up training (PyTorch native or Apex).
4. **Gradient Accumulation**: Simulate larger batch sizes without extra memory.
5. **Custom Collate Function**: Implement dynamic padding to reduce padding tokens and speed up training.
6. **Class Weighting**: Weight loss by class frequency to handle potential imbalance.

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Backbone** | BERT Base Uncased (110M params) |
| **Classification Head** | Dropout → Linear(768 → 10) |
| **Token Extraction** | CLS token (not pooler_output) |
| **Num Emotion Classes** | 10 (balanced dataset) |
| **Total Parameters** | ~110.1M (all trainable) |
| **Optimizer** | AdamW with weight decay grouping |
| **Learning Rate** | 2e-5 |
| **Warmup** | 10% of training steps, linear increase then linear decay |
| **Config Source** | `src/config.py` (no hardcoding) |
| **What's Not Done** | Training loop, validation, inference, checkpointing, loss function |

---

## Files Modified / Created

- **src/config.py**: Updated `model.num_labels = 10`.
- **src/model.py**: Implemented `EmotionClassifier` with forward() returning logits; uses num_labels from config.
- **src/optimizer.py**: Implemented `build_optimizer()` and `build_scheduler()`.
- **docs/PHASE_3_MODEL_OPTIMIZATION_SUMMARY.md**: This document.
