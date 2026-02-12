**Phase 2 — Data Layer Summary**

- **Goal**: Implement dataset + tokenization + DataLoader only (no model/training loop).
- **Dataset class**: `EmotionDataset` in `src/dataset.py` — on-the-fly tokenization in `__getitem__` using a passed HuggingFace tokenizer. Returns `input_ids`, `attention_mask`, and `label` as PyTorch tensors (dtype=torch.long).
- **Numeric labels only**: Dataset expects labels as integers (int). No string-to-int label mapping is performed internally.
- **Tokenizer config**: Uses `bert-base-uncased` tokenizer. In `__getitem__`, calls `tokenizer.encode_plus()` with:
  - `padding='max_length'`, `truncation=True`, `return_token_type_ids=False`, `return_tensors='pt'`.
- **Max length**: Confirmed `CONFIG['tokenization']['max_length'] = 72` (from Phase 1 decision).
- **Batch size**: Read from `CONFIG['training']['batch_size']` (no hardcoded defaults in get_dataloaders).
- **DataLoader**: `get_dataloaders(train_dataset, val_dataset, test_dataset=None, num_workers=0)` in `src/data_loader.py` returns (train_loader, val_loader, test_loader_or_None). Shuffles train, no shuffles for val/test. pin_memory enabled when CUDA available.
- **Label tensor**: Labels are stored and returned as `torch.tensor(label, dtype=torch.long)` — no None or float types.
- **Tokenization strategy**: Tokenize per-sample in `__getitem__` with `return_token_type_ids=False`. This keeps peak memory low and is deterministic across runs.
- **Split logic**: No split logic is in Phase 2. Data splitting (stratified 90/10) occurs in Phase 1 and populates the HuggingFace dataset. Phase 2 receives pre-split train/val samples and wraps them with PyTorch Dataset/DataLoader.
- **Integration check**: `scripts/phase2_integration_check.py` (placeholder for now; will be updated to use numeric labels) builds small train/val subsets, constructs `EmotionDataset` instances, wraps them with `get_dataloaders()`, and prints one batch to validate tensor shapes.
- **Files added/modified**:
  - `src/dataset.py` — `EmotionDataset` implementation (numeric labels only, return_token_type_ids=False).
  - `src/data_loader.py` — `get_dataloaders` implementation (batch_size from CONFIG).
  - `scripts/phase2_integration_check.py` — quick integration tester (placeholder).
- **Next steps (optional)**: Pre-tokenize and save to disk for faster training start-up, or implement a custom collate_fn for dynamic padding.

