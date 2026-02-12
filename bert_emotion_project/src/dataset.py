"""Dataset implementation for emotion classification.

This module provides `EmotionDataset` which wraps raw texts and labels and
performs tokenization on-the-fly in ``__getitem__`` using a provided
HuggingFace tokenizer. It returns PyTorch tensors suitable for a training loop.

Design constraints:
- Do not pre-tokenize the entire dataset in memory.
- Labels must be numeric (int). No string-to-int mapping is performed internally.
- Use `tokenizer.encode_plus` per sample with the exact arguments required.
"""

from typing import List

import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification.

    Args:
        texts: List of raw text strings.
        labels: List of numeric labels (int). Must not be string labels.
        tokenizer: A HuggingFace tokenizer instance (e.g., BertTokenizerFast).
        max_length: Integer max sequence length for padding/truncation.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        # encode_plus returns tensors with a leading batch dim (1, L); remove it
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor,
        }
