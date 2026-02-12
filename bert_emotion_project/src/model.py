"""
Model definition module.

Purpose:
    - Define EmotionClassifier that uses a BERT backbone and a simple
      classification head for emotion prediction.

Responsibilities:
    - Provide EmotionClassifier with forward() that returns logits.
    - Extract CLS token (not pooler_output) for fine-tuning.
    - Single-layer classification head (Dropout -> Linear).

Dependencies:
    - torch.nn
    - transformers.BertModel
"""

import torch
import torch.nn as nn
from transformers import BertModel
from bert_emotion_project.src.config import CONFIG


class EmotionClassifier(nn.Module):
    """BERT-based emotion classifier.
    
    Extracts the CLS token from BERT's last hidden state, applies dropout,
    and projects to num_labels via a single linear layer.
    
    Args:
        num_labels (int): Number of emotion classes.
        dropout_prob (float): Dropout probability before classification layer.
        pretrained_model_name (str): HuggingFace model identifier for BertModel.
    """

    def __init__(self, num_labels: int, dropout_prob: float, pretrained_model_name: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_labels)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
        logits = self.classifier(self.dropout(cls_embedding))
        return logits
