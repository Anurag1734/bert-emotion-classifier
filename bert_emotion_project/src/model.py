import torch.nn as nn
from transformers import BertModel


class EmotionClassifier(nn.Module):
    def __init__(
        self, num_labels: int, dropout_prob: float, pretrained_model_name: str
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits
