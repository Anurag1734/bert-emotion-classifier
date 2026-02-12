"""
Optimizer and scheduler builders.

Purpose:
    - Build AdamW optimizer with proper weight decay grouping.
    - Build linear warmup + decay scheduler.

Responsibilities:
    - Implement build_optimizer with weight decay separation (bias/LayerNorm vs others).
    - Implement build_scheduler using transformers.get_linear_schedule_with_warmup.
    - Read hyperparameters from CONFIG.

Dependencies:
    - torch.optim
    - transformers.optimization
"""

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer(model, config):
    """
    Build AdamW optimizer with weight decay grouping.
    
    Parameters with 'bias' or 'LayerNorm.weight' in name get weight_decay=0.0.
    All other parameters get weight_decay from config.
    
    Args:
        model: PyTorch model with parameters to optimize.
        config (dict): Configuration dict with 'training' key.
        
    Returns:
        torch.optim.AdamW: Configured optimizer.
    """
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    # Separate parameters into groups
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_params, lr=learning_rate)
    return optimizer


def build_scheduler(optimizer, num_training_steps, config):
    """
    Build linear warmup + linear decay scheduler.
    
    Warmup steps computed as: warmup_ratio * num_training_steps
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        num_training_steps (int): Total training steps (epochs * steps_per_epoch).
        config (dict): Configuration dict with 'training' key.
        
    Returns:
        transformers.optimization.LambdaLR: Learning rate scheduler.
    """
    warmup_ratio = config['training']['warmup_ratio']
    warmup_steps = int(warmup_ratio * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler
