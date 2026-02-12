from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def build_optimizer(model, config):
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_params, lr=learning_rate)
    return optimizer


def build_scheduler(optimizer, num_training_steps, config):
    warmup_ratio = config["training"]["warmup_ratio"]
    warmup_steps = int(warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler
