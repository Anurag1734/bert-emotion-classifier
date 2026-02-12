from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from bert_emotion_project.src.config import CONFIG


def get_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset=None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    batch_size = CONFIG["training"]["batch_size"]
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader
