"""
Deterministic seed control utilities.

Purpose:
    Provide a single helper to set seeds across Python RNGs and PyTorch to
    maximize reproducibility for experiments.

Responsibilities:
    - Set seeds for `random`, `numpy`, and `torch`.
    - Configure `torch.backends.cudnn` deterministic flags.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): Seed value from CONFIG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
