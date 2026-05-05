from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and downstream ML libraries.

    Sets ``PYTHONHASHSEED`` so child processes inherit the seed. If torch is
    importable it is also seeded, including CUDA generators when available.

    :param seed: Non-negative integer seed.
    :returns: The seed that was applied, for logging.
    :raises ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
