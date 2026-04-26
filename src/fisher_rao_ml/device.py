from __future__ import annotations

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available local PyTorch device.

    Apple Silicon MPS is preferred over CPU when available because it keeps these
    experiments laptop-friendly without requiring CUDA.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
