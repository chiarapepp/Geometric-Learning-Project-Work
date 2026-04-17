import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@contextmanager
def cuda_peak_memory(device):
    use_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    yield
    if use_cuda:
        torch.cuda.synchronize(device)


def peak_memory_mb(device):
    use_cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    if not use_cuda:
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


def time_call(fn, warmup=1, repeats=5, device="cpu"):
    for _ in range(warmup):
        fn()
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / float(repeats)
    return result, elapsed
