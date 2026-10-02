"""Validation-based selection for fresh, single-process/DDP experiments."""
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ValidationTracker:
    patience: int = 0
    best: float = math.inf
    best_epoch: int = 0
    bad_checks: int = 0

    def update(self, value, epoch):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'Non-finite checkpoint selection metric: {value}')
        improved = value < self.best
        if improved:
            self.best, self.best_epoch, self.bad_checks = value, epoch, 0
        else:
            self.bad_checks += 1
        return improved, bool(self.patience and self.bad_checks >= self.patience)


def atomic_save_checkpoint(checkpoint, path):
    """Keep the previous best intact if writing its replacement fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb') as f:
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
