"""Batch data collator for SAM3 training.

This is a minimal stub to satisfy imports when using SAM3 for inference only.
The full implementation is available in the official SAM3 repository.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BatchedDatapoint:
    """Batched datapoint for SAM3 training."""
    pass


def collate_fn_api(*args, **kwargs):
    """Collate function API stub."""
    raise NotImplementedError(
        "Training data collation is not implemented in this inference-only setup"
    )
