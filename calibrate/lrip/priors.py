"""Prior snapshot and storage for LRIP (classification).

At t* (the learning-rate peak) one extra forward pass over the training set
records p(x_n; theta_t*) for every sample. The priors are kept in a single
[N, K] float16 tensor and gathered by sample index during the refinement stage,
so the cost after t* is one indexing op per batch.
"""
import copy
import logging
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, Subset, SubsetRandomSampler
)

logger = logging.getLogger(__name__)

# attribute names used by the datasets of this repo to hold their transform
_TRANSFORM_ATTRS = ("transform", "transformer", "data_transform")
# above this size the prior bank lives in (pinned) host memory
_MAX_DEVICE_BYTES = 1 << 30


def peak_epoch(cfg: DictConfig) -> int:
    """Epoch t* at which the snapshot is taken (number of epochs completed).

    ``loss.snapshot_epoch`` wins when set; otherwise t* is read from the LR
    schedule: first milestone for step decay, ``pct_start * T`` for OneCycle.
    """
    explicit = cfg.loss.get("snapshot_epoch", None)
    if explicit is not None:
        return int(explicit)
    sched = cfg.scheduler
    if sched.name in ("multi_step", "step"):
        milestones = sched.get("milestones", None) or [sched.get("step_size")]
        return int(milestones[0])
    if sched.name == "one_cycle":
        return int(round(sched.pct_start * cfg.train.max_epoch))
    raise ValueError(
        "Cannot infer the LR peak of scheduler '{}': set loss.snapshot_epoch."
        .format(sched.name)
    )


class IndexedDataset(Dataset):
    """Wraps a dataset so that each item is prefixed with its index."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if not isinstance(item, (tuple, list)):
            item = (item,)
        return (idx, *item)


def _sample_indices(loader: DataLoader) -> Optional[Sequence[int]]:
    sampler = loader.sampler
    if isinstance(sampler, SubsetRandomSampler):
        return list(sampler.indices)
    return None


def with_indices(loader: DataLoader) -> DataLoader:
    """Same training loader, but yielding ``(idx, inputs, labels)``."""
    kwargs = dict(
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        persistent_workers=loader.num_workers > 0,
    )
    if isinstance(loader.sampler, SubsetRandomSampler):
        kwargs["sampler"] = loader.sampler
    else:
        kwargs["shuffle"] = isinstance(loader.sampler, RandomSampler)
    return DataLoader(IndexedDataset(loader.dataset), **kwargs)


def view_with_transform_of(dataset: Dataset, reference: Dataset) -> Dataset:
    """Shallow copy of ``dataset`` that uses the transform of ``reference``.

    Used to compute the priors on the deterministic (evaluation) view of the
    training images. Falls back to ``dataset`` if no transform can be found.
    """
    if isinstance(dataset, Subset):
        return Subset(view_with_transform_of(dataset.dataset, reference), dataset.indices)
    while isinstance(reference, Subset):
        reference = reference.dataset
    for attr in _TRANSFORM_ATTRS:
        if hasattr(dataset, attr) and hasattr(reference, attr):
            view = copy.copy(dataset)
            setattr(view, attr, getattr(reference, attr))
            return view
    logger.warning("No transform attribute found: priors use the training transform.")
    return dataset


def snapshot_loader(
    train_loader: DataLoader, dataset: Dataset, batch_size: Optional[int] = None
) -> DataLoader:
    """Ordered, non-augmented-by-sampling pass over the training samples."""
    indices = _sample_indices(train_loader)
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=batch_size or 2 * train_loader.batch_size,
        sampler=indices,  # None -> sequential over the whole dataset
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )


class PriorBank:
    """[N, K] table of predictive distributions, gathered by sample index."""

    def __init__(self, num_samples: int, num_classes: int, device: torch.device) -> None:
        nbytes = num_samples * num_classes * 2
        on_device = device.type == "cuda" and nbytes <= _MAX_DEVICE_BYTES
        self.device = device
        self.table = torch.zeros(
            num_samples, num_classes, dtype=torch.float16,
            device=device if on_device else "cpu",
        )
        if not on_device and device.type == "cuda":
            self.table = self.table.pin_memory()
        self.filled = torch.zeros(num_samples, dtype=torch.bool, device=self.table.device)

    def __len__(self) -> int:
        return self.table.size(0)

    def put(self, idx: torch.Tensor, probs: torch.Tensor) -> None:
        idx = idx.to(self.table.device)
        self.table[idx] = probs.to(self.table.device, torch.float16)
        self.filled[idx] = True

    def get(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.table.device)
        return self.table[idx].to(self.device, torch.float32, non_blocking=True)

    def state_dict(self) -> dict:
        return {"table": self.table.cpu(), "filled": self.filled.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device) -> "PriorBank":
        bank = cls(*state["table"].shape, device=device)
        bank.table.copy_(state["table"])
        bank.filled.copy_(state["filled"])
        return bank


@torch.no_grad()
def collect_priors(
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    forward: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    num_classes: int,
    device: torch.device,
) -> PriorBank:
    """Fill a PriorBank from ``(idx, inputs)`` batches with softmax(forward(inputs))."""
    bank = PriorBank(num_samples, num_classes, device)
    for idx, inputs in batches:
        logits = forward(inputs.to(device, non_blocking=True))
        bank.put(idx, F.softmax(logits.float(), dim=1))
    n = int(bank.filled.sum())
    logger.info("LRIP: stored priors for {} samples".format(n))
    return bank
