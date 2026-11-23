"""Fast CPU checks of the LRIP components (no dataset download, a few seconds)."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from calibrate.losses import LRIPLoss
from calibrate.lrip import (
    PriorBank, collect_priors, peak_epoch, snapshot_loader, with_indices,
    SegPriorStore,
)


def test_loss_matches_definition():
    torch.manual_seed(0)
    logits, target = torch.randn(8, 5), torch.randint(0, 5, (8,))
    prior = F.softmax(torch.randn(8, 5), dim=1)
    eps = 0.1
    total, ce, kl = LRIPLoss(eps)(logits, target, prior)
    log_p = F.log_softmax(logits, 1)
    ref_kl = F.kl_div(log_p, prior, reduction="batchmean")
    ref = (1 - eps) * F.cross_entropy(logits, target) + eps * ref_kl
    assert torch.allclose(total, ref, atol=1e-6)
    assert torch.allclose(kl, ref_kl, atol=1e-6)


def test_loss_is_ce_before_snapshot():
    logits, target = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
    total, ce, kl = LRIPLoss(0.1)(logits, target)
    assert torch.allclose(total, F.cross_entropy(logits, target)) and kl == 0


def test_uniform_prior_recovers_label_smoothing():
    # Sec. 4.1: with u = uniform, LRIP = LS up to the constant eps * log K
    logits, target = torch.randn(6, 4), torch.randint(0, 4, (6,))
    eps, K = 0.2, 4
    total, _, _ = LRIPLoss(eps)(logits, target, torch.full((6, 4), 1 / K))
    ls = F.cross_entropy(logits, target, label_smoothing=eps)
    assert torch.allclose(total, ls - eps * math.log(K), atol=1e-6)


def test_segmentation_ignore_index():
    logits = torch.randn(2, 3, 4, 4, requires_grad=True)
    target = torch.randint(0, 3, (2, 4, 4))
    target[0, 0] = 255
    prior = F.softmax(torch.randn(2, 3, 4, 4), dim=1)
    prior_bad = prior.clone()
    prior_bad[0, :, 0] = 255.0  # padded area: must not matter
    a = LRIPLoss(0.1, ignore_index=255)(logits, target, prior)[0]
    b = LRIPLoss(0.1, ignore_index=255)(logits, target, prior_bad)[0]
    assert torch.allclose(a, b)
    a.backward()
    assert torch.isfinite(logits.grad).all()


def test_peak_epoch_from_schedule():
    cfg = OmegaConf.create({
        "loss": {"snapshot_epoch": None},
        "train": {"max_epoch": 160},
        "scheduler": {"name": "multi_step", "milestones": [50, 100]},
    })
    assert peak_epoch(cfg) == 50
    cfg.scheduler = {"name": "one_cycle", "pct_start": 0.3}
    assert peak_epoch(cfg) == 48
    cfg.loss.snapshot_epoch = 100
    assert peak_epoch(cfg) == 100


def test_prior_bank_follows_sample_index():
    x = torch.randn(20, 3)
    y = torch.randint(0, 4, (20,))
    ds = TensorDataset(x, y)
    train_idx = list(range(0, 20, 2))
    loader = DataLoader(ds, batch_size=4, sampler=SubsetRandomSampler(train_idx))
    model = torch.nn.Linear(3, 4)
    indexed = with_indices(loader)
    snap = snapshot_loader(indexed, ds)
    bank = collect_priors(((i, xb) for i, xb, _ in snap), model, len(ds), 4, torch.device("cpu"))
    assert bank.filled.sum() == len(train_idx)
    for idx, xb, _ in indexed:  # shuffled batches get their own priors back
        ref = F.softmax(model(xb), 1)
        assert torch.allclose(bank.get(idx), ref, atol=1e-3)
    restored = PriorBank.from_state_dict(bank.state_dict(), torch.device("cpu"))
    assert torch.equal(restored.table, bank.table)


def test_seg_prior_store_resize_is_a_distribution():
    small = np.random.dirichlet(np.ones(21), size=(8, 10)).astype(np.float16)
    prior = SegPriorStore([small], 21).get(0, (64, 80))
    assert prior.shape == (64, 80, 21)
    assert np.allclose(prior.sum(-1), 1, atol=1e-4)


@pytest.mark.parametrize("h,w", [(600, 640), (480, 640), (333, 500)])
def test_voc_prior_follows_pad_crop_and_flip(h=480, w=640):
    pytest.importorskip("albumentations")
    from calibrate.data.voc import train_augment, pad_to_crop
    img = np.random.randint(0, 255, (h, w, 3), np.uint8)
    mask = np.random.randint(0, 21, (h, w)).astype(np.uint8)
    # prior = one-hot of the mask: after the joint transform they must still agree
    prior = np.eye(21, dtype=np.float32)[mask]
    img, mask, prior = pad_to_crop(img, 0), pad_to_crop(mask, 255), pad_to_crop(prior, 0.0)
    out = train_augment(with_prior=True)(image=img, mask=mask, prior=prior)
    m = out["mask"].numpy()
    p = torch.as_tensor(out["prior"]).numpy()
    valid = m != 255
    assert p.shape[:2] == m.shape == (512, 512)
    assert (p.argmax(-1)[valid] == m[valid]).all()
