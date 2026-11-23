"""Prior snapshot and storage for LRIP (semantic segmentation).

A pixel-wise prior must stay aligned with the random crops and flips applied
after t*. The snapshot is therefore taken on the deterministic geometric view
of each training image (the resize that precedes the random augmentations), and
the stored prior map goes through the same random crop / flip as the image and
the mask (it is passed to albumentations as an extra mask target).

Maps are stored at 1/stride resolution in float16 (DeepLabV3 predicts at
output stride 8, so stride 8 loses nothing) and upsampled on the fly.
"""
import logging
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SegPriorStore:
    def __init__(self, maps: List[np.ndarray], num_classes: int) -> None:
        self.maps = maps            # list of [h, w, C] float16 arrays
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.maps)

    def get(self, index: int, size: Tuple[int, int]) -> np.ndarray:
        """Prior map of sample ``index`` resized to ``size`` = (H, W), [H, W, C] float32."""
        small = self.maps[index].astype(np.float32)
        h, w = size
        prior = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if prior.ndim == 2:  # cv2 drops the channel axis when C == 1
            prior = prior[..., None]
        prior /= np.maximum(prior.sum(axis=-1, keepdims=True), 1e-6)
        return prior

    def state_dict(self) -> dict:
        return {"maps": self.maps, "num_classes": self.num_classes}

    @classmethod
    def from_state_dict(cls, state: dict) -> "SegPriorStore":
        return cls(state["maps"], state["num_classes"])


@torch.no_grad()
def collect_seg_priors(
    loader: DataLoader,
    forward: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    num_classes: int,
    device: torch.device,
    stride: int = 8,
    size_divisor: int = 32,
) -> SegPriorStore:
    """``loader`` yields ``(idx, image[1, 3, H, W], mask)`` one image at a time."""
    maps: List[np.ndarray] = [None] * num_samples
    for idx, image, _ in loader:
        image = image.to(device, non_blocking=True)
        h, w = image.shape[-2:]
        ph, pw = (-h) % size_divisor, (-w) % size_divisor
        if ph or pw:
            image = F.pad(image, (0, pw, 0, ph))
        probs = F.softmax(forward(image).float(), dim=1)[..., :h, :w]
        small = F.adaptive_avg_pool2d(probs, (-(-h // stride), -(-w // stride)))
        for i, j in enumerate(idx.tolist()):
            maps[j] = small[i].permute(1, 2, 0).cpu().numpy().astype(np.float16)
    missing = sum(m is None for m in maps)
    if missing:
        raise RuntimeError("LRIP: {} training images have no prior".format(missing))
    nbytes = sum(m.nbytes for m in maps)
    logger.info("LRIP: stored prior maps for {} images ({:.0f} MB)".format(
        num_samples, nbytes / 2 ** 20))
    return SegPriorStore(maps, num_classes)
