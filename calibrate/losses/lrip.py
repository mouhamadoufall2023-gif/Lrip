"""Learning Rate Informed Priors (LRIP) objective.

    L(x_n) = (1 - eps) * CE(y_n, p_n) + eps * KL(u_LR(x_n) || p_n)

where u_LR(x_n) = p(x_n; theta_t*) is the predictive distribution of the model
at the learning-rate peak t*. Before t* no prior exists yet and the loss is the
plain cross-entropy (exploration stage).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRIPLoss(nn.Module):
    names = ("loss", "loss_ce", "loss_kl")

    def __init__(self, epsilon: float = 0.1, ignore_index: int = -100) -> None:
        super().__init__()
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1), got {}".format(epsilon))
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        logits : [N, C] or [N, C, *spatial]
        target : [N]    or [N, *spatial]          (class indices)
        prior  : same shape as logits, a distribution over dim 1, or None
        """
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        if prior is None:
            return ce, ce.detach(), torch.zeros_like(ce)

        log_p = F.log_softmax(logits, dim=1)
        prior = prior.to(log_p.dtype)
        # KL(u || p) = sum_k u_k (log u_k - log p_k), computed per sample / pixel
        kl = (torch.xlogy(prior, prior) - prior * log_p).sum(dim=1)
        valid = target != self.ignore_index
        kl = kl[valid].mean() if valid.any() else kl.sum() * 0.0

        total = (1.0 - self.epsilon) * ce + self.epsilon * kl
        return total, ce.detach(), kl.detach()

    def extra_repr(self) -> str:
        return "epsilon={}, ignore_index={}".format(self.epsilon, self.ignore_index)
