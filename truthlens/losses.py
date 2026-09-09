"""Training losses."""

from __future__ import annotations


def focal_loss(inputs, targets, alpha: float = 1.0, gamma: float = 2.0):
    import torch
    import torch.nn.functional as functional

    cross_entropy = functional.cross_entropy(inputs, targets, reduction="none")
    probability = torch.exp(-cross_entropy)
    return (alpha * (1 - probability) ** gamma * cross_entropy).mean()


def make_focal_loss(alpha: float = 1.0, gamma: float = 2.0):
    import torch.nn as nn

    class FocalLoss(nn.Module):
        def forward(self, inputs, targets):
            return focal_loss(inputs, targets, alpha=alpha, gamma=gamma)

    return FocalLoss()
