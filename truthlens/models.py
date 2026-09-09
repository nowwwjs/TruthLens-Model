"""Backbone creation and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_ARCHITECTURES = {"efficientnet_b0", "mobilenet_v3"}


def create_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    pretrained: bool = True,
):
    """Create a transfer-learning backbone.

    ``pretrained=False`` is the deterministic, offline-safe path used for
    checkpoint loading and tests. ImageNet weights are requested only when the
    caller explicitly keeps ``pretrained=True``.
    """

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")

    import torch.nn as nn
    from torchvision import models

    if architecture == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model


def extract_state_dict(checkpoint: Any) -> dict:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def load_model_checkpoint(spec, device: str):
    import torch

    checkpoint_path = Path(spec.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = create_model(
        architecture=spec.architecture,
        num_classes=spec.num_classes,
        dropout_rate=spec.dropout,
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(extract_state_dict(checkpoint))
    return model.to(device).eval()


def set_transfer_learning_stage(model, architecture: str, stage: str) -> None:
    """Apply head-only, partial, or full fine-tuning."""

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if stage not in {"head", "partial", "full"}:
        raise ValueError("stage must be one of: head, partial, full")

    for parameter in model.parameters():
        parameter.requires_grad = stage == "full"
    if stage == "full":
        return

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if stage == "partial":
        last_block = model.features[-1]
        for parameter in last_block.parameters():
            parameter.requires_grad = True
