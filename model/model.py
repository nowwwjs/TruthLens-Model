"""Backward-compatible model factory.

New code should import from :mod:`truthlens.models`.
"""

from truthlens.models import create_model as _create_model


def create_model(
    arch: str = "efficientnet_b0",
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    pretrained: bool = True,
):
    return _create_model(
        architecture=arch,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )
