"""Backward-compatible configuration view.

The canonical typed configuration lives in :mod:`truthlens.config`.
"""

from truthlens.config import DEFAULT_PIPELINE_CONFIG, PROJECT_ROOT

MODEL_LIST = {
    spec.name: {
        "weights": spec.checkpoint,
        "arch": spec.architecture,
        "num_classes": spec.num_classes,
        "threshold": DEFAULT_PIPELINE_CONFIG.threshold,
        "dropout": spec.dropout,
    }
    for spec in DEFAULT_PIPELINE_CONFIG.models
}
MODEL_LIST["efficientnet_b0_dfdc"] = {
    "weights": PROJECT_ROOT / "weights" / "dfdc_efficientnet_b0_focal.pth",
    "arch": "efficientnet_b0",
    "num_classes": 2,
    "threshold": 0.5,
    "dropout": 0.5,
}

DEFAULT_MODEL_NAME = DEFAULT_PIPELINE_CONFIG.models[0].name
ENSEMBLE_CONFIG = {
    "models": [
        {"name": spec.name, "weight": spec.ensemble_weight}
        for spec in DEFAULT_PIPELINE_CONFIG.models
    ],
    "threshold": DEFAULT_PIPELINE_CONFIG.threshold,
    "version": DEFAULT_PIPELINE_CONFIG.version,
}
