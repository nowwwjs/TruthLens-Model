"""Typed configuration for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    """A checkpoint participating in an ensemble."""

    name: str
    architecture: str
    checkpoint: Path
    ensemble_weight: float
    dropout: float = 0.5
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.architecture not in {"efficientnet_b0", "mobilenet_v3"}:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_classes != 2:
            raise ValueError("TruthLens currently supports binary classification only")
        if self.ensemble_weight <= 0:
            raise ValueError("ensemble_weight must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration shared by inference and evaluation."""

    models: Tuple[ModelSpec, ...]
    threshold: float = 0.5
    use_face_crop: bool = True
    version: str = "portfolio-refactor-v1"

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("At least one model is required")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        total = sum(model.ensemble_weight for model in self.models)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total:.6f}")


DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    models=(
        ModelSpec(
            name="eff_b0_finetuned",
            architecture="efficientnet_b0",
            checkpoint=PROJECT_ROOT / "model" / "weights" / "eff_b0_finetuned.pth",
            ensemble_weight=0.8,
        ),
        ModelSpec(
            name="mobilenet_v3_dfdc",
            architecture="mobilenet_v3",
            checkpoint=PROJECT_ROOT / "weights" / "dfdc_mobilenet_v3_focal.pth",
            ensemble_weight=0.2,
        ),
    )
)


@dataclass(frozen=True)
class TrainingConfig:
    """Regularization and optimization defaults used by training scripts."""

    seed: int = 42
    epochs: int = 10
    warmup_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.5
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    early_stopping_patience: int = 3
    min_delta: float = 1e-4
    metric: str = "roc_auc"
    augmentation: dict = field(
        default_factory=lambda: {
            "horizontal_flip": 0.5,
            "rotation_degrees": 15,
            "color_jitter": 0.2,
            "random_erasing": 0.3,
        }
    )
