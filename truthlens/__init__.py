"""TruthLens model portfolio package.

The package keeps training, evaluation, and inference concerns separate while
preserving the legacy ``model`` imports used by the team backend.
"""

from .config import DEFAULT_PIPELINE_CONFIG, ModelSpec, PipelineConfig
from .ensemble import validate_ensemble_weights, weighted_soft_vote

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_PIPELINE_CONFIG",
    "ModelSpec",
    "PipelineConfig",
    "validate_ensemble_weights",
    "weighted_soft_vote",
]
