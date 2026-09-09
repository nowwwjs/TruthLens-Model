"""Dependency-light weighted soft-voting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def validate_ensemble_weights(weights: Mapping[str, float]) -> None:
    if not weights:
        raise ValueError("At least one ensemble weight is required")
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("Every ensemble weight must be positive")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ensemble weights must sum to 1.0, got {total:.6f}")


def weighted_soft_vote(
    probabilities: Mapping[str, float | Sequence[float]],
    weights: Mapping[str, float],
) -> float | list[float]:
    """Combine fake-class probabilities without assuming a tensor library."""

    validate_ensemble_weights(weights)
    if set(probabilities) != set(weights):
        raise ValueError("Probability and weight model names must match")

    first = next(iter(probabilities.values()))
    if isinstance(first, (int, float)):
        score = sum(float(probabilities[name]) * weight for name, weight in weights.items())
        if not 0.0 <= score <= 1.0:
            raise ValueError("Probabilities must be in [0, 1]")
        return score

    lengths = {len(probabilities[name]) for name in weights}  # type: ignore[arg-type]
    if len(lengths) != 1:
        raise ValueError("Every probability sequence must have the same length")
    result = [
        sum(float(probabilities[name][index]) * weight for name, weight in weights.items())  # type: ignore[index]
        for index in range(lengths.pop())
    ]
    if any(not 0.0 <= score <= 1.0 for score in result):
        raise ValueError("Probabilities must be in [0, 1]")
    return result
