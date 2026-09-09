"""Evaluation, validation-only threshold selection, and result persistence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


def confusion_counts(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict[str, int]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    predictions = [int(probability >= threshold) for probability in probabilities]
    return {
        "tn": sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions)),
        "fp": sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions)),
        "fn": sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions)),
        "tp": sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions)),
    }


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict:
    counts = confusion_counts(labels, probabilities, threshold)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total = max(1, tn + fp + fn + tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else None
    except ImportError:
        roc_auc = None
    return {
        "threshold": round(threshold, 6),
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_threshold_on_validation(
    labels: Sequence[int],
    probabilities: Sequence[float],
    candidates: Iterable[float] | None = None,
) -> tuple[float, dict]:
    """Select a threshold using validation data only; test data is not accepted."""

    thresholds = list(candidates or (index / 100 for index in range(5, 96)))
    scored = [(threshold, binary_metrics(labels, probabilities, threshold)) for threshold in thresholds]
    threshold, metrics = max(scored, key=lambda item: (item[1]["f1"], item[1]["recall"], -abs(item[0] - 0.5)))
    return threshold, metrics


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_evaluation(
    results: dict[str, dict],
    output_dir: str | Path,
    metadata: dict | None = None,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "metrics.json"
    csv_path = output / "metrics.csv"
    payload = {"metrics": results, "metadata": metadata or {}}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["model", "threshold", "accuracy", "roc_auc", "precision", "recall", "f1", "confusion_matrix"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in results.items():
            row = {"model": name, **metrics}
            row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
            writer.writerow({field: row.get(field) for field in fieldnames})
    return json_path, csv_path
