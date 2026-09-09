"""Compare both backbones and their weighted soft-voting ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

from truthlens.config import DEFAULT_PIPELINE_CONFIG
from truthlens.data import ManifestDataset, load_manifest
from truthlens.ensemble import weighted_soft_vote
from truthlens.evaluation import binary_metrics, save_evaluation, select_threshold_on_validation
from truthlens.models import load_model_checkpoint


def collect_probabilities(models, loader, device: str) -> tuple[list[int], dict[str, list[float]]]:
    import torch

    labels_out: list[int] = []
    scores = {name: [] for name in models}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_out.extend(int(value) for value in labels.tolist())
            for name, model in models.items():
                values = torch.softmax(model(images), dim=1)[:, 1].cpu().tolist()
                scores[name].extend(float(value) for value in values)
    return labels_out, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="TruthLens standalone/ensemble evaluator")
    parser.add_argument("--test-manifest", type=Path, default=Path("dataset/manifests/combined_test.csv"))
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args()

    if args.calibrate_threshold and args.val_manifest is None:
        raise SystemExit("--calibrate-threshold requires --val-manifest")

    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {spec.name: load_model_checkpoint(spec, device) for spec in DEFAULT_PIPELINE_CONFIG.models}
    weights = {spec.name: spec.ensemble_weight for spec in DEFAULT_PIPELINE_CONFIG.models}

    threshold = DEFAULT_PIPELINE_CONFIG.threshold
    threshold_evidence = {"source": "default", "threshold": threshold}
    if args.calibrate_threshold:
        val_loader = DataLoader(ManifestDataset(load_manifest(args.val_manifest), train=False), batch_size=args.batch_size)
        val_labels, val_scores = collect_probabilities(models, val_loader, device)
        val_ensemble = weighted_soft_vote(val_scores, weights)
        threshold, val_metrics = select_threshold_on_validation(val_labels, val_ensemble)
        threshold_evidence = {"source": "validation", "threshold": threshold, "validation_metrics": val_metrics}

    test_loader = DataLoader(ManifestDataset(load_manifest(args.test_manifest), train=False), batch_size=args.batch_size)
    test_labels, test_scores = collect_probabilities(models, test_loader, device)
    ensemble_scores = weighted_soft_vote(test_scores, weights)
    results = {
        name: binary_metrics(test_labels, probabilities, threshold)
        for name, probabilities in test_scores.items()
    }
    results["weighted_ensemble"] = binary_metrics(test_labels, ensemble_scores, threshold)
    save_evaluation(results, args.output_dir, metadata={"threshold_evidence": threshold_evidence})


if __name__ == "__main__":
    main()
