"""Train one regularized backbone; train both backbones for the ensemble."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from truthlens.config import TrainingConfig
from truthlens.data import ManifestDataset, load_manifest, manifest_sha256
from truthlens.models import create_model
from truthlens.training import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="TruthLens regularized transfer-learning trainer")
    parser.add_argument("--arch", choices=["efficientnet_b0", "mobilenet_v3"], default="efficientnet_b0")
    parser.add_argument("--train-manifest", type=Path, default=Path("dataset/manifests/combined_train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("dataset/manifests/combined_val.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    config = replace(
        TrainingConfig(),
        seed=args.seed,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
    )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_records = load_manifest(args.train_manifest)
    val_records = load_manifest(args.val_manifest)
    train_loader = DataLoader(
        ManifestDataset(train_records, train=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        ManifestDataset(val_records, train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    model = create_model(args.arch, dropout_rate=config.dropout, pretrained=True).to(device)
    output = args.output or Path("weights") / f"combined_{args.arch}_focal.pth"
    fit(model, args.arch, train_loader, val_loader, config, output, device)

    metadata = {
        "architecture": args.arch,
        "checkpoint": output.name,
        "train_manifest": args.train_manifest.name,
        "validation_manifest": args.val_manifest.name,
        "train_manifest_sha256": manifest_sha256(train_records),
        "validation_manifest_sha256": manifest_sha256(val_records),
        "training_config": asdict(config),
        "device_type": device,
    }
    output.with_suffix(".experiment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
