"""Reusable training loop with staged transfer learning and early stopping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import set_transfer_learning_stage


@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 1e-4
    mode: str = "max"
    best: float | None = None
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        improved = self.best is None or (
            value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_epoch(model, loader, criterion, optimizer, device: str) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((outputs.argmax(1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def evaluate_epoch(model, loader, criterion, device: str) -> dict:
    import torch
    from .evaluation import binary_metrics

    model.eval()
    total_loss = total = 0
    labels_out: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            labels_out.extend(int(value) for value in labels.cpu().tolist())
            probabilities.extend(float(value) for value in torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
    metrics = binary_metrics(labels_out, probabilities)
    metrics["loss"] = total_loss / max(1, total)
    return metrics


def fit(
    model,
    architecture: str,
    train_loader,
    val_loader,
    config,
    output_path: str | Path,
    device: str,
) -> list[dict]:
    import torch

    from .losses import make_focal_loss

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    criterion = make_focal_loss(config.focal_alpha, config.focal_gamma)
    stopper = EarlyStopping(config.early_stopping_patience, config.min_delta, mode="max")
    history: list[dict] = []

    set_transfer_learning_stage(model, architecture, "head")
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

    for epoch in range(1, config.epochs + 1):
        if epoch == config.warmup_epochs + 1:
            set_transfer_learning_stage(model, architecture, "partial")
            optimizer = torch.optim.AdamW(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=config.learning_rate * 0.2,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - epoch + 1))
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        validation = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        score = validation.get(config.metric)
        if score is None:
            score = -validation["loss"]
        previous_best = stopper.best
        should_stop = stopper.update(float(score))
        if previous_best != stopper.best:
            torch.save(
                {
                    "architecture": architecture,
                    "model_state_dict": model.state_dict(),
                    "best_validation_metric": stopper.best,
                    "training_config": asdict(config),
                },
                output,
            )
        if should_stop:
            break

    output.with_suffix(".history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return history
