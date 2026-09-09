"""Manifest creation, leakage-safe splitting, and image validation."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    label: int
    source_dataset: str
    group_id: str
    split: str = ""

    def __post_init__(self) -> None:
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 (REAL) or 1 (FAKE)")
        if not self.source_dataset.strip() or not self.group_id.strip():
            raise ValueError("source_dataset and group_id are required")


def infer_group_id(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"(?:_frame|_f)\d+$", "", stem, flags=re.IGNORECASE)
    return stem


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        ManifestRecord(
            image_path=row["image_path"],
            label=int(row["label"]),
            source_dataset=row["source_dataset"],
            group_id=row.get("group_id") or infer_group_id(row["image_path"]),
            split=row.get("split", ""),
        )
        for row in rows
    ]


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(ManifestRecord("x", 0, "x", "x"))))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def combine_manifests(*record_sets: Sequence[ManifestRecord]) -> list[ManifestRecord]:
    combined = [record for records in record_sets for record in records]
    seen: set[tuple[str, str]] = set()
    for record in combined:
        key = (record.source_dataset, record.image_path)
        if key in seen:
            raise ValueError(f"Duplicate manifest row: {key}")
        seen.add(key)
    return combined


def _allocation(group_count: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if group_count < 3:
        return group_count, 0
    train_count = max(1, int(group_count * train_ratio))
    val_count = max(1, int(group_count * val_ratio))
    if train_count + val_count >= group_count:
        train_count = max(1, group_count - 2)
        val_count = 1
    return train_count, val_count


def split_records_by_group(
    records: Sequence[ManifestRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[ManifestRecord]]:
    """Stratify by dataset and label, then split complete video groups."""

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must leave a non-zero test partition")

    groups: dict[tuple[str, str], list[ManifestRecord]] = defaultdict(list)
    for record in records:
        groups[(record.source_dataset, record.group_id)].append(record)
    for key, group in groups.items():
        if len({record.label for record in group}) != 1:
            raise ValueError(f"Mixed labels inside group: {key}")

    strata: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for key, group in groups.items():
        strata[(key[0], group[0].label)].append(key)

    rng = random.Random(seed)
    assignment: dict[tuple[str, str], str] = {}
    for keys in strata.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        train_count, val_count = _allocation(len(keys), train_ratio, val_ratio)
        for index, key in enumerate(keys):
            assignment[key] = "train" if index < train_count else "val" if index < train_count + val_count else "test"

    result = {"train": [], "val": [], "test": []}
    for key, group in groups.items():
        split = assignment[key]
        result[split].extend(
            ManifestRecord(**{**asdict(record), "split": split}) for record in group
        )
    assert_no_group_leakage(result)
    return result


def assert_no_group_leakage(splits: dict[str, Sequence[ManifestRecord]]) -> None:
    observed: dict[tuple[str, str], str] = {}
    for split_name, records in splits.items():
        for record in records:
            key = (record.source_dataset, record.group_id)
            previous = observed.setdefault(key, split_name)
            if previous != split_name:
                raise ValueError(f"Group leakage detected for {key}: {previous}/{split_name}")


def validate_images(records: Sequence[ManifestRecord]) -> tuple[list[ManifestRecord], dict[str, int]]:
    from PIL import Image

    valid: list[ManifestRecord] = []
    errors: Counter[str] = Counter()
    for record in records:
        try:
            with Image.open(record.image_path) as image:
                image.verify()
            valid.append(record)
        except FileNotFoundError:
            errors["missing"] += 1
        except Exception:
            errors["corrupt"] += 1
    return valid, dict(errors)


def manifest_sha256(records: Sequence[ManifestRecord]) -> str:
    canonical = "\n".join(
        f"{record.source_dataset}|{record.group_id}|{record.label}|{record.image_path}|{record.split}"
        for record in sorted(records, key=lambda item: (item.source_dataset, item.group_id, item.image_path))
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_transforms(train: bool, image_size: int = 224):
    from torchvision import transforms

    operations = [transforms.Resize((image_size, image_size))]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if train:
        operations.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)))
    return transforms.Compose(operations)


class ManifestDataset:
    """Dataset that fails on bad inputs instead of silently training on black images."""

    def __init__(self, records: Sequence[ManifestRecord], train: bool = True):
        self.records = list(records)
        self.transform = build_transforms(train=train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.label
