"""Build a combined, group-safe Celeb-DF/DFDC image manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from truthlens.data import (
    ManifestRecord,
    infer_group_id,
    manifest_sha256,
    split_records_by_group,
    validate_images,
    write_manifest,
)


KNOWN_LAYOUT = {
    "celebdf": {
        0: "Celeb_real_face_only",
        1: "Celeb_fake_face_only",
    },
    "dfdc": {
        0: "DFDC_REAL_Face_only_data",
        1: "DFDC_FAKE_Face_only_data",
    },
}


def discover_records(dataset_root: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for source_dataset, labels in KNOWN_LAYOUT.items():
        for label, folder_name in labels.items():
            folder = dataset_root / folder_name
            if not folder.exists():
                continue
            for image_path in sorted(folder.rglob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                relative = image_path.relative_to(dataset_root).as_posix()
                records.append(
                    ManifestRecord(
                        image_path=str(image_path),
                        label=label,
                        source_dataset=source_dataset,
                        group_id=infer_group_id(relative),
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe TruthLens manifests")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-image-validation", action="store_true")
    args = parser.parse_args()

    records = discover_records(args.dataset_root)
    if not records:
        raise SystemExit(f"No supported images found under {args.dataset_root}")
    errors: dict[str, int] = {}
    if not args.skip_image_validation:
        records, errors = validate_images(records)
    splits = split_records_by_group(records, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[ManifestRecord] = []
    for split_name, split_records in splits.items():
        write_manifest(split_records, args.output_dir / f"combined_{split_name}.csv")
        all_records.extend(split_records)
    write_manifest(all_records, args.output_dir / "combined_manifest.csv")
    print(f"records={len(all_records)} sha256={manifest_sha256(all_records)}")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    if errors:
        print(f"excluded={errors}")


if __name__ == "__main__":
    main()
