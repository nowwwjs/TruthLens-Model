from pathlib import Path

from PIL import Image

from truthlens.data import (
    ManifestRecord,
    assert_no_group_leakage,
    split_records_by_group,
    validate_images,
)


def _records():
    records = []
    for source in ("celebdf", "dfdc"):
        for label in (0, 1):
            for group_index in range(5):
                for frame_index in range(2):
                    records.append(
                        ManifestRecord(
                            image_path=f"{source}/{label}/video_{group_index}_frame{frame_index}.jpg",
                            label=label,
                            source_dataset=source,
                            group_id=f"video_{label}_{group_index}",
                        )
                    )
    return records


def test_group_split_is_reproducible_and_leak_free():
    first = split_records_by_group(_records(), seed=42)
    second = split_records_by_group(_records(), seed=42)
    assert first == second
    assert_no_group_leakage(first)
    assert all(first[name] for name in ("train", "val", "test"))


def test_corrupt_and_missing_images_are_excluded(tmp_path: Path):
    valid_path = tmp_path / "valid.png"
    corrupt_path = tmp_path / "corrupt.png"
    Image.new("RGB", (8, 8), "white").save(valid_path)
    corrupt_path.write_bytes(b"not an image")
    records = [
        ManifestRecord(str(valid_path), 0, "test", "valid"),
        ManifestRecord(str(corrupt_path), 1, "test", "corrupt"),
        ManifestRecord(str(tmp_path / "missing.png"), 1, "test", "missing"),
    ]
    valid, errors = validate_images(records)
    assert valid == [records[0]]
    assert errors == {"corrupt": 1, "missing": 1}
