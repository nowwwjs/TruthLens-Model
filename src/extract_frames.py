# src/extract_frames.py 

from pathlib import Path
import cv2
import random
import csv
from tqdm import tqdm
import os
import argparse

# --- 경로 설정 및 프로젝트 루트 (모든 파일이 이를 기준으로 함) ---
# NOTE: 이 코드는 프로젝트 루트 폴더에 있다고 가정합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# 영상당 뽑을 프레임 개수
FRAME_PER_VIDEO = 5
random.seed(42)

# ============================================
# I. 데이터셋 정의 (Manager 역할)
# ============================================

DATASETS = [
    {
        "name": "FaceForensics++",
        # 🚨 FF++ 원본 영상 경로
        "input_root": DATA_DIR / "raw" / "ffpp_c23" / "FaceForensics++_C23",
        "output_dir": DATA_DIR / "interim" / "frames_ffpp",
        "output_csv": DATA_DIR / "processed" / "indices" / "frames_ffpp.csv",
    },
    {
        "name": "Celeb-DF v2",
        # 🚨 Celeb-DF 원본 영상 경로
        "input_root": DATA_DIR / "raw" / "Celeb_DF" / "celebdf_v2",
        "output_dir": DATA_DIR / "interim" / "frames_celebdf",
        "output_csv": DATA_DIR / "processed" / "indices" / "frames_celebdf.csv",
    },
]

# ============================================
# II. 프레임 추출 로직 (Worker 역할)
# ============================================


def sample_indices(num_frames: int, k: int):
    """전체 프레임 중 k개를 균일하게 샘플링."""
    if num_frames <= k:
        return list(range(num_frames))
    return sorted(random.sample(range(num_frames), k))


def extract_frames_from_video(video_path: Path, out_dir: Path):
    """단일 영상 파일에서 일부 프레임을 추출하여 이미지로 저장."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return []

    indices = sample_indices(total, FRAME_PER_VIDEO)
    saved_paths = []
    category = video_path.parent.name  # original, Deepfakes 등 카테고리 추출

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        filename = f"{category}_{video_path.stem}_f{idx:05d}.jpg"
        out_path = out_dir / filename

        cv2.imwrite(str(out_path), frame)
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


# ============================================
# III. 메인 실행 루프 (단일 파일 실행)
# ============================================


def run_extraction_pipeline(dataset):
    """주어진 데이터셋 정의에 따라 프레임 추출 및 인덱스 생성을 실행."""

    RAW_DATA_ROOT = dataset["input_root"]
    FRAMES_DIR = dataset["output_dir"]
    INDEX_CSV = dataset["output_csv"]

    # 출력 폴더 및 인덱스 CSV 디렉토리 생성
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)

    videos = list(RAW_DATA_ROOT.rglob("*.mp4"))
    print(f"[INFO] found {len(videos)} videos under {RAW_DATA_ROOT.name}")

    with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_path"])

        for v in tqdm(videos, desc=f"Extracting {dataset['name']}"):
            saved_frames = extract_frames_from_video(v, FRAMES_DIR)

            for fp in saved_frames:
                rel_video_path = v.relative_to(PROJECT_ROOT)
                rel_frame_path = fp.relative_to(PROJECT_ROOT)
                writer.writerow([str(rel_video_path), str(rel_frame_path)])

    print(f"[DONE] frames saved to: {FRAMES_DIR.name}")
    print(f"[DONE] index csv: {INDEX_CSV.name}")


def main():
    print("\n[MASTER PROCESS] Starting Dual-Dataset Frame Extraction...")

    for dataset in DATASETS:
        print("\n" + "=" * 50)
        print(f"🚀 Processing Dataset: {dataset['name']}")
        print("=" * 50)

        run_extraction_pipeline(dataset)

    print("\n[MASTER PROCESS] All frame extractions complete.")


if __name__ == "__main__":
    main()
