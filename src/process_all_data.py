# src/process_all_data.py

from pathlib import Path
import cv2
import random
import csv
from tqdm import tqdm
import os
import sys

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from .paths import DATA_DIR

# 🚨 [Worker 모듈 임포트]
# 얼굴 크롭(2단계)과 인덱싱(3단계)은 외부 모듈의 함수를 호출합니다.
try:
    from .extract_faces import run_face_extraction as run_face_extraction_runner
    from .build_index import run_indexing as run_indexing_runner
except ImportError as e:
    print(f"[ERROR] Failed to import worker modules: {e}")
    print("Check if 'extract_faces.py' and 'build_index.py' exist in src/ folder.")
    sys.exit(1)


# 상수 정의
FRAME_PER_VIDEO = 5
VIDEO_LIMIT_PER_DATASET = 0
random.seed(42)


# ============================================
# I. 데이터셋 정의 (Manager 역할)
# ============================================

DATASETS = [
    {
        "name": "FaceForensics++ (Training)",
        "input_root": DATA_DIR / "raw" / "ffpp_c23" / "FaceForensics++_C23",
        "frame_output_dir": DATA_DIR / "interim" / "frames_ffpp",
        "frame_csv": DATA_DIR / "processed" / "indices" / "frames_ffpp.csv",
        "face_output_dir": DATA_DIR / "processed" / "faces_ffpp",
        "face_csv": DATA_DIR / "processed" / "indices" / "faces_ffpp.csv",
        "index_prefix": DATA_DIR / "processed" / "indices" / "ffpp",
    },
    {
        "name": "Celeb-DF v2 (Cross-Eval)",
        "input_root": DATA_DIR / "raw" / "Celeb_DF" / "celebdf_v2",
        "frame_output_dir": DATA_DIR / "interim" / "frames_celebdf",
        "frame_csv": DATA_DIR / "processed" / "indices" / "frames_celebdf.csv",
        "face_output_dir": DATA_DIR / "processed" / "faces_celebdf",
        "face_csv": DATA_DIR / "processed" / "indices" / "faces_celebdf.csv",
        "index_prefix": DATA_DIR / "processed" / "indices" / "celebdf",
    },
]


# ============================================
# II. 1단계: 프레임 추출 로직 (Worker 역할)
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

    # 카테고리명 추출 (예: Celeb-real, Deepfakes)
    category_name = video_path.parent.name

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        filename = f"{category_name}_{video_path.stem}_f{idx:05d}.jpg"
        out_path = out_dir / filename

        cv2.imwrite(str(out_path), frame)
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def run_extraction_pipeline(dataset):
    """
    프레임 추출 파이프라인 (Real/Fake 밸런싱 로직 포함)
    """
    RAW_DATA_ROOT = dataset["input_root"]
    FRAMES_DIR = dataset["frame_output_dir"]
    INDEX_CSV = dataset["frame_csv"]

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Searching in absolute path: {RAW_DATA_ROOT.resolve()}")

    # 모든 mp4 파일 찾기
    all_videos = list(RAW_DATA_ROOT.rglob("*.mp4"))

    if not all_videos:
        print(f"[ERROR] No MP4 files found in {RAW_DATA_ROOT}. Check path structure.")
        return

    # ====================================================
    # 🚨 [수정] Real/Fake 밸런싱 로직 추가
    # ====================================================
    real_videos = []
    fake_videos = []

    for v in all_videos:
        path_str = str(v).lower()
        # Real 폴더명 키워드 확인 (FF++: original, Celeb-DF: celeb-real, youtube-real)
        if (
            "original" in path_str
            or "celeb-real" in path_str
            or "youtube-real" in path_str
        ):
            real_videos.append(v)
        else:
            fake_videos.append(v)

    print(
        f"[INFO] Found total: {len(all_videos)} (Real: {len(real_videos)}, Fake: {len(fake_videos)})"
    )

    # 제한 적용 (각각 절반씩 가져오기)
    final_videos = []
    if VIDEO_LIMIT_PER_DATASET > 0:
        half_limit = VIDEO_LIMIT_PER_DATASET // 2

        # 섞어서 랜덤하게 뽑기 위해 shuffle
        random.shuffle(real_videos)
        random.shuffle(fake_videos)

        # 데이터가 부족하면 있는 만큼만 가져오기
        selected_reals = real_videos[:half_limit]
        selected_fakes = fake_videos[:half_limit]

        final_videos = selected_reals + selected_fakes
        print(
            f"[WARN] Balanced Limiting: Used {len(selected_reals)} Real + {len(selected_fakes)} Fake videos."
        )
    else:
        final_videos = all_videos
    # ====================================================

    with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_path"])

        for v in tqdm(final_videos, desc=f"Extracting {dataset['name']}"):
            saved_frames = extract_frames_from_video(v, FRAMES_DIR)
            for fp in saved_frames:
                rel_video_path = v.relative_to(PROJECT_ROOT)
                rel_frame_path = fp.relative_to(PROJECT_ROOT)
                writer.writerow([str(rel_video_path), str(rel_frame_path)])

    print(f"[DONE] frames saved to: {FRAMES_DIR.name}")
    print(f"[DONE] index csv: {INDEX_CSV.name}")


# ============================================
# III. 메인 실행 루프
# ============================================


def main():
    print("\n[MASTER PROCESS] Starting Dual-Dataset Preprocessing Pipeline...")

    for dataset in DATASETS:
        print("\n" + "=" * 60)
        print(f"🚀 Processing Dataset: {dataset['name']}")
        print("=" * 60)

        # ---------------------------------------------
        # 1. 프레임 추출 (Frame Extraction)
        # ---------------------------------------------
        print("\n[STEP 1/3] Starting Frame Extraction (Balanced)...")
        run_extraction_pipeline(dataset)

        # ---------------------------------------------
        # 2. 얼굴 크롭 및 정렬 (Face Cropping)
        # ---------------------------------------------
        print("\n[STEP 2/3] Starting Face Cropping...")
        try:
            run_face_extraction_runner(
                dataset["frame_csv"], dataset["face_output_dir"], dataset["face_csv"]
            )
            print(f"[DONE] Step 2 Complete.")
        except Exception as e:
            print(f"[FATAL ERROR] Step 2 failed: {e}")
            continue

        # ---------------------------------------------
        # 3. 인덱스 파일 생성 (Indexing & Split)
        # ---------------------------------------------
        print("\n[STEP 3/3] Starting Train/Val/Test Indexing...")
        try:
            run_indexing_runner(dataset["face_csv"], dataset["index_prefix"])
            print(f"[DONE] Step 3 Complete.")
        except Exception as e:
            print(f"[FATAL ERROR] Step 3 failed: {e}")
            continue

    print("\n\n🎉 [MASTER PROCESS] ALL DATA PREPARATION COMPLETE!")


if __name__ == "__main__":
    main()
