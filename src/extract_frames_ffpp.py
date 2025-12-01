# src/extract_frames_ffpp.py

from pathlib import Path
import cv2
import random
import csv
from tqdm import tqdm

from .paths import PROJECT_ROOT


# 경로 설정
# data 루트 (TruthLens-Model/data)
DATA_ROOT = PROJECT_ROOT / "data"

# 원본 FF++ 데이터 루트
RAW_FFPP_ROOT = DATA_ROOT / "raw" / "ffpp_c23" / "FaceForensics++_C23"

# 추출된 프레임이 저장될 디렉토리
FRAMES_DIR = DATA_ROOT / "interim" / "frames_ffpp"

# 프레임 인덱스 CSV 경로
INDEX_CSV = DATA_ROOT / "processed" / "indices" / "frames_ffpp.csv"

# 영상당 뽑을 프레임 개수
FRAME_PER_VIDEO = 5

# 랜덤 시드 고정
random.seed(42)


# ============================================
# 유틸 함수
# ============================================

def sample_indices(num_frames: int, k: int):
    """
    전체 프레임 수(num_frames) 중에서 k개를 균일하게 샘플링.
    프레임 수가 k 이하인 경우는 전체 프레임을 사용.
    """
    if num_frames <= k:
        return list(range(num_frames))
    return sorted(random.sample(range(num_frames), k))


def extract_frames_from_video(video_path: Path, out_dir: Path):
    """
    단일 영상 파일에서 일부 프레임을 추출하여 이미지로 저장.

    - video_path: 원본 영상 경로
    - out_dir: 프레임 이미지가 저장될 디렉토리
    - 반환: 저장된 프레임 이미지 경로 리스트 (List[Path])
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        print(f"[WARN] no frames: {video_path}")
        cap.release()
        return []

    indices = sample_indices(total, FRAME_PER_VIDEO)
    saved_paths = []

    # 상위 폴더 이름(카테고리)을 파일명에 포함
    category = video_path.parent.name

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        # 예: original_000_f00000.jpg 형태
        filename = f"{category}_{video_path.stem}_f{idx:05d}.jpg"
        out_path = out_dir / filename

        cv2.imwrite(str(out_path), frame)
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


# ============================================
# 메인 처리 로직
# ============================================

def main():
    # 출력 폴더 및 인덱스 CSV 디렉토리 생성
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)

    # FF++ C23 폴더 아래의 모든 mp4 영상 탐색
    videos = list(RAW_FFPP_ROOT.rglob("*.mp4"))
    print(f"[INFO] found {len(videos)} videos under {RAW_FFPP_ROOT}")

    with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # CSV 헤더: 원본 영상 경로, 프레임 이미지 경로
        writer.writerow(["video_path", "frame_path"])

        for v in tqdm(videos):
            # v: 원본 영상의 절대 경로 or PROJECT_ROOT 기준 경로
            saved_frames = extract_frames_from_video(v, FRAMES_DIR)

            for fp in saved_frames:
                # CSV에는 프로젝트 루트 기준의 상대 경로로 저장
                # (레포를 이동해도 경로가 깨지지 않도록)
                rel_video_path = v.relative_to(PROJECT_ROOT)
                rel_frame_path = fp.relative_to(PROJECT_ROOT)

                writer.writerow([str(rel_video_path), str(rel_frame_path)])

    print("[DONE] frames saved to:", FRAMES_DIR)
    print("[DONE] index csv:", INDEX_CSV)


if __name__ == "__main__":
    main()
