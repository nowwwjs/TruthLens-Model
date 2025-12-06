# src/build_index.py

from pathlib import Path
import csv
import random
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 분할을 위해 설치 필요

from .paths import PROJECT_ROOT

# 랜덤 시드 고정 (분할 결과의 재현성을 위해)
SEED = 42
random.seed(SEED)

# ============================================
# 유틸 함수 1: 라벨 추출 (FF++ 및 Celeb-DF 모두 커버)
# ============================================


def get_label_from_path(video_path: str) -> int:
    """
    영상 경로를 분석하여 딥페이크 레이블 (0: Real, 1: Fake)을 결정합니다.
    (FF++의 'original' 폴더명 또는 Celeb-DF의 'Celeb-real' 폴더명으로 판단)
    """
    p = Path(video_path)
    parts = [s.lower() for s in p.parts]

    # 'original' (FF++) 또는 'celeb-real' (Celeb-DF)이 포함되면 Real (0)
    if "original" in parts or "celeb-real" in parts or "youtube-real" in parts:
        return 0  # Real

    # 'Deepfakes', 'FaceSwap', 'Celeb-synthesis' 등이 포함되면 Fake (1)
    return 1  # Fake


def group_by_video(faces_rows):
    """얼굴 행들을 영상 단위로 그룹화합니다."""
    by_video = {}

    for row in faces_rows:
        video_path = row["video_path"]

        if video_path not in by_video:
            # get_label_from_path 함수를 사용해 라벨을 결정
            label = get_label_from_path(video_path)
            by_video[video_path] = {"label": label, "rows": []}

        by_video[video_path]["rows"].append(row)

    return by_video


# ============================================
# 유틸 함수 2: 영상 단위 분할 (80:10:10)
# ============================================


def split_videos(video_paths, train_ratio=0.8, val_ratio=0.1):
    """영상 단위로 train / val / test 분할 (Stratify 없이 단순 랜덤 분할)"""

    video_paths = list(video_paths)

    # scikit-learn의 train_test_split을 사용하여 분할
    # 1차 분리: Test set 분리
    train_val_v, test_v = train_test_split(
        video_paths, test_size=val_ratio, random_state=SEED
    )

    # 2차 분리: Train set과 Validation set 분리
    val_size = val_ratio / (train_ratio + val_ratio)  # 예: 0.1 / 0.9 = 0.111...
    train_v, val_v = train_test_split(
        train_val_v, test_size=val_size, random_state=SEED
    )

    return set(train_v), set(val_v), set(test_v)


def write_csv(path: Path, rows, fieldnames):
    """주어진 rows 리스트를 path 위치의 CSV 파일로 저장"""

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================
# 메인 처리 로직
# ============================================


def run_indexing(input_csv, output_prefix):

    FACES_CSV = Path(input_csv)
    OUTPUT_PREFIX = Path(output_prefix)

    # 1. faces_csv 읽기 (이전에 extract_faces.py가 생성한 CSV)
    faces_rows = []
    with FACES_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Loading faces from {FACES_CSV.name}"):
            faces_rows.append(row)

    print(f"[INFO] Loaded {len(faces_rows)} face rows")

    # 2. 영상단위로 그룹화 및 분할
    by_video = group_by_video(faces_rows)
    video_paths = list(by_video.keys())

    train_v, val_v, test_v = split_videos(video_paths)

    # 3. 각 row에 라벨 및 split 정보 추가
    all_rows = []
    train_rows, val_rows, test_rows = [], [], []

    for video_path, info in by_video.items():
        # ... (split 정보 추가 및 row 할당 로직은 유지) ...

        if video_path in train_v:
            split = "train"
            target_list = train_rows
        elif video_path in val_v:
            split = "val"
            target_list = val_rows
        else:
            split = "test"
            target_list = test_rows

        for row in info["rows"]:
            new_row = dict(row)
            new_row["label"] = info[
                "label"
            ]  # 이미 get_label_from_path에서 정해진 라벨 사용
            new_row["split"] = split

            all_rows.append(new_row)
            target_list.append(new_row)

    # 4. CSV 저장
    fieldnames = (
        list(all_rows[0].keys())
        if all_rows
        else ["video_path", "frame_path", "face_path", "label", "split"]
    )

    write_csv(
        OUTPUT_PREFIX.parent / (OUTPUT_PREFIX.name + "_all.csv"), all_rows, fieldnames
    )
    write_csv(
        OUTPUT_PREFIX.parent / (OUTPUT_PREFIX.name + "_train.csv"),
        train_rows,
        fieldnames,
    )
    write_csv(
        OUTPUT_PREFIX.parent / (OUTPUT_PREFIX.name + "_val.csv"), val_rows, fieldnames
    )
    write_csv(
        OUTPUT_PREFIX.parent / (OUTPUT_PREFIX.name + "_test.csv"), test_rows, fieldnames
    )

    print(
        f"[DONE] Saved indices. Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}"
    )


def main():
    # 🚨 [추가] 명령줄 인자를 처리하는 로직
    parser = argparse.ArgumentParser(
        description="Creates train/val/test splits at the video level for any dataset."
    )
    # 인자 정의
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input CSV containing cropped face paths (e.g., data/processed/indices/faces_ffpp.csv)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Base path and prefix for output CSVs (e.g., data/processed/indices/ffpp)",
    )

    args = parser.parse_args()

    # run_indexing 함수 호출
    run_indexing(args.input_csv, args.output_prefix)


if __name__ == "__main__":
    main()
