# src/build_ffpp_index.py

from pathlib import Path
import csv
import random

from .paths import PROJECT_ROOT


# 경로 설정
# data 폴더
DATA_DIR = PROJECT_ROOT / "data"

# faces_ffpp.csv 경로
FACES_CSV = DATA_DIR / "processed" / "indices" / "faces_ffpp.csv"

# 출력 CSV 경로들
OUT_ALL = DATA_DIR / "processed" / "indices" / "ffpp_all.csv"
OUT_TRAIN = DATA_DIR / "processed" / "indices" / "ffpp_train.csv"
OUT_VAL = DATA_DIR / "processed" / "indices" / "ffpp_val.csv"
OUT_TEST = DATA_DIR / "processed" / "indices" / "ffpp_test.csv"

# 랜덤 시드 고정
SEED = 42
random.seed(SEED)


# ============================================
# 유틸 함수
# ============================================

def get_label_from_path(video_path: str) -> int:
    # FF++ 폴더 구조에서 video_path 문자열을 보고 라벨 결정.
    # - 'original'이 경로에 포함 → 0 (real)
    # - 그 외 → 1 (fake)
    p = Path(video_path)
    parts = [s.lower() for s in p.parts]

    return 0 if "original" in parts else 1


def group_by_video(faces_rows):
    
    # 같은 video_path에 속한 얼굴들을 묶어줌.
    """ 반환 형태:
    {
        "path/to/video": {
            "label": 0 또는 1,
            "rows": [ ... face rows ... ]
        },
        ...
    }
    """
    by_video = {}

    for row in faces_rows:
        video_path = row["video_path"]

        if video_path not in by_video:
            label = get_label_from_path(video_path)
            by_video[video_path] = {
                "label": label,
                "rows": []
            }

        by_video[video_path]["rows"].append(row)

    return by_video


def split_videos(video_paths, train_ratio=0.7, val_ratio=0.15):
    # 영상 단위로 train / val / test 분할

    video_paths = list(video_paths)
    random.shuffle(video_paths)

    n = len(video_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_v = video_paths[:n_train]
    val_v = video_paths[n_train:n_train + n_val]
    test_v = video_paths[n_train + n_val:]

    return set(train_v), set(val_v), set(test_v)


def write_csv(path: Path, rows):
    # 주어진 rows 리스트를 path 위치의 CSV 파일로 저장
    
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================
# 메인 처리 로직
# ============================================

def main():
    # faces_ffpp.csv 읽기
    faces_rows = []
    with FACES_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            faces_rows.append(row)

    print(f"[INFO] Loaded {len(faces_rows)} face rows")

    # 영상단위로 그룹화
    by_video = group_by_video(faces_rows)
    video_paths = list(by_video.keys())
    print(f"[INFO] Unique videos with faces: {len(video_paths)}")

    # Train / Val / Test split
    train_v, val_v, test_v = split_videos(video_paths)

    print(f"[INFO] Train videos: {len(train_v)}")
    print(f"[INFO] Val videos:   {len(val_v)}")
    print(f"[INFO] Test videos:  {len(test_v)}")

    # 각 row에 라벨 및 split 정보 추가
    all_rows = []
    train_rows, val_rows, test_rows = [], [], []

    for video_path, info in by_video.items():
        label = info["label"]

        if video_path in train_v:
            split = "train"
            target_list = train_rows
        elif video_path in val_v:
            split = "val"
            target_list = val_rows
        else:
            split = "test"
            target_list = test_rows

        # 하나의 영상에 속한 얼굴 row들 추가
        for row in info["rows"]:
            new_row = dict(row)
            new_row["label"] = label
            new_row["split"] = split

            all_rows.append(new_row)
            target_list.append(new_row)

    # CSV 저장
    write_csv(OUT_ALL, all_rows)
    write_csv(OUT_TRAIN, train_rows)
    write_csv(OUT_VAL, val_rows)
    write_csv(OUT_TEST, test_rows)

    print("[DONE] Saved:")
    print("  ", OUT_ALL)
    print("  ", OUT_TRAIN)
    print("  ", OUT_VAL)
    print("  ", OUT_TEST)


if __name__ == "__main__":
    main()
