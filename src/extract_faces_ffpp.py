# src/extract_faces_ffpp.py

from pathlib import Path
import csv

import cv2
from tqdm import tqdm

from .paths import PROJECT_ROOT


# 경로 설정
# data 디렉토리
DATA_DIR = PROJECT_ROOT / "data"

# 프레임 정보 CSV
FRAMES_CSV = DATA_DIR / "processed" / "indices" / "frames_ffpp.csv"

# 얼굴 이미지가 저장될 디렉토리
FACES_DIR = DATA_DIR / "processed" / "faces_ffpp"

# 얼굴 정보 CSV
FACES_CSV = DATA_DIR / "processed" / "indices" / "faces_ffpp.csv"

# 얼굴 crop 후 resize 크기
FACE_SIZE = 224


# ============================================
# 얼굴 검출 관련 함수
# ============================================

def load_face_detector():
    # OpenCV Haar Cascade 로 얼굴 검출기 로드

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(f"Failed to load face cascade from {cascade_path}")

    return face_cascade


def detect_main_face(gray, face_cascade):
    """
    가장 큰 얼굴 하나만 반환
    - 입력: gray (그레이스케일 이미지), face_cascade (Haar 모델)
    - 반환: (x, y, w, h) 또는 None
    """
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    # 영역(너비 * 높이)이 가장 큰 얼굴 하나 선택
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return faces[0]


# ============================================
# 메인 처리 로직
# ============================================

def main():
    # 얼굴 검출기 로드
    face_cascade = load_face_detector()

    # 출력 디렉토리 / CSV 경로 생성
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    FACES_CSV.parent.mkdir(parents=True, exist_ok=True)

    # frames_ffpp.csv를 읽어서 각 프레임에서 얼굴 검출 후 저장
    with FRAMES_CSV.open("r", encoding="utf-8") as f_in, \
            FACES_CSV.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(
            f_out,
            fieldnames=["video_path", "frame_path", "face_path"]
        )
        writer.writeheader()

        for row in tqdm(reader):
            raw_frame_path = Path(row["frame_path"])

            # frame_path가 상대경로일 수도 있으므로 PROJECT_ROOT 기준으로 보정
            if not raw_frame_path.is_absolute():
                frame_path = PROJECT_ROOT / raw_frame_path
            else:
                frame_path = raw_frame_path

            # 프레임 이미지 로드
            img = cv2.imread(str(frame_path))
            if img is None:
                # 파일이 없거나 깨진 경우 스킵
                continue

            # 그레이스케일 변환 후 얼굴 검출
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bbox = detect_main_face(gray, face_cascade)
            if bbox is None:
                # 얼굴이 검출되지 않은 경우 스킵
                continue

            x, y, w, h = bbox
            face = img[y:y + h, x:x + w]

            # 잘못된 영역이면 스킵
            if face.size == 0:
                continue

            # 얼굴 영역을 고정 크기로 resize
            face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))

            # 저장 경로 설정 (프레임 파일명 기반)
            out_path = FACES_DIR / f"{frame_path.stem}_face.jpg"
            cv2.imwrite(str(out_path), face)

            # CSV에는 문자열 경로로 기록
            writer.writerow({
                "video_path": row["video_path"],
                "frame_path": row["frame_path"],
                "face_path": str(out_path),
            })

    print("\n[DONE] Faces saved to:", FACES_DIR)
    print("[DONE] CSV created:", FACES_CSV)


if __name__ == "__main__":
    main()
