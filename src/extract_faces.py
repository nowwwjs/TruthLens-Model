# src/extract_faces.py

from pathlib import Path
import csv
import cv2
from tqdm import tqdm
import argparse  # 🚨 [추가] 명령줄 인자를 받기 위해 임포트

from .paths import PROJECT_ROOT


# 경로 설정 (하드코딩된 경로를 모두 제거하고 상수로 유지할 것만 남깁니다.)
FACE_SIZE = 224


def load_face_detector():
    """OpenCV Haar Cascade 로 얼굴 검출기 로드"""
    # OpenCV 설치 경로에서 XML 파일 위치를 찾습니다.
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        # 이 에러가 발생하면 OpenCV가 제대로 설치되지 않았거나 XML 파일 경로 문제입니다.
        raise RuntimeError(f"Failed to load face cascade from {cascade_path}")

    return face_cascade


def detect_main_face(gray, face_cascade):
    """
    가장 큰 얼굴 하나만 검출하여 반환
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


def run_face_extraction(
    input_csv_path, output_dir, output_csv_path
):  # 🚨 [새 함수] main 로직을 이리로 옮김

    # 🚨 [수정] 경로 객체로 변환
    FRAMES_CSV = Path(input_csv_path)
    FACES_DIR = Path(output_dir)
    FACES_CSV = Path(output_csv_path)

    # 얼굴 검출기 로드
    face_cascade = load_face_detector()

    # 출력 디렉토리 / CSV 경로 생성
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    FACES_CSV.parent.mkdir(parents=True, exist_ok=True)

    # frames_ffpp.csv 대신 인자로 받은 FRAMES_CSV를 읽습니다.
    with FRAMES_CSV.open("r", encoding="utf-8") as f_in, FACES_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(
            f_out, fieldnames=["video_path", "frame_path", "face_path"]
        )
        writer.writeheader()

        for row in tqdm(
            reader, desc=f"Cropping faces for {FRAMES_CSV.name}"
        ):  # tqdm에 설명 추가

            # (프레임 로드 로직 유지)
            raw_frame_path = Path(row["frame_path"])

            if not raw_frame_path.is_absolute():
                frame_path = PROJECT_ROOT / raw_frame_path
            else:
                frame_path = raw_frame_path

            img = cv2.imread(str(frame_path))

            if img is None:
                continue

            # 그레이스케일 변환 후 얼굴 검출
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bbox = detect_main_face(gray, face_cascade)

            if bbox is None:
                continue

            # ... (크롭 및 리사이즈 로직 유지) ...
            x, y, w, h = bbox
            face = img[y : y + h, x : x + w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))

            # 저장 경로 설정 (프레임 파일명 기반)
            out_path = FACES_DIR / f"{frame_path.stem}_face.jpg"
            cv2.imwrite(str(out_path), face)

            # CSV에는 문자열 경로로 기록
            writer.writerow(
                {
                    "video_path": row["video_path"],
                    "frame_path": row["frame_path"],
                    # 🚨 [수정] face_path를 상대경로로 저장해야 프로젝트 이동 시 경로가 깨지지 않습니다.
                    "face_path": str(out_path.relative_to(PROJECT_ROOT)),
                }
            )


if __name__ == "__main__":
    # 🚨 [추가] 명령줄 인자를 처리하는 로직
    parser = argparse.ArgumentParser(
        description="Detects and crops faces from frame images based on an index CSV."
    )

    # 인자 정의
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input CSV containing frame paths (e.g., data/processed/indices/frames_ffpp.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the final cropped face images (e.g., data/processed/faces_ffpp).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to the output index CSV linking frames to faces.",
    )

    args = parser.parse_args()

    # run_face_extraction 함수 호출
    run_face_extraction(args.input_csv, args.output_dir, args.output_csv)
