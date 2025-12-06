# src/dataset.py

from pathlib import Path
import csv
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys

# 모듈 경로 문제 방지를 위한 로직
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# paths.py에서 가져오되, 실패하면 현재 경로 기준으로 설정
try:
    from .paths import PROJECT_ROOT
except ImportError:
    pass


class FaceDataset(Dataset):
    """
    범용 얼굴 이미지 데이터셋 클래스 (FF++, Celeb-DF 등 모두 지원)
    CSV 파일에 정의된 경로를 통해 이미지와 레이블을 로드합니다.
    """

    def __init__(
        self,
        csv_path: str | Path,
        train: bool = True,
        load_mask: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.train = train
        self.load_mask = load_mask

        self.samples: List[Tuple[Path, int]] = []

        # 1. CSV 로드
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_face_path = Path(row["face_path"])

                # 절대 경로로 변환
                if not raw_face_path.is_absolute():
                    face_path = PROJECT_ROOT / raw_face_path
                else:
                    face_path = raw_face_path

                label = int(row["label"])
                self.samples.append((face_path, label))

        # 2. 이미지 변환(Transform) 정의
        if train:
            # 학습용: Augmentation 적용
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    # ImageNet 표준 정규화
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            # 검증/테스트용: 기본 변환만 적용
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        face_path, label = self.samples[idx]

        # 이미지 로드
        try:
            img = Image.open(face_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {face_path}: {e}")
            return self.__getitem__(0)  # 에러 시 0번 인덱스로 대체

        # 변환 적용
        x = self.transform(img)

        # 라벨 텐서 변환
        y = torch.tensor(label, dtype=torch.long)

        return x, y
