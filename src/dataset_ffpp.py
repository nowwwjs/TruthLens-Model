# src/dataset_ffpp.py

from pathlib import Path
import csv
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .paths import PROJECT_ROOT


class FFPPFaceDataset(Dataset):
    """
    FF++ 얼굴 이미지용 Dataset 클래스

    CSV 파일 형식 (헤더):
    - video_path
    - frame_path
    - face_path
    - label
    - split
    """

    def __init__(
        self,
        csv_path: str | Path,
        train: bool = True,
    ):
        # CSV 파일 경로
        self.csv_path = Path(csv_path)
        self.train = train

        # (face_path, label) 튜플을 담는 리스트
        self.samples: List[Tuple[Path, int]] = []

        # CSV 로드
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_face_path = Path(row["face_path"])

                # face_path가 절대 경로가 아니면 프로젝트 루트를 기준으로 보정
                if not raw_face_path.is_absolute():
                    face_path = PROJECT_ROOT / raw_face_path
                else:
                    face_path = raw_face_path

                label = int(row["label"])
                self.samples.append((face_path, label))

        # 이미지 변환(augmentation + normalize)
        if train:
            # 학습용: 간단한 augmentation 포함
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])
        else:
            # 검증/테스트용: deterministic 변환만 적용
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])

    def __len__(self):
        # 전체 샘플 개수 반환
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        하나의 샘플 반환
        - x: 전처리된 이미지 텐서
        - y: 라벨 (LongTensor)
        """
        face_path, label = self.samples[idx]

        # 이미지 로드 및 RGB 변환
        img = Image.open(face_path).convert("RGB")

        # 변환 적용
        x = self.transform(img)

        # 라벨 텐서 변환
        y = torch.tensor(label, dtype=torch.long)

        return x, y
