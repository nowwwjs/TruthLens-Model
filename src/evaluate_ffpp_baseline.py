# src/evaluate_ffpp_baseline.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from .dataset_ffpp import FFPPFaceDataset
from .paths import WEIGHTS_DIR, PROJECT_ROOT

from pathlib import Path

# 데이터 및 모델 경로 설정
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = DATA_DIR / "processed" / "indices"
TEST_CSV = INDICES_DIR / "ffpp_test.csv"

MODEL_PATH = WEIGHTS_DIR / "ffpp_resnet18_baseline.pth"

# 모델 정의 및 체크포인트 로드
def get_model(num_classes=2, device="cpu"):
    """Load the baseline ResNet18 model with pretrained weights."""

    # 기본 ResNet18 불러오기
    model = models.resnet18(weights=None)

    # 마지막 FC 레이어를 이진 분류에 맞게 교체
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 학습된 가중치 불러오기
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# 메인 평가 함수
def main():
    # 디바이스 설정 (GPU 우선)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 테스트 데이터셋 로드
    test_dataset = FFPPFaceDataset(TEST_CSV, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    print(f"[INFO] Test samples: {len(test_dataset)}")

    # 모델 및 Loss 함수 준비
    model = get_model(device=device)
    criterion = nn.CrossEntropyLoss()

    # 평가 지표 변수 초기화
    total = 0
    correct = 0
    running_loss = 0.0

    # 평가 루프
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 모델 추론
            outputs = model(images)

            # Loss 계산
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # 정확도 계산
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / total
    test_acc = correct / total

    print(f"[RESULT] Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")


# 모듈 실행 시 main() 실행
if __name__ == "__main__":
    main()
