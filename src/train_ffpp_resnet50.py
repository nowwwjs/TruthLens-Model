# src/train_ffpp_resnet50.py

from pathlib import Path
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from .dataset_ffpp import FFPPFaceDataset
from .paths import PROJECT_ROOT, WEIGHTS_DIR


# 경로 설정
# data 디렉토리
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = DATA_DIR / "processed" / "indices"

# 학습/검증용 CSV 경로
TRAIN_CSV = INDICES_DIR / "ffpp_train.csv"
VAL_CSV = INDICES_DIR / "ffpp_val.csv"

# 모델 가중치 저장 디렉토리 (weights/)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 데이터로더 구성
# ============================================

def get_dataloaders(batch_size: int = 64):
    """
    FFPP 얼굴 데이터셋을 이용해 Train / Val DataLoader를 생성.
    ResNet50 모델 특성상 기본 배치 크기를 64로 설정.
    """
    train_dataset = FFPPFaceDataset(TRAIN_CSV, train=True)
    val_dataset = FFPPFaceDataset(VAL_CSV, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


# ============================================
# 모델 생성
# ============================================

def get_model(num_classes: int = 2):
    """
    torchvision의 ResNet50을 기반으로 한 이진 분류 모델 생성.
    - ImageNet 사전학습 가중치를 사용.
    """
    # ImageNet으로 사전 학습된 ResNet50 불러오기
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 마지막 FC 레이어를 2 클래스 분류에 맞게 교체
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ============================================
# 학습 epoch 루프
# ============================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    모델 한 epoch 학습.
    loss와 accuracy를 반환.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================
# 검증 epoch 루프
# ============================================

def eval_one_epoch(model, loader, criterion, device):
    """
    검증 데이터셋에 대해 1 epoch 평가.
    loss와 accuracy를 반환.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================
# 메인 학습 루프
# ============================================

def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 데이터로더 준비
    train_loader, val_loader = get_dataloaders(batch_size=64)
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples:   {len(val_loader.dataset)}")

    # 모델, Loss, Optimizer, 스케줄러 설정
    model = get_model(num_classes=2).to(device)

    # label smoothing 적용 (일반 CrossEntropy보다 과적합 완화에 도움)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW 옵티마이저 (weight decay 포함)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 10

    # CosineAnnealingLR 스케줄러 (epoch에 따라 lr를 코사인 형태로 감소)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0

    # 베스트 모델 저장 경로
    # -> 여기서 저장한 .pth를 evaluate_ffpp_resnet50에서 그대로 사용
    best_model_path = WEIGHTS_DIR / "ffpp_resnet50_advanced.pth"

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # 한 epoch 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        # 스케줄러 스텝
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        # 베스트 모델 저장 (검증 정확도 기준)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(
                f"  -> New best model saved to {best_model_path} "
                f"(val_acc={best_val_acc:.4f})"
            )

    print("[DONE] Training finished.")
    print(f"[BEST] Best val acc: {best_val_acc:.4f}")
    print(f"[MODEL] Saved to: {best_model_path}")


if __name__ == "__main__":
    main()
