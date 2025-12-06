# src/train_resnet50.py

from pathlib import Path
import time
import sys
import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

# --- 모듈 경로 수정 로직 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import FaceDataset
from .paths import WEIGHTS_DIR, PROJECT_ROOT

# 가중치 저장 폴더 생성
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 🚨 [경로 정의] 두 데이터셋의 CSV 및 가중치 경로 하드코딩
# =========================================================
INDICES_DIR = PROJECT_ROOT / "data" / "processed" / "indices"

# 1. FF++ 경로 (Stage 1)
FFPP_TRAIN_CSV = INDICES_DIR / "ffpp_train.csv"
FFPP_VAL_CSV = INDICES_DIR / "ffpp_val.csv"
FFPP_WEIGHTS_NAME = "ffpp_resnet50_advanced.pth"

# 2. Celeb-DF 경로 (Stage 2)
CELEBDF_TRAIN_CSV = INDICES_DIR / "celebdf_train.csv"
CELEBDF_VAL_CSV = INDICES_DIR / "celebdf_val.csv"
CELEBDF_WEIGHTS_NAME = "celebdf_resnet50_finetuned.pth"


# ============================================
# 데이터로더 & 모델 유틸리티
# ============================================


def get_dataloaders(train_csv, val_csv, batch_size=64):  # ResNet50은 배치 64 권장
    print(f"[LOADER] Loading Train: {train_csv.name} | Val: {val_csv.name}")
    train_dataset = FaceDataset(train_csv, train=True)
    val_dataset = FaceDataset(val_csv, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def get_model(num_classes=2, resume_path=None, device="cpu"):
    """
    ResNet50 모델 생성 (ImageNet Pretrained).
    resume_path가 있으면 가중치를 로드하여 이어서 학습(Fine-tuning).
    """
    # 🚨 [수정] ResNet50 사용
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if resume_path:
        print(f"[INFO] Loading weights for Fine-tuning: {resume_path}")
        try:
            state = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print("   -> Weights loaded successfully!")
        except FileNotFoundError:
            print(f"   -> [ERROR] Weights file not found: {resume_path}")
            return None

    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


# ============================================
# 개별 학습 세션 실행 함수
# ============================================


def run_training_session(
    session_name, train_csv, val_csv, output_name, resume_from=None, epochs=10, lr=1e-4
):
    print(f"\n{'='*20} [{session_name}] START {'='*20}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 데이터 준비
    try:
        train_loader, val_loader = get_dataloaders(
            train_csv, val_csv, batch_size=32
        )  # 메모리 부족 시 배치 줄임
    except FileNotFoundError:
        print(f"[SKIP] CSV not found for {session_name}. Check preprocessing.")
        return None

    # 모델 준비
    model = get_model(resume_path=resume_from, device=device)
    if model is None:
        return None

    # 🚨 [ResNet50 전용 최적화] Label Smoothing & AdamW
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 스케줄러 추가 (CosineAnnealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    output_path = WEIGHTS_DIR / output_name

    for epoch in range(1, epochs + 1):
        start = time.time()
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        v_loss, v_acc = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step()  # 스케줄러 업데이트
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{epochs} | Train: loss={t_loss:.4f}, acc={t_acc:.4f} | Val: loss={v_loss:.4f}, acc={v_acc:.4f} | {elapsed:.1f}s"
        )

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), output_path)
            print(f"  --> Best model saved: {output_name} (acc: {best_acc:.4f})")

    print(f"{'='*20} [{session_name}] DONE {'='*20}\n")
    return output_path


# ============================================
# 메인 파이프라인 (순차 실행)
# ============================================


def main():
    print("[PIPELINE] Starting ResNet50 2-Stage Training Pipeline...\n")

    # 1단계: FF++ 학습 (Base Training)
    ffpp_weights = run_training_session(
        session_name="Stage 1: FF++ Base Training (ResNet50)",
        train_csv=FFPP_TRAIN_CSV,
        val_csv=FFPP_VAL_CSV,
        output_name=FFPP_WEIGHTS_NAME,
        resume_from=None,
        epochs=10,  # ResNet50은 더 깊어서 epoch을 좀 더 늘림
        lr=1e-4,
    )

    # 1단계 실패 시 중단
    if ffpp_weights is None or not ffpp_weights.exists():
        print("[ERROR] Stage 1 failed. Stopping pipeline.")
        return

    # 2단계: Celeb-DF 학습 (Fine-tuning)
    run_training_session(
        session_name="Stage 2: Celeb-DF Fine-tuning (ResNet50)",
        train_csv=CELEBDF_TRAIN_CSV,
        val_csv=CELEBDF_VAL_CSV,
        output_name=CELEBDF_WEIGHTS_NAME,
        resume_from=ffpp_weights,  # 1단계 결과물 이어서 학습
        epochs=5,  # Fine-tuning은 적은 epoch
        lr=1e-5,  # 낮은 학습률
    )

    print("[PIPELINE] All training sessions finished!")


if __name__ == "__main__":
    main()
