# src/evaluate_resnet50.py

from pathlib import Path
import argparse
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

# 모듈 경로 수정 로직
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import FaceDataset
from .paths import WEIGHTS_DIR, PROJECT_ROOT

# ---------------------------------------------------------
# 🚨 [평가 시나리오 정의]
# ---------------------------------------------------------
INDICES_DIR = PROJECT_ROOT / "data" / "processed" / "indices"

EVALUATION_TASKS = [
    {
        "name": "Scenario 1: FF++ Evaluation (ResNet50)",
        "csv_path": INDICES_DIR / "ffpp_test.csv",
        "model_path": WEIGHTS_DIR / "ffpp_resnet50_advanced.pth",
    },
    {
        "name": "Scenario 2: Celeb-DF Evaluation (ResNet50 Fine-tuned)",
        "csv_path": INDICES_DIR / "celebdf_test.csv",
        "model_path": WEIGHTS_DIR / "celebdf_resnet50_finetuned.pth",
    },
]


def get_model(model_path: Path, num_classes=2, device="cpu"):
    """Load the ResNet50 model with trained weights."""

    # 🚨 [수정] ResNet50 사용
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 가중치 로드
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[INFO] Loaded weights from: {model_path.name}")
    except FileNotFoundError:
        print(f"[ERROR] Weights not found: {model_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    model.to(device)
    model.eval()
    return model


def run_evaluation(task_name, csv_path, model_path, batch_size, device):
    print(f"\n{'='*10} {task_name} {'='*10}")

    # 1. 모델 로드
    model = get_model(model_path, device=device)
    if model is None:
        print("[SKIP] Skipping this task due to model loading error.")
        return

    # 2. 데이터셋 로드
    try:
        test_dataset = FaceDataset(csv_path, train=False)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    if len(test_dataset) == 0:
        print("[WARN] Dataset is empty. Skipping.")
        return

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"[INFO] Samples: {len(test_dataset)} | Model: {model_path.name}")

    # 3. 평가 루프
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # AUC 계산을 위해 확률값(Softmax) 저장
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 4. 결과 계산
    test_loss = running_loss / total
    test_acc = correct / total

    try:
        if len(np.unique(all_labels)) > 1:
            test_auc = roc_auc_score(all_labels, all_probs)
        else:
            test_auc = 0.0
            print("[WARN] AUC not calculated (single class in batch)")
    except:
        test_auc = 0.0

    # 5. 출력
    print("-" * 50)
    print(f"[RESULT] Loss : {test_loss:.4f}")
    print(f"[RESULT] Acc  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"[RESULT] AUC  : {test_auc:.4f}")
    print("-" * 50)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 정의된 모든 평가 작업 순차 실행
    for task in EVALUATION_TASKS:
        run_evaluation(
            task["name"], task["csv_path"], task["model_path"], args.batch_size, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet50 Models Sequentially"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    main(args)
