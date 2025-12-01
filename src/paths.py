from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"        # 나중에 쓸 데이터 폴더
WEIGHTS_DIR = PROJECT_ROOT / "weights"  # .pth 저장한 곳
