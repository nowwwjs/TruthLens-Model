# TruthLens: 딥페이크 탐지 모델

제한된 데이터와 연산 환경에서 얼굴 기반 딥페이크 분류기의 과적합과 일반화 문제를 다룬 졸업작품 AI 모델 파트입니다.

## 문제와 접근

추출 프레임을 무작위로 분할하면 같은 원본 영상의 유사 프레임이 train과 test에 섞일 수 있습니다. 이 저장소는 `source_dataset + group_id`를 분할 단위로 관리해 이 누수를 막고, 전이학습과 regularization을 결합한 실험 흐름을 구현합니다.

- EfficientNet-B0와 MobileNetV3-Large의 weighted soft voting
- ImageNet 전이학습, head warm-up, partial unfreeze
- Dropout 0.5, weight decay, augmentation, Focal Loss
- validation 데이터에서만 threshold를 선택하고 test set에는 고정값을 한 번 적용
- 그룹 누수, threshold, weighted vote, CPU smoke test를 포함한 단위 테스트

## 검증된 기록과 한계

보존된 체크포인트와 연결되는 실험 기록은 DeepFake-Eval-2024의 EfficientNet-B0 fine-tuning 한 건입니다. 최대 200개 train 영상과 영상당 8프레임 설정에서 epoch 3의 validation ROC-AUC는 **0.6731**, accuracy는 **0.6272**였습니다.

이 기록은 앙상블의 성능 향상, 외부 데이터 일반화, 확률 보정을 입증하지 않습니다. 이 프로젝트는 교육·연구용 프로토타입이며, 출력은 포렌식 증거나 자동 제재 판단에 사용할 수 없습니다.

## 구조

```text
truthlens/   # 데이터, 모델, 학습, 평가, 앙상블, 추론의 기준 패키지
model/       # 기존 백엔드 import 호환 계층
src/         # manifest 생성, 학습, 평가 CLI
tests/       # 외부 데이터 없이 실행하는 검증 코드
artifacts/   # 대용량 가중치의 메타데이터와 해시
```

가중치와 원본 데이터는 저장소에 포함하지 않습니다. 파일명과 기존 코드만으로 추정한 학습 출처는 검증된 성능과 구분합니다.

## 실행

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

새 학습에는 각 데이터셋의 접근 조건과 라이선스 확인이 필요합니다.
