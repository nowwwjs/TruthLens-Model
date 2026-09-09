"""Unified, backend-friendly ensemble inference pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig
from .ensemble import weighted_soft_vote
from .models import load_model_checkpoint


def preprocess_pil(image: Image.Image):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


class DeepfakeDetectionPipeline:
    """Run face crop, per-backbone inference, and weighted soft voting."""

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        device: str | None = None,
        use_face_crop: bool | None = None,
        model_loader: Callable[[Any, str], Any] = load_model_checkpoint,
        models: dict[str, Any] | None = None,
        preprocessor: Callable[[Image.Image], Any] = preprocess_pil,
    ) -> None:
        import torch

        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_face_crop = config.use_face_crop if use_face_crop is None else use_face_crop
        self.preprocessor = preprocessor
        self._models = models or {
            spec.name: model_loader(spec, self.device) for spec in config.models
        }
        expected = {spec.name for spec in config.models}
        if set(self._models) != expected:
            raise ValueError(f"Configured and loaded model names differ: {expected} / {set(self._models)}")
        self.face_detector = None
        if self.use_face_crop:
            self._init_face_detector()

    def _init_face_detector(self) -> None:
        try:
            from model.face_detectors import make_detector

            self.face_detector = make_detector("mtcnn", device=self.device)
        except Exception:
            self.use_face_crop = False

    def run(self, image: Image.Image) -> dict[str, Any]:
        import torch

        if not isinstance(image, Image.Image):
            return {"success": False, "error": "invalid_input", "message": "PIL 이미지가 필요합니다."}
        image = image.convert("RGB")
        if self.use_face_crop:
            image = self._crop_face(image)
            if image is None:
                return {
                    "success": False,
                    "error": "no_face",
                    "message": "얼굴을 찾을 수 없습니다. 얼굴이 잘 보이는 이미지를 사용해 주세요.",
                }
        try:
            tensor = self.preprocessor(image).to(self.device)
            scores: dict[str, float] = {}
            with torch.no_grad():
                for name, model in self._models.items():
                    scores[name] = float(torch.softmax(model(tensor), dim=1)[0, 1].item())
            weights = {spec.name: spec.ensemble_weight for spec in self.config.models}
            fake_probability = round(float(weighted_soft_vote(scores, weights)), 6)
        except Exception as error:
            return {"success": False, "error": "model_error", "message": f"모델 추론 오류: {error}"}

        max_distance = max(self.config.threshold, 1.0 - self.config.threshold, 1e-12)
        confidence_score = round(min(1.0, abs(fake_probability - self.config.threshold) / max_distance), 6)
        return {
            "success": True,
            "label": "FAKE" if fake_probability >= self.config.threshold else "REAL",
            "fake_probability": fake_probability,
            "real_probability": round(1.0 - fake_probability, 6),
            "confidence": self._confidence_level(confidence_score),
            "confidence_score": confidence_score,
            "model_scores": {name: round(score, 6) for name, score in scores.items()},
            "threshold": self.config.threshold,
            "model_version": self.config.version,
        }

    def _crop_face(self, image: Image.Image) -> Image.Image | None:
        try:
            import cv2
            import numpy as np
            from model.face_detectors import crop_face

            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            detection = self.face_detector.detect_main_face(image_bgr)
            if detection is None:
                return None
            face_bgr = crop_face(image_bgr, detection, output_size=224)
            if face_bgr is None or face_bgr.size == 0:
                return None
            return Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            return None

    @staticmethod
    def _confidence_level(score: float) -> str:
        if score < 0.2:
            return "low"
        if score < 0.5:
            return "medium"
        return "high"


class DeepfakeDetector:
    """Legacy path-based adapter backed by the unified pipeline."""

    def __init__(self, device: str | None = None, **kwargs: Any) -> None:
        self.pipeline = DeepfakeDetectionPipeline(device=device, **kwargs)

    def predict(self, image_path: str) -> dict[str, Any]:
        try:
            with Image.open(image_path) as image:
                result = self.pipeline.run(image.convert("RGB"))
        except Exception as error:
            return {"status": "error", "message": str(error)}
        if not result.get("success"):
            return {"status": "error", "message": result.get("message", "Unknown error")}
        return {
            "status": "success",
            "label": result["label"],
            "score": round(result["fake_probability"] * 100, 2),
            "details": result["model_scores"],
        }
