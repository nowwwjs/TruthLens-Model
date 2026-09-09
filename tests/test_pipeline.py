import pytest
from PIL import Image


torch = pytest.importorskip("torch")

from truthlens.config import ModelSpec, PipelineConfig  # noqa: E402
from truthlens.pipeline import DeepfakeDetectionPipeline  # noqa: E402


class FixedModel(torch.nn.Module):
    def __init__(self, fake_logit: float):
        super().__init__()
        self.fake_logit = fake_logit

    def forward(self, value):
        return torch.tensor([[0.0, self.fake_logit]], dtype=torch.float32).repeat(value.shape[0], 1)


def test_pipeline_preserves_legacy_keys_and_adds_portfolio_fields(tmp_path):
    config = PipelineConfig(
        models=(
            ModelSpec("efficientnet", "efficientnet_b0", tmp_path / "a.pth", 0.8),
            ModelSpec("mobilenet", "mobilenet_v3", tmp_path / "b.pth", 0.2),
        ),
        use_face_crop=False,
        version="test",
    )
    pipeline = DeepfakeDetectionPipeline(
        config=config,
        device="cpu",
        models={"efficientnet": FixedModel(2.0), "mobilenet": FixedModel(-1.0)},
        preprocessor=lambda image: torch.zeros(1, 3, 8, 8),
    )
    result = pipeline.run(Image.new("RGB", (8, 8), "white"))
    assert result["success"] is True
    for key in ("label", "fake_probability", "real_probability", "confidence"):
        assert key in result
    assert result["model_version"] == "test"
    assert set(result["model_scores"]) == {"efficientnet", "mobilenet"}
