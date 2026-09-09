import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from truthlens.models import create_model, set_transfer_learning_stage  # noqa: E402


@pytest.mark.parametrize("architecture", ["efficientnet_b0", "mobilenet_v3"])
def test_offline_model_creation_and_transfer_stages(architecture):
    model = create_model(architecture, dropout_rate=0.4, pretrained=False)
    set_transfer_learning_stage(model, architecture, "head")
    assert any(parameter.requires_grad for parameter in model.classifier.parameters())
    assert any(not parameter.requires_grad for parameter in model.features.parameters())
    set_transfer_learning_stage(model, architecture, "full")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_one_cpu_training_step():
    model = create_model("mobilenet_v3", pretrained=False)
    model.train()
    images = torch.rand(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
