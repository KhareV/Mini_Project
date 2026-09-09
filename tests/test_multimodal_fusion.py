import torch

from models.multimodal_fusion import MultimodalFusionConfig, MultimodalFusionModel


def test_fusion_model_supports_missing_modalities():
    model = MultimodalFusionModel(MultimodalFusionConfig(signal_length=2500))
    inputs = (
        torch.randn(2, 1, 2500), torch.randn(2, 1, 2500),
        torch.randn(2, 2), torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    )
    logits = model(*inputs)
    assert logits.shape == (2, 2)


def test_fusion_model_rejects_unknown_mask_width():
    model = MultimodalFusionModel()
    try:
        model(torch.randn(1, 1, 2500), torch.randn(1, 1, 2500),
              torch.randn(1, 2), torch.ones(1, 3))
    except ValueError as exc:
        assert "presence_mask" in str(exc)
    else:
        raise AssertionError("mask width must be validated")
