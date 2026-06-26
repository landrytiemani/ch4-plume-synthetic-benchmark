import pytest

torch = pytest.importorskip("torch")

from ch4l1c.train import build_model


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def test_phys_tau_net_is_distinct_from_attention_unet():
    attn = build_model("attn_unet", in_channels=14)
    phys = build_model("phys_tau_net", in_channels=14)

    assert type(attn).__name__ != type(phys).__name__
    assert _param_count(attn) != _param_count(phys)


def test_phys_tau_net_forward_shape():
    model = build_model("phys_tau_net", in_channels=14)
    x = torch.randn(2, 14, 128, 128)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (2, 1, 128, 128)
    assert torch.isfinite(y).all()
