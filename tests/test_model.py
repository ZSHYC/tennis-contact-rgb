from pathlib import Path

import pytest
import torch

from tennis_contact.model import TennisContactNet, load_model


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_loads_strictly_and_has_expected_layout():
    model, metadata = load_model(ROOT / "weights/best.pt")
    initialization, _ = load_model(ROOT / "weights/initialization.pt")

    assert sum(parameter.numel() for parameter in model.parameters()) == 6_797_590
    assert len(model.state_dict()) == 583
    assert len(initialization.state_dict()) == 583
    assert metadata["thresholds"] == {"hit": 0.60, "bounce": 0.48}
    assert metadata["secondary_thresholds"] == {"hit": 0.67, "bounce": 0.57}


def test_forward_and_trunk2_backward():
    torch.manual_seed(7)
    model = TennisContactNet().train()
    rgb = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)

    output = model(rgb)
    assert output["eventness_logit"].shape == (2,)
    assert output["type_logits"].shape == (2, 2)
    assert output["spatial_logits"].shape[0] == 2
    assert output["presence_logit"].shape == (2,)
    assert output["state"].shape == (2, 68)
    assert all(not module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm3d))

    (output["eventness_logit"].square().mean() + output["type_logits"].square().mean()).backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if name.startswith("trunk.2.") and parameter.grad is not None
    ) > 0


def test_input_and_precision_validation():
    model = TennisContactNet()
    with pytest.raises(ValueError, match="uint8"):
        model(torch.zeros(2, 3, 32, 32))
    with pytest.raises(ValueError, match="precision"):
        model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), precision="fp16")


def test_bf16_is_limited_to_visual_backbone():
    model = TennisContactNet().eval()
    observed = {}
    conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv3d))
    hooks = [
        conv.register_forward_hook(lambda _module, _args, value: observed.__setitem__("visual", value.dtype)),
        model.gru.register_forward_pre_hook(
            lambda _module, args: observed.__setitem__("temporal", args[0].dtype)
        ),
    ]
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), precision="bf16")
    for hook in hooks:
        hook.remove()

    assert observed == {"visual": torch.bfloat16, "temporal": torch.float32}
    assert all(value.dtype == torch.float32 for value in output.values())
