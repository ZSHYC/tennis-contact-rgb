import copy
from pathlib import Path

import pytest
import torch

from tennis_contact.model import TennisContactNet, _FrameSE, load_model


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_loads_strictly_and_has_expected_layout():
    model, metadata = load_model(ROOT / "weights/best.pt")
    initialization, _ = load_model(ROOT / "weights/initialization.pt")

    assert sum(parameter.numel() for parameter in model.parameters()) == 6_797_590
    assert len(model.state_dict()) == 583
    assert len(initialization.state_dict()) == 583
    assert metadata["thresholds"] == {"hit": 0.60, "bounce": 0.48}
    assert metadata["secondary_thresholds"] == {"hit": 0.67, "bounce": 0.57}


def test_constructor_preserves_frozen_x3d_seed_contract():
    torch.manual_seed(2026)
    model = TennisContactNet()
    expected = {
        "trunk.0.conv.conv_t.weight": [-0.023615488782525063, 0.09282617270946503,
                                        0.12029710412025452, -0.0778420940041542],
        "trunk.3.res_blocks.10.branch2.norm_b.1.block.2.weight":
            [-0.004059214144945145, 0.0022257249802351,
             -0.00022152686142362654, -0.1371549665927887],
        "pool.post_conv.weight": [-0.022027993574738503, 0.014491512440145016,
                                  -0.0024628571700304747, 0.02690165489912033],
        "gru.weight_ih_l0": [0.019463106989860535, -0.03992495685815811,
                             -0.04006730020046234, 0.04196435958147049],
        "state_adapter.0.weight": [-0.08995778858661652, 0.09162204712629318,
                                   0.03869125619530678, 0.04250316694378853],
    }
    state = model.state_dict()
    for name, values in expected.items():
        torch.testing.assert_close(state[name].flatten()[:4], torch.tensor(values), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.rand(4),
        torch.tensor([0.7665562629699707, 0.6637804508209229,
                      0.7691547274589539, 0.9119697213172913]),
        rtol=0,
        atol=0,
    )


def test_inference_fusion_is_complete_close_and_idempotent():
    torch.manual_seed(11)
    model = TennisContactNet().eval()
    assert sum(isinstance(module, torch.nn.BatchNorm3d) for module in model.modules()) == 83
    fused = copy.deepcopy(model)
    rgb = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    with torch.inference_mode():
        expected = model(rgb)
        returned = fused.fuse_for_inference()
        actual = fused(rgb)
    assert returned is fused
    assert not any(isinstance(module, torch.nn.BatchNorm3d) for module in fused.modules())
    assert sum(isinstance(module, _FrameSE) for module in fused.modules()) == 15
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
    state = {key: value.clone() for key, value in fused.state_dict().items()}
    assert fused.fuse_for_inference() is fused
    assert all(torch.equal(state[key], value) for key, value in fused.state_dict().items())

    with pytest.raises(ValueError, match="eval"):
        TennisContactNet().fuse_for_inference()


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
        model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), precision="int8")


@pytest.mark.parametrize("precision,visual_dtype", [("bf16", torch.bfloat16), ("fp16", torch.float16)])
def test_reduced_precision_is_limited_to_visual_backbone(precision, visual_dtype):
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
        output = model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), precision=precision)
    for hook in hooks:
        hook.remove()

    assert observed == {"visual": visual_dtype, "temporal": torch.float32}
    assert all(value.dtype == torch.float32 for value in output.values())
