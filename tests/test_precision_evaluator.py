import numpy as np
import pytest
import torch

from scripts.compare_precision import (
    _forward_modes,
    _load_reference_scores,
    _modes_to_compute,
    quality_differences,
)


class _ModeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, rgb, *, precision):
        self.calls.append((
            precision,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ))
        values = rgb.float().mean(dim=(1, 2, 3))
        return {"eventness_logit": values, "type_logits": torch.stack((values, -values), dim=1)}


def _report(tp=9, fp=1, fn=1, f1=.9, mean_error=1.0, fp_per_minute=.1):
    metrics = {
        "tp": tp, "fp": fp, "fn": fn, "precision": .9, "recall": .9, "f1": f1,
        "mean_error": mean_error, "fp_per_minute": fp_per_minute,
    }
    return {
        "overall": metrics,
        "by_type": {"hit": metrics, "bounce": metrics},
        "by_source": {"source": {"overall": metrics, "by_type": {"hit": metrics, "bounce": metrics}}},
    }


def test_precision_modes_use_declared_order_precision_and_tf32():
    base, fused = _ModeModel(), _ModeModel()
    modes = ("fp32", "tf32", "bf16", "bf16_tf32", "fp16", "fused_fp32", "fused_fp16")
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        values, seconds = _forward_modes(
            {"base": base, "fused": fused},
            torch.zeros(2, 3, 4, 4, dtype=torch.uint8), modes, torch.device("cpu")
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    assert base.calls == [
        ("fp32", False, False), ("fp32", True, True),
        ("bf16", False, False), ("bf16", True, True),
        ("fp16", False, False),
    ]
    assert fused.calls == [("fp32", False, False), ("fp16", False, False)]
    assert list(values) == list(seconds) == list(modes)


def test_cached_reference_scores_are_reused_and_require_exact_frame_times(tmp_path):
    root = tmp_path / "reference"
    path = root / "validation" / "sample.npz"
    path.parent.mkdir(parents=True)
    frame_times = np.array([0.0, 0.04, 0.08])
    scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    np.savez(path, frame_times=frame_times, fp32=scores)
    row = {"id": "sample", "axis": "validation"}

    assert np.array_equal(_load_reference_scores(root, row, frame_times), scores)
    assert _modes_to_compute(("fp32", "fused_fp32", "fused_fp16"), root) == (
        "fused_fp32", "fused_fp16",
    )
    with pytest.raises(ValueError, match="frame_times"):
        _load_reference_scores(root, row, frame_times + 1e-12)


def test_quality_comparison_exposes_count_and_f1_regressions():
    differences = quality_differences(
        _report(), _report(tp=8, fp=2, fn=2, f1=.8, mean_error=1.25, fp_per_minute=.2)
    )
    assert len(differences) == 6
    assert differences[0]["cell"] == "overall"
    assert differences[0]["regressions"] == pytest.approx({
        "tp": -1, "fp": 1, "fn": 1, "f1": -.1,
        "mean_error": .25, "fp_per_minute": .1,
    })
