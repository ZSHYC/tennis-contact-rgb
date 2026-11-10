import threading
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.compare_precision as precision_evaluator
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


def test_predict_modes_prefetches_each_decode_once(monkeypatch):
    main_thread = threading.get_ident()
    reads = []

    class FakeSampler:
        count = 101

        def __init__(self, _frame_times):
            pass

        def window(self, start):
            return {
                "indices": np.array([start // 100]),
                "core_mask": np.array([True]),
            }

    class FakeReader:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self, _video, indices):
            reads.append((threading.get_ident(), indices.item()))
            return torch.zeros(1, 3, 4, 4, dtype=torch.uint8)

    class FakeAccumulator:
        def __init__(self, frame_times):
            self.length = len(frame_times)

        def add(self, _window, _values):
            pass

        def finish(self):
            return np.zeros((self.length, 3), dtype=np.float32)

    def fake_forward(_models, _rgb, modes, _target):
        return ({mode: torch.zeros(1, 3) for mode in modes}, {mode: 0.0 for mode in modes})

    monkeypatch.setattr(precision_evaluator, "WindowSampler", FakeSampler)
    monkeypatch.setattr(precision_evaluator, "FrameReader", FakeReader)
    monkeypatch.setattr(precision_evaluator, "_LogitAccumulator", FakeAccumulator)
    monkeypatch.setattr(precision_evaluator, "_forward_modes", fake_forward)

    _scores, cost = precision_evaluator._predict_modes(
        {}, Path("unused.mp4"), np.array([0.0, 0.04]), ("fp32",), torch.device("cpu")
    )
    assert [index for _thread, index in reads] == [0, 1]
    assert {thread for thread, _index in reads}.isdisjoint({main_thread})
    assert cost["windows"] == 2


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
