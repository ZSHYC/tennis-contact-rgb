from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tennis_contact import training


@dataclass
class Sample:
    id: str
    video: Path
    split: str
    source: str
    events: list[dict]
    frame_times: np.ndarray


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(3)])
        self.head = nn.Linear(1, 3)

    def forward(self, rgb, *, precision="fp32"):
        values = rgb[:, :1, :1, :1].float().reshape(-1, 1) / 255
        values = self.trunk[2](values)
        logits = self.head(values)
        return {
            "eventness_logit": logits[:, 0],
            "type_logits": logits[:, 1:],
            "spatial_logits": logits[:, :1],
            "presence_logit": logits[:, 0],
            "state": logits,
        }


def _fake_video_io(monkeypatch, window, frames):
    class Sampler:
        def __init__(self, _times):
            pass

        def window(self, start, **kwargs):
            return window(None, start, **kwargs)

    class Reader:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            pass

        def read(self, video, indices):
            return frames(video, indices)

    monkeypatch.setattr(training, "WindowSampler", Sampler)
    monkeypatch.setattr(training, "FrameReader", Reader)


def test_joint_loss_uses_soft_eventness_and_actual_joint_scores():
    sample = Sample(
        "pts", Path("unused.mp4"), "train", "test",
        [{"frame_number": 0, "event_type": "hit"}], np.asarray([10.0, 10.1]),
    )
    native = training._targets(
        sample,
        {
            "indices": np.asarray([0, 1]),
            "times": np.asarray([10.0, 10.033]),
            "weights": np.ones(2),
            "core_mask": np.ones(2, dtype=bool),
        },
        torch.device("cpu"),
    )
    assert float(native["eventness"][1]) == pytest.approx(np.exp(-0.5 * (0.1 / 0.04) ** 2))

    output = {
        "eventness_logit": torch.tensor([0.3, -0.2], requires_grad=True),
        "type_logits": torch.tensor([[0.7, -0.1], [-0.4, 0.5]], requires_grad=True),
    }
    targets = {
        "weights": torch.tensor([1.0, 0.5]),
        "eventness": torch.tensor([0.6, 0.0]),
        "types": torch.tensor([0, 1]),
        "type_mask": torch.tensor([True, False]),
    }
    event, joint = training.loss_sums(output, targets)

    expected_event = (
        F.binary_cross_entropy_with_logits(
            output["eventness_logit"],
            targets["eventness"],
            pos_weight=torch.tensor(training.POSITIVE_WEIGHT),
            reduction="none",
        )
        * targets["weights"]
    ).sum()
    adjusted = training.POSITIVE_WEIGHT * 0.6 / (training.POSITIVE_WEIGHT * 0.6 + 0.4)
    probabilities = output["eventness_logit"].sigmoid()[:, None] * output["type_logits"].softmax(-1)
    expected_joint = F.binary_cross_entropy(
        probabilities[0], torch.tensor([adjusted, 0.0]), reduction="sum"
    )
    assert float(event.detach()) == pytest.approx(float(expected_event.detach()))
    assert float(joint.detach()) == pytest.approx(float(expected_joint.detach()))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        training.loss_sums(
            {**output, "eventness_logit": torch.tensor([float("nan"), 0.0])}, targets
        )


def test_native_targets_match_full_distance_oracle_for_ties():
    rng = np.random.default_rng(7)
    irregular = np.cumsum(rng.uniform(0.02, 0.06, 200))
    irregular[50] = (irregular[40] + irregular[60]) / 2
    cases = [
        (
            np.array([0.0, 0.0999999996, 0.2, 0.31, 0.5]),
            [0, 2, 4],
            np.array([0, 1, 0]),
        ),
        (irregular, [4, 40, 60, 110, 170], np.array([1, 0, 1, 1, 0])),
    ]
    for frame_times, frames, event_types in cases:
        sample = Sample(
            "ties", Path("unused.mp4"), "train", "test",
            [
                {"frame_number": frame, "event_type": "bounce" if event_type else "hit"}
                for frame, event_type in zip(frames, event_types, strict=True)
            ],
            frame_times,
        )
        distances = np.abs(frame_times[:, None] - frame_times[frames][None, :])
        chosen = distances.argmin(axis=1)
        nearest = distances[np.arange(len(frame_times)), chosen]
        expected_types = event_types[chosen]
        tied = np.isclose(distances, nearest[:, None], atol=1e-9, rtol=0.0)
        expected_mask = (nearest <= training.TYPE_RADIUS_SECONDS + 1e-9) & ~np.any(
            tied & (event_types[None, :] != expected_types[:, None]), axis=1
        )

        targets = training._native_targets(sample)

        np.testing.assert_allclose(
            targets["eventness"],
            np.asarray(
                np.exp(-0.5 * np.square(nearest / training.EVENT_SIGMA_SECONDS)), dtype=np.float32
            ),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(targets["types"], expected_types)
        np.testing.assert_array_equal(targets["type_mask"], expected_mask)


def test_class_balance_ignores_repeated_epochs():
    sample = Sample(
        "one", Path("unused.mp4"), "train", "test",
        [{"frame_number": 0, "event_type": "hit"}], np.arange(0, 4.01, 0.04),
    )
    class CountingSampler:
        def __init__(self):
            self.sampler = training.WindowSampler(sample.frame_times)
            self.calls = 0

        def window(self, *args, **kwargs):
            self.calls += 1
            return self.sampler.window(*args, **kwargs)

    one_epoch = [(0, 0, False), (0, 100, True)]
    expected = training._class_balance([sample], one_epoch)
    sampler = CountingSampler()
    actual = training._class_balance([sample], one_epoch * 3, samplers=[sampler])

    assert actual == expected
    assert sampler.calls == len(one_epoch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction semantics")
def test_cuda_denominator_matches_per_window_device_reductions():
    times = np.arange(12, dtype=np.float64) / 10
    weights = torch.as_tensor(training.WindowSampler(times).window(0)["weights"], dtype=torch.float32)
    targets = [{"weights": weights}, {"weights": weights}]
    expected = sum(float(target["weights"].cuda().sum()) for target in targets)

    actual = training._event_denominator(targets, torch.device("cuda"))

    assert actual.hex() == expected.hex()
    assert all(target["weights"].is_pinned() for target in targets)


def test_train_updates_trunk2_and_resume_matches_continuous(tmp_path, monkeypatch):
    sample = Sample(
        "one",
        tmp_path / "one.mp4",
        "train",
        "test",
        [{"frame_number": 0, "event_type": "hit"}],
        np.arange(0, 4.01, 0.04),
    )

    def window(_times, start, **_kwargs):
        indices = np.arange(start, min(start + 2, len(sample.frame_times)))
        return {
            "indices": indices,
            "times": sample.frame_times[indices],
            "core_mask": np.ones(len(indices), dtype=bool),
            "core_indices": indices,
            "logical_indices": indices,
            "weights": np.ones(len(indices), dtype=np.float32),
        }

    def frames(_video, indices):
        return torch.full((len(indices), 3, 1, 1), 128, dtype=torch.uint8)

    monkeypatch.setattr(training, "TennisContactNet", TinyModel)
    monkeypatch.setattr(training, "load_dataset", lambda *_args, **_kwargs: [sample])
    _fake_video_io(monkeypatch, window, frames)

    torch.manual_seed(11)
    initial = TinyModel()
    initial_path = tmp_path / "initial.pt"
    torch.save({"model": initial.state_dict(), "thresholds": {"hit": 0.6, "bounce": 0.48}}, initial_path)
    order_path = tmp_path / "order.json"
    order_path.write_text(json.dumps({
        "train_sample_ids": ["one"],
        "positive_weight": training._class_balance([sample], [(0, 0, False), (0, 100, False)]),
        "requests": [
            {"sample_id": "one", "start": 100, "mirror": True},
            {"sample_id": "one", "start": 0, "mirror": False},
        ],
    }))

    continuous = tmp_path / "continuous"
    training.train(
        "manifest.json", continuous, initial_path, precision="fp32", lr=1e-3, batch_size=1,
        order_path=order_path,
    )
    first = tmp_path / "resume"
    paused = training.train(
        "manifest.json", first, initial_path, precision="fp32", lr=1e-3, batch_size=1,
        max_steps=1, order_path=order_path,
    )
    assert paused["status"] == "paused"
    monkeypatch.setattr(
        training, "_class_balance",
        lambda *_args, **_kwargs: pytest.fail("resume rebuilt class balance"),
    )
    resumed = training.train(
        "manifest.json", first, precision="fp32", lr=1e-3, batch_size=1,
        resume=first / "training.pt", max_steps=2, order_path=order_path,
    )
    uninterrupted = torch.load(continuous / "training.pt", weights_only=False)
    resumed_state = torch.load(first / "training.pt", weights_only=False)
    inference = torch.load(first / "last.pt", weights_only=True)

    assert resumed["status"] == "complete"
    assert set(inference) == {"model", "thresholds", "secondary_thresholds", "description"}
    assert resumed["cursor"] == resumed["requests"]
    assert resumed["updates_this_run"] == 1
    assert any(
        not torch.equal(initial.state_dict()[name], resumed_state["model"][name])
        for name in initial.state_dict()
        if name.startswith("trunk.2.")
    )
    assert all(
        torch.equal(initial.state_dict()[name], resumed_state["model"][name])
        for name in initial.state_dict()
        if not name.startswith("trunk.2.")
    )
    assert all(
        torch.equal(resumed_state["model"][name], uninterrupted["model"][name])
        for name in resumed_state["model"]
    )
    for parameter, state in resumed_state["optimizer"]["state"].items():
        for name, value in state.items():
            expected = uninterrupted["optimizer"]["state"][parameter][name]
            assert torch.equal(value, expected) if isinstance(value, torch.Tensor) else value == expected


@pytest.mark.parametrize("interrupt", [True, False])
def test_training_saves_once_on_interrupt_or_final_periodic_boundary(tmp_path, monkeypatch, interrupt):
    sample = Sample(
        "one", tmp_path / "one.mp4", "train", "test",
        [{"frame_number": 0, "event_type": "hit"}], np.arange(0, 4.01, 0.04),
    )

    def window(_times, start, **_kwargs):
        indices = np.array([start])
        return {
            "indices": indices,
            "times": sample.frame_times[indices],
            "core_mask": np.ones(1, dtype=bool),
            "core_indices": indices,
            "logical_indices": indices,
            "weights": np.ones(1, dtype=np.float32),
        }

    class InterruptingModel(TinyModel):
        def forward(self, rgb, *, precision="fp32"):
            training._STOP_REQUESTED = interrupt
            return super().forward(rgb, precision=precision)

    saved = []
    monkeypatch.setattr(training, "TennisContactNet", InterruptingModel)
    monkeypatch.setattr(training, "load_dataset", lambda *_args, **_kwargs: [sample])
    _fake_video_io(
        monkeypatch,
        window,
        lambda _video, indices: torch.full((len(indices), 3, 1, 1), 128, dtype=torch.uint8),
    )
    monkeypatch.setattr(training, "_atomic_save", lambda path, _packet: saved.append(path.name))
    initial = InterruptingModel()
    initial_path = tmp_path / "initial.pt"
    torch.save({"model": initial.state_dict()}, initial_path)

    result = training.train(
        "manifest.json", tmp_path / "run", initial_path, precision="fp32", batch_size=1, epochs=32,
    )

    assert result["status"] == ("interrupted" if interrupt else "complete")
    assert result["steps"] == (1 if interrupt else 64)
    assert saved == ["training.pt", "last.pt"]


def test_training_prefetch_keeps_only_one_window_ahead(tmp_path, monkeypatch):
    sample = Sample(
        "one", tmp_path / "one.mp4", "train", "test",
        [{"frame_number": 0, "event_type": "hit"}], np.arange(0, 10.01, 0.04),
    )
    events = []

    def window(_times, start, **_kwargs):
        indices = np.array([min(start, len(sample.frame_times) - 1)])
        return {
            "indices": indices,
            "times": sample.frame_times[indices],
            "core_mask": np.ones(1, dtype=bool),
            "core_indices": np.array([0]),
            "logical_indices": np.array([start]),
            "weights": np.ones(1, dtype=np.float32),
        }

    def frames(_video, indices):
        events.append("read")
        return torch.full((len(indices), 3, 1, 1), 128, dtype=torch.uint8)

    class RecordingModel(TinyModel):
        def forward(self, rgb, *, precision="fp32"):
            events.append("forward")
            return super().forward(rgb, precision=precision)

    monkeypatch.setattr(training, "TennisContactNet", RecordingModel)
    monkeypatch.setattr(training, "load_dataset", lambda *_args, **_kwargs: [sample])
    monkeypatch.setattr(
        training,
        "_request_order",
        lambda *_args, **_kwargs: ([(0, start, False) for start in (0, 100, 200, 250)], None),
    )
    _fake_video_io(monkeypatch, window, frames)
    initial = RecordingModel()
    initial_path = tmp_path / "initial.pt"
    torch.save({"model": initial.state_dict()}, initial_path)

    result = training.train(
        "manifest.json", tmp_path / "run", initial_path, precision="fp32", batch_size=4,
    )

    third_read = [index for index, event in enumerate(events) if event == "read"][2]
    assert events.index("forward") < third_read
    assert result["steps"] == 1


def test_nonfinite_batch_does_not_update_parameters(tmp_path, monkeypatch):
    sample = Sample(
        "one", tmp_path / "one.mp4", "train", "test",
        [{"frame_number": 0, "event_type": "hit"}], np.arange(0, 4.01, 0.04),
    )

    def window(_times, start, **_kwargs):
        indices = np.arange(start, min(start + 2, len(sample.frame_times)))
        return {
            "indices": indices,
            "times": sample.frame_times[indices],
            "core_mask": np.ones(len(indices), dtype=bool),
            "core_indices": np.arange(len(indices)),
            "logical_indices": indices,
            "weights": np.ones(len(indices), dtype=np.float32),
        }

    class NonfiniteModel(TinyModel):
        trained = None

        def __init__(self):
            super().__init__()
            NonfiniteModel.trained = self

        def forward(self, rgb, *, precision="fp32"):
            result = super().forward(rgb, precision=precision)
            result["eventness_logit"] = result["eventness_logit"] * float("nan")
            return result

    monkeypatch.setattr(training, "TennisContactNet", NonfiniteModel)
    monkeypatch.setattr(training, "load_dataset", lambda *_args, **_kwargs: [sample])
    _fake_video_io(
        monkeypatch,
        window,
        lambda _video, indices: torch.full((len(indices), 3, 1, 1), 128, dtype=torch.uint8),
    )
    initial = NonfiniteModel()
    initial_state = {name: value.clone() for name, value in initial.state_dict().items()}
    initial_path = tmp_path / "initial.pt"
    torch.save({"model": initial_state}, initial_path)

    with pytest.raises(FloatingPointError, match="nonfinite"):
        training.train(
            "manifest.json", tmp_path / "run", initial_path, precision="fp32", batch_size=2,
        )

    assert all(
        torch.equal(value, initial_state[name])
        for name, value in NonfiniteModel.trained.state_dict().items()
    )
