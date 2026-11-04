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
    monkeypatch.setattr(training, "sample_window", window)
    monkeypatch.setattr(training, "read_frames", frames)

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
