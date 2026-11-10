import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tennis_contact import data
from tennis_contact.data import (
    FrameReader,
    WindowSampler,
    load_dataset,
    prepare_dataset,
    probe_video,
    read_frames,
    sample_window,
)


def _video(path: Path, count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for value in range(count):
        writer.write(np.full((24, 32, 3), (value * 50, 0, 255 - value * 50), np.uint8))
    writer.release()


def _reference_frames(path: Path, indices: np.ndarray) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    decoded = {}
    try:
        for index in np.unique(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            assert ok
            frame = cv2.resize(frame, (796, 448), interpolation=cv2.INTER_AREA)
            encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert encoded
            frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            decoded[int(index)] = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1)
    finally:
        capture.release()
    return torch.stack([decoded[int(index)] for index in indices])


def _reference_window(frame_times, start, core_steps, halo, rate):
    count = int(np.ceil((frame_times[-1] - frame_times[0]) * rate - 1e-8)) + 1
    stop = min(count, start + core_steps)
    logical = np.arange(start - halo, stop + halo, dtype=np.int64)
    offsets = logical / rate
    targets = frame_times[0] + offsets
    tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.abs(frame_times).max()), float(np.abs(offsets).max(initial=0.0))
    ) * 4.0
    valid = (targets >= frame_times[0] - tolerance) & (targets <= frame_times[-1] + tolerance)
    logical, targets = logical[valid], targets[valid]
    right = np.searchsorted(frame_times, targets).clip(max=len(frame_times) - 1)
    left = (right - 1).clip(min=0)
    indices = np.where(
        np.abs(frame_times[right] - targets) < np.abs(frame_times[left] - targets), right, left
    )
    full_logical = np.arange(count, dtype=np.int64)
    full_targets = frame_times[0] + full_logical / rate
    full_valid = (full_targets >= frame_times[0] - tolerance) & (full_targets <= frame_times[-1] + tolerance)
    full_right = np.searchsorted(frame_times, full_targets[full_valid]).clip(max=len(frame_times) - 1)
    full_left = (full_right - 1).clip(min=0)
    full_indices = np.where(
        np.abs(frame_times[full_right] - full_targets[full_valid])
        < np.abs(frame_times[full_left] - full_targets[full_valid]),
        full_right,
        full_left,
    )
    repeated = np.bincount(full_indices, minlength=len(frame_times))
    core_mask = (logical >= start) & (logical < stop)
    return {
        "indices": indices.astype(np.int64, copy=False),
        "logical_indices": logical,
        "times": targets,
        "core_mask": core_mask,
        "core_indices": np.flatnonzero(core_mask),
        "weights": core_mask.astype(np.float32) / np.maximum(repeated[indices], 1),
    }


def test_prepare_csv_load_and_read_jpeg95_frames(tmp_path):
    video = tmp_path / "match.MP4"
    _video(video)
    labels = tmp_path / "events.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_number", "label"])
        writer.writeheader()
        writer.writerows([{"frame_number": 1, "label": "near_hit"}, {"frame_number": 3, "label": "bounce"}])
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"samples": [{
        "id": "match", "video": video.name, "split": "train", "source": "real",
        "annotations": labels.name,
    }]}), encoding="utf-8")

    output = tmp_path / "prepared" / "dataset.json"
    result = prepare_dataset(source, output)
    samples = load_dataset(output, "train")
    assert result["frame_numbering"] == "zero_based"
    assert [(event["frame_number"], event["event_type"]) for event in samples[0].events] == [(1, "hit"), (3, "bounce")]
    assert len(samples[0].frame_times) == 4
    assert np.array_equal(probe_video(video), samples[0].frame_times)
    frames = read_frames(samples[0].video, np.array([2, 0, 2]))
    assert frames.shape == (3, 3, 448, 796)
    assert frames.dtype == torch.uint8
    assert torch.equal(frames[0], frames[2])


def test_frame_reader_reuses_processed_frames_without_changing_pixels(tmp_path):
    video = tmp_path / "frames.mp4"
    _video(video, count=12)
    with FrameReader(max_cached_frames=5) as reader:
        first = reader.read(video, np.array([0, 1, 2, 3, 4, 2]))
        capture = reader._capture
        second = reader.read(video, np.array([3, 4, 5, 6, 3]))
        backward = reader.read(video, np.array([1, 7, 1]))
        assert reader._capture is capture
        assert len(reader._frames) <= 5
    assert reader._capture is None
    assert not reader._frames

    assert torch.equal(first, _reference_frames(video, np.array([0, 1, 2, 3, 4, 2])))
    assert torch.equal(second, _reference_frames(video, np.array([3, 4, 5, 6, 3])))
    assert torch.equal(backward, _reference_frames(video, np.array([1, 7, 1])))


def test_frame_reader_seeks_late_random_requests_instead_of_scanning_from_zero(tmp_path, monkeypatch):
    videos = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    for video in videos:
        _video(video, count=20)
    expected = [
        _reference_frames(videos[0], np.array([18])),
        _reference_frames(videos[1], np.array([17])),
        _reference_frames(videos[0], np.array([19])),
    ]
    real_capture = cv2.VideoCapture
    captures = []

    class CaptureSpy:
        def __init__(self, path):
            self.capture = real_capture(path)
            self.grabs = 0
            self.seeks = []
            captures.append(self)

        def isOpened(self):
            return self.capture.isOpened()

        def set(self, prop, value):
            self.seeks.append((prop, value))
            return self.capture.set(prop, value)

        def grab(self):
            self.grabs += 1
            return self.capture.grab()

        def read(self):
            return self.capture.read()

        def release(self):
            self.capture.release()

    monkeypatch.setattr(cv2, "VideoCapture", CaptureSpy)
    with FrameReader(max_cached_frames=2) as reader:
        actual = [
            reader.read(videos[0], np.array([18])),
            reader.read(videos[1], np.array([17])),
            reader.read(videos[0], np.array([19])),
        ]

    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert sum(capture.grabs for capture in captures) == 0
    assert all(capture.seeks for capture in captures)


def test_labelme_and_explicit_one_based_csv(tmp_path):
    video = tmp_path / "match.mov"
    second_video = tmp_path / "second.mov"
    _video(video)
    _video(second_video)
    labelme = tmp_path / "labelme"
    labelme.mkdir()
    (labelme / "000002.json").write_text(json.dumps({"shapes": [{"label": "far_bounce"}]}), encoding="utf-8")
    csv_path = tmp_path / "one.csv"
    csv_path.write_text("frame_number,event_type\n1,hit\n", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"samples": [
        {"id": "labelme", "video": video.name, "split": "validation", "source": "x", "annotations": labelme.name},
        {"id": "csv", "video": second_video.name, "split": "holdout", "source": "x",
         "annotations": {"path": csv_path.name, "frame_numbering": "one_based"}},
    ]}), encoding="utf-8")
    output = tmp_path / "dataset.json"
    prepare_dataset(source, output)
    samples = load_dataset(output)
    assert samples[0].events == [{"frame_number": 2, "event_type": "bounce"}]
    assert samples[1].events == [{"frame_number": 0, "event_type": "hit"}]


def test_sample_window_matches_pts_axis_ties_duplicates_weights_and_edges():
    frame_times = np.array([10.0, 10.02, 10.08, 10.20])
    window = sample_window(frame_times, start=0, core_steps=6, halo=2, rate=25.0)
    assert window["logical_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    assert window["indices"].tolist() == [0, 1, 2, 2, 3, 3]
    assert window["times"].tolist() == pytest.approx([10, 10.04, 10.08, 10.12, 10.16, 10.20])
    assert window["core_mask"].tolist() == [True] * 6
    assert window["core_indices"].tolist() == list(range(6))
    assert window["weights"].tolist() == pytest.approx([1, 1, .5, .5, .5, .5])

    middle = sample_window(frame_times, start=3, core_steps=2, halo=2, rate=25.0)
    assert middle["logical_indices"].tolist() == [1, 2, 3, 4, 5]
    assert middle["core_mask"].tolist() == [False, False, True, True, False]
    assert middle["weights"].tolist() == pytest.approx([0, 0, .5, .5, 0])

    tied = sample_window(np.array([0.0, 0.1]), start=0, core_steps=2, halo=0, rate=20.0)
    assert tied["indices"].tolist() == [0, 0]

    sampler = WindowSampler(frame_times, rate=25.0)
    assert sampler.count == 6
    for start, core_steps, halo in ((0, 6, 2), (3, 2, 2)):
        expected = sample_window(frame_times, start, core_steps, halo, 25.0)
        actual = sampler.window(start, core_steps, halo)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_window_sampler_matches_original_algorithm_for_irregular_pts():
    rng = np.random.default_rng(20261110)
    frame_times = 1234.5 + np.cumsum(rng.uniform(0.005, 0.08, size=80))
    for rate in (7.5, 30.0, 1000.0):
        sampler = WindowSampler(frame_times, rate)
        for start in (0, 1, sampler.count // 2, sampler.count - 1):
            expected = _reference_window(frame_times, start, 13, 7, rate)
            actual = sampler.window(start, 13, 7)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])


def test_prepare_probes_each_shared_video_once(tmp_path, monkeypatch):
    video = tmp_path / "shared.mp4"
    _video(video)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"samples": [
        {"id": "a", "video": video.name, "split": "train", "events": []},
        {"id": "b", "video": video.name, "split": "train", "events": []},
    ]}), encoding="utf-8")
    calls = 0
    original = data.probe_video

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(data, "probe_video", counted)
    prepare_dataset(source, tmp_path / "dataset.json")
    assert calls == 1


def test_manifest_rejects_duplicate_ids_missing_split_and_visualized_video(tmp_path):
    video = tmp_path / "match.mp4"
    _video(video)
    for rows, message in [
        ([{"id": "x", "video": video.name, "events": []}], "needs split"),
        ([{"id": "x", "video": video.name, "split": "train", "events": []}] * 2, "duplicate"),
    ]:
        source = tmp_path / "bad.json"
        source.write_text(json.dumps({"samples": rows}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            prepare_dataset(source, tmp_path / "out.json")

    visualized = tmp_path / "match_visualized.mp4"
    visualized.write_bytes(video.read_bytes())
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"samples": [{
        "id": "x", "video": visualized.name, "split": "train", "events": [],
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="visualized"):
        prepare_dataset(source, tmp_path / "out.json")

    source.write_text(json.dumps({"samples": [
        {"id": "a", "video": video.name, "split": "train", "events": []},
        {"id": "b", "video": video.name, "split": "holdout", "events": []},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple splits"):
        prepare_dataset(source, tmp_path / "out.json")

    source.write_text(json.dumps({"samples": [{
        "id": "../escape", "video": video.name, "split": "train", "events": [],
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="sample id"):
        prepare_dataset(source, tmp_path / "out.json")

    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"samples": [{
        "id": "x", "video": video.name, "split": "train", "frame_times": [0.0],
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        load_dataset(prepared)

    prepared.write_text(json.dumps({"samples": [{
        "id": "x", "video": video.name, "split": "train", "frame_times": [[0.0]], "events": [],
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="frame_times"):
        load_dataset(prepared)
