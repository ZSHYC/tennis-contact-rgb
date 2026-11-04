import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tennis_contact.data import load_dataset, prepare_dataset, probe_video, read_frames, sample_window


def _video(path: Path, count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for value in range(count):
        writer.write(np.full((24, 32, 3), (value * 50, 0, 255 - value * 50), np.uint8))
    writer.release()


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
