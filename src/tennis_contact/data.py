from __future__ import annotations

import csv
from collections import OrderedDict
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


FRAME_SIZE = (448, 796)
_SPLITS = {"train", "validation", "holdout"}
_LABELS = {
    "hit": "hit",
    "near_hit": "hit",
    "far_hit": "hit",
    "bounce": "bounce",
    "near_bounce": "bounce",
    "far_bounce": "bounce",
    "net": None,
    "near_net": None,
    "far_net": None,
}


@dataclass(frozen=True)
class VideoSample:
    id: str
    video: Path
    split: str
    source: str
    events: list[dict]
    frame_times: np.ndarray


def _video_path(path: Path) -> Path:
    if path.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError(f"unsupported video extension: {path.suffix}")
    if "visualization" in {part.lower() for part in path.parts} or "_visualized" in path.stem.lower():
        raise ValueError(f"visualized video is not model input: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def probe_video(video: Path) -> np.ndarray:
    video = _video_path(Path(video).resolve())
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp:stream=time_base", "-of", "json", str(video),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "")
        raise ValueError(f"ffprobe failed for {video}: {detail}") from exc
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        if len(streams) != 1:
            raise ValueError
        time_base = float(Fraction(streams[0]["time_base"]))
        values = [int(row["best_effort_timestamp"]) * time_base for row in payload["frames"]]
    except (KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe returned invalid frame PTS: {video}") from exc
    times = np.asarray(values, dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"video PTS must be finite and strictly increasing: {video}")
    return times


def _event(frame: object, label: object, one_based: bool, origin: str) -> dict | None:
    try:
        frame_number = int(frame) - int(one_based)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid event frame: {origin}") from exc
    event_type = _LABELS.get(str(label).strip().lower())
    if str(label).strip().lower() not in _LABELS:
        raise ValueError(f"unsupported event label {label!r}: {origin}")
    if frame_number < 0:
        raise ValueError(f"event frame must be zero-based and non-negative: {origin}")
    return None if event_type is None else {"frame_number": frame_number, "event_type": event_type}


def _load_annotations(spec: object, base: Path) -> list[dict]:
    if isinstance(spec, list):
        rows = [
            _event(row.get("frame_number"), row.get("event_type", row.get("label")), False, "inline event")
            for row in spec
        ]
    else:
        options = {"path": spec} if isinstance(spec, str) else spec
        if not isinstance(options, dict) or "path" not in options:
            raise ValueError("annotations must be an event list, path, or {path, frame_numbering}")
        numbering = options.get("frame_numbering", "zero_based")
        if numbering not in {"zero_based", "one_based"}:
            raise ValueError("frame_numbering must be zero_based or one_based")
        one_based = numbering == "one_based"
        path = (base / str(options["path"])).resolve()
        rows = []
        if path.is_file() and path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                label_field = "event_type" if "event_type" in fields else "label"
                if "frame_number" not in fields or label_field not in fields:
                    raise ValueError(f"annotation CSV needs frame_number and event_type/label: {path}")
                rows.extend(
                    _event(row["frame_number"], row[label_field], one_based, f"{path}:{line}")
                    for line, row in enumerate(reader, 2)
                )
        elif path.is_dir():
            for file in sorted(path.glob("*.json")):
                match = re.match(r"(\d+)", file.stem)
                if not match:
                    raise ValueError(f"LabelMe filename must start with a frame number: {file}")
                payload = json.loads(file.read_text(encoding="utf-8"))
                rows.extend(
                    _event(match.group(1), shape.get("label"), one_based, str(file))
                    for shape in payload.get("shapes", [])
                )
        else:
            raise ValueError(f"unsupported annotations: {path}")
    events = [row for row in rows if row is not None]
    keys = [(row["frame_number"], row["event_type"]) for row in events]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate normalized event")
    by_frame: dict[int, set[str]] = {}
    for frame, event_type in keys:
        by_frame.setdefault(frame, set()).add(event_type)
    if any(len(types) > 1 for types in by_frame.values()):
        raise ValueError("a frame cannot be both hit and bounce")
    return sorted(events, key=lambda row: (row["frame_number"], row["event_type"]))


def _read_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("dataset manifest needs a samples list")
    return rows


def prepare_dataset(input_manifest: Path, output: Path) -> dict:
    input_manifest, output = Path(input_manifest).resolve(), Path(output).resolve()
    prepared, seen, video_splits, frame_times = [], set(), {}, {}
    for row in _read_manifest(input_manifest):
        if not isinstance(row, dict):
            raise ValueError("each sample must be an object")
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or "/" in sample_id or "\\" in sample_id or sample_id in seen:
            raise ValueError(f"missing or duplicate sample id: {sample_id!r}")
        seen.add(sample_id)
        split = row.get("split")
        if split not in _SPLITS:
            raise ValueError(f"sample {sample_id} needs split train/validation/holdout")
        if "events" not in row and "annotations" not in row:
            raise ValueError(f"sample {sample_id} needs labels for training/evaluation")
        video = _video_path((input_manifest.parent / str(row.get("video", ""))).resolve())
        if video in video_splits and video_splits[video] != split:
            raise ValueError(f"video appears in multiple splits: {video}")
        video_splits[video] = split
        times = frame_times.get(video)
        if times is None:
            times = frame_times[video] = probe_video(video)
        events = _load_annotations(row.get("events", row.get("annotations")), input_manifest.parent)
        if events and events[-1]["frame_number"] >= len(times):
            raise ValueError(f"event frame exceeds video: {sample_id}:{events[-1]['frame_number']}")
        prepared.append({
            "id": sample_id,
            "video": os.path.relpath(video, output.parent),
            "split": split,
            "source": str(row.get("source", "")),
            "events": events,
            "frame_times": times.tolist(),
        })
    result = {"schema": 1, "frame_numbering": "zero_based", "samples": prepared}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def load_dataset(manifest: Path, split: str | None = None) -> list[VideoSample]:
    manifest = Path(manifest).resolve()
    if split is not None and split not in _SPLITS:
        raise ValueError(f"unknown split: {split}")
    samples, seen, video_splits = [], set(), {}
    for row in _read_manifest(manifest):
        if not isinstance(row, dict):
            raise ValueError("each sample must be an object")
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or "/" in sample_id or "\\" in sample_id or sample_id in seen:
            raise ValueError(f"missing or duplicate sample id: {sample_id!r}")
        seen.add(sample_id)
        row_split = row.get("split")
        if row_split not in _SPLITS:
            raise ValueError(f"sample {sample_id} needs split train/validation/holdout")
        video = _video_path((manifest.parent / str(row.get("video", ""))).resolve())
        if video in video_splits and video_splits[video] != row_split:
            raise ValueError(f"video appears in multiple splits: {video}")
        video_splits[video] = row_split
        if split is not None and row_split != split:
            continue
        times = np.asarray(row.get("frame_times", []), dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"invalid frame_times: {sample_id}")
        if "events" not in row:
            raise ValueError(f"sample {sample_id} needs labels for training/evaluation")
        events = _load_annotations(row["events"], manifest.parent)
        if events and events[-1]["frame_number"] >= len(times):
            raise ValueError(f"event frame exceeds video: {sample_id}:{events[-1]['frame_number']}")
        samples.append(VideoSample(sample_id, video, row_split, str(row.get("source", "")), events, times))
    return samples


class FrameReader:
    """读取并有界复用已完成模型像素处理的帧。"""

    # 超过完整模型窗口的间隔直接 seek，避免随机请求线性扫描。
    _MAX_SEQUENTIAL_GAP = 156

    def __init__(self, max_cached_frames: int = 256, *, pin_memory: bool = False) -> None:
        if isinstance(max_cached_frames, bool) or not isinstance(max_cached_frames, int) or max_cached_frames < 1:
            raise ValueError("max_cached_frames must be a positive integer")
        self.max_cached_frames = max_cached_frames
        self.pin_memory = pin_memory
        self._frames: OrderedDict[tuple[Path, int], torch.Tensor] = OrderedDict()
        self._capture = None
        self._video: Path | None = None
        self._next_index = 0

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._video = None
        self._next_index = 0
        self._frames.clear()

    def _open(self, video: Path) -> bool:
        import cv2

        if self._video == video and self._capture is not None:
            return False
        if self._capture is not None:
            self._capture.release()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"cannot open video: {video}")
        self._capture, self._video, self._next_index = capture, video, 0
        return True

    def _decode(self, video: Path, index: int) -> torch.Tensor:
        import cv2

        opened = self._open(video)
        if index < self._next_index or (
            index > self._next_index
            and (opened or index - self._next_index > self._MAX_SEQUENTIAL_GAP)
        ):
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next_index = index
        while self._next_index < index:
            if not self._capture.grab():
                raise ValueError(f"cannot advance to frame {self._next_index}: {video}")
            self._next_index += 1
        ok, frame = self._capture.read()
        if not ok:
            raise ValueError(f"cannot decode frame {index}: {video}")
        self._next_index = index + 1
        frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]), interpolation=cv2.INTER_AREA)
        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR) if encoded else None
        if frame is None:
            raise ValueError(f"JPEG95 roundtrip failed for frame {index}: {video}")
        return torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1)

    def read(self, video: Path, frame_indices: np.ndarray) -> torch.Tensor:
        video = _video_path(Path(video).resolve())
        indices = np.asarray(frame_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
            raise ValueError("frame_indices must be a one-dimensional non-negative integer array")
        if not len(indices):
            return torch.empty((0, 3, *FRAME_SIZE), dtype=torch.uint8)

        requested = list(map(int, np.unique(indices)))
        decoded: dict[int, torch.Tensor] = {}
        for index in requested:
            key = (video, index)
            frame = self._frames.get(key)
            if frame is not None:
                self._frames.move_to_end(key)
                decoded[index] = frame
        for index in requested:
            if index in decoded:
                continue
            frame = self._decode(video, index)
            self._frames[(video, index)] = decoded[index] = frame
            if len(self._frames) > self.max_cached_frames:
                self._frames.popitem(last=False)
        # TCHW API with CTHW storage makes the model's NCTHW view contiguous.
        window = torch.empty(
            (3, len(indices), *FRAME_SIZE), dtype=torch.uint8, pin_memory=self.pin_memory
        ).permute(1, 0, 2, 3)
        return torch.stack([decoded[int(index)] for index in indices], out=window)


def read_frames(video: Path, frame_indices: np.ndarray) -> torch.Tensor:
    with FrameReader() as reader:
        return reader.read(video, frame_indices)


def _nearest(frame_times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(frame_times, targets).clip(max=len(frame_times) - 1)
    left = (right - 1).clip(min=0)
    choose_right = np.abs(frame_times[right] - targets) < np.abs(frame_times[left] - targets)
    return np.where(choose_right, right, left)


def _logical_targets(frame_times: np.ndarray, logical: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    offsets = logical / rate
    targets = frame_times[0] + offsets
    tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.abs(frame_times).max()), float(np.abs(offsets).max(initial=0.0))
    ) * 4.0
    valid = (targets >= frame_times[0] - tolerance) & (targets <= frame_times[-1] + tolerance)
    return targets, valid


class WindowSampler:
    """一次构造视频的30Hz到原生帧映射，再切出各窗口。"""

    def __init__(self, frame_times: np.ndarray, rate: float = 30.0) -> None:
        times = np.asarray(frame_times, dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("frame_times must be finite, one-dimensional, and strictly increasing")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        self.count = math.ceil(float(times[-1] - times[0]) * rate - 1e-8) + 1
        self._logical = np.arange(self.count, dtype=np.int64)
        self._targets, valid = _logical_targets(times, self._logical, rate)
        self._logical, self._targets = self._logical[valid], self._targets[valid]
        self._indices = _nearest(times, self._targets).astype(np.int64, copy=False)
        self._repeated = np.bincount(self._indices, minlength=len(times))

    def window(self, start: int, core_steps: int = 100, halo: int = 28) -> dict[str, np.ndarray]:
        if not isinstance(start, (int, np.integer)) or start < 0 or core_steps < 1 or halo < 0:
            raise ValueError("start/halo must be non-negative integers and core_steps must be positive")
        if start >= self.count:
            raise ValueError("window start is outside the logical video axis")
        stop = min(self.count, start + core_steps)
        first = np.searchsorted(self._logical, start - halo, side="left")
        last = np.searchsorted(self._logical, stop + halo, side="left")
        logical = self._logical[first:last]
        targets = self._targets[first:last]
        indices = self._indices[first:last]
        core_mask = (logical >= start) & (logical < stop)
        weights = core_mask.astype(np.float32) / np.maximum(self._repeated[indices], 1)
        return {
            "indices": indices,
            "logical_indices": logical,
            "times": targets,
            "core_mask": core_mask,
            "core_indices": np.flatnonzero(core_mask),
            "weights": weights,
        }


def sample_window(
    frame_times: np.ndarray, start: int, core_steps: int = 100, halo: int = 28, rate: float = 30.0
) -> dict[str, np.ndarray]:
    return WindowSampler(frame_times, rate).window(start, core_steps, halo)
