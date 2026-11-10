#!/usr/bin/env python3
"""Compare CUDA inference precision modes on fixed event metrics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from tennis_contact.data import FrameReader, VideoSample, WindowSampler
from tennis_contact.evaluation import evaluate_predictions
from tennis_contact.inference import _LogitAccumulator, _packed_logits
from tennis_contact.model import load_model


ROOT = Path(__file__).resolve().parents[1]
MODE_SETTINGS = {
    "fp32": ("base", "fp32", False),
    "tf32": ("base", "fp32", True),
    "bf16": ("base", "bf16", False),
    "bf16_tf32": ("base", "bf16", True),
    "fp16": ("base", "fp16", False),
    "fused_fp32": ("fused", "fp32", False),
    "fused_fp16": ("fused", "fp16", False),
}
DEFAULT_MODES = ("fp32", "fused_fp32", "fused_fp16")
THRESHOLDS = {
    "primary": {"hit": 0.60, "bounce": 0.48},
    "secondary": {"hit": 0.67, "bounce": 0.57},
}
EXPECTED_COVERAGE = {"validation": 505, "holdout": 45, "official": 1622, "admin": 6}


def _identity(path: Path) -> dict[str, str | int]:
    path = path.resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rows(path: Path, axes: set[str]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a list")
    rows, seen = [], set()
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"id", "base", "axis", "source", "events"}:
            raise ValueError("each manifest row needs exactly id, base, axis, source and canonical events")
        sample_id, axis = str(row["id"]), str(row["axis"])
        base = Path(row["base"]).resolve()
        if (
            not sample_id or "/" in sample_id or "\\" in sample_id
            or sample_id in seen or axis not in EXPECTED_COVERAGE or not base.is_file()
            or not isinstance(row["events"], list)
        ):
            raise ValueError(f"invalid manifest row: {row}")
        seen.add(sample_id)
        if axis in axes:
            rows.append({
                "id": sample_id, "base": str(base), "axis": axis,
                "source": str(row["source"]), "events": row["events"],
            })
    if not rows:
        raise ValueError("selected axes contain no videos")
    return rows


def _forward_modes(models, rgb, modes, target):
    """Run one already-transferred window in deterministic mode order."""
    values, seconds = {}, {}
    for mode in modes:
        model_name, precision, tf32 = MODE_SETTINGS[mode]
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = perf_counter()
        output = models[model_name](rgb, precision=precision)
        packed = _packed_logits(output)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        seconds[mode] = perf_counter() - started
        if not bool(packed.isfinite().all()):
            raise FloatingPointError(f"{mode} produced nonfinite logits")
        values[mode] = packed
    return values, seconds


def _predict_modes(models, video: Path, frame_times: np.ndarray, modes, target):
    sampler = WindowSampler(frame_times)
    accumulators = {mode: _LogitAccumulator(frame_times) for mode in modes}
    timings = {mode: 0.0 for mode in modes}
    input_seconds = 0.0
    windows = 0
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        with FrameReader(max_cached_frames=56, pin_memory=target.type == "cuda") as reader, torch.inference_mode():
            for start in range(0, sampler.count, 100):
                window = sampler.window(start)
                if not window["core_mask"].any():
                    continue
                tick = perf_counter()
                rgb = reader.read(video, window["indices"]).to(target, non_blocking=True)
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                input_seconds += perf_counter() - tick
                packed, elapsed = _forward_modes(models, rgb, modes, target)
                for mode in modes:
                    accumulators[mode].add(window, packed[mode])
                    timings[mode] += elapsed[mode]
                windows += 1
                del rgb, packed
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
    scores = {}
    for mode, accumulator in accumulators.items():
        logits = torch.from_numpy(accumulator.finish())
        scores[mode] = (logits[:, :1].sigmoid() * logits[:, 1:].softmax(-1)).numpy()
    return scores, {"input_seconds": input_seconds, "mode_seconds": timings, "windows": windows}


def _sample(row) -> tuple[VideoSample, Path]:
    packet = torch.load(row["base"], map_location="cpu", weights_only=True)
    times = np.asarray(packet["frame_times"], dtype=np.float64)
    video = Path(packet["metadata"]["source"]["video"]["path"]).resolve()
    sample = VideoSample(row["id"], video, row["axis"], row["source"], list(row["events"]), times)
    return sample, video


def _metric_rows(report):
    yield "overall", report["overall"]
    for event_type, metrics in report["by_type"].items():
        yield f"by_type/{event_type}", metrics
    for source, values in report["by_source"].items():
        yield f"by_source/{source}/overall", values["overall"]
        for event_type, metrics in values["by_type"].items():
            yield f"by_source/{source}/by_type/{event_type}", metrics


def quality_differences(reference, candidate) -> list[dict]:
    """Report every count or P/R/F1 change, marking quality regressions."""
    left, right = dict(_metric_rows(reference)), dict(_metric_rows(candidate))
    if left.keys() != right.keys():
        raise ValueError("quality reports have different cells")
    differences = []
    for cell in left:
        delta = {
            key: right[cell][key] - left[cell][key]
            for key in ("tp", "fp", "fn", "precision", "recall", "f1", "mean_error", "fp_per_minute")
        }
        regressions = {
            key: value for key, value in delta.items()
            if (key in {"tp", "precision", "recall", "f1"} and value < 0)
            or (key in {"fp", "fn", "mean_error", "fp_per_minute"} and value > 0)
        }
        if any(value != 0 for value in delta.values()):
            differences.append({"cell": cell, "delta": delta, "regressions": regressions})
    return differences


def _summarize(rows, output: Path, modes, costs) -> dict:
    samples_by_axis, predictions = {}, {mode: {} for mode in modes}
    coverage = {axis: 0 for axis in EXPECTED_COVERAGE}
    completed = []
    for row in rows:
        path = output / "scores" / row["axis"] / f"{row['id']}.npz"
        if not path.is_file():
            continue
        sample, _video = _sample(row)
        with np.load(path) as saved:
            if not np.array_equal(saved["frame_times"], sample.frame_times):
                raise ValueError(f"saved frame_times changed: {row['id']}")
            for mode in modes:
                values = saved[mode]
                if values.shape != (len(sample.frame_times), 2) or not np.isfinite(values).all():
                    raise ValueError(f"invalid saved scores: {row['id']}:{mode}")
                predictions[mode][row["id"]] = values
        samples_by_axis.setdefault(row["axis"], []).append(sample)
        coverage[row["axis"]] += 1
        completed.append(row["id"])

    reports, differences, regressions = {}, [], []
    for axis, samples in samples_by_axis.items():
        reports[axis] = {}
        ids = {sample.id for sample in samples}
        for profile, thresholds in THRESHOLDS.items():
            reports[axis][profile] = {}
            for mode in modes:
                reports[axis][profile][mode] = evaluate_predictions(
                    samples, {sample_id: predictions[mode][sample_id] for sample_id in ids}, thresholds
                )
            reference = reports[axis][profile]["fp32"]
            for mode in modes:
                if mode == "fp32":
                    continue
                rows_diff = quality_differences(reference, reports[axis][profile][mode])
                for row_diff in rows_diff:
                    item = {"axis": axis, "profile": profile, "mode": mode, **row_diff}
                    differences.append(item)
                    if item["regressions"]:
                        regressions.append(item)
    complete = coverage == EXPECTED_COVERAGE
    return {
        "status": "pass" if complete and not regressions else "needs_decision" if complete else "partial",
        "coverage": coverage,
        "completed_ids": completed,
        "modes": list(modes),
        "thresholds": THRESHOLDS,
        "costs": costs,
        "reports": reports,
        "differences": differences,
        "regressions": regressions,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axes", nargs="+", choices=tuple(EXPECTED_COVERAGE), default=["validation"])
    parser.add_argument("--modes", nargs="+", choices=tuple(MODE_SETTINGS), default=list(DEFAULT_MODES))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/best.pt")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    modes = tuple(dict.fromkeys(arguments.modes))
    if "fp32" not in modes:
        parser.error("modes must include fp32 as the quality reference")
    if len(modes) < 2:
        parser.error("modes must include at least one candidate besides fp32")
    rows = _rows(arguments.manifest.resolve(), set(arguments.axes))
    arguments.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest": _identity(arguments.manifest), "checkpoint": _identity(arguments.checkpoint),
        "axes": list(dict.fromkeys(arguments.axes)), "modes": list(modes), "device": arguments.device,
    }
    metadata_path = arguments.output / "metadata.json"
    if metadata_path.is_file() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("output metadata differs; use another output directory")
    _write_json(metadata_path, metadata)
    progress_path = arguments.output / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file() else {"completed": [], "costs": {}}
    )

    torch.set_num_threads(4)
    target = torch.device(arguments.device)
    model, _model_metadata = load_model(arguments.checkpoint, "cpu")
    fused = copy.deepcopy(model).fuse_for_inference() if any(
        MODE_SETTINGS[mode][0] == "fused" for mode in modes
    ) else None
    models = {"base": model.to(target)}
    if fused is not None:
        models["fused"] = fused.to(target)
    for index, row in enumerate(rows, 1):
        destination = arguments.output / "scores" / row["axis"] / f"{row['id']}.npz"
        produced = not destination.is_file()
        if produced:
            destination.parent.mkdir(parents=True, exist_ok=True)
            sample, video = _sample(row)
            scores, cost = _predict_modes(models, video, sample.frame_times, modes, target)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, frame_times=sample.frame_times, **scores)
            temporary.replace(destination)
        else:
            cost = progress["costs"].get(row["id"], {"resumed": True})
        if produced or row["id"] not in progress["costs"]:
            progress["costs"][row["id"]] = cost
        if row["id"] not in progress["completed"]:
            progress["completed"].append(row["id"])
        _write_json(progress_path, progress)
        print(json.dumps({
            "video": index, "of": len(rows), "sample_id": row["id"], "axis": row["axis"],
            "cost": cost, "completed": len(progress["completed"]),
        }), flush=True)
    report = _summarize(rows, arguments.output, modes, progress["costs"])
    _write_json(arguments.output / "report.json", report)
    if report["status"] in {"pass", "needs_decision"}:
        _write_json(arguments.output / "quality-comparison.json", {
            "status": report["status"], "coverage": report["coverage"],
            "regressions": report["regressions"], "differences": report["differences"],
        })
    print(json.dumps({
        "status": report["status"], "coverage": report["coverage"],
        "regressions": len(report["regressions"]), "report": str(arguments.output / "report.json"),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
