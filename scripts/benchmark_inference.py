#!/usr/bin/env python3
"""Compare new-video inference against a frozen source tree."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def _shape(value):
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0] if value else None
    return shape


def _numbers(value):
    if isinstance(value, list):
        for child in value:
            yield from _numbers(child)
    else:
        yield float(value)


def compare_predictions(expected, actual):
    """Return compact, actionable differences in inference semantics."""
    mismatches = []
    for field in ("frame_times", "scores"):
        left, right = expected.get(field), actual.get(field)
        if left == right:
            continue
        mismatch = {"field": field}
        left_shape, right_shape = _shape(left), _shape(right)
        if left_shape != right_shape:
            mismatch.update(expected_shape=left_shape, actual_shape=right_shape)
        else:
            mismatch["max_absolute_difference"] = max(
                (abs(a - b) for a, b in zip(_numbers(left), _numbers(right), strict=True)),
                default=0.0,
            )
        mismatches.append(mismatch)
    if expected.get("events") != actual.get("events"):
        left, right = expected.get("events", []), actual.get("events", [])
        mismatch = {"field": "events", "expected_count": len(left), "actual_count": len(right)}
        first = next(
            (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
            min(len(left), len(right)) if len(left) != len(right) else None,
        )
        if first is not None:
            mismatch["first_difference"] = {
                "index": first,
                "expected": left[first] if first < len(left) else None,
                "actual": right[first] if first < len(right) else None,
            }
        mismatches.append(mismatch)
    if expected.get("thresholds") != actual.get("thresholds"):
        mismatches.append({
            "field": "thresholds",
            "expected": expected.get("thresholds"),
            "actual": actual.get("thresholds"),
        })
    return mismatches


def _manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("videos manifest needs a nonempty list")
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "video"}:
            raise ValueError("each video needs exactly id and video")
        sample_id = str(row["id"])
        video = Path(row["video"]).resolve()
        if not sample_id or sample_id in seen or not video.is_file():
            raise ValueError(f"invalid benchmark video: {row}")
        seen.add(sample_id)
        result.append({"id": sample_id, "source": sample_id, "video": video})
    return result


def _stdout_json(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "command_elapsed_seconds" in value:
            return value
    raise RuntimeError(f"predict did not print its cost JSON: {stdout[-1000:]}")


def _run(source, row, output, device):
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source / "src") + (os.pathsep + inherited if inherited else "")
    command = [
        sys.executable, "-m", "tennis_contact", "--threads", "4", "predict",
        "--video", str(row["video"]), "--checkpoint", str(ROOT / "weights" / "best.pt"),
        "--device", device, "--output", str(output),
    ]
    started = perf_counter()
    process = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    elapsed = perf_counter() - started
    if process.returncode:
        raise RuntimeError(
            f"predict failed for {row['id']} from {source}: exit={process.returncode}\n"
            f"stdout={process.stdout[-2000:]}\nstderr={process.stderr[-2000:]}"
        )
    prediction = json.loads(output.read_text(encoding="utf-8"))
    cost = _stdout_json(process.stdout)
    return prediction, {
        "external_wall_seconds": elapsed,
        "command_elapsed_seconds": float(cost["command_elapsed_seconds"]),
        "prediction_wall_seconds": float(cost["wall_seconds"]),
        "read_seconds": float(cost["read_seconds"]),
        "forward_seconds": float(cost["forward_seconds"]),
        "restore_and_decode_seconds": float(cost["restore_and_decode_seconds"]),
        "peak_allocated_bytes": int(cost["peak_allocated_bytes"]),
        "native_frames": int(cost["native_frames"]),
        "windows": int(cost["windows"]),
    }


def _stats(values):
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _summarize(measurements, repeats):
    by_source = defaultdict(lambda: defaultdict(list))
    by_video = defaultdict(lambda: defaultdict(list))
    suite = defaultdict(lambda: [0.0] * repeats)
    command_suite = defaultdict(lambda: [0.0] * repeats)
    peaks = defaultdict(int)
    for row in measurements:
        implementation = row["implementation"]
        by_source[implementation][row["source"]].append(row["external_wall_seconds"])
        by_video[implementation][row["id"]].append(row["external_wall_seconds"])
        suite[implementation][row["repeat"]] += row["external_wall_seconds"]
        command_suite[implementation][row["repeat"]] += row["command_elapsed_seconds"]
        peaks[implementation] = max(peaks[implementation], row["peak_allocated_bytes"])
    return {
        implementation: {
            "suite_external_wall_seconds": _stats(suite[implementation]),
            "suite_command_elapsed_seconds": _stats(command_suite[implementation]),
            "by_source_external_wall_seconds": {
                source: _stats(values) for source, values in sorted(by_source[implementation].items())
            },
            "by_video_external_wall_seconds": {
                sample_id: _stats(values) for sample_id, values in sorted(by_video[implementation].items())
            },
            "peak_allocated_bytes": peaks[implementation],
        }
        for implementation in ("baseline", "current")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    if arguments.repeats < 2 or not (arguments.baseline / "src" / "tennis_contact").is_dir():
        parser.error("repeats must be at least 2 and baseline must contain src/tennis_contact")
    rows = _manifest(arguments.videos)
    sources = {"baseline": arguments.baseline.resolve(), "current": ROOT}
    predictions = defaultdict(lambda: defaultdict(list))
    measurements = []
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tennis-contact-benchmark-") as temporary:
        temporary = Path(temporary)
        # Warm both implementations before measured runs to remove one-time driver and filesystem startup.
        for implementation in ("baseline", "current"):
            prediction, _cost = _run(
                sources[implementation], rows[0], temporary / f"warm-{implementation}.json", arguments.device
            )
            predictions[rows[0]["id"]][implementation].append(prediction)
        for repeat in range(arguments.repeats):
            for video_index, row in enumerate(rows):
                order = ("baseline", "current") if (repeat + video_index) % 2 == 0 else ("current", "baseline")
                for implementation in order:
                    prediction, cost = _run(
                        sources[implementation], row,
                        temporary / f"{repeat}-{row['id']}-{implementation}.json", arguments.device,
                    )
                    predictions[row["id"]][implementation].append(prediction)
                    measurements.append({
                        "implementation": implementation, "repeat": repeat,
                        "id": row["id"], "source": row["source"], **cost,
                    })

    mismatches = []
    for row in rows:
        sample_id = row["id"]
        reference = predictions[sample_id]["baseline"][0]
        for implementation in ("baseline", "current"):
            for run, prediction in enumerate(predictions[sample_id][implementation]):
                differences = compare_predictions(reference, prediction)
                if differences:
                    mismatches.append({
                        "id": sample_id, "implementation": implementation,
                        "run": run, "differences": differences,
                    })
    summary = _summarize(measurements, arguments.repeats)
    baseline_suite = summary["baseline"]["suite_external_wall_seconds"]
    current_suite = summary["current"]["suite_external_wall_seconds"]
    source_regressions = []
    for source, current in summary["current"]["by_source_external_wall_seconds"].items():
        baseline = summary["baseline"]["by_source_external_wall_seconds"][source]
        if current["median"] > baseline["median"] * 1.02:
            source_regressions.append({
                "source": source,
                "baseline_median": baseline["median"],
                "current_median": current["median"],
            })
    semantic_pass = not mismatches
    performance_pass = (
        current_suite["maximum"] < baseline_suite["minimum"] and not source_regressions
    )
    report = {
        "status": "pass" if semantic_pass and performance_pass else "fail",
        "semantic_pass": semantic_pass,
        "performance_pass": performance_pass,
        "performance_contract": {
            "suite": "current maximum repeat wall must be below baseline minimum repeat wall",
            "sources": "no source median may regress by more than 2 percent",
        },
        "speedup": baseline_suite["median"] / current_suite["median"],
        "mismatches": mismatches,
        "source_regressions": source_regressions,
        "summary": summary,
        "measurements": measurements,
    }
    if not math.isfinite(report["speedup"]):
        raise RuntimeError("benchmark produced a nonfinite speedup")
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "speedup": report["speedup"],
        "semantic_pass": semantic_pass, "performance_pass": performance_pass,
        "output": str(arguments.output),
    }, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
