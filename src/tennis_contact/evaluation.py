"""事件解码、一对一匹配与仅validation选阈值。"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import Any

import numpy as np


EVENT_TYPES = ("hit", "bounce")
PRIMARY_THRESHOLDS = {"hit": 0.60, "bounce": 0.48}
DEFAULT_THRESHOLD_CANDIDATES = tuple(value / 100 for value in range(1, 102))


def _scores(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1:] != (2,):
        raise ValueError("scores must be a numpy array with shape [N, 2]")
    if not np.isfinite(value).all():
        raise ValueError("scores must be finite")
    if np.any(value < 0) or np.any(value > 1):
        raise ValueError("scores must be probabilities in [0, 1]")
    return value


def _thresholds(value: dict[str, float]) -> dict[str, float]:
    if set(value) != set(EVENT_TYPES):
        raise ValueError("thresholds must contain exactly hit and bounce")
    result = {event_type: float(value[event_type]) for event_type in EVENT_TYPES}
    if not all(np.isfinite(threshold) and 0 <= threshold <= 1.01 for threshold in result.values()):
        raise ValueError("thresholds must be finite and between 0 and 1.01")
    return result


def _peak_indices(values: np.ndarray) -> list[int]:
    groups: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        score = float(value)
        if not groups or score != groups[-1][1]:
            groups.append((index, score))
    return [
        start
        for index, (start, score) in enumerate(groups)
        if score > 0
        and (index == 0 or score > groups[index - 1][1])
        and (index == len(groups) - 1 or score > groups[index + 1][1])
    ]


def decode_events(
    scores: np.ndarray,
    thresholds: dict[str, float],
    nms_radius: int = 5,
) -> list[dict[str, Any]]:
    """将单视频原生帧轴上的密集hit/bounce分数解码为事件。"""
    values = _scores(scores)
    limits = _thresholds(thresholds)
    if isinstance(nms_radius, bool) or not isinstance(nms_radius, int) or nms_radius < 0:
        raise ValueError("nms_radius must be a non-negative integer")

    events: list[dict[str, Any]] = []
    for column, event_type in enumerate(EVENT_TYPES):
        candidates = [
            (frame_number, float(values[frame_number, column]))
            for frame_number in _peak_indices(values[:, column])
            if values[frame_number, column] >= limits[event_type]
        ]
        kept: list[int] = []
        for frame_number, score in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if any(abs(frame_number - other) <= nms_radius for other in kept):
                continue
            kept.append(frame_number)
            events.append({"frame_number": frame_number, "event_type": event_type, "score": score})
    return sorted(events, key=lambda event: (event["frame_number"], event["event_type"]))


def _best(candidates):
    return max(candidates, key=lambda item: (item[0], -item[1]))


def _match_type(ground_truth, predictions, tolerance):
    empty = (0, 0, ())
    table = [[empty for _ in range(len(predictions) + 1)] for _ in range(len(ground_truth) + 1)]
    for row in range(1, len(ground_truth) + 1):
        for column in range(1, len(predictions) + 1):
            choices = [table[row - 1][column], table[row][column - 1]]
            error = abs(ground_truth[row - 1]["frame_number"] - predictions[column - 1]["frame_number"])
            if error <= tolerance:
                previous = table[row - 1][column - 1]
                choices.append(
                    (
                        previous[0] + 1,
                        previous[1] + error,
                        previous[2] + ((ground_truth[row - 1], predictions[column - 1], error),),
                    )
                )
            table[row][column] = _best(choices)
    return list(table[-1][-1][2])


def match_events(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    tolerance: int = 5,
) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    """对单视频先最大化命中数，再最小化总帧误差。"""
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer")
    matches = []
    for event_type in EVENT_TYPES:
        truth = sorted(
            (event for event in ground_truth if event["event_type"] == event_type),
            key=lambda event: event["frame_number"],
        )
        predicted = sorted(
            (event for event in predictions if event["event_type"] == event_type),
            key=lambda event: event["frame_number"],
        )
        matches.extend(_match_type(truth, predicted, tolerance))
    return matches


def _counts(tp: int, fp: int, fn: int, errors: list[int], minutes: float) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_error": float(np.mean(errors)) if errors else 0.0,
        "fp_per_minute": fp / minutes if minutes > 0 else 0.0,
    }


def _sample_minutes(sample) -> float:
    times = np.asarray(sample.frame_times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"frame_times must be finite and strictly increasing: {sample.id}")
    return float(times[-1] - times[0]) / 60 if len(times) > 1 else 0.0


def _report(records, minutes, tolerance):
    by_type = {}
    all_errors = []
    totals = [0, 0, 0]
    for event_type in EVENT_TYPES:
        truth = [event for _, gt, _ in records for event in gt if event["event_type"] == event_type]
        predicted = [event for _, _, pred in records for event in pred if event["event_type"] == event_type]
        matches = []
        for _, gt, pred in records:
            matches.extend(
                match_events(
                    [event for event in gt if event["event_type"] == event_type],
                    [event for event in pred if event["event_type"] == event_type],
                    tolerance,
                )
            )
        errors = [match[2] for match in matches]
        counts = (len(matches), len(predicted) - len(matches), len(truth) - len(matches))
        by_type[event_type] = _counts(*counts, errors, minutes)
        totals = [left + right for left, right in zip(totals, counts, strict=True)]
        all_errors.extend(errors)
    return _counts(*totals, all_errors, minutes), by_type


def evaluate_predictions(
    samples,
    predictions: dict[str, np.ndarray],
    thresholds: dict[str, float],
    tolerance: int = 5,
) -> dict[str, Any]:
    """对密集预测计算总体与各来源事件指标。"""
    sample_list = list(samples)
    sample_ids = [sample.id for sample in sample_list]
    if len(sample_ids) != len(set(sample_ids)) or set(predictions) != set(sample_ids):
        raise ValueError("predictions must contain exactly one entry for every sample id")

    records = []
    minutes_by_source: dict[str, float] = {}
    for sample in sample_list:
        minutes = _sample_minutes(sample)
        scores = _scores(predictions[sample.id])
        if len(scores) != len(sample.frame_times):
            raise ValueError(f"prediction length does not match frame_times: {sample.id}")
        records.append((sample.source, list(sample.events), decode_events(scores, thresholds)))
        minutes_by_source[sample.source] = minutes_by_source.get(sample.source, 0.0) + minutes

    overall, by_type = _report(records, sum(minutes_by_source.values()), tolerance)
    by_source = {}
    for source in sorted(minutes_by_source):
        source_overall, source_by_type = _report(
            [record for record in records if record[0] == source], minutes_by_source[source], tolerance
        )
        by_source[source] = {"overall": source_overall, "by_type": source_by_type}
    return {"overall": overall, "by_type": by_type, "by_source": by_source}


def search_thresholds(
    samples,
    predictions: dict[str, np.ndarray],
    candidates: Iterable[float] = DEFAULT_THRESHOLD_CANDIDATES,
) -> dict[str, float]:
    """仅在validation视频上选择hit/bounce联合工作点。"""
    sample_list = list(samples)
    if not sample_list or any(sample.split != "validation" for sample in sample_list):
        raise ValueError("threshold search requires validation samples only")
    values = tuple(dict.fromkeys(float(value) for value in candidates))
    if not values or any(not np.isfinite(value) or value < 0 or value > 1.01 for value in values):
        raise ValueError("threshold candidates must be finite values in [0, 1.01]")

    retained = {
        sample.id: decode_events(predictions[sample.id], {"hit": 0.0, "bounce": 0.0})
        for sample in sample_list
    }
    counts_by_type = {}
    for event_type in EVENT_TYPES:
        choices = []
        for threshold in values:
            tp = fp = fn = 0
            for sample in sample_list:
                truth = [event for event in sample.events if event["event_type"] == event_type]
                predicted = [
                    event
                    for event in retained[sample.id]
                    if event["event_type"] == event_type and event["score"] >= threshold
                ]
                matched = len(match_events(truth, predicted))
                tp += matched
                fp += len(predicted) - matched
                fn += len(truth) - matched
            choices.append((threshold, tp, fp, fn))
        counts_by_type[event_type] = choices

    ranked = []
    for hit, bounce in product(counts_by_type["hit"], counts_by_type["bounce"]):
        tp, fp, fn = hit[1] + bounce[1], hit[2] + bounce[2], hit[3] + bounce[3]
        f1 = 2 * tp / (2 * tp + fp + fn) if tp or fp or fn else 0.0
        ranked.append((f1, -fp, hit[0], bounce[0]))
    _, _, hit_threshold, bounce_threshold = max(ranked)
    return {"hit": hit_threshold, "bounce": bounce_threshold}


__all__ = [
    "DEFAULT_THRESHOLD_CANDIDATES",
    "PRIMARY_THRESHOLDS",
    "decode_events",
    "evaluate_predictions",
    "match_events",
    "search_thresholds",
]
