"""事件解码、一对一匹配与仅validation选阈值。"""

from __future__ import annotations

from collections.abc import Iterable
from collections import defaultdict
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
    if not len(values):
        return []
    starts = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1))
    grouped = values[starts]
    peaks = grouped > 0
    peaks[1:] &= grouped[1:] > grouped[:-1]
    peaks[:-1] &= grouped[:-1] > grouped[1:]
    return starts[peaks].tolist()


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
        occupied = np.zeros(len(values), dtype=bool)
        for frame_number, score in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if occupied[frame_number]:
                continue
            occupied[max(0, frame_number - nms_radius) : frame_number + nms_radius + 1] = True
            events.append({"frame_number": frame_number, "event_type": event_type, "score": score})
    return sorted(events, key=lambda event: (event["frame_number"], event["event_type"]))


def _better(matches: int, error: int, best_matches: int, best_error: int) -> bool:
    return matches > best_matches or matches == best_matches and error < best_error


def _match_component(ground_truth, predictions, tolerance):
    """Compact traceback with the legacy up, left, diagonal tie priority."""
    previous_matches = [0] * (len(predictions) + 1)
    previous_errors = [0] * (len(predictions) + 1)
    directions = [bytearray(len(predictions) + 1) for _ in range(len(ground_truth) + 1)]
    for row in range(1, len(ground_truth) + 1):
        current_matches = [0] * (len(predictions) + 1)
        current_errors = [0] * (len(predictions) + 1)
        for column in range(1, len(predictions) + 1):
            best_matches = previous_matches[column]
            best_error = previous_errors[column]
            direction = 1
            if _better(current_matches[column - 1], current_errors[column - 1], best_matches, best_error):
                best_matches = current_matches[column - 1]
                best_error = current_errors[column - 1]
                direction = 2
            error = abs(ground_truth[row - 1]["frame_number"] - predictions[column - 1]["frame_number"])
            if error <= tolerance:
                matches = previous_matches[column - 1] + 1
                total_error = previous_errors[column - 1] + error
                if _better(matches, total_error, best_matches, best_error):
                    best_matches, best_error, direction = matches, total_error, 3
            current_matches[column] = best_matches
            current_errors[column] = best_error
            directions[row][column] = direction
        previous_matches, previous_errors = current_matches, current_errors

    matches = []
    row, column = len(ground_truth), len(predictions)
    while row and column:
        direction = directions[row][column]
        if direction == 1:
            row -= 1
        elif direction == 2:
            column -= 1
        else:
            truth, predicted = ground_truth[row - 1], predictions[column - 1]
            matches.append((truth, predicted, abs(truth["frame_number"] - predicted["frame_number"])))
            row -= 1
            column -= 1
    matches.reverse()
    return matches


def _match_type(ground_truth, predictions, tolerance):
    """Split the tolerance graph, then solve each independent ordered component."""
    if not ground_truth or not predictions:
        return []
    parents = list(range(len(ground_truth) + len(predictions)))

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    edges = []
    first_prediction = 0
    for truth_index, truth in enumerate(ground_truth):
        frame = truth["frame_number"]
        while (
            first_prediction < len(predictions)
            and predictions[first_prediction]["frame_number"] < frame - tolerance
        ):
            first_prediction += 1
        prediction_index = first_prediction
        while (
            prediction_index < len(predictions)
            and predictions[prediction_index]["frame_number"] <= frame + tolerance
        ):
            union(truth_index, len(ground_truth) + prediction_index)
            edges.append((truth_index, prediction_index))
            prediction_index += 1

    components = defaultdict(lambda: (set(), set()))
    for truth_index, prediction_index in edges:
        truth_indices, prediction_indices = components[find(truth_index)]
        truth_indices.add(truth_index)
        prediction_indices.add(prediction_index)

    matches = []
    ordered = sorted(components.values(), key=lambda component: min(component[0]))
    for truth_indices, prediction_indices in ordered:
        truth = [ground_truth[index] for index in sorted(truth_indices)]
        predicted = [predictions[index] for index in sorted(prediction_indices)]
        matches.extend(_match_component(truth, predicted, tolerance))
    return matches


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


def _match_count(ground_truth, predictions, tolerance):
    """Maximum cardinality for ordered points; threshold search does not need a traceback."""
    truth_frames = sorted(event["frame_number"] for event in ground_truth)
    prediction_frames = sorted(event["frame_number"] for event in predictions)
    truth_index = prediction_index = matched = 0
    while truth_index < len(truth_frames) and prediction_index < len(prediction_frames):
        truth, predicted = truth_frames[truth_index], prediction_frames[prediction_index]
        if abs(truth - predicted) <= tolerance:
            matched += 1
            truth_index += 1
            prediction_index += 1
        elif truth < predicted:
            truth_index += 1
        else:
            prediction_index += 1
    return matched


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


def _report(records, minutes):
    by_type = {}
    all_errors = []
    totals = [0, 0, 0]
    for event_type in EVENT_TYPES:
        truth = sum(record[1][event_type][0] for record in records)
        predicted = sum(record[1][event_type][1] for record in records)
        errors = [error for record in records for error in record[1][event_type][2]]
        counts = (len(errors), predicted - len(errors), truth - len(errors))
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
        truth = list(sample.events)
        predicted = decode_events(scores, thresholds)
        matches = match_events(truth, predicted, tolerance)
        statistics = {}
        for event_type in EVENT_TYPES:
            statistics[event_type] = (
                sum(event["event_type"] == event_type for event in truth),
                sum(event["event_type"] == event_type for event in predicted),
                [error for gt, _prediction, error in matches if gt["event_type"] == event_type],
            )
        records.append((sample.source, statistics))
        minutes_by_source[sample.source] = minutes_by_source.get(sample.source, 0.0) + minutes

    overall, by_type = _report(records, sum(minutes_by_source.values()))
    by_source = {}
    for source in sorted(minutes_by_source):
        source_overall, source_by_type = _report(
            [record for record in records if record[0] == source], minutes_by_source[source]
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
                matched = _match_count(truth, predicted, 5)
                tp += matched
                fp += len(predicted) - matched
                fn += len(truth) - matched
            choices.append((threshold, tp, fp, fn))
        counts_by_type[event_type] = choices

    def rank(pair):
        hit, bounce = pair
        tp, fp, fn = hit[1] + bounce[1], hit[2] + bounce[2], hit[3] + bounce[3]
        return (2 * tp / (2 * tp + fp + fn) if tp or fp or fn else 0.0, -fp, hit[0], bounce[0])

    hit, bounce = max(product(counts_by_type["hit"], counts_by_type["bounce"]), key=rank)
    hit_threshold, bounce_threshold = hit[0], bounce[0]
    return {"hit": hit_threshold, "bounce": bounce_threshold}


__all__ = [
    "DEFAULT_THRESHOLD_CANDIDATES",
    "PRIMARY_THRESHOLDS",
    "decode_events",
    "evaluate_predictions",
    "match_events",
    "search_thresholds",
]
