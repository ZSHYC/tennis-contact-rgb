from types import SimpleNamespace
import random

import numpy as np
import pytest

import tennis_contact.evaluation as evaluation

from tennis_contact.evaluation import (
    PRIMARY_THRESHOLDS,
    decode_events,
    evaluate_predictions,
    match_events,
    search_thresholds,
)


def sample(sample_id, source, split, events, frame_count=20):
    return SimpleNamespace(
        id=sample_id,
        source=source,
        split=split,
        events=events,
        frame_times=np.arange(frame_count, dtype=np.float64) / 10,
    )


def test_decode_uses_left_edge_of_wide_plateau_and_primary_thresholds():
    scores = np.zeros((9, 2), dtype=np.float64)
    scores[2:6, 0] = 0.8
    scores[7, 1] = 0.5

    assert PRIMARY_THRESHOLDS == {"hit": 0.60, "bounce": 0.48}
    assert decode_events(scores, PRIMARY_THRESHOLDS) == [
        {"frame_number": 2, "event_type": "hit", "score": 0.8},
        {"frame_number": 7, "event_type": "bounce", "score": 0.5},
    ]


def test_decode_nms_breaks_equal_score_ties_by_earlier_frame():
    scores = np.zeros((10, 2), dtype=np.float64)
    scores[[2, 6], 0] = 0.9

    assert decode_events(scores, {"hit": 0.5, "bounce": 0.5}) == [
        {"frame_number": 2, "event_type": "hit", "score": 0.9}
    ]


def test_match_events_never_crosses_types_and_maximizes_cardinality_before_error():
    hit_zero = {"frame_number": 0, "event_type": "hit"}
    hit_two = {"frame_number": 2, "event_type": "hit"}
    bounce = {"frame_number": 0, "event_type": "bounce"}
    pred_one = {"frame_number": 1, "event_type": "hit", "score": 0.9}
    pred_zero = {"frame_number": 0, "event_type": "hit", "score": 0.8}

    matches = match_events([hit_zero, hit_two, bounce], [pred_one, pred_zero], tolerance=1)

    assert matches == [(hit_zero, pred_zero, 0), (hit_two, pred_one, 1)]


def _legacy_match_type(ground_truth, predictions, tolerance):
    empty = (0, 0, ())
    table = [[empty for _ in range(len(predictions) + 1)] for _ in range(len(ground_truth) + 1)]
    for row in range(1, len(ground_truth) + 1):
        for column in range(1, len(predictions) + 1):
            choices = [table[row - 1][column], table[row][column - 1]]
            error = abs(ground_truth[row - 1]["frame_number"] - predictions[column - 1]["frame_number"])
            if error <= tolerance:
                previous = table[row - 1][column - 1]
                choices.append((
                    previous[0] + 1,
                    previous[1] + error,
                    previous[2] + ((ground_truth[row - 1], predictions[column - 1], error),),
                ))
            table[row][column] = max(choices, key=lambda item: (item[0], -item[1]))
    return list(table[-1][-1][2])


def test_match_events_preserves_legacy_tie_breaking_on_random_inputs():
    rng = random.Random(7)
    for _ in range(100):
        truth = [
            {"frame_number": rng.randrange(20), "event_type": "hit", "id": index}
            for index in range(rng.randrange(8))
        ]
        predicted = [
            {"frame_number": rng.randrange(20), "event_type": "hit", "score": rng.random(), "id": index}
            for index in range(rng.randrange(8))
        ]
        expected = _legacy_match_type(
            sorted(truth, key=lambda event: event["frame_number"]),
            sorted(predicted, key=lambda event: event["frame_number"]),
            2,
        )
        assert match_events(truth, predicted, tolerance=2) == expected
        assert evaluation._match_count(truth, predicted, tolerance=2) == len(expected)


def test_evaluate_aggregates_sources_and_handles_a_source_with_no_ground_truth():
    samples = [
        sample("a", "broadcast", "holdout", [{"frame_number": 2, "event_type": "hit"}], 11),
        sample("b", "phone", "holdout", [], 11),
    ]
    scores_a = np.zeros((11, 2), dtype=np.float64)
    scores_b = np.zeros((11, 2), dtype=np.float64)
    scores_a[2, 0] = 0.9
    scores_b[8, 1] = 0.8

    report = evaluate_predictions(
        samples,
        {"a": scores_a, "b": scores_b},
        {"hit": 0.6, "bounce": 0.48},
        tolerance=0,
    )

    assert report["overall"]["tp"] == 1
    assert report["overall"]["fp"] == 1
    assert report["overall"]["fn"] == 0
    assert report["overall"]["f1"] == pytest.approx(2 / 3)
    assert report["overall"]["mean_error"] == 0.0
    assert report["by_type"]["hit"]["tp"] == 1
    assert report["by_type"]["bounce"]["fp"] == 1
    assert report["by_source"]["broadcast"]["overall"]["tp"] == 1
    assert report["by_source"]["phone"]["overall"]["fp"] == 1


def test_evaluate_honors_the_requested_matching_tolerance():
    video = sample("v", "broadcast", "holdout", [{"frame_number": 2, "event_type": "hit"}], 6)
    scores = np.zeros((6, 2), dtype=np.float64)
    scores[3, 0] = 0.9

    report = evaluate_predictions([video], {"v": scores}, PRIMARY_THRESHOLDS, tolerance=0)

    assert report["overall"]["tp"] == 0
    assert report["overall"]["fp"] == 1
    assert report["overall"]["fn"] == 1


def test_evaluate_matches_each_video_once_before_aggregating(monkeypatch):
    samples = [
        sample("a", "same", "holdout", [{"frame_number": 2, "event_type": "hit"}], 10),
        sample("b", "same", "holdout", [{"frame_number": 4, "event_type": "bounce"}], 10),
    ]
    predictions = {sample.id: np.zeros((10, 2), dtype=np.float64) for sample in samples}
    calls = 0
    original = evaluation.match_events

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "match_events", counted)
    evaluate_predictions(samples, predictions, PRIMARY_THRESHOLDS)

    assert calls == len(samples)


def test_search_thresholds_uses_joint_f1_not_independent_class_optima():
    events = [{"frame_number": frame, "event_type": "hit"} for frame in range(0, 100, 10)]
    events.append({"frame_number": 110, "event_type": "bounce"})
    validation = sample("v", "broadcast", "validation", events, 210)
    scores = np.zeros((210, 2), dtype=np.float64)
    scores[range(0, 100, 10), 0] = 0.9
    scores[110, 1] = 0.4
    scores[range(120, 210, 10), 1] = 0.4

    assert search_thresholds([validation], {"v": scores}, candidates=(0.3, 0.7)) == {
        "hit": 0.7,
        "bounce": 0.7,
    }


def test_search_thresholds_rejects_non_validation_samples():
    holdout = sample("h", "phone", "holdout", [], 2)

    with pytest.raises(ValueError, match="validation"):
        search_thresholds([holdout], {"h": np.zeros((2, 2))}, candidates=(0.5,))


@pytest.mark.parametrize(
    "call",
    [
        lambda: decode_events(np.array([[np.nan, 0.0]]), PRIMARY_THRESHOLDS),
        lambda: evaluate_predictions(
            [sample("v", "phone", "holdout", [], 1)],
            {"v": np.array([[0.0, np.nan]])},
            PRIMARY_THRESHOLDS,
        ),
    ],
)
def test_non_finite_scores_are_rejected(call):
    with pytest.raises(ValueError, match="finite"):
        call()
