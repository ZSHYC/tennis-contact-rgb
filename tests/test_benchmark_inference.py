import pytest

from scripts.benchmark_inference import compare_predictions


def _prediction():
    return {
        "frame_times": [0.0, 0.04],
        "scores": [[0.1, 0.2], [0.3, 0.4]],
        "events": [{"frame_number": 1, "event_type": "hit", "score": 0.3}],
        "thresholds": {"hit": 0.6, "bounce": 0.48},
    }


def test_prediction_comparison_detects_score_and_event_changes():
    expected = _prediction()
    assert compare_predictions(expected, _prediction()) == []

    changed_score = _prediction()
    changed_score["scores"][1][0] += 0.01
    score_mismatches = compare_predictions(expected, changed_score)
    assert score_mismatches[0]["field"] == "scores"
    assert score_mismatches[0]["max_absolute_difference"] == pytest.approx(0.01)

    changed_events = _prediction()
    changed_events["events"] = []
    mismatch = compare_predictions(expected, changed_events)[0]
    assert mismatch == {
        "field": "events", "expected_count": 1, "actual_count": 0,
        "first_difference": {"index": 0, "expected": expected["events"][0], "actual": None},
    }
