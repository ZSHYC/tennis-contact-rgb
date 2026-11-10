import json

import pytest

from scripts.benchmark_inference import (
    compare_predictions,
    prediction_acceptance,
    read_quality_report,
)


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


def test_score_changes_need_a_passing_full_quality_report(tmp_path):
    changed = _prediction()
    changed["scores"][0][0] += 0.01
    differences = compare_predictions(_prediction(), changed)
    assert not prediction_acceptance(differences, None)

    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"status": "fail", "coverage": {
        "validation": 505, "holdout": 45, "official": 1622, "admin": 6,
    }}))
    assert not prediction_acceptance(differences, read_quality_report(quality)["pass"])

    quality.write_text(json.dumps({"status": "pass", "coverage": {
        "validation": 505, "holdout": 45, "official": 1622,
    }}))
    assert not prediction_acceptance(differences, read_quality_report(quality)["pass"])

    quality.write_text(json.dumps({"status": "pass", "coverage": {
        "validation": 505, "holdout": 45, "official": 1622, "admin": 6,
    }}))
    assert prediction_acceptance(differences, read_quality_report(quality)["pass"])


def test_quality_report_never_allows_pts_or_threshold_changes(tmp_path):
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"status": "pass", "coverage": {
        "validation": 505, "holdout": 45, "official": 1622, "admin": 6,
    }}))
    quality_pass = read_quality_report(quality)["pass"]

    changed_pts = _prediction()
    changed_pts["frame_times"][1] += 0.001
    assert not prediction_acceptance(compare_predictions(_prediction(), changed_pts), quality_pass)

    changed_threshold = _prediction()
    changed_threshold["thresholds"]["hit"] = 0.61
    assert not prediction_acceptance(compare_predictions(_prediction(), changed_threshold), quality_pass)
