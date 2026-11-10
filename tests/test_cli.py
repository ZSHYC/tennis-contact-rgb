import json

from tennis_contact.cli import main


def test_commands_have_help(capsys):
    for command in ([], ["prepare"], ["train"], ["predict"], ["evaluate"]):
        try:
            main([*command, "--help"])
        except SystemExit as error:
            assert error.code == 0
        assert "usage:" in capsys.readouterr().out


def test_write_json_keeps_chinese_and_rejects_nan(tmp_path):
    import pytest
    from tennis_contact.cli import write_json
    output = tmp_path / "result.json"
    write_json(output, {"模型": "网球"})
    assert json.loads(output.read_text())["模型"] == "网球"
    with pytest.raises(ValueError):
        write_json(output, {"bad": float("nan")})
    assert json.loads(output.read_text())["模型"] == "网球"
    assert not output.with_name(output.name + ".tmp").exists()


def test_predict_loads_checkpoint_on_cpu_before_inference(tmp_path, monkeypatch):
    import tennis_contact.data
    import tennis_contact.inference
    import tennis_contact.model

    model = object()
    calls = {}

    def load_model(checkpoint):
        calls["checkpoint"] = checkpoint
        return model, {"thresholds": {"hit": .6, "bounce": .48}}

    def predict_video(loaded, video, frame_times, *, device, thresholds):
        calls.update(model=loaded, video=video, frame_times=frame_times,
                     device=device, thresholds=thresholds)
        return {"cost": {}, "scores": [], "events": []}

    monkeypatch.setattr(tennis_contact.model, "load_model", load_model)
    monkeypatch.setattr(tennis_contact.data, "probe_video", lambda _video: [0.])
    monkeypatch.setattr(tennis_contact.inference, "predict_video", predict_video)
    output = tmp_path / "prediction.json"
    checkpoint = tmp_path / "model.pt"
    video = tmp_path / "video.mp4"

    assert main(["predict", "--video", str(video), "--checkpoint", str(checkpoint),
                 "--device", "cuda", "--output", str(output)]) == 0
    assert calls == {
        "checkpoint": checkpoint, "model": model, "video": video,
        "frame_times": [0.], "device": "cuda",
        "thresholds": {"hit": .6, "bounce": .48},
    }
