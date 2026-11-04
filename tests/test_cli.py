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
