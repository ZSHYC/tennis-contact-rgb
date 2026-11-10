import json
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from tennis_contact import data


@pytest.mark.parametrize("codec,suffix,variable,cut", [
    ("libx264", ".mp4", True, False),
    ("libx265", ".mov", False, False),
    ("mpeg4", ".mp4", False, False),
    ("libx264", ".mp4", False, True),
])
def test_probe_uses_packet_pts_in_display_order_without_decoding(tmp_path, monkeypatch, codec, suffix, variable, cut):
    video = tmp_path / f"reordered{suffix}"
    command = [
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25:duration=1",
        "-vf", "setpts=(N+floor(N/5))/(25*TB)+0.4/TB" if variable else "setpts=PTS+0.4/TB",
        "-vsync", "vfr", "-c:v", codec, "-bf", "2", "-g", "12", "-threads", "1",
    ]
    if codec == "libx265":
        command += ["-x265-params", "pools=none:frame-threads=1:log-level=error"]
    subprocess.run([*command, str(video)], check=True, capture_output=True)
    if cut:
        clipped = tmp_path / "cut.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-ss", "0.24", "-i", str(video), "-t", "0.5",
            "-c:v", "copy", str(clipped),
        ], check=True, capture_output=True)
        video = clipped
    reference = json.loads(subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_packets",
        "-show_entries", "frame=best_effort_timestamp:packet=pts,flags:stream=time_base", "-of", "json", str(video),
    ], check=True, capture_output=True, text=True).stdout)
    rows = reference["packets_and_frames"]
    timestamps = [int(row["best_effort_timestamp"]) for row in rows if row["type"] == "frame"]
    packets = [int(row["pts"]) for row in rows if row["type"] == "packet"]
    assert packets != sorted(packets), "fixture must exercise B-frame presentation reordering"
    time_base = float(Fraction(reference["streams"][0]["time_base"]))
    expected = np.array([timestamp * time_base for timestamp in timestamps])
    if cut:
        assert any("D" in row["flags"] for row in rows if row["type"] == "packet")
        assert len(expected) < len(packets)
    else:
        assert len(expected) == 25 and expected[0] > 0
    if variable:
        assert len(set(np.diff(timestamps))) > 1

    run = subprocess.run

    def without_frame_decode(command, **kwargs):
        assert "-show_frames" not in command, "PTS probing must not decode the video"
        return run(command, **kwargs)

    monkeypatch.setattr(data.subprocess, "run", without_frame_decode)
    np.testing.assert_array_equal(data.probe_video(video), expected)
