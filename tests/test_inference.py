import numpy as np
import pytest
import torch

from tennis_contact.inference import restore_logits


def test_restore_accepts_streams_without_retaining_windows():
    windows = [{"indices": np.array([0, 0, 2]), "core_mask": np.ones(3, dtype=bool)}]
    outputs = [{"eventness_logit": torch.tensor([1., 3., 8.]),
                "type_logits": torch.tensor([[2., 0.], [4., 2.], [9., 7.]])}]
    result = restore_logits(iter(windows), iter(outputs), np.array([0., .02, .06]))
    np.testing.assert_allclose(result, [[2., 3., 1.], [4., 5., 3.], [8., 9., 7.]])


def test_restore_averages_duplicate_native_frames_before_interpolation():
    # 重复采样若直接插值，会选中最后一次输出而非平均。
    windows = [{"indices": np.array([0, 0, 2]), "core_mask": np.ones(3, dtype=bool)}]
    outputs = [{"eventness_logit": torch.tensor([1., 3., 8.]),
                "type_logits": torch.tensor([[2., 0.], [4., 2.], [9., 7.]])}]
    result = restore_logits(windows, outputs, np.array([0., .02, .06]))
    np.testing.assert_allclose(result[:, 0], [2., 4., 8.])
    np.testing.assert_allclose(result[:, 1:], [[3., 1.], [5., 3.], [9., 7.]])


def test_restore_excludes_halo_outputs_and_extends_endpoint_values():
    windows = [{"indices": np.array([0, 1, 2]), "core_mask": np.array([False, True, False])}]
    outputs = [{"eventness_logit": torch.tensor([99., 2., 99.]),
                "type_logits": torch.tensor([[99., 99.], [3., 4.], [99., 99.]])}]
    result = restore_logits(windows, outputs, np.array([0., .03, .06]))
    np.testing.assert_array_equal(result, [[2., 3., 4.]] * 3)


@pytest.mark.parametrize("fps, count, logical_count, expected_windows", [(10, 110, 328, 4), (29.97, 100, 101, 1)])
def test_predict_prefetch_matches_sequential_windows_on_real_video(
    tmp_path, monkeypatch, fps, count, logical_count, expected_windows
):
    import cv2
    from tennis_contact import data
    from tennis_contact.evaluation import decode_events
    from tennis_contact.inference import predict_video

    monkeypatch.setattr(data, "FRAME_SIZE", (16, 24))
    video = tmp_path / "match.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (24, 16))
    assert writer.isOpened()
    for frame in range(count):
        writer.write(np.full((16, 24, 3), (frame * 11) % 256, dtype=np.uint8))
    writer.release()
    times = data.probe_video(video)

    class PixelModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = False
            self.precisions = []

        def fuse_for_inference(self):
            self.fused = True
            return self

        def forward(self, rgb, *, precision):
            self.precisions.append(precision)
            values = rgb.float().mean(dim=(1, 2, 3)) / 255
            # Include window context so changing halo or order changes the result.
            values = values + values.mean()
            return {"eventness_logit": values, "type_logits": torch.stack((values, -values), dim=1)}

    model = PixelModel()
    frames, logits = [], []
    for start in range(0, logical_count, 100):
        window = data.sample_window(times, start)
        if not window["core_mask"].any():
            continue
        output = model(data.read_frames(video, window["indices"]), precision="fp32")
        frames.extend(window["indices"][window["core_mask"]])
        logits.extend(torch.cat((output["eventness_logit"][:, None], output["type_logits"]), 1)[window["core_mask"]])
    unique, inverse, counts = np.unique(frames, return_inverse=True, return_counts=True)
    averaged = torch.zeros(len(unique), 3)
    averaged.index_add_(0, torch.from_numpy(inverse), torch.stack(logits))
    averaged /= torch.from_numpy(counts)[:, None]
    restored = torch.from_numpy(np.stack([
        np.interp(times, times[unique], column.numpy()) for column in averaged.T
    ], 1).astype(np.float32))
    expected = (restored[:, :1].sigmoid() * restored[:, 1:].softmax(-1)).numpy()
    thresholds = {"hit": .60, "bounce": .48}
    reads = []

    class CountingReader(data.FrameReader):
        def read(self, video, indices):
            reads.append(len(indices))
            return super().read(video, indices)

    monkeypatch.setattr(data, "FrameReader", CountingReader)
    result = predict_video(model, video, times, thresholds=thresholds)
    np.testing.assert_array_equal(result["scores"], expected)
    events = decode_events(expected, thresholds)
    for event in events:
        event["time_seconds"] = float(times[event["frame_number"]] - times[0])
    assert result["events"] == events
    assert model.fused
    assert model.precisions == ["fp32"] * (expected_windows * 2)
    assert result["cost"]["windows"] == expected_windows
    assert len(reads) == expected_windows
