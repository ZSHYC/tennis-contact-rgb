"""原视频逐窗推理，以及完整原生时间轴还原。"""

from time import perf_counter

import numpy as np
import torch


class _LogitAccumulator:
    def __init__(self, frame_times):
        self.times = np.asarray(frame_times, dtype=np.float64)
        self.sums = torch.zeros(len(self.times), 3, dtype=torch.float32)
        self.counts = np.zeros(len(self.times), dtype=np.int64)

    def add(self, window, values):
        frames = window["indices"][window["core_mask"]]
        self.sums.index_add_(0, torch.from_numpy(frames), values[window["core_mask"]].float())
        np.add.at(self.counts, frames, 1)

    def finish(self):
        unique = np.flatnonzero(self.counts)
        averaged = self.sums[unique] / torch.from_numpy(self.counts[unique])[:, None]
        return np.stack([
            np.interp(self.times, self.times[unique], column.numpy())
            for column in averaged.T
        ], axis=1).astype(np.float32)


def _packed_logits(output):
    return torch.cat((output["eventness_logit"][:, None], output["type_logits"]), dim=1).detach().cpu()


def restore_logits(windows, outputs, frame_times):
    """重复 native 槽先平均 logits，再按真实 PTS 插值；halo 不参与输出。"""
    accumulator = _LogitAccumulator(frame_times)
    for window, output in zip(windows, outputs, strict=True):
        accumulator.add(window, _packed_logits(output))
    return accumulator.finish()


def predict_video(model, video, frame_times, *, device="cpu", thresholds=None):
    from .data import FrameReader, WindowSampler
    from .evaluation import decode_events
    from .pipeline import prefetch

    started = perf_counter()
    target = torch.device(device)
    model.to(target).eval()
    thresholds = thresholds or {"hit": .60, "bounce": .48}
    sampler = WindowSampler(frame_times)
    accumulator = _LogitAccumulator(frame_times)
    reading, forward, restoring, slots, windows = 0., 0., 0., 0, 0
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(target)

    with FrameReader() as reader, torch.inference_mode():
        def load(start):
            tick = perf_counter()
            window = sampler.window(start)
            if not window["core_mask"].any():
                return window, None, perf_counter() - tick
            rgb = reader.read(video, window["indices"])
            if target.type == "cuda":
                rgb = rgb.pin_memory()
            return window, rgb, perf_counter() - tick

        with prefetch(range(0, sampler.count, 100), load) as prepared:
            for window, rgb, elapsed in prepared:
                reading += elapsed
                if not window["core_mask"].any():
                    continue
                tick = perf_counter()
                rgb = rgb.to(target, non_blocking=True)
                output = model(rgb, precision="fp32")
                values = _packed_logits(output)
                if not bool(values.isfinite().all()):
                    raise FloatingPointError("模型输出包含非有限值")
                forward += perf_counter() - tick
                slots += len(rgb)
                windows += 1
                tick = perf_counter()
                accumulator.add(window, values)
                restoring += perf_counter() - tick
                del output, rgb

    tick = perf_counter()
    logits = accumulator.finish()
    packed = torch.from_numpy(logits)
    scores = (packed[:, :1].sigmoid() * packed[:, 1:].softmax(-1)).numpy()
    events = decode_events(scores, thresholds)
    for event in events:
        event["time_seconds"] = float(frame_times[event["frame_number"]] - frame_times[0])
    restoring += perf_counter() - tick
    result = {
        "model": "TennisContactNet",
        "video": str(video),
        "frame_times": np.asarray(frame_times).tolist(),
        "scores": scores.tolist(),
        "events": events,
        "thresholds": thresholds,
    }
    wall = perf_counter() - started
    result["cost"] = {
        "wall_seconds": wall,
        "read_seconds": reading,
        "forward_seconds": forward,
        "restore_and_decode_seconds": restoring,
        "native_frames": len(frame_times),
        "windows": windows,
        "logical_rgb_slots": slots,
        "native_frames_per_second": len(frame_times) / wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else 0,
    }
    return result
