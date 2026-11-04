"""原视频逐窗推理，以及完整原生时间轴还原。"""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch


def restore_logits(windows, outputs, frame_times):
    """重复 native 槽先平均 logits，再按真实 PTS 插值；halo 不参与输出。"""
    frames = np.concatenate([w["indices"][w["core_mask"]] for w in windows])
    unique, inverse, counts = np.unique(frames, return_inverse=True, return_counts=True)
    values = torch.cat([
        torch.cat((o["eventness_logit"][:, None], o["type_logits"]), dim=1)[w["core_mask"]]
        for w, o in zip(windows, outputs, strict=True)
    ]).detach().cpu()
    averaged = torch.zeros(len(unique), 3, dtype=torch.float32)
    averaged.index_add_(0, torch.from_numpy(inverse), values.float())
    averaged /= torch.from_numpy(counts)[:, None]
    return np.stack([
        np.interp(frame_times, frame_times[unique], column.numpy())
        for column in averaged.T
    ], axis=1).astype(np.float32)


def predict_video(model, video, frame_times, *, device="cpu", thresholds=None):
    from .data import read_frames, sample_window
    from .evaluation import decode_events

    started = perf_counter()
    target = torch.device(device)
    model.to(target).eval()
    thresholds = thresholds or {"hit": .60, "bounce": .48}
    windows, outputs = [], []
    reading, forward, slots = 0., 0., 0
    count = int(np.ceil((frame_times[-1] - frame_times[0]) * 30 - 1e-8)) + 1
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(target)
    with torch.inference_mode():
        for start in range(0, count, 100):
            window = sample_window(frame_times, start)
            if not window["core_mask"].any():
                continue
            tick = perf_counter()
            rgb = read_frames(Path(video), window["indices"]).to(target)
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            reading += perf_counter() - tick
            tick = perf_counter()
            output = model(rgb, precision="fp32")
            output = {k: output[k].detach().cpu() for k in ("eventness_logit", "type_logits")}
            if any(not bool(v.isfinite().all()) for v in output.values()):
                raise FloatingPointError("模型输出包含非有限值")
            forward += perf_counter() - tick
            slots += len(rgb)
            windows.append(window)
            outputs.append(output)
    tick = perf_counter()
    logits = restore_logits(windows, outputs, frame_times)
    packed = torch.from_numpy(logits)
    scores = (packed[:, :1].sigmoid() * packed[:, 1:].softmax(-1)).numpy()
    events = decode_events(scores, thresholds)
    for event in events:
        event["time_seconds"] = float(frame_times[event["frame_number"]] - frame_times[0])
    restoring = perf_counter() - tick
    wall = perf_counter() - started
    return {
        "model": "TennisContactNet",
        "video": str(video),
        "frame_times": np.asarray(frame_times).tolist(),
        "scores": scores.tolist(),
        "events": events,
        "thresholds": thresholds,
        "cost": {
            "wall_seconds": wall,
            "read_seconds": reading,
            "forward_seconds": forward,
            "restore_and_decode_seconds": restoring,
            "native_frames": len(frame_times),
            "windows": len(windows),
            "logical_rgb_slots": slots,
            "native_frames_per_second": len(frame_times) / wall,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else 0,
        },
    }
