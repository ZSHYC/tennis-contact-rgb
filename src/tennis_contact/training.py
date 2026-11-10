"""独立纯RGB事件模型的训练与恢复。"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import signal
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .data import FrameReader, WindowSampler, load_dataset
from .model import TennisContactNet
from .pipeline import prefetch


POSITIVE_WEIGHT = 14.330945259756955
TYPE_WEIGHT = 0.5
TYPE_DENOMINATOR = 36.39117789626757
EVENT_SIGMA_SECONDS = 0.04
TYPE_RADIUS_SECONDS = 0.08
_STOP_REQUESTED = False


def _targets(
    sample: Any,
    window: dict[str, Any],
    device: torch.device,
    native: dict[str, np.ndarray] | None = None,
) -> dict[str, torch.Tensor]:
    """按观测到的原生帧PTS构造事件软目标。"""
    indices = np.asarray(window["indices"], dtype=np.int64)
    weights = torch.as_tensor(window["weights"], dtype=torch.float32, device=device)
    core = torch.as_tensor(window["core_mask"], dtype=torch.bool, device=device)
    if len(indices) == 0 or weights.shape != core.shape or weights.shape != (len(indices),):
        raise ValueError("window indices, weights, and core_mask must have one common nonempty length")
    weights = weights * core

    native = _native_targets(sample) if native is None else native
    return {
        "weights": weights,
        "eventness": torch.as_tensor(native["eventness"][indices], device=device),
        "types": torch.as_tensor(native["types"][indices], device=device),
        "type_mask": torch.as_tensor(native["type_mask"][indices], device=device),
    }


def _native_targets(sample: Any) -> dict[str, np.ndarray]:
    """为一个视频的原生帧一次性构造目标。"""
    times = np.asarray(sample.frame_times, dtype=np.float64)

    events = [event for event in sample.events if event.get("event_type") in {"hit", "bounce"}]
    if not events:
        return {
            "eventness": np.zeros(len(times), dtype=np.float32),
            "types": np.zeros(len(times), dtype=np.int64),
            "type_mask": np.zeros(len(times), dtype=bool),
        }

    frames = np.asarray([int(event["frame_number"]) for event in events], dtype=np.int64)
    if np.any(frames < 0) or np.any(frames >= len(sample.frame_times)):
        raise ValueError(f"event frame is outside video {sample.id!r}")
    event_times = np.asarray(sample.frame_times, dtype=np.float64)[frames]
    event_types = np.asarray([event["event_type"] == "bounce" for event in events], dtype=np.int64)
    order = np.argsort(event_times, kind="stable")
    sorted_times = event_times[order]
    right = np.searchsorted(sorted_times, times).clip(max=len(sorted_times) - 1)
    left = (right - 1).clip(min=0)
    left_distance = np.abs(times - sorted_times[left])
    right_distance = np.abs(times - sorted_times[right])
    choose_right = (right_distance < left_distance) | (
        (right_distance == left_distance) & (order[right] < order[left])
    )
    chosen = order[np.where(choose_right, right, left)]
    nearest = np.minimum(left_distance, right_distance)
    types = event_types[chosen]

    nearest_by_type = []
    for event_type in (0, 1):
        typed = np.sort(event_times[event_types == event_type])
        if not len(typed):
            nearest_by_type.append(np.full(len(times), np.inf))
            continue
        typed_right = np.searchsorted(typed, times).clip(max=len(typed) - 1)
        typed_left = (typed_right - 1).clip(min=0)
        nearest_by_type.append(
            np.minimum(np.abs(times - typed[typed_left]), np.abs(times - typed[typed_right]))
        )
    opposite_distance = np.where(types == 0, nearest_by_type[1], nearest_by_type[0])
    ambiguous = np.isclose(opposite_distance, nearest, atol=1e-9, rtol=0.0)
    return {
        "eventness": np.asarray(
            np.exp(-0.5 * np.square(nearest / EVENT_SIGMA_SECONDS)), dtype=np.float32
        ),
        "types": types,
        "type_mask": (nearest <= TYPE_RADIUS_SECONDS + 1e-9) & ~ambiguous,
    }


def loss_sums(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    positive_weight: float = POSITIVE_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算加权事件BCE与联合分数类型BCE的分子。"""
    event_sum, joint_sum = _loss_sums(output, targets, positive_weight)
    if not bool(torch.stack((event_sum, joint_sum)).isfinite().all()):
        raise FloatingPointError("training loss is nonfinite")
    return event_sum, joint_sum


def _loss_sums(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    positive_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    event = output["eventness_logit"].float()
    types = output["type_logits"].float()
    weights, y = targets["weights"], targets["eventness"]
    labels, mask = targets["types"], targets["type_mask"]
    length = len(weights)
    if event.shape != (length,) or types.shape != (length, 2):
        raise ValueError("model output and target lengths differ")

    event_sum = (
        F.binary_cross_entropy_with_logits(
            event,
            y,
            pos_weight=event.new_tensor(positive_weight),
            reduction="none",
        )
        * weights
    ).sum()

    log_q = F.logsigmoid(event)
    log_not_q = F.logsigmoid(-event)
    log_r = F.log_softmax(types, dim=-1)
    log_p = log_q[:, None] + log_r
    log_not_p = torch.logaddexp(log_not_q[:, None], log_q[:, None] + log_r.flip(-1))
    adjusted = positive_weight * y / (positive_weight * y + 1 - y)
    joint_target = torch.zeros_like(types).scatter(1, labels[:, None], adjusted[:, None])
    joint_rows = -(joint_target * log_p + (1 - joint_target) * log_not_p).sum(dim=-1)
    joint_sum = (joint_rows * weights * mask).sum()
    return event_sum, joint_sum


def _request_order(
    samples: list[Any], epochs: int, order_path: Path | None = None
) -> tuple[list[tuple[int, int, bool]], dict[str, Any] | None]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    natural: list[tuple[int, int]] = []
    for sample_index, sample in enumerate(samples):
        times = np.asarray(sample.frame_times, dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError(f"invalid frame_times for {sample.id!r}")
        logical_count = math.ceil(float(times[-1] - times[0]) * 30.0 - 1e-8) + 1
        natural.extend((sample_index, start) for start in range(0, logical_count, 100))
    if not natural:
        raise ValueError("training split contains no windows")
    if order_path is None:
        one_epoch = [(sample_index, start, False) for sample_index, start in natural]
        return one_epoch * epochs, None

    specification = json.loads(order_path.read_text(encoding="utf-8"))
    if not isinstance(specification, dict) or not isinstance(specification.get("requests"), list):
        raise ValueError("training order needs a requests list")
    sample_ids = [sample.id for sample in samples]
    recorded_ids = specification.get("train_sample_ids")
    if (
        not isinstance(recorded_ids, list)
        or len(recorded_ids) != len(set(recorded_ids))
        or set(recorded_ids) != set(sample_ids)
    ):
        raise ValueError("training order split ids differ from the manifest train split")
    by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    one_epoch = []
    for row in specification["requests"]:
        if not isinstance(row, dict) or set(row) != {"sample_id", "start", "mirror"}:
            raise ValueError("each training request needs sample_id, start, and mirror")
        if row["sample_id"] not in by_id or type(row["start"]) is not int or row["start"] < 0:
            raise ValueError("training request sample or start is invalid")
        if type(row["mirror"]) is not bool:
            raise ValueError("training request mirror must be boolean")
        one_epoch.append((by_id[row["sample_id"]], row["start"], row["mirror"]))
    requested = sorted((sample_index, start) for sample_index, start, _mirror in one_epoch)
    if requested != sorted(natural):
        raise ValueError("training order must contain every manifest window exactly once")
    return one_epoch * epochs, specification


def _set_scope(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    if scope not in {"backbone2", "all"}:
        raise ValueError("scope must be 'backbone2' or 'all'")
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(scope == "all" or name.startswith("trunk.2."))
        if parameter.requires_grad:
            selected.append(parameter)
    if not selected:
        raise ValueError("scope selected no parameters")
    return selected


def _class_balance(
    samples: list[Any],
    order: list[tuple[int, int, bool]],
    native: list[dict[str, np.ndarray]] | None = None,
    samplers: list[WindowSampler] | None = None,
) -> float:
    total = positive = 0.0
    seen = set()
    native = [_native_targets(sample) for sample in samples] if native is None else native
    samplers = [WindowSampler(sample.frame_times) for sample in samples] if samplers is None else samplers
    for sample_index, start, _mirror in order:
        if (sample_index, start) in seen:
            continue
        seen.add((sample_index, start))
        window = samplers[sample_index].window(start, core_steps=100, halo=28)
        indices = np.asarray(window["indices"], dtype=np.int64)
        weights = np.asarray(window["weights"], dtype=np.float32)
        weight_tensor = torch.from_numpy(weights)
        total += float(weight_tensor.sum())
        positive += float(
            (weight_tensor * torch.from_numpy(native[sample_index]["eventness"][indices])).sum()
        )
    if not 0 < positive < total:
        raise ValueError("training split must contain both event and background mass")
    return (total - positive) / positive


def _event_denominator(targets: list[dict[str, torch.Tensor]], device: torch.device) -> float:
    if device.type == "cuda":
        for target in targets:
            if not target["weights"].is_pinned():
                target["weights"] = target["weights"].pin_memory()
        sums = torch.stack([
            target["weights"].to(device, non_blocking=True).sum() for target in targets
        ])
        return sum(sums.cpu().tolist())
    return sum(float(target["weights"].sum()) for target in targets)


def _rng_state(device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": deepcopy(random.getstate()),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if device.type == "cuda"
        else [],
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if device.type != "cuda":
            raise ValueError("resume checkpoint contains CUDA RNG but target device is not CUDA")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_save(path: Path, packet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(packet, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _signal_handler(_number: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


@contextmanager
def _signals():
    previous = {number: signal.signal(number, _signal_handler) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def train(
    manifest: str | Path,
    output: str | Path,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-6,
    scope: str = "backbone2",
    precision: str = "bf16",
    seed: int = 20261028,
    max_steps: int | None = None,
    resume: str | Path | None = None,
    order_path: str | Path | None = None,
) -> dict[str, Any]:
    """按固定请求顺序训练，同时保存推理权重与恢复状态。"""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    started = perf_counter()
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    inference_path, training_path = output / "last.pt", output / "training.pt"
    if resume is None and (inference_path.exists() or training_path.exists()):
        raise FileExistsError(f"training output already exists: {output}")
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    if batch_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("batch_size and lr must be positive")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be 'fp32' or 'bf16'")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")

    samples = load_dataset(manifest, split="train")
    resolved_order_path = None if order_path is None else Path(order_path).resolve()
    order, order_specification = _request_order(samples, epochs, resolved_order_path)
    dataset = [
        {
            "id": sample.id,
            "video": str(sample.video),
            "source": sample.source,
            "frame_times": np.asarray(sample.frame_times, dtype=np.float64).tolist(),
            "events": deepcopy(sample.events),
        }
        for sample in samples
    ]
    packet = None if resume is None else torch.load(Path(resume), map_location="cpu", weights_only=False)
    native_targets = None
    samplers = None
    if packet is None:
        native_targets = [_native_targets(sample) for sample in samples]
        samplers = [WindowSampler(sample.frame_times) for sample in samples]
        calculated_positive_weight = _class_balance(samples, order, native_targets, samplers)
        positive_weight = calculated_positive_weight
        if order_specification is not None and "positive_weight" in order_specification:
            recorded = order_specification["positive_weight"]
            if (
                not isinstance(recorded, (int, float))
                or not math.isfinite(recorded)
                or not math.isclose(recorded, calculated_positive_weight, rel_tol=0.0, abs_tol=1e-5)
            ):
                raise ValueError("recorded positive_weight differs from rebuilt training targets")
            positive_weight = float(recorded)
    else:
        stored_config = packet.get("training_config", {})
        calculated_positive_weight = stored_config.get("calculated_positive_weight")
        positive_weight = stored_config.get("positive_weight")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in (calculated_positive_weight, positive_weight)
        ):
            raise ValueError("resume checkpoint contains invalid class balance")
        del stored_config
    config = {
        "manifest": str(manifest),
        "order_path": None if resolved_order_path is None else str(resolved_order_path),
        "order_specification": order_specification,
        "dataset": dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(lr),
        "scope": scope,
        "precision": precision,
        "seed": seed,
        "device": str(device),
        "positive_weight": positive_weight,
        "calculated_positive_weight": calculated_positive_weight,
        "type_denominator": TYPE_DENOMINATOR,
    }
    metadata: dict[str, Any] = {}
    if resume is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    model = TennisContactNet().to(device)
    if resume is None:
        if checkpoint is not None:
            initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(initial, dict) or not isinstance(initial.get("model"), dict):
                raise ValueError("initial checkpoint must contain model")
            model.load_state_dict(initial["model"], strict=True)
            metadata = {key: deepcopy(value) for key, value in initial.items() if key != "model"}
            del initial
        parameters = _set_scope(model, scope)
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
        cursor = steps = 0
        previous_cost = {
            "wall_seconds": 0.0,
            "video_read_seconds": 0.0,
            "forward_backward_seconds": 0.0,
            "updates": 0,
        }
    else:
        if packet.get("training_config") != config or packet.get("order") != order:
            raise ValueError("resume training configuration or request order differs")
        model.load_state_dict(packet["model"], strict=True)
        parameters = _set_scope(model, scope)
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
        optimizer.load_state_dict(packet["optimizer"])
        _optimizer_to(optimizer, device)
        cursor, steps = int(packet["cursor"]), int(packet["steps"])
        metadata = deepcopy(packet.get("metadata", {}))
        previous_cost = deepcopy(packet.get("cost", {}))
        if set(previous_cost) != {
            "wall_seconds", "video_read_seconds", "forward_backward_seconds", "updates"
        }:
            raise ValueError("resume cost state differs")
        _restore_rng(packet["rng"], device)
        del packet
    model.train()
    start_steps = steps
    video_read_seconds = forward_backward_seconds = 0.0
    if cursor < len(order) and (max_steps is None or steps < max_steps) and native_targets is None:
        native_targets = [_native_targets(sample) for sample in samples]
    if cursor < len(order) and (max_steps is None or steps < max_steps) and samplers is None:
        samplers = [WindowSampler(sample.frame_times) for sample in samples]

    def save(status: str) -> dict[str, Any]:
        inference = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "thresholds": metadata.get("thresholds", {"hit": 0.60, "bounce": 0.48}),
            "secondary_thresholds": metadata.get(
                "secondary_thresholds", {"hit": 0.67, "bounce": 0.57}
            ),
            "description": metadata.get("description", "TennisContactNet training checkpoint"),
        }
        packet = {
            "model": inference["model"],
            "optimizer": optimizer.state_dict(),
            "cursor": cursor,
            "steps": steps,
            "order": order,
            "rng": _rng_state(device),
            "training_config": config,
            "status": status,
            "metadata": metadata,
            "cost": {
                "wall_seconds": previous_cost["wall_seconds"] + perf_counter() - started,
                "video_read_seconds": previous_cost["video_read_seconds"] + video_read_seconds,
                "forward_backward_seconds": (
                    previous_cost["forward_backward_seconds"] + forward_backward_seconds
                ),
                "updates": previous_cost["updates"] + steps - start_steps,
            },
        }
        _atomic_save(training_path, packet)
        _atomic_save(inference_path, inference)
        return packet

    remaining_updates = math.inf if max_steps is None else max_steps - steps
    request_stop = (
        max(cursor, min(len(order), cursor + int(remaining_updates * batch_size)))
        if math.isfinite(remaining_updates)
        else len(order)
    )

    def requests():
        position = cursor
        while position < request_stop:
            prepared = []
            for sample_index, start, mirror in order[position : min(request_stop, position + batch_size)]:
                sample = samples[sample_index]
                window = samplers[sample_index].window(start, core_steps=100, halo=28)
                targets = _targets(sample, window, torch.device("cpu"), native_targets[sample_index])
                prepared.append((sample, window, mirror, targets))
            batch_targets = [row[3] for row in prepared]
            for index, row in enumerate(prepared, 1):
                yield *row, batch_targets if index == 1 else None, index == len(prepared)
            position += len(prepared)

    with FrameReader(max_cached_frames=160) as reader:
        def load(row):
            sample, window, mirror, targets, batch_targets, batch_last = row
            tick = perf_counter()
            rgb = reader.read(sample.video, np.asarray(window["indices"], dtype=np.int64))
            if device.type == "cuda":
                rgb = rgb.pin_memory()
                targets = {key: value.pin_memory() for key, value in targets.items()}
            return mirror, targets, batch_targets, batch_last, rgb, perf_counter() - tick

        with _signals(), prefetch(requests(), load) as loaded:
            batch_count = 0
            for mirror, cpu_targets, batch_targets, batch_last, rgb, read_seconds in loaded:
                if batch_count == 0:
                    event_denominator = _event_denominator(batch_targets, device)
                    if not math.isfinite(event_denominator) or event_denominator <= 0:
                        raise ValueError("batch has no weighted core frames")
                    optimizer.zero_grad(set_to_none=True)
                    cuda_timings = []
                    transfer_timings = []
                    batch_finite = None
                batch_count += 1
                video_read_seconds += read_seconds
                if device.type == "cuda":
                    transfer = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    transfer[0].record()
                    transfer_timings.append(transfer)
                rgb = rgb.to(device, non_blocking=device.type == "cuda")
                targets = {
                    key: value.to(device, non_blocking=device.type == "cuda")
                    for key, value in cpu_targets.items()
                }
                if device.type == "cuda":
                    transfer[1].record()
                if mirror:
                    rgb = torch.flip(rgb, (-1,))
                if device.type == "cuda":
                    timing = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    timing[0].record()
                    cuda_timings.append(timing)
                else:
                    tick = perf_counter()
                event_sum, joint_sum = _loss_sums(
                    model(rgb, precision=precision), targets, positive_weight
                )
                finite = torch.stack((event_sum, joint_sum)).isfinite().all().detach()
                batch_finite = finite if batch_finite is None else batch_finite & finite
                loss = event_sum / event_denominator
                loss = loss + TYPE_WEIGHT * joint_sum / TYPE_DENOMINATOR
                loss.backward()
                if device.type == "cuda":
                    timing[1].record()
                else:
                    forward_backward_seconds += perf_counter() - tick
                if not batch_last:
                    continue
                if not bool(batch_finite):
                    raise FloatingPointError("training loss is nonfinite")
                if device.type == "cuda":
                    timing = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    timing[0].record()
                    cuda_timings.append(timing)
                else:
                    tick = perf_counter()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                if device.type == "cuda":
                    timing[1].record()
                    timing[1].synchronize()
                    video_read_seconds += sum(start.elapsed_time(end) for start, end in transfer_timings) / 1000
                    forward_backward_seconds += sum(start.elapsed_time(end) for start, end in cuda_timings) / 1000
                else:
                    forward_backward_seconds += perf_counter() - tick
                cursor += batch_count
                batch_count = 0
                steps += 1
                if _STOP_REQUESTED:
                    break
                if steps % 64 == 0 and cursor < len(order) and (max_steps is None or steps < max_steps):
                    save("running")
    status = "interrupted" if _STOP_REQUESTED else "complete" if cursor == len(order) else "paused"
    save(status)
    wall_seconds = perf_counter() - started
    return {
        "status": status,
        "steps": steps,
        "cursor": cursor,
        "requests": len(order),
        "positive_weight": positive_weight,
        "wall_seconds": wall_seconds,
        "updates_this_run": steps - start_steps,
        "video_read_seconds": video_read_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "model": str(inference_path),
        "resume": str(training_path),
    }


__all__ = ["loss_sums", "train"]
