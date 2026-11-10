"""数据处理、训练、推理与评价四个命令。"""

import argparse
import json
from pathlib import Path
from time import perf_counter


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parser():
    root = argparse.ArgumentParser(description="TennisContactNet：纯 RGB 网球 hit/bounce 检测")
    root.add_argument("--threads", type=int, default=4, help="PyTorch CPU 线程数，默认4")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="解析标注与真实 PTS，写数据清单")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="直接读取视频训练，支持断点恢复")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, default=Path("weights/initialization.pt"))
    train.add_argument("--resume", type=Path)
    train.add_argument("--order", type=Path, help="显式训练请求顺序，例如 docs/training-order.json")
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=4, help="每次更新累积的视频窗口数")
    train.add_argument("--lr", type=float, default=1e-6)
    train.add_argument("--scope", choices=("backbone2", "all"), default="backbone2")
    train.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    train.add_argument("--seed", type=int, default=20261028)
    train.add_argument("--max-steps", type=int, help="累计更新上限，用于短训或分段恢复")
    predict = commands.add_parser("predict", help="原视频直接推理，保存完整分数和事件")
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--dataset", type=Path)
    predict.add_argument("--split", choices=("train", "validation", "holdout"), default="validation")
    predict.add_argument("--checkpoint", type=Path, default=Path("weights/best.pt"))
    predict.add_argument("--profile", choices=("primary", "secondary"), default="primary")
    predict.add_argument("--thresholds", type=Path, help="使用 evaluate 输出内的 thresholds")
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="一对一事件评价，按来源和类别报告")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True, help="predict --dataset 的输出目录")
    evaluate.add_argument("--split", choices=("train", "validation", "holdout"), default="validation")
    evaluate.add_argument("--hit-threshold", type=float, default=.60)
    evaluate.add_argument("--bounce-threshold", type=float, default=.48)
    evaluate.add_argument("--tune", action="store_true", help="仅 validation 搜索阈值")
    evaluate.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None):
    started = perf_counter()
    arguments = parser().parse_args(argv)
    if arguments.threads < 1:
        raise ValueError("threads 必须为正整数")
    import torch
    torch.set_num_threads(arguments.threads)
    if arguments.command == "prepare":
        from .data import prepare_dataset
        result = prepare_dataset(arguments.input, arguments.output)
        print(json.dumps({"output": str(arguments.output), "videos": len(result["samples"])}, ensure_ascii=False))
    elif arguments.command == "train":
        from .training import train
        result = train(
            arguments.dataset, arguments.output, checkpoint=arguments.checkpoint,
            device=arguments.device, epochs=arguments.epochs, batch_size=arguments.batch_size,
            lr=arguments.lr, scope=arguments.scope, precision=arguments.precision,
            seed=arguments.seed, max_steps=arguments.max_steps, resume=arguments.resume,
            order_path=arguments.order,
        )
        print(json.dumps(result, ensure_ascii=False))
    elif arguments.command == "predict":
        from .data import load_dataset, probe_video
        from .inference import predict_video
        from .model import load_model
        model, metadata = load_model(arguments.checkpoint)
        thresholds = metadata["thresholds" if arguments.profile == "primary" else "secondary_thresholds"]
        if arguments.thresholds:
            thresholds = json.loads(arguments.thresholds.read_text(encoding="utf-8"))["thresholds"]
        if arguments.video:
            result = predict_video(model, arguments.video, probe_video(arguments.video),
                                   device=arguments.device, thresholds=thresholds)
            write_json(arguments.output, result)
            print(json.dumps({**result["cost"], "command_elapsed_seconds": perf_counter() - started}))
        else:
            samples = load_dataset(arguments.dataset, arguments.split)
            if not samples:
                raise ValueError("所选 split 没有视频")
            for sample in samples:
                result = predict_video(model, sample.video, sample.frame_times,
                                       device=arguments.device, thresholds=thresholds)
                result["sample_id"] = sample.id
                write_json(arguments.output / f"{sample.id}.json", result)
                print(json.dumps({"sample_id": sample.id, **result["cost"],
                                  "command_elapsed_seconds": perf_counter() - started}), flush=True)
    else:
        import numpy as np
        from .data import load_dataset
        from .evaluation import evaluate_predictions, search_thresholds
        samples = load_dataset(arguments.dataset, arguments.split)
        if not samples:
            raise ValueError("所选 split 没有视频")
        if arguments.tune and arguments.split != "validation":
            raise ValueError("阈值只能在 validation 上选择")
        predictions = {}
        for sample in samples:
            saved = json.loads((arguments.predictions / f"{sample.id}.json").read_text(encoding="utf-8"))
            if saved["sample_id"] != sample.id or not np.array_equal(saved["frame_times"], sample.frame_times):
                raise ValueError(f"预测视频或 PTS 不匹配：{sample.id}")
            predictions[sample.id] = np.asarray(saved["scores"], dtype=np.float32)
        thresholds = {"hit": arguments.hit_threshold, "bounce": arguments.bounce_threshold}
        if arguments.tune:
            thresholds = search_thresholds(samples, predictions)
        result = evaluate_predictions(samples, predictions, thresholds)
        result["thresholds"] = thresholds
        result["split"] = arguments.split
        result["threshold_origin"] = "validation_search" if arguments.tune else "fixed"
        write_json(arguments.output, result)
        print(json.dumps(result["overall"]))
    return 0
