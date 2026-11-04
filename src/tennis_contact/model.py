"""独立纯RGB网球击球与落地检测网络。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from fvcore.nn.squeeze_excitation import SqueezeExcitation
from pytorchvideo.layers.swish import Swish
from pytorchvideo.models.hub.x3d import x3d_s
from torch import nn
from torch.nn import functional as F


class _FrameSE(nn.Module):
    """为每个时刻独立计算通道门控的X3D激励层。"""

    is_3d = True

    def __init__(self, original: SqueezeExcitation) -> None:
        super().__init__()
        self.block = original.block

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.block(values.mean(dim=(3, 4), keepdim=True))


class _SpatialMean(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=(3, 4), keepdim=True)


def _enable_frame_se(model: nn.Module) -> None:
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, SqueezeExcitation)]
    if len(targets) != 15:
        raise RuntimeError("X3D-S must contain 15 squeeze-excitation layers")
    for name, module in targets:
        parent_name, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent_name), attribute, _FrameSE(module))


def _enable_standard_silu(model: nn.Module) -> None:
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, Swish)]
    if len(targets) != 26:
        raise RuntimeError("X3D-S must contain 26 Swish layers")
    for name, _module in targets:
        parent_name, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent_name), attribute, nn.SiLU())


class TennisContactNet(nn.Module):
    """逐帧FrameSE、S3局部状态与完整窗口双向GRU。"""

    def __init__(self) -> None:
        super().__init__()
        blocks = x3d_s(pretrained=False).blocks
        head = blocks[5]
        self.trunk = nn.Sequential(*list(blocks[:5]))
        self.pool = head.pool
        self.pool.pool = _SpatialMean()
        self.gru = nn.GRU(2048, 256, batch_first=True, bidirectional=True)
        self.eventness_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 1))
        self.type_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 2))
        self.register_buffer("image_mean", torch.full((1, 3, 1, 1), 0.45))
        self.register_buffer("image_std", torch.full((1, 3, 1, 1), 0.225))
        self.spatial_locator = nn.Linear(96, 1, bias=False)
        self.presence_head = nn.Linear(96, 1)
        self.value_projection = nn.Linear(96, 64, bias=False)
        self.state_adapter = nn.Sequential(nn.Linear(68, 128), nn.SiLU(), nn.Linear(128, 2048))
        nn.init.zeros_(self.state_adapter[-1].weight)
        nn.init.zeros_(self.state_adapter[-1].bias)
        _enable_frame_se(self)
        _enable_standard_silu(self)
        self._freeze_batch_norm()

    def _freeze_batch_norm(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()

    def train(self, mode: bool = True) -> TennisContactNet:
        super().train(mode)
        self._freeze_batch_norm()
        return self

    @staticmethod
    def _grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        y = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5) * (2 / height) - 1
        x = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5) * (2 / width) - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)

    def _local_state(self, s3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = s3.float().permute(0, 2, 3, 4, 1)
        time, height, width = features.shape[1:4]
        flat = features.reshape(time, height * width, 96)
        spatial_logits = self.spatial_locator(flat).squeeze(-1)
        attention = spatial_logits.softmax(dim=-1)
        presence_logit = self.presence_head(flat.mean(dim=1)).squeeze(-1)
        presence = presence_logit.sigmoid()
        xy = attention @ self._grid(height, width, attention)
        entropy = -(attention * attention.clamp_min(torch.finfo(attention.dtype).tiny).log()).sum(dim=-1)
        entropy = entropy / math.log(height * width)
        weighted = (attention.unsqueeze(-1) * flat).sum(dim=1)
        local = presence.unsqueeze(-1) * F.layer_norm(self.value_projection(weighted), (64,))
        state = torch.cat((local, xy, entropy.unsqueeze(-1), presence.unsqueeze(-1)), dim=-1)
        return spatial_logits, presence_logit, state

    def forward(self, rgb: torch.Tensor, *, precision: str = "fp32") -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype != torch.uint8 or not len(rgb):
            raise ValueError("rgb must be nonempty uint8 [T,3,H,W]")
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be 'fp32' or 'bf16'")
        if rgb.device != self.image_mean.device:
            raise ValueError("rgb and model must be on the same device")

        device_type = rgb.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            values = (rgb.float().div(255) - self.image_mean) / self.image_std
            values = values.permute(1, 0, 2, 3).unsqueeze(0)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                for stage in self.trunk[:4]:
                    values = stage(values)
                s3 = values
                values = self.trunk[4](values)
                values = self.pool(values)
            spatial_logits, presence_logit, state = self._local_state(s3)
            sequence = values.float().squeeze(-1).squeeze(-1).transpose(1, 2)
            sequence = sequence + self.state_adapter(state).unsqueeze(0)
            sequence, _ = self.gru(sequence)
            eventness = self.eventness_head(sequence).squeeze(0).squeeze(-1)
            types = self.type_head(sequence).squeeze(0)
        return {
            "eventness_logit": eventness,
            "type_logits": types,
            "spatial_logits": spatial_logits,
            "presence_logit": presence_logit,
            "state": state,
        }


def load_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[TennisContactNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    model = TennisContactNet()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, {key: value for key, value in checkpoint.items() if key != "model"}


__all__ = ["TennisContactNet", "load_model"]
