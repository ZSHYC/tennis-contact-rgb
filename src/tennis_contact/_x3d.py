"""Fixed X3D-S graph derived from PyTorchVideo 0.1.5 (Apache-2.0)."""

from __future__ import annotations

import torch
from torch import nn


class _FrameSE(nn.Module):
    """Apply X3D channel excitation independently at every time step."""

    is_3d = True

    def __init__(self, channels: int, reduced: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv3d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.block(values.mean(dim=(3, 4), keepdim=True))


class Conv2plus1d(nn.Module):
    def __init__(self, conv_t: nn.Module, conv_xy: nn.Module) -> None:
        super().__init__()
        self.conv_t = conv_t
        self.norm = None
        self.activation = None
        self.conv_xy = conv_xy
        self.conv_xy_first = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv_xy(self.conv_t(values))


class ResNetBasicStem(nn.Module):
    def __init__(self, conv: nn.Module, norm: nn.Module, activation: nn.Module) -> None:
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.activation = activation
        self.pool = None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.conv(values)
        if self.norm is not None:
            values = self.norm(values)
        if self.activation is not None:
            values = self.activation(values)
        return values


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        conv_a: nn.Module,
        norm_a: nn.Module,
        act_a: nn.Module,
        conv_b: nn.Module,
        norm_b: nn.Module,
        act_b: nn.Module,
        conv_c: nn.Module,
        norm_c: nn.Module,
    ) -> None:
        super().__init__()
        self.conv_a = conv_a
        self.norm_a = norm_a
        self.act_a = act_a
        self.conv_b = conv_b
        self.norm_b = norm_b
        self.act_b = act_b
        self.conv_c = conv_c
        self.norm_c = norm_c
        self.norm_c.block_final_bn = True

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.conv_a(values)
        if self.norm_a is not None:
            values = self.norm_a(values)
        values = self.act_a(values)
        values = self.act_b(self.norm_b(self.conv_b(values)))
        values = self.conv_c(values)
        return self.norm_c(values) if self.norm_c is not None else values


class ResBlock(nn.Module):
    def __init__(
        self,
        branch1_conv: nn.Module | None,
        branch1_norm: nn.Module | None,
        branch2: nn.Module,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.branch1_conv = branch1_conv
        self.branch1_norm = branch1_norm
        self.branch2 = branch2
        self.activation = activation
        self.branch_fusion = _add

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.branch1_conv is None:
            values = self.branch_fusion(values, self.branch2(values))
        else:
            shortcut = self.branch1_conv(values)
            if self.branch1_norm is not None:
                shortcut = self.branch1_norm(shortcut)
            values = self.branch_fusion(shortcut, self.branch2(values))
        return self.activation(values)


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right


class ResStage(nn.Module):
    def __init__(self, res_blocks: nn.ModuleList) -> None:
        super().__init__()
        self.res_blocks = res_blocks

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            values = block(values)
        return values


class ProjectedPool(nn.Module):
    def __init__(
        self,
        pre_conv: nn.Module,
        pre_norm: nn.Module,
        pre_act: nn.Module,
        pool: nn.Module,
        post_conv: nn.Module,
        post_norm: nn.Module | None,
        post_act: nn.Module,
    ) -> None:
        super().__init__()
        self.pre_conv = pre_conv
        self.pre_norm = pre_norm
        self.pre_act = pre_act
        self.pool = pool
        self.post_conv = post_conv
        self.post_norm = post_norm
        self.post_act = post_act

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.pre_conv(values)
        if self.pre_norm is not None:
            values = self.pre_norm(values)
        values = self.pre_act(values)
        values = self.post_conv(self.pool(values))
        if self.post_norm is not None:
            values = self.post_norm(values)
        return self.post_act(values)


class ResNetBasicHead(nn.Module):
    def __init__(
        self,
        pool: nn.Module,
        dropout: nn.Module,
        proj: nn.Module,
        activation: nn.Module,
        output_pool: nn.Module,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.dropout = dropout
        self.proj = proj
        self.activation = activation
        self.output_pool = output_pool


class Net(nn.Module):
    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks
        _init_weights(self)


def _bottleneck(dim_in: int, dim_inner: int, dim_out: int, stride: int, se: bool) -> BottleneckBlock:
    conv_a = nn.Conv3d(dim_in, dim_inner, 1, bias=False)
    norm_a = nn.BatchNorm3d(dim_inner)
    act_a = nn.ReLU()
    conv_b = nn.Conv3d(
        dim_inner, dim_inner, 3, stride=(1, stride, stride), padding=1,
        bias=False, groups=dim_inner,
    )
    excitation = _FrameSE(dim_inner, {54: 8, 108: 8, 216: 16, 432: 32}[dim_inner]) if se else nn.Identity()
    norm_b = nn.Sequential(nn.BatchNorm3d(dim_inner), excitation)
    act_b = nn.SiLU()
    conv_c = nn.Conv3d(dim_inner, dim_out, 1, bias=False)
    norm_c = nn.BatchNorm3d(dim_out)
    return BottleneckBlock(conv_a, norm_a, act_a, conv_b, norm_b, act_b, conv_c, norm_c)


def _stage(depth: int, dim_in: int, dim_inner: int, dim_out: int) -> ResStage:
    blocks = []
    for index in range(depth):
        input_channels = dim_in if index == 0 else dim_out
        stride = 2 if index == 0 else 1
        branch1_norm = nn.BatchNorm3d(dim_out) if input_channels != dim_out else None
        branch1_conv = (
            nn.Conv3d(input_channels, dim_out, 1, stride=(1, stride, stride), bias=False)
            if input_channels != dim_out or stride > 1 else None
        )
        blocks.append(ResBlock(
            branch1_conv,
            branch1_norm,
            _bottleneck(input_channels, dim_inner, dim_out, stride, index % 2 == 0),
            nn.ReLU(),
        ))
    return ResStage(nn.ModuleList(blocks))


def _head() -> ResNetBasicHead:
    pre_conv = nn.Conv3d(192, 432, 1, bias=False)
    pre_norm = nn.BatchNorm3d(432)
    pre_act = nn.ReLU()
    pool = nn.AvgPool3d((13, 5, 5), stride=1)
    post_conv = nn.Conv3d(432, 2048, 1, bias=False)
    projected = ProjectedPool(pre_conv, pre_norm, pre_act, pool, post_conv, None, nn.ReLU())
    activation = nn.Softmax(dim=1)
    output_pool = nn.AdaptiveAvgPool3d(1)
    # The discarded classifier preserves the checkpoint-era constructor RNG sequence.
    projection = nn.Linear(2048, 400)
    return ResNetBasicHead(projected, nn.Dropout(0.5), projection, activation, output_pool)


def _init_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.modules.batchnorm._NormBase):
            nn.init.constant_(module.weight, 0.0 if getattr(module, "block_final_bn", False) else 1.0)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def x3d_s() -> Net:
    spatial = nn.Conv3d(3, 24, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
    temporal = nn.Conv3d(24, 24, (5, 1, 1), padding=(2, 0, 0), bias=False, groups=24)
    stem = ResNetBasicStem(Conv2plus1d(spatial, temporal), nn.BatchNorm3d(24), nn.ReLU())
    blocks = nn.ModuleList([
        stem,
        _stage(3, 24, 54, 24),
        _stage(5, 24, 108, 48),
        _stage(11, 48, 216, 96),
        _stage(7, 96, 432, 192),
        _head(),
    ])
    return Net(blocks)


__all__ = ["_FrameSE", "x3d_s"]
