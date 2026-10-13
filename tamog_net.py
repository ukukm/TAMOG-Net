from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[Tensor, Tensor]:
    weight = conv.weight
    if conv.bias is None:
        bias = torch.zeros(weight.size(0), device=weight.device, dtype=weight.dtype)
    else:
        bias = conv.bias
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused_weight = weight * scale.reshape(-1, 1, 1, 1)
    fused_bias = bn.bias + (bias - bn.running_mean) * scale
    return fused_weight, fused_bias


class RepDepthwiseConv(nn.Module):
    """Depthwise 3x3 mixer with train-time branches and deploy-time fusion."""

    def __init__(self, channels: int, deploy: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        if deploy:
            self.reparam = nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=True
            )
        else:
            self.dw3 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.dw1 = nn.Sequential(
                nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.identity = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.reparam(x)
        return self.dw3(x) + self.dw1(x) + self.identity(x)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel3, bias3 = _fuse_conv_bn(self.dw3[0], self.dw3[1])
        kernel1, bias1 = _fuse_conv_bn(self.dw1[0], self.dw1[1])
        kernel1 = F.pad(kernel1, [1, 1, 1, 1])

        identity_kernel = torch.zeros_like(kernel3)
        identity_kernel[:, :, 1, 1] = 1.0
        identity_bn = self.identity
        identity_scale = identity_bn.weight / torch.sqrt(
            identity_bn.running_var + identity_bn.eps
        )
        identity_bias = identity_bn.bias - identity_bn.running_mean * identity_scale
        identity_kernel = identity_kernel * identity_scale.reshape(-1, 1, 1, 1)

        self.reparam = nn.Conv2d(
            self.channels,
            self.channels,
            3,
            padding=1,
            groups=self.channels,
            bias=True,
        )
        self.reparam.weight.copy_(kernel3 + kernel1 + identity_kernel)
        self.reparam.bias.copy_(bias3 + bias1 + identity_bias)
        del self.dw3, self.dw1, self.identity
        self.deploy = True


class DepthwiseConv(nn.Module):
    """Single depthwise 3x3 mixer used by the baseline ablation variant."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        return


class MultiOrderGatedAggregation(nn.Module):
    """Multi-order depthwise context with a per-image routing policy."""

    def __init__(self, channels: int, reduction: int = 4, adaptive_route: bool = True) -> None:
        super().__init__()
        self.adaptive_route = adaptive_route
        hidden = max(channels // reduction, 8)
        self.orders = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=2,
                    dilation=2,
                    groups=channels,
                    bias=False,
                ),
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=3,
                    dilation=3,
                    groups=channels,
                    bias=False,
                ),
            ]
        )
        self.order_norm = nn.BatchNorm2d(channels)
        self.router = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, 3, 1),
            )
            if adaptive_route
            else None
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.project_norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.router is None:
            route = x.new_full((x.shape[0], 3), 1.0 / 3.0)
        else:
            route_logits = self.router(x).flatten(1)
            route = torch.softmax(route_logits, dim=1)
        order_features = torch.stack([branch(x) for branch in self.orders], dim=1)
        mixed = (order_features * route[:, :, None, None, None]).sum(dim=1)
        mixed = self.order_norm(mixed)
        mixed = self.project_norm(self.project(mixed))
        return x + mixed, route


class TAMOGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 2.0,
        use_rep: bool = True,
        use_moa: bool = True,
        adaptive_route: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(int(round(in_channels * expansion)), out_channels)
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.downsample_features = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.token_mixer = RepDepthwiseConv(hidden) if use_rep else DepthwiseConv(hidden)
        self.aggregate = (
            MultiOrderGatedAggregation(hidden, adaptive_route=adaptive_route)
            if use_moa
            else None
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.stride = stride

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        identity = self.downsample(x)
        out = self.downsample_features(self.expand(x))
        out = self.token_mixer(out)
        if self.aggregate is None:
            route = out.new_full((out.shape[0], 3), 1.0 / 3.0)
        else:
            out, route = self.aggregate(out)
        out = self.project(out)
        if out.shape[-2:] != identity.shape[-2:]:
            identity = F.adaptive_avg_pool2d(identity, out.shape[-2:])
        return F.gelu(out + identity), route

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        self.token_mixer.switch_to_deploy()


def _default_widths(num_blocks: int, width_mult: float) -> List[int]:
    base = [32, 48, 80, 128, 192, 256, 320]
    if not 2 <= num_blocks <= len(base):
        raise ValueError("num_blocks must be between 2 and 7")
    return [max(8, int(round(width_mult * width / 8) * 8)) for width in base[:num_blocks]]


class TAMOGNet(nn.Module):
    """Configurable classifier; deployment removes only RepDepthwise branches."""

    def __init__(
        self,
        num_classes: int = 4,
        num_blocks: int = 5,
        width_mult: float = 1.0,
        expansion: float = 2.0,
        aux_loss: bool = True,
        variant: str = "tamog",
    ) -> None:
        super().__init__()
        widths = _default_widths(num_blocks, width_mult)
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.aux_loss = aux_loss
        variant_settings = {
            "baseline": (False, False, False),
            "rep": (True, False, False),
            "moa": (True, True, False),
            "tamog": (True, True, True),
        }
        if variant not in variant_settings and variant != "tamog_lite":
            raise ValueError(f"unknown TAMOG ablation variant: {variant}")
        self.variant = variant
        use_rep, use_moa, adaptive_route = variant_settings.get(
            variant, (True, True, True)
        )
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        self.aux_classifiers = nn.ModuleList()
        in_channels = widths[0]
        for index, out_channels in enumerate(widths):
            stride = 1 if index == 0 else 2
            if variant == "tamog_lite":
                # Keep the expensive multi-order/router path only in the two
                # lowest-resolution stages; early stages retain Rep mixing.
                block_use_moa = index >= len(widths) - 2
                block_adaptive_route = block_use_moa
            else:
                block_use_moa = use_moa
                block_adaptive_route = adaptive_route
            self.blocks.append(
                TAMOGBlock(
                    in_channels,
                    out_channels,
                    stride,
                    expansion=expansion,
                    use_rep=use_rep,
                    use_moa=block_use_moa,
                    adaptive_route=block_adaptive_route,
                )
            )
            if aux_loss and index < len(widths) - 1:
                self.aux_classifiers.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(out_channels, num_classes),
                    )
                )
            in_channels = out_channels
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
        return_routes: bool = False,
    ):
        x = self.stem(x)
        aux_outputs: List[Tensor] = []
        routes: List[Tensor] = []
        for index, block in enumerate(self.blocks):
            x, route = block(x)
            routes.append(route)
            if self.aux_loss and index < len(self.aux_classifiers):
                aux_outputs.append(self.aux_classifiers[index](x))
        features = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.classifier(features)
        if return_aux or return_routes:
            return {
                "logits": logits,
                "aux_logits": aux_outputs,
                "routes": routes,
                "features": features,
            }
        return logits

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        for block in self.blocks:
            block.switch_to_deploy()


class ModelEMA:
    """Legacy EMA helper; not used by the current manuscript experiments."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        teacher_state = self.module.state_dict()
        student_state = model.state_dict()
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name].detach()
            if teacher_value.is_floating_point():
                teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                teacher_value.copy_(student_value)


def logits_distill_loss(student: Tensor, teacher: Tensor, temperature: float = 2.0) -> Tensor:
    teacher_prob = F.softmax(teacher.detach() / temperature, dim=1)
    return F.kl_div(
        F.log_softmax(student / temperature, dim=1), teacher_prob, reduction="batchmean"
    ) * (temperature**2)


def route_consistency_loss(student_routes: Sequence[Tensor], teacher_routes: Sequence[Tensor]) -> Tensor:
    if len(student_routes) != len(teacher_routes):
        raise ValueError("student and teacher must expose the same number of routes")
    losses = []
    for student, teacher in zip(student_routes, teacher_routes):
        losses.append(
            F.kl_div(
                (student.clamp_min(1e-8)).log(),
                teacher.detach().clamp_min(1e-8),
                reduction="batchmean",
            )
        )
    return torch.stack(losses).mean() if losses else torch.zeros((), device=student_routes[0].device)


def class_relation_distill_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Match pairwise class-logit geometry without adding deployment parameters."""
    student = F.normalize(student, dim=1)
    teacher = F.normalize(teacher.detach(), dim=1)
    return F.mse_loss(student @ student.transpose(0, 1), teacher @ teacher.transpose(0, 1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "TAMOGNet",
    "TAMOGBlock",
    "ModelEMA",
    "logits_distill_loss",
    "route_consistency_loss",
    "class_relation_distill_loss",
    "count_parameters",
]
