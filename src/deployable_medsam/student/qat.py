from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .model import LightweightUNet


class FakeQuantizedConv2d(nn.Module):
    """Conv2d wrapper with straight-through fake quantization for QAT."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = copy.deepcopy(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qx = fake_quantize_tensor(x, signed=False)
        qw = fake_quantize_tensor(self.conv.weight, signed=True)
        output = F.conv2d(
            qx,
            qw,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return fake_quantize_tensor(output, signed=True)


def fake_quantize_tensor(values: torch.Tensor, *, bits: int = 8, signed: bool = False, eps: float = 1e-8) -> torch.Tensor:
    qmin = -(2 ** (bits - 1)) if signed else 0
    qmax = (2 ** (bits - 1) - 1) if signed else (2**bits - 1)
    min_value = values.detach().amin()
    max_value = values.detach().amax()
    if signed:
        max_abs = torch.maximum(min_value.abs(), max_value.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        dequantized = torch.clamp(torch.round(values / scale), qmin, qmax) * scale
    else:
        scale = ((max_value - min_value).clamp_min(eps)) / float(qmax - qmin)
        zero_point = torch.clamp(torch.round(qmin - min_value / scale), qmin, qmax)
        dequantized = (torch.clamp(torch.round(values / scale + zero_point), qmin, qmax) - zero_point) * scale
    return values + (dequantized - values).detach()


def prepare_fake_quant_unet(model: LightweightUNet) -> LightweightUNet:
    qat_model = copy.deepcopy(model)
    _replace_convs_with_fake_quant(qat_model)
    return qat_model


def export_clean_unet_from_fake_quant(qat_model: nn.Module, *, base_channels: int = 16) -> LightweightUNet:
    clean = LightweightUNet(base_channels=base_channels)
    source_modules = dict(qat_model.named_modules())
    for name, target in clean.named_modules():
        if name == "":
            continue
        source = source_modules.get(name)
        if source is None:
            continue
        if isinstance(target, nn.Conv2d):
            source_conv = source.conv if isinstance(source, FakeQuantizedConv2d) else source
            if not isinstance(source_conv, nn.Conv2d):
                continue
            target.weight.data.copy_(source_conv.weight.detach().cpu())
            if target.bias is not None and source_conv.bias is not None:
                target.bias.data.copy_(source_conv.bias.detach().cpu())
        elif isinstance(target, nn.BatchNorm2d) and isinstance(source, nn.BatchNorm2d):
            target.weight.data.copy_(source.weight.detach().cpu())
            target.bias.data.copy_(source.bias.detach().cpu())
            target.running_mean.data.copy_(source.running_mean.detach().cpu())
            target.running_var.data.copy_(source.running_var.detach().cpu())
            target.num_batches_tracked.data.copy_(source.num_batches_tracked.detach().cpu())
    return clean


def _replace_convs_with_fake_quant(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, FakeQuantizedConv2d(child))
        else:
            _replace_convs_with_fake_quant(child)
