from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from deployable_medsam.data.prompts import BoxPrompt


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class LightweightUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.in_conv = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.up1 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up3 = UpBlock(c2, c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out_conv(x)


class PromptableLightweightUNet(LightweightUNet):
    """Optional prompt-conditioned U-Net using one extra binary prompt channel.

    This class keeps the current image-only student untouched. It is available
    for a later ablation where the compact student receives an explicit box
    prompt mask, closer to SAM/MedSAM's promptable interface.
    """

    def __init__(self, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__(in_channels=4, out_channels=out_channels, base_channels=base_channels)

    def forward(self, image: torch.Tensor, prompt_mask: torch.Tensor | None = None) -> torch.Tensor:
        if prompt_mask is None:
            prompt_mask = torch.zeros(
                (image.shape[0], 1, image.shape[-2], image.shape[-1]),
                dtype=image.dtype,
                device=image.device,
            )
        if prompt_mask.ndim != 4 or prompt_mask.shape[1] != 1:
            raise ValueError("prompt_mask must have shape [batch, 1, height, width].")
        if prompt_mask.shape[0] != image.shape[0] or prompt_mask.shape[-2:] != image.shape[-2:]:
            raise ValueError("prompt_mask batch and spatial dimensions must match image.")
        return super().forward(torch.cat([image, prompt_mask.to(image.dtype)], dim=1))


def box_prompt_to_mask(box: BoxPrompt, *, height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    mask = torch.zeros((1, height, width), dtype=dtype or torch.float32, device=device)
    x_min = max(0, min(width - 1, int(box.x_min)))
    x_max = max(0, min(width - 1, int(box.x_max)))
    y_min = max(0, min(height - 1, int(box.y_min)))
    y_max = max(0, min(height - 1, int(box.y_max)))
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    mask[:, y_min : y_max + 1, x_min : x_max + 1] = 1
    return mask


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
