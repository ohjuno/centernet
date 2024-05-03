from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_n_tuple


__all__ = ["ConvNormAct", "UpConv"]


# Types

Tensor = torch.Tensor


# Modules

class ConvNormAct(nn.Module):

    def __init__(
        self,
        i: int,
        o: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool | None = None,
        normalization: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        inplace: bool | None = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                conv_nd = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = make_n_tuple(kernel_size, conv_nd)
                dilation = make_n_tuple(dilation, conv_nd)
                padding = (kernel_size[i] - 1 // 2 * dilation[i] for i in range(conv_nd))

        conv = nn.Conv2d

        if bias is None:
            bias = normalization is None

        layers: list[nn.Module] = [
            conv(i, o, kernel_size, stride, padding, dilation, groups, bias, **factory_kwargs)
        ]

        if normalization is not None:
            layers.append(normalization(o, **factory_kwargs))

        if activation is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation(**params))

        self.layer = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class UpConv(nn.Module):

    def __init__(self, i: int, o: int, activation: Callable[..., nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvNormAct(i, o, 3, 1, 1, activation=activation),
            ConvNormAct(o, o, 1, 1, 0, activation=activation),
        )

    def forward(self, u: Tensor, l: Tensor) -> Tensor:  # noqa: E741
        x = F.interpolate(l, u.shape[2:], mode="nearest")  # noqa: E741
        x = torch.cat([u, x], 1)
        x = self.conv(x)
        return x
