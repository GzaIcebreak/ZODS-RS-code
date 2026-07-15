"""Utilities for constructing prior maps used by UAM merging.

提供三类训练无关、设备自适应的先验图构造方法：

``margin_prior``
    基于图像边缘的距离模拟边界衰减。

``norm_prior``
    根据掩码质心的归一化距离生成高斯式先验。

``combine_priors``
    按照乘积或平均方式组合多个先验图。
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence

import torch


def _ensure_3d(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        return mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError("mask must have shape (N,H,W) or (H,W)")
    return mask


def margin_prior(
    masks: torch.Tensor,
    margin: float = 0.1,
    sharpen: float = 2.0,
) -> torch.Tensor:
    """生成边缘衰减先验。

    Parameters
    ----------
    masks:
        掩码张量 ``(N,H,W)`` 或 ``(H,W)``。
    margin:
        边缘宽度占输入尺寸的比例。
    sharpen:
        边界衰减的锐化系数。
    """

    masks = _ensure_3d(masks)
    margin = float(max(0.0, min(0.5, margin)))
    sharpen = float(max(0.1, sharpen))

    n, h, w = masks.shape
    device = masks.device

    # 生成像素到四条边界的归一化距离
    y = torch.linspace(0.0, 1.0, steps=h, device=device)
    x = torch.linspace(0.0, 1.0, steps=w, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    dist = torch.stack((grid_y, 1.0 - grid_y, grid_x, 1.0 - grid_x), dim=0)
    dist_min = dist.min(dim=0).values  # (H, W)

    margin_mask = (dist_min >= margin).float()
    prior = margin_mask + (1.0 - margin_mask) * (dist_min / margin).pow(sharpen)
    return (prior.unsqueeze(0) * masks).clamp(0.0, 1.0)


def norm_prior(
    masks: torch.Tensor,
    center: Optional[Sequence[float]] = None,
    sigma: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """基于掩码质心的高斯先验。

    Parameters
    ----------
    masks:
        掩码张量。
    center:
        归一化坐标 ``(cy, cx)``，默认从掩码估计。
    sigma:
        高斯标准差，控制衰减速度。
    """

    masks = _ensure_3d(masks)
    sigma = float(max(eps, sigma))

    n, h, w = masks.shape
    device = masks.device

    y = torch.linspace(0.0, 1.0, steps=h, device=device)
    x = torch.linspace(0.0, 1.0, steps=w, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    if center is None:
        mass = masks.sum(dim=(-1, -2), keepdim=True).clamp(min=eps)
        cy = (masks * grid_y).sum(dim=(-1, -2), keepdim=True) / mass
        cx = (masks * grid_x).sum(dim=(-1, -2), keepdim=True) / mass
    else:
        cy, cx = map(float, center)
        cy = torch.full((n, 1, 1), cy, device=device)
        cx = torch.full((n, 1, 1), cx, device=device)

    dist_sq = (grid_y - cy).pow(2) + (grid_x - cx).pow(2)
    prior = torch.exp(-dist_sq / (2 * sigma ** 2))
    return (prior * masks).clamp(0.0, 1.0)


def combine_priors(
    priors: Iterable[torch.Tensor],
    mode: Literal["multiply", "mean", "augment"] = "multiply",
    eps: float = 1e-6,
) -> torch.Tensor:
    """组合多个先验图。

    Parameters
    ----------
    priors:
        形状一致的先验张量序列。
    mode:
        ``"multiply"`` 表示乘法叠加，``"mean"`` 表示算术平均，
        ``"augment"`` 表示将多个先验视为互补证据（概率并集）。
    eps:
        防止乘法过程中的数值下溢。
    """

    priors = list(priors)
    if not priors:
        raise ValueError("priors must be non-empty")

    base = priors[0]
    for p in priors[1:]:
        if p.shape != base.shape:
            raise ValueError("all priors must share identical shape")

    if mode == "multiply":
        out = torch.ones_like(base)
        for p in priors:
            out = out * p.clamp(min=eps, max=1.0)
    elif mode == "mean":
        out = torch.stack(priors, dim=0).mean(dim=0)
    elif mode == "augment":
        comp = torch.ones_like(base)
        for p in priors:
            comp = comp * (1.0 - p.clamp(min=0.0, max=1.0))
        out = 1.0 - comp
    else:
        raise ValueError(f"Unsupported combine mode: {mode}")

    return out.clamp(0.0, 1.0)


__all__ = ["margin_prior", "norm_prior", "combine_priors"]


