"""自适应置信先验（Adaptive Confidence Priors）用于UAM。

通过Margin、Norm、Attention三种证据构建像素级置信先验图A(p)∈[0,1]，
并以logit/prob/weight三种方式注入UAM的像素分布计算。

核心接口：
- robust_minmax: 稳健归一化
- margin_map: 分数间隔先验
- norm_map: 特征范数先验
- attention_map: 注意力/显著性先验
- build_confidence_prior: 组合先验构建
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def robust_minmax(
    x: torch.Tensor,
    p_low: float = 0.02,
    p_high: float = 0.98,
    eps: float = 1e-6,
) -> torch.Tensor:
    """百分位裁剪后的Min-Max归一化。
    
    参数
    ----
    x : torch.Tensor
        输入张量，形状 (H, W) 或 (B, H, W)
    p_low : float
        下百分位
    p_high : float
        上百分位
    eps : float
        数值稳定项
        
    返回
    ----
    torch.Tensor
        归一化后的张量，范围 [0, 1]
    """
    # 百分位裁剪
    q_low = torch.quantile(x.flatten(), p_low)
    q_high = torch.quantile(x.flatten(), p_high)
    
    x_clipped = x.clamp(min=q_low.item(), max=q_high.item())
    
    # Min-Max归一化
    x_min = x_clipped.min()
    x_max = x_clipped.max()
    
    if (x_max - x_min).abs() < eps:
        return torch.zeros_like(x_clipped)
    
    x_normalized = (x_clipped - x_min) / (x_max - x_min + eps)
    
    return x_normalized


@torch.no_grad()
def margin_map(
    score_maps: List[torch.Tensor],
    k: int = 2,
    tau: Optional[float] = None,
    mode: str = "logit",
) -> torch.Tensor:
    """计算像素级分数间隔（Margin）图。
    
    参数
    ----
    score_maps : List[torch.Tensor]
        相似度图列表，每个形状 (H, W)
    k : int
        top-k差值（2表示top1-top2）
    tau : Optional[float]
        温度参数（mode='logit'时使用）
    mode : str
        计算模式：'logit' 或 'prob'
        
    返回
    ----
    torch.Tensor
        Margin图，形状 (H, W)，未归一化
    """
    if len(score_maps) == 0:
        raise ValueError("score_maps cannot be empty")
    
    if len(score_maps) == 1:
        # 单张图，无margin
        return torch.zeros_like(score_maps[0])
    
    # 堆叠所有目标的分数
    scores_stacked = torch.stack(score_maps, dim=0)  # (T, H, W)
    
    if mode == "logit":
        # 对logits计算margin
        if tau is None:
            tau = 1.0
        logits = scores_stacked / tau
        
        # top-k
        topk_vals = torch.topk(logits, k=min(k, logits.shape[0]), dim=0).values
        
        if topk_vals.shape[0] >= 2:
            margin = topk_vals[0] - topk_vals[1]  # top1 - top2
        else:
            margin = topk_vals[0]
        
    elif mode == "prob":
        # 先softmax再计算margin
        probs = F.softmax(scores_stacked, dim=0)
        
        # top-k
        topk_vals = torch.topk(probs, k=min(k, probs.shape[0]), dim=0).values
        
        if topk_vals.shape[0] >= 2:
            margin = topk_vals[0] - topk_vals[1]
        else:
            margin = topk_vals[0]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return margin


@torch.no_grad()
def norm_map(
    feat: torch.Tensor,
    mode: str = "l2",
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算特征范数图。
    
    参数
    ----
    feat : torch.Tensor
        特征图，形状 (C, H, W)
    mode : str
        范数类型：'l2' 或 'l1'
    eps : float
        数值稳定项
        
    返回
    ----
    torch.Tensor
        范数图，形状 (H, W)，未归一化
    """
    if feat.dim() != 3:
        raise ValueError("feat must have shape (C, H, W)")
    
    if mode == "l2":
        norm = torch.sqrt((feat ** 2).sum(dim=0) + eps)
    elif mode == "l1":
        norm = feat.abs().sum(dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return norm


@torch.no_grad()
def attention_map(
    attn: Optional[torch.Tensor],
    feat: Optional[torch.Tensor] = None,
    fallback: str = "none",
) -> torch.Tensor:
    """构建注意力/显著性图。
    
    参数
    ----
    attn : Optional[torch.Tensor]
        注意力图，形状 (H, W) 或 (N, N) 需要聚合
    feat : Optional[torch.Tensor]
        特征图，形状 (C, H, W)（用于fallback）
    fallback : str
        回退策略：'grad'（特征平方和）或 'none'
        
    返回
    ----
    torch.Tensor
        注意力图，形状 (H, W)，未归一化
    """
    if attn is not None:
        if attn.dim() == 2:
            h, w = attn.shape
            if h == w:
                # 可能是 (H, W) 或 (N, N)
                # 假设已经是 (H, W)
                return attn
            else:
                # 尝试重塑
                n = h
                size = int(n ** 0.5)
                if size * size == n:
                    return attn.reshape(size, size)
                else:
                    return attn
        else:
            return attn.squeeze()
    
    # 回退策略
    if fallback == "grad" and feat is not None:
        # 使用特征平方和作为显著性近似
        saliency = (feat ** 2).sum(dim=0)
        return saliency
    elif fallback == "none":
        # 返回零图（不影响）
        if feat is not None:
            return torch.zeros(feat.shape[1:], device=feat.device, dtype=feat.dtype)
        else:
            # 无法确定尺寸，返回单元素零张量（调用方需处理）
            return torch.tensor(0.0)
    else:
        raise ValueError(f"Unknown fallback: {fallback}")


@torch.no_grad()
def build_confidence_prior(
    score_maps: List[torch.Tensor],
    feat: Optional[torch.Tensor] = None,
    attn: Optional[torch.Tensor] = None,
    tau: Optional[float] = None,
    norm_cfg: Dict = None,
    margin_cfg: Dict = None,
    attn_cfg: Dict = None,
    weights: Dict = None,
    combine: str = "logit",
    gamma: float = 1.0,
    scale: str = "percentile",
    p_low: float = 0.02,
    p_high: float = 0.98,
) -> Dict:
    """构建自适应置信先验图。
    
    组合Norm、Margin、Attention三种证据：
        A_raw(p) = a·Norm*(p) + b·Margin*(p) + c·Attn*(p) + d
        A(p) = σ(A_raw(p)) ∈ (0,1)
    
    参数
    ----
    score_maps : List[torch.Tensor]
        相似度图列表 R_{k,m}
    feat : Optional[torch.Tensor]
        特征图 (C, H, W)
    attn : Optional[torch.Tensor]
        注意力图
    tau : Optional[float]
        温度参数（margin计算用）
    norm_cfg : Dict
        Norm配置：{'mode': 'l2'}
    margin_cfg : Dict
        Margin配置：{'k': 2, 'mode': 'logit'}
    attn_cfg : Dict
        Attention配置：{'fallback': 'none'}
    weights : Dict
        权重字典：{'a': 0.5, 'b': 0.5, 'c': 0.0, 'd': 0.0}
    combine : str
        融合方式：'logit', 'prob', 'weight'
    gamma : float
        先验强度系数
    scale : str
        归一化方式：'percentile', 'minmax', 'zscore'
    p_low : float
        下百分位
    p_high : float
        上百分位
        
    返回
    ----
    Dict
        包含：
        - A: 置信先验图 (H, W)
        - parts: {'norm': ..., 'margin': ..., 'attn': ...}
        - gamma: 先验强度
        - combine: 融合方式
    """
    # 设置默认值
    norm_cfg = norm_cfg or {"mode": "l2"}
    margin_cfg = margin_cfg or {"k": 2, "mode": "logit"}
    attn_cfg = attn_cfg or {"fallback": "none"}
    weights = weights or {"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0}
    
    a = weights.get("a", 0.5)
    b = weights.get("b", 0.5)
    c = weights.get("c", 0.0)
    d = weights.get("d", 0.0)
    
    # 确定目标尺寸
    if len(score_maps) > 0:
        h, w = score_maps[0].shape
        device = score_maps[0].device
        dtype = score_maps[0].dtype
    elif feat is not None:
        h, w = feat.shape[1:]
        device = feat.device
        dtype = feat.dtype
    else:
        raise ValueError("Must provide score_maps or feat")
    
    parts = {}
    
    # 1. Norm component
    norm_raw = None
    if a > 0 and feat is not None:
        norm_raw = norm_map(feat, mode=norm_cfg.get("mode", "l2"))
        
        # 归一化
        if scale == "percentile":
            norm_normalized = robust_minmax(norm_raw, p_low=p_low, p_high=p_high)
        elif scale == "minmax":
            norm_normalized = (norm_raw - norm_raw.min()) / (norm_raw.max() - norm_raw.min() + 1e-8)
        elif scale == "zscore":
            norm_normalized = (norm_raw - norm_raw.mean()) / (norm_raw.std() + 1e-8)
            norm_normalized = norm_normalized.clamp(-3, 3) / 6 + 0.5  # 映射到[0,1]
        else:
            norm_normalized = robust_minmax(norm_raw, p_low=p_low, p_high=p_high)
        
        parts["norm"] = norm_normalized
    else:
        parts["norm"] = torch.zeros((h, w), device=device, dtype=dtype)
    
    # 2. Margin component
    margin_raw = None
    if b > 0 and len(score_maps) > 0:
        margin_raw = margin_map(
            score_maps,
            k=margin_cfg.get("k", 2),
            tau=tau,
            mode=margin_cfg.get("mode", "logit"),
        )
        
        # 归一化
        if scale == "percentile":
            margin_normalized = robust_minmax(margin_raw, p_low=p_low, p_high=p_high)
        else:
            margin_normalized = (margin_raw - margin_raw.min()) / (margin_raw.max() - margin_raw.min() + 1e-8)
        
        parts["margin"] = margin_normalized
    else:
        parts["margin"] = torch.zeros((h, w), device=device, dtype=dtype)
    
    # 3. Attention component
    attn_raw = None
    if c > 0:
        attn_raw = attention_map(
            attn,
            feat=feat,
            fallback=attn_cfg.get("fallback", "none"),
        )
        
        # 确保尺寸匹配
        if attn_raw.shape != (h, w):
            attn_raw = F.interpolate(
                attn_raw.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        
        # 归一化
        if scale == "percentile":
            attn_normalized = robust_minmax(attn_raw, p_low=p_low, p_high=p_high)
        else:
            attn_normalized = (attn_raw - attn_raw.min()) / (attn_raw.max() - attn_raw.min() + 1e-8)
        
        parts["attn"] = attn_normalized
    else:
        parts["attn"] = torch.zeros((h, w), device=device, dtype=dtype)
    
    # 组合
    A_raw = a * parts["norm"] + b * parts["margin"] + c * parts["attn"] + d
    
    # Sigmoid映射到(0,1)
    A = torch.sigmoid(A_raw)
    
    return {
        "A": A,
        "parts": parts,
        "gamma": gamma,
        "combine": combine,
    }


__all__ = [
    "robust_minmax",
    "margin_map",
    "norm_map",
    "attention_map",
    "build_confidence_prior",
]

