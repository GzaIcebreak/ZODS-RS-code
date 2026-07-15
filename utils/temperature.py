"""Temperature scaling utilities for uncertainty-aware merging.

自动温度调节工具，用于优化 softmax 分布的锐度与平滑度平衡。

Functions
---------
compute_auto_tau
    根据logits统计自动计算最优温度参数。

summarize_smax  
    分析softmax分布特征，输出统计摘要。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_auto_tau(
    logits: torch.Tensor,
    method: str = "entropy_target", 
    target_entropy: float = 0.5,
    percentile: float = 90.0,
    tau_range: Tuple[float, float] = (0.1, 3.0),
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """自动计算温度参数以达到目标分布特性。
    
    Parameters
    ----------
    logits:
        输入logits，形状 (B, C, H, W) 或 (C, H, W)
    method:
        计算方法，'entropy_target' | 'variance_based' | 'confidence_percentile'
    target_entropy:
        目标熵值 (仅用于 entropy_target 方法)
    percentile:
        置信度百分位 (仅用于 confidence_percentile 方法)
    tau_range:
        温度搜索范围
    eps:
        数值稳定项

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (最优温度, 统计信息字典)
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if logits.dim() != 4:
        raise ValueError("logits must be 3D or 4D")

    tau_min, tau_max = tau_range
    device = logits.device
    
    # 候选温度
    tau_candidates = torch.linspace(tau_min, tau_max, 20, device=device)
    
    if method == "entropy_target":
        best_tau = _find_entropy_target_tau(logits, tau_candidates, target_entropy, eps)
    elif method == "variance_based":
        best_tau = _find_variance_optimal_tau(logits, tau_candidates, eps)
    elif method == "confidence_percentile":
        best_tau = _find_percentile_tau(logits, tau_candidates, percentile, eps)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 计算最终统计
    final_probs = F.softmax(logits / best_tau, dim=1)
    stats = _compute_distribution_stats(final_probs, eps)
    
    return best_tau.item(), stats


def summarize_smax(
    probs: torch.Tensor,
    eps: float = 1e-6
) -> Dict[str, float]:
    """分析softmax分布的统计特征。
    
    Parameters
    ----------  
    probs:
        概率分布，形状 (B, C, H, W) 或 (C, H, W)
    eps:
        数值稳定项
        
    Returns
    -------
    Dict[str, float]
        包含熵、方差、峰度等统计信息
    """
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)
    
    return _compute_distribution_stats(probs, eps)


def _find_entropy_target_tau(
    logits: torch.Tensor, 
    tau_candidates: torch.Tensor, 
    target_entropy: float,
    eps: float
) -> torch.Tensor:
    """寻找使平均熵接近目标值的温度。"""
    best_tau = tau_candidates[0]
    best_diff = float('inf')
    
    for tau in tau_candidates:
        probs = F.softmax(logits / tau, dim=1)
        entropy = -(probs * torch.log(probs.clamp(min=eps))).sum(dim=1)
        avg_entropy = entropy.mean().item()
        diff = abs(avg_entropy - target_entropy)
        
        if diff < best_diff:
            best_diff = diff
            best_tau = tau
            
    return best_tau


def _find_variance_optimal_tau(
    logits: torch.Tensor,
    tau_candidates: torch.Tensor,
    eps: float
) -> torch.Tensor:
    """寻找使概率方差最优的温度。"""
    max_variance = 0.0
    best_tau = tau_candidates[0]
    
    for tau in tau_candidates:
        probs = F.softmax(logits / tau, dim=1)
        # 计算每个像素的概率方差
        prob_var = probs.var(dim=1).mean().item()
        
        if prob_var > max_variance:
            max_variance = prob_var
            best_tau = tau
            
    return best_tau


def _find_percentile_tau(
    logits: torch.Tensor,
    tau_candidates: torch.Tensor, 
    percentile: float,
    eps: float
) -> torch.Tensor:
    """寻找使指定百分位置信度达到目标的温度。"""
    target_conf = percentile / 100.0
    best_tau = tau_candidates[0]
    best_diff = float('inf')
    
    for tau in tau_candidates:
        probs = F.softmax(logits / tau, dim=1)
        max_probs = probs.max(dim=1).values
        p_val = torch.quantile(max_probs, percentile / 100.0).item()
        diff = abs(p_val - target_conf)
        
        if diff < best_diff:
            best_diff = diff
            best_tau = tau
            
    return best_tau


def _compute_distribution_stats(
    probs: torch.Tensor, 
    eps: float
) -> Dict[str, float]:
    """计算概率分布的统计特征。"""
    # 熵计算
    entropy = -(probs * torch.log(probs.clamp(min=eps))).sum(dim=1)
    
    # 最大概率 (置信度)
    max_probs = probs.max(dim=1).values
    
    # 方差计算
    prob_variance = probs.var(dim=1)
    
    # 能量计算 (负对数似然的近似)
    energy = -torch.logsumexp(probs.log().clamp(min=-50), dim=1)
    
    return {
        "entropy_mean": entropy.mean().item(),
        "entropy_std": entropy.std().item(),
        "confidence_mean": max_probs.mean().item(),
        "confidence_std": max_probs.std().item(),
        "confidence_p50": torch.quantile(max_probs, 0.5).item(),
        "confidence_p90": torch.quantile(max_probs, 0.9).item(),
        "variance_mean": prob_variance.mean().item(),
        "energy_mean": energy.mean().item(),
        "energy_std": energy.std().item(),
    }


__all__ = ["compute_auto_tau", "summarize_smax"]
