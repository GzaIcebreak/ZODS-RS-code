"""一致性加权的多层聚合（Consistency-Weighted Layer Aggregation, CWLA）。

通过层间一致性/不确定性度量生成权重，闭式融合多层相似度图与不确定性图。
适用于DINOv3等多层视觉Transformer的特征融合。

核心接口：
- layer_consistency_weights: 计算层权重β
- fuse_layers_with_weights: 加权融合多层相似度
- fuse_uncertainty_layers: 加权融合多层不确定性
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def layer_consistency_weights(
    R_per_layer: List[List[torch.Tensor]],
    tau_mode: str = "auto",
    tau: Optional[float] = None,
    tau_strategy: str = "std",
    tau_k: float = 1.0,
    tau_p: float = 0.85,
    metric: str = "entropy",
    sigma: float = 0.15,
    smooth: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算每层的一致性权重。
    
    基于层间不确定性差异生成权重：
        β_ℓ = softmax_ℓ( -ȗ^[ℓ] / σ )
    
    其中 ȗ^[ℓ] 为该层的平均不确定性度量。
    
    参数
    ----
    R_per_layer : List[List[torch.Tensor]]
        结构 [layers][targets]，每个元素形状 (H, W)
    tau_mode : str
        温度模式：'auto'（自动计算）或 'fixed'
    tau : Optional[float]
        固定温度值（tau_mode='fixed'时使用）
    tau_strategy : str
        自动温度策略：'std', 'mad', 'percentile-std'
    tau_k : float
        温度计算的缩放因子
    tau_p : float
        百分位参数
    metric : str
        一致性度量：'entropy', 'margin', 'jsd', 'agreement'
    sigma : float
        softmax温度（控制权重集中度）
    smooth : float
        对ȗ进行1D平滑的强度（0=不平滑）
        
    返回
    ----
    Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        (β, info)
        - β: 层权重 (num_layers,)
        - info: 调试信息（U_layers, tau等）
    """
    num_layers = len(R_per_layer)
    
    if num_layers == 0:
        raise ValueError("R_per_layer cannot be empty")
    
    device = R_per_layer[0][0].device if R_per_layer[0] else torch.device("cpu")
    
    # 计算每层的不确定性度量
    U_per_layer = []
    
    for layer_idx, layer_targets in enumerate(R_per_layer):
        if len(layer_targets) == 0:
            # 空层，使用高不确定性
            U_per_layer.append(1.0)
            continue
        
        # 堆叠所有目标的相似度图
        R_stacked = torch.stack(layer_targets, dim=0)  # (T, H, W)
        
        if metric == "entropy":
            # 计算像素级熵
            # 先转为概率分布
            if tau_mode == "auto":
                from utils.temperature import compute_auto_tau
                # tau_strategy 应该映射到 method
                if tau_strategy in ["std", "mad"]:
                    method_for_tau = "entropy_target"
                elif tau_strategy == "percentile-std":
                    method_for_tau = "confidence_percentile"
                else:
                    method_for_tau = tau_strategy
                
                tau_val, _ = compute_auto_tau(
                    R_stacked.unsqueeze(0),  # (1, T, H, W)
                    method=method_for_tau,
                    target_entropy=0.5,
                    percentile=tau_p * 100,
                    tau_range=(0.1, 3.0),
                )
            else:
                tau_val = tau if tau is not None else 1.0
            
            # Softmax 跨目标维度
            probs = F.softmax(R_stacked / tau_val, dim=0)  # (T, H, W)
            
            # 像素级熵
            log_probs = torch.log(probs.clamp(min=1e-10))
            entropy = -(probs * log_probs).sum(dim=0)  # (H, W)
            
            # 平均熵作为该层的不确定性
            U_layer = entropy.mean().item()
            
        elif metric == "margin":
            # top1 - top2 margin（负号使大margin→小不确定）
            top2_vals = R_stacked.topk(k=min(2, R_stacked.shape[0]), dim=0).values
            if top2_vals.shape[0] >= 2:
                margin = top2_vals[0] - top2_vals[1]  # (H, W)
                U_layer = -margin.mean().item()  # 负号转换
            else:
                U_layer = 0.0
        
        elif metric == "jsd":
            # Jensen-Shannon 散度（简化：用熵近似）
            if R_stacked.shape[0] >= 2:
                probs = F.softmax(R_stacked, dim=0)
                log_probs = torch.log(probs.clamp(min=1e-10))
                entropy = -(probs * log_probs).sum(dim=0).mean().item()
                U_layer = entropy
            else:
                U_layer = 0.0
        
        elif metric == "agreement":
            # argmax 一致性
            if R_stacked.shape[0] >= 2:
                argmax_labels = R_stacked.argmax(dim=0)  # (H, W)
                # 模式（出现最多的标签）
                mode_val = argmax_labels.mode().values.item()
                agreement = (argmax_labels == mode_val).float().mean().item()
                U_layer = 1.0 - agreement  # 一致率高→不确定性低
            else:
                U_layer = 0.0
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        U_per_layer.append(U_layer)
    
    # 转为tensor
    U_tensor = torch.tensor(U_per_layer, device=device, dtype=torch.float32)
    
    # 可选平滑
    if smooth > 0 and num_layers > 1:
        # 简单1D平滑（移动平均）
        kernel_size = 3
        padding = kernel_size // 2
        U_smoothed = F.avg_pool1d(
            U_tensor.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ).squeeze()
        U_tensor = (1 - smooth) * U_tensor + smooth * U_smoothed
    
    # 计算权重 β = softmax(-U / σ)
    beta = F.softmax(-U_tensor / (sigma + 1e-8), dim=0)
    
    info = {
        "U_layers": U_tensor,
        "metric": metric,
        "sigma": sigma,
        "tau": tau_val if metric == "entropy" and tau_mode == "auto" else tau,
    }
    
    return beta, info


@torch.no_grad()
def fuse_layers_with_weights(
    R_per_layer: List[List[torch.Tensor]],
    beta: torch.Tensor,
    base_size: Optional[Tuple[int, int]] = None,
    dtype: Optional[torch.dtype] = None,
) -> List[torch.Tensor]:
    """按权重融合多层相似度图。
    
    对每个目标t：R_t = Σ_ℓ β_ℓ · Up(R_t^[ℓ])
    
    参数
    ----
    R_per_layer : List[List[torch.Tensor]]
        结构 [layers][targets]
    beta : torch.Tensor
        层权重，形状 (num_layers,)
    base_size : Optional[Tuple[int, int]]
        目标分辨率（默认为最大尺寸）
    dtype : Optional[torch.dtype]
        输出数据类型
        
    返回
    ----
    List[torch.Tensor]
        融合后的相似度图列表，每个目标一个 (H, W)
    """
    num_layers = len(R_per_layer)
    
    if num_layers == 0:
        return []
    
    if beta.shape[0] != num_layers:
        raise ValueError("beta length must match number of layers")
    
    # 确定目标数量与尺寸
    num_targets = len(R_per_layer[0]) if R_per_layer[0] else 0
    
    if num_targets == 0:
        return []
    
    if base_size is None:
        # 找最大尺寸
        max_h = max(R[0].shape[0] for R in R_per_layer if len(R) > 0)
        max_w = max(R[0].shape[1] for R in R_per_layer if len(R) > 0)
        base_size = (max_h, max_w)
    
    device = R_per_layer[0][0].device
    out_dtype = dtype or R_per_layer[0][0].dtype
    
    fused_list = []
    
    # 对每个目标融合
    for t_idx in range(num_targets):
        R_fused = torch.zeros(base_size, device=device, dtype=torch.float32)
        
        for layer_idx in range(num_layers):
            if t_idx >= len(R_per_layer[layer_idx]):
                continue
            
            R_layer_t = R_per_layer[layer_idx][t_idx]
            
            # 上采样到base_size
            if R_layer_t.shape != base_size:
                R_up = F.interpolate(
                    R_layer_t.unsqueeze(0).unsqueeze(0),
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            else:
                R_up = R_layer_t
            
            # 加权累加
            R_fused += beta[layer_idx] * R_up
        
        # 转换数据类型
        if out_dtype != torch.float32:
            R_fused = R_fused.to(dtype=out_dtype)
        
        fused_list.append(R_fused)
    
    return fused_list


@torch.no_grad()
def fuse_uncertainty_layers(
    U_layers: List[torch.Tensor],
    beta: torch.Tensor,
    mode: str = "weighted_sum",
) -> torch.Tensor:
    """融合多层不确定性图。
    
    参数
    ----
    U_layers : List[torch.Tensor]
        每层的不确定性图，形状 (H, W)
    beta : torch.Tensor
        层权重，形状 (num_layers,)
    mode : str
        融合模式：'weighted_sum' 或 'jsd'
        
    返回
    ----
    torch.Tensor
        融合后的不确定性图，形状 (H, W)
    """
    if len(U_layers) == 0:
        raise ValueError("U_layers cannot be empty")
    
    if mode == "weighted_sum":
        # 默认：加权和
        # 确保尺寸一致
        base_size = U_layers[0].shape
        device = U_layers[0].device
        
        U_fused = torch.zeros(base_size, device=device, dtype=torch.float32)
        
        for layer_idx, U_layer in enumerate(U_layers):
            if U_layer.shape != base_size:
                U_up = F.interpolate(
                    U_layer.unsqueeze(0).unsqueeze(0),
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            else:
                U_up = U_layer
            
            U_fused += beta[layer_idx] * U_up
        
        return U_fused
        
    elif mode == "jsd":
        # JSD 模式（层间差异作为不确定性）
        # 简化：计算各层的方差作为不一致性
        base_size = U_layers[0].shape
        device = U_layers[0].device
        
        # 堆叠所有层
        U_stacked = []
        for U_layer in U_layers:
            if U_layer.shape != base_size:
                U_up = F.interpolate(
                    U_layer.unsqueeze(0).unsqueeze(0),
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            else:
                U_up = U_layer
            U_stacked.append(U_up)
        
        U_stacked = torch.stack(U_stacked, dim=0)  # (num_layers, H, W)
        
        # 像素级方差（层间不一致性）
        U_var = U_stacked.var(dim=0)  # (H, W)
        
        # 结合加权平均
        U_mean = (U_stacked * beta.view(-1, 1, 1)).sum(dim=0)
        
        # 组合
        U_fused = U_mean + U_var * 0.5  # 可调权重
        
        return U_fused
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


__all__ = [
    "layer_consistency_weights",
    "fuse_layers_with_weights",
    "fuse_uncertainty_layers",
]

