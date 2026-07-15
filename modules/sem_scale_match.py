"""Semantic Scale-aware Matching (SEM) utilities.

多尺度语义匹配模块，用于在不同尺度上计算原型与目标特征的匹配度，
并通过匈牙利算法进行最优分配。

主要接口：
-----------
build_feature_pyramid
    构建多尺度特征金字塔

multiscale_similarity
    计算跨尺度的相似度矩阵

fuse_scales_with_kernel
    使用可学习核或固定权重融合多尺度结果

build_cost_matrix
    构建用于匈牙利算法的代价矩阵

hungarian_assign
    执行匈牙利最优分配

match_one_prototype
    单个原型的匹配流程

match_all
    批量处理所有原型的匹配
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import warnings

import torch
import torch.nn.functional as F

# Optional dependencies
try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, will use greedy assignment", RuntimeWarning)

try:
    import kornia.geometry.transform as K_transform
    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False


@dataclass
class SEMMatchResult:
    """SEM 匹配结果容器。
    
    Attributes
    ----------
    matched_indices : torch.Tensor
        匹配的目标索引，形状 (N,)
    similarity_scores : torch.Tensor
        匹配相似度分数，形状 (N,)
    scale_weights : Optional[torch.Tensor]
        各尺度的融合权重，形状 (N, num_scales)
    cost_matrix : Optional[torch.Tensor]
        完整代价矩阵，用于调试
    """
    matched_indices: torch.Tensor
    similarity_scores: torch.Tensor
    scale_weights: Optional[torch.Tensor] = None
    cost_matrix: Optional[torch.Tensor] = None


def build_feature_pyramid(
    features: torch.Tensor,
    scales: List[float] = [1.0, 0.5, 0.25],
    mode: str = "bilinear",
) -> List[torch.Tensor]:
    """构建多尺度特征金字塔。
    
    Parameters
    ----------
    features : torch.Tensor
        输入特征图，形状 (B, C, H, W) 或 (C, H, W)
    scales : List[float]
        尺度因子列表，1.0 表示原始尺度
    mode : str
        插值模式，'bilinear' 或 'nearest'
        
    Returns
    -------
    List[torch.Tensor]
        多尺度特征列表，每个元素形状 (B, C, H_s, W_s)
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    pyramid = []
    b, c, h, w = features.shape
    
    for scale in scales:
        if scale == 1.0:
            pyramid.append(features)
        else:
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            scaled = F.interpolate(
                features,
                size=(new_h, new_w),
                mode=mode,
                align_corners=False if mode == "bilinear" else None,
            )
            pyramid.append(scaled)
    
    return pyramid


def multiscale_similarity(
    proto_pyramid: List[torch.Tensor],
    target_pyramid: List[torch.Tensor],
    normalize: bool = True,
    eps: float = 1e-6,
) -> List[torch.Tensor]:
    """计算多尺度相似度。
    
    Parameters
    ----------
    proto_pyramid : List[torch.Tensor]
        原型特征金字塔，每个 (1, C, H_p, W_p) 或 (C, H_p, W_p)
    target_pyramid : List[torch.Tensor]
        目标特征金字塔，每个 (1, C, H_t, W_t) 或 (C, H_t, W_t)
    normalize : bool
        是否 L2 归一化特征
    eps : float
        归一化稳定项
        
    Returns
    -------
    List[torch.Tensor]
        每个尺度的相似度图，形状 (H_t, W_t)
    """
    if len(proto_pyramid) != len(target_pyramid):
        raise ValueError("Pyramid lengths must match")
    
    similarities = []
    
    for proto_feat, target_feat in zip(proto_pyramid, target_pyramid):
        # 确保 4D
        if proto_feat.dim() == 3:
            proto_feat = proto_feat.unsqueeze(0)
        if target_feat.dim() == 3:
            target_feat = target_feat.unsqueeze(0)
        
        # 全局池化原型
        proto_pooled = F.adaptive_avg_pool2d(proto_feat, (1, 1))  # (1, C, 1, 1)
        
        if normalize:
            proto_pooled = F.normalize(proto_pooled, p=2, dim=1, eps=eps)
            target_feat = F.normalize(target_feat, p=2, dim=1, eps=eps)
        
        # 逐像素相似度
        sim = (proto_pooled * target_feat).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        similarities.append(sim.squeeze())  # (H, W)
    
    return similarities


def fuse_scales_with_kernel(
    sim_maps: List[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    target_size: Optional[Tuple[int, int]] = None,
    mode: str = "learned",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """融合多尺度相似度图。
    
    Parameters
    ----------
    sim_maps : List[torch.Tensor]
        各尺度的相似度图
    weights : Optional[torch.Tensor]
        尺度权重，形状 (num_scales,)，若为 None 则均匀权重
    target_size : Optional[Tuple[int, int]]
        目标输出尺寸 (H, W)
    mode : str
        融合模式：'learned', 'uniform', 'variance'
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (融合后的相似度图, 尺度权重)
    """
    num_scales = len(sim_maps)
    
    if target_size is None:
        target_size = sim_maps[0].shape[-2:]
    
    # 统一尺寸
    resized_maps = []
    for sim in sim_maps:
        if sim.shape[-2:] != target_size:
            sim_4d = sim.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            sim_resized = F.interpolate(
                sim_4d, size=target_size, mode="bilinear", align_corners=False
            )
            resized_maps.append(sim_resized.squeeze())
        else:
            resized_maps.append(sim)
    
    # 计算权重
    if mode == "uniform" or weights is None:
        weights = torch.ones(num_scales, device=sim_maps[0].device) / num_scales
    elif mode == "variance":
        # 方差加权：方差越大，权重越高（信息量更丰富）
        variances = torch.tensor([s.var().item() for s in resized_maps], device=sim_maps[0].device)
        weights = F.softmax(variances, dim=0)
    elif mode == "learned" and weights is not None:
        weights = F.softmax(weights, dim=0)
    
    # 加权融合
    stacked = torch.stack(resized_maps, dim=0)  # (num_scales, H, W)
    fused = (stacked * weights.view(-1, 1, 1)).sum(dim=0)  # (H, W)
    
    return fused, weights


def build_cost_matrix(
    proto_features: torch.Tensor,
    target_features: torch.Tensor,
    similarity_maps: List[torch.Tensor],
    alpha: float = 0.5,
    normalize: bool = True,
) -> torch.Tensor:
    """构建匹配代价矩阵。
    
    Parameters
    ----------
    proto_features : torch.Tensor
        原型特征，形状 (N_proto, C)
    target_features : torch.Tensor
        目标特征，形状 (N_target, C)
    similarity_maps : List[torch.Tensor]
        多尺度相似度图（用于空间加权）
    alpha : float
        特征相似度权重，范围 [0, 1]
    normalize : bool
        是否归一化特征
        
    Returns
    -------
    torch.Tensor
        代价矩阵，形状 (N_proto, N_target)，值越小越相似
    """
    if normalize:
        proto_features = F.normalize(proto_features, p=2, dim=-1)
        target_features = F.normalize(target_features, p=2, dim=-1)
    
    # 特征相似度
    feat_sim = proto_features @ target_features.t()  # (N_proto, N_target)
    feat_sim = (feat_sim + 1.0) * 0.5  # 归一化到 [0, 1]
    
    # 空间相似度（从多尺度图采样）
    if similarity_maps:
        spatial_sim = similarity_maps[0].mean()  # 简化：使用平均值
    else:
        spatial_sim = 0.0
    
    # 组合代价（越小越好，所以取负）
    cost = -(alpha * feat_sim + (1 - alpha) * spatial_sim)
    
    return cost


def hungarian_assign(
    cost_matrix: torch.Tensor,
    return_cost: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """匈牙利算法最优分配。
    
    Parameters
    ----------
    cost_matrix : torch.Tensor
        代价矩阵，形状 (N, M)
    return_cost : bool
        是否返回分配代价
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (row_indices, col_indices) 或 (row_indices, col_indices, total_cost)
    """
    if not _SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for hungarian assignment")
    
    cost_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    
    row_ind = torch.from_numpy(row_ind).to(cost_matrix.device)
    col_ind = torch.from_numpy(col_ind).to(cost_matrix.device)
    
    if return_cost:
        total_cost = cost_matrix[row_ind, col_ind].sum()
        return row_ind, col_ind, total_cost
    
    return row_ind, col_ind


def match_one_prototype(
    proto_feat: torch.Tensor,
    target_feats: torch.Tensor,
    scales: List[float] = [1.0, 0.5],
    alpha: float = 0.5,
    top_k: int = 5,
) -> SEMMatchResult:
    """单个原型的多尺度匹配。
    
    Parameters
    ----------
    proto_feat : torch.Tensor
        原型特征，形状 (C, H_p, W_p) 或 (1, C, H_p, W_p)
    target_feats : torch.Tensor
        目标特征图，形状 (C, H_t, W_t) 或 (1, C, H_t, W_t)
    scales : List[float]
        尺度列表
    alpha : float
        特征-空间权重平衡
    top_k : int
        返回top-k匹配结果
        
    Returns
    -------
    SEMMatchResult
        匹配结果，包含索引、分数、权重等
    """
    # 构建金字塔
    proto_pyr = build_feature_pyramid(proto_feat, scales=scales)
    target_pyr = build_feature_pyramid(target_feats, scales=scales)
    
    # 多尺度相似度
    sim_maps = multiscale_similarity(proto_pyr, target_pyr, normalize=True)
    
    # 融合尺度
    fused_sim, scale_weights = fuse_scales_with_kernel(
        sim_maps, mode="variance", target_size=target_feats.shape[-2:]
    )
    
    # 提取 top-k 位置
    h, w = fused_sim.shape
    sim_flat = fused_sim.view(-1)
    top_scores, top_indices = torch.topk(sim_flat, k=min(top_k, sim_flat.numel()))
    
    return SEMMatchResult(
        matched_indices=top_indices,
        similarity_scores=top_scores,
        scale_weights=scale_weights,
        cost_matrix=None,
    )


def match_all(
    proto_features: List[torch.Tensor],
    target_features: torch.Tensor,
    scales: List[float] = [1.0, 0.5, 0.25],
    alpha: float = 0.5,
    method: str = "greedy",
) -> List[SEMMatchResult]:
    """批量匹配所有原型。
    
    Parameters
    ----------
    proto_features : List[torch.Tensor]
        原型特征列表，每个 (C, H, W)
    target_features : torch.Tensor
        目标特征图，形状 (C, H, W)
    scales : List[float]
        尺度列表
    alpha : float
        特征-空间权重
    method : str
        匹配方法：'greedy' 或 'hungarian'
        
    Returns
    -------
    List[SEMMatchResult]
        每个原型的匹配结果
    """
    results = []
    
    if method == "greedy":
        # 贪心：逐个匹配
        for proto in proto_features:
            result = match_one_prototype(proto, target_features, scales=scales, alpha=alpha)
            results.append(result)
    
    elif method == "hungarian":
        # 匈牙利：全局最优分配
        # 简化实现：先计算所有相似度，再全局分配
        all_sim_maps = []
        for proto in proto_features:
            proto_pyr = build_feature_pyramid(proto, scales=scales)
            target_pyr = build_feature_pyramid(target_features, scales=scales)
            sim_maps = multiscale_similarity(proto_pyr, target_pyr)
            fused_sim, _ = fuse_scales_with_kernel(sim_maps, mode="uniform")
            all_sim_maps.append(fused_sim)
        
        # 构建代价矩阵 (N_proto, H*W)
        stacked = torch.stack([s.view(-1) for s in all_sim_maps], dim=0)
        cost = -stacked  # 转为代价
        
        # 每个原型找最佳位置
        for i, proto in enumerate(proto_features):
            best_score, best_idx = stacked[i].max(dim=0)
            results.append(
                SEMMatchResult(
                    matched_indices=best_idx.unsqueeze(0),
                    similarity_scores=best_score.unsqueeze(0),
                    scale_weights=None,
                    cost_matrix=cost,
                )
            )
    
    else:
        raise ValueError(f"Unknown matching method: {method}")
    
    return results


def build_layer_pyramid(
    layer_features: List[torch.Tensor],
    scales: List[float] = [1.0, 0.5],
) -> List[List[torch.Tensor]]:
    """构建层×尺度双重金字塔。
    
    Parameters
    ----------
    layer_features : List[torch.Tensor]
        各层特征列表，每个 (B, N, C) 或 (B, C, H, W)
    scales : List[float]
        尺度列表
        
    Returns
    -------
    List[List[torch.Tensor]]
        双重金字塔 [layers][scales]
    """
    layer_pyramid = []
    
    for layer_feat in layer_features:
        # 如果是 token 格式，需要重塑为 2D
        if layer_feat.dim() == 3:  # (B, N, C)
            b, n, c = layer_feat.shape
            h = w = int(n ** 0.5)
            if h * w != n:
                # 非完全平方，尝试推断
                h = w = int(n ** 0.5)
            layer_feat = layer_feat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        scale_pyr = build_feature_pyramid(layer_feat, scales=scales)
        layer_pyramid.append(scale_pyr)
    
    return layer_pyramid


def similarity_per_scale_layer(
    proto_layer_pyr: List[List[torch.Tensor]],
    target_layer_pyr: List[List[torch.Tensor]],
    normalize: bool = True,
) -> List[List[torch.Tensor]]:
    """计算层×尺度相似度矩阵。
    
    Returns
    -------
    List[List[torch.Tensor]]
        相似度矩阵 [layers][scales]，每个元素形状 (H, W)
    """
    n_layers = len(proto_layer_pyr)
    n_scales = len(proto_layer_pyr[0])
    
    sim_matrix = []
    for layer_idx in range(n_layers):
        layer_sims = multiscale_similarity(
            proto_layer_pyr[layer_idx],
            target_layer_pyr[layer_idx],
            normalize=normalize,
        )
        sim_matrix.append(layer_sims)
    
    return sim_matrix


def fuse_scales_alpha(
    sim_matrix: List[List[torch.Tensor]],
    alpha_weights: Optional[torch.Tensor] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """在每一层内融合多尺度（α融合）。
    
    Returns
    -------
    Tuple[List[torch.Tensor], torch.Tensor]
        (每层融合后的相似度, 尺度权重)
    """
    n_layers = len(sim_matrix)
    
    layer_fused = []
    all_weights = []
    
    for layer_sims in sim_matrix:
        fused, weights = fuse_scales_with_kernel(
            layer_sims,
            weights=alpha_weights,
            target_size=target_size,
            mode="variance",
        )
        layer_fused.append(fused)
        all_weights.append(weights)
    
    # 取平均权重
    avg_weights = torch.stack(all_weights).mean(dim=0)
    
    return layer_fused, avg_weights


def fuse_layers_beta(
    layer_sims: List[torch.Tensor],
    beta_weights: Optional[torch.Tensor] = None,
    mode: str = "learned",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """跨层融合（β融合）。
    
    Parameters
    ----------
    layer_sims : List[torch.Tensor]
        各层的相似度图，每个 (H, W)
    beta_weights : Optional[torch.Tensor]
        层权重
    mode : str
        'learned', 'uniform', 'variance'
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (融合后相似度, 层权重)
    """
    n_layers = len(layer_sims)
    
    if mode == "uniform" or beta_weights is None:
        beta_weights = torch.ones(n_layers, device=layer_sims[0].device) / n_layers
    elif mode == "variance":
        variances = torch.tensor([s.var().item() for s in layer_sims], device=layer_sims[0].device)
        beta_weights = F.softmax(variances, dim=0)
    elif mode == "learned":
        beta_weights = F.softmax(beta_weights, dim=0)
    
    stacked = torch.stack(layer_sims, dim=0)  # (n_layers, H, W)
    fused = (stacked * beta_weights.view(-1, 1, 1)).sum(dim=0)
    
    return fused, beta_weights


def match_all_layers_scales(
    proto_layer_features: List[torch.Tensor],
    target_layer_features: List[torch.Tensor],
    scales: List[float] = [1.0, 0.5],
    alpha_weights: Optional[torch.Tensor] = None,
    beta_weights: Optional[torch.Tensor] = None,
    attn_prior: Optional[torch.Tensor] = None,
    gamma: float = 0.5,
) -> Tuple[torch.Tensor, Dict]:
    """层×尺度联合匹配。
    
    Parameters
    ----------
    proto_layer_features : List[torch.Tensor]
        原型的多层特征
    target_layer_features : List[torch.Tensor]
        目标的多层特征
    scales : List[float]
        尺度列表
    alpha_weights : Optional[torch.Tensor]
        尺度权重
    beta_weights : Optional[torch.Tensor]
        层权重
    attn_prior : Optional[torch.Tensor]
        注意力先验图，形状 (H, W)
    gamma : float
        注意力先验混合系数
        
    Returns
    -------
    Tuple[torch.Tensor, Dict]
        (最终相似度图, 调试信息字典)
    """
    # 构建双重金字塔
    proto_pyr = build_layer_pyramid(proto_layer_features, scales=scales)
    target_pyr = build_layer_pyramid(target_layer_features, scales=scales)
    
    # 计算层×尺度相似度
    sim_matrix = similarity_per_scale_layer(proto_pyr, target_pyr, normalize=True)
    
    # α融合（尺度）
    layer_fused, alpha_w = fuse_scales_alpha(sim_matrix, alpha_weights=alpha_weights)
    
    # β融合（层）
    final_sim, beta_w = fuse_layers_beta(layer_fused, beta_weights=beta_weights, mode="variance")
    
    # 应用注意力先验
    if attn_prior is not None and gamma > 0:
        if attn_prior.shape != final_sim.shape:
            attn_resized = F.interpolate(
                attn_prior.unsqueeze(0).unsqueeze(0),
                size=final_sim.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        else:
            attn_resized = attn_prior
        
        final_sim = (1 - gamma) * final_sim + gamma * attn_resized
    
    debug_info = {
        "alpha_weights": alpha_w,
        "beta_weights": beta_w,
        "layer_sims": layer_fused,
        "sim_matrix": sim_matrix,
        "attn_contribution": gamma if attn_prior is not None else 0.0,
    }
    
    return final_sim, debug_info


# ═══════════════════════════════════════════════════════════════
# Rotation-Equivariant SEM (R-SEM) Extensions
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def build_rotation_views(
    feat: torch.Tensor,
    angles: Sequence[float],
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = False,
) -> List[torch.Tensor]:
    """构建旋转视图集合（中心旋转）。
    
    Parameters
    ----------
    feat : torch.Tensor
        输入特征图，形状 (C, H, W) 或 (1, C, H, W)
    angles : Sequence[float]
        旋转角度列表（度数），正值为逆时针
    mode : str
        插值模式，'bilinear' 或 'nearest'
    padding_mode : str
        边界填充模式，'border', 'reflection', 'zeros'
    align_corners : bool
        是否对齐角点
        
    Returns
    -------
    List[torch.Tensor]
        旋转后的特征视图列表，每个形状 (C, H, W)
    """
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)  # (1, C, H, W)
    
    b, c, h, w = feat.shape
    device = feat.device
    dtype = feat.dtype
    
    rotated_views = []
    
    for angle_deg in angles:
        if angle_deg == 0.0:
            rotated_views.append(feat.squeeze(0))
            continue
        
        # 优先使用 kornia
        if _KORNIA_AVAILABLE:
            angle_tensor = torch.tensor([angle_deg], device=device, dtype=dtype)
            rotated = K_transform.rotate(
                feat, angle_tensor, mode=mode, padding_mode=padding_mode, align_corners=align_corners
            )
            rotated_views.append(rotated.squeeze(0))
        else:
            # 回退：使用 affine_grid + grid_sample
            angle_rad = torch.tensor(angle_deg * 3.14159265 / 180.0, device=device, dtype=dtype)
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            
            # 旋转矩阵 (2x3)
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], device=device, dtype=dtype).unsqueeze(0)  # (1, 2, 3)
            
            grid = F.affine_grid(theta, feat.shape, align_corners=align_corners)
            rotated = F.grid_sample(feat, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
            rotated_views.append(rotated.squeeze(0))
    
    return rotated_views


@torch.no_grad()
def build_scale_rotation_pyramid(
    feat: torch.Tensor,
    scales: Sequence[float],
    angles: Sequence[float],
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = False,
) -> List[List[torch.Tensor]]:
    """构建尺度×旋转双重金字塔。
    
    Parameters
    ----------
    feat : torch.Tensor
        输入特征图，形状 (C, H, W)
    scales : Sequence[float]
        尺度因子列表
    angles : Sequence[float]
        旋转角度列表（度数）
    mode : str
        插值模式
    padding_mode : str
        边界填充模式
    align_corners : bool
        是否对齐角点
        
    Returns
    -------
    List[List[torch.Tensor]]
        双重金字塔 F_pyr_rot[s][θ]，每个元素形状 (C, H_s, W_s)
    """
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)  # (1, C, H, W)
    
    pyr_rot = []
    
    # 先按尺度构建金字塔
    scale_pyr = build_feature_pyramid(feat, scales=list(scales), mode=mode)
    
    # 对每个尺度构建旋转视图
    for scaled_feat in scale_pyr:
        rot_views = build_rotation_views(
            scaled_feat, angles, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
        pyr_rot.append(rot_views)
    
    return pyr_rot


@torch.no_grad()
def multiscale_multirot_similarity(
    F_pyr_rot: List[List[torch.Tensor]],
    proto: torch.Tensor,
    reduce: Literal["dot", "mean"] = "dot",
) -> List[List[torch.Tensor]]:
    """计算尺度×旋转相似度矩阵。
    
    Parameters
    ----------
    F_pyr_rot : List[List[torch.Tensor]]
        特征金字塔 [scales][angles]，每个 (C, H, W)
    proto : torch.Tensor
        原型向量，形状 (C,) 或 (1, C) 或 (C, 1, 1)
    reduce : Literal["dot", "mean"]
        相似度计算方式
        
    Returns
    -------
    List[List[torch.Tensor]]
        相似度矩阵 S[s][θ]，每个形状 (H_s, W_s)
    """
    # 归一化原型
    if proto.dim() == 1:
        proto = proto.view(-1, 1, 1)  # (C, 1, 1)
    elif proto.dim() == 2:
        proto = proto.view(-1, 1, 1)
    
    proto = F.normalize(proto, p=2, dim=0, eps=1e-6)
    
    sim_matrix = []
    
    for scale_views in F_pyr_rot:
        angle_sims = []
        for feat_view in scale_views:
            # 归一化特征
            feat_norm = F.normalize(feat_view, p=2, dim=0, eps=1e-6)
            
            # 通道点积 -> 空间相似度图
            if reduce == "dot":
                sim = torch.einsum("chw,c->hw", feat_norm, proto.squeeze())
            elif reduce == "mean":
                sim = (feat_norm * proto).mean(dim=0)
            else:
                raise ValueError(f"Unknown reduce: {reduce}")
            
            angle_sims.append(sim)
        sim_matrix.append(angle_sims)
    
    return sim_matrix


@torch.no_grad()
def fuse_scales_angles_with_kernels(
    sims: List[List[torch.Tensor]],
    gamma: float = 0.2,
    eta: float = 0.35,
    base_size: Optional[Tuple[int, int]] = None,
    stats: Literal["mean", "topk"] = "mean",
    topk: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用可分核融合尺度×角度相似度。
    
    数学公式：
        α^{(s)} = softmax_s( -(μ^{(s)} - μ*)^2 / (2γ^2) )
        β^{(θ)} = softmax_θ( -(ν^{(θ)} - ν*)^2 / (2η^2) )
        ŵ^{(s,θ)} = α^{(s)}β^{(θ)} / Σ
        R = Σ_{s,θ} ŵ^{(s,θ)} · Up(S^{(s,θ)})
    
    Parameters
    ----------
    sims : List[List[torch.Tensor]]
        相似度矩阵 [scales][angles]
    gamma : float
        尺度核带宽
    eta : float
        角度核带宽
    base_size : Optional[Tuple[int, int]]
        目标分辨率，默认为最大尺寸
    stats : Literal["mean", "topk"]
        统计量类型
    topk : int
        top-k 平均时的 k 值
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (融合相似度图 R, 尺度权重 α, 角度权重 β)
    """
    num_scales = len(sims)
    num_angles = len(sims[0]) if sims else 0
    
    if num_scales == 0 or num_angles == 0:
        raise ValueError("Empty similarity matrix")
    
    device = sims[0][0].device
    dtype = sims[0][0].dtype
    
    # 确定目标尺寸
    if base_size is None:
        base_size = (
            max(s[0].shape[0] for s in sims),
            max(s[0].shape[1] for s in sims),
        )
    
    # 计算统计量 μ^{(s)} 和 ν^{(θ)}
    mu_s = torch.zeros(num_scales, device=device, dtype=torch.float32)
    nu_theta = torch.zeros(num_angles, device=device, dtype=torch.float32)
    
    for s_idx in range(num_scales):
        if stats == "mean":
            mu_s[s_idx] = torch.stack([sims[s_idx][a].mean() for a in range(num_angles)]).mean()
        elif stats == "topk":
            all_vals = torch.cat([sims[s_idx][a].flatten() for a in range(num_angles)])
            k = min(topk * num_angles, all_vals.numel())
            mu_s[s_idx] = torch.topk(all_vals, k=k).values.mean()
    
    for a_idx in range(num_angles):
        if stats == "mean":
            nu_theta[a_idx] = torch.stack([sims[s][a_idx].mean() for s in range(num_scales)]).mean()
        elif stats == "topk":
            all_vals = torch.cat([sims[s][a_idx].flatten() for s in range(num_scales)])
            k = min(topk * num_scales, all_vals.numel())
            nu_theta[a_idx] = torch.topk(all_vals, k=k).values.mean()
    
    # 计算峰值
    mu_star = mu_s.max()
    nu_star = nu_theta.max()
    
    # 计算核权重
    alpha = torch.exp(-((mu_s - mu_star) ** 2) / (2 * gamma ** 2))
    alpha = alpha / (alpha.sum() + 1e-8)
    
    beta = torch.exp(-((nu_theta - nu_star) ** 2) / (2 * eta ** 2))
    beta = beta / (beta.sum() + 1e-8)
    
    # 可分权重 ŵ^{(s,θ)} = α^{(s)}β^{(θ)}
    w_sep = alpha.unsqueeze(1) * beta.unsqueeze(0)  # (S, Θ)
    w_sep = w_sep / (w_sep.sum() + 1e-8)
    
    # 融合
    R = torch.zeros(base_size, device=device, dtype=dtype)
    
    for s_idx in range(num_scales):
        for a_idx in range(num_angles):
            sim_sa = sims[s_idx][a_idx]
            
            # 上采样到 base_size
            if sim_sa.shape != base_size:
                sim_up = F.interpolate(
                    sim_sa.unsqueeze(0).unsqueeze(0),
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            else:
                sim_up = sim_sa
            
            R += w_sep[s_idx, a_idx] * sim_up
    
    return R, alpha, beta


@torch.no_grad()
def build_cost_matrix_rot(
    R: torch.Tensor,
    proposals: Sequence[torch.Tensor],
    agg: Literal["mean", "median", "max"] = "mean",
    dilate: int = 0,
) -> torch.Tensor:
    """基于融合相似度图构建代价向量。
    
    Parameters
    ----------
    R : torch.Tensor
        融合相似度图，形状 (H, W)
    proposals : Sequence[torch.Tensor]
        候选掩码列表，每个 (H, W) bool
    agg : Literal["mean", "median", "max"]
        聚合方式
    dilate : int
        膨胀半径（用于鲁棒性）
        
    Returns
    -------
    torch.Tensor
        代价向量，形状 (P,)，越小越优（注意取负号）
    """
    if len(proposals) == 0:
        return torch.zeros(0, device=R.device, dtype=R.dtype)
    
    costs = []
    
    for mask in proposals:
        # 确保尺寸匹配
        if mask.shape != R.shape:
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=R.shape,
                mode="nearest",
            ).squeeze().bool()
        else:
            mask_resized = mask
        
        # 可选膨胀
        if dilate > 0:
            kernel_size = 2 * dilate + 1
            mask_dilated = F.max_pool2d(
                mask_resized.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=kernel_size,
                stride=1,
                padding=dilate,
            ).squeeze().bool()
        else:
            mask_dilated = mask_resized
        
        # 聚合
        vals = R[mask_dilated]
        if vals.numel() == 0:
            score = 0.0
        elif agg == "mean":
            score = vals.mean().item()
        elif agg == "median":
            score = vals.median().item()
        elif agg == "max":
            score = vals.max().item()
        else:
            raise ValueError(f"Unknown agg: {agg}")
        
        # 代价 = -分数（分数越大，代价越小）
        costs.append(-score)
    
    return torch.tensor(costs, device=R.device, dtype=R.dtype)


@torch.no_grad()
def match_one_prototype_rot(
    F: torch.Tensor,
    proto: torch.Tensor,
    proposals: Sequence[torch.Tensor],
    scales: Sequence[float],
    angles: Sequence[float],
    gamma: float = 0.2,
    eta: float = 0.35,
    agg: str = "mean",
    dilate: int = 0,
) -> Dict[str, any]:
    """单个原型的旋转等变匹配。
    
    Parameters
    ----------
    F : torch.Tensor
        目标特征图，形状 (C, H, W)
    proto : torch.Tensor
        原型向量，形状 (C,)
    proposals : Sequence[torch.Tensor]
        候选掩码列表
    scales : Sequence[float]
        尺度列表
    angles : Sequence[float]
        旋转角度列表
    gamma : float
        尺度核带宽
    eta : float
        角度核带宽
    agg : str
        聚合方式
    dilate : int
        膨胀半径
        
    Returns
    -------
    Dict[str, any]
        {"R": Tensor(H,W), "alpha": Tensor(|S|), "beta": Tensor(|Θ|), 
         "cost": Tensor(P,), "matches": [(prop_idx, 0)]}
    """
    # 构建金字塔
    F_pyr_rot = build_scale_rotation_pyramid(F, scales, angles)
    
    # 计算相似度
    sims = multiscale_multirot_similarity(F_pyr_rot, proto, reduce="dot")
    
    # 核融合
    R, alpha, beta = fuse_scales_angles_with_kernels(
        sims, gamma=gamma, eta=eta, base_size=F.shape[-2:]
    )
    
    # 构建代价
    cost = build_cost_matrix_rot(R, proposals, agg=agg, dilate=dilate)
    
    # 贪心匹配：选择代价最小的
    if cost.numel() > 0:
        best_idx = cost.argmin().item()
        matches = [(best_idx, 0)]
    else:
        matches = []
    
    return {
        "R": R,
        "alpha": alpha,
        "beta": beta,
        "cost": cost,
        "matches": matches,
    }


@torch.no_grad()
def match_all_rot(
    F: torch.Tensor,
    prototypes: Sequence[torch.Tensor],
    proposals: Sequence[torch.Tensor],
    scales: Sequence[float],
    angles: Sequence[float],
    gamma: float = 0.2,
    eta: float = 0.35,
    agg: str = "mean",
    dilate: int = 0,
    assignment: Literal["hungarian", "greedy"] = "hungarian",
    cache_pyr: bool = True,
    score_dtype: Literal["float16", "float32"] = "float32",
) -> Dict[str, any]:
    """所有原型的旋转等变匹配（批量）。
    
    Parameters
    ----------
    F : torch.Tensor
        目标特征图，形状 (C, H, W)
    prototypes : Sequence[torch.Tensor]
        原型向量列表，每个 (C,)
    proposals : Sequence[torch.Tensor]
        候选掩码列表
    scales : Sequence[float]
        尺度列表
    angles : Sequence[float]
        旋转角度列表
    gamma : float
        尺度核带宽
    eta : float
        角度核带宽
    agg : str
        聚合方式
    dilate : int
        膨胀半径
    assignment : Literal["hungarian", "greedy"]
        分配算法
    cache_pyr : bool
        是否缓存金字塔（加速）
    score_dtype : Literal["float16", "float32"]
        计算精度
        
    Returns
    -------
    Dict[str, any]
        {"R_list": List[Tensor], "cost_mat": Tensor(P,T), 
         "matches": List[(prop, tgt)], "alphas": List[Tensor], "betas": List[Tensor]}
    """
    T = len(prototypes)
    P = len(proposals)
    
    if T == 0 or P == 0:
        return {
            "R_list": [],
            "cost_mat": torch.zeros((P, T), device=F.device),
            "matches": [],
            "alphas": [],
            "betas": [],
        }
    
    # 类型优化
    if score_dtype == "float16":
        F = F.to(dtype=torch.float16)
        prototypes = [p.to(dtype=torch.float16) for p in prototypes]
    
    # 缓存金字塔（所有原型共享）
    if cache_pyr:
        F_pyr_rot = build_scale_rotation_pyramid(F, scales, angles)
    
    R_list = []
    alphas = []
    betas = []
    cost_mat = torch.zeros((P, T), device=F.device, dtype=F.dtype)
    
    for t_idx, proto in enumerate(prototypes):
        # 相似度
        if cache_pyr:
            sims = multiscale_multirot_similarity(F_pyr_rot, proto, reduce="dot")
        else:
            F_pyr_rot_t = build_scale_rotation_pyramid(F, scales, angles)
            sims = multiscale_multirot_similarity(F_pyr_rot_t, proto, reduce="dot")
        
        # 核融合
        R, alpha, beta = fuse_scales_angles_with_kernels(
            sims, gamma=gamma, eta=eta, base_size=F.shape[-2:]
        )
        
        # 代价
        cost_vec = build_cost_matrix_rot(R, proposals, agg=agg, dilate=dilate)
        
        R_list.append(R)
        alphas.append(alpha)
        betas.append(beta)
        cost_mat[:, t_idx] = cost_vec
    
    # 全局分配
    if assignment == "hungarian" and _SCIPY_AVAILABLE:
        row_ind, col_ind = hungarian_assign(cost_mat, return_cost=False)
        matches = [(int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist())]
    elif assignment == "greedy" or not _SCIPY_AVAILABLE:
        if not _SCIPY_AVAILABLE:
            warnings.warn("scipy not available, using greedy assignment", RuntimeWarning)
        # 贪心：每个原型选最优 proposal
        matches = []
        used_props = set()
        for t_idx in range(T):
            costs_t = cost_mat[:, t_idx]
            # 排除已用的 proposal
            for p_idx in range(P):
                if p_idx in used_props:
                    costs_t[p_idx] = float("inf")
            if costs_t.min() < float("inf"):
                best_p = costs_t.argmin().item()
                matches.append((best_p, t_idx))
                used_props.add(best_p)
    else:
        raise ValueError(f"Unknown assignment: {assignment}")
    
    # 回落精度
    if score_dtype == "float16":
        R_list = [r.to(dtype=torch.float32) for r in R_list]
        cost_mat = cost_mat.to(dtype=torch.float32)
    
    return {
        "R_list": R_list,
        "cost_mat": cost_mat,
        "matches": matches,
        "alphas": alphas,
        "betas": betas,
    }


# ═══════════════════════════════════════════════════════════════
# 几何分析与高级代价计算
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def mask_to_obb(mask: torch.Tensor) -> Tuple[float, float, float, float]:
    """从掩码提取 OBB 几何属性。
    
    参数
    ----
    mask : torch.Tensor
        二值掩码，形状 (H, W)
        
    返回
    ----
    Tuple[float, float, float, float]
        (area_px, aspect_ratio, angle_deg, perimeter_px)
        - area_px: 前景像素数
        - aspect_ratio: 长宽比 max(w,h)/min(w,h)
        - angle_deg: 旋转角度 [-90, 90)
        - perimeter_px: 周长（像素）
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    
    # 面积
    area = float(mask_np.sum())
    
    if area == 0:
        return 0.0, 1.0, 0.0, 0.0
    
    # 尝试使用 cv2
    try:
        import cv2
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return area, 1.0, 0.0, 0.0
        
        # 使用最大轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 周长
        perimeter = float(cv2.arcLength(contour, True))
        
        # minAreaRect
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        
        if w == 0 or h == 0:
            aspect_ratio = 1.0
        else:
            aspect_ratio = max(w, h) / min(w, h)
        
        # 角度归一化到 [-90, 90)
        angle = angle % 180
        if angle >= 90:
            angle -= 180
        
    except Exception:
        # cv2 不可用，回退到简单几何估计
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0:
            return area, 1.0, 0.0, 0.0
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        angle = 0.0
        perimeter = 2.0 * (w + h)
    
    return area, aspect_ratio, angle, perimeter


@torch.no_grad()
def infer_gate_params_from_sem(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    ref_area: Optional[float],
    scales: Sequence[float],
) -> Dict[str, float]:
    """从 SEM/R-SEM 的融合权重推断几何门控参数。
    
    参数
    ----
    alpha : torch.Tensor
        尺度权重 (num_scales,)
    beta : torch.Tensor
        角度权重 (num_angles,)
    ref_area : Optional[float]
        参考掩码面积（像素）
    scales : Sequence[float]
        尺度列表
        
    返回
    ----
    Dict[str, float]
        {"angle_star": ..., "scale_star": ..., "area_expect": ...}
    """
    # 峰值尺度
    scale_idx = alpha.argmax().item()
    scale_star = scales[scale_idx] if scale_idx < len(scales) else 1.0
    
    # 峰值角度（如果有）
    if beta.numel() > 0:
        angle_idx = beta.argmax().item()
        # 需要从外部传入 angles 列表，这里简化假设
        angle_star = 0.0  # 简化：需要实际 angles 列表
    else:
        angle_star = 0.0
    
    # 期望面积
    if ref_area is not None and ref_area > 0:
        area_expect = ref_area * (scale_star ** 2)
    else:
        area_expect = 0.0
    
    return {
        "angle_star": angle_star,
        "scale_star": scale_star,
        "area_expect": area_expect,
    }


@torch.no_grad()
def build_cost_matrix_topk_cover(
    R: torch.Tensor,
    proposals: Sequence[torch.Tensor],
    topk_frac: float = 0.2,
    topk_min: int = 50,
    cover_std_k: float = 1.0,
    cover_w: float = 0.2,
    shape_w: float = 0.15,
    angle_star: Optional[float] = None,
    angle_tol: float = 25.0,
    geom_gate: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """使用 Top-k 均值 + 覆盖率 + 几何门控计算代价。
    
    参数
    ----
    R : torch.Tensor
        融合相似度图，形状 (H, W)
    proposals : Sequence[torch.Tensor]
        候选掩码列表，每个 (H, W) bool
    topk_frac : float
        Top-k 比例（0.2 = 前 20%）
    topk_min : int
        Top-k 最小像素数
    cover_std_k : float
        覆盖率阈值系数
    cover_w : float
        覆盖率权重
    shape_w : float
        形状先验权重
    angle_star : Optional[float]
        R-SEM 峰值角度
    angle_tol : float
        角度容差（度）
    geom_gate : Optional[Dict[str, float]]
        几何门控参数
        
    返回
    ----
    Tuple[torch.Tensor, Dict[str, Any]]
        (costs, meta)
        - costs: 代价向量 (P,)，越小越好
        - meta: 调试信息
    """
    if len(proposals) == 0:
        return torch.zeros(0, device=R.device), {}
    
    # 全局统计
    R_mean = R.mean().item()
    R_std = R.std().item()
    cover_thr = R_mean + cover_std_k * R_std
    
    costs = []
    areas = []
    ratios = []
    angles = []
    shapes = []
    covers = []
    topk_means = []
    rejected = {"area": 0, "ratio": 0, "angle": 0}
    
    geom_gate = geom_gate or {}
    
    for mask in proposals:
        # 确保尺寸匹配
        if mask.shape != R.shape:
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=R.shape,
                mode="nearest",
            ).squeeze().bool()
        else:
            mask_resized = mask
        
        # 提取几何属性
        area, aspect_ratio, angle, perimeter = mask_to_obb(mask_resized)
        
        areas.append(area)
        ratios.append(aspect_ratio)
        angles.append(angle)
        
        # 几何门控检查
        reject = False
        
        # 面积门控
        area_expect = geom_gate.get("area_expect", 0.0)
        area_mult = geom_gate.get("area_mult", 8.0)
        if area_expect > 0 and (area < area_expect / area_mult or area > area_expect * area_mult):
            rejected["area"] += 1
            reject = True
        
        # 长宽比门控
        ratio_min = geom_gate.get("ratio_min", 0.0)
        ratio_max = geom_gate.get("ratio_max", 100.0)
        if aspect_ratio < ratio_min or aspect_ratio > ratio_max:
            rejected["ratio"] += 1
            reject = True
        
        # 角度门控
        if angle_star is not None:
            angle_diff = abs(angle - angle_star)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > angle_tol:
                rejected["angle"] += 1
                reject = True
        
        if reject:
            costs.append(float('inf'))
            shapes.append(0.0)
            covers.append(0.0)
            topk_means.append(0.0)
            continue
        
        # 提取掩码内的 R 值
        vals = R[mask_resized]
        
        if vals.numel() == 0:
            costs.append(float('inf'))
            shapes.append(0.0)
            covers.append(0.0)
            topk_means.append(0.0)
            continue
        
        # Top-k 均值
        k = max(topk_min, int(vals.numel() * topk_frac))
        k = min(k, vals.numel())
        topk_vals, _ = torch.topk(vals, k=k)
        topk_mean = topk_vals.mean().item()
        topk_means.append(topk_mean)
        
        # 覆盖率
        cover = (vals > cover_thr).float().mean().item()
        covers.append(cover)
        
        # 形状分数
        slenderness = min(aspect_ratio, 1.0 / aspect_ratio) if aspect_ratio > 0 else 0.0
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6) if perimeter > 0 else 0.0
        shape_score = 0.5 * slenderness + 0.5 * (1 - compactness)
        shapes.append(shape_score)
        
        # 组合代价（越小越好，所以取负号）
        # 主代价：-（topk_mean + cover_w * cover）
        # 形状惩罚：+ shape_w * shape_score
        cost = -(0.8 * topk_mean + cover_w * cover) + shape_w * shape_score
        costs.append(cost)
    
    costs_tensor = torch.tensor(costs, device=R.device, dtype=R.dtype)

    meta = {
        "topk_means": topk_means[:3],  # 前3个
        "covers": covers[:3],
        "shapes": shapes[:3],
        "rejected": rejected,
        "R_mean": R_mean,
        "R_std": R_std,
        "cover_thr": cover_thr,
    }
    
    return costs_tensor, meta


__all__ = [
    "SEMMatchResult",
    "build_feature_pyramid",
    "multiscale_similarity", 
    "fuse_scales_with_kernel",
    "build_cost_matrix",
    "hungarian_assign",
    "match_one_prototype",
    "match_all",
    "build_layer_pyramid",
    "similarity_per_scale_layer",
    "fuse_scales_alpha",
    "fuse_layers_beta",
    "match_all_layers_scales",
    # R-SEM extensions
    "build_rotation_views",
    "build_scale_rotation_pyramid",
    "multiscale_multirot_similarity",
    "fuse_scales_angles_with_kernels",
    "build_cost_matrix_rot",
    "match_one_prototype_rot",
    "match_all_rot",
    # 几何分析与高级代价
    "mask_to_obb",
    "infer_gate_params_from_sem",
    "build_cost_matrix_topk_cover",
]

