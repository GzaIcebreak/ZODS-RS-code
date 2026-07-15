"""Semantic matching engine for prototype-target correspondence."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from modules.sem_scale_match import (
    SEMMatchResult,
    build_feature_pyramid,
    match_all,
    multiscale_similarity,
    match_all_layers_scales,
    fuse_scales_alpha,
    fuse_layers_beta,
    # R-SEM
    match_all_rot,
)


class SEMMatcher:
    """Semantic Scale-aware Matcher.
    
    封装多尺度语义匹配逻辑，用于在原型与目标特征之间建立对应关系。
    """
    
    def __init__(self, cfg: Dict):
        """
        Parameters
        ----------
        cfg : dict
            配置字典，包含 scales, alpha, method 等参数
        """
        self.cfg = cfg
        self.scales = cfg.get("scales", [1.0, 0.5, 0.25])
        self.alpha = cfg.get("alpha", 0.5)
        self.method = cfg.get("method", "greedy")
        self.cache_pyr = cfg.get("cache_pyr", False)
        self.score_dtype = cfg.get("score_dtype", "fp32")
        
        # DINOv3 multi-layer config
        self.dino_cfg = cfg.get("dino", {})
        self.use_multilayer = self.dino_cfg.get("use_multilayer", False)
        self.layers = self.dino_cfg.get("layers", [6, 10, -1])
        self.use_attn_prior = self.dino_cfg.get("use_attn_prior", False)
        self.gamma = self.dino_cfg.get("gamma", 0.5)
        self.gamma_auto = self.dino_cfg.get("gamma_auto", False)
        
        # R-SEM (Rotation-Equivariant) config
        self.rot_cfg = cfg.get("rot", {})
        self.use_rotation = self.rot_cfg.get("enable", False)
        self.angles = self.rot_cfg.get("angles", [-30, -15, 0, 15, 30])
        self.eta = self.rot_cfg.get("eta", 0.35)
        self.rot_mode = self.rot_cfg.get("mode", "bilinear")
        self.rot_padding = self.rot_cfg.get("padding_mode", "border")
        self.rot_align = self.rot_cfg.get("align_corners", False)
        self.rot_agg = cfg.get("agg", "mean")
        self.rot_dilate = cfg.get("dilate", 1)
        self.rot_assignment = cfg.get("assignment", "hungarian")
        self.rot_gamma = cfg.get("gamma", 0.2)  # 尺度核带宽（与 dino.gamma 不同）
        
        self._cached_target_pyr = None
    
    def match(
        self,
        proto_features: List[torch.Tensor],
        target_features: torch.Tensor,
        proto_layer_features: Optional[List[List[torch.Tensor]]] = None,
        target_layer_features: Optional[List[torch.Tensor]] = None,
        attn_map: Optional[torch.Tensor] = None,
        proposals: Optional[List[torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> Tuple[List[SEMMatchResult], Dict]:
        """执行多尺度匹配（支持多层、旋转等变）。
        
        Parameters
        ----------
        proto_features : List[torch.Tensor]
            原型特征列表（单层），每个 (C,) 或 (C, H, W)
        target_features : torch.Tensor
            目标特征图（单层），形状 (C, H, W)
        proto_layer_features : Optional[List[List[torch.Tensor]]]
            原型多层特征 [n_proto][n_layers]
        target_layer_features : Optional[List[torch.Tensor]]
            目标多层特征 [n_layers]
        attn_map : Optional[torch.Tensor]
            注意力图
        proposals : Optional[List[torch.Tensor]]
            候选掩码列表（用于 R-SEM），每个 (H, W) bool
        return_aux : bool
            是否返回辅助信息
            
        Returns
        -------
        Tuple[List[SEMMatchResult], Dict]
            (每个原型的匹配结果, 调试信息)
        """
        aux_info = {}
        
        # 数据类型优化
        if self.score_dtype == "fp16":
            target_features = target_features.to(dtype=torch.float16)
            proto_features = [p.to(dtype=torch.float16) for p in proto_features]
        
        # R-SEM mode (旋转等变匹配)
        if self.use_rotation and proposals is not None and len(proposals) > 0:
            results = self._match_rotation(
                proto_features,
                target_features,
                proposals,
                aux_info,
            )
        # Multi-layer mode
        elif self.use_multilayer and proto_layer_features and target_layer_features:
            results = self._match_multilayer(
                proto_layer_features,
                target_layer_features,
                attn_map,
                aux_info,
            )
        else:
            # Single layer mode (backward compatible)
            if self.cache_pyr and self._cached_target_pyr is None:
                self._cached_target_pyr = build_feature_pyramid(
                    target_features, scales=self.scales
                )
            
            results = match_all(
                proto_features,
                target_features,
                scales=self.scales,
                alpha=self.alpha,
                method=self.method,
            )
        
        if self.score_dtype == "fp16":
            for r in results:
                r.similarity_scores = r.similarity_scores.to(dtype=torch.float32)
        
        return results, aux_info
    
    def _match_rotation(
        self,
        proto_features: List[torch.Tensor],
        target_features: torch.Tensor,
        proposals: List[torch.Tensor],
        aux_info: Dict,
    ) -> List[SEMMatchResult]:
        """旋转等变匹配实现（R-SEM）。
        
        Parameters
        ----------
        proto_features : List[torch.Tensor]
            原型特征列表，每个 (C,)
        target_features : torch.Tensor
            目标特征图，形状 (C, H, W)
        proposals : List[torch.Tensor]
            候选掩码列表
        aux_info : Dict
            用于存储调试信息
            
        Returns
        -------
        List[SEMMatchResult]
            匹配结果列表
        """
        # 调用 match_all_rot
        rot_result = match_all_rot(
            F=target_features,
            prototypes=proto_features,
            proposals=proposals,
            scales=self.scales,
            angles=self.angles,
            gamma=self.rot_gamma,
            eta=self.eta,
            agg=self.rot_agg,
            dilate=self.rot_dilate,
            assignment=self.rot_assignment,
            cache_pyr=self.cache_pyr,
            score_dtype="float16" if self.score_dtype == "fp16" else "float32",
        )
        
        # 记录调试信息
        aux_info["R_list"] = rot_result["R_list"]
        aux_info["cost_mat"] = rot_result["cost_mat"]
        aux_info["rot_matches"] = rot_result["matches"]
        aux_info["alphas"] = rot_result["alphas"]
        aux_info["betas"] = rot_result["betas"]
        
        # 记录配置信息（用于日志）
        aux_info["rot_cfg"] = {
            "scales": self.scales,
            "angles": self.angles,
            "gamma": self.rot_gamma,
            "eta": self.eta,
            "assignment": self.rot_assignment,
        }
        
        # 打印日志
        print(f"[R-SEM] scales={self.scales}, angles={self.angles}, "
              f"gamma={self.rot_gamma:.3f}, eta={self.eta:.3f}, "
              f"assignment={self.rot_assignment}")
        if rot_result["alphas"]:
            alpha_peak = rot_result["alphas"][0].argmax().item()
            beta_peak = rot_result["betas"][0].argmax().item()
            print(f"[R-SEM] Peak scale_idx={alpha_peak}, Peak angle_idx={beta_peak} "
                  f"({self.angles[beta_peak]}°)")
        
        # 转换为 SEMMatchResult 格式（向后兼容）
        results = []
        matches_dict = {}  # {proto_idx: prop_idx}
        for prop_idx, proto_idx in rot_result["matches"]:
            matches_dict[proto_idx] = prop_idx
        
        for proto_idx in range(len(proto_features)):
            if proto_idx in matches_dict:
                prop_idx = matches_dict[proto_idx]
                # 使用代价的负值作为相似度分数
                score = -rot_result["cost_mat"][prop_idx, proto_idx].item()
                matched_idx = torch.tensor([prop_idx], device=target_features.device)
                sim_score = torch.tensor([score], device=target_features.device)
            else:
                # 未匹配
                matched_idx = torch.tensor([], dtype=torch.long, device=target_features.device)
                sim_score = torch.tensor([], device=target_features.device)
            
            result = SEMMatchResult(
                matched_indices=matched_idx,
                similarity_scores=sim_score,
                scale_weights=rot_result["alphas"][proto_idx] if proto_idx < len(rot_result["alphas"]) else None,
                cost_matrix=rot_result["cost_mat"],
            )
            results.append(result)
        
        return results
    
    def _match_multilayer(
        self,
        proto_layer_features: List[List[torch.Tensor]],
        target_layer_features: List[torch.Tensor],
        attn_map: Optional[torch.Tensor],
        aux_info: Dict,
    ) -> List[SEMMatchResult]:
        """多层匹配实现。"""
        results = []
        
        # γ 自适应
        gamma = self.gamma
        if self.gamma_auto and attn_map is not None:
            # 根据注意力图的方差自动调整 γ
            attn_var = attn_map.var().item()
            gamma = min(0.8, max(0.1, attn_var * 2.0))
            aux_info["gamma_auto"] = gamma
        
        for proto_layers in proto_layer_features:
            # 层×尺度联合匹配
            final_sim, debug_info = match_all_layers_scales(
                proto_layers,
                target_layer_features,
                scales=self.scales,
                attn_prior=attn_map,
                gamma=gamma,
            )
            
            # 提取 top-k 匹配
            k = min(5, final_sim.numel())
            sim_flat = final_sim.view(-1)
            top_scores, top_indices = torch.topk(sim_flat, k=k)
            
            result = SEMMatchResult(
                matched_indices=top_indices,
                similarity_scores=top_scores,
                scale_weights=debug_info.get("alpha_weights"),
                cost_matrix=None,
            )
            results.append(result)
            
            # 收集调试信息
            if not aux_info.get("alpha_weights"):
                aux_info["alpha_weights"] = debug_info.get("alpha_weights")
                aux_info["beta_weights"] = debug_info.get("beta_weights")
                aux_info["attn_contribution"] = debug_info.get("attn_contribution", 0.0)
        
        return results
    
    def clear_cache(self):
        """清空缓存的金字塔。"""
        self._cached_target_pyr = None


# Backward compatibility wrapper
class _SEMMatchResultCompat:
    """Compatibility wrapper for old match API."""
    def __init__(self, results, aux):
        self.results = results
        self.aux = aux
    
    def __iter__(self):
        return iter(self.results)
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        return self.results[idx]


def build_matcher(cfg: Dict) -> SEMMatcher:
    """构建 SEM matcher 实例。
    
    Parameters
    ----------
    cfg : dict
        SEM 配置字典
        
    Returns
    -------
    SEMMatcher
        匹配器实例
    """
    return SEMMatcher(cfg)


__all__ = ["SEMMatcher", "build_matcher"]

