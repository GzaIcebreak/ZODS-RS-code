"""Train-free Uncertainty-Aware Merging (UAM) utilities.

This module implements inference-only helpers for combining multi-source
pixel-level predictions via uncertainty estimation. All APIs are designed to
be:

* **训练-free** – the functions operate purely on inference tensors.
* **设备自适应** – tensors keep their input `device`, falling back to CPU only
  when necessary (e.g. NumPy interop for CRF).
* **类型安全** – public functions include type annotations and exhaustive
  docstrings.

主要提供三个接口：

``pixelwise_distribution``
    计算像素级的类分布与不确定性度量（熵 `E` 与能量 `U`）。

``bayes_merge``
    基于融合分布执行贝叶斯式合并，输出最终掩码与辅助信息。

``crf_refine`` (可选)
    尝试使用全连接条件随机场对概率图进行细化。若 `pydensecrf`
    不存在，则自动降级并返回原始概率。

Example
-------
>>> probs = torch.rand(2, 3, 256, 256)
>>> uam = pixelwise_distribution(probs)
>>> merged = bayes_merge(uam["logits"], uam["entropy"], threshold=0.7)

Notes
-----
* 为保障稳定性，所有接口均假定输入已经过 clamp 到 `[0, 1]`。
* `bayes_merge` 可以同时处理单图和批量输入（通道维为类别数）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

try:
    import pydensecrf.densecrf as dcrf  # type: ignore

    _HAS_DENSECRF = True
except Exception:  # pragma: no cover - optional dependency
    dcrf = None
    _HAS_DENSECRF = False


@dataclass
class PixelwiseStats:
    """Container for uncertainty-aware pixel statistics.

    Attributes
    ----------
    logits: torch.Tensor
        Logits tensor shaped as ``(B, C, H, W)``.
    probs: torch.Tensor
        Softmax probabilities with the same shape as ``logits``.
    entropy: Optional[torch.Tensor]
        Shannon entropy per pixel, shape ``(B, 1, H, W)``.
    energy: Optional[torch.Tensor]
        Negative log partition (energy) per pixel, same shape as ``entropy``.
    """

    logits: torch.Tensor
    probs: torch.Tensor
    entropy: Optional[torch.Tensor]
    energy: Optional[torch.Tensor]


def _safe_log(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def pixelwise_distribution(
    logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
    chunk_size: Optional[int] = None,
    mixed_precision: bool = False,
    memory_sparsify: bool = False,
    auto_tau_cfg: Optional[Dict] = None,
    verbose: bool = False,
    prior: Optional[Dict] = None,
    combine: Optional[str] = None,
    gamma_prior: Optional[float] = None,
) -> PixelwiseStats:
    """Compute pixel-level class distribution and uncertainty metrics.

    Parameters
    ----------
    logits:
        Raw logits or unnormalised scores, shape ``(B, C, H, W)`` or
        ``(C, H, W)``; lower-dimensional inputs are auto-expanded.
    temperature:
        Softmax temperature. Values >1 soften the distribution. Ignored if auto_tau_cfg is provided.
    eps:
        Numerical stabiliser applied to probability clamps.
    chunk_size:
        Process logits in chunks to save memory.
    mixed_precision:
        Use half precision for intermediate calculations.
    memory_sparsify:
        Use sparse representation (only keep top class per pixel).
    auto_tau_cfg:
        Automatic temperature configuration dict with keys: 'enable', 'method', 'target_entropy', etc.
    verbose:
        Print temperature computation logs.

    Returns
    -------
    PixelwiseStats
        包含 logits、概率与不确定性度量的封装对象。
    """

    logit_was_3d = False
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
        logit_was_3d = True
    if logits.dim() != 4:
        raise ValueError("logits must be 3D or 4D tensor (B, C, H, W)")

    # Auto temperature computation if enabled
    if auto_tau_cfg is not None and auto_tau_cfg.get("enable", False):
        from utils.temperature import compute_auto_tau
        
        method = auto_tau_cfg.get("method", "entropy_target")
        target_entropy = auto_tau_cfg.get("target_entropy", 0.5) 
        percentile = auto_tau_cfg.get("percentile", 90.0)
        tau_range = auto_tau_cfg.get("tau_range", [0.1, 3.0])
        
        auto_temp, stats = compute_auto_tau(
            logits, 
            method=method,
            target_entropy=target_entropy, 
            percentile=percentile,
            tau_range=tau_range,
            eps=eps
        )
        temperature = auto_temp
        
        if verbose:
            print(f"🌡️ Auto temperature: τ={temperature:.3f} (method={method})")
            print(f"   Distribution stats: entropy={stats['entropy_mean']:.3f}±{stats['entropy_std']:.3f}, "
                  f"conf={stats['confidence_mean']:.3f}±{stats['confidence_std']:.3f}")
    else:
        temperature = max(temperature, eps)

    logits_orig_dtype = logits.dtype
    if mixed_precision:
        logits = logits.to(dtype=torch.float16)

    # 置信先验注入（B1: logit 模式）
    if prior is not None:
        combine_mode = combine or prior.get("combine", "none")
        gamma = gamma_prior if gamma_prior is not None else prior.get("gamma", 1.0)
        A = prior.get("A")  # (H, W)
        
        if combine_mode == "logit" and A is not None:
            # 注入先验到 logits: Z = R/τ + γ*log(A+eps)
            # A 需要扩展到 (B, 1, H, W)
            A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            if A_expanded.shape[-2:] != logits.shape[-2:]:
                A_expanded = F.interpolate(A_expanded, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            
            # 对所有类别添加先验偏置
            prior_bias = gamma * torch.log(A_expanded + eps)
            logits = logits + prior_bias
    
    if chunk_size is None or chunk_size <= 0 or logits.shape[1] <= chunk_size:
        scaled_logits = logits / temperature
        if memory_sparsify:
            probs = torch.zeros_like(scaled_logits)
            top_idx = torch.argmax(scaled_logits, dim=1, keepdim=True)
            probs.scatter_(1, top_idx, 1.0)
        else:
            probs = F.softmax(scaled_logits, dim=1)
        probs = torch.clamp(probs, min=eps, max=1.0)

        log_probs = _safe_log(probs, eps=eps)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        energy = -torch.logsumexp(scaled_logits, dim=1, keepdim=True)
    else:
        scaled_logits = logits / temperature
        n_chunks = max(1, (scaled_logits.shape[1] + chunk_size - 1) // chunk_size)
        chunks = torch.chunk(scaled_logits, n_chunks, dim=1)
        max_logits = None
        for chunk in chunks:
            chunk_max = torch.amax(chunk, dim=1, keepdim=True)
            max_logits = chunk_max if max_logits is None else torch.maximum(max_logits, chunk_max)

        denom = torch.zeros_like(max_logits)
        for chunk in chunks:
            denom += torch.exp(chunk - max_logits).sum(dim=1, keepdim=True)

        probs_list = []
        entropy = torch.zeros_like(max_logits)
        for chunk in chunks:
            exp_chunk = torch.exp(chunk - max_logits)
            prob_chunk = exp_chunk / torch.clamp(denom, min=eps)
            if memory_sparsify:
                top_idx = torch.argmax(prob_chunk, dim=1, keepdim=True)
                sparse_chunk = torch.zeros_like(prob_chunk)
                sparse_chunk.scatter_(1, top_idx, 1.0)
                prob_chunk = sparse_chunk
            prob_chunk = torch.clamp(prob_chunk, min=eps, max=1.0)
            probs_list.append(prob_chunk)
            entropy -= (prob_chunk * _safe_log(prob_chunk, eps=eps)).sum(dim=1, keepdim=True)

        probs = torch.cat(probs_list, dim=1)
        energy = -(max_logits + torch.log(torch.clamp(denom, min=eps)))

    if mixed_precision:
        probs = probs.to(dtype=logits_orig_dtype)
        entropy = entropy.to(dtype=logits_orig_dtype)
        energy = energy.to(dtype=logits_orig_dtype)

    logits_out = logits.to(dtype=logits_orig_dtype)
    if logit_was_3d:
        logits_out = logits_out.squeeze(0)
        probs = probs.squeeze(0)
        entropy = entropy.squeeze(0)
        energy = energy.squeeze(0)

    return PixelwiseStats(
        logits=logits_out,
        probs=probs,
        entropy=entropy,
        energy=energy,
    )


def bayes_merge(
    logits: torch.Tensor,
    uncertainty: torch.Tensor,
    threshold: float = 0.5,
    calibrate: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    prior_map: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Merge logits into binary masks using uncertainty-aware filtering.

    Parameters
    ----------
    logits:
        Logits tensor ``(B, C, H, W)``.
    uncertainty:
        形状兼容的像素不确定性（例如熵）。较高值表示更不确定。
    threshold:
        Posterior 概率阈值。仅当某像素最大概率高于该阈值时保留。
    calibrate:
        若为 True，将使用 sigmoid 温标校准落入 `[0, 1]` 范围的置信度。
    class_weights:
        可选的类别权重，形状 ``(C,)`` 或 ``(B, C, 1, 1)``，用于重加权 logits。
    prior_map:
        先验激活图，形状 ``(B, 1, H, W)`` 或 ``(B, H, W)``，会在 softmax 前加到
        logits 上，用于对中部或边缘区域施加先验权重。

    Returns
    -------
    dict
        ``{"mask": 二值掩码, "prob": 最大概率, "confidence": 校准置信度, "label": argmax 类别, "uncertainty": 输入不确定性}``。
    """

    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if uncertainty.dim() == 3:
        uncertainty = uncertainty.unsqueeze(0)

    if logits.shape[0] != uncertainty.shape[0]:
        raise ValueError("Batch size mismatch between logits and uncertainty")

    if class_weights is not None:
        if class_weights.dim() == 1:
            class_weights = class_weights.view(1, -1, 1, 1)
        logits = logits * class_weights

    if prior_map is not None:
        if prior_map.dim() == 2:
            prior_map = prior_map.unsqueeze(0)
        if prior_map.dim() == 3:
            prior_map = prior_map.unsqueeze(1)
        if prior_map.shape[-2:] != logits.shape[-2:]:
            raise ValueError("prior_map spatial size must match logits")
        if prior_map.shape[0] not in (1, logits.shape[0]):
            raise ValueError("prior_map batch must match logits or be broadcastable")
        logits = logits + prior_map.to(logits.dtype)

    probs = F.softmax(logits, dim=1)
    max_prob, labels = probs.max(dim=1, keepdim=True)

    if class_weights is not None:
        if class_weights.dim() == 1:
            class_weights = class_weights.view(1, -1, 1, 1)
        max_prob = max_prob * class_weights.gather(1, labels)

    if calibrate:
        # 简单的温标校准：sigmoid with learn-free slope.
        calibrated = torch.sigmoid((max_prob - threshold) * 10.0)
        confidence = calibrated
    else:
        confidence = max_prob

    mask = (max_prob >= threshold).to(dtype=torch.bool)

    return {
        "mask": mask,
        "prob": max_prob,
        "confidence": confidence,
        "label": labels,
        "uncertainty": uncertainty,
    }


def crf_refine(
    image: torch.Tensor,
    prob_map: torch.Tensor,
    n_iters: int = 5,
    sxy: Tuple[int, int] = (3, 3),
    srgb: Tuple[int, int, int] = (5, 5, 5),
) -> torch.Tensor:
    """Apply DenseCRF refinement if available.

    Parameters
    ----------
    image:
        Input image tensor ``(3, H, W)`` or ``(B, 3, H, W)``.
    prob_map:
        类别概率，形状 ``(C, H, W)`` 或 ``(B, C, H, W)``。
    n_iters:
        DenseCRF 迭代次数。
    sxy / srgb:
        双边滤波的空间/颜色参数。

    Returns
    -------
    torch.Tensor
        Refined probability map，若 `pydensecrf` 缺失则返回原始输入。
    """

    if prob_map.dim() == 4:
        # Process batch independently
        refined = []
        for img, p in zip(image, prob_map):
            refined.append(crf_refine(img, p, n_iters=n_iters, sxy=sxy, srgb=srgb))
        return torch.stack(refined, dim=0)

    if not _HAS_DENSECRF or dcrf is None:  # pragma: no cover - optional path
        return prob_map

    if image.dim() != 3 or prob_map.dim() != 3:
        raise ValueError("Expect image and prob_map to have shapes (3,H,W)/(C,H,W)")

    device = prob_map.device
    c, h, w = prob_map.shape

    unary = -_safe_log(prob_map).clamp(max=50.0)
    unary = unary.view(c, -1).cpu().numpy()

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(unary)

    img_np = image.detach().cpu().numpy().astype("float32")
    img_np = img_np.transpose(1, 2, 0)

    d.addPairwiseGaussian(sxy=sxy, compat=3)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=img_np, compat=5)

    q = d.inference(n_iters)
    refined = torch.tensor(q, device=device, dtype=prob_map.dtype)
    refined = refined.view(c, h, w)
    refined = refined / refined.sum(dim=0, keepdim=True).clamp(min=1e-6)
    return refined


# ═══════════════════════════════════════════════════════════════
# B2: 负原型采样（仅用于 UAM softmax 竞争）
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def make_negative_prototypes_from_image(
    feat: torch.Tensor,
    proposals: list,
    prior_A: Optional[torch.Tensor] = None,
    R_list: Optional[list] = None,
    num_neg: int = 3,
    min_area: int = 128,
) -> list:
    """从图像中采样负原型区域（低置信/低响应）。
    
    仅用于 UAM 的 softmax 竞争，不参与最终匹配输出。
    
    参数
    ----
    feat : torch.Tensor
        DINOv3 密特征，形状 (C, H, W)
    proposals : list
        候选掩码列表，每个 (H, W) bool
    prior_A : Optional[torch.Tensor]
        置信先验图 (H, W)
    R_list : Optional[list]
        相似度图列表
    num_neg : int
        负原型数量
    min_area : int
        最小面积阈值（像素）
        
    返回
    ----
    list
        负原型向量列表，每个 (C,)
    """
    if len(proposals) == 0:
        return []
    
    c, h, w = feat.shape
    device = feat.device
    
    # 计算每个候选的"负性分数"（越高越适合作为负样本）
    neg_scores = []
    
    for idx, mask in enumerate(proposals):
        # 确保尺寸匹配
        if mask.shape != (h, w):
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="nearest",
            ).squeeze().bool()
        else:
            mask_resized = mask
        
        # 面积过滤
        area = mask_resized.sum().item()
        if area < min_area:
            neg_scores.append(-1.0)
            continue
        
        # 负性分数 = (1 - 先验均值) + (1 - 响应均值)
        neg_score = 0.0
        
        if prior_A is not None:
            A_vals = prior_A[mask_resized]
            if A_vals.numel() > 0:
                neg_score += (1.0 - A_vals.mean().item())
        
        if R_list is not None and len(R_list) > 0:
            R_vals_all = []
            for R in R_list:
                R_resized = R
                if R.shape != (h, w):
                    R_resized = F.interpolate(
                        R.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                R_vals_all.append(R_resized[mask_resized].mean().item())
            
            if R_vals_all:
                neg_score += (1.0 - max(R_vals_all))
        
        neg_scores.append(neg_score)
    
    # 选择 top-k 负候选
    neg_scores_tensor = torch.tensor(neg_scores, device=device)
    valid_mask = neg_scores_tensor >= 0
    
    if valid_mask.sum() == 0:
        return []
    
    valid_scores = neg_scores_tensor[valid_mask]
    valid_indices = torch.where(valid_mask)[0]
    
    k = min(num_neg, valid_scores.numel())
    if k == 0:
        return []
    
    topk_vals, topk_idx = torch.topk(valid_scores, k=k)
    selected_indices = valid_indices[topk_idx]
    
    # 提取负原型（掩码区域的特征均值）
    neg_protos = []
    for idx in selected_indices:
        mask = proposals[idx.item()]
        if mask.shape != (h, w):
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="nearest",
            ).squeeze().bool()
        else:
            mask_resized = mask
        
        # 提取特征均值
        feat_vals = feat[:, mask_resized]  # (C, N)
        if feat_vals.numel() > 0:
            neg_proto = feat_vals.mean(dim=1)  # (C,)
            neg_proto = F.normalize(neg_proto, p=2, dim=0, eps=1e-6)
            neg_protos.append(neg_proto)
    
    return neg_protos


