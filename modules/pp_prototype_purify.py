"""Prototype Purification (PP) utilities.

该模块实现跨源谱净化（Prototype Purification）流程，提供纯矩阵运算的
原型净化与子原型提取能力，支持可选的 CLIP 图像/文本先验，并在缺失时
自动退化为单源谱净化。所有函数均在 ``torch.no_grad`` 环境下工作，以保
证训练-free 推理。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def percent_clip(
    x: torch.Tensor,
    p_low: float = 0.0,
    p_high: float = 0.999,
) -> Tuple[torch.Tensor, float, float]:
    """对一维张量执行分位裁剪。

    参数
    ----
    x : torch.Tensor
        输入一维张量，若为多维将被展平。
    p_low : float
        下分位阈值，范围 ``[0, 1)``。
    p_high : float
        上分位阈值，范围 ``(0, 1]`` 且需大于 ``p_low``。

    返回
    ----
    Tuple[torch.Tensor, float, float]
        (裁剪后的张量, 下阈值, 上阈值)。阈值以浮点数形式返回，便于日志或
        调试分析。
    """

    if x.numel() == 0:
        raise ValueError("percent_clip 需要非空张量。")

    if p_low < 0.0 or p_high > 1.0 or p_low >= p_high:
        raise ValueError("分位阈值需满足 0 <= p_low < p_high <= 1。")

    if x.ndim != 1:
        x = x.reshape(-1)

    q_low = torch.quantile(x, p_low)
    q_high = torch.quantile(x, p_high)
    clipped = x.clamp(min=q_low.item(), max=q_high.item())

    return clipped, float(q_low.item()), float(q_high.item())


@torch.no_grad()
def compute_cov_eig(
    X: torch.Tensor,
    top_r: int = 32,
    method: str = "eigh",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算协方差谱分解的前 ``r`` 个主方向。

    参数
    ----
    X : torch.Tensor
        形状 ``(n, d)`` 的样本矩阵，应当已经做过均值去除或归一化。
    top_r : int
        需要返回的谱方向数量 ``r``，会自动裁剪到有效 rank 范围。
    method : str
        期望的谱分解方法，支持 ``"eigh"`` 与 ``"svd"``。当 ``n << d`` 时
        会优先使用 SVD 以降低计算开销。

    返回
    ----
    Tuple[torch.Tensor, torch.Tensor]
        ``(U_r, S_r)``，其中 ``U_r`` 形状为 ``(d, r)``，``S_r`` 为特征值
        向量 ``(r,)``。
    """

    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵。")

    n, d = X.shape
    if n < 1 or d < 1:
        raise ValueError("X 的形状无效。")

    r = max(1, min(top_r, d))

    chosen = method.lower()
    if chosen not in {"eigh", "svd"}:
        raise ValueError("method 仅支持 'eigh' 或 'svd'。")

    if chosen == "eigh" and n <= d // 2:
        chosen = "svd"
    elif chosen == "svd" and d <= n:
        chosen = "eigh"

    if chosen == "svd":
        # X = U * diag(s) * Vt, 协方差特征值为 s^2 / (n - 1)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        if Vh is None:
            raise RuntimeError("SVD 计算失败：Vh 为 None。")
        eig_vecs = Vh.transpose(0, 1)
        eig_vals = (S ** 2) / max(1, n - 1)
    else:
        cov = (X.transpose(0, 1) @ X) / max(1, n - 1)
        eig_vals, eig_vecs = torch.linalg.eigh(cov)

    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real

    # 按特征值降序排列
    sorted_vals, indices = torch.sort(eig_vals, descending=True)
    eig_vecs = eig_vecs[:, indices]

    r = min(r, eig_vecs.shape[1])
    U_r = eig_vecs[:, :r]
    S_r = sorted_vals[:r]

    return U_r.contiguous(), S_r.contiguous()


@torch.no_grad()
def cross_source_scores(
    U_r: torch.Tensor,
    clip_img_proto: Optional[torch.Tensor] = None,
    clip_txt_proto: Optional[torch.Tensor] = None,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    """计算谱方向与 CLIP 先验的一致性得分。

    若 CLIP 图像或文本先验不可用，则返回全 ``1`` 向量作为统一权重。

    参数
    ----
    U_r : torch.Tensor
        谱方向矩阵，形状 ``(d, r)``。
    clip_img_proto : Optional[torch.Tensor]
        CLIP 图像原型，形状 ``(d,)`` 或 ``(c,)``，需与 ``U_r`` 的特征维度匹配。
    clip_txt_proto : Optional[torch.Tensor]
        CLIP 文本原型，同上。
    alpha : float
        图像先验的混合系数。
    beta : float
        文本先验的混合系数。

    返回
    ----
    torch.Tensor
        形状 ``(r,)`` 的得分向量。
    """

    if U_r.ndim != 2:
        raise ValueError("U_r 必须是二维矩阵。")

    d, r = U_r.shape
    device = U_r.device
    dtype = U_r.dtype

    img = clip_img_proto
    txt = clip_txt_proto

    if img is not None:
        img = img.to(device=device, dtype=dtype)
        img = F.normalize(img, dim=0)
        if img.shape[0] != d:
            raise ValueError("clip_img_proto 维度与 U_r 不匹配。")

    if txt is not None:
        txt = txt.to(device=device, dtype=dtype)
        txt = F.normalize(txt, dim=0)
        if txt.shape[0] != d:
            raise ValueError("clip_txt_proto 维度与 U_r 不匹配。")

    if img is None and txt is None:
        return torch.ones(r, device=device, dtype=dtype)

    comps = []
    weights = []

    if img is not None and alpha > 0:
        img_scores = (U_r.transpose(0, 1) @ img).abs()
        comps.append(img_scores)
        weights.append(alpha)

    if txt is not None and beta > 0:
        txt_scores = (U_r.transpose(0, 1) @ txt).abs()
        comps.append(txt_scores)
        weights.append(beta)

    if not comps:
        return torch.ones(r, device=device, dtype=dtype)

    total_weight = sum(weights)
    normalized_scores = sum(w * s for w, s in zip(weights, comps)) / max(total_weight, 1e-6)

    return normalized_scores


@torch.no_grad()
def prototype_purify(
    X_k: torch.Tensor,
    top_r: int = 32,
    alpha: float = 0.7,
    beta: float = 0.3,
    use_whitening: bool = False,
    clip_img_proto: Optional[torch.Tensor] = None,
    clip_txt_proto: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """对单类参考特征执行跨源谱净化。

    参数
    ----
    X_k : torch.Tensor
        形状 ``(n_k, d)`` 的 DINOv3 特征集合。
    top_r : int
        选取的谱方向数量。
    alpha, beta : float
        CLIP 图像/文本得分的混合权重。
    use_whitening : bool
        是否在构建投影矩阵时执行白化 ``S_r^{-1/2}``。
    clip_img_proto : Optional[torch.Tensor]
        CLIP 图像原型，形状 ``(d,)``。
    clip_txt_proto : Optional[torch.Tensor]
        CLIP 文本原型，形状 ``(d,)``。

    返回
    ----
    Dict[str, Any]
        包含净化后的主原型、谱子空间、投影矩阵等信息。
    """

    if X_k.ndim != 2:
        raise ValueError("X_k 必须是二维矩阵。")

    n_k, d = X_k.shape
    device = X_k.device
    dtype = X_k.dtype

    if n_k == 0:
        raise ValueError("X_k 至少包含一个样本。")

    X_norm = F.normalize(X_k, p=2, dim=1)
    p_bar = X_norm.mean(dim=0)

    if n_k < 3:
        p_hat = F.normalize(p_bar, dim=0)
        return {
            "p_hat": p_hat,
            "U_r": torch.zeros(d, 0, device=device, dtype=dtype),
            "S_r": torch.zeros(0, device=device, dtype=dtype),
            "P_r": torch.eye(d, device=device, dtype=dtype),
            "scores": torch.ones(0, device=device, dtype=dtype),
            "p_bar": p_bar,
        }

    X_centered = X_norm - p_bar
    U_r, S_r = compute_cov_eig(X_centered, top_r=top_r, method="eigh")

    raw_scores = cross_source_scores(
        U_r,
        clip_img_proto=clip_img_proto,
        clip_txt_proto=clip_txt_proto,
        alpha=alpha,
        beta=beta,
    )
    weights = F.softmax(raw_scores, dim=0)

    if use_whitening and S_r.numel() > 0:
        scales = torch.where(S_r > 1e-8, S_r.rsqrt(), torch.zeros_like(S_r))
        U_proj = U_r * scales.unsqueeze(0)
    else:
        U_proj = U_r

    weighted_basis = U_proj * weights.unsqueeze(0)
    P_r = weighted_basis @ U_proj.transpose(0, 1)

    p_hat_raw = P_r @ p_bar
    p_hat = F.normalize(p_hat_raw, dim=0)

    return {
        "p_hat": p_hat,
        "U_r": U_r,
        "S_r": S_r,
        "P_r": P_r,
        "scores": raw_scores,
        "p_bar": p_bar,
    }


def _torch_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """轻量级 k-means 聚类实现（torch 版）。"""

    n, d = points.shape
    if num_clusters >= n:
        centers = points.clone()
        labels = torch.arange(n, device=points.device)
        inertia = 0.0
        return centers, labels, inertia

    perm = torch.randperm(n, device=points.device)
    centers = points[perm[:num_clusters]].clone()

    for _ in range(num_iters):
        dist = torch.cdist(points, centers, p=2)
        labels = torch.argmin(dist, dim=1)
        new_centers = []
        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                new_centers.append(points[mask].mean(dim=0))
            else:
                # 若某簇为空，随机重采样
                new_centers.append(points[torch.randint(0, n, (1,), device=points.device)[0]])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    dist = torch.cdist(points, centers, p=2)
    labels = torch.argmin(dist, dim=1)
    inertia = torch.sum((points - centers[labels]) ** 2).item()

    return centers, labels, inertia


@torch.no_grad()
def subprototypes_spectral(
    X_k: torch.Tensor,
    U_r: torch.Tensor,
    method: str = "hdbscan",
    min_cluster_size: int = 10,
    max_k: int = 4,
) -> List[torch.Tensor]:
    """在谱子空间中提取子原型。

    参数
    ----
    X_k : torch.Tensor
        类别特征集合，形状 ``(n_k, d)``。
    U_r : torch.Tensor
        谱基矩阵，形状 ``(d, r)``。
    method : str
        聚类方法，支持 ``"hdbscan"``、``"kmeans"``、``"none"``。
    min_cluster_size : int
        HDBSCAN 的最小簇大小。
    max_k : int
        KMeans 的最大簇数量。

    返回
    ----
    List[torch.Tensor]
        子原型列表，每个元素形状 ``(d,)``。
    """

    if X_k.ndim != 2 or U_r.ndim != 2:
        raise ValueError("X_k 与 U_r 均需为二维矩阵。")

    n_k, d = X_k.shape
    if n_k == 0:
        raise ValueError("X_k 至少包含一个样本。")

    if U_r.shape[0] != d:
        raise ValueError("U_r 特征维度与 X_k 不一致。")

    if method.lower() == "none" or n_k < max(3, min_cluster_size):
        proto = F.normalize(X_k.mean(dim=0), dim=0)
        return [proto]

    X_norm = F.normalize(X_k, p=2, dim=1)
    coords = X_norm @ U_r

    labels = None
    method = method.lower()

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore

            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels_np = clusterer.fit_predict(coords.cpu().numpy())
            labels = torch.from_numpy(labels_np).to(coords.device)
            valid_labels = labels[labels >= 0]
            if valid_labels.numel() == 0:
                labels = None
        except Exception:
            labels = None

    if labels is None:
        k_upper = min(max_k, n_k)
        best_inertia = float("inf")
        best_labels: Optional[torch.Tensor] = None
        best_k = 1

        for k in range(1, k_upper + 1):
            centers, _labels, inertia = _torch_kmeans(coords, k)
            if inertia < best_inertia * 0.9:
                best_inertia = inertia
                best_labels = _labels
                best_k = k
            elif k == 1 and best_labels is None:
                best_inertia = inertia
                best_labels = _labels
                best_k = k

        if best_labels is None:
            proto = F.normalize(X_norm.mean(dim=0), dim=0)
            return [proto]

        labels = best_labels

    unique_labels = torch.unique(labels[labels >= 0]) if (labels >= 0).any() else torch.unique(labels)
    if unique_labels.numel() == 0:
        proto = F.normalize(X_norm.mean(dim=0), dim=0)
        return [proto]

    prototypes: List[torch.Tensor] = []
    projector = U_r @ U_r.transpose(0, 1)

    for lab in unique_labels.tolist():
        mask = labels == lab
        if not mask.any():
            continue
        cluster_mean = X_norm[mask].mean(dim=0)
        proj = projector @ cluster_mean
        proto = F.normalize(proj, dim=0)
        prototypes.append(proto)

    if not prototypes:
        proto = F.normalize(X_norm.mean(dim=0), dim=0)
        prototypes.append(proto)

    return prototypes


@torch.no_grad()
def class_prototypes(
    ref_feats_by_class: Dict[int, torch.Tensor],
    cfg: Dict[str, Any],
    clip_hooks: Optional[Dict[str, Callable[[int], Optional[torch.Tensor]]]] = None,
    debug_store: Optional[Dict[int, Dict[str, Any]]] = None,
    store_raw_feats: bool = False,
) -> Dict[int, List[torch.Tensor]]:
    """批量计算类别原型或子原型。

    参数
    ----
    ref_feats_by_class : Dict[int, torch.Tensor]
        ``{cls_id: (n_k, d)}`` 的参考特征字典。
    cfg : Dict[str, Any]
        Prototype Purification 配置字典（通常来自 ``pp`` 节点）。
    clip_hooks : Optional[Dict[str, Callable]]
        可选的 CLIP 先验获取回调，键可为 ``"img"``/``"image"`` 与
        ``"txt"``/``"text"``，函数签名 ``fn(cls_id) -> Optional[Tensor]``。
    debug_store : Optional[Dict[int, Dict[str, Any]]]
        若提供，则会写入每个类别的谱分解信息（``p_hat``、``U_r`` 等）。

    返回
    ----
    Dict[int, List[torch.Tensor]]
        每个类别的主原型或子原型列表。
    """
    pp_cfg = cfg or {}

    results: Dict[int, List[torch.Tensor]] = {}

    top_r = int(pp_cfg.get("top_r", 32))
    alpha = float(pp_cfg.get("alpha", 0.7))
    beta = float(pp_cfg.get("beta", 0.3))
    use_whitening = bool(pp_cfg.get("use_whitening", False))
    use_sub = bool(pp_cfg.get("use_subprototypes", False))
    cluster_cfg = pp_cfg.get("cluster", {}) if isinstance(pp_cfg.get("cluster", {}), dict) else {}
    sub_method = str(cluster_cfg.get("method", "hdbscan"))
    min_cluster_size = int(cluster_cfg.get("min_size", 10))
    max_k = int(cluster_cfg.get("max_k", 4))

    img_hook = None
    txt_hook = None
    if clip_hooks:
        img_hook = clip_hooks.get("img") or clip_hooks.get("image")
        txt_hook = clip_hooks.get("txt") or clip_hooks.get("text")

    for cls_id, feats in ref_feats_by_class.items():
        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"类别 {cls_id} 的特征必须是 torch.Tensor。")

        clip_img = img_hook(cls_id) if callable(img_hook) else None
        clip_txt = txt_hook(cls_id) if callable(txt_hook) else None

        purify_res = prototype_purify(
            feats,
            top_r=top_r,
            alpha=alpha,
            beta=beta,
            use_whitening=use_whitening,
            clip_img_proto=clip_img,
            clip_txt_proto=clip_txt,
        )

        if use_sub and purify_res["U_r"].numel() > 0:
            subs = subprototypes_spectral(
                feats,
                purify_res["U_r"],
                method=sub_method,
                min_cluster_size=min_cluster_size,
                max_k=max_k,
            )
            results[cls_id] = subs
        else:
            results[cls_id] = [purify_res["p_hat"]]

        if debug_store is not None:
            debug_store[cls_id] = {
                "purify": purify_res,
                "prototypes": results[cls_id],
            }
            if store_raw_feats:
                max_samples = min(feats.shape[0], int(cfg.get("debug", {}).get("max_samples", 2048)))
                debug_store[cls_id]["raw_feats"] = feats[:max_samples].detach().cpu()

    return results


# ═══════════════════════════════════════════════════════════════
# Robust-PP: Tyler's M-Estimator + Spectral Purification
# ═══════════════════════════════════════════════════════════════

import warnings


@torch.no_grad()
def tyler_covariance(
    X: torch.Tensor,
    iters: int = 20,
    tol: float = 1e-4,
    init: str = "identity",
    eps: float = 1e-6,
    trace_norm: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """计算 Tyler's M-estimator 协方差矩阵。
    
    Tyler's M-estimator 是一种稳健的协方差估计方法，对重尾分布和离群值不敏感。
    通过迭代固定点算法求解：
        Σ = (d/n) Σᵢ xᵢxᵢᵀ / (xᵢᵀΣ⁻¹xᵢ)
    
    参数
    ----
    X : torch.Tensor
        形状 ``(n, d)`` 的样本矩阵，应已L2归一化或零均值化
    iters : int
        最大迭代次数
    tol : float
        收敛容差 ||Σ_{t+1} - Σ_t||_F / ||Σ_t||_F
    init : str
        初始化方式：'identity' 或 'sample'
    eps : float
        数值稳定项（防止奇异）
    trace_norm : bool
        是否规范化到 trace(Σ) = d
    verbose : bool
        是否打印迭代信息
        
    返回
    ----
    torch.Tensor
        Tyler's 协方差矩阵 Σ，形状 ``(d, d)``，float32
        
    注意
    ----
    若迭代不收敛或数值异常，会回退到样本协方差 + εI 并发出警告
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D matrix (n, d)")
    
    n, d = X.shape
    
    if n < 3:
        warnings.warn(f"Sample size n={n} too small for Tyler's estimator, using sample covariance", RuntimeWarning)
        Sigma = X.T @ X / max(n, 1) + eps * torch.eye(d, device=X.device, dtype=torch.float32)
        if trace_norm:
            Sigma = d * Sigma / (Sigma.trace() + eps)
        return Sigma
    
    # 确保 float32 精度
    X_orig_dtype = X.dtype
    X = X.to(dtype=torch.float32)
    device = X.device
    
    # 初始化
    if init == "identity":
        Sigma = torch.eye(d, device=device, dtype=torch.float32)
    elif init == "sample":
        Sigma = X.T @ X / n + eps * torch.eye(d, device=device, dtype=torch.float32)
        if trace_norm:
            Sigma = d * Sigma / (Sigma.trace() + eps)
    else:
        raise ValueError(f"Unknown init: {init}")
    
    # 迭代求解
    for t in range(iters):
        Sigma_old = Sigma.clone()
        
        try:
            # 计算 mahalanobis 距离：m_i = xᵢᵀ Σ⁻¹ xᵢ
            # 使用 cholesky 分解加速
            L = torch.linalg.cholesky(Sigma + eps * torch.eye(d, device=device, dtype=torch.float32))
            X_inv = torch.linalg.solve_triangular(L, X.T, upper=False)  # L⁻¹X^T
            m = (X_inv ** 2).sum(dim=0)  # (n,)
            
            # 计算权重：wᵢ = d / max(mᵢ, ε)
            w = d / torch.clamp(m, min=eps)
            
            # 更新 Σ = (1/n) Σᵢ wᵢ xᵢ xᵢᵀ
            X_weighted = X.T * w.unsqueeze(0)  # (d, n)
            Sigma = (X_weighted @ X) / n
            
            # Trace 规范化
            if trace_norm:
                Sigma = d * Sigma / (Sigma.trace() + eps)
            
            # 检查收敛
            diff = (Sigma - Sigma_old).norm('fro').item()
            rel_change = diff / (Sigma_old.norm('fro').item() + eps)
            
            if verbose:
                print(f"[Tyler] iter {t+1}/{iters}: rel_change={rel_change:.6f}")
            
            if rel_change < tol:
                if verbose:
                    print(f"[Tyler] Converged at iteration {t+1}")
                break
                
        except Exception as e:
            # 数值异常，回退
            warnings.warn(
                f"Tyler's estimator failed at iteration {t}: {e}. Falling back to sample covariance.",
                RuntimeWarning
            )
            Sigma = X.T @ X / n + eps * torch.eye(d, device=device, dtype=torch.float32)
            if trace_norm:
                Sigma = d * Sigma / (Sigma.trace() + eps)
            break
    
    # 回落到原始精度
    if X_orig_dtype != torch.float32:
        Sigma = Sigma.to(dtype=X_orig_dtype)
    
    return Sigma


@torch.no_grad()
def robust_cov_eig(
    X: torch.Tensor,
    top_r: int = 32,
    **tyler_kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算稳健协方差的谱分解。
    
    使用 Tyler's M-estimator 计算协方差，然后谱分解获取主方向。
    
    参数
    ----
    X : torch.Tensor
        形状 ``(n, d)`` 的样本矩阵
    top_r : int
        保留的主成分数量
    **tyler_kwargs :
        传递给 tyler_covariance 的参数（iters, tol, init等）
        
    返回
    ----
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(U_r, S_r, Sigma)``，其中：
        - U_r: 形状 ``(d, r)``，前 r 个特征向量
        - S_r: 形状 ``(r,)``，对应的特征值
        - Sigma: 形状 ``(d, d)``，完整的 Tyler's 协方差矩阵
    """
    n, d = X.shape
    
    # 计算 Tyler's 协方差
    Sigma = tyler_covariance(X, **tyler_kwargs)
    
    # 谱分解
    try:
        # 使用 eigh 求解对称矩阵
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        
        # 降序排列
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # 取前 r 个
        r = max(1, min(top_r, d))
        U_r = eigvecs[:, :r]
        S_r = eigvals[:r]
        
    except Exception as e:
        warnings.warn(f"Eigendecomposition failed: {e}. Using SVD fallback.", RuntimeWarning)
        # SVD 回退
        try:
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            r = max(1, min(top_r, S.shape[0]))
            U_r = Vt[:r].T  # (d, r)
            S_r = S[:r] ** 2 / (n - 1)
        except Exception as e2:
            warnings.warn(f"SVD also failed: {e2}. Returning identity.", RuntimeWarning)
            r = max(1, min(top_r, d))
            U_r = torch.eye(d, r, device=X.device, dtype=X.dtype)
            S_r = torch.ones(r, device=X.device, dtype=X.dtype)
    
    return U_r, S_r, Sigma


@torch.no_grad()
def prototype_purify_robust(
    X_k: torch.Tensor,
    top_r: int = 32,
    alpha: float = 0.7,
    beta: float = 0.3,
    use_whitening: bool = False,
    clip_img_proto: Optional[torch.Tensor] = None,
    clip_txt_proto: Optional[torch.Tensor] = None,
    tyler: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """使用 Tyler's 估计器进行稳健原型净化。
    
    完整流程：
    1) Tyler's M-estimator 计算稳健协方差 Σ
    2) 谱分解得到稳健子空间 U_r 和特征值 S_r
    3) 跨源一致性打分（CLIP 图像/文本先验）得到权重 w
    4) 稳健谱投影：p̂ = normalize(P_r · p̄)
    
    参数
    ----
    X_k : torch.Tensor
        类 k 的样本特征，形状 ``(n, d)``
    top_r : int
        保留的主成分数量
    alpha : float
        CLIP 图像先验权重
    beta : float
        CLIP 文本先验权重
    use_whitening : bool
        是否使用白化投影
    clip_img_proto : Optional[torch.Tensor]
        CLIP 图像原型，形状 ``(d,)``
    clip_txt_proto : Optional[torch.Tensor]
        CLIP 文本原型，形状 ``(d,)``
    tyler : Optional[Dict[str, Any]]
        Tyler's 估计器参数（iters, tol, init等）
        
    返回
    ----
    Dict[str, Any]
        包含：
        - p_hat: 净化后的原型 ``(d,)``
        - U_r: 稳健子空间 ``(d, r)``
        - S_r: 特征值 ``(r,)``
        - P_r: 投影矩阵 ``(d, d)``
        - scores: 一致性打分 ``(r,)``
        - p_bar: 原始均值原型 ``(d,)``
        - Sigma: Tyler's 协方差 ``(d, d)``
    """
    n, d = X_k.shape
    
    # 计算原始均值原型
    p_bar = X_k.mean(dim=0)
    p_bar = F.normalize(p_bar, p=2, dim=0, eps=1e-6)
    
    # Tyler's 参数
    tyler_params = tyler or {}
    tyler_params.setdefault("iters", 20)
    tyler_params.setdefault("tol", 1e-4)
    tyler_params.setdefault("init", "identity")
    tyler_params.setdefault("eps", 1e-6)
    tyler_params.setdefault("trace_norm", True)
    tyler_params.setdefault("verbose", False)
    
    # 计算稳健协方差与谱分解
    U_r, S_r, Sigma = robust_cov_eig(X_k, top_r=top_r, **tyler_params)
    
    # 跨源一致性打分
    if clip_img_proto is not None or clip_txt_proto is not None:
        scores = torch.zeros(top_r, device=X_k.device, dtype=torch.float32)
        
        for i in range(top_r):
            u_i = U_r[:, i]
            u_i_norm = F.normalize(u_i, p=2, dim=0, eps=1e-6)
            
            score_i = 0.0
            if clip_img_proto is not None:
                clip_img_norm = F.normalize(clip_img_proto, p=2, dim=0, eps=1e-6)
                score_i += alpha * (u_i_norm @ clip_img_norm).item()
            
            if clip_txt_proto is not None:
                clip_txt_norm = F.normalize(clip_txt_proto, p=2, dim=0, eps=1e-6)
                score_i += beta * (u_i_norm @ clip_txt_norm).item()
            
            scores[i] = score_i
        
            # 归一化到 [0, 1] 并 softmax
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            w = F.softmax(scores, dim=0)
    else:
            # 无 CLIP 先验，使用 log 特征值作为权重
            eps_val = tyler_params.get("eps", 1e-6)
            log_S = torch.log(S_r.clamp(min=eps_val))
            w = F.softmax(log_S, dim=0)
            scores = log_S
    
    # 构建投影矩阵
    eps_val = tyler_params.get("eps", 1e-6)
    if use_whitening:
        # 白化：P_r = U_r diag(w ⊙ λ^{-1/2}) U_r^T
        sqrt_inv = torch.sqrt(1.0 / S_r.clamp(min=eps_val))
        P_r = U_r @ torch.diag(w * sqrt_inv) @ U_r.T
    else:
        # 标准投影：P_r = U_r diag(w) U_r^T
        P_r = U_r @ torch.diag(w) @ U_r.T
    
    # 稳健投影
    p_hat = P_r @ p_bar
    p_hat = F.normalize(p_hat, p=2, dim=0, eps=1e-6)
    
    return {
        "p_hat": p_hat,
        "U_r": U_r,
        "S_r": S_r,
        "P_r": P_r,
        "scores": scores,
        "p_bar": p_bar,
        "Sigma": Sigma,
    }


@torch.no_grad()
def class_prototypes_robust(
    ref_feats_by_class: Dict[int, torch.Tensor],
    cfg: Dict[str, Any],
    clip_hooks: Optional[Dict[str, Callable]] = None,
    debug_store: Optional[Dict[int, Dict[str, Any]]] = None,
    store_raw_feats: bool = False,
) -> Dict[int, List[torch.Tensor]]:
    """使用稳健 PP 为每个类生成原型/子原型。
    
    参数
    ----
    ref_feats_by_class : Dict[int, torch.Tensor]
        按类别索引的参考特征，每个 ``(n_k, d)``
    cfg : Dict[str, Any]
        PP 配置字典
    clip_hooks : Optional[Dict[str, Callable]]
        CLIP 钩子函数字典
    debug_store : Optional[Dict[int, Dict[str, Any]]]
        调试信息存储
    store_raw_feats : bool
        是否存储原始特征
        
    返回
    ----
    Dict[int, List[torch.Tensor]]
        每个类的原型列表 ``{cls_id: [p_hat, ...]}``
    """
    results = {}
    
    # Tyler's 参数
    robust_cfg = cfg.get("robust", {})
    tyler_params = {
        "iters": robust_cfg.get("iters", 20),
        "tol": robust_cfg.get("tol", 1e-4),
        "init": robust_cfg.get("init", "identity"),
        "eps": robust_cfg.get("eps", 1e-6),
        "trace_norm": robust_cfg.get("trace_norm", True),
        "verbose": robust_cfg.get("verbose", False),
    }
    
    top_r = cfg.get("top_r", 32)
    alpha = cfg.get("alpha", 0.7)
    beta = cfg.get("beta", 0.3)
    use_whitening = cfg.get("use_whitening", False)
    use_subs = cfg.get("use_subprototypes", False)
    
    for cls_id, feats in ref_feats_by_class.items():
        if feats.numel() == 0:
            results[cls_id] = []
            continue
        
        n = feats.shape[0]
        
        # 归一化
        feats_norm = F.normalize(feats, p=2, dim=-1, eps=1e-6)
        
        # 获取 CLIP 先验
        clip_img = None
        clip_txt = None
        if clip_hooks:
            img_hook = clip_hooks.get("get_image_proto")
            txt_hook = clip_hooks.get("get_text_proto")
            
            if img_hook is not None:
                try:
                    clip_img = img_hook(cls_id)
                except Exception:
                    pass
            
            if txt_hook is not None:
                try:
                    clip_txt = txt_hook(cls_id)
                except Exception:
                    pass
        
        # 稳健净化
        purify_res = prototype_purify_robust(
            X_k=feats_norm,
            top_r=top_r,
            alpha=alpha,
            beta=beta,
            use_whitening=use_whitening,
            clip_img_proto=clip_img,
            clip_txt_proto=clip_txt,
            tyler=tyler_params,
        )
        
        # 主原型
        p_hat = purify_res["p_hat"]
        results[cls_id] = [p_hat]
        
        # 子原型（可选）
        if use_subs and n > 10:  # 样本足够多才做聚类
            try:
                cluster_cfg = cfg.get("cluster", {})
                method = cluster_cfg.get("method", "hdbscan")
                max_k = cluster_cfg.get("max_k", 4)
                
                # 投影到稳健子空间
                Z = (purify_res["U_r"].T @ feats_norm.T).T  # (n, r)
                
                # 聚类
                if method == "hdbscan":
                    try:
                        import hdbscan
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=cluster_cfg.get("min_size", 10),
                            min_samples=1,
                        )
                        labels = clusterer.fit_predict(Z.cpu().numpy())
                        labels = torch.from_numpy(labels).to(Z.device)
                    except Exception:
                        warnings.warn("HDBSCAN failed, using k-means", RuntimeWarning)
                        from sklearn.cluster import KMeans
                        km = KMeans(n_clusters=min(max_k, n // 10), random_state=42)
                        labels = torch.from_numpy(km.fit_predict(Z.cpu().numpy())).to(Z.device)
                else:
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=min(max_k, n // 10), random_state=42)
                    labels = torch.from_numpy(km.fit_predict(Z.cpu().numpy())).to(Z.device)
                
                # 每簇计算子原型
                unique_labels = torch.unique(labels)
                unique_labels = unique_labels[unique_labels >= 0]  # 排除噪声点（-1）
                
                for label in unique_labels[:max_k]:
                    mask = (labels == label)
                    if mask.sum() < 3:
                        continue
                    
                    # 簇均值
                    cluster_feats = feats_norm[mask]
                    p_cluster = cluster_feats.mean(dim=0)
                    
                    # 稳健投影
                    p_cluster_purified = purify_res["P_r"] @ p_cluster
                    p_cluster_purified = F.normalize(p_cluster_purified, p=2, dim=0, eps=1e-6)
                    
                    results[cls_id].append(p_cluster_purified)
                    
            except Exception as e:
                warnings.warn(f"Subprototype clustering failed for class {cls_id}: {e}", RuntimeWarning)
        
        # 存储调试信息
        if debug_store is not None:
            debug_store[cls_id] = {
                "purify": purify_res,
                "prototypes": results[cls_id],
            }
            if store_raw_feats:
                max_samples = min(feats.shape[0], 2048)
                debug_store[cls_id]["raw_feats"] = feats[:max_samples].detach().cpu()
    
    return results


# ═══════════════════════════════════════════════════════════════
# OT-PP: DINOv3↔CLIP Sinkhorn-OT Alignment
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def prototype_purify_ot(
    X_k: torch.Tensor,
    top_r: int = 32,
    mode: str = "prior",
    kappa: float = 2.0,
    use_whitening: bool = False,
    reg: float = 0.05,
    debias: bool = True,
    max_iters: int = 200,
    tol: float = 1e-6,
    stabilize: str = "log",
    a: str = "uniform",
    b: str = "uniform",
    sample_cap_n: int = 2000,
    sample_cap_m: int = 256,
    clip_img_proto: Optional[torch.Tensor] = None,
    clip_txt_proto: Optional[torch.Tensor] = None,
    robust: bool = False,
    tyler_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """使用 Sinkhorn-OT 对齐进行原型净化。
    
    完整流程：
    1) 组装 CLIP 锚集合 G_k
    2) 计算 DINO-CLIP 的 OT 耦合 Π*
    3) 重心映射得到对齐向量 ẑp_k
    4) 结合谱净化（prior/barycentric模式）
    
    参数
    ----
    X_k : torch.Tensor
        DINO类内特征，形状 ``(n, d)``
    top_r : int
        保留的主成分数量
    mode : str
        对齐模式：'prior' 或 'barycentric'
    kappa : float
        prior模式的softmax温度
    use_whitening : bool
        是否白化投影
    reg : float
        熵正则化参数
    debias : bool
        是否计算Sinkhorn散度（去偏）
    max_iters : int
        Sinkhorn最大迭代次数
    tol : float
        Sinkhorn收敛容差
    stabilize : str
        稳定化方法：'log' 或 'kernel'
    a, b : str
        边际分布：'uniform'
    sample_cap_n : int
        DINO侧子采样上限
    sample_cap_m : int
        CLIP侧子采样上限
    clip_img_proto : Optional[torch.Tensor]
        CLIP图像原型，形状 ``(m, d)`` 或 ``(d,)``
    clip_txt_proto : Optional[torch.Tensor]
        CLIP文本原型，形状 ``(m, d)`` 或 ``(d,)``
    robust : bool
        是否使用Tyler's估计器
    tyler_kwargs : Optional[Dict[str, Any]]
        Tyler's参数
        
    返回
    ----
    Dict[str, Any]
        包含：p_hat, p_bar, p_tilde, U_r, S_r, P_r, scores, Pi, C, ot_cost, ot_div
    """
    from utils.ot import cosine_distance_matrix, sinkhorn_cost, sinkhorn_divergence, barycentric_map
    
    n, d = X_k.shape
    
    # 归一化DINO特征
    X_norm = F.normalize(X_k, p=2, dim=-1, eps=1e-6)
    
    # 计算原始均值
    p_bar = X_norm.mean(dim=0)
    p_bar = F.normalize(p_bar, p=2, dim=0, eps=1e-6)
    
    # 组装CLIP锚集合
    G_k_list = []
    if clip_img_proto is not None:
        if clip_img_proto.ndim == 1:
            clip_img_proto = clip_img_proto.unsqueeze(0)
        G_k_list.append(clip_img_proto)
    
    if clip_txt_proto is not None:
        if clip_txt_proto.ndim == 1:
            clip_txt_proto = clip_txt_proto.unsqueeze(0)
        G_k_list.append(clip_txt_proto)
    
    if len(G_k_list) == 0:
        # 无CLIP先验，回退到常规PP
        warnings.warn("No CLIP anchors provided for OT-PP, falling back to standard PP", RuntimeWarning)
        
        if robust:
            return prototype_purify_robust(
                X_k=X_norm,
                top_r=top_r,
                alpha=0.7,
                beta=0.3,
                use_whitening=use_whitening,
                tyler=tyler_kwargs,
            )
        else:
            return prototype_purify(
                X_k=X_norm,
                top_r=top_r,
                alpha=0.7,
                beta=0.3,
                use_whitening=use_whitening,
            )
    
    G_k = torch.cat(G_k_list, dim=0)  # (m, d)
    m = G_k.shape[0]
    
    # 子采样（如果需要）
    if n > sample_cap_n:
        from utils.ot import ot_subsample
        X_sub, indices_n, weights_n = ot_subsample(X_norm, sample_cap_n, method="random")
        a_vec = weights_n
    else:
        X_sub = X_norm
        a_vec = torch.ones(n, device=X_norm.device, dtype=torch.float32) / n
    
    if m > sample_cap_m:
        from utils.ot import ot_subsample
        G_sub, indices_m, weights_m = ot_subsample(G_k, sample_cap_m, method="random")
        b_vec = weights_m
    else:
        G_sub = G_k
        b_vec = torch.ones(m, device=G_k.device, dtype=torch.float32) / m
    
    # 计算成本矩阵
    C = cosine_distance_matrix(X_sub, G_sub)
    
    # Sinkhorn OT
    ot_cost_val, Pi = sinkhorn_cost(a_vec, b_vec, C, reg=reg, max_iters=max_iters, tol=tol, stabilize=stabilize)
    
    # 可选：计算Sinkhorn散度
    ot_div = None
    if debias:
        try:
            div, div_info = sinkhorn_divergence(X_sub, G_sub, reg=reg, max_iters=max_iters, tol=tol, stabilize=stabilize)
            ot_div = div.item()
        except Exception as e:
            warnings.warn(f"Sinkhorn divergence computation failed: {e}", RuntimeWarning)
    
    # 重心映射
    p_tilde = barycentric_map(X_sub, Pi, to="A")
    
    # 计算协方差与谱分解
    if robust:
        tyler_params = tyler_kwargs or {}
        U_r, S_r, Sigma = robust_cov_eig(X_norm, top_r=top_r, **tyler_params)
    else:
        U_r, S_r = compute_cov_eig(X_norm, top_r=top_r, method="eigh")
        Sigma = None
    
    # 根据模式构建投影
    if mode == "prior":
        # prior模式：用ẑp与谱基的相似度打分
        scores = torch.zeros(top_r, device=X_norm.device, dtype=torch.float32)
        for i in range(top_r):
            u_i = F.normalize(U_r[:, i], p=2, dim=0, eps=1e-6)
            scores[i] = (u_i @ p_tilde).item()
        
        # softmax with temperature
        w = F.softmax(kappa * scores, dim=0)
        
        # 投影矩阵
        if use_whitening:
            sqrt_inv = torch.sqrt(1.0 / S_r.clamp(min=1e-6))
            P_r = U_r @ torch.diag(w * sqrt_inv) @ U_r.T
        else:
            P_r = U_r @ torch.diag(w) @ U_r.T
        
        # 投影均值
        p_hat = P_r @ p_bar
        p_hat = F.normalize(p_hat, p=2, dim=0, eps=1e-6)
        
    elif mode == "barycentric":
        # barycentric模式：直接投影对齐向量
        if use_whitening:
            sqrt_inv = torch.sqrt(1.0 / S_r.clamp(min=1e-6))
            # 均匀权重白化
            w = torch.ones(top_r, device=X_norm.device) / top_r
            P_r = U_r @ torch.diag(w * sqrt_inv) @ U_r.T
        else:
            # 标准投影（均匀权重）
            w = torch.ones(top_r, device=X_norm.device) / top_r
            P_r = U_r @ torch.diag(w) @ U_r.T
        
        # 投影对齐向量
        p_hat = P_r @ p_tilde
        p_hat = F.normalize(p_hat, p=2, dim=0, eps=1e-6)
        
        scores = torch.zeros(top_r, device=X_norm.device)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return {
        "p_hat": p_hat,
        "p_bar": p_bar,
        "p_tilde": p_tilde,
        "U_r": U_r,
        "S_r": S_r,
        "P_r": P_r,
        "scores": scores,
        "Pi": Pi,
        "C": C,
        "ot_cost": ot_cost_val.item() if hasattr(ot_cost_val, 'item') else float(ot_cost_val),
        "ot_div": ot_div,
        "Sigma": Sigma,
    }


@torch.no_grad()
def class_prototypes_ot(
    ref_feats_by_class: Dict[int, torch.Tensor],
    cfg: Dict[str, Any],
    clip_hooks: Optional[Dict[str, Callable]] = None,
    debug_store: Optional[Dict[int, Dict[str, Any]]] = None,
    store_raw_feats: bool = False,
) -> Dict[int, List[torch.Tensor]]:
    """使用OT-PP为每个类生成原型/子原型。
    
    参数
    ----
    ref_feats_by_class : Dict[int, torch.Tensor]
        按类别索引的参考特征
    cfg : Dict[str, Any]
        PP配置字典
    clip_hooks : Optional[Dict[str, Callable]]
        CLIP钩子函数
    debug_store : Optional[Dict[int, Dict[str, Any]]]
        调试信息存储
    store_raw_feats : bool
        是否存储原始特征
        
    返回
    ----
    Dict[int, List[torch.Tensor]]
        每个类的原型列表
    """
    results = {}
    
    # OT参数
    ot_cfg = cfg.get("ot", {})
    mode = ot_cfg.get("mode", "prior")
    kappa = ot_cfg.get("kappa", 2.0)
    reg = ot_cfg.get("reg", 0.05)
    debias = ot_cfg.get("debias", True)
    
    # PP参数
    top_r = cfg.get("top_r", 32)
    use_whitening = cfg.get("use_whitening", False)
    use_subs = cfg.get("use_subprototypes", False)
    robust_enable = cfg.get("robust", {}).get("enable", False)
    
    for cls_id, feats in ref_feats_by_class.items():
        if feats.numel() == 0:
            results[cls_id] = []
            continue
        
        n = feats.shape[0]
        
        # 获取CLIP先验
        clip_img = None
        clip_txt = None
        if clip_hooks:
            img_hook = clip_hooks.get("get_image_proto")
            txt_hook = clip_hooks.get("get_text_proto")
            
            if img_hook:
                try:
                    clip_img = img_hook(cls_id)
                except Exception:
                    pass
            
            if txt_hook:
                try:
                    clip_txt = txt_hook(cls_id)
                except Exception:
                    pass
        
        # OT净化
        purify_res = prototype_purify_ot(
            X_k=feats,
            top_r=top_r,
            mode=mode,
            kappa=kappa,
            use_whitening=use_whitening,
            reg=reg,
            debias=debias,
            max_iters=max_iters,
            tol=tol,
            stabilize=stabilize,
            sample_cap_n=sample_cap_n,
            sample_cap_m=sample_cap_m,
            clip_img_proto=clip_img,
            clip_txt_proto=clip_txt,
            robust=robust_enable,
            tyler_kwargs=cfg.get("robust", {}) if robust_enable else None,
        )
        
        p_hat = purify_res["p_hat"]
        results[cls_id] = [p_hat]
        
        # 子原型（可选）
        if use_subs and n > 10:
            try:
                cluster_cfg = cfg.get("cluster", {})
                method = cluster_cfg.get("method", "hdbscan")
                max_k = cluster_cfg.get("max_k", 4)
                
                # 投影到子空间
                Z = (purify_res["U_r"].T @ feats.T).T
                
                # 聚类
                if method == "hdbscan":
                    try:
                        import hdbscan
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=cluster_cfg.get("min_size", 10),
                            min_samples=1,
                        )
                        labels = clusterer.fit_predict(Z.cpu().numpy())
                        labels = torch.from_numpy(labels).to(Z.device)
                    except Exception:
                        from sklearn.cluster import KMeans
                        km = KMeans(n_clusters=min(max_k, n // 10), random_state=42)
                        labels = torch.from_numpy(km.fit_predict(Z.cpu().numpy())).to(Z.device)
                else:
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=min(max_k, n // 10), random_state=42)
                    labels = torch.from_numpy(km.fit_predict(Z.cpu().numpy())).to(Z.device)
                
                # 簇子原型
                unique_labels = torch.unique(labels)
                unique_labels = unique_labels[unique_labels >= 0]
                
                for label in unique_labels[:max_k]:
                    mask = (labels == label)
                    if mask.sum() < 3:
                        continue
                    
                    cluster_feats = feats[mask]
                    cluster_feats_norm = F.normalize(cluster_feats, p=2, dim=-1, eps=1e-6)
                    p_cluster = cluster_feats_norm.mean(dim=0)
                    
                    p_cluster_purified = purify_res["P_r"] @ p_cluster
                    p_cluster_purified = F.normalize(p_cluster_purified, p=2, dim=0, eps=1e-6)
                    
                    results[cls_id].append(p_cluster_purified)
                    
            except Exception as e:
                warnings.warn(f"OT-PP subprototype clustering failed for class {cls_id}: {e}", RuntimeWarning)
        
        # 存储调试信息
        if debug_store is not None:
            debug_store[cls_id] = {
                "purify": purify_res,
                "prototypes": results[cls_id],
            }
            if store_raw_feats:
                max_samples = min(feats.shape[0], 2048)
                debug_store[cls_id]["raw_feats"] = feats[:max_samples].detach().cpu()
    
    return results


__all__ = [
    "percent_clip",
    "compute_cov_eig",
    "cross_source_scores",
    "prototype_purify",
    "subprototypes_spectral",
    "class_prototypes",
    # Robust-PP extensions
    "tyler_covariance",
    "robust_cov_eig",
    "prototype_purify_robust",
    "class_prototypes_robust",
    # OT-PP extensions
    "prototype_purify_ot",
    "class_prototypes_ot",
]


