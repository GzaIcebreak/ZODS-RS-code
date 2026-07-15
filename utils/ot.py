"""最优传输（Optimal Transport）工具集。

实现熵正则化最优传输（Sinkhorn算法）及相关工具，用于跨模态对齐。
所有函数均为训练-free、设备自适应、数值稳定。

核心接口：
- sinkhorn: 对数域稳定的 Sinkhorn-Knopp 算法
- sinkhorn_cost: 计算 OT 成本
- sinkhorn_divergence: 去偏的 Sinkhorn 散度
- barycentric_map: 重心映射
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_distance_matrix(
    A: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算余弦距离矩阵。
    
    参数
    ----
    A : torch.Tensor
        形状 ``(n, d)``
    B : torch.Tensor
        形状 ``(m, d)``
    eps : float
        归一化稳定项
        
    返回
    ----
    torch.Tensor
        余弦距离矩阵 C，形状 ``(n, m)``，C_ij = 1 - cos(A_i, B_j)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    
    if A.shape[1] != B.shape[1]:
        raise ValueError("Feature dimensions must match")
    
    # 确保 float32 精度
    orig_dtype = A.dtype
    A = A.to(dtype=torch.float32)
    B = B.to(dtype=torch.float32)
    
    # L2 归一化
    A_norm = F.normalize(A, p=2, dim=-1, eps=eps)
    B_norm = F.normalize(B, p=2, dim=-1, eps=eps)
    
    # 余弦相似度
    cosine_sim = A_norm @ B_norm.T  # (n, m)
    
    # 转为距离
    C = 1.0 - cosine_sim
    C = C.clamp(min=0.0, max=2.0)  # 理论范围 [0, 2]
    
    # 回落精度
    if orig_dtype != torch.float32:
        C = C.to(dtype=orig_dtype)
    
    return C


@torch.no_grad()
def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float = 0.05,
    max_iters: int = 200,
    tol: float = 1e-6,
    stabilize: str = "log",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """熵正则化最优传输（Sinkhorn-Knopp算法）。
    
    求解：
        min_{Π} <Π, C> + reg·H(Π)
        s.t. Π·1 = a, Π^T·1 = b
    
    参数
    ----
    a : torch.Tensor
        源分布，形状 ``(n,)``，应满足 sum(a)=1
    b : torch.Tensor
        目标分布，形状 ``(m,)``，应满足 sum(b)=1
    C : torch.Tensor
        成本矩阵，形状 ``(n, m)``
    reg : float
        熵正则化参数（越小越接近精确OT）
    max_iters : int
        最大迭代次数
    tol : float
        收敛容差
    stabilize : str
        稳定化方法：'log'（对数域）或 'kernel'（kernel版本）
        
    返回
    ----
    Tuple[torch.Tensor, Dict[str, Any]]
        (Π, info)，其中：
        - Π: 耦合矩阵 ``(n, m)``
        - info: 调试信息字典（iters, err, converged等）
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D vectors")
    
    if C.shape != (a.shape[0], b.shape[0]):
        raise ValueError("C shape must match (len(a), len(b))")
    
    # 归一化边际
    a = a / (a.sum() + 1e-10)
    b = b / (b.sum() + 1e-10)
    
    # 确保 float32
    orig_dtype = C.dtype
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    C = C.to(dtype=torch.float32)
    
    device = C.device
    n, m = C.shape
    
    if stabilize == "log":
        # 对数域 Sinkhorn（更稳定）
        log_a = torch.log(a + 1e-10)
        log_b = torch.log(b + 1e-10)
        
        # 初始化对偶变量
        f = torch.zeros(n, device=device, dtype=torch.float32)
        g = torch.zeros(m, device=device, dtype=torch.float32)
        
        # Gibbs kernel: K_ij = exp(-C_ij / reg)
        # 对数域：log K = -C/reg
        log_K = -C / reg
        
        err = float('inf')
        for it in range(max_iters):
            f_old = f.clone()
            
            # 更新 g
            # log b = log_K^T f + g
            log_sum = torch.logsumexp(log_K.T + f.unsqueeze(0), dim=1)
            g = log_b - log_sum
            
            # 更新 f
            # log a = log_K g + f
            log_sum = torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
            f = log_a - log_sum
            
            # 检查收敛
            err = (f - f_old).abs().max().item()
            
            if err < tol:
                break
        
        # 计算耦合矩阵
        log_Pi = f.unsqueeze(1) + log_K + g.unsqueeze(0)
        Pi = torch.exp(log_Pi)
        
        converged = (err < tol)
        
    elif stabilize == "kernel":
        # Kernel 版本（数值不稳定，但简单）
        K = torch.exp(-C / reg)
        
        u = torch.ones(n, device=device, dtype=torch.float32) / n
        
        err = float('inf')
        for it in range(max_iters):
            u_old = u.clone()
            
            v = b / (K.T @ u + 1e-10)
            u = a / (K @ v + 1e-10)
            
            err = (u - u_old).abs().max().item()
            
            if err < tol:
                break
        
        Pi = u.unsqueeze(1) * K * v.unsqueeze(0)
        converged = (err < tol)
        
    else:
        raise ValueError(f"Unknown stabilize method: {stabilize}")
    
    # 归一化（数值误差修正）
    Pi = Pi / (Pi.sum() + 1e-10)
    
    # 回落精度
    if orig_dtype != torch.float32:
        Pi = Pi.to(dtype=orig_dtype)
    
    info = {
        "iters": it + 1,
        "err": err,
        "converged": converged,
        "reg": reg,
    }
    
    if not converged:
        warnings.warn(f"Sinkhorn did not converge after {max_iters} iterations (err={err:.2e})", RuntimeWarning)
    
    return Pi, info


@torch.no_grad()
def sinkhorn_cost(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    reg: float,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算熵正则化 OT 成本。
    
    参数
    ----
    a : torch.Tensor
        源分布 ``(n,)``
    b : torch.Tensor
        目标分布 ``(m,)``
    C : torch.Tensor
        成本矩阵 ``(n, m)``
    reg : float
        熵正则化参数
    **kwargs :
        传递给 sinkhorn 的其他参数
        
    返回
    ----
    Tuple[torch.Tensor, torch.Tensor]
        (OT成本, Π)
        - OT成本: 标量，<Π, C> + reg·KL(Π||ab^T)
        - Π: 耦合矩阵 ``(n, m)``
    """
    Pi, info = sinkhorn(a, b, C, reg=reg, **kwargs)
    
    # 计算成本：<Π, C> + reg·KL(Π||ab^T)
    # KL(Π||ab^T) = Σ Π_ij (log Π_ij - log(a_i b_j))
    transport_cost = (Pi * C).sum()
    
    # 熵项（数值稳定）
    Pi_pos = Pi.clamp(min=1e-10)
    entropy = -(Pi_pos * torch.log(Pi_pos)).sum()
    
    ot_cost = transport_cost - reg * entropy
    
    return ot_cost, Pi


@torch.no_grad()
def sinkhorn_divergence(
    A: torch.Tensor,
    B: torch.Tensor,
    reg: float = 0.05,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """计算 Sinkhorn 散度（去偏版本）。
    
    S_ε(A,B) = OT_ε(A,B) - 0.5[OT_ε(A,A) + OT_ε(B,B)]
    
    参数
    ----
    A : torch.Tensor
        源特征集，形状 ``(n, d)``
    B : torch.Tensor
        目标特征集，形状 ``(m, d)``
    reg : float
        熵正则化参数
    **kwargs :
        传递给 sinkhorn 的其他参数
        
    返回
    ----
    Tuple[torch.Tensor, Dict[str, Any]]
        (散度, 调试信息)
        - 散度: 标量，去偏后的距离
        - 调试信息: 包含三个OT成本
    """
    n = A.shape[0]
    m = B.shape[0]
    
    # 均匀边际
    a = torch.ones(n, device=A.device, dtype=torch.float32) / n
    b = torch.ones(m, device=B.device, dtype=torch.float32) / m
    a_self = a.clone()
    b_self = b.clone()
    
    # 计算三个 OT 成本
    C_AB = cosine_distance_matrix(A, B)
    ot_AB, _ = sinkhorn_cost(a, b, C_AB, reg=reg, **kwargs)
    
    C_AA = cosine_distance_matrix(A, A)
    ot_AA, _ = sinkhorn_cost(a_self, a_self, C_AA, reg=reg, **kwargs)
    
    C_BB = cosine_distance_matrix(B, B)
    ot_BB, _ = sinkhorn_cost(b_self, b_self, C_BB, reg=reg, **kwargs)
    
    # 去偏
    div = ot_AB - 0.5 * (ot_AA + ot_BB)
    
    info = {
        "ot_AB": ot_AB.item(),
        "ot_AA": ot_AA.item(),
        "ot_BB": ot_BB.item(),
        "divergence": div.item(),
    }
    
    return div, info


@torch.no_grad()
def barycentric_map(
    X: torch.Tensor,
    Pi: torch.Tensor,
    to: str = "A",
) -> torch.Tensor:
    """计算重心映射。
    
    参数
    ----
    X : torch.Tensor
        特征矩阵，形状 ``(n, d)`` 或 ``(m, d)``
    Pi : torch.Tensor
        耦合矩阵，形状 ``(n, m)``
    to : str
        映射方向：'A'（映射到源空间）或 'B'（映射到目标空间）
        
    返回
    ----
    torch.Tensor
        重心向量，形状 ``(d,)``
        
    说明
    ----
    to='A': 返回 X·(Π·1_b) / ||Π·1_b||₁
    to='B': 返回 X·(Π^T·1_a) / ||Π^T·1_a||₁
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D matrix")
    
    if to == "A":
        # 映射到 A 空间（X=A）
        # 列边际：Π·1_b
        marginal = Pi.sum(dim=1)  # (n,)
        marginal = marginal / (marginal.sum() + 1e-10)
        
        # 重心
        barycenter = (X.T @ marginal)  # (d,)
        
    elif to == "B":
        # 映射到 B 空间（X=B）
        # 行边际：Π^T·1_a
        marginal = Pi.sum(dim=0)  # (m,)
        marginal = marginal / (marginal.sum() + 1e-10)
        
        # 重心
        barycenter = (X.T @ marginal)  # (d,)
        
    else:
        raise ValueError(f"Unknown target space: {to}")
    
    # L2 归一化
    barycenter = F.normalize(barycenter, p=2, dim=0, eps=1e-8)
    
    return barycenter


@torch.no_grad()
def ot_subsample(
    X: torch.Tensor,
    cap: int,
    method: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对大规模数据进行子采样（用于加速OT）。
    
    参数
    ----
    X : torch.Tensor
        特征矩阵 ``(n, d)``
    cap : int
        子采样上限
    method : str
        子采样方法：'random', 'fps'（farthest point sampling）
        
    返回
    ----
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (X_sub, indices, weights)
        - X_sub: 子采样后的特征 ``(cap, d)``
        - indices: 选中的索引 ``(cap,)``
        - weights: 重权重 ``(cap,)``，用于还原边际
    """
    n = X.shape[0]
    
    if n <= cap:
        # 无需子采样
        indices = torch.arange(n, device=X.device)
        weights = torch.ones(n, device=X.device, dtype=torch.float32) / n
        return X, indices, weights
    
    if method == "random":
        # 随机子采样
        indices = torch.randperm(n, device=X.device)[:cap]
        X_sub = X[indices]
        weights = torch.ones(cap, device=X.device, dtype=torch.float32) / cap
        
    elif method == "fps":
        # Farthest Point Sampling（简化版）
        indices = [torch.randint(0, n, (1,), device=X.device).item()]
        
        for _ in range(cap - 1):
            # 计算当前点到已选点的最小距离
            selected = X[indices]
            dists = torch.cdist(X, selected, p=2)  # (n, len(indices))
            min_dists = dists.min(dim=1).values
            
            # 选择最远点
            farthest = min_dists.argmax().item()
            indices.append(farthest)
        
        indices = torch.tensor(indices, device=X.device, dtype=torch.long)
        X_sub = X[indices]
        weights = torch.ones(cap, device=X.device, dtype=torch.float32) / cap
        
    else:
        raise ValueError(f"Unknown subsample method: {method}")
    
    return X_sub, indices, weights


__all__ = [
    "cosine_distance_matrix",
    "sinkhorn",
    "sinkhorn_cost",
    "sinkhorn_divergence",
    "barycentric_map",
    "ot_subsample",
]

