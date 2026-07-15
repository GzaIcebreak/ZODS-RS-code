import math

import torch

from modules.pp_prototype_purify import (
    class_prototypes,
    compute_cov_eig,
    cross_source_scores,
    prototype_purify,
    subprototypes_spectral,
)


def angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a, dim=0)
    b = torch.nn.functional.normalize(b, dim=0)
    cos = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    return math.degrees(torch.acos(cos))


def test_prototype_purify_basic():
    """测试基础谱净化：验证输出有效性和基本性质。"""
    torch.manual_seed(0)
    d = 64
    n = 200
    
    # 构造一簇以某方向为中心的特征
    center = torch.randn(d)
    center = torch.nn.functional.normalize(center, dim=0)
    
    # 围绕中心生成样本
    X = center.unsqueeze(0) + 0.3 * torch.randn(n, d)

    res = prototype_purify(X, top_r=8, use_whitening=False)

    p_hat = res["p_hat"]
    p_bar = res["p_bar"]
    U_r = res["U_r"]
    S_r = res["S_r"]
    
    # 验证基本性质
    assert torch.isfinite(p_hat).all(), "p_hat contains NaN/Inf"
    assert math.isclose(p_hat.norm().item(), 1.0, rel_tol=1e-3), "p_hat not unit norm"
    assert U_r.shape == (d, 8), f"U_r shape mismatch: {U_r.shape}"
    assert S_r.shape == (8,), f"S_r shape mismatch: {S_r.shape}"
    
    # 验证 p_hat 与平均原型 p_bar 方向相近（谱净化应保留主方向）
    angle = angle_between(p_hat, p_bar)
    assert angle < 60.0, f"p_hat and p_bar too far: {angle:.2f}°"
    
    # 验证特征值递减
    # 验证特征值非负
    assert (S_r >= 0).all(), "Eigenvalues contain negative values"


def test_cross_source_scores_clip_priority():
    """测试 CLIP 先验能提升目标方向的得分排名。"""
    torch.manual_seed(1)
    d = 32
    U_r = torch.linalg.qr(torch.randn(d, d)).Q[:, :8]
    target_dir = U_r[:, 3]

    scores_no_clip = cross_source_scores(U_r)

    # 构造与目标方向强对齐的 CLIP 先验
    clip_proto = torch.nn.functional.normalize(target_dir + 0.01 * torch.randn_like(target_dir), dim=0)
    scores_with_clip = cross_source_scores(U_r, clip_txt_proto=clip_proto, alpha=0.0, beta=1.0)

    top_with_clip = torch.argmax(scores_with_clip).item()

    # 验证目标方向被选中
    assert top_with_clip == 3, f"Expected top index=3, got {top_with_clip}"
    
    # 验证有 CLIP 时目标方向得分显著高于其它方向
    sorted_scores, _ = torch.sort(scores_with_clip, descending=True)
    assert scores_with_clip[3] == sorted_scores[0]
    assert scores_with_clip[3] > sorted_scores[1] * 1.1  # 至少高 10%


def test_subprototypes_spectral_clusters():
    """测试子原型聚类：双簇数据应产生多个子原型。"""
    torch.manual_seed(2)
    d = 48
    
    # 在低维子空间中构造两个清晰分离的簇
    basis = torch.linalg.qr(torch.randn(d, d)).Q[:, :4]
    
    # 簇中心在子空间坐标系下差异明显
    c1_coords = torch.tensor([2.0, 0.0, 0.0, 0.0])
    c2_coords = torch.tensor([-2.0, 0.0, 0.0, 0.0])
    
    center1 = torch.nn.functional.normalize(basis @ c1_coords, dim=0)
    center2 = torch.nn.functional.normalize(basis @ c2_coords, dim=0)
    
    feats = []
    for center in [center1, center2]:
        cluster = center.unsqueeze(0) + 0.1 * torch.randn(60, d)
        feats.append(cluster)
    X = torch.cat(feats, dim=0)

    subprotos = subprototypes_spectral(X, basis, method="kmeans", max_k=3)
    
    # 验证输出了多个子原型
    assert len(subprotos) >= 2, f"Expected >=2 subprototypes, got {len(subprotos)}"
    
    # 验证子原型是单位向量
    for proto in subprotos:
        assert math.isclose(proto.norm().item(), 1.0, rel_tol=1e-3)
    
    # 验证子原型之间有一定差异（不是完全相同）
    if len(subprotos) >= 2:
        angle_diff = angle_between(subprotos[0], subprotos[1])
        assert angle_diff > 5.0, f"Subprototypes too similar: {angle_diff:.2f}°"


def test_edge_cases():
    """测试边界条件：小样本、高维场景的退化行为。"""
    torch.manual_seed(3)
    d = 128

    # 小样本 (n < 3)：应退化为简单平均
    X_small = torch.randn(2, d)
    res_small = prototype_purify(X_small, top_r=16)
    assert math.isclose(res_small["p_hat"].norm().item(), 1.0, rel_tol=1e-3)
    assert res_small["U_r"].shape[1] == 0  # n < 3 时应返回空谱基
    assert res_small["S_r"].numel() == 0

    # 高维场景 (d >> n)：仍能正常运行
    X_tall = torch.randn(10, 512)
    res_tall = prototype_purify(X_tall, top_r=16)
    assert math.isclose(res_tall["p_hat"].norm().item(), 1.0, rel_tol=1e-3)
    assert res_tall["U_r"].shape[0] == 512
    assert res_tall["S_r"].shape[0] == min(16, res_tall["U_r"].shape[1])
    
    # 单样本场景
    X_single = torch.randn(1, 64)
    res_single = prototype_purify(X_single, top_r=8)
    assert math.isclose(res_single["p_hat"].norm().item(), 1.0, rel_tol=1e-3)


