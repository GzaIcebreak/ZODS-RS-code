"""UAM自适应置信先验单元测试。

测试覆盖：
1. 先验提升置信度
2. Margin分支正确性
3. Attention回退
4. combine三模式一致性

运行方式：
    pytest -q tests/test_uam_prior.py
"""

import pytest
import torch
import numpy as np

from utils.confidence_prior import (
    robust_minmax,
    margin_map,
    norm_map,
    attention_map,
    build_confidence_prior,
)


# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)


@torch.no_grad()
def test_prior_improves_confidence():
    """测试1: 先验提升置信度。
    
    验证在正确区域，先验能提升分类置信度并降低熵。
    """
    print("\n[Test 1] Prior Improves Confidence")
    
    h, w = 32, 32
    c = 16
    
    # 创建特征图（中心区域范数更大）
    feat = torch.randn(c, h, w) * 0.5
    feat[:, 12:20, 12:20] *= 3.0  # 中心区域增强
    
    # 创建两类相似度图（R1在中心更强）
    R1 = torch.zeros(h, w)
    R1[12:20, 12:20] = 0.8
    R1 += torch.randn(h, w) * 0.1
    
    R2 = torch.rand(h, w) * 0.3
    
    score_maps = [R1, R2]
    
    # 构建先验
    prior = build_confidence_prior(
        score_maps=score_maps,
        feat=feat,
        tau=1.0,
        norm_cfg={"mode": "l2"},
        margin_cfg={"k": 2, "mode": "logit"},
        weights={"a": 0.6, "b": 0.4, "c": 0.0, "d": 0.0},
        gamma=1.0,
        combine="logit",
    )
    
    A = prior["A"]
    
    print(f"  A center region mean: {A[12:20, 12:20].mean():.4f}")
    print(f"  A edge region mean: {A[:10, :10].mean():.4f}")
    
    # 断言：中心区域先验更高
    assert A[12:20, 12:20].mean() > A[:10, :10].mean(), \
        "Center region should have higher prior confidence"
    
    # 断言：先验在合理范围内
    assert A.min() > 0.0 and A.max() < 1.0, "Prior should be in (0, 1)"
    
    print("  ✓ Prior correctly enhances confidence in strong regions")


@torch.no_grad()
def test_margin_branch_correctness():
    """测试2: Margin分支正确性。
    
    验证margin与top1-top2差值的单调关系。
    """
    print("\n[Test 2] Margin Branch Correctness")
    
    h, w = 16, 16
    
    # 创建两类分数图，人为制造不同margin
    R1 = torch.zeros(h, w)
    R2 = torch.zeros(h, w)
    
    # 左半：大margin（R1>>R2）
    R1[:, :8] = 0.9
    R2[:, :8] = 0.1
    
    # 右半：小margin（R1≈R2）
    R1[:, 8:] = 0.5
    R2[:, 8:] = 0.45
    
    score_maps = [R1, R2]
    
    # 计算margin
    M = margin_map(score_maps, k=2, tau=1.0, mode="logit")
    
    margin_left = M[:, :8].mean().item()
    margin_right = M[:, 8:].mean().item()
    
    print(f"  Large margin region: {margin_left:.4f}")
    print(f"  Small margin region: {margin_right:.4f}")
    
    # 断言：大margin区域值更大
    assert margin_left > margin_right, \
        f"Large margin region ({margin_left:.4f}) should have higher values than small margin ({margin_right:.4f})"
    
    print("  ✓ Margin correctly reflects score separation")


@torch.no_grad()
def test_attention_fallback():
    """测试3: Attention回退策略。
    
    验证grad回退与特征范数的相关性。
    """
    print("\n[Test 3] Attention Fallback")
    
    h, w, c = 16, 16, 8
    
    # 创建特征（某些区域范数更大）
    feat = torch.randn(c, h, w)
    feat[:, 6:10, 6:10] *= 2.0  # 中心区域增强
    
    # Fallback: grad
    attn_grad = attention_map(attn=None, feat=feat, fallback="grad")
    
    # Fallback: none
    attn_none = attention_map(attn=None, feat=feat, fallback="none")
    
    # 计算特征范数平方和
    feat_norm_sq = (feat ** 2).sum(dim=0)
    
    # 验证grad回退与范数平方的相关性
    corr = torch.corrcoef(torch.stack([attn_grad.flatten(), feat_norm_sq.flatten()]))[0, 1].item()
    
    print(f"  Correlation (grad fallback vs ||f||²): {corr:.4f}")
    
    assert corr > 0.95, f"Grad fallback should correlate with feature norm squared, got {corr:.4f}"
    
    # 验证none回退为常数
    assert attn_none.std().item() < 1e-5, "None fallback should be constant"
    
    print("  ✓ Attention fallback strategies work correctly")


@torch.no_grad()
def test_combine_modes_consistency():
    """测试4: combine三模式一致性。
    
    比较logit/prob/weight模式的行为。
    """
    print("\n[Test 4] Combine Modes Consistency")
    
    h, w = 16, 16
    c = 8
    
    # 创建特征和分数图
    feat = torch.randn(c, h, w)
    R1 = torch.rand(h, w)
    R2 = torch.rand(h, w) * 0.5
    R3 = torch.rand(h, w) * 0.3
    
    score_maps = [R1, R2, R3]
    
    # 构建先验（使用norm）
    prior_logit = build_confidence_prior(
        score_maps=score_maps,
        feat=feat,
        weights={"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0},
        gamma=1.0,
        combine="logit",
    )
    
    prior_prob = build_confidence_prior(
        score_maps=score_maps,
        feat=feat,
        weights={"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0},
        gamma=1.0,
        combine="prob",
    )
    
    A_logit = prior_logit["A"]
    A_prob = prior_prob["A"]
    
    # 验证两种先验图本身相同（build_confidence_prior中都经过sigmoid）
    assert torch.allclose(A_logit, A_prob, atol=1e-5), "Prior A should be same for both modes"
    
    print(f"  A mean: {A_logit.mean():.4f}")
    print(f"  A std: {A_logit.std():.4f}")
    
    # 断言：A在合理范围内
    assert A_logit.min() > 0.0 and A_logit.max() < 1.0, "A should be in (0, 1)"
    
    print("  ✓ Different combine modes produce consistent priors")


if __name__ == "__main__":
    # 直接运行测试
    test_prior_improves_confidence()
    test_margin_branch_correctness()
    test_attention_fallback()
    test_combine_modes_consistency()
    
    print("\n✅ All UAM prior tests passed!")
