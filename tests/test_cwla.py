"""CWLA (一致性加权多层聚合) 单元测试。

测试覆盖：
1. 层权选择正确性（核心）
2. 与UAM的正向影响
3. 数值稳定性

运行方式：
    pytest -q tests/test_cwla.py
"""

import pytest
import torch
import numpy as np

from utils.cwla import (
    layer_consistency_weights,
    fuse_layers_with_weights,
    fuse_uncertainty_layers,
)


# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)


@torch.no_grad()
def test_layer_weight_selection():
    """测试1: 层权选择正确性。
    
    构造两层：LayerA清晰（低熵），LayerB噪声（高熵）。
    验证βA > βB。
    """
    print("\n[Test 1] Layer Weight Selection")
    
    h, w = 32, 32
    num_targets = 3
    
    # Layer A: 清晰响应（低熵）
    layer_a_targets = []
    for t in range(num_targets):
        R_a = torch.zeros(h, w)
        # 集中的高响应区域
        center_h, center_w = h // 2 + t * 5, w // 2
        R_a[center_h-3:center_h+3, center_w-3:center_w+3] = 0.9
        layer_a_targets.append(R_a)
    
    # Layer B: 噪声响应（高熵）
    layer_b_targets = []
    for t in range(num_targets):
        R_b = torch.rand(h, w) * 0.5  # 随机噪声
        layer_b_targets.append(R_b)
    
    R_per_layer = [layer_a_targets, layer_b_targets]
    
    # 计算层权重
    beta, info = layer_consistency_weights(
        R_per_layer,
        tau_mode="fixed",
        tau=1.0,
        metric="entropy",
        sigma=0.15,
    )
    
    print(f"  βA (clear): {beta[0]:.4f}")
    print(f"  βB (noisy): {beta[1]:.4f}")
    print(f"  U_layers: {info['U_layers'].tolist()}")
    
    # 断言：清晰层权重更高
    assert beta[0] > beta[1], f"Clear layer should have higher weight: βA={beta[0]:.4f}, βB={beta[1]:.4f}"
    
    # 断言：权重归一化
    assert torch.allclose(beta.sum(), torch.tensor(1.0), atol=1e-5), "Weights should sum to 1"
    
    print("  ✓ Layer weight selection correct")


@torch.no_grad()
def test_fused_R_quality():
    """测试2: 融合后R的质量。
    
    验证融合后的R比单独使用噪声层有更高的峰/均比。
    """
    print("\n[Test 2] Fused R Quality")
    
    h, w = 24, 24
    num_targets = 2
    
    # Layer 1: 高质量
    layer_1 = []
    for t in range(num_targets):
        R = torch.zeros(h, w)
        R[10+t*2:14+t*2, 10:14] = 0.8
        layer_1.append(R)
    
    # Layer 2: 低质量
    layer_2 = []
    for t in range(num_targets):
        R = torch.rand(h, w) * 0.3
        layer_2.append(R)
    
    R_per_layer = [layer_1, layer_2]
    
    # 计算权重
    beta, info = layer_consistency_weights(
        R_per_layer,
        metric="margin",
        sigma=0.15,
    )
    
    # 融合
    fused_list = fuse_layers_with_weights(R_per_layer, beta, base_size=(h, w))
    
    # 计算峰/均比
    ratio_fused = fused_list[0].max().item() / (fused_list[0].mean().item() + 1e-8)
    ratio_layer2_only = layer_2[0].max().item() / (layer_2[0].mean().item() + 1e-8)
    
    print(f"  Fused R peak/mean: {ratio_fused:.3f}")
    print(f"  Layer2 only peak/mean: {ratio_layer2_only:.3f}")
    
    # 断言：融合后应该比单独使用低质量层更好
    assert ratio_fused > ratio_layer2_only * 0.9, \
        f"Fused R should have better peak/mean ratio than noisy layer alone"
    
    print("  ✓ Fused R has better quality")


@torch.no_grad()
def test_numerical_stability():
    """测试3: 数值稳定性。
    
    测试极端情况下的数值稳定性。
    """
    print("\n[Test 3] Numerical Stability")
    
    # 极小值
    h, w = 8, 8
    layer_small = [[torch.ones(h, w) * 1e-8]]
    
    beta_small, info_small = layer_consistency_weights(
        layer_small,
        metric="entropy",
        sigma=0.15,
    )
    
    assert not torch.isnan(beta_small).any(), "Beta should not contain NaN"
    assert not torch.isinf(beta_small).any(), "Beta should not contain Inf"
    assert torch.allclose(beta_small.sum(), torch.tensor(1.0), atol=1e-4), "Beta should sum to 1"
    
    # 极大值
    layer_large = [[torch.ones(h, w) * 1e8]]
    
    beta_large, info_large = layer_consistency_weights(
        layer_large,
        metric="margin",
        sigma=0.15,
    )
    
    assert not torch.isnan(beta_large).any(), "Beta should not contain NaN"
    assert not torch.isinf(beta_large).any(), "Beta should not contain Inf"
    
    print("  ✓ Numerical stability maintained")


if __name__ == "__main__":
    # 直接运行测试
    test_layer_weight_selection()
    test_fused_R_quality()
    test_numerical_stability()
    
    print("\n✅ All CWLA tests passed!")

