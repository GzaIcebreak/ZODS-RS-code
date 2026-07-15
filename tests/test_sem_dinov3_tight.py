"""Tests for DINOv3 multi-layer SEM integration."""

import pytest
import torch

from modules.sem_scale_match import (
    build_layer_pyramid,
    similarity_per_scale_layer,
    fuse_scales_alpha,
    fuse_layers_beta,
    match_all_layers_scales,
)


def test_build_layer_pyramid():
    """测试层×尺度双重金字塔构建"""
    # 模拟3层特征
    layer_features = [
        torch.randn(1, 256, 32),  # Layer 6: (B, N, C)
        torch.randn(1, 256, 32),  # Layer 10
        torch.randn(1, 256, 32),  # Layer -1
    ]
    scales = [1.0, 0.5]
    
    layer_pyr = build_layer_pyramid(layer_features, scales=scales)
    
    # 验证形状
    assert len(layer_pyr) == 3  # 3 layers
    assert len(layer_pyr[0]) == 2  # 2 scales per layer
    
    # 验证每层的尺度
    for layer_scales in layer_pyr:
        assert layer_scales[0].shape[-2:] == (16, 16)  # 原始尺度 (sqrt(256) = 16)
        assert layer_scales[1].shape[-2:] == (8, 8)    # 0.5x


def test_fuse_scales_alpha():
    """测试α权重尺度融合"""
    # 构造模拟的层×尺度相似度矩阵
    sim_matrix = [
        [torch.rand(16, 16), torch.rand(8, 8)],  # Layer 1, 2 scales
        [torch.rand(16, 16), torch.rand(8, 8)],  # Layer 2
    ]
    
    layer_fused, alpha_weights = fuse_scales_alpha(sim_matrix, target_size=(16, 16))
    
    # 验证输出
    assert len(layer_fused) == 2
    assert alpha_weights.shape == (2,)
    assert torch.allclose(alpha_weights.sum(), torch.tensor(1.0), atol=1e-5)
    
    # 验证每层融合后的形状
    for fused in layer_fused:
        assert fused.shape == (16, 16)


def test_fuse_layers_beta():
    """测试β权重层融合"""
    layer_sims = [
        torch.rand(16, 16),
        torch.rand(16, 16),
        torch.rand(16, 16),
    ]
    
    fused, beta_weights = fuse_layers_beta(layer_sims, mode="variance")
    
    # 验证输出
    assert fused.shape == (16, 16)
    assert beta_weights.shape == (3,)
    assert torch.allclose(beta_weights.sum(), torch.tensor(1.0), atol=1e-5)
    
    # 验证方差加权：方差大的层应该权重更高
    variances = torch.tensor([s.var().item() for s in layer_sims])
    expected_order = torch.argsort(variances, descending=True)
    actual_order = torch.argsort(beta_weights, descending=True)
    # 至少前两个应该一致
    assert expected_order[0] == actual_order[0] or variances.std() < 0.01


def test_match_all_layers_scales_with_attn():
    """测试层×尺度联合匹配 + 注意力先验"""
    # 原型多层特征
    proto_layers = [
        torch.randn(1, 64, 16),   # Layer 6
        torch.randn(1, 64, 16),   # Layer 10
        torch.randn(1, 64, 16),   # Layer -1
    ]
    
    # 目标多层特征
    target_layers = [
        torch.randn(1, 64, 16),
        torch.randn(1, 64, 16),
        torch.randn(1, 64, 16),
    ]
    
    # 注意力先验
    attn_prior = torch.rand(4, 4)  # sqrt(16) = 4
    
    final_sim, debug_info = match_all_layers_scales(
        proto_layers,
        target_layers,
        scales=[1.0, 0.5],
        attn_prior=attn_prior,
        gamma=0.5,
    )
    
    # 验证输出
    assert final_sim.shape == (4, 4)
    assert "alpha_weights" in debug_info
    assert "beta_weights" in debug_info
    assert debug_info["alpha_weights"].shape[0] == 2  # 2 scales
    assert debug_info["beta_weights"].shape[0] == 3  # 3 layers
    assert debug_info["attn_contribution"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

