"""Tests for Semantic Scale-aware Matching (SEM) module."""

import pytest
import torch

from modules.sem_scale_match import (
    build_feature_pyramid,
    multiscale_similarity,
    fuse_scales_with_kernel,
    match_one_prototype,
    match_all,
    SEMMatchResult,
)


def test_build_feature_pyramid():
    """测试多尺度特征金字塔构建"""
    features = torch.randn(1, 64, 32, 32)
    scales = [1.0, 0.5, 0.25]
    
    pyramid = build_feature_pyramid(features, scales=scales)
    
    # 验证金字塔尺度数
    assert len(pyramid) == len(scales)
    
    # 验证各尺度形状
    assert pyramid[0].shape == (1, 64, 32, 32)  # 原始尺度
    assert pyramid[1].shape == (1, 64, 16, 16)  # 0.5x
    assert pyramid[2].shape == (1, 64, 8, 8)    # 0.25x
    
    # 验证特征维度不变
    for p in pyramid:
        assert p.shape[1] == 64


def test_multiscale_similarity():
    """测试多尺度相似度计算"""
    # 创建原型和目标金字塔
    proto_pyramid = [
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 4, 4),
    ]
    target_pyramid = [
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 32, 8, 8),
    ]
    
    sim_maps = multiscale_similarity(proto_pyramid, target_pyramid, normalize=True)
    
    # 验证输出数量
    assert len(sim_maps) == 2
    
    # 验证形状与目标对齐
    assert sim_maps[0].shape == (16, 16)
    assert sim_maps[1].shape == (8, 8)
    
    # 验证归一化后的相似度范围
    for sim in sim_maps:
        assert sim.min() >= -1.0
        assert sim.max() <= 1.0


def test_fuse_scales_with_kernel():
    """测试尺度融合"""
    # 创建不同尺度的相似度图
    sim_maps = [
        torch.rand(16, 16) * 0.8,
        torch.rand(8, 8) * 0.6,
        torch.rand(4, 4) * 0.4,
    ]
    
    # 测试均匀融合
    fused, weights = fuse_scales_with_kernel(sim_maps, mode="uniform", target_size=(16, 16))
    
    # 验证输出形状
    assert fused.shape == (16, 16)
    assert weights.shape == (3,)
    
    # 验证权重归一化
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    
    # 测试方差加权
    fused_var, weights_var = fuse_scales_with_kernel(sim_maps, mode="variance", target_size=(16, 16))
    assert fused_var.shape == (16, 16)
    assert torch.allclose(weights_var.sum(), torch.tensor(1.0), atol=1e-5)


def test_match_one_prototype():
    """测试单原型匹配"""
    proto_feat = torch.randn(64, 8, 8)
    target_feats = torch.randn(64, 16, 16)
    scales = [1.0, 0.5]
    
    result = match_one_prototype(proto_feat, target_feats, scales=scales, top_k=5)
    
    # 验证结果类型
    assert isinstance(result, SEMMatchResult)
    
    # 验证匹配数量
    assert result.matched_indices.shape[0] == 5
    assert result.similarity_scores.shape[0] == 5
    
    # 验证尺度权重
    assert result.scale_weights is not None
    assert result.scale_weights.shape[0] == 2


def test_match_all_greedy():
    """测试批量贪心匹配"""
    proto_features = [
        torch.randn(32, 4, 4),
        torch.randn(32, 4, 4),
        torch.randn(32, 4, 4),
    ]
    target_features = torch.randn(32, 8, 8)
    
    results = match_all(proto_features, target_features, method="greedy")
    
    # 验证输出数量
    assert len(results) == 3
    
    # 验证每个结果的结构
    for r in results:
        assert isinstance(r, SEMMatchResult)
        assert r.matched_indices.numel() > 0
        assert r.similarity_scores.numel() > 0


def test_match_all_hungarian():
    """测试批量匈牙利匹配"""
    proto_features = [
        torch.randn(16, 4, 4),
        torch.randn(16, 4, 4),
    ]
    target_features = torch.randn(16, 8, 8)
    
    results = match_all(proto_features, target_features, method="hungarian")
    
    # 验证输出数量
    assert len(results) == 2
    
    # 验证结果有效性
    for r in results:
        assert isinstance(r, SEMMatchResult)
        assert r.matched_indices.numel() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

