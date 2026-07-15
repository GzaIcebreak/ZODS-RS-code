"""Tests for automatic temperature adjustment in UAM."""

import pytest
import torch

from utils.temperature import compute_auto_tau, summarize_smax
from utils.prior_maps import combine_priors, margin_prior, norm_prior
from modules.uam_uncert_merge import pixelwise_distribution


def test_compute_auto_tau_entropy_target():
    """测试基于目标熵的温度计算"""
    # 创建测试 logits
    logits = torch.randn(1, 3, 16, 16)
    
    # 计算自动温度
    tau, stats = compute_auto_tau(
        logits, 
        method="entropy_target", 
        target_entropy=0.8,
        tau_range=(0.5, 2.0)
    )
    
    # 验证温度在合理范围内
    assert 0.5 <= tau <= 2.0
    
    # 验证统计信息
    assert "entropy_mean" in stats
    assert "confidence_mean" in stats
    assert stats["entropy_mean"] > 0
    assert 0 < stats["confidence_mean"] <= 1
    
    # 验证熵接近目标值
    assert abs(stats["entropy_mean"] - 0.8) < 0.3  # 允许一定偏差


def test_compute_auto_tau_confidence_percentile():
    """测试基于置信度百分位的温度计算"""
    # 创建有明显峰值的 logits
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 1, 2:6, 2:6] = 3.0  # 高置信度区域
    
    tau, stats = compute_auto_tau(
        logits,
        method="confidence_percentile",
        percentile=75.0,
        tau_range=(0.1, 3.0)
    )
    
    # 验证温度合理
    assert 0.1 <= tau <= 3.0
    
    # 验证统计
    assert stats["confidence_p90"] > stats["confidence_mean"]
    assert stats["confidence_mean"] > 0.5  # 应该有较高置信度


def test_pixelwise_distribution_with_auto_tau():
    """测试 pixelwise_distribution 集成自动温度功能"""
    logits = torch.randn(2, 10, 10)
    
    # 配置自动温度
    auto_tau_cfg = {
        "enable": True,
        "method": "variance_based",
        "tau_range": [0.2, 2.0],
    }
    
    # 不带自动温度
    stats_manual = pixelwise_distribution(logits, temperature=1.0)
    
    # 带自动温度
    stats_auto = pixelwise_distribution(
        logits, 
        temperature=1.0,  # 应被忽略
        auto_tau_cfg=auto_tau_cfg,
        verbose=False
    )
    
    # 验证输出形状一致
    assert stats_manual.probs.shape == stats_auto.probs.shape
    assert stats_manual.entropy.shape == stats_auto.entropy.shape
    
    # 验证自动温度产生了不同的结果
    prob_diff = torch.abs(stats_manual.probs - stats_auto.probs).mean()
    assert prob_diff > 0.01  # 应该有明显差异


def test_summarize_smax():
    """测试 softmax 分布统计摘要"""
    # 创建已知分布
    probs = torch.zeros(1, 3, 4, 4)
    probs[:, 0] = 0.7  # 第一类占主导
    probs[:, 1] = 0.2
    probs[:, 2] = 0.1
    
    stats = summarize_smax(probs)
    
    # 验证统计项存在
    required_keys = [
        "entropy_mean", "confidence_mean", "confidence_p50",
        "confidence_p90", "variance_mean", "energy_mean"
    ]
    for key in required_keys:
        assert key in stats
        assert isinstance(stats[key], float)
    
    # 验证置信度合理（应该接近0.7）
    assert 0.6 < stats["confidence_mean"] < 0.8
    
    # 验证百分位数合理
    assert stats["confidence_p50"] <= stats["confidence_p90"]


def test_combine_priors_augment():
    """测试新增的 augment 组合模式可以覆盖补充区域"""
    masks = torch.zeros(2, 8, 8)
    masks[0, 1:5, 1:5] = 1.0  # margin prior
    masks[1, 3:7, 3:7] = 1.0  # norm prior

    margin = margin_prior(masks[0:1])
    norm = norm_prior(masks[1:2])

    augmented = combine_priors([margin.squeeze(0), norm.squeeze(0)], mode="augment")

    assert augmented.shape == margin.squeeze(0).shape
    # augment 应覆盖至少一个区域的激活
    assert augmented.max() <= 1.0
    assert augmented.min() >= 0.0
    # 边界区域应被保留
    assert augmented[2, 2] > 0.0
    assert augmented[4, 4] > 0.0


if __name__ == "__main__":
    pytest.main([__file__])
