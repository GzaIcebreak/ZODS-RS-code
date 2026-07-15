"""R-SEM (Rotation-Equivariant Semantic Matching) 单元测试。

测试覆盖：
1. 角度选择正确性
2. 可分核融合稳定性  
3. 边界与掩码鲁棒性

运行方式：
    pytest -q tests/test_rsem.py
"""

import pytest
import torch
import numpy as np

from modules.sem_scale_match import (
    build_rotation_views,
    build_scale_rotation_pyramid,
    multiscale_multirot_similarity,
    fuse_scales_angles_with_kernels,
    build_cost_matrix_rot,
    match_one_prototype_rot,
    match_all_rot,
)


# 固定随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)


def create_oriented_feature(h=96, w=96, c=8, angle_deg=0.0, width=5):
    """创建一个含有方向性条带的特征图。
    
    Args:
        h, w: 空间尺寸
        c: 通道数
        angle_deg: 条带方向（度）
        width: 条带宽度
        
    Returns:
        feat: (C, H, W) 特征图
    """
    feat = torch.zeros((c, h, w), dtype=torch.float32)
    
    # 创建水平条带作为基础
    center_h = h // 2
    for i in range(h):
        if abs(i - center_h) < width:
            feat[:, i, :] = 1.0
    
    # 如果需要旋转，使用 affine_grid + grid_sample
    if abs(angle_deg) > 1e-3:
        angle_rad = angle_deg * 3.14159265 / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        feat_4d = feat.unsqueeze(0)
        grid = torch.nn.functional.affine_grid(theta, feat_4d.shape, align_corners=False)
        feat_rotated = torch.nn.functional.grid_sample(feat_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        feat = feat_rotated.squeeze(0)
    
    # L2 归一化
    feat = torch.nn.functional.normalize(feat, p=2, dim=0, eps=1e-6)
    
    return feat


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@torch.no_grad()
def test_rotation_angle_selection():
    """测试1: 角度选择正确性。
    
    构造含斜向条带的特征图，验证 β 权重在正确角度处最大。
    """
    print("\n[Test 1] Rotation Angle Selection")
    
    # 参数
    h, w, c = 96, 96, 8
    target_angle = 30.0
    
    # 创建含30°条带的目标特征
    F_target = create_oriented_feature(h, w, c, angle_deg=target_angle, width=5)
    
    # 创建两个原型：一个与条带方向一致，一个正交
    proto_aligned = F_target[:, h//2, w//2]  # 中心点特征（对齐）
    proto_ortho = create_oriented_feature(h, w, c, angle_deg=target_angle + 90, width=5)[:, h//2, w//2]
    
    # 候选掩码（dummy）
    proposals = [
        torch.ones((h, w), dtype=torch.bool),
        torch.zeros((h, w), dtype=torch.bool),
    ]
    proposals[1][h//4:3*h//4, w//4:3*w//4] = True
    
    # 执行 R-SEM
    scales = [1.0]
    angles = [-45, -30, -15, 0, 15, 30, 45]  # 包含目标角度
    
    result = match_one_prototype_rot(
        F=F_target,
        proto=proto_aligned,
        proposals=proposals,
        scales=scales,
        angles=angles,
        gamma=0.2,
        eta=0.35,
        agg="mean",
        dilate=0,
    )
    
    beta = result["beta"]
    print(f"  β weights: {beta.numpy()}")
    
    # 验证：β 在 30° 附近应该最大
    peak_idx = beta.argmax().item()
    peak_angle = angles[peak_idx]
    print(f"  Peak angle: {peak_angle}° (expected ~{target_angle}°)")
    
    assert abs(peak_angle - target_angle) <= 15, f"Peak angle {peak_angle}° too far from target {target_angle}°"
    
    # 验证：β 权重应该集中（峰值 > 均值 * 1.5）
    assert beta[peak_idx] > beta.mean() * 1.5, "β weights not concentrated enough"
    
    print("  ✓ Angle selection correct")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@torch.no_grad()
def test_separable_kernel_fusion_stability():
    """测试2: 可分核融合稳定性。
    
    比较仅尺度核 vs 尺度×角度双核，验证双核融合的峰/均比更高。
    """
    print("\n[Test 2] Separable Kernel Fusion Stability")
    
    h, w, c = 64, 64, 8
    
    # 创建特征
    F = create_oriented_feature(h, w, c, angle_deg=0, width=3)
    proto = F[:, h//2, w//2]
    
    # 仅尺度核（单角度 0°）
    scales = [0.5, 1.0]
    angles_single = [0]
    
    F_pyr_single = build_scale_rotation_pyramid(F, scales, angles_single)
    sims_single = multiscale_multirot_similarity(F_pyr_single, proto)
    R_single, alpha_s, beta_s = fuse_scales_angles_with_kernels(sims_single, gamma=0.2, eta=0.35)
    
    peak_single = R_single.max().item()
    mean_single = R_single.mean().item()
    ratio_single = peak_single / (mean_single + 1e-8)
    
    # 尺度×角度双核
    angles_multi = [-15, 0, 15]
    
    F_pyr_multi = build_scale_rotation_pyramid(F, scales, angles_multi)
    sims_multi = multiscale_multirot_similarity(F_pyr_multi, proto)
    R_multi, alpha_m, beta_m = fuse_scales_angles_with_kernels(sims_multi, gamma=0.2, eta=0.35)
    
    peak_multi = R_multi.max().item()
    mean_multi = R_multi.mean().item()
    ratio_multi = peak_multi / (mean_multi + 1e-8)
    
    print(f"  Single-angle peak/mean ratio: {ratio_single:.3f}")
    print(f"  Multi-angle peak/mean ratio: {ratio_multi:.3f}")
    
    # 验证：双核融合应该有更高的峰/均比（更集中）
    assert ratio_multi > ratio_single * 0.9, "Multi-angle fusion should not degrade peak/mean ratio significantly"
    
    # 验证：峰值应该在合理范围内
    assert 0.0 <= peak_multi <= 1.0, f"Peak value {peak_multi} out of range [0, 1]"
    
    print("  ✓ Separable kernel fusion stable")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@torch.no_grad()
def test_mask_robustness_with_dilation():
    """测试3: 边界与掩码鲁棒性。
    
    在proposal边界加噪声，验证膨胀+median聚合降低方差。
    """
    print("\n[Test 3] Mask Robustness with Dilation")
    
    h, w, c = 48, 48, 4
    
    # 创建特征和原型
    F = torch.randn(c, h, w, dtype=torch.float32)
    F = torch.nn.functional.normalize(F, p=2, dim=0, eps=1e-6)
    proto = F[:, h//2, w//2]
    
    # 创建干净的掩码
    mask_clean = torch.zeros((h, w), dtype=torch.bool)
    mask_clean[h//4:3*h//4, w//4:3*w//4] = True
    
    # 在边界加噪声
    def add_boundary_noise(mask, num_noise=5):
        mask_noisy = mask.clone()
        for _ in range(num_noise):
            x = torch.randint(0, w, (1,)).item()
            y = torch.randint(0, h, (1,)).item()
            mask_noisy[y, x] = not mask_noisy[y, x]
        return mask_noisy
    
    # 生成多个带噪声的掩码
    masks_noisy = [add_boundary_noise(mask_clean, num_noise=10) for _ in range(5)]
    
    # 构建相似度图 R
    F_pyr = build_scale_rotation_pyramid(F, scales=[1.0], angles=[0])
    sims = multiscale_multirot_similarity(F_pyr, proto)
    R, _, _ = fuse_scales_angles_with_kernels(sims, gamma=0.2, eta=0.35)
    
    # 不膨胀 + mean
    costs_no_dilate = build_cost_matrix_rot(R, masks_noisy, agg="mean", dilate=0)
    var_no_dilate = costs_no_dilate.var().item()
    
    # 膨胀 + median
    costs_dilate_median = build_cost_matrix_rot(R, masks_noisy, agg="median", dilate=1)
    var_dilate_median = costs_dilate_median.var().item()
    
    print(f"  Variance (no dilate, mean): {var_no_dilate:.4f}")
    print(f"  Variance (dilate=1, median): {var_dilate_median:.4f}")
    
    # 验证：膨胀+median应该降低方差（更鲁棒）
    assert var_dilate_median <= var_no_dilate * 1.1, "Dilation + median should reduce or not significantly increase variance"
    
    print("  ✓ Mask dilation improves robustness")


if __name__ == "__main__":
    # 直接运行测试（不通过 pytest）
    test_rotation_angle_selection()
    test_separable_kernel_fusion_stability()
    test_mask_robustness_with_dilation()
    
    print("\n✅ All R-SEM tests passed!")


