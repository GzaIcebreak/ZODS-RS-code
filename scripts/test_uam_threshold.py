"""直接测试 UAM 阈值效果的简化脚本"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 添加 modules 路径
import sys
sys.path.insert(0, '.')

from modules.uam_uncert_merge import pixelwise_distribution, bayes_merge


def test_threshold_effects():
    """测试不同阈值对 UAM 过滤的影响"""
    
    print("=== UAM 阈值效果测试 ===\n")
    
    # 模拟一些掩码和分数（类似真实数据）
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_masks = 50
    height, width = 64, 64
    
    # 生成模拟掩码分数（有高有低）
    mask_scores = torch.tensor([
        0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75, 0.72, 0.68, 0.65,  # 高置信度
        0.62, 0.58, 0.55, 0.52, 0.48, 0.45, 0.42, 0.38, 0.35, 0.32,  # 中等置信度
        0.28, 0.25, 0.22, 0.18, 0.15, 0.12, 0.08, 0.05, 0.02, 0.01   # 低置信度
    ] + [np.random.uniform(0.1, 0.9) for _ in range(20)])  # 随机填充
    
    mask_scores = mask_scores[:n_masks]
    
    # 模拟掩码
    masks = torch.rand(n_masks, height, width) > 0.7
    
    print(f"输入掩码数量: {n_masks}")
    print(f"分数范围: [{mask_scores.min().item():.3f}, {mask_scores.max().item():.3f}]")
    print()
    
    # 测试不同阈值
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        # 模拟 UAM 过滤逻辑
        keep_mask = mask_scores > threshold
        kept_count = keep_mask.sum().item()
        
        print(f"阈值 {threshold:.1f}: 保留 {kept_count:2d}/{n_masks} 个掩码 ({kept_count/n_masks*100:.1f}%)")
        
        if kept_count == 0:
            print("         → 阈值过高，无掩码通过！")
        elif kept_count < 5:
            print("         → 阈值较高，仅保留高置信度掩码")
        elif kept_count > 40:
            print("         → 阈值较低，大部分掩码通过")
    
    print()
    print("=== 建议阈值设置 ===")
    print("• 0.1-0.3: 宽松过滤，适合召回优先")
    print("• 0.4-0.6: 平衡过滤，常规使用") 
    print("• 0.7-0.9: 严格过滤，精度优先")
    
    return mask_scores


def analyze_real_debug_files():
    """分析真实的调试文件"""
    debug_dir = Path("./results_analysis/uam_debug")
    
    conf_file = debug_dir / "target_mask_confidences.pt"
    uncert_file = debug_dir / "target_uncertainty.pt"
    indices_file = debug_dir / "target_kept_indices.pt"
    
    if conf_file.exists():
        confidences = torch.load(conf_file)
        print(f"\n=== 真实掩码置信度分析 ===")
        print(f"掩码数量: {len(confidences)}")
        print(f"置信度范围: [{confidences.min().item():.3f}, {confidences.max().item():.3f}]")
        print(f"平均置信度: {confidences.mean().item():.3f}")
        print(f"置信度 > 0.5: {(confidences > 0.5).sum().item()}/{len(confidences)}")
        print(f"置信度 > 0.7: {(confidences > 0.7).sum().item()}/{len(confidences)}")
        print(f"置信度 > 0.9: {(confidences > 0.9).sum().item()}/{len(confidences)}")
        
        # 显示置信度分布
        plt.figure(figsize=(10, 6))
        plt.hist(confidences.numpy(), bins=30, alpha=0.7, color='blue')
        plt.axvline(x=0.5, color='red', linestyle='--', label='threshold=0.5')
        plt.axvline(x=0.7, color='orange', linestyle='--', label='threshold=0.7') 
        plt.axvline(x=0.9, color='green', linestyle='--', label='threshold=0.9')
        plt.xlabel('Mask Confidence')
        plt.ylabel('Count')
        plt.title('Mask Confidence Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(debug_dir / "confidence_histogram.png", dpi=150)
        plt.show()
        
    else:
        print(f"未找到置信度文件: {conf_file}")
    
    if indices_file.exists():
        indices = torch.load(indices_file)
        print(f"实际保留的掩码索引: {len(indices)} 个")
        print(f"索引范围: {indices.min().item()} - {indices.max().item()}")


if __name__ == "__main__":
    test_threshold_effects()
    analyze_real_debug_files()
