"""演示 UAM 像素级阈值效果的独立脚本"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加路径以导入 UAM 模块
sys.path.insert(0, '.')
from modules.uam_uncert_merge import pixelwise_distribution, bayes_merge

def demo_pixel_filtering():
    """演示像素级阈值过滤的效果"""
    
    print("=== UAM 像素级过滤演示 ===\n")
    
    # 模拟掩码数据
    torch.manual_seed(42)
    n_masks = 5
    h, w = 64, 64
    
    # 生成不同质量的掩码
    masks = []
    scores = []
    
    # 高质量掩码 (清晰边界)
    mask1 = torch.zeros(h, w)
    mask1[20:40, 20:40] = 1.0  # 方形区域
    masks.append(mask1)
    scores.append(0.9)
    
    # 中等质量掩码 (有噪声边界)
    mask2 = torch.zeros(h, w)
    mask2[15:45, 15:45] = torch.rand(30, 30) > 0.3  # 有噪声
    masks.append(mask2)
    scores.append(0.7)
    
    # 低质量掩码 (很多噪声)
    mask3 = torch.zeros(h, w)
    mask3[10:50, 10:50] = torch.rand(40, 40) > 0.6  # 稀疏噪声
    masks.append(mask3)
    scores.append(0.5)
    
    # 边缘模糊掩码
    mask4 = torch.zeros(h, w)
    x, y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    center_dist = torch.sqrt((x - 32)**2 + (y - 32)**2)
    mask4 = (center_dist < 15).float() + (center_dist < 20).float() * 0.5
    mask4 = torch.clamp(mask4, 0, 1)
    masks.append(mask4)
    scores.append(0.6)
    
    # 非常低质量掩码
    mask5 = torch.rand(h, w) > 0.8  # 随机稀疏点
    masks.append(mask5.float())
    scores.append(0.3)
    
    masks = torch.stack(masks)  # (5, h, w)
    scores = torch.tensor(scores)
    
    print(f"生成 {n_masks} 个测试掩码，分数: {scores.tolist()}")
    
    # 测试不同阈值的像素级过滤效果
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(len(thresholds), n_masks + 1, figsize=(18, 12))
    fig.suptitle('UAM 像素级阈值过滤效果对比', fontsize=16)
    
    # 显示原始掩码
    for j in range(n_masks):
        axes[0, j + 1].imshow(masks[j], cmap='gray')
        axes[0, j + 1].set_title(f'Mask {j+1} (score={scores[j]:.1f})')
        axes[0, j + 1].axis('off')
    axes[0, 0].text(0.5, 0.5, '原始掩码', ha='center', va='center', fontsize=14)
    axes[0, 0].axis('off')
    
    for t_idx, threshold in enumerate(thresholds):
        if t_idx == 0:
            continue  # 第一行已经显示原始掩码
        
        axes[t_idx, 0].text(0.5, 0.5, f'阈值 {threshold}', ha='center', va='center', fontsize=12)
        axes[t_idx, 0].axis('off')
        
        pixel_retained = []
        
        for j in range(n_masks):
            mask = masks[j]
            score = scores[j].item()
            
            # 模拟像素级概率：前景像素概率 = sigmoid(score)
            fg_prob = torch.sigmoid(torch.tensor(score)) * mask
            bg_prob = 1 - fg_prob
            
            # 应用阈值过滤
            confident_pixels = fg_prob > threshold
            refined_mask = mask.bool() & confident_pixels
            
            retained_ratio = refined_mask.sum().float() / mask.bool().sum().float() if mask.bool().sum() > 0 else 0
            pixel_retained.append(retained_ratio.item())
            
            # 可视化：绿色=保留，红色=过滤掉
            viz_mask = torch.zeros(h, w, 3)
            viz_mask[refined_mask, 1] = 1.0  # 绿色：保留的像素
            viz_mask[mask.bool() & (~confident_pixels), 0] = 1.0  # 红色：过滤的像素
            
            axes[t_idx, j + 1].imshow(viz_mask)
            axes[t_idx, j + 1].set_title(f'{retained_ratio*100:.0f}% kept')
            axes[t_idx, j + 1].axis('off')
        
        print(f"阈值 {threshold}: 像素保留率 = {[f'{x*100:.0f}%' for x in pixel_retained]}")
    
    plt.tight_layout()
    plt.savefig('./results_analysis/uam_debug/pixel_filtering_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 演示图已保存: ./results_analysis/uam_debug/pixel_filtering_demo.png")
    
    return masks, scores

def analyze_threshold_sensitivity():
    """分析阈值敏感性"""
    print("\n=== 阈值建议 ===")
    print("• 0.1-0.2: 宽松过滤，保留大部分边缘像素")
    print("• 0.3-0.5: 适中过滤，去除不确定边缘") 
    print("• 0.6-0.8: 严格过滤，只保留高置信度核心")
    print("• 0.9+: 极严格，可能过度去除有效像素")
    print("\n你的设置 threshold=0.9 属于极严格过滤，")
    print("建议先试试 0.5-0.7 看效果是否更好。")

if __name__ == "__main__":
    # 确保输出目录存在
    Path("./results_analysis/uam_debug").mkdir(parents=True, exist_ok=True)
    
    demo_pixel_filtering()
    analyze_threshold_sensitivity()

