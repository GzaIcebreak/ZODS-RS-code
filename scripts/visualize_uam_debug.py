"""可视化 UAM 调试产物 (.pt 文件)"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_uam_outputs(debug_dir, image_stem="target", save_dir=None):
    """
    可视化 UAM 调试输出
    
    参数:
        debug_dir: 调试文件所在目录 (如 ./results_analysis/uam_debug)
        image_stem: 图像文件名前缀 (如 "target")
        save_dir: 可视化结果保存目录 (默认与 debug_dir 相同)
    """
    debug_dir = Path(debug_dir)
    if save_dir is None:
        save_dir = debug_dir
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载文件
    probs_file = debug_dir / f"{image_stem}_fg_probs.pt"
    entropy_file = debug_dir / f"{image_stem}_entropy.pt"
    energy_file = debug_dir / f"{image_stem}_qualities.pt"
    masks_file = debug_dir / f"{image_stem}_refined_masks.pt"
    prior_file = debug_dir / f"{image_stem}_prior.pt"
    
    print(f"Loading UAM debug files from: {debug_dir}")
    
    outputs = {}
    if probs_file.exists():
        probs = torch.load(probs_file, map_location='cpu')
        outputs['probs'] = probs
        print(f"  [OK] Probs: {probs.shape}")
    
    if entropy_file.exists():
        entropy = torch.load(entropy_file, map_location='cpu')
        outputs['entropy'] = entropy
        print(f"  [OK] Entropy: {entropy.shape}")
    
    if energy_file.exists():
        energy = torch.load(energy_file, map_location='cpu')
        outputs['energy'] = energy
        print(f"  [OK] Energy: {type(energy)}")
    
    if masks_file.exists():
        masks = torch.load(masks_file, map_location='cpu')
        outputs['masks'] = masks
        print(f"  [OK] Masks: {masks.shape}")

    if prior_file.exists():
        prior = torch.load(prior_file, map_location='cpu')
        outputs['prior'] = prior
        print(f"  [OK] Prior: {prior.shape}")
    
    if not outputs:
        print("[ERROR] No debug files found!")
        return
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'UAM Debug Visualization - {image_stem}', fontsize=16)
    
    # 1. 最大概率
    if 'probs' in outputs:
        probs = outputs['probs']
        if probs.dim() == 3:
            avg_prob = probs.mean(dim=0)
            im0 = axes[0, 0].imshow(avg_prob.numpy(), cmap='jet')
            axes[0, 0].set_title('FG Probability (avg)')
            axes[0, 0].axis('off')
            plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
            
            im1 = axes[0, 1].imshow(probs.max(dim=0).values.numpy(), cmap='Blues')
            axes[0, 1].set_title('FG Max Probability')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        else:
            axes[0, 0].text(0.5, 0.5, f'Probs shape: {probs.shape}', ha='center')
            axes[0, 0].axis('off')
            axes[0, 1].axis('off')
    else:
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
    
    # 3. 熵图 (不确定性)
    if 'entropy' in outputs:
        entropy = outputs['entropy']
        if entropy.dim() == 3:  # (N, H, W) - 取平均或最大
            entropy_avg = entropy.mean(dim=0)  # 平均熵
            im2 = axes[0, 2].imshow(entropy_avg.numpy(), cmap='hot')
            axes[0, 2].set_title(f'Avg Entropy ({entropy.shape[0]} masks)')
        elif entropy.dim() == 2:  # (H, W)
            im2 = axes[0, 2].imshow(entropy.numpy(), cmap='hot')
            axes[0, 2].set_title('Entropy (Uncertainty)')
        else:
            axes[0, 2].text(0.5, 0.5, f'Entropy shape: {entropy.shape}', ha='center')
            im2 = None
        
        axes[0, 2].axis('off')
        if im2 is not None:
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    else:
        axes[0, 2].axis('off')
    
    # 4. 能量图
    if 'energy' in outputs:
        energy = outputs['energy']
        if isinstance(energy, dict):
            qualities = energy.get('mask_qualities')
            calibrated = energy.get('calibrated_conf')
            if qualities is not None:
                axes[1, 0].plot(qualities.numpy(), label='quality')
            if calibrated is not None:
                axes[1, 0].plot(calibrated.numpy(), label='calibrated')
            axes[1, 0].set_title('Mask Qualities')
            axes[1, 0].legend()
        else:
            energy = torch.tensor(energy)
            im3 = axes[1, 0].imshow(energy.numpy(), cmap='viridis')
            axes[1, 0].set_title('Energy')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].axis('off')
    
    # 5. 掩码数量分布
    if 'masks' in outputs:
        masks = outputs['masks']
        if masks.dim() == 3:
            mask_sum = masks.sum(dim=0)
            im4 = axes[1, 1].imshow(mask_sum.numpy(), cmap='plasma')
            axes[1, 1].set_title(f'Refined Masks Sum (N={masks.shape[0]})')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        else:
            axes[1, 1].text(0.5, 0.5, f'Masks shape: {masks.shape}', ha='center')
            axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')

    if 'prior' in outputs:
        prior = outputs['prior']
        if prior.dim() >= 2:
            prior_map = prior.squeeze()
            im_prior = axes[0, 2].imshow(prior_map.numpy(), cmap='inferno')
            axes[0, 2].set_title('Prior Map')
            axes[0, 2].axis('off')
            plt.colorbar(im_prior, ax=axes[0, 2], fraction=0.046)
    
    # 6. 温度/熵统计直方图
    if 'probs' in outputs:
        probs = outputs['probs']
        if probs.dim() >= 3:
            # 计算softmax分布统计
            if probs.dim() == 4:
                max_probs = probs.max(dim=1).values  # (B, H, W)
            elif probs.dim() == 3:
                max_probs = probs.max(dim=0).values  # (H, W)
            else:
                max_probs = probs
            max_probs_flat = max_probs.flatten().numpy()
            
            # 双子图：概率分布 + 熵分布
            axes[1, 2].hist(max_probs_flat, bins=50, color='blue', alpha=0.6, label='Max Prob')
            axes[1, 2].set_xlabel('Probability / Entropy')
            axes[1, 2].set_ylabel('Pixel Count')
            
            if 'entropy' in outputs:
                entropy = outputs['entropy']
                if entropy.dim() >= 2:
                    entropy_flat = entropy.flatten().numpy()
                    # 第二个y轴用于熵
                    ax2 = axes[1, 2].twinx()
                    ax2.hist(entropy_flat, bins=50, color='orange', alpha=0.6, label='Entropy')
                    ax2.set_ylabel('Entropy Count')
                    
            axes[1, 2].set_title('Softmax & Entropy Stats')
            axes[1, 2].grid(alpha=0.3)
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, 'No softmax data', ha='center')
            axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存可视化
    save_path = save_dir / f"{image_stem}_uam_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Visualization saved to: {save_path}")
    
    # 显示
    plt.show()
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 UAM 调试产物")
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./results_analysis/uam_debug",
        help="UAM 调试文件目录"
    )
    parser.add_argument(
        "--image_stem",
        type=str,
        default="target",
        help="图像文件名前缀"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="可视化结果保存目录（默认同 debug_dir）"
    )
    
    args = parser.parse_args()
    visualize_uam_outputs(args.debug_dir, args.image_stem, args.save_dir)

