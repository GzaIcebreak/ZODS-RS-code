"""可视化 R-SEM (旋转等变语义匹配) 调试产物。

用法:
    python scripts/visualize_rsem_debug.py --debug_dir results_analysis/sem_debug --image_stem target
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Fix for UnicodeEncodeError on Windows
sys.stdout.reconfigure(encoding='utf8')


def visualize_rsem_outputs(debug_dir, image_stem="target", save_dir=None):
    """可视化 R-SEM 调试输出。
    
    参数:
        debug_dir: 调试文件所在目录
        image_stem: 图像文件名前缀
        save_dir: 可视化结果保存目录
    """
    debug_dir = Path(debug_dir)
    if save_dir is None:
        save_dir = debug_dir
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载文件（R-SEM specific）
    alpha_file = debug_dir / f"{image_stem}_scale_weights.pt"  # α权重
    beta_file = debug_dir / f"{image_stem}_angle_weights.pt"   # β权重
    heatmap_file = debug_dir / f"{image_stem}_R_heatmap.pt"    # 融合相似度图 R
    matches_file = debug_dir / f"{image_stem}_matches.pt"
    
    print(f"Loading R-SEM debug files from: {debug_dir}")
    
    outputs = {}
    if alpha_file.exists():
        alpha = torch.load(alpha_file, map_location='cpu', weights_only=False)
        outputs['alpha'] = alpha
        print(f"  [OK] Scale weights (α): {alpha.shape if hasattr(alpha, 'shape') else type(alpha)}")
    
    if beta_file.exists():
        beta = torch.load(beta_file, map_location='cpu', weights_only=False)
        outputs['beta'] = beta
        print(f"  [OK] Angle weights (β): {beta.shape if hasattr(beta, 'shape') else type(beta)}")
    
    if heatmap_file.exists():
        R = torch.load(heatmap_file, map_location='cpu', weights_only=False)
        outputs['R'] = R
        print(f"  [OK] Fused heatmap (R): {R.shape if hasattr(R, 'shape') else type(R)}")
    
    if matches_file.exists():
        try:
            matches = torch.load(matches_file, map_location='cpu', weights_only=False)
            outputs['matches'] = matches
            print(f"  [OK] Matches: {type(matches)}")
        except Exception as e:
            print(f"  [WARN] Failed to load matches: {e}")
    
    if not outputs:
        print("[ERROR] No R-SEM debug files found!")
        return
    
    # 可视化 - 2x2 布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'R-SEM Debug Visualization - {image_stem}', fontsize=16)
    
    # 1. α权重（尺度）
    if 'alpha' in outputs:
        alpha = outputs['alpha']
        if isinstance(alpha, torch.Tensor):
            a_np = alpha.cpu().numpy()
            scales_labels = [f"S{i}" for i in range(len(a_np))]
            axes[0, 0].bar(scales_labels, a_np, color='skyblue', alpha=0.8, edgecolor='navy')
            axes[0, 0].set_title('α Weights (Scale Fusion)', fontweight='bold')
            axes[0, 0].set_ylabel('Weight')
            axes[0, 0].set_ylim([0, max(a_np) * 1.2])
            axes[0, 0].grid(axis='y', alpha=0.3)
            # 标注峰值
            peak_idx = a_np.argmax()
            axes[0, 0].axvline(peak_idx, color='red', linestyle='--', alpha=0.6, label=f'Peak @ S{peak_idx}')
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, f'Alpha type: {type(alpha)}', ha='center', va='center')
    else:
        axes[0, 0].text(0.5, 0.5, 'No α data', ha='center', va='center')
    axes[0, 0].set_facecolor('#f9f9f9')
    
    # 2. β权重（角度）
    if 'beta' in outputs:
        beta = outputs['beta']
        if isinstance(beta, torch.Tensor):
            b_np = beta.cpu().numpy()
            # 假设角度为 [-30, -15, 0, 15, 30]
            num_angles = len(b_np)
            angle_labels = [f"θ{i}" for i in range(num_angles)]
            axes[0, 1].bar(angle_labels, b_np, color='coral', alpha=0.8, edgecolor='darkred')
            axes[0, 1].set_title('β Weights (Angle Fusion)', fontweight='bold')
            axes[0, 1].set_ylabel('Weight')
            axes[0, 1].set_ylim([0, max(b_np) * 1.2])
            axes[0, 1].grid(axis='y', alpha=0.3)
            # 标注峰值
            peak_idx = b_np.argmax()
            axes[0, 1].axvline(peak_idx, color='red', linestyle='--', alpha=0.6, label=f'Peak @ θ{peak_idx}')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, f'Beta type: {type(beta)}', ha='center', va='center')
    else:
        axes[0, 1].text(0.5, 0.5, 'No β data', ha='center', va='center')
    axes[0, 1].set_facecolor('#f9f9f9')
    
    # 3. 融合相似度热力图 R
    if 'R' in outputs:
        R = outputs['R']
        if isinstance(R, torch.Tensor) and R.dim() >= 2:
            R_np = R.squeeze().cpu().numpy()
            im = axes[1, 0].imshow(R_np, cmap='jet', interpolation='bilinear')
            axes[1, 0].set_title('Fused Similarity Map (R)', fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        else:
            axes[1, 0].text(0.5, 0.5, f'R shape: {R.shape if hasattr(R, "shape") else "N/A"}', ha='center')
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'No R heatmap', ha='center', va='center')
        axes[1, 0].axis('off')
    
    # 4. R的分布直方图
    if 'R' in outputs:
        R = outputs['R']
        if isinstance(R, torch.Tensor) and R.dim() >= 2:
            R_flat = R.flatten().cpu().numpy()
            axes[1, 1].hist(R_flat, bins=50, color='green', alpha=0.7, edgecolor='darkgreen')
            axes[1, 1].set_title('R Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Similarity Score')
            axes[1, 1].set_ylabel('Pixel Count')
            axes[1, 1].grid(alpha=0.3)
            mean_val = R_flat.mean()
            max_val = R_flat.max()
            axes[1, 1].axvline(mean_val, color='blue', linestyle='--', label=f'Mean={mean_val:.3f}')
            axes[1, 1].axvline(max_val, color='red', linestyle='--', label=f'Max={max_val:.3f}')
            axes[1, 1].legend()
        else:
            axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存可视化
    save_path = save_dir / f"{image_stem}_rsem_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] R-SEM visualization saved to: {save_path}")
    
    plt.show()
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 R-SEM 调试产物")
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./results_analysis/sem_debug",
        help="R-SEM 调试文件目录"
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
        help="可视化结果保存目录"
    )
    
    args = parser.parse_args()
    visualize_rsem_outputs(args.debug_dir, args.image_stem, args.save_dir)


