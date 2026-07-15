"""可视化 SEM (语义尺度匹配) 调试产物。"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Fix for UnicodeEncodeError on Windows
sys.stdout.reconfigure(encoding='utf8')

# Ensure repo root is importable (so that `modules.*` can be resolved during unpickling)
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
except Exception:
    pass


def visualize_sem_outputs(debug_dir, image_stem="target", save_dir=None):
    """可视化 SEM 调试输出。
    
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
    
    # 加载文件
    weights_file = debug_dir / f"{image_stem}_scale_weights.pt"
    heatmap_file = debug_dir / f"{image_stem}_sim_heatmap.pt"
    matches_file = debug_dir / f"{image_stem}_matches.pt"
    alpha_beta_file = debug_dir / f"{image_stem}_alpha_beta.pt"
    attn_file = debug_dir / f"{image_stem}_attn.pt"
    
    print(f"Loading SEM debug files from: {debug_dir}")
    
    outputs = {}
    if weights_file.exists():
        weights = torch.load(weights_file, map_location='cpu')
        outputs['weights'] = weights
        print(f"  [OK] Scale weights: {weights.shape if hasattr(weights, 'shape') else type(weights)}")
    
    if heatmap_file.exists():
        heatmap = torch.load(heatmap_file, map_location='cpu')
        outputs['heatmap'] = heatmap
        print(f"  [OK] Similarity heatmap: {heatmap.shape}")
    
    if matches_file.exists():
        # PyTorch 2.6 默认 weights_only=True；带自定义类型需允许白名单或关闭 weights_only。
        try:
            matches = torch.load(matches_file, map_location='cpu')
        except Exception:
            matches = None
            # Try safe_globals allowlist
            try:
                from modules.sem_scale_match import SEMMatchResult  # type: ignore
                try:
                    from torch.serialization import add_safe_globals, safe_globals  # type: ignore
                except Exception:
                    add_safe_globals = None  # type: ignore
                    safe_globals = None  # type: ignore
                try:
                    if add_safe_globals is not None:
                        add_safe_globals([SEMMatchResult])
                    if safe_globals is not None:
                        with safe_globals([SEMMatchResult]):  # type: ignore
                            matches = torch.load(matches_file, map_location='cpu')
                    else:
                        # Fallback: try again after add_safe_globals
                        matches = torch.load(matches_file, map_location='cpu')
                except Exception:
                    pass
            except Exception:
                # If import fails due to path, we already injected REPO_ROOT above.
                pass
            if matches is None:
                # 最后手段：显式关闭 weights_only（仅在你信任产物来源时使用）
                try:
                    matches = torch.load(matches_file, map_location='cpu', weights_only=False)  # type: ignore
                except TypeError:
                    # 兼容旧版 torch 无 weights_only 参数
                    matches = torch.load(matches_file, map_location='cpu')
        outputs['matches'] = matches
        print(f"  [OK] Match results: {type(matches)}")
    
    if alpha_beta_file.exists():
        alpha_beta = torch.load(alpha_beta_file, map_location='cpu')
        outputs['alpha_beta'] = alpha_beta
        print(f"  [OK] Alpha/Beta weights: {type(alpha_beta)}")
    
    if attn_file.exists():
        attn = torch.load(attn_file, map_location='cpu')
        outputs['attn'] = attn
        print(f"  [OK] Attention map: {attn.shape if hasattr(attn, 'shape') else type(attn)}")
    
    if not outputs:
        print("[ERROR] No SEM debug files found!")
        return
    
    # 可视化 - 扩展到 3x3 布局
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f'SEM Debug Visualization (Multi-Layer) - {image_stem}', fontsize=16)
    
    # 1. 尺度权重柱状图
    if 'weights' in outputs:
        weights = outputs['weights']
        if isinstance(weights, torch.Tensor):
            w_np = weights.numpy()
            scales = [f"Scale {i}" for i in range(len(w_np))]
            axes[0, 0].bar(scales, w_np, color='skyblue', alpha=0.8)
            axes[0, 0].set_title('Multi-Scale Fusion Weights')
            axes[0, 0].set_ylabel('Weight')
            axes[0, 0].grid(axis='y', alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, f'Weights type: {type(weights)}', ha='center')
    else:
        axes[0, 0].text(0.5, 0.5, 'No weights data', ha='center')
    axes[0, 0].axis('off') if 'weights' not in outputs else None
    
    # 2. 相似度热力图
    if 'heatmap' in outputs:
        heatmap = outputs['heatmap']
        if heatmap.dim() >= 2:
            hm = heatmap.squeeze().numpy() if heatmap.dim() > 2 else heatmap.numpy()
            im1 = axes[0, 1].imshow(hm, cmap='jet', interpolation='bilinear')
            axes[0, 1].set_title('Similarity Heatmap')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        else:
            axes[0, 1].text(0.5, 0.5, f'Heatmap shape: {heatmap.shape}', ha='center')
            axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'No heatmap data', ha='center')
        axes[0, 1].axis('off')
    
    # 3. 相似度分布直方图
    if 'heatmap' in outputs:
        heatmap = outputs['heatmap']
        if heatmap.dim() >= 2:
            hm_flat = heatmap.flatten().numpy()
            axes[1, 0].hist(hm_flat, bins=50, color='green', alpha=0.7)
            axes[1, 0].set_title('Similarity Distribution')
            axes[1, 0].set_xlabel('Similarity Score')
            axes[1, 0].set_ylabel('Pixel Count')
            axes[1, 0].grid(alpha=0.3)
            axes[1, 0].axvline(hm_flat.mean(), color='r', linestyle='--', label=f'Mean={hm_flat.mean():.3f}')
            axes[1, 0].legend()
        else:
            axes[1, 0].axis('off')
    else:
        axes[1, 0].axis('off')
    
    # 4. α权重（尺度融合）
    if 'alpha_beta' in outputs and 'alpha_weights' in outputs['alpha_beta']:
        alpha_w = outputs['alpha_beta']['alpha_weights']
        if isinstance(alpha_w, torch.Tensor):
            axes[1, 1].bar(range(len(alpha_w)), alpha_w.numpy(), color='coral', alpha=0.8)
            axes[1, 1].set_title('α Weights (Scale Fusion)')
            axes[1, 1].set_xlabel('Scale Index')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No α weights', ha='center')
            axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
    
    # 5. β权重（层融合）
    if 'alpha_beta' in outputs and 'beta_weights' in outputs['alpha_beta']:
        beta_w = outputs['alpha_beta']['beta_weights']
        if isinstance(beta_w, torch.Tensor):
            layers = [f"L{i}" for i in range(len(beta_w))]
            axes[1, 2].bar(layers, beta_w.numpy(), color='teal', alpha=0.8)
            axes[1, 2].set_title('β Weights (Layer Fusion)')
            axes[1, 2].set_ylabel('Weight')
            axes[1, 2].grid(axis='y', alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No β weights', ha='center')
            axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')
    
    # 6. 注意力图
    if 'attn' in outputs:
        attn = outputs['attn']
        if isinstance(attn, dict) and 'map' in attn:
            attn_map = attn['map']
            if attn_map.dim() >= 2:
                im_attn = axes[2, 0].imshow(attn_map.squeeze().numpy(), cmap='viridis')
                axes[2, 0].set_title('Attention Prior')
                axes[2, 0].axis('off')
                plt.colorbar(im_attn, ax=axes[2, 0], fraction=0.046)
            else:
                axes[2, 0].axis('off')
        else:
            axes[2, 0].axis('off')
    else:
        axes[2, 0].axis('off')
    
    # 7. 注意力前后对比
    if 'heatmap' in outputs and 'alpha_beta' in outputs:
        gamma = outputs['alpha_beta'].get('gamma', 0.0)
        if isinstance(gamma, (int, float)) and gamma > 0:
            axes[2, 1].text(0.5, 0.5, f'Attn γ={gamma:.3f}\nContribution: {gamma*100:.1f}%',
                           ha='center', va='center', fontsize=14, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[2, 1].set_title('Attention Contribution')
        else:
            axes[2, 1].text(0.5, 0.5, 'No attention used', ha='center')
        axes[2, 1].axis('off')
    else:
        axes[2, 1].axis('off')
    
    # 8. 匹配统计
    if 'matches' in outputs:
        matches = outputs['matches']
        axes[2, 2].text(0.5, 0.5, f'Match results:\n{type(matches).__name__}', 
                       ha='center', va='center', fontsize=12)
        axes[2, 2].set_title('Match Statistics')
    else:
        axes[2, 2].text(0.5, 0.5, 'No match data', ha='center')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存可视化
    save_path = save_dir / f"{image_stem}_sem_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Visualization saved to: {save_path}")
    
    plt.show()
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 SEM 调试产物")
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./results_analysis/sem_debug",
        help="SEM 调试文件目录"
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
    visualize_sem_outputs(args.debug_dir, args.image_stem, args.save_dir)

