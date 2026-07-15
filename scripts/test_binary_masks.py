#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试二值掩码输出功能

用法:
    python scripts/test_binary_masks.py results_analysis/ship/binary_masks/6
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image


def analyze_binary_masks(mask_dir: str):
    """分析二值掩码目录"""
    mask_dir = Path(mask_dir)
    
    if not mask_dir.exists():
        print(f"❌ 目录不存在: {mask_dir}")
        print(f"\n请先运行推理生成结果：")
        print(f"  cd zods-rs")
        print(f"  python run_lightening.py test \\")
        print(f"    --config zods_rs/pl_configs/ship_dinov3.yaml \\")
        print(f"    --model.test_mode=test \\")
        print(f"    --ckpt_path ./tmp_ckpts/ship/ship_refs_memory_postprocessed.pth")
        return
    
    masks = sorted(mask_dir.glob('*.png'))
    
    if not masks:
        print(f"❌ 目录中没有掩码文件: {mask_dir}")
        return
    
    print("=" * 60)
    print(f"📊 二值掩码分析")
    print("=" * 60)
    print(f"目录: {mask_dir}")
    print(f"掩码数量: {len(masks)}")
    print()
    
    total_area = 0
    
    for idx, mask_path in enumerate(masks):
        # 读取掩码
        mask = np.array(Image.open(mask_path))
        
        # 解析文件名
        # 格式: {stem}_{label}_{score}_mask_{idx}.png
        parts = mask_path.stem.split('_')
        if len(parts) >= 4:
            stem = parts[0]
            label = parts[1]
            score = int(parts[2]) / 100.0
            mask_idx = parts[-1]
        else:
            stem = "unknown"
            label = "unknown"
            score = 0.0
            mask_idx = str(idx)
        
        # 计算统计信息
        area = np.sum(mask > 0)
        total_area += area
        
        height, width = mask.shape
        ratio = area / (height * width)
        
        # 边界框
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox_w = x2 - x1 + 1
            bbox_h = y2 - y1 + 1
            bbox_area = bbox_w * bbox_h
            fill_ratio = area / bbox_area
        else:
            bbox_w = bbox_h = 0
            fill_ratio = 0.0
        
        # 打印信息
        print(f"[{idx}] {mask_path.name}")
        print(f"    类别: {label} | 置信度: {score:.2f}")
        print(f"    图像尺寸: {width} × {height}")
        print(f"    掩码面积: {area:,} 像素 ({ratio:.2%})")
        print(f"    边界框: {bbox_w} × {bbox_h} | 填充率: {fill_ratio:.2%}")
        
        # 检查掩码值
        unique_vals = np.unique(mask)
        if not np.array_equal(unique_vals, [0, 255]) and not np.array_equal(unique_vals, [0]):
            print(f"    ⚠️  警告: 掩码包含非二值值: {unique_vals}")
        else:
            print(f"    ✓ 二值掩码正确（值: {unique_vals.tolist()}）")
        
        print()
    
    # 总结
    print("=" * 60)
    print(f"总计: {len(masks)} 个掩码，总面积: {total_area:,} 像素")
    print("=" * 60)
    
    # 生成合并图
    if masks:
        print(f"\n📦 生成语义分割合并图...")
        
        # 读取第一个掩码获取尺寸
        base = Image.open(masks[0])
        semantic_map = np.zeros(base.size[::-1], dtype=np.uint8)
        
        # 叠加所有掩码
        for idx, mask_path in enumerate(masks, start=1):
            mask = np.array(Image.open(mask_path))
            semantic_map[mask > 0] = min(idx * 50, 255)  # 不同灰度值
        
        # 保存
        output_path = mask_dir.parent / f"{mask_dir.name}_merged.png"
        Image.fromarray(semantic_map).save(output_path)
        
        print(f"✓ 保存合并图: {output_path}")
        print(f"  （不同实例显示为不同灰度值）")


def compare_outputs(result_dir: str, img_stem: str):
    """对比不同输出类型"""
    result_dir = Path(result_dir)
    
    print("\n" + "=" * 60)
    print(f"📁 输出文件对比 - {img_stem}")
    print("=" * 60)
    
    outputs = {
        "可视化图": result_dir / "predictions" / f"{img_stem}_prediction.png",
        "JSON": result_dir / "json" / f"{img_stem}_prediction.json",
        "实例图像": result_dir / "instances" / img_stem,
        "二值掩码": result_dir / "binary_masks" / img_stem,
    }
    
    for name, path in outputs.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"✓ {name}: {path} ({size:,} 字节)")
            else:
                files = list(path.glob('*.png'))
                print(f"✓ {name}: {path} ({len(files)} 个文件)")
        else:
            print(f"✗ {name}: {path} (不存在)")
    
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mask_dir = sys.argv[1]
        analyze_binary_masks(mask_dir)
        
        # 尝试对比输出
        mask_path = Path(mask_dir)
        if mask_path.exists():
            result_dir = mask_path.parent.parent
            img_stem = mask_path.name
            compare_outputs(result_dir, img_stem)
    else:
        print("用法: python scripts/test_binary_masks.py <掩码目录>")
        print()
        print("示例:")
        print("  python scripts/test_binary_masks.py results_analysis/ship/binary_masks/6")
        print()
        print("或者分析所有图像:")
        print("  for dir in results_analysis/ship/binary_masks/*; do")
        print("      python scripts/test_binary_masks.py \"$dir\"")
        print("  done")

