#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 JSON 预测结果重新生成可视化输出

用法:
    python scripts/regenerate_from_json.py <json_path> [--img_path <图像路径>] [--output_dir <输出目录>]

示例:
    python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json
    python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json --img_path data/FAR1M/Ship/images/000000.jpg
    python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json --output_dir ./regenerated_output
"""

import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import warnings

try:
    import cv2
except ImportError:
    cv2 = None
    warnings.warn("cv2 not available, some visualizations may be limited")

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None
    raise ImportError("pycocotools is required. Install with: pip install pycocotools")


def decode_rle_segmentation(segmentation: Dict[str, Any]) -> np.ndarray:
    """解码 RLE 格式的 segmentation 为二值掩码"""
    if mask_utils is None:
        raise RuntimeError("pycocotools not available")
    
    # 确保 counts 是字符串
    if isinstance(segmentation['counts'], str):
        rle = segmentation.copy()
    else:
        rle = segmentation.copy()
        rle['counts'] = segmentation['counts'].decode('utf-8') if isinstance(segmentation['counts'], bytes) else str(segmentation['counts'])
    
    # 解码为二值掩码
    mask = mask_utils.decode(rle)
    return mask.astype(np.uint8)


def bbox_xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """将 [x, y, w, h] 转换为 [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """加载 JSON 预测结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"JSON 文件应包含一个列表，但得到: {type(data)}")
    
    return data


def extract_class_names_from_instances(instances: List[Dict[str, Any]]) -> List[str]:
    """从实例列表中提取类别名称
    
    参数:
        instances: 实例列表
    
    返回:
        类别名称列表（去重后）
    """
    class_names_set = set()
    
    for inst in instances:
        category_name = inst.get('category_name', '')
        if category_name:
            # 从 "airplane=33" 中提取 "airplane"
            if '=' in category_name:
                cat_name = category_name.split('=', 1)[0].strip()
            else:
                cat_name = category_name.strip()
            
            if cat_name:
                class_names_set.add(cat_name)
    
    # 如果没有找到类别名称，返回默认值
    if not class_names_set:
        return ['unknown']
    
    # 返回排序后的列表
    return sorted(list(class_names_set))


def find_image_path(stem: str, search_dirs: List[Path]) -> Optional[Path]:
    """根据文件名查找图像路径"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for ext in image_extensions:
            img_path = search_dir / f"{stem}{ext}"
            if img_path.exists():
                return img_path
            img_path = search_dir / f"{stem}.{ext.lstrip('.')}"
            if img_path.exists():
                return img_path
    
    return None


def save_binary_masks(
    instances: List[Dict[str, Any]],
    output_dir: Path,
    stem: str,
    score_thr: float = 0.0
):
    """保存纯二值掩码
    
    参数:
        instances: 实例列表
        output_dir: 输出目录
        stem: 文件名前缀
        score_thr: 置信度阈值（默认0.0，保存所有实例）
    """
    import shutil
    
    # 删除旧的目录和文件
    if output_dir.exists():
        old_files = list(output_dir.glob("*.png"))
        if old_files:
            print(f"   🗑️  清理旧二值掩码: {len(old_files)} 个文件")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for idx, ins in enumerate(instances):
        # 应用置信度阈值过滤
        if ins.get('score', 0.0) < score_thr:
            continue
        # 解码掩码
        mask = decode_rle_segmentation(ins['segmentation'])
        
        # 转换为二值掩码（0 或 255）
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        # 保存为灰度图
        mask_img = Image.fromarray(mask_binary, mode='L')
        
        # 解析类别名称和置信度
        category_name = ins.get('category_name', 'unknown')
        if '=' in category_name:
            cat_name, score_str = category_name.split('=', 1)
            score_int = int(float(score_str))
        else:
            cat_name = category_name
            score_int = int(ins.get('score', 0) * 100)
        
        filename = f"{stem}_{cat_name}_{score_int:03d}_mask_{saved_count:03d}.png"
        mask_path = output_dir / filename
        
        mask_img.save(mask_path)
        saved_count += 1
    
    print(f"✓ 保存了 {saved_count} 个二值掩码到: {output_dir}")
    filtered_count = sum(1 for ins in instances if ins.get('score', 0.0) < score_thr)
    if filtered_count > 0:
        print(f"  (过滤了 {filtered_count} 个低置信度实例, threshold={score_thr})")
    return saved_count


def save_instance_images(
    instances: List[Dict[str, Any]],
    base_img: Image.Image,
    output_dir: Path,
    stem: str,
    class_names: Optional[List[str]] = None,
    score_thr: float = 0.0
):
    """保存单实例可视化图像
    
    参数:
        instances: 实例列表
        base_img: 基础图像
        output_dir: 输出目录
        stem: 文件名前缀
        class_names: 类别名称列表
        score_thr: 置信度阈值（默认0.0，保存所有实例）
    """
    import shutil
    
    # 删除旧的目录和文件
    if output_dir.exists():
        old_files = list(output_dir.glob("*.png"))
        if old_files:
            print(f"   🗑️  清理旧实例图像: {len(old_files)} 个文件")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 实例颜色列表
    INSTANCE_COLORS = [
        np.array([255, 0, 0]),
        np.array([0, 255, 0]),
        np.array([0, 0, 255]),
        np.array([255, 255, 0]),
        np.array([255, 0, 255]),
        np.array([0, 255, 255]),
        np.array([128, 0, 0]),
        np.array([0, 128, 0]),
        np.array([0, 0, 128]),
        np.array([128, 128, 0]),
    ]
    
    saved_count = 0
    filtered_count = 0
    for idx, ins in enumerate(instances):
        # 应用置信度阈值过滤
        if ins.get('score', 0.0) < score_thr:
            filtered_count += 1
            continue
        # 创建图像副本
        img_copy = base_img.copy()
        img_np = np.array(img_copy)
        
        # 解码掩码
        mask = decode_rle_segmentation(ins['segmentation'])
        mask_bool = mask > 0
        
        # 选择颜色
        color = INSTANCE_COLORS[idx % len(INSTANCE_COLORS)].tolist()
        
        # 绘制轮廓
        if cv2 is not None:
            seg_thickness = max(2, int(img_np.shape[1] * 0.003))
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_np, contours, -1, color, seg_thickness)
        
        # 绘制边界框
        bbox_xyxy = bbox_xywh_to_xyxy(ins['bbox'])
        
        # 解析类别名称和置信度
        category_name = ins.get('category_name', 'unknown')
        if '=' in category_name:
            cat_name, score_str = category_name.split('=', 1)
            score_int = int(float(score_str))
        else:
            cat_name = category_name
            score_int = int(ins.get('score', 0) * 100)
        
        label_str = cat_name
        text = f"{label_str}={score_int}"
        
        # 绘制边界框和标签
        if cv2 is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            
            # 绘制文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, img_np.shape[1] / 1000)
            thickness = max(1, int(font_scale * 2))
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img_np, (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(img_np, text, (x1, y1 - baseline - 2), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # 保存
        overlay_img = Image.fromarray(img_np)
        filename = f"{stem}_mask_{saved_count:03d}.png"
        overlay_path = output_dir / filename
        
        overlay_img.save(overlay_path)
        saved_count += 1
    
    print(f"✓ 保存了 {saved_count} 个实例图像到: {output_dir}")
    if filtered_count > 0:
        print(f"  (过滤了 {filtered_count} 个低置信度实例, threshold={score_thr})")
    return saved_count


def save_prediction_image(
    instances: List[Dict[str, Any]],
    base_img: Image.Image,
    output_path: Path,
    img_path: Path,
    class_names: Optional[List[str]] = None,
    score_thr: float = 0.0,
    dataset_name: Optional[str] = None
):
    """保存所有实例叠加的可视化图像
    
    参数:
        instances: 实例列表
        base_img: 基础图像
        output_path: 输出路径
        img_path: 图像路径
        class_names: 类别名称列表
        score_thr: 置信度阈值（默认0.0，显示所有实例）
        dataset_name: 数据集名称（可选，用于兼容性）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 先过滤掉低于阈值的实例
    filtered_instances = [ins for ins in instances if ins.get('score', 0.0) >= score_thr]
    
    if len(filtered_instances) == 0:
        print(f"   ⚠️  警告: 没有实例满足置信度阈值 {score_thr}")
        # 创建一个空的可视化图像
        base_img.save(output_path)
        return
    
    # 如果没有提供类别名称，尝试从实例中提取
    if class_names is None or len(class_names) == 0:
        class_names = extract_class_names_from_instances(filtered_instances)
    
    # 如果没有数据集名称，尝试从类别名称推断
    if dataset_name is None and class_names:
        dataset_name = class_names[0].lower()
    
    try:
        from zods_rs.dataset.visualization import vis_coco
    except ImportError:
        warnings.warn("vis_coco not available, using simple visualization")
        vis_coco = None
    
    if vis_coco is not None:
        # 使用 vis_coco 生成可视化
        scores_np = np.array([ins.get('score', 0.0) for ins in filtered_instances])
        labels_np = np.array([ins.get('category_id', 0) for ins in filtered_instances])
        bboxes_np = np.array([bbox_xywh_to_xyxy(ins['bbox']) for ins in filtered_instances])
        masks_np = np.array([decode_rle_segmentation(ins['segmentation']) for ins in filtered_instances])
        
        # 准备 GT 数据（空）
        gt_bboxes_np = np.array([])
        gt_labels_np = np.array([])
        gt_masks_np = np.array([])
        
        # 调用 vis_coco (参数顺序: gt_bboxes, gt_labels, gt_masks, scores, labels, bboxes, masks, score_thr, img_path, out_path, ...)
        vis_coco(
            gt_bboxes_np,
            gt_labels_np,
            gt_masks_np,
            scores_np,
            labels_np,
            bboxes_np,
            masks_np,
            score_thr=score_thr,  # 使用传入的阈值
            img_path=str(img_path),
            out_path=str(output_path),
            show_scores=True,
            class_names=class_names,
            dataset_name=dataset_name or 'unknown'
        )
    else:
        # 简单可视化：叠加所有掩码
        img_np = np.array(base_img.copy())
        
        for idx, ins in enumerate(filtered_instances):
            mask = decode_rle_segmentation(ins['segmentation'])
            
            # 创建半透明叠加
            color = np.random.randint(0, 255, 3)
            overlay = np.zeros_like(img_np)
            overlay[mask > 0] = color
            
            # 混合
            alpha = 0.3
            img_np = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
            
            # 绘制边界框
            if cv2 is not None:
                bbox_xyxy = bbox_xywh_to_xyxy(ins['bbox'])
                x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                cv2.rectangle(img_np, (x1, y1), (x2, y2), color.tolist(), 2)
        
        vis_img = Image.fromarray(img_np)
        vis_img.save(output_path)
    
    print(f"✓ 保存了预测可视化图像到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="根据 JSON 预测结果重新生成可视化输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认路径
  python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json
  
  # 指定图像路径
  python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json --img_path data/FAR1M/Ship/images/000000.jpg
  
  # 指定输出目录
  python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json --output_dir ./regenerated_output
        """
    )
    
    parser.add_argument('json_path', help='JSON 预测结果文件路径')
    parser.add_argument('--img_path', help='原始图像路径（如果不提供会自动查找）')
    parser.add_argument('--output_dir', help='输出目录（默认：results_analysis/ship）')
    parser.add_argument('--class_names', nargs='+', default=None, help='类别名称列表（如果不提供，将从 JSON 中自动提取）')
    parser.add_argument('--score_thr', type=float, default=0.0, help='置信度阈值（默认0.0，保存所有实例。建议0.3-0.8）')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"❌ JSON 文件不存在: {json_path}")
        sys.exit(1)
    
    # 加载 JSON 数据
    print(f"📖 加载 JSON 文件: {json_path}")
    instances = load_json_data(str(json_path))
    print(f"✓ 找到 {len(instances)} 个实例")
    
    # 自动提取类别名称（如果未提供）
    if args.class_names is None:
        class_names = extract_class_names_from_instances(instances)
        print(f"✓ 自动提取类别名称: {class_names}")
    else:
        class_names = args.class_names
        print(f"✓ 使用指定的类别名称: {class_names}")
    
    # 确定图像路径
    stem = json_path.stem.replace('_prediction', '')
    
    if args.img_path:
        img_path = Path(args.img_path)
        if not img_path.exists():
            print(f"❌ 图像文件不存在: {img_path}")
            sys.exit(1)
    else:
        # 自动查找图像
        # 尝试从 JSON 路径推断数据集名称（例如：results_analysis/tomb/json -> tomb）
        dataset_name = None
        json_parts = json_path.parts
        if 'results_analysis' in json_parts:
            idx = json_parts.index('results_analysis')
            if idx + 1 < len(json_parts):
                dataset_name = json_parts[idx + 1]
        
        # 构建搜索目录列表
        search_dirs = []
        
        # 1. 如果推断出数据集名称，尝试常见的数据路径
        if dataset_name:
            # 尝试通用数据集路径
            search_dirs.append(json_path.parent.parent.parent / 'data' / dataset_name / 'images')
            # 尝试 FAR1M 数据集路径
            search_dirs.append(json_path.parent.parent.parent / 'data' / 'FAR1M' / dataset_name.capitalize() / 'images')
            # 尝试 Ship 数据集路径（兼容旧路径）
            if dataset_name.lower() == 'ship':
                search_dirs.append(json_path.parent.parent.parent / 'data' / 'FAR1M' / 'Ship' / 'images')
        
        # 2. 通用搜索路径
        search_dirs.extend([
            json_path.parent.parent / '..' / '..' / 'data' / 'FAR1M' / 'Ship' / 'images',
            json_path.parent.parent.parent / 'images',
            json_path.parent.parent.parent,
        ])
        
        # 3. 如果提供了数据集名称，也尝试直接搜索
        if dataset_name:
            # 尝试在项目根目录下查找
            project_root = json_path.parent.parent.parent
            search_dirs.append(project_root / 'data' / dataset_name / 'images')
            search_dirs.append(project_root / 'data' / dataset_name.capitalize() / 'images')
        
        search_dirs = [Path(str(d)).resolve() for d in search_dirs]
        
        # 去重并过滤不存在的目录
        seen = set()
        unique_search_dirs = []
        for d in search_dirs:
            if d not in seen:
                seen.add(d)
                unique_search_dirs.append(d)
        search_dirs = unique_search_dirs
        
        img_path = find_image_path(stem, search_dirs)
        
        if img_path is None:
            print(f"⚠️  未找到图像文件（stem: {stem}）")
            print(f"   搜索目录: {search_dirs}")
            print(f"   请使用 --img_path 指定图像路径")
            sys.exit(1)
    
    print(f"📷 使用图像: {img_path}")
    
    # 加载图像
    base_img = Image.open(img_path).convert('RGB')
    
    # 确定输出目录
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = json_path.parent.parent
    
    print(f"📁 输出目录: {output_root}")
    print()
    
    # 生成输出
    print("=" * 60)
    print("🔄 开始重新生成可视化输出")
    print("=" * 60)
    print()
    
    # 统计信息
    total_instances = len(instances)
    filtered_instances = sum(1 for ins in instances if ins.get('score', 0.0) < args.score_thr)
    valid_instances = total_instances - filtered_instances
    
    print(f"📊 实例统计:")
    print(f"   总计: {total_instances}")
    print(f"   置信度阈值: {args.score_thr}")
    print(f"   将保存: {valid_instances} 个实例")
    if filtered_instances > 0:
        print(f"   将过滤: {filtered_instances} 个低置信度实例")
    print()
    
    # 1. 保存二值掩码
    print("1️⃣  生成二值掩码...")
    binary_masks_dir = output_root / "binary_masks" / stem
    save_binary_masks(instances, binary_masks_dir, stem, score_thr=args.score_thr)
    print()
    
    # 2. 保存实例图像
    print("2️⃣  生成实例图像...")
    instances_dir = output_root / "instances" / stem
    save_instance_images(instances, base_img, instances_dir, stem, class_names, score_thr=args.score_thr)
    print()
    
    # 3. 保存预测可视化
    print("3️⃣  生成预测可视化...")
    predictions_dir = output_root / "predictions"
    prediction_path = predictions_dir / f"{stem}_prediction.png"
    # 从 JSON 路径推断数据集名称（用于兼容性）
    dataset_name = None
    json_parts = json_path.parts
    if 'results_analysis' in json_parts:
        idx = json_parts.index('results_analysis')
        if idx + 1 < len(json_parts):
            dataset_name = json_parts[idx + 1]
    
    save_prediction_image(instances, base_img, prediction_path, img_path, class_names, score_thr=args.score_thr, dataset_name=dataset_name)
    
    # 显示过滤统计
    filtered_count = sum(1 for ins in instances if ins.get('score', 0.0) < args.score_thr)
    if filtered_count > 0:
        print(f"  (过滤了 {filtered_count} 个低置信度实例, threshold={args.score_thr})")
    print()
    
    print("=" * 60)
    print("✅ 重新生成完成！")
    print("=" * 60)
    print(f"✓ 二值掩码: {binary_masks_dir}")
    print(f"✓ 实例图像: {instances_dir}")
    print(f"✓ 预测可视化: {prediction_path}")


if __name__ == "__main__":
    main()

