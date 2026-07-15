"""
XML 标注转换为 OBB JSON 工具

支持格式：
1. PASCAL VOC XML (LabelImg) - 轴对齐矩形
2. DOTA XML - 带旋转的 OBB
3. 自定义 XML - 四点标注

输出：COCO + OBB JSON 格式

用法：
    python scripts/xml_to_obb_json.py \
        --xml_dir data/annotations/xml \
        --img_dir data/images \
        --output annotations.json \
        --format voc
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"错误: 缺少依赖 {e}")
    print("请运行: pip install opencv-python pillow numpy")
    exit(1)


def parse_voc_xml(xml_path: Path) -> Dict[str, Any]:
    """解析 PASCAL VOC 格式的 XML 文件。
    
    参数:
        xml_path: XML 文件路径
    
    返回:
        包含图像信息和标注的字典
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 解析图像信息
    filename = root.find('filename').text if root.find('filename') is not None else xml_path.stem + '.jpg'
    size = root.find('size')
    width = int(size.find('width').text) if size.find('width') is not None else 0
    height = int(size.find('height').text) if size.find('height') is not None else 0
    
    # 解析对象标注
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        
        # 检查是否有 OBB 标注（四点格式）
        robndbox = obj.find('robndbox')
        if robndbox is not None:
            # DOTA 格式：四个点
            x1 = float(robndbox.find('x1').text)
            y1 = float(robndbox.find('y1').text)
            x2 = float(robndbox.find('x2').text)
            y2 = float(robndbox.find('y2').text)
            x3 = float(robndbox.find('x3').text)
            y3 = float(robndbox.find('y3').text)
            x4 = float(robndbox.find('x4').text)
            y4 = float(robndbox.find('y4').text)
            obb = [x1, y1, x2, y2, x3, y3, x4, y4]
        else:
            # 普通 bndbox：需要转换为 OBB（无旋转）
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            # 转换为 OBB（4个角点，无旋转）
            obb = [
                xmin, ymin,  # 左上
                xmax, ymin,  # 右上
                xmax, ymax,  # 右下
                xmin, ymax   # 左下
            ]
        
        objects.append({
            'name': name,
            'obb': obb,
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects,
    }


def obb_to_bbox_xywh(obb: List[float]) -> List[float]:
    """从 OBB 计算轴对齐的 bbox (xywh)。"""
    points = np.array(obb).reshape(4, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def create_mask_from_obb(obb: List[float], width: int, height: int) -> np.ndarray:
    """从 OBB 创建二值掩码。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(obb).reshape(4, 2).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def encode_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """将掩码编码为 COCO RLE 格式。"""
    try:
        import pycocotools.mask as mask_utils
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    except ImportError:
        warnings.warn("pycocotools not available, segmentation will be empty")
        return {"size": [mask.shape[0], mask.shape[1]], "counts": ""}


def convert_xml_dir_to_json(
    xml_dir: Path,
    img_dir: Path,
    output_path: Path,
    format_type: str = 'voc',
    category_mapping: Optional[Dict[str, int]] = None,
) -> None:
    """批量转换 XML 标注为 COCO + OBB JSON。
    
    参数:
        xml_dir: XML 文件目录
        img_dir: 对应图像目录
        output_path: 输出 JSON 文件路径
        format_type: XML 格式类型（'voc' 或 'dota'）
        category_mapping: 类别名称 -> id 的映射
    """
    
    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"警告: 在 {xml_dir} 中未找到 XML 文件")
        return
    
    print(f"找到 {len(xml_files)} 个 XML 文件")
    
    # 构建 COCO JSON 结构
    coco_data = {
        "info": {
            "description": "Converted from XML annotations",
            "version": "1.0",
            "year": 2025,
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    
    # 收集所有类别
    all_categories = set()
    parsed_data = []
    
    for xml_path in xml_files:
        try:
            data = parse_voc_xml(xml_path)
            parsed_data.append(data)
            for obj in data['objects']:
                all_categories.add(obj['name'])
        except Exception as e:
            print(f"警告: 解析 {xml_path.name} 失败: {e}")
            continue
    
    # 创建类别映射
    if category_mapping is None:
        category_mapping = {name: idx for idx, name in enumerate(sorted(all_categories))}
    
    coco_data['categories'] = [
        {"id": cat_id, "name": cat_name}
        for cat_name, cat_id in category_mapping.items()
    ]
    
    print(f"类别映射: {category_mapping}")
    
    # 转换每个文件
    image_id = 1
    annotation_id = 1
    
    for data in parsed_data:
        filename = data['filename']
        width = data['width']
        height = data['height']
        
        # 如果 XML 中没有尺寸信息，从图像读取
        if width == 0 or height == 0:
            img_path = img_dir / filename
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                except Exception as e:
                    print(f"警告: 无法读取图像 {filename}: {e}")
                    continue
            else:
                print(f"警告: 图像文件不存在: {filename}")
                continue
        
        # 添加图像信息
        coco_data['images'].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        
        # 处理每个对象
        for obj in data['objects']:
            cat_name = obj['name']
            if cat_name not in category_mapping:
                print(f"警告: 未知类别 {cat_name}，跳过")
                continue
            
            cat_id = category_mapping[cat_name]
            obb = obj['obb']
            bbox_xywh = obb_to_bbox_xywh(obb)
            
            # 从 OBB 创建掩码
            mask = create_mask_from_obb(obb, width, height)
            segmentation = encode_mask_to_rle(mask)
            
            # 添加标注
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox_xywh,
                "area": float(np.sum(mask)),
                "segmentation": segmentation,
                "obb": obb,
                "iscrowd": 0,
            }
            coco_data['annotations'].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    # 保存 JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 转换完成!")
    print(f"  图像数: {len(coco_data['images'])}")
    print(f"  标注数: {len(coco_data['annotations'])}")
    print(f"  类别数: {len(coco_data['categories'])}")
    print(f"  输出文件: {output_path}")


def visualize_obb_json(json_path: Path, img_dir: Path, output_dir: Path) -> None:
    """可视化 OBB JSON 标注。
    
    参数:
        json_path: COCO + OBB JSON 文件
        img_dir: 图像目录
        output_dir: 可视化输出目录
    """
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建图像 ID 到信息的映射
    img_map = {img['id']: img for img in data['images']}
    cat_map = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 按图像组织标注
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每张图像生成可视化
    for img_id, anns in img_anns.items():
        img_info = img_map[img_id]
        img_path = img_dir / img_info['file_name']
        
        if not img_path.exists():
            print(f"警告: 图像不存在 {img_path}")
            continue
        
        # 加载图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 绘制每个标注
        for ann in anns:
            cat_name = cat_map.get(ann['category_id'], 'unknown')
            obb = np.array(ann['obb']).reshape(4, 2).astype(np.int32)
            
            # 随机颜色（基于类别 ID）
            np.random.seed(ann['category_id'])
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # 绘制 OBB
            cv2.polylines(img, [obb], True, color, 2)
            
            # 绘制标签
            center = obb.mean(axis=0).astype(int)
            label = f"{cat_name}"
            cv2.putText(img, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 保存
        out_path = output_dir / img_info['file_name']
        cv2.imwrite(str(out_path), img)
    
    print(f"\n✓ 可视化完成: {len(img_anns)} 张图像保存到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="XML 标注转换为 OBB JSON")
    parser.add_argument('--xml_dir', type=str, required=True, help='XML 文件目录')
    parser.add_argument('--img_dir', type=str, required=True, help='图像文件目录')
    parser.add_argument('--output', type=str, required=True, help='输出 JSON 路径')
    parser.add_argument('--format', type=str, default='voc', choices=['voc', 'dota'], help='XML 格式类型')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图像')
    parser.add_argument('--vis_dir', type=str, default='./xml_vis', help='可视化输出目录')
    parser.add_argument('--categories', type=str, default=None, help='类别映射 JSON，格式: {"cat1": 0, "cat2": 1}')
    
    args = parser.parse_args()
    
    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir)
    output_path = Path(args.output)
    
    # 解析类别映射
    category_mapping = None
    if args.categories:
        try:
            category_mapping = json.loads(args.categories)
        except:
            cat_file = Path(args.categories)
            if cat_file.exists():
                with cat_file.open('r') as f:
                    category_mapping = json.load(f)
    
    # 执行转换
    print(f"开始转换 XML → OBB JSON")
    print(f"  XML 目录: {xml_dir}")
    print(f"  图像目录: {img_dir}")
    print(f"  输出路径: {output_path}")
    print(f"  格式类型: {args.format}")
    
    convert_xml_dir_to_json(xml_dir, img_dir, output_path, args.format, category_mapping)
    
    # 可视化
    if args.visualize:
        print(f"\n生成可视化...")
        vis_dir = Path(args.vis_dir)
        visualize_obb_json(output_path, img_dir, vis_dir)


if __name__ == '__main__':
    main()

