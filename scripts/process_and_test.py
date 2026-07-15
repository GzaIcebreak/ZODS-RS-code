#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动处理图片并执行推理（支持单张和批处理）

用法:
    # 单张图片
    python scripts/process_and_test.py <图片路径>
    
    # 多张图片
    python scripts/process_and_test.py <图片1> <图片2> <图片3> ...
    
    # 目录批处理
    python scripts/process_and_test.py --dir <目录路径>
    
    # 只处理不推理（快速模式）
    python scripts/process_and_test.py --no-inference <图片路径>

示例:
    python scripts/process_and_test.py path\to\data\my_tomb.jpg
    python scripts/process_and_test.py img1.jpg img2.jpg img3.jpg
    python scripts/process_and_test.py --dir path\to\data\images
    python scripts/process_and_test.py --dir ./test_images --no-inference
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from PIL import Image
import subprocess
from typing import Tuple, List, Optional


def get_image_size(image_path: str) -> Tuple[int, int]:
    """获取图片尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        raise ValueError(f"无法读取图片尺寸: {e}")


def copy_image_to_target(src_path: str, target_dir: str, verbose: bool = True) -> str:
    """复制图片到目标目录，返回目标路径"""
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(f"图片不存在: {src_path}")
    
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取文件名（保持原扩展名）
    filename = src_path.name
    target_path = target_dir / filename
    
    # 复制文件
    shutil.copy2(src_path, target_path)
    if verbose:
        print(f"✓ 图片已复制到: {target_path}")
    
    return str(target_path)


def update_custom_targets_json(json_path: str, file_name: str, width: int, height: int, verbose: bool = True):
    """更新 custom_targets.json 文件"""
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
    
    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 更新 images[0]
    if len(data.get("images", [])) == 0:
        # 如果没有 images，创建一个
        data["images"] = [{"id": 1001}]
    
    data["images"][0]["file_name"] = file_name
    data["images"][0]["width"] = width
    data["images"][0]["height"] = height
    
    # 确保 id 存在
    if "id" not in data["images"][0]:
        data["images"][0]["id"] = 1001
    
    # 写回 JSON（保持格式）
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"✓ JSON 已更新: file_name={file_name}, width={width}, height={height}")


def run_inference(work_dir: str = None, verbose: bool = True):
    """执行推理命令"""
    cmd = [
        "python", "-X", "utf8", "run_lightening.py", "test",
        "--config", "zods_rs/pl_configs/ship_dinov3.yaml",
        "--model.test_mode=test",
        "--ckpt_path", "./tmp_ckpts/ship/ship_refs_memory_postprocessed.pth"
    ]
    
    if verbose:
        print("=" * 60)
        print("🚀 开始执行推理...")
        print("=" * 60)
        print(f"命令: {' '.join(cmd)}")
        print()
    
    # 如果指定了工作目录，切换到该目录
    if work_dir:
        original_dir = os.getcwd()
        os.chdir(work_dir)
        if verbose:
            print(f"工作目录: {os.getcwd()}")
            print()
    
    try:
        # 执行命令（批处理模式下不显示详细输出）
        result = subprocess.run(
            cmd, 
            check=False,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None
        )
        
        if verbose:
            if result.returncode == 0:
                print()
                print("=" * 60)
                print("✅ 推理完成！")
                print("=" * 60)
            else:
                print()
                print("=" * 60)
                print(f"⚠️  推理返回非零退出码: {result.returncode}")
                print("=" * 60)
        
        return result.returncode
    
    finally:
        # 恢复原目录
        if work_dir:
            os.chdir(original_dir)


def process_single_image(
    image_path: str,
    images_dir: Path,
    json_path: Path,
    script_dir: Path,
    run_inference_flag: bool = True,
    verbose: bool = True
) -> Tuple[bool, Optional[str]]:
    """处理单张图片
    
    参数:
        image_path: 图片路径
        images_dir: 目标图片目录
        json_path: JSON 配置文件路径
        script_dir: 脚本目录（用于推理）
        run_inference_flag: 是否执行推理
        verbose: 是否显示详细信息
    
    返回:
        (成功标志, 错误信息)
    """
    try:
        if verbose:
            print("=" * 60)
            print(f"📋 处理图片: {Path(image_path).name}")
            print("=" * 60)
        
        # 1. 获取图片尺寸
        if verbose:
            print("📏 读取图片尺寸...")
        width, height = get_image_size(image_path)
        if verbose:
            print(f"✓ 图片尺寸: {width} × {height}")
            print()
        
        # 2. 复制图片到目标目录
        if verbose:
            print("📁 复制图片到目标目录...")
        target_path = copy_image_to_target(image_path, images_dir, verbose=verbose)
        file_name = Path(target_path).name
        if verbose:
            print()
        
        # 3. 更新 JSON 文件
        if verbose:
            print("✏️  更新 JSON 文件...")
        update_custom_targets_json(json_path, file_name, width, height, verbose=verbose)
        if verbose:
            print()
        
        # 4. 执行推理（如果需要）
        if run_inference_flag:
            if verbose:
                print()
            return_code = run_inference(work_dir=str(script_dir), verbose=verbose)
            if return_code != 0:
                return False, f"推理失败，退出码: {return_code}"
        
        if verbose:
            print()
            print("=" * 60)
            print("📊 处理完成")
            print("=" * 60)
            print(f"✓ 图片已保存: {target_path}")
            print(f"✓ JSON 已更新: {json_path}")
            if run_inference_flag:
                print(f"✓ 推理结果保存在: {script_dir / 'results_analysis' / 'ship'}")
            print()
        
        return True, None
    
    except FileNotFoundError as e:
        return False, f"文件不存在: {e}"
    except ValueError as e:
        return False, f"值错误: {e}"
    except Exception as e:
        return False, f"未预期的错误: {e}"


def get_image_files(paths: List[str]) -> List[str]:
    """从路径列表中提取图片文件
    
    支持:
    - 单个图片文件
    - 目录（递归查找所有图片）
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if not path.exists():
            print(f"⚠️  路径不存在，跳过: {path}")
            continue
        
        if path.is_file():
            # 单个文件
            if path.suffix.lower() in image_extensions:
                image_files.append(str(path.absolute()))
            else:
                print(f"⚠️  不是图片文件，跳过: {path}")
        elif path.is_dir():
            # 目录：递归查找所有图片
            for ext in image_extensions:
                image_files.extend([str(p.absolute()) for p in path.rglob(f"*{ext}")])
                image_files.extend([str(p.absolute()) for p in path.rglob(f"*{ext.upper()}")])
    
    return sorted(set(image_files))  # 去重并排序


def batch_process(
    image_paths: List[str],
    images_dir: Path,
    json_path: Path,
    script_dir: Path,
    run_inference_flag: bool = True
):
    """批处理多张图片"""
    total = len(image_paths)
    
    print("=" * 60)
    print(f"📦 批处理模式: {total} 张图片")
    print("=" * 60)
    print()
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{total}] 处理: {Path(image_path).name}")
        print("-" * 60)
        
        success, error = process_single_image(
            image_path=image_path,
            images_dir=images_dir,
            json_path=json_path,
            script_dir=script_dir,
            run_inference_flag=run_inference_flag,
            verbose=False  # 批处理模式下不显示详细信息
        )
        
        if success:
            success_count += 1
            print(f"✅ [{idx}/{total}] 成功: {Path(image_path).name}")
        else:
            fail_count += 1
            failed_files.append((image_path, error))
            print(f"❌ [{idx}/{total}] 失败: {Path(image_path).name}")
            print(f"   错误: {error}")
    
    # 打印统计信息
    print()
    print("=" * 60)
    print("📊 批处理统计")
    print("=" * 60)
    print(f"总计: {total} 张图片")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    
    if failed_files:
        print()
        print("失败的文件:")
        for img_path, error in failed_files:
            print(f"  - {Path(img_path).name}: {error}")
    
    print("=" * 60)
    
    return fail_count == 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="自动处理图片并执行推理（支持单张和批处理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片
  python scripts/process_and_test.py image.jpg
  
  # 多张图片
  python scripts/process_and_test.py img1.jpg img2.jpg img3.jpg
  
  # 目录批处理
  python scripts/process_and_test.py --dir ./images
  
  # 只处理不推理（快速模式）
  python scripts/process_and_test.py --dir ./images --no-inference
        """
    )
    
    parser.add_argument(
        'images',
        nargs='*',
        help='图片文件路径或目录路径'
    )
    
    parser.add_argument(
        '--dir',
        dest='directory',
        help='批处理模式：处理目录下所有图片'
    )
    
    parser.add_argument(
        '--no-inference',
        action='store_true',
        help='只处理图片和更新JSON，不执行推理'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 确定图片列表
    image_paths = []
    
    if args.directory:
        # 目录模式
        image_paths = get_image_files([args.directory])
        if not image_paths:
            print(f"❌ 目录中没有找到图片文件: {args.directory}")
            sys.exit(1)
    elif args.images:
        # 文件列表模式
        image_paths = get_image_files(args.images)
        if not image_paths:
            print("❌ 没有找到有效的图片文件")
            sys.exit(1)
    else:
        # 没有参数，显示帮助
        parser.print_help()
        sys.exit(1)
    
    # 解析路径
    script_dir = Path(__file__).parent.parent.absolute()
    images_dir = script_dir / "data" / "FAR1M" / "Ship" / "images"
    json_path = script_dir / "data" / "FAR1M" / "Ship" / "annotations" / "custom_targets.json"
    
    # 处理图片
    if len(image_paths) == 1:
        # 单张图片：显示详细信息
        success, error = process_single_image(
            image_path=image_paths[0],
            images_dir=images_dir,
            json_path=json_path,
            script_dir=script_dir,
            run_inference_flag=not args.no_inference,
            verbose=True
        )
        
        if not success:
            print(f"❌ 处理失败: {error}")
            sys.exit(1)
    else:
        # 多张图片：批处理模式
        success = batch_process(
            image_paths=image_paths,
            images_dir=images_dir,
            json_path=json_path,
            script_dir=script_dir,
            run_inference_flag=not args.no_inference
        )
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()

