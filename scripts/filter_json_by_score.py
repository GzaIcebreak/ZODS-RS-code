#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据置信度阈值过滤 JSON 文件中的实例

用法:
    python scripts/filter_json_by_score.py <json_path> --threshold <阈值> [选项]
    python scripts/filter_json_by_score.py <目录> --threshold <阈值> --batch [选项]

示例:
    # 过滤单个 JSON 文件，删除 score < 0.75 的实例
    python scripts/filter_json_by_score.py results_analysis/tomb/json/000084_prediction.json --threshold 0.75
    
    # 批量处理目录中的所有 JSON 文件
    python scripts/filter_json_by_score.py results_analysis/tomb/json --threshold 0.75 --batch
    
    # 不备份原文件
    python scripts/filter_json_by_score.py results_analysis/tomb/json/000084_prediction.json --threshold 0.75 --no-backup
    
    # 过滤后重新生成可视化
    python scripts/filter_json_by_score.py results_analysis/tomb/json/000084_prediction.json --threshold 0.75 --regenerate
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess


def filter_json_by_score(
    json_path: Path,
    threshold: float,
    backup: bool = True
) -> Dict[str, Any]:
    """根据置信度阈值过滤 JSON 文件中的实例
    
    参数:
        json_path: JSON 文件路径
        threshold: 置信度阈值（保留 score >= threshold 的实例）
        backup: 是否备份原文件
    
    返回:
        包含统计信息的字典
    """
    # 读取 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            instances = json.load(f)
    except Exception as e:
        return {
            'success': False,
            'error': f"无法读取 JSON 文件: {e}",
            'total': 0,
            'kept': 0,
            'removed': 0
        }
    
    if not isinstance(instances, list):
        return {
            'success': False,
            'error': f"JSON 文件应包含一个列表，但得到: {type(instances)}",
            'total': 0,
            'kept': 0,
            'removed': 0
        }
    
    total_count = len(instances)
    
    if total_count == 0:
        return {
            'success': True,
            'error': None,
            'total': 0,
            'kept': 0,
            'removed': 0,
            'message': 'JSON 文件为空，无需处理'
        }
    
    # 备份原文件
    if backup:
        backup_path = json_path.with_suffix('.json.bak')
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(instances, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return {
                'success': False,
                'error': f"无法备份文件: {e}",
                'total': total_count,
                'kept': 0,
                'removed': 0
            }
    
    # 过滤实例
    filtered_instances = []
    removed_scores = []
    
    for inst in instances:
        score = float(inst.get('score', 0.0))
        if score >= threshold:
            filtered_instances.append(inst)
        else:
            removed_scores.append(score)
    
    kept_count = len(filtered_instances)
    removed_count = total_count - kept_count
    
    # 保存筛选后的 JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_instances, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return {
            'success': False,
            'error': f"无法保存 JSON 文件: {e}",
            'total': total_count,
            'kept': kept_count,
            'removed': removed_count
        }
    
    return {
        'success': True,
        'error': None,
        'total': total_count,
        'kept': kept_count,
        'removed': removed_count,
        'backup_path': backup_path if backup else None,
        'removed_scores': removed_scores
    }


def regenerate_visualizations(
    json_path: Path,
    script_dir: Path,
    score_thr: float = 0.0,
    img_path: Optional[Path] = None
) -> bool:
    """调用 regenerate_from_json.py 重新生成可视化
    
    参数:
        json_path: JSON 文件路径
        script_dir: 脚本目录
        score_thr: 置信度阈值（传递给 regenerate_from_json.py）
        img_path: 图像路径（可选）
    
    返回:
        是否成功
    """
    cmd = [
        sys.executable,
        str(script_dir / "scripts" / "regenerate_from_json.py"),
        str(json_path),
        "--score_thr", str(score_thr)
    ]
    
    if img_path:
        cmd.extend(["--img_path", str(img_path)])
    
    result = subprocess.run(cmd, cwd=str(script_dir), capture_output=True, text=True)
    return result.returncode == 0


def process_single_file(
    json_path: Path,
    threshold: float,
    backup: bool,
    regenerate: bool,
    script_dir: Path,
    img_path: Optional[Path] = None
) -> bool:
    """处理单个 JSON 文件
    
    返回:
        是否成功
    """
    print(f"📄 处理文件: {json_path.name}")
    
    # 过滤 JSON
    result = filter_json_by_score(json_path, threshold, backup=backup)
    
    if not result['success']:
        print(f"   ❌ 错误: {result['error']}")
        return False
    
    if result.get('message'):
        print(f"   ℹ️  {result['message']}")
        return True
    
    print(f"   ✓ 原始实例数: {result['total']}")
    print(f"   ✓ 保留实例数: {result['kept']} (score >= {threshold})")
    print(f"   ✓ 删除实例数: {result['removed']}")
    
    if result['removed'] > 0:
        removed_scores = result['removed_scores']
        if removed_scores:
            min_score = min(removed_scores)
            max_score = max(removed_scores)
            print(f"   📊 删除的分数范围: [{min_score:.3f}, {max_score:.3f})")
    
    if result.get('backup_path'):
        print(f"   💾 备份文件: {result['backup_path'].name}")
    
    # 重新生成可视化（如果需要）
    if regenerate:
        print(f"   🔄 重新生成可视化...")
        success = regenerate_visualizations(json_path, script_dir, score_thr=threshold, img_path=img_path)
        if success:
            print(f"   ✓ 可视化已重新生成")
        else:
            print(f"   ⚠️  可视化重新生成失败（但 JSON 已更新）")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="根据置信度阈值过滤 JSON 文件中的实例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 过滤单个 JSON 文件，删除 score < 0.75 的实例
  python scripts/filter_json_by_score.py results_analysis/tomb/json/000084_prediction.json --threshold 0.75
  
  # 批量处理目录中的所有 JSON 文件
  python scripts/filter_json_by_score.py results_analysis/tomb/json --threshold 0.75 --batch
  
  # 不备份原文件
  python scripts/filter_json_by_score.py results_analysis/tomb/json/000084_prediction.json --threshold 0.75 --no-backup
  
  # 过滤后重新生成可视化
  python scripts/filter_json_by_score.py results_analysis/tomb/json/000084_prediction.json --threshold 0.75 --regenerate
  
  # 批量处理并重新生成可视化
  python scripts/filter_json_by_score.py results_analysis/tomb/json --threshold 0.75 --batch --regenerate
        """
    )
    
    parser.add_argument('path', help='JSON 文件路径或包含 JSON 文件的目录')
    parser.add_argument('--threshold', '-t', type=float, required=True, help='置信度阈值（保留 score >= threshold 的实例）')
    parser.add_argument('--batch', action='store_true', help='批量处理模式：如果 path 是目录，处理其中的所有 JSON 文件')
    parser.add_argument('--regenerate', action='store_true', help='过滤后重新生成可视化')
    parser.add_argument('--no-backup', dest='backup', action='store_false', default=True, help='不备份原文件')
    parser.add_argument('--img-path', help='图像路径（用于重新生成可视化，可选）')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"❌ 路径不存在: {path}")
        sys.exit(1)
    
    # 解析脚本目录
    script_dir = Path(__file__).parent.parent.absolute()
    
    # 确定要处理的文件列表
    json_files = []
    
    if path.is_file():
        if path.suffix.lower() != '.json':
            print(f"❌ 不是 JSON 文件: {path}")
            sys.exit(1)
        json_files = [path]
    elif path.is_dir():
        if args.batch:
            # 批量处理模式：查找所有 JSON 文件（排除备份文件）
            json_files = [f for f in path.glob('*.json') if not f.name.endswith('.bak')]
            json_files.sort()
        else:
            print(f"❌ 路径是目录，请使用 --batch 选项进行批量处理")
            print(f"   或指定具体的 JSON 文件路径")
            sys.exit(1)
    else:
        print(f"❌ 无效的路径: {path}")
        sys.exit(1)
    
    if len(json_files) == 0:
        print(f"⚠️  未找到 JSON 文件")
        sys.exit(1)
    
    print("=" * 60)
    print("🔍 根据置信度阈值过滤 JSON 文件")
    print("=" * 60)
    print(f"阈值: {args.threshold}")
    print(f"文件数量: {len(json_files)}")
    print(f"备份: {'是' if args.backup else '否'}")
    print(f"重新生成可视化: {'是' if args.regenerate else '否'}")
    if args.img_path:
        print(f"图像路径: {args.img_path}")
    print()
    
    # 处理文件
    success_count = 0
    fail_count = 0
    
    img_path = Path(args.img_path) if args.img_path else None
    
    for json_file in json_files:
        success = process_single_file(
            json_file,
            args.threshold,
            args.backup,
            args.regenerate,
            script_dir,
            img_path
        )
        if success:
            success_count += 1
        else:
            fail_count += 1
        print()
    
    # 显示总结
    print("=" * 60)
    print("📊 处理总结")
    print("=" * 60)
    print(f"✓ 成功: {success_count}")
    if fail_count > 0:
        print(f"❌ 失败: {fail_count}")
    print()


if __name__ == "__main__":
    main()

