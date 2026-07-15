#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 JSON 中筛选指定 ID 的实例并重新生成可视化

用法:
    python scripts/filter_json_by_ids.py <json_path> <id1> <id2> ... [选项]

示例:
    # 只重新生成可视化，不进行筛选
    python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json --regenerate-only
    
    # 按索引：只保留索引 0, 1, 2 的实例（保留模式，默认）
    python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 1 2
    
    # 删除索引 0, 1, 2，保留其他的实例（删除模式）
    python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 1 2 --remove
    
    # 保留索引 0, 3, 5，并重新生成所有可视化
    python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 3 5 --regenerate
    
    # 删除索引 0, 3, 5，并重新生成所有可视化
    python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 3 5 --remove --regenerate
    
    # 只更新JSON文件，不重新生成可视化
    python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 1 2 --no-regenerate
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Set, Optional
import subprocess


def extract_score_from_category_name(category_name: str) -> int:
    """从 category_name 中提取置信度值
    
    例如: "ship=95" -> 95
    """
    if '=' in category_name:
        try:
            return int(float(category_name.split('=')[1]))
        except (ValueError, IndexError):
            return None
    return None


def filter_json_by_ids(json_path: Path, keep_values: Set[int], backup: bool = True, by_score: bool = False, remove_mode: bool = False) -> List[dict]:
    """从 JSON 文件中筛选指定 ID 或置信度值的实例
    
    参数:
        json_path: JSON 文件路径
        keep_values: 要保留（或删除）的实例索引（from0开始）或置信度值（如95）
        backup: 是否备份原文件
        by_score: True=按置信度值筛选，False=按索引筛选
        remove_mode: True=删除指定的ID，False=保留指定的ID（默认）
    
    返回:
        筛选后的实例列表
    """
    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        instances = json.load(f)
    
    if not isinstance(instances, list):
        raise ValueError(f"JSON 文件应包含一个列表，但得到: {type(instances)}")
    
    total_count = len(instances)
    
    if total_count == 0:
        print("⚠️  警告: JSON 文件为空，无法筛选")
        # 尝试从备份文件恢复
        backup_path = json_path.with_suffix('.json.bak')
        if backup_path.exists():
            print(f"   发现备份文件: {backup_path}")
            try:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_instances = json.load(f)
                if isinstance(backup_instances, list) and len(backup_instances) > 0:
                    print(f"   备份文件包含 {len(backup_instances)} 个实例")
                    print(f"   是否要恢复备份文件？(y/n): ", end='')
                    response = input().strip().lower()
                    if response == 'y':
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(backup_instances, f, indent=2, ensure_ascii=False)
                        print(f"   ✓ 已从备份文件恢复")
                        instances = backup_instances
                        total_count = len(instances)
                    else:
                        return []
                else:
                    print(f"   备份文件也是空的")
                    return []
            except Exception as e:
                print(f"   ✗ 无法读取备份文件: {e}")
                return []
        else:
            return []
    
    # 备份原文件
    if backup:
        backup_path = json_path.with_suffix('.json.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(instances, f, indent=2, ensure_ascii=False)
        print(f"✓ 已备份原文件到: {backup_path}")
    
    # 筛选实例
    if by_score:
        # 按置信度值筛选（从 category_name 中提取）
        filtered_instances = []
        matched_scores = []
        for inst in instances:
            category_name = inst.get('category_name', '')
            score_val = extract_score_from_category_name(category_name)
            should_keep = score_val is not None and score_val in keep_values
            if remove_mode:
                # 删除模式：保留不在 keep_values 中的实例
                if score_val is None or score_val not in keep_values:
                    filtered_instances.append(inst)
                elif score_val in keep_values:
                    matched_scores.append(score_val)
            else:
                # 保留模式：只保留在 keep_values 中的实例
                if should_keep:
                    filtered_instances.append(inst)
                    matched_scores.append(score_val)
        
        # 检查是否有未匹配的值
        unmatched_values = keep_values - set(matched_scores)
        if unmatched_values and not remove_mode:
            print(f"⚠️  警告: 以下置信度值未找到匹配实例（已忽略）: {sorted(unmatched_values)}")
        elif unmatched_values and remove_mode:
            print(f"ℹ️  信息: 以下置信度值未找到匹配实例（无需删除）: {sorted(unmatched_values)}")
    else:
        # 按索引筛选
        if remove_mode:
            # 删除模式：保留不在 keep_values 中的索引
            filtered_instances = [instances[i] for i in range(total_count) if i not in keep_values]
            # 检查是否有无效的 ID
            invalid_ids = [i for i in keep_values if i < 0 or i >= total_count]
            if invalid_ids:
                print(f"ℹ️  信息: 以下索引无效（无需删除）: {sorted(invalid_ids)}")
        else:
            # 保留模式：只保留在 keep_values 中的索引
            filtered_instances = [instances[i] for i in keep_values if 0 <= i < total_count]
            # 检查是否有无效的 ID
            invalid_ids = [i for i in keep_values if i < 0 or i >= total_count]
            if invalid_ids:
                print(f"⚠️  警告: 以下索引无效（已忽略）: {sorted(invalid_ids)}")
    
    # 保存筛选后的 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_instances, f, indent=2, ensure_ascii=False)
    
    mode_str = "删除" if remove_mode else "保留"
    print(f"✓ JSON 已更新 ({mode_str}模式):")
    print(f"   原始实例数: {total_count}")
    print(f"   保留实例数: {len(filtered_instances)}")
    print(f"   删除实例数: {total_count - len(filtered_instances)}")
    
    return filtered_instances


def regenerate_visualizations(json_path: Path, script_dir: Path, score_thr: float = 0.0, img_path: Optional[Path] = None):
    """调用 regenerate_from_json.py 重新生成可视化
    
    参数:
        json_path: JSON 文件路径
        script_dir: 脚本目录
        score_thr: 置信度阈值
        img_path: 图像路径（可选）
    """
    print()
    print("=" * 60)
    print("🔄 重新生成可视化输出...")
    print("=" * 60)
    
    # 调用 regenerate_from_json.py
    cmd = [
        sys.executable,
        str(script_dir / "scripts" / "regenerate_from_json.py"),
        str(json_path),
        "--score_thr", str(score_thr)  # 保存所有实例（因为已经筛选过了）
    ]
    
    # 如果提供了图像路径，添加到命令中
    if img_path:
        cmd.extend(["--img_path", str(img_path)])
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=str(script_dir))
    
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("✅ 可视化重新生成完成！")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print(f"⚠️  可视化重新生成返回非零退出码: {result.returncode}")
        print("=" * 60)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="从 JSON 中筛选指定 ID 的实例并重新生成可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只重新生成可视化，不进行筛选
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json --regenerate-only
  
  # 按索引：只保留索引 0, 1, 2 的实例
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 1 2
  
  # 按索引：删除索引 0, 1, 2，保留其他的实例
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 1 2 --remove
  
  # 按置信度值：只保留 ship=95, ship=92, ship=65 的实例
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 95 92 65 --by-score
  
  # 按置信度值：删除 ship=95, ship=92，保留其他的实例
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 95 92 --by-score --remove
  
  # 保留索引 0, 3, 5，并重新生成所有可视化
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 3 5 --regenerate
  
  # 删除索引 0, 3, 5，并重新生成所有可视化
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 3 5 --remove --regenerate
  
  # 只更新JSON文件，不重新生成可视化
  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 1 2 --no-regenerate
        """
    )
    
    parser.add_argument('json_path', help='JSON 文件路径')
    parser.add_argument('ids', type=int, nargs='*', help='要保留（默认）或删除（使用 --remove）的值：索引（默认）或置信度值（使用 --by-score）。如果不提供，且使用 --regenerate-only，则只重新生成可视化')
    parser.add_argument('--by-score', action='store_true', help='按置信度值筛选（从 category_name 中提取，如 ship=95 中的 95）')
    parser.add_argument('--by-index', action='store_true', help='按索引筛选（从0开始，默认模式）')
    parser.add_argument('--remove', action='store_true', help='删除模式：删除指定的ID，保留其他的（默认是保留模式）')
    parser.add_argument('--regenerate-only', action='store_true', help='只重新生成可视化，不进行筛选（忽略 ids 参数）')
    parser.add_argument('--regenerate', action='store_true', default=True, help='重新生成可视化（默认启用）')
    parser.add_argument('--no-regenerate', dest='regenerate', action='store_false', help='不重新生成可视化')
    parser.add_argument('--no-backup', dest='backup', action='store_false', default=True, help='不备份原文件')
    parser.add_argument('--score-thr', type=float, default=0.0, help='传递给 regenerate_from_json.py 的置信度阈值（默认0.0）')
    parser.add_argument('--img-path', help='图像路径（可选，如果不提供会自动查找）')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"❌ JSON 文件不存在: {json_path}")
        sys.exit(1)
    
    # 解析脚本目录
    script_dir = Path(__file__).parent.parent.absolute()
    
    # 如果使用 --regenerate-only，直接重新生成并退出
    if args.regenerate_only:
        print("=" * 60)
        print("🔄 重新生成可视化（不进行筛选）")
        print("=" * 60)
        print(f"JSON 文件: {json_path}")
        print()
        
        # 读取 JSON 以显示信息
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                instances = json.load(f)
            if isinstance(instances, list):
                print(f"✓ JSON 文件包含 {len(instances)} 个实例")
                print()
        except Exception as e:
            print(f"⚠️  警告: 无法读取 JSON 文件: {e}")
            print()
        
        # 重新生成可视化
        img_path = Path(args.img_path) if args.img_path else None
        success = regenerate_visualizations(json_path, script_dir, score_thr=args.score_thr, img_path=img_path)
        if not success:
            print()
            print("⚠️  可视化重新生成失败")
            sys.exit(1)
        
        print()
        print("=" * 60)
        print("✅ 完成！")
        print("=" * 60)
        print(f"✓ 可视化已重新生成")
        print()
        sys.exit(0)
    
    # 检查是否提供了 IDs
    if not args.ids:
        print("❌ 错误: 必须提供至少一个 ID，或使用 --regenerate-only 只重新生成可视化")
        print()
        print("示例:")
        print("  # 只重新生成可视化，不筛选")
        print("  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json --regenerate-only")
        print()
        print("  # 筛选并重新生成")
        print("  python scripts/filter_json_by_ids.py results_analysis/ship/json/000000_prediction.json 0 1 2")
        sys.exit(1)
    
    print("=" * 60)
    print("🔍 筛选 JSON 实例")
    print("=" * 60)
    print(f"JSON 文件: {json_path}")
    
    # 确定筛选模式
    by_score = args.by_score and not args.by_index
    remove_mode = args.remove
    
    if by_score:
        mode_str = "删除" if remove_mode else "保留"
        print(f"筛选模式: 按置信度值 ({mode_str}模式)")
        print(f"{mode_str}置信度值: {sorted(args.ids)}")
    else:
        mode_str = "删除" if remove_mode else "保留"
        print(f"筛选模式: 按索引 ({mode_str}模式)")
        print(f"{mode_str}实例索引: {sorted(args.ids)}")
    print()
    
    # 筛选 JSON
    keep_values = set(args.ids)
    filtered_instances = filter_json_by_ids(
        json_path=json_path,
        keep_values=keep_values,
        backup=args.backup,
        by_score=by_score,
        remove_mode=remove_mode
    )
    
    if len(filtered_instances) == 0:
        print()
        print("⚠️  警告: 没有保留任何实例！")
        print("   请检查 ID 是否正确")
        print()
        # 即使没有实例，也允许继续重新生成可视化（生成空的可视化）
        if args.regenerate:
            print("ℹ️  将继续重新生成可视化（将生成空的可视化结果）")
            print()
            img_path = Path(args.img_path) if args.img_path else None
            success = regenerate_visualizations(json_path, script_dir, score_thr=args.score_thr, img_path=img_path)
            if not success:
                print()
                print("⚠️  可视化重新生成失败，但 JSON 已更新")
                sys.exit(1)
            print()
            print("=" * 60)
            print("✅ 完成！")
            print("=" * 60)
            print(f"✓ JSON 已更新: {json_path} (现在为空)")
            if args.regenerate:
                print(f"✓ 可视化已重新生成（空结果）")
            if args.backup:
                print(f"✓ 原文件已备份: {json_path.with_suffix('.json.bak')}")
            print()
            sys.exit(0)
        else:
            print("ℹ️  跳过可视化重新生成")
            print()
            print("=" * 60)
            print("✅ JSON 已更新（现在为空）")
            print("=" * 60)
            if args.backup:
                print(f"✓ 原文件已备份: {json_path.with_suffix('.json.bak')}")
            print()
            sys.exit(0)
    
    # 显示保留的实例信息
    print()
    print("保留的实例:")
    for idx, inst in enumerate(filtered_instances):
        category_name = inst.get('category_name', 'unknown')
        score = inst.get('score', 0.0)
        bbox = inst.get('bbox', [])
        print(f"  [{idx}] {category_name} (score: {score:.3f}, bbox: {bbox})")
    
    # 重新生成可视化（如果需要）
    if args.regenerate:
        img_path = Path(args.img_path) if args.img_path else None
        success = regenerate_visualizations(json_path, script_dir, score_thr=args.score_thr, img_path=img_path)
        if not success:
            print()
            print("⚠️  可视化重新生成失败，但 JSON 已更新")
            sys.exit(1)
    else:
        print()
        print("ℹ️  跳过可视化重新生成（使用 --regenerate 启用）")
    
    print()
    print("=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"✓ JSON 已更新: {json_path}")
    if args.regenerate:
        print(f"✓ 可视化已重新生成")
    if args.backup:
        print(f"✓ 原文件已备份: {json_path.with_suffix('.json.bak')}")
    print()


if __name__ == "__main__":
    main()

