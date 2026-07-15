#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 bbox 面积的统计特性过滤 JSON 文件中的异常实例，并可选地重新生成可视化。

核心特性:
  * 自动计算 bbox 面积 (width * height) 并在对数空间中执行稳健的离群检测；
  * 默认使用 MAD (Median Absolute Deviation) 估计，阈值以 z-score（绝对值）表示；
  * 支持按图片整体或按类别（如 "tomb"）分别建模；
  * 支持批量目录处理、备份、干跑 (dry-run) 和联动 regenerate_from_json.py 重新生成输出。

用法示例:
  # 过滤单个 JSON，删除 z-score > 3.5 的大面积离群框
  python scripts/filter_json_by_area.py results_analysis/tomb/json/000150_prediction.json --threshold 3.5

  # 按类别（如 tomb）分别建模，阈值 3.0，并重新生成可视化
  python scripts/filter_json_by_area.py results_analysis/tomb/json/000150_prediction.json \
      --threshold 3.0 --per-category --regenerate --img-path data/my_build/images/000150.jpg

  # 批量处理目录，阈值 3.2，仅打印将被删除的实例，不落盘
  python scripts/filter_json_by_area.py results_analysis/tomb/json --batch --threshold 3.2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import subprocess


# ------------------------------ 数据结构 ------------------------------ #

@dataclass
class InstanceInfo:
    index: int
    area: float
    log_area: float
    category_name: str
    base_category: str
    bbox: List[float]
    score: float


@dataclass
class RemovalDecision:
    index: int
    area: float
    log_area: float
    z_score: float
    base_category: str


@dataclass
class FilterResult:
    success: bool
    total: int
    kept: int
    removed: int
    removals: List[RemovalDecision]
    backup_path: Optional[Path] = None
    error: Optional[str] = None


# ------------------------------ 工具函数 ------------------------------ #

def parse_base_category(category_name: str) -> str:
    """从 "tomb=91" 提取基础类别 "tomb"。"""
    if not category_name:
        return "unknown"
    if "=" in category_name:
        return category_name.split("=", 1)[0].strip() or "unknown"
    return category_name.strip() or "unknown"


def compute_instance_info(instances: List[Dict[str, object]], *, per_category: bool) -> Dict[str, List[InstanceInfo]]:
    """整理实例信息，按类别或整张图聚类。"""
    groups: Dict[str, List[InstanceInfo]] = {}
    for idx, inst in enumerate(instances):
        bbox = inst.get("bbox", inst.get("bbox_xywh"))
        if not bbox or len(bbox) < 4:
            # 缺失 bbox 时跳过（不参与建模，也不删除）
            continue
        try:
            w = float(bbox[2])
            h = float(bbox[3])
        except (TypeError, ValueError):
            continue
        area = max(w, 0.0) * max(h, 0.0)
        log_area = math.log(area + 1.0)  # 避免 0 面积取对数问题
        category_name = str(inst.get("category_name", ""))
        base_category = parse_base_category(category_name) if per_category else "__ALL__"
        info = InstanceInfo(
            index=idx,
            area=area,
            log_area=log_area,
            category_name=category_name,
            base_category=base_category,
            bbox=list(bbox) if isinstance(bbox, (list, tuple)) else [float(x) for x in bbox],
            score=float(inst.get("score", 0.0)),
        )
        groups.setdefault(base_category, []).append(info)
    return groups


def robust_z_scores(values: List[float]) -> Tuple[List[float], str]:
    """使用 MAD 估计稳健标准化分数，若 MAD≈0 则回退到样本标准差。"""
    if not values:
        return [], "empty"
    median = statistics.median(values)
    abs_dev = [abs(v - median) for v in values]
    mad = statistics.median(abs_dev)
    if mad > 1e-9:
        factor = 0.6745 / mad  # 0.6745 ≈ inverse Φ^-1(3/4)
        z_scores = [(v - median) * factor for v in values]
        return z_scores, "mad"
    # 回退：整体方差极小
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev <= 1e-9:
        return [0.0 for _ in values], "constant"
    z_scores = [(v - mean) / stdev for v in values]
    return z_scores, "std"


def detect_large_outliers(
    groups: Dict[str, List[InstanceInfo]],
    *,
    z_threshold: float,
    min_samples: int,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
) -> List[RemovalDecision]:
    """针对每个分组找出大面积离群点。"""
    removals: List[RemovalDecision] = []
    for key, infos in groups.items():
        if len(infos) < max(min_samples, 3):
            continue  # 数据太少，跳过
        log_areas = [info.log_area for info in infos]
        z_scores, method = robust_z_scores(log_areas)
        for info, z in zip(infos, z_scores):
            remove = False
            reason = []
            if max_area is not None and info.area > max_area:
                remove = True
                reason.append(f"area>{max_area}")
            if min_area is not None and info.area < min_area:
                # 仅当用户显式设置 min_area 时才考虑过小
                remove = True
                reason.append(f"area<{min_area}")
            if z_threshold is not None and z > z_threshold:
                remove = True
                reason.append(f"z>{z_threshold:.2f} ({method})")
            if remove:
                removals.append(
                    RemovalDecision(
                        index=info.index,
                        area=info.area,
                        log_area=info.log_area,
                        z_score=z,
                        base_category=info.base_category,
                    )
                )
    # 去重（同一实例可能由于不同约束重复）
    unique: Dict[int, RemovalDecision] = {}
    for item in removals:
        if item.index not in unique:
            unique[item.index] = item
        else:
            # 保留更大的 z-score 以提供更具说明力的统计
            if item.z_score > unique[item.index].z_score:
                unique[item.index] = item
    return sorted(unique.values(), key=lambda x: x.index)


def backup_file(json_path: Path) -> Optional[Path]:
    backup_path = json_path.with_suffix('.json.bak')
    with open(json_path, 'r', encoding='utf-8') as f_in, open(backup_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f_in.read())
    return backup_path


def regenerate_visualizations(
    json_path: Path,
    script_root: Path,
    *,
    score_thr: float = 0.0,
    img_path: Optional[Path] = None,
) -> bool:
    cmd = [
        sys.executable,
        str(script_root / 'scripts' / 'regenerate_from_json.py'),
        str(json_path),
        '--score_thr', str(score_thr),
    ]
    if img_path:
        cmd.extend(['--img_path', str(img_path)])
    result = subprocess.run(cmd, cwd=str(script_root))
    return result.returncode == 0


# ------------------------------ 核心逻辑 ------------------------------ #

import sys


def filter_json_file(
    json_path: Path,
    *,
    z_threshold: float,
    min_samples: int,
    per_category: bool,
    min_area: Optional[float],
    max_area: Optional[float],
    backup: bool,
    dry_run: bool,
) -> FilterResult:
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            instances = json.load(f)
        except json.JSONDecodeError as exc:
            return FilterResult(False, 0, 0, 0, [], error=f"JSON 解析失败: {exc}")

    if not isinstance(instances, list):
        return FilterResult(False, 0, 0, 0, [], error=f"JSON 顶层应为列表，但得到 {type(instances)}")

    total = len(instances)
    groups = compute_instance_info(instances, per_category=per_category)
    removals = detect_large_outliers(
        groups,
        z_threshold=z_threshold,
        min_samples=min_samples,
        min_area=min_area,
        max_area=max_area,
    )

    if not removals:
        return FilterResult(True, total, total, 0, [], None)

    keep_indices = {r.index for r in removals}
    filtered_instances = [inst for idx, inst in enumerate(instances) if idx not in keep_indices]

    backup_path = None
    if not dry_run and backup:
        backup_path = backup_file(json_path)

    if not dry_run:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_instances, f, ensure_ascii=False, indent=2)

    return FilterResult(
        success=True,
        total=total,
        kept=len(filtered_instances),
        removed=len(removals),
        removals=removals,
        backup_path=backup_path,
    )


# ------------------------------ CLI ------------------------------ #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据 bbox 面积异常过滤 JSON 实例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
主要参数:
  --threshold/-t         MAD (或 std) z-score 阈值，默认 3.5，仅删除 z-score 超过阈值（左侧保留）
  --min-area             额外设置的最小面积（像素^2），小于该值的实例也会被删除
  --max-area             额外设置的最大面积（像素^2），大于该值的实例也会被删除
  --per-category         按类别分别建模（例如 tomb），默认对整张图的所有实例建模
  --min-samples          每个组至少需要的样本数（默认 8），不足则跳过过滤
  --regenerate           过滤后调用 regenerate_from_json.py 重新生成输出
  --img-path             重新生成时显式指定原图路径
  --score-thr            重新生成时传递给 regenerate_from_json.py 的置信度阈值
  --dry-run              只打印将要删除的实例，不改写 JSON
  --batch                如果 path 是目录，则批量处理其中的所有 JSON 文件
        """
    )

    parser.add_argument('path', help='JSON 文件路径或包含 JSON 的目录')
    parser.add_argument('--threshold', '-t', type=float, default=3.5, help='MAD z-score 阈值 (默认 3.5)')
    parser.add_argument('--min-area', type=float, default=None, help='最小面积阈值 (像素^2)')
    parser.add_argument('--max-area', type=float, default=None, help='最大面积阈值 (像素^2)')
    parser.add_argument('--min-samples', type=int, default=8, help='每组最少样本数 (默认 8)')
    parser.add_argument('--per-category', action='store_true', help='按类别分别建模 (默认整张图)')
    parser.add_argument('--batch', action='store_true', help='对目录内所有 JSON 文件进行批量处理')
    parser.add_argument('--dry-run', action='store_true', help='只打印结果，不写回文件')
    parser.add_argument('--no-backup', dest='backup', action='store_false', default=True, help='不备份原 JSON')
    parser.add_argument('--regenerate', action='store_true', help='过滤完成后重新生成可视化输出')
    parser.add_argument('--img-path', help='重新生成可视化时指定原始图像路径')
    parser.add_argument('--score-thr', type=float, default=0.0, help='重新生成时的置信度阈值 (默认 0.0)')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    target_path = Path(args.path)
    if not target_path.exists():
        print(f"❌ 路径不存在: {target_path}")
        sys.exit(1)

    if target_path.is_dir() and not args.batch:
        print("❌ 目标是目录，请使用 --batch 或提供具体的 JSON 文件")
        sys.exit(1)

    script_root = Path(__file__).parent.parent.resolve()
    img_path = Path(args.img_path) if args.img_path else None

    json_files: List[Path]
    if target_path.is_dir():
        json_files = sorted(f for f in target_path.glob('*.json') if not f.name.endswith('.json.bak'))
    else:
        json_files = [target_path]

    if not json_files:
        print("⚠️  未找到 JSON 文件")
        sys.exit(1)

    print("=" * 60)
    print("🧠 面积驱动的异常实例过滤")
    print("=" * 60)
    print(f"阈值 (z-score): {args.threshold}")
    print(f"最小样本数: {args.min_samples}")
    print(f"按类别建模: {'是' if args.per_category else '否'}")
    print(f"最小面积: {args.min_area if args.min_area is not None else '未设定'}")
    print(f"最大面积: {args.max_area if args.max_area is not None else '未设定'}")
    print(f"干跑 (dry-run): {'是' if args.dry_run else '否'}")
    print(f"备份原文件: {'是' if args.backup else '否'}")
    print(f"重新生成可视化: {'是' if args.regenerate else '否'}")
    if img_path:
        print(f"图像路径: {img_path}")
    print(f"目标文件数: {len(json_files)}")
    print()

    success = 0
    failure = 0

    for json_file in json_files:
        print("-" * 60)
        print(f"📄 处理: {json_file}")
        result = filter_json_file(
            json_file,
            z_threshold=args.threshold,
            min_samples=args.min_samples,
            per_category=args.per_category,
            min_area=args.min_area,
            max_area=args.max_area,
            backup=args.backup,
            dry_run=args.dry_run,
        )

        if not result.success:
            print(f"   ❌ 失败: {result.error}")
            failure += 1
            continue

        print(f"   总实例: {result.total}")
        print(f"   保留实例: {result.kept}")
        print(f"   删除实例: {result.removed}")
        if result.backup_path:
            print(f"   💾 备份文件: {result.backup_path}")

        if result.removed:
            print("   移除列表 (index / area / log_area / z_score / group):")
            for item in result.removals:
                print(
                    f"      - idx={item.index:<4d} | area={item.area:>10.1f} | "
                    f"log_area={item.log_area:>8.3f} | z={item.z_score:>6.3f} | group={item.base_category}"
                )
        else:
            print("   ℹ️  未检测到异常实例")

        if args.regenerate and not args.dry_run:
            print("   🔄 重新生成可视化输出...")
            regen_success = regenerate_visualizations(
                json_file,
                script_root,
                score_thr=args.score_thr,
                img_path=img_path,
            )
            if regen_success:
                print("   ✅ 可视化已更新")
            else:
                print("   ⚠️ 可视化重新生成失败 (JSON 已更新)")

        success += 1
        print()

    print("=" * 60)
    print("📊 统计总结")
    print("=" * 60)
    print(f"成功处理: {success}")
    print(f"失败处理: {failure}")
    if args.dry_run and success:
        print("⚠️ 当前为 dry-run 模式，JSON 未被修改")
    print()


if __name__ == '__main__':
    main()
