#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
恢复 JSON 备份文件

用法:
    python scripts/restore_json_backup.py <json_path>
"""

import sys
import json
from pathlib import Path


def restore_backup(json_path: Path):
    """恢复 JSON 备份文件"""
    backup_path = json_path.with_suffix('.json.bak')
    
    if not backup_path.exists():
        print(f"❌ 备份文件不存在: {backup_path}")
        return False
    
    # 读取备份文件
    with open(backup_path, 'r', encoding='utf-8') as f:
        backup_data = json.load(f)
    
    if not isinstance(backup_data, list):
        print(f"❌ 备份文件格式错误: 应为列表")
        return False
    
    # 恢复 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已从备份文件恢复: {json_path}")
    print(f"   实例数量: {len(backup_data)}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/restore_json_backup.py <json_path>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    if restore_backup(json_path):
        print("✅ 恢复成功！")
    else:
        print("❌ 恢复失败！")
        sys.exit(1)

