# JSON 实例筛选功能

## 🆕 功能说明

`scripts/filter_json_by_ids.py` 可以从 JSON 文件中筛选指定 ID 的实例，删除其余实例，并重新生成可视化输出。

---

## 📋 使用方法

### 基本用法
```bash
python scripts/filter_json_by_ids.py <json_path> <id1> <id2> ... [选项]
```

### 示例

#### 1. 只保留索引 0, 1, 2 的实例
```bash
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 1 2
```

#### 2. 保留索引 0, 3, 5，并重新生成所有可视化
```bash
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 3 5 \
  --regenerate
```

#### 3. 只更新 JSON 文件，不重新生成可视化
```bash
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 1 2 \
  --no-regenerate
```

#### 4. 不备份原文件
```bash
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 1 2 \
  --no-backup
```

---

## 🚀 功能特性

### 自动备份
- 默认自动备份原 JSON 文件为 `.json.bak`
- 使用 `--no-backup` 禁用备份

### 自动重新生成
- 默认自动重新生成所有可视化输出
- 使用 `--no-regenerate` 禁用重新生成

### 实例索引
- 使用 **从 0 开始的索引**（列表索引）
- 可以指定多个 ID
- 无效的 ID 会被自动忽略

---

## 📊 输出示例

### 基本输出
```
============================================================
🔍 筛选 JSON 实例
============================================================
JSON 文件: results_analysis/ship/json/000000_prediction.json
保留实例 ID: [0, 1, 2]

✓ 已备份原文件到: results_analysis/ship/json/000000_prediction.json.bak
✓ JSON 已更新:
   原始实例数: 14
   保留实例数: 3
   删除实例数: 11

保留的实例:
  [0] ship=95 (score: 0.954, bbox: [1221.0, 60.0, 268.0, 369.0])
  [1] ship=92 (score: 0.917, bbox: [543.0, 1338.0, 234.0, 161.0])
  [2] ship=65 (score: 0.653, bbox: [278.0, 0.0, 352.0, 412.0])

============================================================
🔄 重新生成可视化输出...
============================================================
[可视化生成输出...]

============================================================
✅ 完成！
============================================================
✓ JSON 已更新: results_analysis/ship/json/000000_prediction.json
✓ 可视化已重新生成
✓ 原文件已备份: results_analysis/ship/json/000000_prediction.json.bak
```

---

## 💡 使用场景

### 场景 1：删除误检实例
```bash
# JSON 中有 14 个实例，但只有前 3 个是正确的
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 1 2
```

### 场景 2：保留高置信度实例
```bash
# 先查看 JSON，确定哪些是高置信度的
# 然后只保留这些实例

# 例如：只保留 ship=95, ship=92, ship=91（索引 0, 1, 2）
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 1 2
```

### 场景 3：手动筛选特定实例
```bash
# 查看 JSON 文件，确定要保留的索引
# 然后运行脚本

python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 3 5 7 9
```

### 场景 4：批量处理
```bash
# PowerShell
$jsonFiles = Get-ChildItem results_analysis\ship\json\*.json
foreach ($json in $jsonFiles) {
    python scripts/filter_json_by_ids.py $json.FullName 0 1 2
}

# Linux/Mac Bash
for json in results_analysis/ship/json/*.json; do
    python scripts/filter_json_by_ids.py "$json" 0 1 2
done
```

---

## ⚙️ 命令行参数

### 必需参数
- `json_path`: JSON 文件路径
- `ids`: 要保留的实例索引（从0开始，可指定多个）

### 可选参数
- `--regenerate`: 重新生成可视化（默认启用）
- `--no-regenerate`: 不重新生成可视化
- `--no-backup`: 不备份原文件
- `--score-thr`: 传递给 `regenerate_from_json.py` 的置信度阈值（默认 0.0）

### 查看帮助
```bash
python scripts/filter_json_by_ids.py --help
```

---

## 🔍 技术细节

### 实例索引
- JSON 文件是一个列表，索引从 0 开始
- 例如：`[{"id": 0}, {"id": 1}, {"id": 2}]` 中：
  - 索引 0 对应第一个实例
  - 索引 1 对应第二个实例
  - 索引 2 对应第三个实例

### 备份文件
- 备份文件格式：`原文件名.json.bak`
- 例如：`000000_prediction.json` → `000000_prediction.json.bak`

### 重新生成
- 自动调用 `regenerate_from_json.py` 重新生成：
  - `binary_masks/` - 二值掩码
  - `instances/` - 实例图像
  - `predictions/` - 预测可视化

---

## 📝 注意事项

### 1. 索引有效性
- 索引必须是有效的（0 到 `len(instances)-1`）
- 无效的索引会被自动忽略
- 如果所有索引都无效，脚本会退出并提示

### 2. 备份文件
- 默认会备份原文件
- 备份文件可以帮助恢复原始数据
- 使用 `--no-backup` 可以禁用备份

### 3. 可视化重新生成
- 默认会自动重新生成可视化
- 如果只想更新 JSON，使用 `--no-regenerate`
- 重新生成会调用 `regenerate_from_json.py`，需要确保该脚本可用

### 4. JSON 格式
- JSON 文件必须是一个实例列表
- 每个实例必须包含 `category_name`, `score`, `bbox`, `segmentation` 等字段

---

## 🎯 完整示例

### 示例 1：保留前 3 个实例
```bash
cd .

python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 1 2
```

**结果**：
- JSON 只保留索引 0, 1, 2 的实例
- 自动备份原文件
- 自动重新生成所有可视化

### 示例 2：保留指定实例（不重新生成）
```bash
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 1 2 \
  --no-regenerate
```

**结果**：
- JSON 只保留索引 0, 1, 2 的实例
- 自动备份原文件
- **不**重新生成可视化

### 示例 3：查看 JSON 后手动筛选
```bash
# 1. 先查看 JSON 文件，确定要保留的索引
cat results_analysis/ship/json/000000_prediction.json | python -m json.tool

# 2. 假设要保留索引 0, 2, 4
python scripts/filter_json_by_ids.py \
  results_analysis/ship/json/000000_prediction.json \
  0 2 4
```

---

## 🔄 与 regenerate_from_json.py 的关系

本脚本会调用 `regenerate_from_json.py` 来重新生成可视化：

```bash
python scripts/regenerate_from_json.py \
  <json_path> \
  --score_thr 0.0
```

因此，重新生成的可视化会基于筛选后的 JSON 文件。

---

## 📚 相关文档

- **JSON 格式说明**：`JSON_CATEGORY_NAME_UPDATE.md`
- **重新生成可视化**：`scripts/REGENERATE_FROM_JSON.md`
- **脚本源码**：`scripts/filter_json_by_ids.py`

---

## 🆕 更新日志

### v1.0 (2025-11-05)
- ✨ 初始版本
- ✅ 支持从 JSON 筛选指定 ID 的实例
- ✅ 自动备份原文件
- ✅ 自动重新生成可视化
- ✅ 完整的错误处理
- ✅ 详细的输出信息

