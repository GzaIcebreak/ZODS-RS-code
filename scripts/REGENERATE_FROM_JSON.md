# 从 JSON 重新生成可视化输出

## 🆕 功能说明

`scripts/regenerate_from_json.py` 可以根据 JSON 预测结果文件重新生成所有可视化输出，包括：
- ✅ **二值掩码** (`binary_masks/`)
- ✅ **实例图像** (`instances/`)
- ✅ **预测可视化** (`predictions/`)

---

## 📋 使用方法

### 基本用法
```bash
python scripts/regenerate_from_json.py <json_path>
```

### 指定图像路径
```bash
python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json --img_path data/FAR1M/Ship/images/000000.jpg
```

### 指定输出目录
```bash
python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json --output_dir ./regenerated_output
```

### 应用置信度阈值（匹配原始输出）
```bash
# 使用与原始推理相同的阈值（如 0.3）
python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json --score_thr 0.3
```

### 完整示例
```bash
cd .

python scripts/regenerate_from_json.py \
  results_analysis/ship/json/000000_prediction.json \
  --img_path data/FAR1M/Ship/images/000000.jpg \
  --output_dir ./regenerated_output
```

---

## 🚀 功能特性

### 自动图像查找
如果不提供 `--img_path`，脚本会自动在以下目录查找：
- `data/FAR1M/Ship/images/`
- `results_analysis/ship/`
- JSON 文件所在目录的父目录

### 支持的输出格式
- **二值掩码**：PNG 灰度图（0/255）
- **实例图像**：PNG RGB 图（带边界框和标签）
- **预测可视化**：PNG RGB 图（所有实例叠加）

---

## 📊 输出示例

### 运行输出
```
📖 加载 JSON 文件: results_analysis/ship/json/000000_prediction.json
✓ 找到 14 个实例
📷 使用图像: data/FAR1M/Ship/images/000000.jpg
📁 输出目录: results_analysis/ship

============================================================
🔄 开始重新生成可视化输出
============================================================

1️⃣  生成二值掩码...
✓ 保存了 14 个二值掩码到: results_analysis/ship/binary_masks/000000

2️⃣  生成实例图像...
✓ 保存了 14 个实例图像到: results_analysis/ship/instances/000000

3️⃣  生成预测可视化...
✓ 保存了预测可视化图像到: results_analysis/ship/predictions/000000_prediction.png

============================================================
✅ 重新生成完成！
============================================================
✓ 二值掩码: results_analysis/ship/binary_masks/000000
✓ 实例图像: results_analysis/ship/instances/000000
✓ 预测可视化: results_analysis/ship/predictions/000000_prediction.png
```

---

## 💡 使用场景

### 场景 1：恢复丢失的可视化文件
```bash
# JSON 文件还在，但可视化文件丢失了
python scripts/regenerate_from_json.py results_analysis/ship/json/000000_prediction.json
```

### 场景 2：批量重新生成
```bash
# Windows PowerShell
$jsonFiles = Get-ChildItem results_analysis\ship\json\*.json
foreach ($json in $jsonFiles) {
    python scripts/regenerate_from_json.py $json.FullName
}

# Linux/Mac Bash
for json in results_analysis/ship/json/*.json; do
    python scripts/regenerate_from_json.py "$json"
done
```

### 场景 3：生成到不同目录
```bash
python scripts/regenerate_from_json.py \
  results_analysis/ship/json/000000_prediction.json \
  --output_dir ./backup_output
```

### 场景 4：使用不同的图像
```bash
python scripts/regenerate_from_json.py \
  results_analysis/ship/json/000000_prediction.json \
  --img_path ./different_image.jpg
```

---

## ⚙️ 命令行参数

### 必需参数
- `json_path`: JSON 预测结果文件路径

### 可选参数
- `--img_path`: 原始图像路径（如果不提供会自动查找）
- `--output_dir`: 输出目录（默认：`results_analysis/ship`）
- `--class_names`: 类别名称列表（默认：`['ship']`）
- `--score_thr`: 置信度阈值（默认：`0.0`，保存所有实例。要与原始输出匹配，请使用 `0.3`）

### 查看帮助
```bash
python scripts/regenerate_from_json.py --help
```

---

## 🔍 技术细节

### JSON 格式要求
JSON 文件应包含一个实例列表，每个实例包含：
```json
{
  "image_id": 1001,
  "category_id": 0,
  "category_name": "ship=90",
  "bbox": [x, y, w, h],
  "score": 0.9,
  "segmentation": {
    "size": [height, width],
    "counts": "RLE编码字符串"
  },
  "obb": [x1, y1, x2, y2, x3, y3, x4, y4]
}
```

### RLE 解码
使用 `pycocotools.mask.decode()` 解码 RLE 格式的 segmentation：
```python
mask = mask_utils.decode(segmentation)
```

### 二值掩码生成
```python
mask_binary = (mask > 0).astype(np.uint8) * 255
mask_img = Image.fromarray(mask_binary, mode='L')
```

### 实例可视化
- 使用 `cv2.drawContours()` 绘制轮廓
- 使用 `cv2.rectangle()` 绘制边界框
- 使用 `cv2.putText()` 绘制标签

### 预测可视化
- 优先使用 `vis_coco()` 生成专业可视化
- 如果不可用，使用简单的掩码叠加

---

## ⚠️ 重要：置信度阈值

### 问题说明
**JSON 文件包含所有实例**（包括低置信度的），但**原始可视化输出只保存高置信度实例**（默认 `vis_thr=0.3`）。

### 解决方案

#### 选项 1：匹配原始输出（推荐）
```bash
# 使用与原始推理相同的阈值（从配置文件读取，通常是 0.3）
python scripts/regenerate_from_json.py \
  results_analysis/ship/json/000000_prediction.json \
  --score_thr 0.3
```

#### 选项 2：保存所有实例
```bash
# 默认保存所有实例（包括低置信度的）
python scripts/regenerate_from_json.py \
  results_analysis/ship/json/000000_prediction.json
```

### 如何确定阈值
查看配置文件 `zods_rs/pl_configs/ship_dinov3.yaml`：
```yaml
model:
  vis_thr: 0.3  # 这就是原始阈值
```

---

## 📝 注意事项

### 1. 依赖库
需要安装以下依赖：
```bash
pip install pycocotools pillow numpy opencv-python
```

### 2. 置信度阈值
- **默认值**：`0.0`（保存所有实例）
- **匹配原始输出**：使用 `--score_thr 0.3`（或配置文件中的 `vis_thr` 值）
- JSON 文件包含所有实例，但可视化输出只保存高置信度的

### 3. 图像路径
- 如果自动查找失败，请使用 `--img_path` 指定
- 支持常见的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

### 4. 输出目录
- 如果输出目录已存在，文件会被覆盖
- 建议使用 `--output_dir` 指定不同的输出目录以避免覆盖

### 5. 类别名称
- 默认类别名称从 `category_name` 字段解析（如 `"ship=90"`）
- 可以使用 `--class_names` 指定类别名称列表

---

## 🎯 完整示例

### 示例 1：单文件重新生成
```bash
cd .

python scripts/regenerate_from_json.py \
  results_analysis/ship/json/000000_prediction.json
```

### 示例 2：批量重新生成
```bash
# PowerShell
Get-ChildItem results_analysis\ship\json\*.json | ForEach-Object {
    Write-Host "处理: $($_.Name)"
    python scripts/regenerate_from_json.py $_.FullName
}
```

### 示例 3：自定义输出目录
```bash
python scripts/regenerate_from_json.py \
  results_analysis/ship/json/000000_prediction.json \
  --output_dir ./regenerated_output \
  --img_path data/FAR1M/Ship/images/000000.jpg
```

---

## 🔄 与原始输出的对比

### 原始输出（推理时生成）
- 由 `Sam2MatchingBaseline_noAMG` 在推理时自动生成
- 包含完整的推理过程信息

### 重新生成输出（从 JSON）
- 只根据 JSON 文件重新生成可视化
- 不包含推理过程信息
- 可以用于恢复丢失的文件或生成不同格式的输出

---

## 📚 相关文档

- **JSON 格式说明**：`JSON_CATEGORY_NAME_UPDATE.md`
- **可视化函数**：`zods_rs/dataset/visualization.py`
- **脚本源码**：`scripts/regenerate_from_json.py`

---

## 🆕 更新日志

### v1.0 (2025-11-05)
- ✨ 初始版本
- ✅ 支持从 JSON 重新生成二值掩码
- ✅ 支持从 JSON 重新生成实例图像
- ✅ 支持从 JSON 重新生成预测可视化
- ✅ 自动图像路径查找
- ✅ 完整的错误处理

