# 自动处理图片并执行推理脚本

## 功能说明

`scripts/process_and_test.py` 是一个自动化脚本，用于：
1. ✅ 将输入图片复制到 `data/FAR1M/Ship/images/` 目录
2. ✅ 自动读取图片尺寸（width × height）
3. ✅ 更新 `custom_targets.json` 中的图片信息
4. ✅ 自动执行推理命令

---

## 使用方法

### 基本用法
```bash
cd .

python scripts/process_and_test.py <图片路径>
```

### 示例

#### Windows 路径
```bash
python scripts/process_and_test.py path\to\data\my_ship.jpg
python scripts/process_and_test.py D:\test_images\ship.png
```

#### 相对路径
```bash
python scripts/process_and_test.py ./test_images/ship.jpg
python scripts/process_and_test.py ../data/my_ship.png
```

#### Linux/Mac 路径
```bash
python scripts/process_and_test.py /home/user/images/ship.jpg
python scripts/process_and_test.py ~/Pictures/my_ship.png
```

---

## 执行流程

### 1. 读取图片尺寸
```python
# 自动从图片文件中读取 width 和 height
width, height = get_image_size(image_path)
```

### 2. 复制图片
```bash
# 将图片复制到目标目录，保持原文件名
源文件: path\to\data\my_ship.jpg
目标文件: zods-rs/data/FAR1M/Ship/images/my_ship.jpg
```

### 3. 更新 JSON
**更新前** (`custom_targets.json`):
```json
{
  "images": [
    { "id": 1001, "file_name": "13.jpg", "width": 1500, "height": 1500 }
  ]
}
```

**更新后**:
```json
{
  "images": [
    { "id": 1001, "file_name": "my_ship.jpg", "width": 2048, "height": 1536 }
  ]
}
```

### 4. 执行推理
```bash
python -X utf8 run_lightening.py test \
  --config zods_rs/pl_configs/ship_dinov3.yaml \
  --model.test_mode=test \
  --ckpt_path ./tmp_ckpts/ship/ship_refs_memory_postprocessed.pth
```

---

## 输出示例

```
============================================================
📋 自动处理图片并执行推理
============================================================
输入图片: path\to\data\my_ship.jpg
目标目录: .\data\FAR1M\Ship\images
JSON 文件: .\data\FAR1M\Ship\annotations\custom_targets.json

📏 读取图片尺寸...
✓ 图片尺寸: 2048 × 1536

📁 复制图片到目标目录...
✓ 图片已复制到: .\data\FAR1M\Ship\images\my_ship.jpg

✏️  更新 JSON 文件...
✓ JSON 已更新: file_name=my_ship.jpg, width=2048, height=1536

============================================================
🚀 开始执行推理...
============================================================
命令: python -X utf8 run_lightening.py test --config zods_rs/pl_configs/ship_dinov3.yaml --model.test_mode=test --ckpt_path ./tmp_ckpts/ship/ship_refs_memory_postprocessed.pth

工作目录: .

[推理输出...]

============================================================
✅ 推理完成！
============================================================

============================================================
📊 处理完成
============================================================
✓ 图片已保存: .\data\FAR1M\Ship\images\my_ship.jpg
✓ JSON 已更新: .\data\FAR1M\Ship\annotations\custom_targets.json
✓ 推理结果保存在: .\results_analysis\ship
```

---

## 错误处理

### 1. 图片不存在
```
❌ 错误: 图片不存在: path\to\data\my_ship.jpg
```

### 2. 无法读取图片尺寸
```
❌ 错误: 无法读取图片尺寸: [Errno 2] No such file or directory
```

### 3. JSON 文件不存在
```
❌ 错误: JSON 文件不存在: .../custom_targets.json
```

---

## 注意事项

### 1. 文件覆盖
- 如果目标目录中已存在同名文件，**会被覆盖**
- 建议使用唯一的文件名

### 2. JSON 格式
- 脚本会自动保持 JSON 的格式和缩进
- 如果 `images` 数组为空，会自动创建第一个元素
- `id` 字段会自动设置为 `1001`（如果不存在）

### 3. 工作目录
- 脚本会自动切换到 `zods-rs` 目录执行推理
- 推理完成后会恢复原目录

### 4. 图片格式支持
支持所有 PIL 支持的格式：
- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP
- 等...

---

## 高级用法

### 批量处理（使用循环）
```bash
# Windows PowerShell
$images = @("image1.jpg", "image2.jpg", "image3.jpg")
foreach ($img in $images) {
    python scripts/process_and_test.py "path\to\data\$img"
}

# Linux/Mac Bash
for img in image1.jpg image2.jpg image3.jpg; do
    python scripts/process_and_test.py "./data/$img"
done
```

### 与其他脚本集成
```python
import subprocess
import sys

def process_image(image_path):
    """在其他脚本中调用"""
    result = subprocess.run(
        [sys.executable, "scripts/process_and_test.py", image_path],
        cwd="zods-rs"
    )
    return result.returncode == 0

# 使用
if process_image("path/to/data/my_ship.jpg"):
    print("处理成功！")
```

---

## 技术细节

### 文件位置
- **脚本**：`zods-rs/scripts/process_and_test.py`
- **目标图片目录**：`zods-rs/data/FAR1M/Ship/images/`
- **JSON 文件**：`zods-rs/data/FAR1M/Ship/annotations/custom_targets.json`
- **推理结果**：`zods-rs/results_analysis/ship/`

### 依赖库
- `PIL` (Pillow) - 图片处理
- `json` - JSON 文件操作
- `shutil` - 文件复制
- `subprocess` - 命令执行
- `pathlib` - 路径处理

### 函数说明

#### `get_image_size(image_path) -> Tuple[int, int]`
读取图片尺寸，返回 `(width, height)`

#### `copy_image_to_target(src_path, target_dir) -> str`
复制图片到目标目录，返回目标路径

#### `update_custom_targets_json(json_path, file_name, width, height)`
更新 JSON 文件中的图片信息

#### `run_inference(work_dir=None)`
执行推理命令

---

## 常见问题

### Q1: 如何修改目标目录？
编辑脚本中的 `images_dir` 变量：
```python
images_dir = script_dir / "data" / "FAR1M" / "Ship" / "images"
```

### Q2: 如何修改推理命令？
编辑 `run_inference()` 函数中的 `cmd` 列表：
```python
cmd = [
    "python", "-X", "utf8", "run_lightening.py", "test",
    "--config", "zods_rs/pl_configs/ship_dinov3.yaml",
    # ... 其他参数
]
```

### Q3: 如何跳过推理步骤？
修改脚本，注释掉 `run_inference()` 调用：
```python
# return_code = run_inference(work_dir=str(script_dir))
print("推理步骤已跳过")
```

### Q4: 如何批量处理多个图片？
使用循环或创建批处理脚本（见"高级用法"部分）

---

## 更新日志

### v1.0 (2025-11-05)
- ✨ 初始版本
- ✅ 支持图片复制和 JSON 更新
- ✅ 自动执行推理命令
- ✅ 完整的错误处理
- ✅ 友好的输出信息

