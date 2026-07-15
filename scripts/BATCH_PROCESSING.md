# 批处理功能说明

## 🆕 新增批处理功能

`process_and_test.py` 现在支持**批处理模式**，可以一次性处理多张图片！

---

## 📋 使用方法

### 1. 单张图片（原有功能）
```bash
python scripts/process_and_test.py image.jpg
```

### 2. 多张图片（新功能）
```bash
python scripts/process_and_test.py img1.jpg img2.jpg img3.jpg
```

### 3. 目录批处理（新功能）
```bash
# 处理目录下所有图片
python scripts/process_and_test.py --dir ./images

# Windows 路径
python scripts/process_and_test.py --dir path\to\data\images
```

### 4. 快速模式（只处理不推理）
```bash
# 只复制图片和更新JSON，不执行推理
python scripts/process_and_test.py --dir ./images --no-inference
```

---

## 🚀 功能特性

### 批处理模式特性
- ✅ **自动识别图片格式**：支持 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`
- ✅ **递归搜索**：自动查找子目录中的所有图片
- ✅ **进度显示**：显示 `[当前/总数]` 进度
- ✅ **错误处理**：单张失败不影响其他图片
- ✅ **统计报告**：处理完成后显示成功/失败统计

### 快速模式特性
- ✅ **跳过推理**：只处理图片和更新JSON，大幅提升速度
- ✅ **批量准备**：适合批量准备数据后统一推理

---

## 📊 输出示例

### 单张图片模式
```
============================================================
📋 处理图片: my_ship.jpg
============================================================
📏 读取图片尺寸...
✓ 图片尺寸: 2048 × 1536

📁 复制图片到目标目录...
✓ 图片已复制到: .../images/my_ship.jpg

✏️  更新 JSON 文件...
✓ JSON 已更新: file_name=my_ship.jpg, width=2048, height=1536

🚀 开始执行推理...
[推理输出...]
✅ 推理完成！
```

### 批处理模式
```
============================================================
📦 批处理模式: 5 张图片
============================================================

[1/5] 处理: img1.jpg
------------------------------------------------------------
✅ [1/5] 成功: img1.jpg

[2/5] 处理: img2.jpg
------------------------------------------------------------
✅ [2/5] 成功: img2.jpg

[3/5] 处理: img3.jpg
------------------------------------------------------------
✅ [3/5] 成功: img3.jpg

[4/5] 处理: img4.jpg
------------------------------------------------------------
❌ [4/5] 失败: img4.jpg
   错误: 文件不存在: img4.jpg

[5/5] 处理: img5.jpg
------------------------------------------------------------
✅ [5/5] 成功: img5.jpg

============================================================
📊 批处理统计
============================================================
总计: 5 张图片
✅ 成功: 4
❌ 失败: 1

失败的文件:
  - img4.jpg: 文件不存在: img4.jpg
============================================================
```

---

## 💡 使用场景

### 场景 1：批量处理测试图片
```bash
# 准备测试图片目录
mkdir test_images
# 复制测试图片到 test_images/

# 批量处理
python scripts/process_and_test.py --dir test_images
```

### 场景 2：快速准备数据（不推理）
```bash
# 只复制图片和更新JSON，不执行推理
python scripts/process_and_test.py --dir ./raw_images --no-inference

# 之后可以手动执行推理
python run_lightening.py test \
  --config zods_rs/pl_configs/ship_dinov3.yaml \
  --model.test_mode=test \
  --ckpt_path ./tmp_ckpts/ship/ship_refs_memory_postprocessed.pth
```

### 场景 3：混合路径处理
```bash
# 可以混合文件和目录
python scripts/process_and_test.py \
  img1.jpg \
  img2.jpg \
  ./subdir1 \
  ./subdir2/img3.jpg
```

### 场景 4：从不同目录收集图片
```bash
# Windows PowerShell
python scripts/process_and_test.py \
  D:\images\*.jpg \
  E:\test\images \
  F:\data\*.png
```

---

## ⚙️ 命令行参数

### 基本参数
```bash
python scripts/process_and_test.py [图片路径...]
```

### 可选参数

#### `--dir <目录路径>`
批处理模式：处理目录下所有图片（递归搜索）

```bash
python scripts/process_and_test.py --dir ./images
```

#### `--no-inference`
只处理图片和更新JSON，不执行推理（快速模式）

```bash
python scripts/process_and_test.py --dir ./images --no-inference
```

### 查看帮助
```bash
python scripts/process_and_test.py --help
```

---

## 🔍 技术细节

### 图片格式支持
- JPEG: `.jpg`, `.jpeg`
- PNG: `.png`
- BMP: `.bmp`
- TIFF: `.tiff`, `.tif`
- WebP: `.webp`

### 路径处理
- **绝对路径**：直接使用
- **相对路径**：相对于当前工作目录
- **通配符**：支持 `*.jpg` 等（取决于 shell）

### 批处理流程
1. **收集图片**：从输入路径收集所有图片文件
2. **去重排序**：自动去重并按文件名排序
3. **逐个处理**：按顺序处理每张图片
4. **统计报告**：显示成功/失败统计

### 错误处理
- ✅ 单张图片失败不影响其他图片
- ✅ 自动跳过不存在的路径
- ✅ 自动跳过非图片文件
- ✅ 记录所有失败信息

---

## 📝 注意事项

### 1. JSON 文件覆盖
- 每张图片处理时都会更新 `custom_targets.json`
- **最后一张图片的信息会保留在JSON中**
- 如果需要保存每张图片的JSON，需要手动备份

### 2. 推理执行
- 批处理模式下，**每张图片都会执行一次推理**
- 如果图片很多，这会非常耗时
- 建议使用 `--no-inference` 先准备数据，然后统一推理

### 3. 文件覆盖
- 如果目标目录中已存在同名文件，**会被覆盖**
- 建议使用唯一的文件名

### 4. 内存使用
- 批处理模式不会一次性加载所有图片到内存
- 每张图片处理完后会释放资源

---

## 🎯 性能优化建议

### 快速模式
```bash
# 先批量准备数据（不推理）
python scripts/process_and_test.py --dir ./images --no-inference

# 然后统一执行推理（只处理一张图片）
python scripts/process_and_test.py ./images/000001.jpg
```

### 批量推理
如果确实需要批量推理，建议：
1. 使用较小的批次（如 10-20 张）
2. 监控 GPU 内存使用
3. 考虑使用多进程（需要自定义脚本）

---

## 📚 完整示例

### 示例 1：批量处理目录
```bash
# 处理 images 目录下所有图片
python scripts/process_and_test.py --dir ./images

# 输出：
# ============================================================
# 📦 批处理模式: 10 张图片
# ============================================================
# [1/10] 处理: img001.jpg
# ✅ [1/10] 成功: img001.jpg
# ...
```

### 示例 2：快速准备数据
```bash
# 只复制和更新JSON，不推理
python scripts/process_and_test.py --dir ./raw_images --no-inference

# 输出：
# ============================================================
# 📦 批处理模式: 50 张图片
# ============================================================
# [1/50] 处理: img001.jpg
# ✅ [1/50] 成功: img001.jpg
# ...
# ============================================================
# 📊 批处理统计
# ============================================================
# 总计: 50 张图片
# ✅ 成功: 50
# ❌ 失败: 0
```

### 示例 3：混合路径
```bash
python scripts/process_and_test.py \
  img1.jpg \
  img2.jpg \
  ./subdir1 \
  ./subdir2/img3.jpg
```

---

## 🔄 向后兼容性

- ✅ **完全兼容**：原有的单图片模式仍然可用
- ✅ **参数兼容**：原有命令格式不变
- ✅ **功能增强**：新增批处理功能不影响现有使用

---

## 🆕 更新日志

### v2.0 (2025-11-05)
- ✨ 新增批处理模式
- ✨ 支持目录递归搜索
- ✨ 新增 `--dir` 参数
- ✨ 新增 `--no-inference` 快速模式
- ✅ 改进错误处理和统计报告
- ✅ 改进进度显示

### v1.0 (2025-11-05)
- ✨ 初始版本
- ✅ 单图片处理
- ✅ 自动推理

---

## 📖 相关文档

- **完整使用文档**：`scripts/PROCESS_AND_TEST_README.md`
- **快速开始**：`scripts/QUICK_START.md`
- **脚本源码**：`scripts/process_and_test.py`

