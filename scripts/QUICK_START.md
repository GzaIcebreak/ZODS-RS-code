# 快速开始：process_and_test.py

## 🚀 一行命令完成所有操作

```bash
python scripts/process_and_test.py <图片路径>
```

## 📋 完整示例

```bash
# 1. 进入项目目录
cd .

# 2. 运行脚本（指定图片路径）
python scripts/process_and_test.py path\to\data\my_ship.jpg
```

**脚本会自动**：
1. ✅ 读取图片尺寸
2. ✅ 复制图片到 `data/FAR1M/Ship/images/`
3. ✅ 更新 `custom_targets.json`
4. ✅ 执行推理命令

## 📁 输出位置

推理结果保存在：
```
results_analysis/ship/
├── predictions/          # 可视化图
├── json/                 # JSON 结果
├── instances/            # 单实例可视化
└── binary_masks/         # 二值掩码 🆕
```

## 🔍 验证步骤

### 1. 检查图片是否复制成功
```bash
dir data\FAR1M\Ship\images\my_ship.jpg
```

### 2. 检查 JSON 是否更新
```bash
type data\FAR1M\Ship\annotations\custom_targets.json
```

应该看到：
```json
{
  "images": [
    { "id": 1001, "file_name": "my_ship.jpg", "width": 2048, "height": 1536 }
  ]
}
```

### 3. 查看推理结果
```bash
dir results_analysis\ship\binary_masks\my_ship\
```

## ❓ 常见问题

**Q: 图片路径包含空格怎么办？**
```bash
# Windows PowerShell
python scripts/process_and_test.py "path\to\data\my ship.jpg"

# Linux/Mac
python scripts/process_and_test.py "/path/to/my ship.jpg"
```

**Q: 如何批量处理？**
```bash
# Windows PowerShell
foreach ($img in @("img1.jpg", "img2.jpg", "img3.jpg")) {
    python scripts/process_and_test.py "path\to\data\$img"
}
```

**Q: 只想更新 JSON，不执行推理？**
编辑脚本，注释掉 `run_inference()` 调用即可。

## 📚 详细文档

完整文档请参考：`scripts/PROCESS_AND_TEST_README.md`

