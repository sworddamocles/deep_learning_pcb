# 颜色过拟合测试工具使用说明

本文档说明如何使用 test_color_overfit.py 对检测模型进行颜色鲁棒性评估。

## 1. 工具作用

test_color_overfit.py 会对同一批图像施加多种颜色扰动，再与原图检测结果做一致性对比，用于判断模型是否对颜色过于敏感。

扰动类型包括：
- gray3: 灰度三通道
- low_saturation: 低饱和度
- hue_shift: 色相偏移
- dark: 变暗
- bright: 变亮
- warm_cast: 暖色偏置

## 2. 运行前准备

需要准备：
- 训练完成的权重文件（.pt）
- 测试图像目录（建议使用验证集）

示例路径：
- 模型: runs/train/yolo11_cbam_SimAM_color_robust_1024/weights/best.pt
- 图像目录: D:/deep_learning/ultralytics-8.3.163/dataset_split/images/val

## 3. 基本运行命令

```bash
python test_color_overfit.py --model runs/train/yolo11_cbam_SimAM_color_robust_1024/weights/best.pt --source D:/deep_learning/ultralytics-8.3.163/dataset_split/images/val --imgsz 1024 --conf 0.25 --iou 0.5 --max-images 300 --device 0
```

如果在 CPU 上测试：

```bash
python test_color_overfit.py --model runs/train/yolo11_cbam_SimAM_color_robust_1024/weights/best.pt --source D:/deep_learning/ultralytics-8.3.163/dataset_split/images/val --device cpu
```

## 4. 参数说明

- --model: 模型权重路径
- --source: 图像来源，可为单图、目录或 glob
- --imgsz: 推理尺寸，建议与训练一致
- --conf: 置信度阈值
- --iou: NMS IoU 阈值
- --device: 推理设备，如 0、1 或 cpu
- --max-images: 最多测试图像数量，0 表示全部
- --save-dir: 结果输出目录，默认 runs/color_overfit_test

## 5. 输出文件

脚本会生成：
- color_overfit_detail.csv: 每张图、每种扰动的详细指标
- color_overfit_summary.json: 每种扰动的均值统计

默认输出目录：
- runs/color_overfit_test

## 6. 指标解读

关键指标：
- match_ratio: 扰动图与原图的匹配比例，越高越好
- conf_ratio: 扰动后置信度与原图的比值，越接近 1 越好
- count_ratio: 扰动后检测数量与原图比值，越接近 1 越好
- score: 综合分数，越高越好

经验判断：
- global score >= 0.80: 颜色鲁棒性较好
- 0.65 <= global score < 0.80: 中等，可继续优化
- global score < 0.65: 颜色过拟合风险较高

## 7. 推荐评估流程

1. 用旧模型跑一次，保存 summary.json。
2. 用新模型跑同一批图像，保持参数一致。
3. 对比每个扰动的 score_mean 与 global score。
4. 若 gray3、hue_shift 明显偏低，通常说明颜色依赖仍然存在。

## 8. 常见问题

1. 报错 No images found from source
- 检查 --source 路径是否正确
- Windows 路径建议使用 / 或使用引号包裹

2. 报错 No valid predictions collected
- 检查 --model 是否正确
- 降低 --conf，例如改为 0.1

3. 推理太慢
- 减小 --max-images
- 使用 GPU 设备，如 --device 0
- 适当降低 --imgsz
