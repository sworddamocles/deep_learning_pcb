import torch

from ultralytics import YOLO

if __name__ == "__main__":
    # 检查GPU可用性
    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"使用设备: {device} (GPU内存: {gpu_memory:.1f}GB)")

    # 加载模型
    model = YOLO(r"D:\deep_learning\ultralytics-8.3.163\yolo11_cbam_SimAM.yaml")

    # 优化的训练参数
    model.train(
        # ==================== 核心训练参数 ====================
        data=r"D:\deep_learning\ultralytics-8.3.163\dataset_split\data.yaml",
        epochs=350,  # 基于您之前的300epochs，效果不错
        imgsz=1024,  # 保持640，RTX 3050 4GB内存友好
        # ==================== 硬件适配参数 ====================
        batch=4,  # 从8降低到4，避免OOM（Out of Memory）
        workers=0,  # 从4降低到2，减少CPU负担
        cache="ram",  # 使用RAM缓存，速度比disk快
        # ==================== 优化器与学习率 ====================
        optimizer="AdamW",  # 改为AdamW，对小目标检测更友好
        lr0=0.001,  # 降低初始学习率，避免震荡
        lrf=0.01,  # 最终学习率为初始的1%
        momentum=0.937,
        weight_decay=0.0001,  # 降低权重衰减
        # ==================== 学习率调度 ====================
        cos_lr=True,  # 启用余弦退火，提高收敛性
        warmup_epochs=3,  # 减少预热epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # ==================== 损失函数优化 ====================
        box=5.0,  # 降低边界框损失权重
        cls=1.0,  # 提高分类损失权重，解决spur/spurious_copper混淆
        dfl=1.0,  # 保持DFL损失权重
        # ==================== PCB缺陷专属数据增强 ====================
        augment=True,
        hsv_h=0.01,  # 轻微色调变化
        hsv_s=0.5,  # 中度饱和度增强
        hsv_v=0.3,  # 轻微明度变化
        degrees=2.0,  # 减小旋转角度，PCB板通常水平放置
        translate=0.05,  # 减小平移
        scale=0.2,  # 减小缩放范围
        shear=0.0,
        perspective=0.0,
        flipud=0.0,  # 关闭上下翻转
        fliplr=0.5,  # 保持左右翻转
        mosaic=0.8,  # 降低mosaic强度
        mixup=0.1,  # 启用少量mixup
        # ==================== 小目标检测优化 ====================
        # copy_paste=0.1,  # 小目标复制粘贴增强
        # erasing=0.1,  # 随机擦除
        # crop_fraction=0.8,  # 裁剪保留比例
        # ==================== 训练控制策略 ====================
        device=device,
        patience=30,  # 设置早停耐心值
        close_mosaic=15,  # 最后15个epoch关闭mosaic
        # ==================== 验证与保存 ====================
        val=True,
        save=True,
        save_period=20,  # 每20个epoch保存一次
        save_json=True,  # 保存验证结果JSON
        # ==================== 项目管理 ====================
        project="runs/train",
        name="yolo11_cbam_SimAM_optimized2_1024_2",  # 新实验名称
        exist_ok=True,
        pretrained=True,
        resume=False,
        # ==================== 训练优化 ====================
        amp=True,  # 保持混合精度训练
        overlap_mask=False,
        mask_ratio=4,
        dropout=0.0,
        # ==================== 输出与监控 ====================
        verbose=True,
        plots=True,
        deterministic=True,  # 启用确定性训练
    )
