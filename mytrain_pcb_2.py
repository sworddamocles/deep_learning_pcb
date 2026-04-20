from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # 检查GPU可用性
    device = 0 if torch.cuda.is_available() else 'cpu'
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"使用设备: {device} (GPU内存: {gpu_memory:.1f}GB)")

    # 加载模型
    model = YOLO(r"D:\deep_learning\ultralytics-8.3.163\yolo11_cbam_SimAM.yaml")

    # 优化的训练参数（重点修复颜色过拟合）
    model.train(
        # ==================== 核心训练参数 ====================
        data=r"D:\deep_learning\ultralytics-8.3.163\dataset_split\data.yaml",
        epochs=500,
        imgsz=1024,
        
        # ==================== 硬件适配参数 ====================
        batch=32,
        workers=8,
        cache='ram',
        
        # ==================== 优化器与学习率 ====================
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,  # 提高正则，缓解过拟合
        nbs=64,  # 显式设置标称batch，稳定损失与正则缩放
        
        # ==================== 学习率调度 ====================
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        
        # ==================== 损失函数优化 ====================
        box=5.0,
        cls=1.0,
        dfl=1.0,
        
        # ==================== PCB缺陷专属数据增强 ====================
        # 关键思路：增大颜色扰动，让模型更多学习几何边缘和纹理，而不是固定色彩分布
        augment=True,
        hsv_h=0.06,  # 明显色相扰动
        hsv_s=0.85,  # 强饱和度扰动
        hsv_v=0.55,  # 强亮度扰动
        bgr=0.20,  # 20%概率交换通道，抑制颜色依赖
        degrees=2.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.6,
        mixup=0.05,
        copy_paste=0.15,
        cutmix=0.10,
        copy_paste_mode="flip",
        
        # ==================== 训练控制策略 ====================
        device=device,
        patience=80,
        close_mosaic=20,
        
        # ==================== 验证与保存 ====================
        val=True,
        save=True,
        save_period=20,
        save_json=True,
        
        # ==================== 项目管理 ====================
        project="runs/train",
        name="yolo11_cbam_SimAM_color_robust_1024",
        exist_ok=True,
        pretrained=True,
        resume=False,
        
        # ==================== 训练优化 ====================
        amp=True,
        overlap_mask=False,
        mask_ratio=4,
        dropout=0.0,
        
        # ==================== 输出与监控 ====================
        verbose=True,
        plots=True,
        deterministic=True,
    )