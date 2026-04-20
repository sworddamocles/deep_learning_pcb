from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # 检查GPU可用性
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载模型
    model = YOLO(r"D:\deep_learning\ultralytics-8.3.163\yolo11_cbam_SimAM.yaml",verbose=True)

    # 优化的训练参数
    model.train(
        # 必需参数
        data=r"D:\deep_learning\ultralytics-8.3.163\dataset_split\data.yaml",
        epochs=300,  # 减少epochs，先测试
        imgsz=640,

        # 数据加载参数
        batch=8,  # 根据您的GPU（RTX 3050 4GB）调整
        workers=4,  # 增加workers提高数据加载效率
        cache=True,  # 启用缓存加速训练

        # 优化器和学习率
        optimizer="SGD",  # 指定优化器
        lr0=0.01,  # 初始学习率
        lrf=0.1,  # 最终学习率系数
        momentum=0.937,
        weight_decay=0.0005,

        # 学习率预热
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 数据增强
        augment=True,  # 启用数据增强
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        degrees=5.0,  # 旋转角度
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=0.0,  # 错切
        perspective=0.0,  # 透视变换
        flipud=0.0,  # 上下翻转
        fliplr=0.5,  # 左右翻转
        mosaic=1.0,  # 马赛克增强
        #mixup=0.0,  # MixUp增强

        # 训练控制
        # device=device,
        # patience=50,  # 早停耐心值
        # close_mosaic=10,  # 最后10个epoch关闭马赛克

        # 验证和保存
        val=True,  # 启用验证
        save=True,
        save_period=10,  # 每10个epoch保存一次
        save_json=False,
        save_hybrid=False,

        # 训练管理
        project="runs/train",
        name="yolo11_cbam_SimAM_v1",  # 实验名称
        exist_ok=True,  # 允许覆盖
        pretrained=True,  # 使用预训练权重
        resume=False,  # 不恢复训练

        # 混合精度训练
        amp=True,  # 自动混合精度

        # 确定性
        # deterministic=False,  # 关闭确定性避免警告

        # 类别平衡
        # single_cls=False,  # 多类别训练

        # 输出控制
        verbose=True,
        plots=True,  # 保存训练曲线
    )