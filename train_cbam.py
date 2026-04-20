import torch

from ultralytics import YOLO


def main():
    # 创建模型
    model = YOLO("yolov8n-cbam.yaml")  # 从配置文件创建

    # 或者从预训练权重开始
    # model = YOLO("yolov8n.pt")

    # 训练参数
    train_args = {
        "data": "pcb_dataset.yaml",  # 你的数据配置文件
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "workers": 4,
        "patience": 50,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "optimizer": "AdamW",  # 推荐使用AdamW优化器
        "lr0": 0.001,  # 初始学习率
        "lrf": 0.01,  # 最终学习率
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "box": 7.5,  # box loss gain
        "cls": 0.5,  # cls loss gain
        "dfl": 1.5,  # dfl loss gain
        "save": True,
        "save_period": 10,
        "project": "runs/train",
        "name": "yolov8n_cbam",
        "exist_ok": True,
        "pretrained": True,  # 使用预训练权重
        "amp": True,  # 自动混合精度
    }

    # 开始训练
    model.train(**train_args)

    # 验证
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map50:.3f}")


if __name__ == "__main__":
    main()
