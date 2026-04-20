from ultralytics import YOLO

# 加载您的自定义模型配置文件
model = YOLO("yolo11_cbam_SimAM.yaml", verbose=True)  # 使用您修改后的配置文件
