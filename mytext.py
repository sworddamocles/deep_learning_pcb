from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
print(model.task)
print(model.names)
print(sum(p.numel() for p in model.parameters()))