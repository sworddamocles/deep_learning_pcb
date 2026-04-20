from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
#    source=r"ultralytics\assets",
    source=0,
    save=False,
    show=True,
    line_width=8,
)