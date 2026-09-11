from ultralytics import YOLO

model = YOLO(
    "runs/detect/train/weights/best.pt"
)

model.predict(
    source="edge_cases",
    conf=0.25,
    save=True
)