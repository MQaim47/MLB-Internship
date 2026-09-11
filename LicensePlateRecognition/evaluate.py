from ultralytics import YOLO

model = YOLO(
    "runs/detect/train/weights/best.pt"
)

results = model.predict(
    source="dataset/test/images",
    conf=0.25,
    save=True
)