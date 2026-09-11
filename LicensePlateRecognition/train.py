from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=5,
    imgsz=640,
    batch=16,
    workers=4
)