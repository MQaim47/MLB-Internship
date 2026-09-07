# Task 4 - NMS & Evaluation

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = YOLO("runs/detect/weights/best.pt")
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640
)
# ----------------------------------
# 1. Evaluate Model
# ----------------------------------

metrics = model.val()

print("\n===== Evaluation Results =====")
print(f"Precision     : {metrics.box.mp:.4f}")
print(f"Recall        : {metrics.box.mr:.4f}")
print(f"mAP@50        : {metrics.box.map50:.4f}")
print(f"mAP@50:95     : {metrics.box.map:.4f}")

# ----------------------------------
# 2. Test Different NMS IoU Thresholds
# ----------------------------------

image_path = "test.jpg"

iou_values = [0.3, 0.5, 0.7]

for iou in iou_values:

    results = model.predict(
        source=image_path,
        conf=0.25,
        iou=iou,
        save=False
    )

    img = cv2.imread(image_path)

    for r in results:

        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                img,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"NMS IoU Threshold = {iou}")
    plt.axis("off")
    plt.show()

# ----------------------------------
# 3. Test Different Confidence Thresholds
# ----------------------------------

conf_values = [0.25, 0.50, 0.75]

for conf in conf_values:

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=0.5,
        save=False
    )

    total_boxes = len(results[0].boxes)

    print(f"\nConfidence Threshold = {conf}")
    print(f"Detections = {total_boxes}")

print("\nTask 4 Completed Successfully!")