from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# STEP 1: TRAIN YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on custom dataset
results = model.train(
    data="data.yaml",     # dataset config file
    epochs=20,
    imgsz=640,
    batch=8,
    project="YOLO_Project",
    name="custom_detector"
)

print("Training Completed!")

# STEP 2: LOAD TRAINED MODEL

trained_model = YOLO(
    "YOLO_Project/custom_detector/weights/best.pt"
)

# STEP 3: INFERENCE FUNCTION

def detect_objects(image_path, confidence_threshold):

    image = cv2.imread(image_path)

    results = trained_model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=False
    )

    result = results[0]

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        class_id = int(box.cls[0])

        class_name = trained_model.names[class_id]

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        print(f"Class: {class_name}")
        print(f"Confidence: {conf:.2f}")
        print("-"*30)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(
        f"Confidence Threshold = {confidence_threshold}"
    )
    plt.show()

# STEP 4: TEST ON UNSEEN IMAGE


test_image = "test.jpg"

print("\nThreshold = 0.25")
detect_objects(test_image, 0.25)

print("\nThreshold = 0.50")
detect_objects(test_image, 0.50)

print("\nThreshold = 0.75")
detect_objects(test_image, 0.75)