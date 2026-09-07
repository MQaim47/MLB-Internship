"""
application.py

Gradio web app for custom-trained YOLOv8 detector.

Features:
- Upload image
- Detect waste objects
- Show bounding boxes
- Show class names
- Show confidence scores
- Adjust confidence threshold
- Adjust NMS IoU threshold
- Expose Gradio app using ngrok

Run:
    python application.py
"""

import os
import glob

import cv2
import gradio as gr
from ultralytics import YOLO
from pyngrok import ngrok



# 1. CONFIGURATION


# Path to your trained YOLO model
MODEL_PATH = "YOLO_Project/custom_detector/weights/best.pt"

# Waste classes
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45

# Gradio port
PORT = 7860



NGROK_AUTHTOKEN = "3GxAaEbJZkVvAEXGLqpRhh3ryux_2rHizAtkkChFnfXPGzL2c"


# Different color for each class
BOX_COLORS = [
    (71, 99, 255),
    (113, 179, 60),
    (225, 105, 65),
    (0, 165, 255),
    (211, 85, 186),
    (209, 206, 0),
]



# 2. FIND MODEL


def resolve_model_path(path):

    # Check specified model path
    if os.path.exists(path):
        print("Using model:", path)
        return path

    # Search automatically
    matches = glob.glob(
        "YOLO_Project/**/weights/best.pt",
        recursive=True
    )

    if matches:
        print("Specified model not found.")
        print("Using:", matches[0])
        return matches[0]

    # If no custom model found
    print("WARNING: Custom model not found.")
    print("Using yolov8n.pt instead.")

    return "yolov8n.pt"



# 3. LOAD YOLO MODEL


model_path = resolve_model_path(MODEL_PATH)

model = YOLO(model_path)

print("\nModel loaded successfully!")
print("Classes:", model.names)



# 4. OBJECT DETECTION FUNCTION


def detect(image, confidence, iou_threshold):

    if image is None:
        return None, "Please upload an image."

    # Gradio image is RGB
    # OpenCV uses BGR
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # YOLO prediction
    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou_threshold,
        verbose=False
    )

    result = results[0]

    detections = []

    # Loop through detected objects
    for box in result.boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Confidence
        conf = float(box.conf[0])

        # Class ID
        class_id = int(box.cls[0])

        # Class name
        class_name = model.names[class_id]

        # Select color
        color = BOX_COLORS[
            class_id % len(BOX_COLORS)
        ]

        # Label
        label = f"{class_name} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Draw label
        cv2.putText(
            image_bgr,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Save detection
        detections.append(
            (
                class_name,
                conf,
                (x1, y1, x2, y2)
            )
        )

    # Convert BGR back to RGB
    annotated_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    # Detection summary
   

    if not detections:

        summary = (
            "### No objects detected\n\n"
            "Try lowering the confidence threshold."
        )

    else:

        summary = (
            "| Class | Confidence | Bounding Box |\n"
            "|---|---:|---|\n"
        )

        for class_name, conf, box_coords in detections:

            summary += (
                f"| {class_name} "
                f"| {conf:.2%} "
                f"| {box_coords} |\n"
            )

        summary += (
            f"\n**Total detections: "
            f"{len(detections)}**"
        )

    return annotated_rgb, summary




with gr.Blocks(
    title="YOLO Waste Detector"
) as demo:

    gr.Markdown(
        """
        # YOLO Waste Detector

        Upload a waste image and the trained YOLOv8 model
        will detect objects using bounding boxes, class names,
        and confidence scores.
        """
    )

    with gr.Row():

      
        # LEFT SIDE
       

        with gr.Column():

            image_input = gr.Image(
                type="numpy",
                label="Upload Image"
            )

            confidence_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=DEFAULT_CONFIDENCE,
                step=0.05,
                label="Confidence Threshold"
            )

            iou_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=DEFAULT_IOU,
                step=0.05,
                label="NMS IoU Threshold"
            )

            detect_button = gr.Button(
                "Detect",
                variant="primary"
            )

            # Example images
            example_images = glob.glob(
                "images/*.jpg"
            )

            if example_images:

                gr.Examples(
                    examples=example_images,
                    inputs=image_input
                )

       
        # RIGHT SIDE
      

        with gr.Column():

            image_output = gr.Image(
                type="numpy",
                label="Detections"
            )

            detections_output = gr.Markdown()


    # Connect button to detection function
    detect_button.click(
        fn=detect,
        inputs=[
            image_input,
            confidence_slider,
            iou_slider
        ],
        outputs=[
            image_output,
            detections_output
        ]
    )




def start_ngrok_tunnel(port):

    print("\nSetting ngrok authentication...")

    # Set ngrok authentication token
    ngrok.set_auth_token(
        NGROK_AUTHTOKEN
    )

    print("Authentication successful.")

    # Create public tunnel
    tunnel = ngrok.connect(
        port,
        "http"
    )

    public_url = tunnel.public_url

    print("\n========================================")
    print("NGROK PUBLIC URL")
    print("========================================")
    print(public_url)
    print("========================================\n")

    return public_url



if __name__ == "__main__":

    # Check token
    if (
        not NGROK_AUTHTOKEN
        or NGROK_AUTHTOKEN == "YOUR_NGROK_TOKEN"
    ):

        raise ValueError(
            "\nNGROK AUTHTOKEN IS MISSING!\n\n"
            "Open application.py and replace:\n"
            'NGROK_AUTHTOKEN = "YOUR_NGROK_TOKEN"\n'
            "with your actual ngrok token."
        )

    public_url = start_ngrok_tunnel(PORT)

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False
    )