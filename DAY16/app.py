import gradio as gr

# -----------------------------
# IoU Calculation
# -----------------------------
def calculate_iou(box1_str, box2_str):
    try:
        box1 = list(map(float, box1_str.split(",")))
        box2 = list(map(float, box2_str.split(",")))

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0

        return f"IoU = {iou:.4f}"

    except:
        return "Invalid Input"


# -----------------------------
# Confidence Threshold
# -----------------------------
def confidence_demo(threshold):
    detections = [
        ("Dog", 0.95),
        ("Cat", 0.82),
        ("Car", 0.30),
        ("Person", 0.65)
    ]

    result = ""

    for label, conf in detections:
        if conf >= threshold:
            result += f"{label} : {conf}\n"

    return result


# -----------------------------
# Precision Recall
# -----------------------------
def precision_recall(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return (
            f"Precision = {precision:.4f}\n"
            f"Recall = {recall:.4f}"
        )
    except:
        return "Invalid Values"


# -----------------------------
# YOLO Annotation Reader
# -----------------------------
def read_annotation(text):
    try:
        lines = text.strip().split("\n")

        output = ""

        for line in lines:
            values = line.split()

            class_id = values[0]
            x_center = values[1]
            y_center = values[2]
            width = values[3]
            height = values[4]

            output += (
                f"Class ID : {class_id}\n"
                f"X Center : {x_center}\n"
                f"Y Center : {y_center}\n"
                f"Width    : {width}\n"
                f"Height   : {height}\n\n"
            )

        return output

    except:
        return "Invalid Annotation Format"


# -----------------------------
# NMS Demo
# -----------------------------
def nms_demo():
    detections = [
        ("Dog", 0.95),
        ("Dog", 0.91),
        ("Dog", 0.87)
    ]

    best = max(detections, key=lambda x: x[1])

    return (
        f"Original Detections:\n{detections}\n\n"
        f"After NMS:\n{best}"
    )


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Module 15 - Object Detection Concepts") as demo:

    gr.Markdown("# Module 15 - Day 1 Object Detection Concepts")

    with gr.Tab("IoU"):
        box1 = gr.Textbox(
            label="Box 1 (x_min,y_min,x_max,y_max)",
            value="100,100,300,300"
        )

        box2 = gr.Textbox(
            label="Box 2 (x_min,y_min,x_max,y_max)",
            value="150,150,350,350"
        )

        btn = gr.Button("Calculate IoU")
        out = gr.Textbox()

        btn.click(calculate_iou, [box1, box2], out)

    with gr.Tab("Confidence Threshold"):
        threshold = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.05
        )

        btn = gr.Button("Apply Threshold")
        out = gr.Textbox(lines=8)

        btn.click(confidence_demo, threshold, out)

    with gr.Tab("Precision & Recall"):
        tp = gr.Number(label="TP", value=90)
        fp = gr.Number(label="FP", value=10)
        fn = gr.Number(label="FN", value=20)

        btn = gr.Button("Calculate")

        out = gr.Textbox()

        btn.click(
            precision_recall,
            [tp, fp, fn],
            out
        )

    with gr.Tab("YOLO Annotation"):
        annotation = gr.Textbox(
            lines=6,
            label="YOLO Annotation",
            value="0 0.50 0.40 0.30 0.25"
        )

        btn = gr.Button("Read Annotation")

        out = gr.Textbox(lines=10)

        btn.click(
            read_annotation,
            annotation,
            out
        )

    with gr.Tab("NMS Demo"):
        btn = gr.Button("Run NMS")

        out = gr.Textbox(lines=10)

        btn.click(
            nms_demo,
            outputs=out
        )

demo.launch()