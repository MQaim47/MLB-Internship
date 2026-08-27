"""
SmartWaste CNN - Gradio App
============================
A Gradio web interface for the SmartWaste CNN project: a PyTorch CNN
that classifies an image of waste into one of six TrashNet categories
(cardboard, glass, metal, paper, plastic, trash) and suggests how to
dispose of / recycle it.

This app reuses the exact same SimpleCNN architecture and preprocessing
pipeline used during training (see improvement_day3.py / evaluation_day4.py
/ predictImage.py), and auto-locates the trained weights and result
images no matter how you've arranged your project folder - it works
whether files sit flat next to app.py (like the original project) or
inside model/ / assets/ / examples/ subfolders.

Run with:
    python app.py
"""

import glob
import os

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = 128

# Class order matches torchvision.datasets.ImageFolder's alphabetical
# sorting of the dataset/trashnet subfolders used during training.
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Friendly disposal guidance shown alongside the prediction.
DISPOSAL_INFO = {
    "cardboard": (
        "♻️ Recyclable",
        "Flatten the box and put it in your paper/cardboard recycling bin. "
        "Keep it dry and free of food grease.",
    ),
    "glass": (
        "♻️ Recyclable",
        "Rinse it out and place it in your glass recycling bin. Glass can "
        "be recycled endlessly without losing quality.",
    ),
    "metal": (
        "♻️ Recyclable",
        "Rinse cans/metal items and put them in your metal recycling bin. "
        "Aluminum and steel are both highly recyclable.",
    ),
    "paper": (
        "♻️ Recyclable",
        "Place clean, dry paper in your paper recycling bin. Avoid "
        "recycling paper that's soiled or laminated.",
    ),
    "plastic": (
        "♻️ Recyclable (check local rules)",
        "Rinse the item and check the resin code on the bottom - most "
        "local programs accept #1 and #2 plastics; others vary.",
    ),
    "trash": (
        "🗑️ General waste",
        "This doesn't fit standard recycling streams - dispose of it in "
        "your general waste bin.",
    ),
}


# ---------------------------------------------------------------------------
# File auto-discovery - works whether your files are flat next to app.py
# (like the original project folder) or organized into subfolders.
# ---------------------------------------------------------------------------

def find_file(*filenames):
    """Return the first matching path for any of the given filenames,
    checking (in order): flat next to app.py, common subfolders, then a
    full recursive search under BASE_DIR. Returns None if nothing found."""
    subfolders = ["", "model", "assets", "examples"]
    for filename in filenames:
        for sub in subfolders:
            candidate = os.path.join(BASE_DIR, sub, filename) if sub else os.path.join(BASE_DIR, filename)
            if os.path.isfile(candidate):
                return candidate
        matches = glob.glob(os.path.join(BASE_DIR, "**", filename), recursive=True)
        if matches:
            return matches[0]
    return None


def find_dir(*dirnames):
    """Return the first existing directory matching any given name,
    checked flat under BASE_DIR then recursively."""
    for dirname in dirnames:
        candidate = os.path.join(BASE_DIR, dirname)
        if os.path.isdir(candidate):
            return candidate
        matches = glob.glob(os.path.join(BASE_DIR, "**", dirname), recursive=True)
        matches = [m for m in matches if os.path.isdir(m)]
        if matches:
            return matches[0]
    return None


# Prefer the improved/augmented model (waste_classifier_day3.pth); fall
# back to the original one (waste_classifier.pth) if that's all you have.
MODEL_PATH = find_file("waste_classifier_day3.pth", "waste_classifier.pth")


# ---------------------------------------------------------------------------
# Model definition (identical architecture to train_day2.py /
# improvement_day3.py / evaluation_day4.py / predictImage.py)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> SimpleCNN:
    if MODEL_PATH is None:
        raise FileNotFoundError(
            "Could not find 'waste_classifier_day3.pth' (or 'waste_classifier.pth') "
            f"anywhere under '{BASE_DIR}'. Place the .pth file in the same folder as "
            "app.py (flat, or inside a 'model' subfolder) and try again."
        )
    print(f"Loading model weights from: {MODEL_PATH}")
    model = SimpleCNN(num_classes=len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


model = load_model()

# Same preprocessing used for validation/test in the training scripts
# (Resize -> ToTensor -> Normalize(0.5, 0.5, 0.5)). No augmentation here,
# since augmentation is only meant to be applied during training.
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_waste(image: Image.Image):
    """Takes a PIL image, returns (label_scores_dict, guidance_markdown)."""
    if image is None:
        return None, "Upload or select an image to get a prediction."

    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    scores = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

    top_class = max(scores, key=scores.get)
    top_conf = scores[top_class] * 100
    tag, tip = DISPOSAL_INFO[top_class]

    guidance = (
        f"### Prediction: **{top_class.capitalize()}** ({top_conf:.1f}% confidence)\n\n"
        f"**{tag}**\n\n"
        f"{tip}"
    )

    return scores, guidance


# ---------------------------------------------------------------------------
# Example images
# ---------------------------------------------------------------------------

def build_examples():
    # 1) A curated examples/ folder, if present.
    examples_dir = find_dir("examples")
    if examples_dir:
        files = sorted(glob.glob(os.path.join(examples_dir, "*")))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            return files

    # 2) Otherwise, fall back to picking a couple of images per class
    #    straight from dataset/trashnet/<class>/, if that folder exists.
    dataset_dir = find_dir(os.path.join("dataset", "trashnet"))
    if dataset_dir:
        examples = []
        for cls in CLASSES:
            cls_dir = os.path.join(dataset_dir, cls)
            if os.path.isdir(cls_dir):
                for fname in sorted(os.listdir(cls_dir))[:2]:
                    examples.append(os.path.join(cls_dir, fname))
        return examples

    return []


EXAMPLES = build_examples()

# Training-curve / confusion-matrix images for the "About" tab (optional -
# the tab still works fine if some or all of these aren't found).
LOSS_IMG = find_file("train_vs_val_loss.png", "train Vs val.png")
ACC_IMG = find_file("train_vs_val_accuracy.png", "train Vs val Accuracy.png")
CM_IMG = find_file("confusion_matrix.png", "confusion Matrix.png")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: var(--body-text-color-subdued); margin-bottom: 1rem; }
"""

with gr.Blocks(title="SmartWaste CNN") as demo:
    gr.Markdown("# ♻️ SmartWaste CNN — Waste Classifier", elem_id="title")
    gr.Markdown(
        "Upload a photo of an item of waste and the CNN will classify it into "
        "one of six categories — **cardboard, glass, metal, paper, plastic, "
        "or trash** — and tell you how to dispose of it.",
        elem_id="subtitle",
    )

    with gr.Tab("Classify"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Waste image", height=320)
                with gr.Row():
                    clear_btn = gr.ClearButton(value="Clear")
                    submit_btn = gr.Button("Classify", variant="primary")
                if EXAMPLES:
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=image_input,
                        label="Try an example",
                        examples_per_page=12,
                    )
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=6, label="Class probabilities")
                guidance_output = gr.Markdown(label="Disposal guidance")

        clear_btn.add([image_input, label_output, guidance_output])
        submit_btn.click(fn=classify_waste, inputs=image_input, outputs=[label_output, guidance_output])
        image_input.change(fn=classify_waste, inputs=image_input, outputs=[label_output, guidance_output])

    with gr.Tab("About the model"):
        gr.Markdown(
            """
### Architecture
A small CNN (`SimpleCNN`) trained from scratch on the **TrashNet** dataset:

- `Conv2d(3→16, k=3)` → ReLU → `MaxPool2d(2,2)`
- `Conv2d(16→32, k=3)` → ReLU → `MaxPool2d(2,2)`
- `Linear(32*30*30 → 128)` → ReLU → `Linear(128 → 6)`

Input images are resized to 128×128 and normalized with mean/std of 0.5 on
each channel. Training used random horizontal flip + rotation augmentation,
Adam optimizer (lr=0.001), and cross-entropy loss for 10 epochs (80/10/10
train/val/test split).
            """
        )
        if LOSS_IMG or ACC_IMG:
            with gr.Row():
                if LOSS_IMG:
                    gr.Image(LOSS_IMG, label="Train vs Validation Loss", show_label=True)
                if ACC_IMG:
                    gr.Image(ACC_IMG, label="Train vs Validation Accuracy", show_label=True)
        if CM_IMG:
            gr.Image(CM_IMG, label="Confusion Matrix (test set)", show_label=True)
        if not (LOSS_IMG or ACC_IMG or CM_IMG):
            gr.Markdown(
                "_(Training curve / confusion matrix images weren't found in this "
                "project folder, so they're not shown here - the classifier above "
                "still works fine without them.)_"
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)