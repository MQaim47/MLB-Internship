import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Task 1
# -----------------------------
def task1_theory():
    return """
# AI vs ML vs DL

## Artificial Intelligence (AI)
Machines performing tasks requiring human intelligence.

Examples:
- Siri / Alexa
- Self-driving cars
- ChatGPT

## Machine Learning (ML)
Systems learn patterns from data.

Examples:
- Spam Detection
- Netflix Recommendations
- House Price Prediction

## Deep Learning (DL)
Uses neural networks with multiple layers.

Examples:
- Face Recognition
- Medical Imaging
- Image Classification

### Why Deep Learning for Computer Vision?
- Automatic feature extraction
- Object detection
- Face recognition
- Image classification
- High accuracy on image tasks
"""

# -----------------------------
# Task 2
# -----------------------------
def neural_network_diagram():

    fig, ax = plt.subplots(figsize=(8, 6))

    input_layer = [(1,4),(1,2),(1,0)]
    hidden_layer = [(4,3),(4,1)]
    output_layer = [(7,2)]

    for x,y in input_layer:
        ax.add_patch(plt.Circle((x,y),0.25,fill=False))

    for x,y in hidden_layer:
        ax.add_patch(plt.Circle((x,y),0.25,fill=False))

    for x,y in output_layer:
        ax.add_patch(plt.Circle((x,y),0.25,fill=False))

    for i,(x,y) in enumerate(input_layer):
        ax.text(x,y,f"X{i+1}",ha="center")

    for i,(x,y) in enumerate(hidden_layer):
        ax.text(x,y,f"H{i+1}",ha="center")

    ax.text(7,2,"Y",ha="center")

    weight_num = 1

    for ix,iy in input_layer:
        for hx,hy in hidden_layer:
            ax.plot([ix,hx],[iy,hy],'k-')

            mx=(ix+hx)/2
            my=(iy+hy)/2

            ax.text(mx,my,f"w{weight_num}",
                    color="red",
                    fontsize=8)

            weight_num+=1

    for hx,hy in hidden_layer:
        ax.plot([hx,7],[hy,2],'k-')

        mx=(hx+7)/2
        my=(hy+2)/2

        ax.text(mx,my,f"w{weight_num}",
                color="red",
                fontsize=8)

        weight_num+=1

    ax.text(4,4.5,"Bias b1",color="blue")
    ax.text(7,3.5,"Bias b2",color="blue")

    ax.text(1,5,"Input Layer")
    ax.text(4,5,"Hidden Layer")
    ax.text(7,5,"Output Layer")

    ax.set_xlim(0,8)
    ax.set_ylim(-1,6)
    ax.axis("off")

    return fig

# -----------------------------
# Task 3
# -----------------------------
def simple_neuron(inputs_text,
                  weights_text,
                  bias):

    inputs = np.array(
        list(map(float,
        inputs_text.split(",")))
    )

    weights = np.array(
        list(map(float,
        weights_text.split(",")))
    )

    weighted_sum = np.sum(inputs * weights) + bias

    output = max(0, weighted_sum)

    return (
        f"Weighted Sum = {weighted_sum:.4f}\n"
        f"ReLU Output = {output:.4f}"
    )

# -----------------------------
# Task 4
# -----------------------------
def dataset_split(total_samples):

    train = int(total_samples * 0.70)
    val = int(total_samples * 0.15)
    test = total_samples - train - val

    return (
        f"Training Set : {train}\n"
        f"Validation Set : {val}\n"
        f"Test Set : {test}"
    )

# -----------------------------
# Task 5
# -----------------------------
def train_simple_network():

    X = np.array([[1],[2],[3],[4],[5]])
    y = np.array([[2],[4],[6],[8],[10]])

    w = np.random.randn()
    b = np.random.randn()

    lr = 0.01

    losses = []

    for epoch in range(100):

        y_pred = X*w + b

        loss = np.mean((y-y_pred)**2)

        dw = (-2*np.mean(X*(y-y_pred)))
        db = (-2*np.mean(y-y_pred))

        w -= lr*dw
        b -= lr*db

        losses.append(loss)

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(losses)

    ax.set_title("Loss vs Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)

    final_prediction = (X*w+b).flatten()

    prediction_text = (
        f"Learned Weight = {w:.4f}\n"
        f"Learned Bias = {b:.4f}\n\n"
        f"Predictions:\n{final_prediction}"
    )

    return prediction_text, fig

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Module 14 - Deep Learning") as demo:

    gr.Markdown(
        "# Module 14: Introduction to Deep Learning"
    )

    # -------------------------
    # Task 1
    # -------------------------
    with gr.Tab("Task 1"):
        btn1 = gr.Button(
            "Show Explanation"
        )

        output1 = gr.Markdown()

        btn1.click(
            task1_theory,
            outputs=output1
        )

    # -------------------------
    # Task 2
    # -------------------------
    with gr.Tab("Task 2"):

        btn2 = gr.Button(
            "Generate Network Diagram"
        )

        output2 = gr.Plot()

        btn2.click(
            neural_network_diagram,
            outputs=output2
        )

    # -------------------------
    # Task 3
    # -------------------------
    with gr.Tab("Task 3"):

        inputs_box = gr.Textbox(
            value="2,4,6",
            label="Inputs"
        )

        weights_box = gr.Textbox(
            value="0.5,0.2,0.8",
            label="Weights"
        )

        bias_box = gr.Number(
            value=1,
            label="Bias"
        )

        btn3 = gr.Button(
            "Calculate Neuron Output"
        )

        output3 = gr.Textbox()

        btn3.click(
            simple_neuron,
            inputs=[
                inputs_box,
                weights_box,
                bias_box
            ],
            outputs=output3
        )

    # -------------------------
    # Task 4
    # -------------------------
    with gr.Tab("Task 4"):

        samples = gr.Slider(
            10,
            1000,
            value=100,
            step=10,
            label="Total Samples"
        )

        btn4 = gr.Button(
            "Split Dataset"
        )

        output4 = gr.Textbox()

        btn4.click(
            dataset_split,
            inputs=samples,
            outputs=output4
        )

    # -------------------------
    # Task 5
    # -------------------------
    with gr.Tab("Task 5"):

        btn5 = gr.Button(
            "Train Neural Network"
        )

        output5_text = gr.Textbox(
            lines=10
        )

        output5_plot = gr.Plot()

        btn5.click(
            train_simple_network,
            outputs=[
                output5_text,
                output5_plot
            ]
        )

demo.launch()