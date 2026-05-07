import gradio as gr
from ultralytics import YOLO

# Load model ONCE (important for preventing flickering)
model = YOLO("last.pt")

def predict(image):
    if image is None:
        return {}

    # Run inference
    results = model(image)

    # Extract class names + probabilities
    names = results[0].names
    probs = results[0].probs.data.tolist()

    # Convert to dictionary for Gradio Label output
    return {names[i]: float(probs[i]) for i in range(len(probs))}

# Build interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Weather Classifier",
    description="Upload an image to classify the weather condition",
    allow_flagging="never"
)

# Launch app (no SSR mode — keeps it stable on Spaces)
demo.launch()
