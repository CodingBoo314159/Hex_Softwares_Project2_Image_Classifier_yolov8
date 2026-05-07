import gradio as gr
from ultralytics import YOLO

# Load model once (important for stability + no flicker)
model = YOLO("last.pt")

def predict(image):
    if image is None:
        return {}

    results = model(image)

    names = results[0].names
    probs = results[0].probs.data.tolist()

    # return dictionary for UI display
    return {names[i]: float(probs[i]) for i in range(len(probs))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),   # FIX: removes flicker caused by gr.Label
    title="Weather Classifier",
    description="Upload an image to classify the weather condition"
)

demo.launch()
