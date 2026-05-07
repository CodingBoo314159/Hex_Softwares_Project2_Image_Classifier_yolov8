from ultralytics import YOLO

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Predict weather class from image
results = model.predict(
    source="image.jpg",
    imgsz=64
)

# Print predicted class
for result in results:
    probs = result.probs
    predicted_class = probs.top1
    class_name = result.names[predicted_class]

    print(f"Predicted Weather: {class_name}")
