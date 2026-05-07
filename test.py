from ultralytics import YOLO

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Run prediction on a test image
results = model.predict(
    source="test_image.jpg",
    imgsz=64
)

# Display prediction results
for result in results:
    print(result.probs)
