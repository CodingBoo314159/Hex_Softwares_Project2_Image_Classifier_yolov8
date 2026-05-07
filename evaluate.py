from ultralytics import YOLO

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Evaluate model on validation set
metrics = model.val()

# Print evaluation results
print(metrics)
