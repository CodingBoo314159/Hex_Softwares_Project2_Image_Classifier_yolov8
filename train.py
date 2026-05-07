import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")

# Train the model
model.train(
    data="weather_dataset",   # Path to dataset
    epochs=20,
    imgsz=64
)

# Path to training results
results_path = "runs/classify/train/results.csv"

# Load results into DataFrame
results_df = pd.read_csv(results_path)

# Display first rows
print(results_df.head())

# Plot training accuracy
plt.plot(results_df["epoch"], results_df["metrics/accuracy_top1"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()

# Plot training loss
plt.plot(results_df["epoch"], results_df["train/loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
