# Hex_Softwares_Project2_Image_Classifier_yolov8

Please see the website for a working interface: https://huggingface.co/spaces/JenniferAnnKok/weather-classifier

# 🌤 Weather Classifier

A machine learning web app that classifies weather conditions from images using a YOLOv8 classification model.

---

## 🚀 Demo

Upload an image (e.g. rain, sunny sky, cloudy sky) and the model will predict the weather condition with confidence scores.

---

## 🧠 Model

This project uses:
- Ultralytics YOLOv8 classification model
- Custom-trained weather image dataset

---

## 📁 Project Structure

```
.
├── app.py              # Gradio web app
├── last.pt             # Trained YOLO model
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── predict.py          # Single image prediction script
└── README.md
```

---

## ⚙️ Installation

Install dependencies:

```bash
pip install ultralytics gradio numpy pillow
```

---

## ▶️ Run the App

```bash
python app.py
```

Then open the link provided (local or Hugging Face Spaces).

---

## 📊 Output

The model returns:
- Weather class (e.g. Rain, Sunny, Cloudy)
- Confidence score (e.g. 0.9995 = 99.95% confidence)

---

## 📸 Example Output

```
Rain → 0.9995
Cloudy → 0.0003
Sunny → 0.0002
```

---

## 💡 How it works

1. Upload an image
2. YOLOv8 processes the image
3. Model outputs probabilities for each class
4. Highest probability is selected as the prediction

---

## 🌐 Deployment

This project is deployed using **Hugging Face Spaces** with Gradio.

---

## 👩‍💻 Author

Jennifer Ann Kok
