# 🧠 Deep Fake Detection Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green)
![License](https://img.shields.io/badge/License-Educational%20Use-lightgrey)
![Last Update](https://img.shields.io/badge/Last%20Updated-July%202025-brightgreen)

A deep learning-powered application designed to detect AI-generated synthetic facial images, commonly known as deepfakes. This project combines a lightweight convolutional neural network with an interactive web application to provide real-time, interpretable deepfake classification.

---

## 📘 Project Overview

This project addresses the increasing threat of deepfakes in digital media by providing a deep learning-based solution to verify the authenticity of facial images. It utilizes a customized version of the **MobileNetV2** architecture and is deployed through a user-facing **Streamlit** web application. The system enables users to upload an image and receive an immediate prediction of whether the image is real or synthetically generated.

---

## 🧠 Model Architecture

The core model is built upon **MobileNetV2**, a lightweight CNN architecture optimized for mobile and low-power devices. The **feature extraction layers were frozen**, preserving their pre-trained weights, while the classification head was replaced with a custom sequence of layers to suit binary classification tasks.

### Customization Includes:
- Retained MobileNetV2 feature extractor
- Added `MaxPooling2D` and dense layers for dimensionality reduction
- Final activation layer replaced with a **sigmoid function** to produce probabilistic binary output:

σ(x) = 1 / (1 + e^(-x))
​

### Training Specifications:
- Loss Function: `Binary Crossentropy`
- Optimizer: `Adam`
- Image Input Size: `224 x 224`
- Output: Probability score ∈ [0, 1]  
- Regularization: `EarlyStopping`, `ModelCheckpoint`

---

## 🗂️ Dataset

The model was trained on the **Celeb-DF v2** dataset and supplemented with synthetically generated and augmented samples. Preprocessing involved normalization, resizing, and labeling images into real or deepfake categories.

---

## 📈 Performance

- Validation Accuracy: **>90%**
- Inference Time: **~1.5 seconds per image**
- Output: **“Real”** or **“Deepfake”** with confidence score

---

## 🌐 Web Application (Streamlit)

An interactive and minimalistic web interface allows users to test the model in real time. Built with **Streamlit**, the application delivers fast predictions and comprehensive visual feedback.

### 🔍 Core Features
- 📤 Upload facial images (JPG, JPEG, PNG)
- 📊 Instant prediction with label and confidence
- 📄 Auto-generated downloadable PDF report
- 📚 Sidebar explanation of deepfakes
- 🌓 Theme toggle: Light / Dark
- 🕘 Recent prediction history with thumbnails

---

## 🚀 Getting Started

### ✅ Prerequisites
Make sure Python 3.8 or higher is installed.

### 📦 Installation

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
````

### ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── app.py                         # Streamlit Web App
├── Complete Model Training.ipynb  # Model training and evaluation notebook
├── xception_deepfake_stage1.h5    # Trained model file
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation
```

---

## 📄 PDF Report

For each prediction, the application generates a downloadable PDF file summarizing:

* Classification result
* Confidence percentage
* Model type and detection method

---

## 🎓 Academic Context

This project was completed under the academic supervision of **Ma’am Aima Munir**, as part of course of Articifial Intelligence. It serves as both a functional tool and an educational contribution to the field of computer vision and AI ethics.

---

## 👨‍💻 Author

**Muhammad Saad**
Just an ordinary programmer
📫 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)

---

## 📌 License

This project is intended for **educational and research purposes only**. Commercial use is not permitted without explicit authorization. If you use this project or its components, kindly cite it appropriately.

---

## 💬 Acknowledgments

Special thanks to Ma’am Aima Munir for her continued mentorship and guidance, and to the open-source community whose tools and datasets made this work possible.

---

## 🏷️ Tags

`Deep Learning` · `MobileNetV2` · `Deepfake Detection` · `Computer Vision` · `Model Deployment` · `Streamlit` · `Keras` · `TensorFlow` · `AI for Good`


