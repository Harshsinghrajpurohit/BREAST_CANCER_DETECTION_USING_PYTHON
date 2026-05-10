# 🎗️ Breast Cancer Detection Using Python & Machine Learning

A full-stack web application that uses a Neural Network to classify breast tumors as **Benign** or **Malignant** based on 30 cell nucleus features from FNA biopsy images.

> Built with Python, TensorFlow, Flask, and scikit-learn. Achieves **96.49% test accuracy**.

---


---

## 🚀 Features

- Neural network trained on the Breast Cancer Wisconsin dataset
- Full-stack web interface built with Flask
- Real-time predictions via REST API (no page refresh)
- Auto-fill sample buttons for quick testing (benign & malignant)
- Confidence score and probability bars for each prediction
- 569 patient records exported as `dataset.csv`

---

## 🗂️ Project Structure

```
breast_cancer_app/
├── app.py               # Flask backend + model training
├── train_model.py       # Standalone model training script
├── dataset.csv          # Breast Cancer Wisconsin dataset (569 rows)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── templates/
    └── index.html       # Frontend (HTML/CSS/JS)
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/rohancanish/BREAST_CANCER_DETECTION_USING_PYTHON.git
cd BREAST_CANCER_DETECTION_USING_PYTHON
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the web app
```bash
python app.py
```
Then open your browser and go to: **http://127.0.0.1:5000**

### Run standalone model training
```bash
python train_model.py
```

---

## 📊 Dataset

The **Breast Cancer Wisconsin (Diagnostic)** dataset is sourced from scikit-learn's built-in datasets (originally from the UCI Machine Learning Repository).

| Property | Value |
|----------|-------|
| Total samples | 569 |
| Benign | 357 |
| Malignant | 212 |
| Features | 30 |
| Target classes | 2 (Benign / Malignant) |

### Feature Groups
| Group | Features |
|-------|----------|
| Mean | radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| Error (SE) | same 10 features above |
| Worst | same 10 features above |

---

## 🧠 Model Architecture

```
Input Layer  →  30 features
Hidden Layer →  20 neurons (ReLU activation)
Output Layer →  2 neurons (Softmax activation)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 10 |
| Test Accuracy | **96.49%** |

---

## 🌐 API Reference

### `POST /predict`

Send 30 feature values as JSON, get back a prediction.

**Request body:**
```json
{
  "mean radius": 11.76,
  "mean texture": 21.6,
  "mean perimeter": 74.72,
  ...
}
```

**Response:**
```json
{
  "result": "Benign",
  "confidence": 97.3,
  "benign_prob": 97.3,
  "malignant_prob": 2.7
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML / AI | TensorFlow, Keras, scikit-learn |
| Data | NumPy, Pandas |
| Frontend | HTML, CSS, JavaScript |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.

---


