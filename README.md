# 🚦 Traffic Sign Detection with YOLOv8

A deep learning-based traffic sign detection system using the **GTSDB** (German Traffic Sign Detection Benchmark) dataset and **YOLOv8** model.

## 📋 Project Overview

This system detects and localizes traffic signs in images by:
- Drawing bounding boxes around detected signs
- Classifying each sign into predefined categories
- Providing confidence scores for each detection

## 🏗️ Project Structure

```
traffic-sign-detection-yolo/
│
├── backend/              # FastAPI backend for inference
│   ├── main.py           # API endpoints
│   ├── model/            # Trained model files
│   └── requirements.txt  # Backend dependencies
│
├── frontend/             # Angular demo UI
│   └── angular-app/
│
├── ml/                   # Machine learning pipeline
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── prepare_dataset.py # Dataset preparation
│   ├── dataset/          # GTSDB dataset (YOLO format)
│   └── weights/          # Trained model weights
│
├── notebooks/            # Jupyter notebooks for exploration
│
└── README.md
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | YOLOv8 (Ultralytics) |
| Backend | FastAPI (Python) |
| Frontend | Angular |
| Dataset | GTSDB (~900 images) |

## 🚀 Quick Start

### 1. Install ML Dependencies
```bash
cd ml
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python prepare_dataset.py
```

### 3. Train Model
```bash
python train.py
```

### 4. Start Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 5. Start Frontend
```bash
cd frontend/angular-app
ng serve
```

## 📊 Evaluation Metrics

- **mAP** (mean Average Precision)
- **Precision**
- **Recall**
- Loss curves and sample predictions

## 📝 License

This project is for educational purposes (CSE460 Final Project).