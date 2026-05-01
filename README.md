# 🔍 SHAR — Suspicious Human Activity Recognition

> Deep Learning project for detecting suspicious vs normal human activities in video surveillance footage using CNN + LSTM architectures.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU--Accelerated-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b.svg)](https://streamlit.io)

---

## 📋 Project Overview

| Property | Details |
|----------|---------|
| **Task** | Multi-class video activity classification |
| **Classes** | 21 total (13 Suspicious + 8 Normal) |
| **Dataset** | Human Activity & Suspicious Behavior Video Dataset |
| **Total Videos** | 1,334 MP4 clips |
| **Model** | CNN-LSTM with Attention (ResNet50 backbone) |
| **GPU** | NVIDIA RTX 3050 6GB (CUDA accelerated) |

---

## 🏗️ Project Workflow (7 Phases)

```
Phase 1 — Dataset Preparation
  └── Folder structure, label mapping, EDA

Phase 2 — Preprocessing & Feature Extraction
  └── Video → frames (OpenCV), resize 224×224, augmentation

Phase 3 — CNN Baseline
  └── ResNet-50 + temporal mean pooling

Phase 4 — CNN-LSTM (Main Model)
  └── ResNet-50 encoder + Bi-LSTM + Soft Attention

Phase 5 — CNN-GRU + Model Comparison
  └── MobileNetV3 + GRU, speed/accuracy tradeoffs

Phase 6 — Explainability (XAI)
  └── Grad-CAM heatmaps + Temporal Attention visualization

Phase 7 — Research Documentation
  └── IEEE-style writeup with results and future scope
```

---

## 📁 Project Structure

```
SHAR_Project/
├── 📓 notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Phase3_CNN_Baseline.ipynb
│   ├── 03_Phase4_CNNLSTM.ipynb
│   ├── 04_Phase5_CNNGRU_Comparison.ipynb
│   └── 05_Phase6_Explainability.ipynb
├── 🐍 src/
│   ├── utils.py          # Utilities, plotting, seeding
│   ├── dataset.py        # Video dataset & dataloader
│   ├── model.py          # CNN-LSTM, CNN-GRU, CNN Baseline
│   ├── train.py          # GPU training pipeline
│   ├── evaluate.py       # Evaluation & confusion matrix
│   └── predict.py        # Single video inference + Grad-CAM
├── 🌐 webapp/
│   └── app.py            # Streamlit web application
├── 📊 data/
│   ├── raw/              # ← Place Kaggle dataset here
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── processed/
├── 🤖 models/
│   ├── saved/            # Best model checkpoints
│   └── checkpoints/      # Epoch-wise saves
├── 📈 results/
│   ├── plots/            # Training curves, confusion matrix
│   └── reports/
├── 🔧 scripts/
│   └── setup_folders.py
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Clone & Setup
```bash
# Create environment (recommended)
python -m venv venv
.\venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Download Dataset
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/daudshah/video-dataset)
2. Download and extract
3. Place class folders:
```
data/raw/train/Fighting/*.mp4
data/raw/val/Fighting/*.mp4
data/raw/test/Fighting/*.mp4
... (for all 21 classes)
```

### Step 3 — Create Folder Structure
```bash
python scripts/setup_folders.py
```

### Step 4 — Run Notebooks (in order)
```
notebooks/01_EDA_and_Preprocessing.ipynb   → Phase 1 & 2
notebooks/02_Phase3_CNN_Baseline.ipynb     → Phase 3
notebooks/03_Phase4_CNNLSTM.ipynb         → Phase 4 (main)
notebooks/04_Phase5_CNNGRU_Comparison.ipynb → Phase 5
notebooks/05_Phase6_Explainability.ipynb  → Phase 6
```

### Step 5 — Train via CLI (alternative)
```bash
cd SHAR_Project
python src/train.py --model cnn_lstm --epochs 30 --batch_size 8
```

### Step 6 — Evaluate
```bash
python src/evaluate.py --model cnn_lstm --checkpoint models/saved/cnn_lstm_best.pth
```

### Step 7 — Single Video Prediction
```bash
python src/predict.py --video path/to/video.mp4 --checkpoint models/saved/cnn_lstm_best.pth
```

### Step 8 — Launch Web App
```bash
streamlit run webapp/app.py
```

---

## 🧠 Model Architectures

### CNN-LSTM (Recommended)
```
Input: (B, 20, 3, 224, 224)
    ↓
ResNet-50 CNN Encoder (pretrained, partial freeze)
    → (B, 20, 2048) frame features
    ↓
Bidirectional LSTM (256 hidden, 2 layers)
    → (B, 20, 512) temporal context
    ↓
Soft Attention Pooling
    → (B, 512) attended context
    ↓
MLP Classifier (512 → 256 → 21)
    ↓
Output: (B, 21) class logits
```

### CNN-GRU (Lightweight)
- MobileNetV3-Small backbone (faster, less VRAM)
- GRU instead of LSTM (fewer parameters)

---

## 📊 Expected Results

| Model | Val Acc | Test Acc | Params | Infer (ms) |
|-------|---------|----------|--------|-----------|
| CNN Baseline | ~75% | ~73% | ~25M | ~45ms |
| CNN-LSTM | ~85% | ~83% | ~28M | ~60ms |
| CNN-GRU | ~80% | ~78% | ~5M | ~30ms |

*Results are approximate; actual values depend on training.*

---

## 🔍 Explainability

- **Grad-CAM**: Highlights spatial regions in frames that triggered the classification
- **Temporal Attention**: Shows which video frames the LSTM weighted most heavily

---

## 🖥️ Hardware Requirements

| Component | Minimum | Used in This Project |
|-----------|---------|---------------------|
| GPU | 4GB VRAM | RTX 3050 6GB |
| RAM | 8GB | 16GB |
| Storage | 5GB | - |
| OS | Windows/Linux | Windows 11 |




---
*GPU-accelerated on NVIDIA RTX 3050 6GB | PyTorch 2.2 | CUDA 12.x*
