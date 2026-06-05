# 🧠 BRF-TED: Emotional State Prediction using Behavioral Rhythm Fusion and Temporal Emotional Drift

A research-oriented deep learning framework that predicts emotional states from longitudinal behavioral sensing data using Behavioral Rhythm Fusion (BRF) feature extraction and a Temporal Emotional Drift (TED) neural architecture.

---

# System Architecture

```text
StudentLife Dataset / Synthetic Data
              │
              ▼
      synthetic_data.py
              │
              ▼
         brf.py
   (Behavioral Rhythm Fusion)
              │
              ▼
      Sequence Builder
          train.py
              │
              ▼
         ted_model.py
 (LSTM + Temporal Drift Gate)
              │
              ▼
       evaluate.py
              │
              ▼
     Emotion Prediction
  (Negative / Neutral / Positive)
```

---

# Research Pipeline

```text
Behavioral Data
      │
      ▼
Feature Engineering
      │
      ▼
Behavioral Rhythm Fusion (BRF)
      │
      ▼
Personalized Normalization
      │
      ▼
Temporal Sequence Construction
      │
      ▼
TED Network
(LSTM + Drift Modeling)
      │
      ▼
Emotion Classification
      │
      ▼
Evaluation & Ablation Study
```

---

# Quick Start

## 1 — Clone Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/BRF-TED-Emotional-State-Prediction.git

cd BRF-TED-Emotional-State-Prediction
```

---

## 2 — Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4 — Run Pipeline

```bash
python main.py
```

---

# Expected Workflow

```text
Generate Synthetic Dataset
        │
        ▼
BRF Feature Extraction
        │
        ▼
Train/Test Split
        │
        ▼
Sequence Generation
        │
        ▼
TED Model Training
        │
        ▼
Prediction
        │
        ▼
Performance Evaluation
```

---

# Core Components

| Module            | Description                                  |
| ----------------- | -------------------------------------------- |
| synthetic_data.py | StudentLife-style behavioral data generation |
| brf.py            | Behavioral Rhythm Fusion feature extraction  |
| ted_model.py      | Temporal Emotional Drift neural network      |
| train.py          | Sequence building and training pipeline      |
| evaluate.py       | Metrics and evaluation utilities             |
| main.py           | End-to-end execution pipeline                |

---

# Dataset

The project supports:

### StudentLife Dataset

Official Dataset:

https://studentlife.cs.dartmouth.edu/

---

# Features

### Behavioral Rhythm Fusion (BRF)

Extracts and normalizes:

* Inter-Behavior Distance (IBD)
* Inter-Session Information Entropy (ISIE)
* Circadian Irregularity Index (CII)
* Mobile Lifestyle Usage Entropy (MLUE)

### Temporal Emotional Drift (TED)

Models emotional transitions using:

* Two-layer LSTM
* Drift Gate
* Temporal Probability Smoothing

---

# Evaluation Metrics

* Accuracy
* Macro F1 Score
* Cohen's Kappa
* ROC-AUC

---

# Repository Structure

```text
BRF-TED-Emotional-State-Prediction/
│
├── main.py
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── synthetic_data.py
│   ├── brf.py
│   ├── ted_model.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   └── BRF_TED_Colab.ipynb
│
├── assets/
│   ├── architecture.png
│   ├── workflow.png
│   └── results/
│
└── outputs/
```

---

# Future Work

* Transformer-based temporal modeling
* Real StudentLife dataset integration
* Self-supervised behavioral representation learning
* Multi-task emotional forecasting

---

# Author

Anhadbani Anand

B.Tech Computer & Communication Engineering

Manipal University Jaipur
