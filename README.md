# spaceflight-sleep-anomaly-detection
# Spaceflight Sleep Anomaly Detection System

This repository contains the machine learning pipelines and models built to detect sleep anomalies and biomarker deviations in spaceflight contexts, utilizing the **PhysioNet** dataset alongside vital signs, blood panels, and muscle performance parameters.

## 🚀 Project Overview
Our main detector uses a **TensorFlow Autoencoder (6-4-2-4-6 architecture)** trained strictly on baseline (normal) sleep sessions. When a new sleep night is analyzed, the model evaluates it and produces a Reconstruction Mean Squared Error (MSE). If the error exceeds our tuned threshold (based on a targeted 5% false alarm rate), the system flags the night as an anomaly.

## 📊 Benchmark Performance
Our leave-people-out cross-validation benchmark shows our Autoencoder outperforms standard baselines on complex disruptions:
- **False Alarm Rate:** ~6%
- **Extreme Heart Rate Anomalies (+12 bpm):** ~57% detection rate
- **Disrupted Sleep (Insomnia/Arousal Simulations):** **45% detection rate** (significantly beating Isolation Forest at 27% and traditional separate z-scores at 29%)

## 📂 Repository Contents
- `*.ipynb`: The main Google Colab notebook containing step-by-step cleaning pipelines and the benchmark code.
- `clean_1_radiation.csv` to `clean_6_vitals.csv`: The cleaned and formatted subsets extracted from our secondary datasets.
- `final_autoencoder_model.keras`: The trained and serialized neural network model ready for deployment.
- `final_model_config.pkl`: Associated configuration containing feature definitions, population medians, and pooled standard deviations (spreads).

## ⚙️ How to Run locally
1. Clone this repository:
   ```bash
   git clone <your-repository-url>
Install required dependencies:
pip install pandas numpy tensorflow joblib scikit-learn
Load the model and preprocessing configuration to begin analyzing new sleep profiles:
import joblib
import tensorflow as tf

model = tf.keras.models.load_model('final_autoencoder_model.keras')
config = joblib.load('final_model_config.pkl')
print("Model and configuration 
