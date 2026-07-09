# Asthma Attack Prediction Using Deep Learning on Multimodal Time-Series Data

## Overview

This repository contains the implementation of an undergraduate thesis that focuses on predicting asthma attack risk using multimodal time-series data. The proposed approach combines physiological signals collected from wearable devices with environmental information to improve prediction performance.

The project includes data preprocessing, feature preparation, deep learning model development, and comparison with conventional machine learning models.

---

## Project Structure

```
asthma-ta/
│
├── preprocessing/
│   ├── create_label.py
│   ├── explore_questionnaire.py
│   ├── explore_smartwatch.py
│   ├── merge_environment.py
│   ├── merge_label.py
│   ├── prepare_smartwatch.py
│   └── window_smartwatch.py
│
├── training/
│   ├── baseline_rf.py
│   ├── baseline_xgb.py
│   ├── model.py
│   ├── prepare_data.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **Asthma Attack Monitoring with Smartwatch (AAMOS)** dataset provided by the University of Edinburgh.

Dataset:
https://datashare.ed.ac.uk/handle/10283/4761

The dataset contains:

- Heart rate measurements
- Step count
- Physical activity records
- Environmental information
- Daily questionnaire responses
- Asthma symptom records

The dataset is not included in this repository due to licensing restrictions and repository size limitations.

---

## Methodology

The workflow of this project consists of the following stages:

1. Smartwatch data preprocessing
2. Questionnaire preprocessing
3. Time-series window generation
4. Label generation
5. Environmental data integration
6. Model training
7. Performance evaluation

The proposed deep learning model is based on a CNN-LSTM architecture and is compared with two baseline machine learning models:

- Random Forest
- XGBoost

---

## Installation

Clone this repository:

```bash
git clone https://github.com/FikoYoris/Asthma-Attack-Prediction-Using-Deep-Learning-on-Multimodal-Time-Series-Data.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run the deep learning model using:

```bash
python training/train.py
```

To train the baseline models:

```bash
python training/baseline_rf.py
```

```bash
python training/baseline_xgb.py
```

---

## Evaluation

The models are evaluated using several classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Notes

This repository only contains the implementation of the proposed method. The dataset is distributed separately by its original publisher and must be downloaded before running the preprocessing and training scripts.

---

## Author

**Fiko Yorisdwi 'Aliy**

Bachelor of Informatics

Telkom University
