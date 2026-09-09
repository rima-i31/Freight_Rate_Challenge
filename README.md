# Freight Rate Prediction

Machine Learning solution for predicting freight shipping rates based on load characteristics.

## Overview

This project solves a freight rate prediction task using supervised machine learning. The goal is to predict shipping costs for unseen loads based on historical data and shipment features.

## 📁 Repository Structure
```
Freight_Rate_Prediction/
├── data/
│   ├── december-chart-inputs.csv    # December predictions input
│   ├── train-test.csv               # Development dataset (private)
│   ├── validation.csv               # Final predictions input (private)
│   └── validation-predictions.csv   # Generated predictions
├── notebooks/
│   ── 01_eda_and_modeling.ipynb    # Exploratory analysis & experiments
├── outputs/                          # Model outputs and artifacts
├── reports/                          # Final report and visualizations
├── scorer/
│   └── score.py                      # Official evaluation script
├── src/
│   ├── data_preparation.py          # Data loading and cleaning
│   ├── features.py                   # Feature engineering
│   ├── train.py                      # Model training pipeline
│   └── predict.py                    # Prediction generation
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Privacy

**Note:** Training and validation datasets (`train-test.csv`, `validation.csv`) contain proprietary information and are not included in this public repository to maintain data privacy and confidentiality. These files are provided separately as part of the assessment.

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/freight-rate-prediction.git
cd freight-rate-prediction

# Install dependencies
pip install -r requirements.txt