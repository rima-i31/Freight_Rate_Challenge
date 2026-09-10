# Freight Rate Prediction

Machine learning solution for predicting freight shipping rates from historical load and shipment characteristics.

## Overview

This repository contains the solution for the Freight Rate Prediction Challenge.

The objective is to train a regression model using historical freight data and generate rate predictions for unseen loads.

The solution includes:

- exploratory data analysis and data-quality checks;
- temporal train/validation splitting;
- missing-value and invalid-value handling;
- baseline and model comparison experiments;
- validation error analysis;
- final XGBoost regression model;
- reproducible training and prediction pipelines;
- generation of the required validation and December predictions.

The complete exploratory analysis and model selection process can be found in:

```text
notebooks/01_freight_rate_analysis_and_modeling.ipynb
```

## Repository Structure

```text
Freight_Rate_Challenge/
│
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
│
├── notebooks/
│   └── 01_freight_rate_analysis_and_modeling.ipynb
│
├── models/
│   ├── freight_rate_model.joblib
│   └── preprocessing_stats.joblib
│
├── reports/
│   └── freight_rate_report.pdf
│
├── scorer/
│   └── score.py
│
├── src/
│   ├── data_preparation.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── validation_predictions.csv
├── .gitignore
├── README.md
└── requirements.txt
```

Some generated files and directories are created only after running the training, prediction, and scoring pipelines.

## Data

The solution expects the following files inside the `data/` directory:

```text
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

### Development data

`train_test.csv` contains the labeled historical development data used for model development and final training.

The prediction target is:

```text
posted_rate
```

### Validation data

`validation.csv` contains the 12,000 unseen loads for which final predictions must be generated.

Each load is identified by a unique:

```text
load_id
```

### December chart data

`december_chart_inputs.csv` contains the fixed December loads used by the provided scoring script to generate the required December prediction chart.

## Data Privacy

The original training and validation datasets are not included in the public repository because they were provided specifically for the assessment.

To reproduce the solution, place the provided assessment datasets in the `data/` directory using the filenames shown above.

## Validation Strategy

Because the data has a temporal structure and the final prediction period occurs after the development period, model validation uses a chronological split rather than a random split.

The development data covers January through October 2025.

The validation strategy is:

```text
Training:    January 1 – August 31, 2025
Validation:  September 1 – October 31, 2025
```

This setup simulates the final task of predicting future freight rates while avoiding look-ahead leakage from future observations into model training.

After model selection was completed, the selected model was retrained using the complete January–October development dataset before generating the final predictions.

## Data Preparation

The preprocessing pipeline includes:

- parsing shipment dates;
- converting negative shipment weights to their absolute values;
- imputing missing weight values using the training-data median;
- handling missing market index values during general data preparation;
- creating calendar features from the shipment date.

Preprocessing statistics are fitted only on the training data during validation to prevent data leakage.

For final training, preprocessing statistics are refitted using the complete labeled development dataset.

## Features

The final model uses the following shipment information:

```text
pickup
delivery
distance
equipment
weight
date
```

Additional calendar features are derived from `date`:

```text
month
day_of_week
day_of_month
days_since_start
```

Categorical features are encoded using `OneHotEncoder` with unknown-category handling.

Additional feature-engineering experiments were evaluated during model development but did not improve temporal validation performance, so the simpler feature set was retained.

## Model Selection

Several regression approaches were evaluated using the same temporal validation split.

The experiments included:

- median prediction baseline;
- distance-only linear regression;
- Extra Trees;
- CatBoost;
- CatBoost with a log-transformed target;
- CatBoost with an extended feature set;
- XGBoost.

The final XGBoost model achieved the strongest validation MAE.

### Validation Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Median baseline | 1148.924 | 1569.424 | -0.058 |
| Distance linear regression | 196.950 | 654.416 | 0.816 |
| Extra Trees | 165.504 | 691.846 | 0.794 |
| CatBoost common | 112.824 | 635.093 | 0.827 |
| CatBoost extended | 116.455 | 636.398 | 0.826 |
| CatBoost log-target | 112.151 | 634.938 | 0.827 |
| **XGBoost** | **109.118** | **634.018** | **0.827** |

The final XGBoost model achieved a validation MAPE of approximately:

```text
4.79%
```

Error analysis showed that prediction accuracy is substantially better for the majority of loads, while a small number of unusually high freight rates account for most of the largest residual errors.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rima-i31/Freight_Rate_Challenge.git
cd Freight_Rate_Challenge
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies


```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
#### macOS

XGBoost may require the OpenMP runtime on macOS. Install it using Homebrew:

```bash
brew install libomp
```

If Homebrew is not installed, see the official Homebrew installation instructions.

#### Linux / Windows

No additional OpenMP installation is normally required.

## Running the Solution

The complete prediction workflow consists of three steps:

```text
Train model
    ↓
Generate predictions
    ↓
Run official scorer
```

### Step 1 — Prepare the data

Place the four provided assessment files in:

```text
data/
```

The directory should contain:

```text
data/
├── train_test.csv
├── validation.csv
├── validation_predictions_template.csv
└── december_chart_inputs.csv
```

### Step 2 — Train the final model

Run the training pipeline from the repository root:

```bash
python src/train.py
```

The script:

1. loads `data/train_test.csv`;
2. fits the final preprocessing statistics;
3. cleans the complete development dataset;
4. creates the model features;
5. trains the selected XGBoost model on all labeled development rows;
6. saves the trained preprocessing/model artifacts.

The generated artifacts are stored in:

```text
models/
├── freight_rate_model.joblib
└── preprocessing_stats.joblib
```

### Step 3 — Generate predictions

Run:

```bash
python src/predict.py
```

The prediction script loads the saved model and generates predictions for both required datasets.

For `data/validation.csv`, it fills the provided validation template using `load_id` and creates:

```text
validation_predictions.csv
```

The output contains exactly:

```text
load_id,predicted_rate
```

for all 12,000 validation loads.

The script also predicts every row in:

```text
data/december_chart_inputs.csv
```

and fills its `predicted_rate` column for the fixed December scoring chart.

### Step 4 — Run the official scorer

After both prediction files have been generated, run:

```bash
python scorer/score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december-chart-inputs.csv
```

The scorer validates:

- the number of validation predictions;
- expected `load_id` values;
- missing or duplicate predictions;
- numeric and positive predicted rates;
- the required December prediction structure.

If both prediction files are valid, the scorer generates:

```text
scorer_results/candidate_december.png
```

This chart is included in the final assessment report.

## Complete Run

After installing the dependencies and placing the datasets in `data/`, the complete solution can be reproduced with:

```bash
python src/train.py
python src/predict.py
python scorer/score.py \
  --predictions validation-predictions.csv \
  --december-predictions data/december-chart-inputs.csv
```

## Exploratory Analysis

The complete exploratory analysis, validation experiments, model comparisons, and error analysis are available in:

```text
notebooks/01_freight_rate_analysis_and_modeling.ipynb
```

The notebook documents the reasoning behind the final modeling decisions, while the scripts under `src/` provide the reproducible final training and prediction pipeline.

## Output Files

The main generated outputs are:

```text
validation_predictions.csv
data/december_chart_inputs.csv
scorer_results/candidate_december.png
```

`validation_predictions.csv` is the required final prediction file containing:

```text
load_id,predicted_rate
```

## Final Report

The final report is located at:

```text
reports/freight_rate_report.pdf
```

It summarizes:

- data exploration and quality issues;
- temporal validation strategy;
- preprocessing decisions;
- model comparison and selection;
- final validation performance;
- error analysis;
- the fixed December prediction chart generated by the official scorer.

## Reproducibility

The same preprocessing, feature creation, and prediction logic is shared between training and inference through the modules in `src/`.

This ensures that the final predictions can be reproduced from the provided datasets using the commands documented above.