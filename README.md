# Causal Crop Yield Forecasting

**Integrating Causal Inference, Machine Learning, and Deep Learning for Explainable Crop Yield Prediction**

An end-to-end research project that combines **causal inference**, **machine learning**, **deep learning**, and **explainable AI** to improve crop yield forecasting using the **CY-Bench (India subset)** dataset.

Unlike traditional forecasting methods that rely purely on correlations, this project first identifies **causal relationships** among agricultural variables and then evaluates whether causally selected features can produce competitive predictive performance.

---

# Project Objectives

- Build an expert agricultural causal graph (DAG)
- Discover causal relationships from observational data
- Estimate Average Treatment Effects (ATE)
- Validate causal assumptions using robustness checks
- Select causally relevant predictive features
- Compare traditional Machine Learning models
- Compare Deep Learning models
- Interpret predictions using SHAP Explainability
- Evaluate whether causal feature selection improves model interpretability while maintaining predictive performance

---

# Dataset

**Dataset:** CY-Bench (Crop Yield Benchmark)

**Region:** India

The dataset contains agricultural, climatic, vegetation, and soil-related variables, including:

- Crop Yield (Target)
- Harvest Area
- Crop Area
- Surface Soil Moisture
- Root Zone Soil Moisture
- Temperature
- Solar Radiation
- Climate Water Balance
- NDVI
- FPAR
- Available Water Capacity
- Bulk Density
- Drainage Class
- Latitude
- Longitude
- Harvest Year

> **Note:** The raw CY-Bench dataset is intentionally excluded from this repository because of its large size and licensing. Instructions for downloading the dataset are provided inside `data/raw/`.

---

# Project Workflow

```text
Dataset
   │
   ▼
01 Dataset Understanding
   │
   ▼
02 Data Cleaning & Preprocessing
   │
   ▼
03 Exploratory Data Analysis
   │
   ▼
04 Expert Agricultural DAG
   │
   ▼
05 Causal Discovery
   │
   ▼
06 Treatment & Outcome Selection
   │
   ▼
07 Causal Effect Estimation
   │
   ▼
08 Causal Feature Selection
   │
   ▼
09 Traditional Machine Learning
      ├── Linear Regression
      ├── Random Forest
      ├── XGBoost
      └── LightGBM
   │
   ▼
10 Deep Learning
      ├── MLP
      └── LSTM
   │
   ▼
11 SHAP Explainability
   │
   ▼
12 Model Comparison
   │
   ▼
13 Project Summary
```

---

# Project Highlights

## Exploratory Data Analysis

- Data quality assessment
- Missing value analysis
- Correlation analysis
- Distribution analysis
- Feature engineering
- Agricultural variable exploration

---

## Causal Inference

Implemented:

- Expert Agricultural DAG
- Causal Discovery
- Treatment Selection
- Outcome Selection
- Confounder Identification

---

## Average Treatment Effect Estimation

The following causal inference methods were implemented:

- Linear Regression Adjustment
- Propensity Score Matching (PSM)
- Inverse Probability Weighting (IPW)
- Augmented IPW (AIPW)
- Double Machine Learning (LinearDML)

---

## Refutation Analysis

To validate causal estimates, the following robustness checks were performed:

- Random Common Cause Refuter
- Placebo Treatment Refuter
- Data Subset Refuter

---

## Machine Learning Models

Traditional Machine Learning models:

- Linear Regression
- Random Forest
- XGBoost
- LightGBM

---

## Deep Learning Models

Neural Network models:

- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)

---

## Explainable AI

Model interpretation performed using:

- SHAP Summary Plot
- SHAP Feature Importance
- Global Feature Importance Analysis

---

# Key Results

### Causal Analysis

- Estimated treatment effects using multiple causal inference methods.
- Validated causal assumptions through robustness testing.
- Identified causally important predictors for crop yield forecasting.

### Machine Learning

Best traditional model:

**XGBoost**

- R² ≈ **0.904**

### Deep Learning

Best neural network:

**LSTM**

- R² ≈ **0.874**

### Explainability

SHAP analysis identified the most influential variables:

- Crop Area Percentage
- Latitude
- Longitude
- Harvest Year
- Start of Season (SOS)
- Harvest Area

---

# Repository Structure

```
data/
├── raw/
└── processed/

notebooks/
src/
models/
reports/
figures/
references/
```

**Status:** 🚧 Project setup in progress.