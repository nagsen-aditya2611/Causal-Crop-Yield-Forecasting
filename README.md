# Causal Crop Yield Forecasting

An end-to-end research project that integrates **causal inference** and **machine learning** for crop yield forecasting using the **CY-Bench** dataset (India subset).

The project investigates whether environmental and soil variables **causally influence** crop yield, rather than relying solely on correlation-based predictive models.

---

## Project Objectives

- Construct an expert agricultural causal graph (DAG)
- Discover causal relationships using data-driven algorithms
- Estimate causal effects using multiple causal inference methods
- Identify causally relevant features for prediction
- Build machine learning models using causal insights
- Improve interpretability and decision support for precision agriculture

---

## Dataset

**Dataset:** CY-Bench (Crop Yield Benchmark)

**Country:** India

The project uses agricultural, climatic, soil and vegetation variables including:

- Surface Soil Moisture (Treatment)
- Crop Yield (Outcome)
- Temperature
- Solar Radiation
- Climate Water Balance
- Available Water Capacity
- Bulk Density
- Drainage Class
- NDVI
- FPAR

> **Note:** The raw dataset is intentionally excluded from this repository because of its large size. A README inside `data/raw/` explains how to obtain the dataset.

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
04 Expert Causal DAG
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
09 Machine Learning Models
   │
   ▼
10 Model Explainability (SHAP)
```

---

# Completed Work

## Dataset Analysis

- Dataset Understanding
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis

---

## Causal Analysis

- Expert Agricultural DAG
- Data-driven Causal Discovery
- Treatment & Outcome Selection
- Confounder Identification

---

## Causal Effect Estimation

Implemented methods:

- DoWhy
- Linear Regression Adjustment
- Propensity Score Matching (PSM)
- Inverse Probability Weighting (IPW)
- Augmented IPW (AIPW)
- Double Machine Learning (LinearDML)

---

## Model Validation

Robustness checks include:

- Random Common Cause Refuter
- Placebo Treatment Refuter
- Data Subset Refuter

---

## Visualizations

Generated visualizations include:

- Expert DAG
- Propensity Score Distribution
- Love Plot
- Average Treatment Effect Comparison
- Individual Treatment Effect Distribution

---

# Repository Structure

```
Causal-Crop-Yield-Forecasting/

├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── figures/
│
├── reports/
│
├── models/
│
├── references/
│
├── src/
│
└── README.md
```

---

# Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Causal Inference

- DoWhy
- EconML
- CausalLearn

### Graph Analysis

- NetworkX
- Graphviz
- Pydot

---

# Current Status

| Module | Status |
|---------|:------:|
| Dataset Understanding | ✅ |
| Data Preprocessing | ✅ |
| Exploratory Data Analysis | ✅ |
| Expert DAG | ✅ |
| Causal Discovery | ✅ |
| Treatment & Outcome Selection | ✅ |
| Causal Effect Estimation | ✅ |
| Causal Feature Selection | ⏳ |
| Machine Learning Models | ⏳ |
| Explainability | ⏳ |

---

# Future Work

- Causal Feature Selection
- Random Forest
- XGBoost
- LightGBM
- SHAP Explainability
- Model Comparison
- Final Performance Evaluation

---

Research Area: Causal Inference • Machine Learning • Agricultural Analytics
