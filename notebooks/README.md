# Notebooks Documentation

# Crop Yield Forecasting using Causal Inference and Machine Learning

This folder contains the complete notebook pipeline for building the crop yield forecasting dataset used in this project.

The objective is to transform multiple raw datasets into a single machine-learning-ready dataset containing seasonal weather, soil, vegetation and geographical information for every district-year observation.

---

# Notebook Structure

| Notebook | Purpose | Status |
|----------|----------|--------|
| 01_dataset_understanding.ipynb | Dataset Understanding | ✅ Completed |
| 02_dataset_preprocessing.ipynb | Data Preprocessing & Feature Engineering | ✅ Completed |
| 03_exploratory_data_analysis.ipynb | Exploratory Data Analysis | ✅ Completed |
| 04_expert_causal_dag.ipynb | Expert DAG Construction | ✅ Completed |
| 05_causal_discovery.ipynb | Data-driven Causal Discovery (PC & NOTEARS) | ✅ Completed |
| 06_treatment_outcome_selection.ipynb | Treatment, Outcome & Confounder Selection | ✅ Completed |
| 07_causal_effect_estimation.ipynb | Causal Effect Estimation | ✅ Completed |
| 08_causal_feature_selection.ipynb | Causal Feature Selection | ⏳ Upcoming |
| 09_machine_learning_models.ipynb | Machine Learning Models | ⏳ Upcoming |
| 10_model_explainability.ipynb | Explainability (SHAP) | ⏳ Upcoming |

# Notebook 01 : Dataset Understanding

## Objective

Understand every dataset provided in CY-Bench before any preprocessing.

---

## Datasets Loaded

- Yield
- Weather
- Soil
- Soil Moisture
- NDVI
- FPAR
- Location
- Crop Calendar
- Crop Mask

---

## Initial Checks Performed

### Dataset Shapes

Verified the number of observations and variables for every dataset.

### Data Types

Checked:

- Numerical variables
- Categorical variables
- Date variables

---

## Date Conversion

Converted all integer date columns into pandas datetime format.

Example

```
20030101

↓

2003-01-01
```

---

## Time Features Created

For all temporal datasets the following variables were created

- Year
- Month
- Day of Year

These variables are required for seasonal feature extraction.

---

## Common Time Period Identification

The available time ranges of every dataset were compared.

The common period was found to be

**2003–2017**

Therefore all analyses are restricted to this interval.

---

## Yield Dataset Filtering

The yield dataset was filtered to

- Wheat only
- India only
- Harvest years 2003–2017

This resulted in

**6823 district-year observations**

---

## Output of Notebook 01

Prepared raw datasets with consistent date formats and identified the common analysis period.

---

# Notebook 02 : Dataset Preprocessing & Feature Engineering

## Objective

Create a machine-learning-ready dataset by integrating all available data sources.

---

# Step 1

## Merge Crop Calendar

Merged

```
Yield
+

Crop Calendar
```

using

```
adm_id
```

This provided

- Start of Season (SOS)
- End of Season (EOS)

for every district.

---

# Step 2

## Merge Soil Dataset

Merged

```
Yield + Calendar

+

Soil
```

Added

- Available Water Capacity
- Bulk Density
- Drainage Class

---

# Step 3

## Merge Location Dataset

Added

- Latitude
- Longitude
- Region Area

---

# Step 4

## Merge Crop Mask Dataset

Added

- Crop Area
- Crop Area Percentage

These variables quantify wheat cultivation intensity in every district.

---

# Step 5

## Missing Value Investigation

Detected missing static information.

The investigation revealed

- Only **9 districts** were affected.
- These districts were absent from the Crop Mask dataset.
- The remaining datasets contained complete information.

---

# Step 6

## Seasonal Weather Feature Engineering

Instead of using daily weather observations, seasonal summaries were created.

For every district-year

Weather between

```
SOS

↓

EOS
```

was extracted.

The growing season correctly handles seasons crossing calendar years.

For every season the following variables were calculated

- Average Minimum Temperature
- Average Maximum Temperature
- Average Mean Temperature
- Average Solar Radiation
- Average Reference Evapotranspiration
- Average Vapor Pressure Deficit
- Average Climatic Water Balance

---

## Performance Optimization

A grouped dictionary of districts was created before looping.

This reduced execution time from approximately

20–40 minutes

to

less than 1 minute.

---

# Step 7

## Seasonal Soil Moisture Features

Seasonal averages were computed for

- Surface Soil Moisture
- Root Zone Soil Moisture

---

# Step 8

## Seasonal NDVI Feature

Calculated

Average NDVI

during the crop growing season.

NDVI measures crop greenness and vegetation vigor.

---

# Step 9

## Seasonal FPAR Feature

Calculated

Average FPAR

during the crop growing season.

FPAR represents the fraction of incoming photosynthetically active radiation absorbed by vegetation.

---

# Step 10

## Dataset Cleaning

Removed columns with no usable information

Examples

- planting_date
- planting_year
- season_name
- harvest_date
- planted_area

Removed observations with incomplete seasonal information.

This eliminated the 108 incomplete district-year records.

---

## Final Dataset

Rows

```
6715
```

Columns

```
28
```

Duplicate Rows

```
0
```

The final dataset contains

- Yield
- Static soil properties
- Geographic variables
- Crop area variables
- Seasonal weather summaries
- Seasonal soil moisture summaries
- Seasonal vegetation indices

---

## Output of Notebook 02

A clean district-year level dataset ready for

- Exploratory Data Analysis
- Causal Graph Construction
- Machine Learning
- Explainability

---

## Notebook 3 – Exploratory Data Analysis

### Objectives

- Understand dataset structure
- Analyze missing values
- Explore feature distributions
- Detect outliers
- Study relationships among variables
- Examine temporal trends
- Analyze spatial distribution
- Generate publication-quality figures

### Outputs

- Summary statistics
- Correlation matrix
- Yield correlation table
- 40+ EDA figures
- Ready for feature engineering

# Notebook 04 – Expert Causal DAG Construction

## Objective

Construct an expert-defined Directed Acyclic Graph (DAG) representing the assumed causal relationships between climatic, soil, vegetation and agricultural variables affecting wheat yield.

This notebook establishes the causal assumptions that will be used throughout the remainder of the project. Unlike machine learning models, causal inference requires explicit assumptions about the data-generating process. The Expert DAG serves as the primary causal model for estimating treatment effects in later notebooks.

---

## Dataset

Input:

```
data/processed/final_crop_dataset.csv
```

Output:

```
figures/figures_dag/
reports/
```

---

## Workflow

### 1. Load Processed Dataset

- Import cleaned dataset
- Verify dataset dimensions
- Display available variables

---

### 2. Define Causal Variables

Selected variables include:

- Treatment Variable
  - Average Surface Soil Moisture (avg_ssm)

- Outcome Variable
  - Crop Yield (yield)

- Confounding Variables
  - Temperature
  - Radiation
  - Climatic Water Balance
  - Soil Characteristics
  - Geographic Variables
  - Crop Area Percentage
  - Harvest Year

---

### 3. Construct Expert DAG

The causal graph was manually designed using domain knowledge from agricultural science and causal inference literature.

Major assumptions include:

- Climate influences soil moisture.
- Soil properties affect water availability.
- Vegetation indices depend on climatic conditions.
- Soil moisture influences vegetation health.
- Vegetation condition influences crop yield.

---

### 4. DAG Visualization

The notebook generates a publication-quality visualization of the expert causal graph.

Outputs:

```
figures/figures_dag/
```

---

### Deliverables

- Expert DAG figure
- Variable definitions
- Treatment and outcome specification
- Confounder identification
- Causal assumptions documentation

---

# Notebook 05 – Data-Driven Causal Discovery

## Objective

Discover causal relationships directly from observational data and compare the learned structures with the Expert DAG.

Two complementary causal discovery algorithms are implemented:

- PC Algorithm
- NOTEARS Algorithm

The purpose of this notebook is to validate and compare learned causal structures rather than replace expert knowledge.

---

## Dataset

Input:

```
data/processed/final_crop_dataset.csv
```

Output:

```
figures/figures_causal_discovery/
reports/
```

---

## Workflow

### 1. Data Preparation

- Load processed dataset
- Select variables for causal discovery
- Standardize numerical features
- Export variable mapping

---

### 2. PC Algorithm

Constraint-based causal discovery using conditional independence testing.

Generated outputs:

- Learned causal graph
- Adjacency matrix
- Edge list
- Graph statistics

Saved reports:

```
pc_algorithm_edges.csv
pc_adjacency_matrix.csv
decoded_pc_edges.csv
```

---

### 3. NOTEARS Algorithm

Optimization-based causal discovery using continuous optimization.

Generated outputs:

- Learned DAG
- Binary adjacency matrix
- Edge list
- Network visualization
- Graph statistics

Saved reports:

```
notears_adjacency_matrix.csv
notears_edges.csv
```

---

### 4. Graph Visualization

Publication-quality graphs are generated for:

- PC Algorithm DAG
- NOTEARS DAG

Important nodes such as:

- Yield
- Surface Soil Moisture

are highlighted for better interpretability.

---

### 5. Graph Comparison

A comparison table summarizes:

- Expert DAG
- PC Algorithm
- NOTEARS Algorithm

Metrics include:

- Number of Nodes
- Number of Edges
- Graph Type
- Discovery Method
- Intended Use

---

### 6. Interpretation

The notebook evaluates:

- Agronomically meaningful relationships
- Missing causal links
- Spurious relationships
- Differences between learned and expert graphs

Based on this comparison, the Expert DAG is selected as the primary causal graph for downstream causal effect estimation.

---

### Deliverables

Figures

```
PC Learned DAG
NOTEARS Learned DAG
```

Reports

```
Variable Mapping
PC Edge List
Decoded PC Edge List
NOTEARS Edge List
Adjacency Matrices
Graph Statistics
Comparison Table
Causal Discovery Summary
```

---

## Key Outcome

Although the data-driven algorithms successfully recovered several meaningful causal relationships (e.g., climatic variables influencing soil moisture and vegetation), they failed to identify some important expert-defined pathways such as the direct influence of soil moisture on crop yield.

Therefore, the Expert DAG is retained as the primary causal model for the subsequent causal inference analysis using DoWhy and EconML.

# Notebook 06 – Treatment & Outcome Selection

## Objective

Select the treatment variable, outcome variable and adjustment variables using the Expert Causal DAG.

This notebook translates the causal assumptions into variables that can be directly used for causal effect estimation.

---

## Treatment Variable

Average Surface Soil Moisture


---

## Outcome Variable

Crop Yield


---

## Confounding Variables

The following variables are adjusted to satisfy the backdoor criterion:

- Average Temperature
- Solar Radiation
- Climate Water Balance
- Available Water Capacity
- Bulk Density
- Drainage Class
- Harvest Year

---

## Output

- Treatment variable
- Outcome variable
- Confounder list
- Final adjustment set

These variables are directly used in Notebook 07.

# Notebook 07 – Causal Effect Estimation

## Objective

Estimate the causal effect of seasonal surface soil moisture on wheat yield using multiple causal inference methods.

This notebook evaluates whether increasing soil moisture causes an increase or decrease in crop yield after adjusting for confounding variables.

---

## Dataset

Input


---

## Workflow

### 1. Build Causal Model

Using DoWhy:

- Treatment
- Outcome
- Confounders

---

### 2. Identify Causal Estimand

Estimate the Average Treatment Effect (ATE) under the backdoor adjustment criterion.

---

### 3. Linear Regression Adjustment

Baseline causal estimator using linear regression adjustment.

Output

- Average Treatment Effect

---

### 4. Propensity Score Estimation

Estimate treatment probabilities using Logistic Regression.

Generated outputs

- Propensity Scores
- Propensity Score Distribution

---

### 5. Propensity Score Matching (PSM)

Nearest-neighbour matching was performed to create balanced treatment and control groups.

Generated outputs

- Matched Dataset
- Estimated PSM ATE

---

### 6. Love Plot

Evaluate covariate balance before and after matching using Standardized Mean Differences (SMD).

Generated output

- Love Plot

---

### 7. Inverse Probability Weighting (IPW)

Estimate treatment effects using stabilized inverse probability weights.

Generated outputs

- Stabilized Weights
- Weighted ATE

---

### 8. Augmented Inverse Probability Weighting (AIPW)

Estimate doubly robust treatment effects combining outcome modelling and propensity weighting.

Generated output

- AIPW Estimate

---

### 9. Double Machine Learning (LinearDML)

Estimate heterogeneous treatment effects using EconML.

Generated outputs

- Average Treatment Effect
- Individual Treatment Effects (ITE)

---

### 10. Refutation Tests

Validate causal estimates using DoWhy robustness checks.

Implemented

- Random Common Cause
- Placebo Treatment
- Data Subset Refuter

---

### 11. Method Comparison

Compare causal estimates obtained from

- Linear Regression
- PSM
- IPW
- LinearDML

Generated output

- ATE Comparison Figure

---

### Deliverables

Figures

- Propensity Score Distribution
- Love Plot
- ATE Comparison
- ITE Distribution

Reports

- ATE Estimates
- PSM Results
- IPW Results
- AIPW Results
- LinearDML Results
- Refutation Results

---

## Key Outcome

Multiple causal inference methods consistently indicate that seasonal surface soil moisture has a measurable causal effect on crop yield.

The robustness of the estimates is supported by successful refutation tests and comparison across several causal estimation techniques.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- DoWhy
- EconML
- NetworkX
- Graphviz
- Pydot
- CausalLearn
- NOTEARS
- XGBoost