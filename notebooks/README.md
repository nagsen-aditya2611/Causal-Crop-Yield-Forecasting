# Notebooks Documentation

# Crop Yield Forecasting using Causal Inference and Machine Learning

This folder contains the complete notebook pipeline for building the crop yield forecasting dataset used in this project.

The objective is to transform multiple raw datasets into a single machine-learning-ready dataset containing seasonal weather, soil, vegetation and geographical information for every district-year observation.

---

# Notebook Structure

| Notebook | Purpose | Status |
|----------|----------|--------|
| 01_dataset_understanding.ipynb | Data loading and exploration | ✅ Completed |
| 02_dataset_preprocessing.ipynb | Data preprocessing and feature engineering | ✅ Completed |
| 03_exploratory_data_analysis.ipynb | Exploratory Data Analysis | ⏳ Upcoming |
| 04_causal_graph.ipynb | Expert DAG construction | ⏳ Upcoming |
| 05_causal_discovery.ipynb | PC / PCMCI algorithms | ⏳ Upcoming |
| 06_causal_inference.ipynb | Treatment effect estimation | ⏳ Upcoming |
| 07_machine_learning.ipynb | ML models | ⏳ Upcoming |
| 08_interpretability.ipynb | SHAP Analysis | ⏳ Upcoming |

---

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

# Pipeline Completed So Far

```
Raw Datasets
      │
      ▼
Date Processing
      │
      ▼
Time Feature Creation
      │
      ▼
Common Time Filtering
      │
      ▼
Yield Selection
      │
      ▼
Calendar Merge
      │
      ▼
Soil Merge
      │
      ▼
Location Merge
      │
      ▼
Crop Mask Merge
      │
      ▼
Seasonal Weather Features
      │
      ▼
Seasonal Soil Moisture Features
      │
      ▼
Seasonal NDVI Features
      │
      ▼
Seasonal FPAR Features
      │
      ▼
Missing Value Handling
      │
      ▼
Final ML Dataset
```

---

# Current Status

✅ Notebook 01 Completed

✅ Notebook 02 Completed

⬜ Notebook 03 (Exploratory Data Analysis) is the next stage.