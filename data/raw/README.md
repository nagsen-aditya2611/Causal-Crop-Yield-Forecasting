# Raw Data

This directory stores the original, unmodified datasets used in the project.

> **Note:** The raw datasets are **not included** in this GitHub repository because they exceed GitHub's file size limits and are intentionally excluded using `.gitignore`.

---

## Dataset Used

### CY-Bench Dataset

This project uses the **CY-Bench (Crop Yield Benchmark)** dataset.

CY-Bench is a large-scale benchmark dataset developed for crop yield forecasting and agricultural machine learning research.

It combines information from multiple sources, including:

- Crop yield records
- Weather variables
- Soil characteristics
- Satellite observations
- Geographic information
- Temporal information

---

## Country Used

For this project, only the **India** subset of CY-Bench is used.

The analysis focuses on Indian agricultural regions to study the causal relationship between environmental conditions and crop yield.

---

## Main Variables

### Weather Variables

- `avg_tavg` – Average temperature
- `avg_tmin` – Minimum temperature
- `avg_tmax` – Maximum temperature
- `avg_rad` – Solar radiation
- `avg_cwb` – Climate water balance

---

### Soil Variables

- `avg_ssm` – Surface soil moisture *(Treatment Variable)*
- `awc` – Available water capacity
- `bulk_density`
- `drainage_class`

---

### Vegetation Variables

- `avg_ndvi`
- `avg_fpar`

---

### Geographic Variables

- `latitude`
- `longitude`
- `adm_id`
- `country_code`

---

### Temporal Variables

- `harvest_year`

---

### Outcome Variable

- `yield`

---

## Data Status

The files inside this folder should always remain **raw**.

No preprocessing, cleaning, feature engineering, or normalization should be performed directly on these files.

All transformations are performed in later notebooks.

---

## Expected Folder Structure

```
data/
│
├── raw/
│   ├── README.md
│   ├── cybench/
│   │   ├── labels.csv
│   │   ├── features.csv
│   │   ├── metadata.csv
│   │   └── ...
│
├── processed/
└── outputs/
```

---

## How to Obtain the Dataset

Download the CY-Bench dataset from the official project repository or source.

After downloading:

1. Extract the dataset.
2. Copy the dataset into:

```
data/raw/cybench/
```

3. Ensure the directory structure matches the one expected by the notebooks.

---

## Important

This repository contains only the **analysis pipeline**.

Users must download the raw data separately before running the notebooks.

The notebooks assume that the dataset is available under:

```
data/raw/cybench/
```

---

## Dataset Citation

If you use this project or the underlying dataset, please cite the original **CY-Bench** publication and repository according to their official citation guidelines.