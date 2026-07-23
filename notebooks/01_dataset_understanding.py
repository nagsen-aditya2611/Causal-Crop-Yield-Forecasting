import pandas as pd
import numpy as np
import os

import os

print("Current Working Directory:")
print(os.getcwd())

print("\nDoes data folder exist?")
print(os.path.exists("../data"))

print("\nDoes raw folder exist?")
print(os.path.exists("../data/raw"))



print("Libraries imported successfully")


import os

# Folder containing this Python file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Project root (go up one level from notebooks/)
project_root = os.path.dirname(script_dir)

# Raw data folder
data_path = os.path.join(project_root, "data", "raw")

print("Dataset Path:")
print(data_path)

files = os.listdir(data_path)

files = os.listdir(data_path)

print("\nFiles found:\n")

for file in files:
    print(file)


# Load the yield dataset
yield_df = pd.read_csv(os.path.join(data_path, "yield_wheat_IN.csv"))

print("Yield dataset loaded successfully!\n")

print("Shape of the dataset:")
print(yield_df.shape)

print("\nColumn Names:\n")
print(yield_df.columns)

print("\nFirst five rows:\n")
print(yield_df.head())

print("\nDataset Information")
print("-" * 50)

yield_df.info()

print("\nSummary Statistics")
print("-" * 50)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(yield_df.describe(include = "all").T)

print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

# Missing values
print("\nMissing Values")
print("-" * 60)
print(yield_df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows")
print("-" * 60)
print(yield_df.duplicated().sum())

# Zero values in important numerical columns
print("\nZero Values")
print("-" * 60)

for col in ["yield", "production", "harvest_area"]:
    zeros = (yield_df[col] == 0).sum()
    print(f"{col}: {zeros}")

print("\n" + "=" * 60)
print("UNIQUE VALUES IN CATEGORICAL COLUMNS")
print("=" * 60)

categorical_cols = ["crop_name", "country_code", "adm_id"]

for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(f"Number of unique values: {yield_df[col].nunique()}")

    if yield_df[col].nunique() <= 20:
        print(yield_df[col].unique())

# =====================================
# WEATHER DATASET
# =====================================

print("\n" + "="*60)
print("WEATHER DATASET")
print("="*60)

meteo_df = pd.read_csv(
    os.path.join(data_path, "meteo_wheat_IN.csv")
)

print("Weather dataset loaded successfully!")

print("\nShape of Weather Dataset")
print("-"*50)

print(meteo_df.shape)

print("\nColumns")
print("-"*50)

print(meteo_df.columns)

print("\nFirst Five Rows")
print("-"*50)

print(meteo_df.head())

print("\nDataset Information")
print("-" * 50)
meteo_df.info()

print("\nSummary Statistics")
print("-" * 50)
print(meteo_df.describe(include="all").T)

print("\nMissing Values")
print("-" * 50)
print(meteo_df.isnull().sum())

print("\nDuplicate Rows")
print("-" * 50)
print(meteo_df.duplicated().sum())

# =====================================
# SOIL DATASET
# =====================================

print("\n" + "=" * 60)
print("SOIL DATASET")
print("=" * 60)

soil_df = pd.read_csv(
    os.path.join(data_path, "soil_wheat_IN.csv")
)

print("Soil dataset loaded successfully!")

print("\nShape")
print("-" * 50)
print(soil_df.shape)

print("\nColumns")
print("-" * 50)
print(soil_df.columns)

print("\nFirst Five Rows")
print("-" * 50)
print(soil_df.head())

print("\nDataset Information")
print("-" * 50)
soil_df.info()

print("\nSummary Statistics")
print("-" * 50)
print(soil_df.describe(include="all").T)

print("\nMissing Values")
print("-" * 50)
print(soil_df.isnull().sum())

print("\nDuplicate Rows")
print("-" * 50)
print(soil_df.duplicated().sum())

# =====================================
# SOIL MOISTURE DATASET
# =====================================

print("\n" + "=" * 60)
print("SOIL MOISTURE DATASET")
print("=" * 60)

soil_moisture_df = pd.read_csv(
    os.path.join(data_path, "soil_moisture_wheat_IN.csv")
)

print("Soil Moisture dataset loaded successfully!")

# Shape
print("\nShape")
print("-" * 50)
print(soil_moisture_df.shape)

# Columns
print("\nColumns")
print("-" * 50)
print(soil_moisture_df.columns)

# First five rows
print("\nFirst Five Rows")
print("-" * 50)
print(soil_moisture_df.head())

# Dataset Information
print("\nDataset Information")
print("-" * 50)
soil_moisture_df.info()

# Summary Statistics
print("\nSummary Statistics")
print("-" * 50)
print(soil_moisture_df.describe(include="all").T)

# Missing Values
print("\nMissing Values")
print("-" * 50)
print(soil_moisture_df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows")
print("-" * 50)
print(soil_moisture_df.duplicated().sum())

# =====================================
# NDVI DATASET
# =====================================

print("\n" + "=" * 60)
print("NDVI DATASET")
print("=" * 60)

ndvi_df = pd.read_csv(
    os.path.join(data_path, "ndvi_wheat_IN.csv")
)

print("NDVI dataset loaded successfully!")

print("\nShape")
print("-" * 50)
print(ndvi_df.shape)

print("\nColumns")
print("-" * 50)
print(ndvi_df.columns)

print("\nFirst Five Rows")
print("-" * 50)
print(ndvi_df.head())

print("\nDataset Information")
print("-" * 50)
ndvi_df.info()

print("\nSummary Statistics")
print("-" * 50)
print(ndvi_df.describe(include="all").T)

print("\nMissing Values")
print("-" * 50)
print(ndvi_df.isnull().sum())

print("\nDuplicate Rows")
print("-" * 50)
print(ndvi_df.duplicated().sum())


# =====================================
# FPAR DATASET
# =====================================

print("\n" + "=" * 60)
print("FPAR DATASET")
print("=" * 60)

fpar_df = pd.read_csv(
    os.path.join(data_path, "fpar_wheat_IN.csv")
)

print("FPAR dataset loaded successfully!")

# Shape
print("\nShape")
print("-" * 50)
print(fpar_df.shape)

# Columns
print("\nColumns")
print("-" * 50)
print(fpar_df.columns)

# First Five Rows
print("\nFirst Five Rows")
print("-" * 50)
print(fpar_df.head())

# Dataset Information
print("\nDataset Information")
print("-" * 50)
fpar_df.info()

# Summary Statistics
print("\nSummary Statistics")
print("-" * 50)
print(fpar_df.describe(include="all").T)

# Missing Values
print("\nMissing Values")
print("-" * 50)
print(fpar_df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows")
print("-" * 50)
print(fpar_df.duplicated().sum())

# =====================================
# LOCATION DATASET
# =====================================

print("\n" + "=" * 60)
print("LOCATION DATASET")
print("=" * 60)

location_df = pd.read_csv(
    os.path.join(data_path, "location_wheat_IN.csv")
)

print("Location dataset loaded successfully!")

# Shape
print("\nShape")
print("-" * 50)
print(location_df.shape)

# Columns
print("\nColumns")
print("-" * 50)
print(location_df.columns)

# First Five Rows
print("\nFirst Five Rows")
print("-" * 50)
print(location_df.head())

# Dataset Information
print("\nDataset Information")
print("-" * 50)
location_df.info()

# Summary Statistics
print("\nSummary Statistics")
print("-" * 50)
print(location_df.describe(include="all").T)

# Missing Values
print("\nMissing Values")
print("-" * 50)
print(location_df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows")
print("-" * 50)
print(location_df.duplicated().sum())

# =====================================
# CROP CALENDAR DATASET
# =====================================

print("\n" + "=" * 60)
print("CROP CALENDAR DATASET")
print("=" * 60)

calendar_df = pd.read_csv(
    os.path.join(data_path, "crop_calendar_wheat_IN.csv")
)

print("Crop Calendar dataset loaded successfully!")

# Shape
print("\nShape")
print("-" * 50)
print(calendar_df.shape)

# Columns
print("\nColumns")
print("-" * 50)
print(calendar_df.columns)

# First Five Rows
print("\nFirst Five Rows")
print("-" * 50)
print(calendar_df.head())

# Dataset Information
print("\nDataset Information")
print("-" * 50)
calendar_df.info()

# Summary Statistics
print("\nSummary Statistics")
print("-" * 50)
print(calendar_df.describe(include="all").T)

# Missing Values
print("\nMissing Values")
print("-" * 50)
print(calendar_df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows")
print("-" * 50)
print(calendar_df.duplicated().sum())

# =====================================
# CROP MASK DATASET
# =====================================

print("\n" + "=" * 60)
print("CROP MASK DATASET")
print("=" * 60)

mask_df = pd.read_csv(
    os.path.join(data_path, "crop_mask_wheat_IN.csv")
)

print("Crop Mask dataset loaded successfully!")

# Shape
print("\nShape")
print("-" * 50)
print(mask_df.shape)

# Columns
print("\nColumns")
print("-" * 50)
print(mask_df.columns)

# First Five Rows
print("\nFirst Five Rows")
print("-" * 50)
print(mask_df.head())

# Dataset Information
print("\nDataset Information")
print("-" * 50)
mask_df.info()

# Summary Statistics
print("\nSummary Statistics")
print("-" * 50)
print(mask_df.describe(include="all").T)

# Missing Values
print("\nMissing Values")
print("-" * 50)
print(mask_df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows")
print("-" * 50)
print(mask_df.duplicated().sum())