import pandas as pd
import numpy as np
import os

print("Libraries imported successfully.")

# Dataset folder

data_path = os.path.join(os.getcwd(), "data", "raw")

yield_df = pd.read_csv(os.path.join(data_path, "yield_wheat_IN.csv"))

weather_df = pd.read_csv(os.path.join(data_path, "meteo_wheat_IN.csv"))

soil_df = pd.read_csv(os.path.join(data_path, "soil_wheat_IN.csv"))

soil_moisture_df = pd.read_csv(os.path.join(data_path, "soil_moisture_wheat_IN.csv"))

ndvi_df = pd.read_csv(os.path.join(data_path, "ndvi_wheat_IN.csv"))

fpar_df = pd.read_csv(os.path.join(data_path, "fpar_wheat_IN.csv"))

location_df = pd.read_csv(os.path.join(data_path, "location_wheat_IN.csv"))

calendar_df = pd.read_csv(os.path.join(data_path, "crop_calendar_wheat_IN.csv"))

cropmask_df = pd.read_csv(os.path.join(data_path, "crop_mask_wheat_IN.csv"))

print("All datasets loaded successfully!")

print("\nDataset Shapes")
print("-" * 50)

print("Yield           :", yield_df.shape)
print("Weather         :", weather_df.shape)
print("Soil            :", soil_df.shape)
print("Soil Moisture   :", soil_moisture_df.shape)
print("NDVI            :", ndvi_df.shape)
print("FPAR            :", fpar_df.shape)
print("Location        :", location_df.shape)
print("Crop Calendar   :", calendar_df.shape)
print("Crop Mask       :", cropmask_df.shape)


# ============================================================
# CHECK CURRENT DATA TYPES
# ============================================================

print("\nCurrent Data Types")
print("-" * 50)

print("\nWeather")
print(weather_df.dtypes)

print("\nSoil Moisture")
print(soil_moisture_df.dtypes)

print("\nNDVI")
print(ndvi_df.dtypes)

print("\nFPAR")
print(fpar_df.dtypes)

# ============================================================
# CONVERT DATE COLUMNS
# ============================================================

weather_df["date"] = pd.to_datetime(
    weather_df["date"],
    format="%Y%m%d"
)

soil_moisture_df["date"] = pd.to_datetime(
    soil_moisture_df["date"],
    format="%Y%m%d"
)

ndvi_df["date"] = pd.to_datetime(
    ndvi_df["date"],
    format="%Y%m%d"
)

fpar_df["date"] = pd.to_datetime(
    fpar_df["date"],
    format="%Y%m%d"
)

print("\nDate columns converted successfully!")


print("\nUpdated Data Types")
print("-" * 50)

print(weather_df["date"].dtype)
print(soil_moisture_df["date"].dtype)
print(ndvi_df["date"].dtype)
print(fpar_df["date"].dtype)

# ============================================================
# CREATE TIME FEATURES
# ============================================================

# Weather
weather_df["year"] = weather_df["date"].dt.year
weather_df["month"] = weather_df["date"].dt.month
weather_df["day_of_year"] = weather_df["date"].dt.dayofyear

# Soil Moisture
soil_moisture_df["year"] = soil_moisture_df["date"].dt.year
soil_moisture_df["month"] = soil_moisture_df["date"].dt.month
soil_moisture_df["day_of_year"] = soil_moisture_df["date"].dt.dayofyear

# NDVI
ndvi_df["year"] = ndvi_df["date"].dt.year
ndvi_df["month"] = ndvi_df["date"].dt.month
ndvi_df["day_of_year"] = ndvi_df["date"].dt.dayofyear

# FPAR
fpar_df["year"] = fpar_df["date"].dt.year
fpar_df["month"] = fpar_df["date"].dt.month
fpar_df["day_of_year"] = fpar_df["date"].dt.dayofyear

print("Time features created successfully!")

print(weather_df[["date", "year", "month", "day_of_year"]].head())


# ============================================================
# CHECK COMMON TIME PERIOD
# ============================================================

print("\nCOMMON TIME PERIOD")
print("=" * 60)

print(f"Yield Dataset           : {yield_df['harvest_year'].min()} - {yield_df['harvest_year'].max()}")
print(f"Weather Dataset         : {weather_df['year'].min()} - {weather_df['year'].max()}")
print(f"Soil Moisture Dataset   : {soil_moisture_df['year'].min()} - {soil_moisture_df['year'].max()}")
print(f"NDVI Dataset            : {ndvi_df['year'].min()} - {ndvi_df['year'].max()}")
print(f"FPAR Dataset            : {fpar_df['year'].min()} - {fpar_df['year'].max()}")

# ============================================================
# FILTER YIELD DATASET
# ============================================================

yield_df = yield_df[yield_df["harvest_year"] >= 2003].copy()

print("\nFiltered Yield Dataset")
print("-" * 40)

print(yield_df["harvest_year"].min())
print(yield_df["harvest_year"].max())

print("Rows:", len(yield_df))

# ============================================================
# SELECT ONE SAMPLE RECORD
# ============================================================

sample = yield_df.iloc[0]

print(sample)

sample_adm = sample["adm_id"]
sample_year = sample["harvest_year"]

print(sample_adm)
print(sample_year)

calendar = calendar_df[
    calendar_df["adm_id"] == sample_adm
]

print(calendar)

sos = int(calendar.iloc[0]["sos"])
eos = int(calendar.iloc[0]["eos"])

print("SOS:", sos)
print("EOS:", eos)

# ============================================================
# WEATHER FOR ONE DISTRICT
# ============================================================

weather_sample = weather_df[
    weather_df["adm_id"] == sample_adm
]

print(weather_sample.shape)


previous_year = weather_sample[
    (weather_sample["year"] == sample_year - 1)
    &
    (weather_sample["day_of_year"] >= sos)
]

current_year = weather_sample[
    (weather_sample["year"] == sample_year)
    &
    (weather_sample["day_of_year"] <= eos)
]

season_weather = pd.concat(
    [previous_year, current_year],
    ignore_index=True
)

print("\nSeasonal Weather Shape")
print(season_weather.shape)

print()

print(season_weather.head())

print()

print(season_weather.tail())


weather_features = (
    season_weather
    .groupby(["adm_id"])
    .agg(
        avg_tmin=("tmin","mean"),
        avg_tmax=("tmax","mean"),
        avg_tavg=("tavg","mean"),
        total_prec=("prec","sum"),
        avg_rad=("rad","mean"),
        avg_et0=("et0","mean"),
        avg_vpd=("vpd","mean"),
        avg_cwb=("cwb","mean")
    )
    .reset_index()
)

print(weather_features)

# ============================================================
# PREPARE WEATHER DATA FOR FEATURE ENGINEERING
# ============================================================

print("\nPreparing Weather Dataset...")
print("-" * 60)

# Keep only years that can contribute to the yield dataset
weather_df = weather_df[
    weather_df["year"].between(2002, 2017)
].copy()

print("Filtered Weather Shape:", weather_df.shape)

# Sort for faster processing
weather_df = weather_df.sort_values(
    ["adm_id", "year", "day_of_year"]
).reset_index(drop=True)

print("\nWeather Dataset Ready!")

print(weather_df.head())

# ============================================================
# MERGE CROP CALENDAR WITH YIELD DATA
# ============================================================

yield_calendar = yield_df.merge(

    calendar_df[
        ["adm_id", "sos", "eos"]
    ],

    on="adm_id",

    how="left"

)

print("\nYield + Calendar Shape")
print("-"*60)

print(yield_calendar.shape)

print()

print(yield_calendar.head())

# ============================================================
# MERGE SOIL DATA
# ============================================================

yield_calendar = yield_calendar.merge(

    soil_df,

    on=["adm_id", "crop_name"],

    how="left"

)

print("\nYield + Calendar + Soil")
print("-"*60)

print(yield_calendar.shape)

print()

print(yield_calendar.head())

# ============================================================
# MERGE LOCATION DATA
# ============================================================

yield_calendar = yield_calendar.merge(

    location_df,

    on=["adm_id", "crop_name"],

    how="left"

)

print("\nLocation merged")
print("-"*60)

print(yield_calendar.shape)

# ============================================================
# MERGE CROP MASK
# ============================================================

yield_calendar = yield_calendar.merge(

    cropmask_df,

    on=["adm_id", "crop_name"],

    how="left"

)

print("\nCrop Mask merged")
print("-"*60)

print(yield_calendar.shape)

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing Values")
print("-"*60)

print(yield_calendar.isnull().sum())

# ============================================================
# CURRENT DATASET OVERVIEW
# ============================================================

print("\nCurrent Dataset")
print("-"*60)

print(yield_calendar.shape)

print()

print(yield_calendar.columns)

print()

print(yield_calendar.head())

# ============================================================
# GENERIC SEASONAL AGGREGATION FUNCTION
# ============================================================

def aggregate_season(
    data,
    adm_id,
    harvest_year,
    sos,
    eos,
    variables,
    agg_type="mean"
):

    district = data[data["adm_id"] == adm_id]

    season = district[
        (
            (district["year"] == harvest_year - 1)
            &
            (district["day_of_year"] >= sos)
        )
        |
        (
            (district["year"] == harvest_year)
            &
            (district["day_of_year"] <= eos)
        )
    ]

    if season.empty:
        return None

    summary = {}

    for var in variables:

        if agg_type == "mean":
            summary[var] = season[var].mean()

        elif agg_type == "sum":
            summary[var] = season[var].sum()

    return summary

# ============================================================
# TEST THE FUNCTION
# ============================================================

weather_test = aggregate_season(

    data=weather_df,

    adm_id=sample_adm,

    harvest_year=sample_year,

    sos=sos,

    eos=eos,

    variables=[
        "tmin",
        "tmax",
        "tavg",
        "rad",
        "et0",
        "vpd",
        "cwb"
    ],

    agg_type="mean"

)

print(weather_test)

# ============================================================
# INVESTIGATE MISSING STATIC INFORMATION
# ============================================================

missing_static = yield_calendar[
    yield_calendar["sos"].isna()
]

print("\nRows with Missing Static Information")
print("-" * 60)

print("Number of rows:", len(missing_static))

print()

print("Unique districts:")

print(missing_static["adm_id"].unique())

print()

print("Number of unique districts:")

print(missing_static["adm_id"].nunique())


# ============================================================
# CHECK WHETHER MISSING DISTRICTS EXIST IN OTHER DATASETS
# ============================================================

missing_ids = missing_static["adm_id"].unique()

print("\nChecking Missing Districts")
print("="*60)

datasets = {
    "Calendar": calendar_df,
    "Soil": soil_df,
    "Location": location_df,
    "Crop Mask": cropmask_df,
    "Weather": weather_df,
    "Soil Moisture": soil_moisture_df,
    "NDVI": ndvi_df,
    "FPAR": fpar_df
}

for name, df in datasets.items():

    count = df["adm_id"].isin(missing_ids).sum()

    print(f"{name:15s} : {count}")


# ============================================================
# WHICH DISTRICTS ARE MISSING FROM CROP MASK?
# ============================================================

missing_districts = yield_calendar[
    yield_calendar["crop_area"].isna()
]["adm_id"].unique()

print("\nMissing Districts")
print("-"*50)

print(missing_districts)

print()

print("Checking if they exist inside Crop Mask dataset")

for district in missing_districts:

    exists = district in cropmask_df["adm_id"].values

    print(f"{district} -> {exists}")

# ============================================================
# GENERIC FUNCTION TO EXTRACT SEASONAL FEATURES
# ============================================================

def seasonal_feature_summary(
    data,
    adm_id,
    harvest_year,
    sos,
    eos,
    variables,
    agg_type="mean"
):

    district = data[
        data["adm_id"] == adm_id
    ]

    season = district[
        (
            (district["year"] == harvest_year - 1)
            &
            (district["day_of_year"] >= sos)
        )
        |
        (
            (district["year"] == harvest_year)
            &
            (district["day_of_year"] <= eos)
        )
    ]

    if season.empty:

        return {
            variable: np.nan
            for variable in variables
        }

    results = {}

    for variable in variables:

        if agg_type == "mean":
            results[variable] = season[variable].mean()

        elif agg_type == "sum":
            results[variable] = season[variable].sum()

        elif agg_type == "max":
            results[variable] = season[variable].max()

        elif agg_type == "min":
            results[variable] = season[variable].min()

    return results

print("Seasonal Feature Function Created Successfully!")


# ============================================================
# TEST THE FUNCTION
# ============================================================

weather_test = seasonal_feature_summary(

    data=weather_df,

    adm_id=sample_adm,

    harvest_year=sample_year,

    sos=sos,

    eos=eos,

    variables=[
        "tmin",
        "tmax",
        "tavg",
        "rad",
        "et0",
        "vpd",
        "cwb"
    ],

    agg_type="mean"

)

print("\nWeather Test")
print("-"*50)

print(weather_test)


# ============================================================
# CREATE WEATHER FEATURES FOR ALL YIELD RECORDS (FAST VERSION)
# ============================================================

import time

print("\nCreating Seasonal Weather Features...")
print("-" * 60)

start = time.time()

# ------------------------------------------------------------
# Group weather once by district
# ------------------------------------------------------------

weather_groups = {
    adm: df
    for adm, df in weather_df.groupby("adm_id")
}

weather_features = []

# ------------------------------------------------------------
# Loop through yield records
# ------------------------------------------------------------

for i, row in yield_calendar.iterrows():

    adm_id = row["adm_id"]

    # Skip if weather unavailable
    if adm_id not in weather_groups:

        weather_features.append({
            "avg_tmin": np.nan,
            "avg_tmax": np.nan,
            "avg_tavg": np.nan,
            "avg_rad": np.nan,
            "avg_et0": np.nan,
            "avg_vpd": np.nan,
            "avg_cwb": np.nan
        })

        continue

    district_weather = weather_groups[adm_id]

    season = district_weather[
        (
            (district_weather["year"] == row["harvest_year"] - 1)
            &
            (district_weather["day_of_year"] >= row["sos"])
        )
        |
        (
            (district_weather["year"] == row["harvest_year"])
            &
            (district_weather["day_of_year"] <= row["eos"])
        )
    ]

    if season.empty:

        weather_features.append({

            "avg_tmin": np.nan,
            "avg_tmax": np.nan,
            "avg_tavg": np.nan,
            "avg_rad": np.nan,
            "avg_et0": np.nan,
            "avg_vpd": np.nan,
            "avg_cwb": np.nan

        })

    else:

        weather_features.append({

            "avg_tmin": season["tmin"].mean(),
            "avg_tmax": season["tmax"].mean(),
            "avg_tavg": season["tavg"].mean(),
            "avg_rad": season["rad"].mean(),
            "avg_et0": season["et0"].mean(),
            "avg_vpd": season["vpd"].mean(),
            "avg_cwb": season["cwb"].mean()

        })

    if (i + 1) % 500 == 0:

        elapsed = (time.time() - start) / 60

        print(
            f"{i+1}/{len(yield_calendar)} completed "
            f"({elapsed:.2f} minutes)"
        )

# ------------------------------------------------------------
# Convert to dataframe
# ------------------------------------------------------------

weather_features = pd.DataFrame(weather_features)

print("\nWeather Features Shape")
print("-"*50)

print(weather_features.shape)

print()

print(weather_features.head())

print()

print(weather_features.isna().sum())

# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

yield_calendar = pd.concat(
    [
        yield_calendar.reset_index(drop=True),
        weather_features.reset_index(drop=True)
    ],
    axis=1
)
end = time.time()

print("\nWeather Feature Engineering Completed!")
print("-"*50)

print(f"Time Taken : {(end-start)/60:.2f} minutes")

print()

print(yield_calendar.shape)

print()

print(yield_calendar.head())


# ============================================================
# GENERIC SEASONAL FEATURE FUNCTION
# ============================================================

def seasonal_feature_summary(
    data,
    adm_id,
    harvest_year,
    sos,
    eos,
    variables,
    agg="mean"
):
    """
    Computes seasonal summary statistics for any time-series dataset.
    """

    district = weather_groups if False else None  # placeholder to avoid confusion

    district = data[data["adm_id"] == adm_id]

    season = district[
        (
            (district["year"] == harvest_year - 1)
            &
            (district["day_of_year"] >= sos)
        )
        |
        (
            (district["year"] == harvest_year)
            &
            (district["day_of_year"] <= eos)
        )
    ]

    if season.empty:

        return {
            var: np.nan
            for var in variables
        }

    result = {}

    for var in variables:

        if agg == "mean":
            result[var] = season[var].mean()

        elif agg == "sum":
            result[var] = season[var].sum()

        elif agg == "max":
            result[var] = season[var].max()

        elif agg == "min":
            result[var] = season[var].min()

    return result

print("Generic Seasonal Function Created!")


# ============================================================
# PREPARE SOIL MOISTURE
# ============================================================

print("\nPreparing Soil Moisture...")
print("-"*60)

soil_moisture_df = soil_moisture_df[
    soil_moisture_df["year"].between(2002, 2017)
].copy()

soil_groups = {
    adm: df
    for adm, df in soil_moisture_df.groupby("adm_id")
}

print("Soil Moisture Ready!")

print(soil_moisture_df.head())

# ============================================================
# CREATE SOIL MOISTURE FEATURES
# ============================================================

import time

print("\nCreating Soil Moisture Features...")
print("-"*60)

start = time.time()

soil_features = []

for i, row in yield_calendar.iterrows():

    adm = row["adm_id"]

    if adm not in soil_groups:

        soil_features.append({
            "avg_ssm": np.nan,
            "avg_rsm": np.nan
        })

        continue

    district = soil_groups[adm]

    season = district[
        (
            (district["year"] == row["harvest_year"]-1)
            &
            (district["day_of_year"] >= row["sos"])
        )
        |
        (
            (district["year"] == row["harvest_year"])
            &
            (district["day_of_year"] <= row["eos"])
        )
    ]

    if season.empty:

        soil_features.append({
            "avg_ssm": np.nan,
            "avg_rsm": np.nan
        })

    else:

        soil_features.append({

            "avg_ssm": season["ssm"].mean(),

            "avg_rsm": season["rsm"].mean()

        })

    if (i+1)%500==0:

        print(f"{i+1}/{len(yield_calendar)} completed")

soil_features = pd.DataFrame(soil_features)

end = time.time()

print()

print("Time:", round((end-start)/60,2),"minutes")

print()

print(soil_features.head())

print()

print(soil_features.isna().sum())

# ============================================================
# MERGE SOIL MOISTURE FEATURES
# ============================================================

yield_calendar = pd.concat(

    [
        yield_calendar.reset_index(drop=True),
        soil_features.reset_index(drop=True)
    ],

    axis=1
)

print("\nDataset Shape")
print("-"*50)

print(yield_calendar.shape)

print()

print(yield_calendar.head())


# ============================================================
# PREPARE NDVI DATASET
# ============================================================

print("\nPreparing NDVI...")
print("-"*60)

ndvi_df = ndvi_df[
    ndvi_df["year"].between(2002, 2017)
].copy()

ndvi_groups = {
    adm: df
    for adm, df in ndvi_df.groupby("adm_id")
}

print("NDVI Ready!")

print(ndvi_df.head())

# ============================================================
# CREATE NDVI FEATURES
# ============================================================

import time

print("\nCreating NDVI Features...")
print("-"*60)

start = time.time()

ndvi_features = []

for i, row in yield_calendar.iterrows():

    adm = row["adm_id"]

    if adm not in ndvi_groups:

        ndvi_features.append({
            "avg_ndvi": np.nan
        })

        continue

    district = ndvi_groups[adm]

    season = district[
        (
            (district["year"] == row["harvest_year"]-1)
            &
            (district["day_of_year"] >= row["sos"])
        )
        |
        (
            (district["year"] == row["harvest_year"])
            &
            (district["day_of_year"] <= row["eos"])
        )
    ]

    if season.empty:

        ndvi_features.append({
            "avg_ndvi": np.nan
        })

    else:

        ndvi_features.append({

            "avg_ndvi": season["ndvi"].mean()

        })

    if (i+1)%500==0:

        print(f"{i+1}/{len(yield_calendar)} completed")

ndvi_features = pd.DataFrame(ndvi_features)

end = time.time()

print()

print("Time:", round((end-start)/60,2),"minutes")

print()

print(ndvi_features.head())

print()

print(ndvi_features.isna().sum())

# ============================================================
# MERGE NDVI FEATURES
# ============================================================

yield_calendar = pd.concat(

    [
        yield_calendar.reset_index(drop=True),
        ndvi_features.reset_index(drop=True)
    ],

    axis=1

)

print("\nDataset Shape")
print("-"*50)

print(yield_calendar.shape)

print()

print(yield_calendar.head())

# ============================================================
# PREPARE FPAR DATASET
# ============================================================

print("\nPreparing FPAR...")
print("-"*60)

fpar_df = fpar_df[
    fpar_df["year"].between(2002, 2017)
].copy()

fpar_groups = {
    adm: df
    for adm, df in fpar_df.groupby("adm_id")
}

print("FPAR Ready!")

print(fpar_df.head())

# ============================================================
# CREATE FPAR FEATURES
# ============================================================

import time

print("\nCreating FPAR Features...")
print("-"*60)

start = time.time()

fpar_features = []

for i, row in yield_calendar.iterrows():

    adm = row["adm_id"]

    if adm not in fpar_groups:

        fpar_features.append({
            "avg_fpar": np.nan
        })

        continue

    district = fpar_groups[adm]

    season = district[
        (
            (district["year"] == row["harvest_year"] - 1)
            &
            (district["day_of_year"] >= row["sos"])
        )
        |
        (
            (district["year"] == row["harvest_year"])
            &
            (district["day_of_year"] <= row["eos"])
        )
    ]

    if season.empty:

        fpar_features.append({
            "avg_fpar": np.nan
        })

    else:

        fpar_features.append({

            "avg_fpar": season["fpar"].mean()

        })

    if (i + 1) % 500 == 0:

        print(f"{i+1}/{len(yield_calendar)} completed")

fpar_features = pd.DataFrame(fpar_features)

end = time.time()

print()

print("Time:", round((end-start)/60,2), "minutes")

print()

print(fpar_features.head())

print()

print(fpar_features.isna().sum())

# ============================================================
# MERGE FPAR FEATURES
# ============================================================

yield_calendar = pd.concat(

    [
        yield_calendar.reset_index(drop=True),
        fpar_features.reset_index(drop=True)
    ],

    axis=1

)

print("\nFinal Dataset Shape")
print("-"*50)

print(yield_calendar.shape)

print()

print(yield_calendar.head())

# ============================================================
# FINAL MISSING VALUE ANALYSIS
# ============================================================

print("\nMissing Values")
print("-"*60)

missing = yield_calendar.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

# ============================================================
# DROP USELESS COLUMNS
# ============================================================

columns_to_drop = [

    "season_name",

    "planting_year",

    "planting_date",

    "harvest_date",

    "planted_area"

]

yield_calendar.drop(
    columns=columns_to_drop,
    inplace=True
)

print(yield_calendar.shape)

# ============================================================
# REMAINING MISSING VALUES
# ============================================================

print("\nRemaining Missing Values")
print("-"*60)

print(
    yield_calendar.isnull().sum()[
        yield_calendar.isnull().sum() > 0
    ]
)

# ============================================================
# REMOVE INCOMPLETE RECORDS
# ============================================================

yield_calendar = yield_calendar.dropna().reset_index(drop=True)

print("\nFinal Dataset Shape")
print("-"*60)

print(yield_calendar.shape)

# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicates = yield_calendar.duplicated().sum()

print("Duplicate Rows:", duplicates)

# ============================================================
# SAVE FINAL DATASET
# ============================================================

import os

processed_path = os.path.join(
    os.getcwd(),
    "data",
    "processed"
)

os.makedirs(processed_path, exist_ok=True)

yield_calendar.to_csv(

    os.path.join(
        processed_path,
        "final_crop_dataset.csv"
    ),

    index=False

)

print("Final dataset saved successfully!")