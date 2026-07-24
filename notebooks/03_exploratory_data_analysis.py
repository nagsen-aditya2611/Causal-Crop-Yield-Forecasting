import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

pd.set_option("display.max_columns", None)

print("Libraries Imported Successfully!")


# CREATE FOLDERS FOR OUTPUT

FIGURE_DIR = Path("figures/figures_eda")
REPORT_DIR = Path("reports")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Output folders ready!")


#FUNCTION TO SAVE FIGURES


def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved -> {filename}")


#LOAD DATASET


df = pd.read_csv("data/processed/final_crop_dataset.csv")

print("\nDataset Loaded Successfully!")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows")
print("-" * 60)

print(df.head())

#DATASET OVERVIEW


print("="*70)
print("DATASET OVERVIEW")
print("="*70)

print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nData Types\n")

df.info()


# MISSING VALUE ANALYSIS


missing = (

    df
    .isna()
    .sum()
    .sort_values(ascending=False)

)

missing = missing[missing > 0]

print("\nMissing Values")
print("="*70)

print(missing)

if len(missing) == 0:

    print("\nNo Missing Values Found!")

else:

    plt.figure(figsize=(10,5))

    missing.plot(
        kind="bar",
        color="tomato"
    )

    plt.title("Missing Values")

    plt.ylabel("Count")

    save_plot("01_missing_values.png")


# DUPLICATE ANALYSIS


duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)


# DESCRIPTIVE STATISTICS


stats = df.describe().T

stats["Skewness"] = df.select_dtypes(
    include=np.number
).skew()

stats["Kurtosis"] = df.select_dtypes(
    include=np.number
).kurt()

print("\nDescriptive Statistics")
print("=" * 70)

print(stats)

stats.to_csv(
    REPORT_DIR / "summary_statistics.csv",
    index=True
)

print("\nSummary statistics exported successfully!")


# NUMERICAL FEATURES


numeric_cols = df.select_dtypes(
    include=np.number
).columns

print("\nNumerical Features")

print(numeric_cols)

print()

print("Total Numerical Features :", len(numeric_cols))


#  TARGET VARIABLE


plt.figure(figsize=(8,5))

sns.histplot(

    df["yield"],

    bins=30,

    kde=True

)

plt.title("Yield Distribution")

plt.xlabel("Yield")

plt.ylabel("Frequency")

save_plot("02_yield_distribution.png")

print(df["yield"].describe())


#  TARGET BOXPLOT

plt.figure(figsize=(6,4))

sns.boxplot(

    x=df["yield"]

)

plt.title("Yield Boxplot")

save_plot("03_yield_boxplot.png")


# HISTOGRAMS OF ALL NUMERICAL FEATURES


print("\nGenerating Histograms...")

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    sns.histplot(

        df[col],

        bins=30,

        kde=True

    )

    plt.title(col)

    save_plot(f"hist_{col}.png")

print("\nAll Histograms Saved Successfully!")

# Boxplots of all numerical features

print("\nGenerating Boxplots...")
print("-" * 60)

for col in numeric_cols:

    plt.figure(figsize=(7,4))

    sns.boxplot(
        x=df[col],
        color="skyblue"
    )

    plt.title(f"Boxplot of {col}")

    save_plot(f"box_{col}.png")

print("\nAll Boxplots Saved Successfully!")

#correlation Matrix

print("\nCorrelation Matrix")
print("-" * 60)

corr = df.corr(numeric_only=True)

plt.figure(figsize=(18,14))

sns.heatmap(

    corr,

    cmap="coolwarm",

    center=0,

    annot=False,

    square=True

)

plt.title("Correlation Heatmap")

save_plot("04_correlation_heatmap.png")

corr.to_csv(
    REPORT_DIR / "correlation_matrix.csv"
)

print("Correlation Matrix Saved!")

# Correlation with Yield

print("\nCorrelation with Yield")
print("-" * 60)

yield_corr = (

    corr["yield"]

    .drop("yield")

    .sort_values(ascending=False)

)

print(yield_corr)

yield_corr.to_csv(
    REPORT_DIR / "yield_correlations.csv"
)

# BARPLOT OF FEATURE CORRELATIONS

plt.figure(figsize=(10,8))

yield_corr.sort_values().plot(

    kind="barh",

    color="steelblue"

)

plt.title("Correlation of Features with Yield")

plt.xlabel("Correlation")

save_plot("05_yield_correlations.png")

# Pairplot

pair_cols = [

    "yield",

    "avg_tavg",

    "avg_prec" if "avg_prec" in df.columns else "avg_tmin",

    "avg_ndvi",

    "avg_fpar",

    "avg_ssm"

]

sns.pairplot(

    df[pair_cols],

    corner=True

)

plt.savefig(

    FIGURE_DIR / "06_pairplot.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("Pairplot Saved!")

# YIELD VS WEATHER FEATURES

print("\nYield vs Weather Features")
print("-"*60)

weather_cols = [

    "avg_tmin",
    "avg_tmax",
    "avg_tavg",
    "avg_rad",
    "avg_et0",
    "avg_vpd",
    "avg_cwb"

]

for col in weather_cols:

    plt.figure(figsize=(7,5))

    sns.scatterplot(

        data=df,

        x=col,

        y="yield",

        alpha=0.6

    )

    sns.regplot(

        data=df,

        x=col,

        y="yield",

        scatter=False,

        color="red"

    )

    plt.title(f"Yield vs {col}")

    save_plot(f"weather_vs_{col}.png")

print("Weather Relationship Plots Saved!")

# YIELD VS SOIL FEATURES

print("\nYield vs Soil Features")
print("-"*60)

soil_cols = [

    "awc",
    "bulk_density",
    "drainage_class",
    "avg_ssm",
    "avg_rsm"

]

for col in soil_cols:

    plt.figure(figsize=(7,5))

    sns.scatterplot(

        data=df,

        x=col,

        y="yield",

        alpha=0.6

    )

    sns.regplot(

        data=df,

        x=col,

        y="yield",

        scatter=False,

        color="red"

    )

    plt.title(f"Yield vs {col}")

    save_plot(f"soil_vs_{col}.png")

print("Soil Relationship Plots Saved!")


# YIELD VS VEGETATION

print("\nYield vs Vegetation")
print("-"*60)

veg_cols = [

    "avg_ndvi",
    "avg_fpar"

]

for col in veg_cols:

    plt.figure(figsize=(7,5))

    sns.scatterplot(

        data=df,

        x=col,

        y="yield",

        alpha=0.6

    )

    sns.regplot(

        data=df,

        x=col,

        y="yield",

        scatter=False,

        color="red"

    )

    plt.title(f"Yield vs {col}")

    save_plot(f"vegetation_vs_{col}.png")

print("Vegetation Plots Saved!")

# YEAR-WISE YIELD TREND

yearly = (

    df

    .groupby("harvest_year")["yield"]

    .mean()

    .reset_index()

)

plt.figure(figsize=(10,5))

sns.lineplot(

    data=yearly,

    x="harvest_year",

    y="yield",

    marker="o"

)

plt.title("Average Yield Over Time")

plt.xlabel("Harvest Year")

plt.ylabel("Average Yield")

save_plot("07_yearly_yield_trend.png")

# YIELD BY YEAR (BOXPLOT)

plt.figure(figsize=(14,6))

sns.boxplot(

    data=df,

    x="harvest_year",

    y="yield"

)

plt.xticks(rotation=45)

plt.title("Yield Distribution by Harvest Year")

save_plot("08_yield_by_year_boxplot.png")


# GEOGRAPHICAL DISTRIBUTION

plt.figure(figsize=(8,8))

sns.scatterplot(

    data=df,

    x="longitude",

    y="latitude",

    hue="yield",

    palette="viridis",

    alpha=0.8

)

plt.title("Spatial Distribution of Yield")

save_plot("09_spatial_yield_distribution.png")

# FEATURE CORRELATION PAIRS

pairs = [

    ("avg_ndvi","yield"),
    ("avg_fpar","yield"),
    ("avg_ssm","yield"),
    ("avg_tavg","yield"),
    ("avg_cwb","yield")

]

for x,y in pairs:

    plt.figure(figsize=(7,5))

    sns.regplot(

        data=df,

        x=x,

        y=y,

        scatter_kws={"alpha":0.4}

    )

    plt.title(f"{y} vs {x}")

    save_plot(f"{y}_vs_{x}.png")


# ============================================================
# CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\nCategorical Variables")
print("-"*60)

cat_cols = df.select_dtypes(include="object").columns

print(cat_cols)

for col in cat_cols:

    print(f"\n{col}")

    print(df[col].value_counts().head())

    plt.figure(figsize=(8,5))

    df[col].value_counts().head(20).plot(
        kind="bar"
    )

    plt.title(col)

    plt.ylabel("Count")

    save_plot(f"cat_{col}.png")

    # ============================================================
# TOP DISTRICTS BY AVERAGE YIELD
# ============================================================

district_yield = (

    df.groupby("adm_id")["yield"]
      .mean()
      .sort_values(ascending=False)

)

plt.figure(figsize=(10,8))

district_yield.head(20).plot(kind="bar")

plt.title("Top 20 Districts by Average Yield")

save_plot("10_top_districts.png")

# ============================================================
# HARVEST AREA VS YIELD
# ============================================================

plt.figure(figsize=(7,5))

sns.scatterplot(

    data=df,

    x="harvest_area",

    y="yield",

    alpha=0.5

)

sns.regplot(

    data=df,

    x="harvest_area",

    y="yield",

    scatter=False,

    color="red"

)

plt.title("Harvest Area vs Yield")

save_plot("11_harvest_area_vs_yield.png")

# ============================================================
# YEARWISE VIOLIN PLOT
# ============================================================

plt.figure(figsize=(14,6))

sns.violinplot(

    data=df,

    x="harvest_year",

    y="yield"

)

plt.xticks(rotation=45)

plt.title("Yield Distribution Across Years")

save_plot("12_yearly_violin.png")

# ============================================================
# TOP POSITIVE AND NEGATIVE CORRELATIONS
# ============================================================

print("\nTop Positive Correlations")

print(yield_corr.head(10))

print("\nTop Negative Correlations")

print(yield_corr.tail(10))

# ============================================================
# EXPORT EDA SUMMARY
# ============================================================

with open(

    REPORT_DIR / "eda_summary.txt",

    "w"

) as f:

    f.write("DATASET SUMMARY\n")

    f.write("="*60 + "\n\n")

    f.write(f"Rows : {df.shape[0]}\n")

    f.write(f"Columns : {df.shape[1]}\n")

    f.write(f"Duplicate Rows : {duplicates}\n\n")

    f.write("Missing Values\n")

    f.write(str(missing))

    f.write("\n\n")

    f.write("Correlation with Yield\n")

    f.write(str(yield_corr))

