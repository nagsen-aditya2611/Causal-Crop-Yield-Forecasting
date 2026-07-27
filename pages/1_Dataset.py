import streamlit as st
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
# ==========================================================
# PAGE CONFIG
# ==========================================================

st.title("📊 Dataset Overview")

st.markdown(
"""
This page provides an overview of the processed crop yield dataset
used throughout the project.
"""
)

# ==========================================================
# LOAD DATA
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_crop_dataset.csv"

df = pd.read_csv(DATA_PATH)

# ==========================================================
# DATASET SUMMARY
# ==========================================================

rows, cols = df.shape

numeric_features = len(df.select_dtypes(include="number").columns)

categorical_features = len(df.select_dtypes(exclude="number").columns)

missing = int(df.isna().sum().sum())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", rows)

with col2:
    st.metric("Columns", cols)

with col3:
    st.metric("Numeric Features", numeric_features)

with col4:
    st.metric("Missing Values", missing)

st.divider()

# ==========================================================
# TARGET VARIABLE
# ==========================================================

st.subheader("🎯 Target Variable")

st.info("Target Variable : **yield**")

# ==========================================================
# FEATURE LIST
# ==========================================================

st.subheader("📑 Dataset Columns")

st.dataframe(
    pd.DataFrame(df.columns, columns=["Feature"]),
    use_container_width=True
)

# ==========================================================
# DATA PREVIEW
# ==========================================================

st.subheader("🔍 Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

st.subheader("📈 Summary Statistics")

st.dataframe(df.describe().T, use_container_width=True)

# ==========================================================
# TARGET VARIABLE DISTRIBUTION
# ==========================================================

st.subheader("🌾 Yield Distribution")

fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    df["yield"],
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Yield")

ax.set_ylabel("Frequency")

ax.set_title("Distribution of Crop Yield")

st.pyplot(fig)

# ==========================================================
# MISSING VALUE HEATMAP
# ==========================================================

st.subheader("🧹 Missing Value Heatmap")

fig, ax = plt.subplots(figsize=(12,4))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis",
    ax=ax
)

st.pyplot(fig)

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

st.subheader("📈 Correlation Heatmap")

corr = df.select_dtypes(include="number").corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink":0.8},
    ax=ax
)

st.pyplot(fig)

# ==========================================================
# FEATURE TYPE SUMMARY
# ==========================================================

st.subheader("📋 Feature Type Summary")

feature_summary = pd.DataFrame({

    "Feature Type":[

        "Numeric",

        "Categorical"

    ],

    "Count":[

        numeric_features,

        categorical_features

    ]

})

st.dataframe(

    feature_summary,

    use_container_width=True

)

# ==========================================================
# DATASET EXPLORER
# ==========================================================

st.subheader("🔍 Interactive Dataset Explorer")

num_rows = st.slider(

    "Number of rows",

    5,

    len(df),

    10

)

st.dataframe(

    df.head(num_rows),

    use_container_width=True

)

# ==========================================================
# DOWNLOAD
# ==========================================================

st.download_button(
    label="⬇ Download Dataset",
    data=df.to_csv(index=False),
    file_name="final_crop_dataset.csv",
    mime="text/csv"
)