import streamlit as st
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("📈 Exploratory Data Analysis")

st.markdown(
"""
Interactive exploratory analysis of the processed crop yield dataset.
"""
)

# ==========================================================
# LOAD DATA
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_crop_dataset.csv"

df = pd.read_csv(DATA_PATH)

# ==========================================================
# NUMERIC FEATURES
# ==========================================================

numeric_columns = df.select_dtypes(include="number").columns.tolist()

# ==========================================================
# DATASET SNAPSHOT
# ==========================================================

st.subheader("📊 Dataset Snapshot")

st.dataframe(df.head(), use_container_width=True)

# ==========================================================
# FEATURE SELECTOR
# ==========================================================

st.subheader("🎛 Feature Explorer")

selected_feature = st.selectbox(

    "Select Numeric Feature",

    numeric_columns

)

# ==========================================================
# DISTRIBUTION
# ==========================================================

st.subheader("📈 Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(

    df[selected_feature],

    kde=True,

    bins=30,

    color="royalblue",

    ax=ax

)

ax.set_xlabel(selected_feature)

st.pyplot(fig)

# ==========================================================
# BOXPLOT
# ==========================================================

st.subheader("📦 Boxplot")

fig, ax = plt.subplots(figsize=(8,2))

sns.boxplot(

    x=df[selected_feature],

    color="orange",

    ax=ax

)

st.pyplot(fig)

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

st.subheader("🔥 Correlation Heatmap")

corr = df[numeric_columns].corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(

    corr,

    cmap="coolwarm",

    center=0,

    square=True,

    linewidths=0.5,

    cbar_kws={"shrink":0.8},

    ax=ax

)

st.pyplot(fig)

# ==========================================================
# FEATURE VS YIELD
# ==========================================================

st.subheader("🌾 Relationship with Yield")

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(

    x=df[selected_feature],

    y=df["yield"],

    alpha=0.6,

    ax=ax

)

ax.set_xlabel(selected_feature)

ax.set_ylabel("Yield")

st.pyplot(fig)

# ==========================================================
# FEATURE IMPORTANCE (Correlation)
# ==========================================================

st.subheader("📊 Features Most Correlated with Yield")

corr_with_target = (

    corr["yield"]

    .drop("yield")

    .sort_values(

        ascending=False,

        key=abs

    )

)

st.dataframe(

    corr_with_target.to_frame("Correlation"),

    use_container_width=True

)

# ==========================================================
# PAIRPLOT
# ==========================================================

st.subheader("📈 Pairplot (Sampled Data)")

sample_df = df.sample(

    min(300, len(df)),

    random_state=42

)

pairplot_features = st.multiselect(

    "Choose up to 4 variables",

    numeric_columns,

    default=numeric_columns[:4]

)

if 2 <= len(pairplot_features) <= 4:

    pair_fig = sns.pairplot(

        sample_df[pairplot_features]

    )

    st.pyplot(pair_fig.figure)


