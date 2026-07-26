import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "figures" / "figures_model_comparison_visualisation"

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Folders Ready!")

# ============================================================
# LOAD PREDICTION FILES
# ============================================================

xgb = pd.read_csv(
    REPORT_DIR / "xgboost_predictions.csv"
)

lstm = pd.read_csv(
    REPORT_DIR / "lstm_predictions.csv"
)

print("LSTM prediction file loaded successfully!")
print(lstm.head())

print("XGBoost prediction file loaded successfully!")

print(xgb.head())

plt.figure(figsize=(8,8))

plt.scatter(
    xgb["Actual"],
    xgb["XGBoost_All"],
    alpha=0.55,
    s=40,
    edgecolors="black",
    linewidth=0.25
)

plt.plot(
    [xgb["Actual"].min(), xgb["Actual"].max()],
    [xgb["Actual"].min(), xgb["Actual"].max()],
    color="red",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("XGBoost (All Features): Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "xgb_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

r2 = r2_score(
    xgb["Actual"],
    xgb["XGBoost_All"]
)

print("="*60)
print("XGBoost R²")
print("="*60)

print(r2)


# ============================================================
# LSTM ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8,8))

plt.scatter(
    lstm["Actual"],
    lstm["LSTM_All"],
    alpha=0.55,
    s=40,
    edgecolors="black",
    linewidth=0.25
)

plt.plot(
    [lstm["Actual"].min(), lstm["Actual"].max()],
    [lstm["Actual"].min(), lstm["Actual"].max()],
    color="red",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("LSTM (All Features): Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "lstm_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

lstm_r2 = r2_score(
    lstm["Actual"],
    lstm["LSTM_All"]
)

print("="*60)
print("LSTM R²")
print("="*60)

print(lstm_r2)

# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

comparison = pd.read_csv(
    REPORT_DIR / "model_comparison.csv"
)

print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(comparison.to_string(index=False))

# ============================================================
# ADD DEEP LEARNING MODELS TO COMPARISON TABLE
# ============================================================

deep_learning_rows = pd.DataFrame({

    "Model": [

        "MLP",
        "LSTM"

    ],

    "All Features R2": [

        0.871,
        0.874

    ],

    "Causal Features R2": [

        0.798,
        0.792

    ],

    "Performance Retained (%)": [

        91.585,
        (0.792 / 0.874) * 100

    ]

})

comparison = pd.concat(

    [comparison, deep_learning_rows],

    ignore_index=True

)

print("="*60)
print("UPDATED MODEL COMPARISON")
print("="*60)

print(comparison.to_string(index=False))

# ============================================================
# R² COMPARISON BAR PLOT
# ============================================================

comparison = comparison.sort_values(
    by="All Features R2",
    ascending=False
)

plt.figure(figsize=(10,6))

bars = plt.bar(
    comparison["Model"],
    comparison["All Features R2"],
    edgecolor="black",
    linewidth=0.8
)

plt.ylabel("R² Score", fontsize=12)

plt.xlabel("Models", fontsize=12)

plt.title(
    "Comparison of Machine Learning and Deep Learning Models",
    fontsize=14,
    fontweight="bold"
)

plt.ylim(0.65, 0.95)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.005,
        f"{height:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "model_r2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()