import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.inspection import permutation_importance
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns

import shap
import joblib

print("Libraries Imported Successfully!")
print("SHAP Version :", shap.__version__)

# ============================================================
# OUTPUT FOLDERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "figures" / "figures_shap"

REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR = PROJECT_ROOT / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Output folders created successfully!")

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_crop_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("="*70)
print("DATASET LOADED")
print("="*70)

print(df.shape)

# ============================================================
# FEATURE PREPARATION
# ============================================================

target = "yield"

causal_features = [

    "avg_ssm",
    "avg_tavg",
    "avg_rad",
    "avg_cwb",

    "avg_ndvi",
    "avg_fpar",

    "awc",
    "bulk_density",
    "drainage_class"

]

all_features = df.select_dtypes(include=[np.number]).columns.tolist()

all_features.remove(target)

if "production" in all_features:
    all_features.remove("production")

X_all = df[all_features]

X_causal = df[causal_features]

y = df[target]

from sklearn.model_selection import train_test_split

X_train_all, X_test_all, y_train, y_test = train_test_split(

    X_all,
    y,

    test_size=0.20,

    random_state=42

)

X_train_causal = X_train_all[causal_features]

X_test_causal = X_test_all[causal_features]

print("Train/Test Split Ready")

# ============================================================
# LOAD SAVED MODELS
# ============================================================

rf_all = joblib.load(MODEL_DIR / "rf_all.pkl")
rf_causal = joblib.load(MODEL_DIR / "rf_causal.pkl")

xgb_all = joblib.load(MODEL_DIR / "xgb_all.pkl")
xgb_causal = joblib.load(MODEL_DIR / "xgb_causal.pkl")

lgb_all = joblib.load(MODEL_DIR / "lgb_all.pkl")
lgb_causal = joblib.load(MODEL_DIR / "lgb_causal.pkl")

print("="*70)
print("ALL MODELS LOADED SUCCESSFULLY")
print("="*70)

# ============================================================
# RANDOM FOREST (ALL FEATURES)
# SHAP EXPLAINER
# ============================================================

explainer_rf = shap.TreeExplainer(rf_all)

shap_values_rf = explainer_rf.shap_values(X_test_all)

print("="*70)
print("SHAP VALUES GENERATED")
print("="*70)

print(np.array(shap_values_rf).shape)

# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

plt.figure(figsize=(10,8))

shap.summary_plot(

    shap_values_rf,

    X_test_all,

    show=False

)

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "rf_summary_plot.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ============================================================
# SHAP BAR PLOT
# ============================================================

plt.figure(figsize=(10,8))

shap.summary_plot(

    shap_values_rf,

    X_test_all,

    plot_type="bar",

    show=False

)

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "rf_bar_plot.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

importance = np.abs(shap_values_rf).mean(axis=0)

rf_importance = pd.DataFrame({

    "Feature": X_test_all.columns,

    "Mean_SHAP": importance

})

rf_importance = rf_importance.sort_values(

    by="Mean_SHAP",

    ascending=False

)

print(rf_importance)

rf_importance.to_csv(

    REPORT_DIR / "rf_shap_importance.csv",

    index=False

)

# ============================================================
# DEBUG XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*70)
print("XGBOOST DEBUG")
print("="*70)

print("Number of dataset features :")
print(len(X_train_all.columns))

print("\nNumber of feature importances :")
print(len(xgb_all.feature_importances_))

print("\nDataset Feature Names:")
print(X_train_all.columns.tolist())

print("\nXGBoost Feature Importances:")
print(xgb_all.feature_importances_)