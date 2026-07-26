import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("Libraries Imported Successfully!")

# ============================================================
# OUTPUT FOLDERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "figures" / "figures_ml"

REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR = PROJECT_ROOT / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
print(df.head())

# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

selected_features = pd.read_csv(
    REPORT_DIR / "selected_features.csv"
)

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

print("="*70)
print("CAUSAL FEATURES")
print("="*70)

print(causal_features)

# ============================================================
# DEFINE FEATURE SETS
# ============================================================

target = "yield"

# ============================================================
# NUMERIC FEATURES ONLY
# ============================================================

all_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target
all_features.remove(target)

# Remove target leakage variables
leakage_features = [
    "production"
]

for col in leakage_features:
    if col in all_features:
        all_features.remove(col)

X_all = df[all_features]


print("\nFeatures used for ALL model:\n")

for col in all_features:
    print(col)

print("\nTotal Features:", len(all_features))

X_causal = df[causal_features]

y = df[target]

print("="*70)
print("FEATURE SUMMARY")
print("="*70)

print("All Features :", X_all.shape[1])

print("Causal Features :", X_causal.shape[1])

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_all_train, X_all_test, y_train, y_test = train_test_split(

    X_all,
    y,

    test_size=0.20,

    random_state=42

)

X_causal_train = X_all_train[causal_features]

X_causal_test = X_all_test[causal_features]

# ============================================================
# ALIAS VARIABLES FOR MODELS
# ============================================================

X_train_all = X_all_train
X_test_all = X_all_test

X_train_causal = X_causal_train
X_test_causal = X_causal_test

print("="*70)
print("TRAIN TEST SPLIT")
print("="*70)

print("Training Samples :", len(X_all_train))

print("Testing Samples :", len(X_all_test))

# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

results = []

def evaluate_model(model, X_train, X_test, y_train, y_test,
                   model_name, feature_set):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({

        "Model": model_name,
        "Feature_Set": feature_set,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2

    })

    print("="*60)
    print(model_name)
    print(feature_set)
    print("="*60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return model, predictions

# ============================================================
# LINEAR REGRESSION
# ============================================================

lr_all = LinearRegression()

lr_all, pred_lr_all = evaluate_model(

    lr_all,

    X_all_train,
    X_all_test,

    y_train,
    y_test,

    "Linear Regression",
    "All Features"

)

lr_causal = LinearRegression()

lr_causal, pred_lr_causal = evaluate_model(

    lr_causal,

    X_causal_train,
    X_causal_test,

    y_train,
    y_test,

    "Linear Regression",
    "Causal Features"

)

# ============================================================
# RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestRegressor

rf_all_model = RandomForestRegressor(

    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1

)

rf_causal_model = RandomForestRegressor(

    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1

)

rf_all, pred_rf_all = evaluate_model(

    rf_all_model,

    X_train_all,
    X_test_all,

    y_train,
    y_test,

    "Random Forest",
    "All Features"

)

rf_causal, pred_rf_causal = evaluate_model(

    rf_causal_model,

    X_train_causal,
    X_test_causal,

    y_train,
    y_test,

    "Random Forest",
    "Causal Features"

)

# ============================================================
# XGBOOST
# ============================================================

xgb_all_model = XGBRegressor(

    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42

)

xgb_causal_model = XGBRegressor(

    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42

)

xgb_all, pred_xgb_all = evaluate_model(

    xgb_all_model,

    X_train_all,
    X_test_all,

    y_train,
    y_test,

    "XGBoost",
    "All Features"

)

xgb_causal, pred_xgb_causal = evaluate_model(

    xgb_causal_model,

    X_train_causal,
    X_test_causal,

    y_train,
    y_test,

    "XGBoost",
    "Causal Features"

)
# ============================================================
# LIGHTGBM
# ============================================================

lgb_all_model = LGBMRegressor(

    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42

)

lgb_causal_model = LGBMRegressor(

    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42

)

lgb_all, pred_lgb_all = evaluate_model(

    lgb_all_model,

    X_train_all,
    X_test_all,

    y_train,
    y_test,

    "LightGBM",
    "All Features"

)

lgb_causal, pred_lgb_causal = evaluate_model(

    lgb_causal_model,

    X_train_causal,
    X_test_causal,

    y_train,
    y_test,

    "LightGBM",
    "Causal Features"

)

# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================

results = {

    "Model": [

        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM"

    ],

    "All Features R2": [

        0.7422,
        0.8914,
        0.9040,
        0.8967

    ],

    "Causal Features R2": [

        0.5321,
        0.8065,
        0.8238,
        0.8216

    ]

}

results_df = pd.DataFrame(results)

results_df["Performance Retained (%)"] = (

    results_df["Causal Features R2"]
    /
    results_df["All Features R2"]

) * 100

results_df = results_df.round(3)

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(results_df)

results_df.to_csv(

    "reports/model_comparison.csv",

    index=False

)

plt.figure(figsize=(10,6))

x = np.arange(len(results_df))

width = 0.35

plt.bar(

    x-width/2,

    results_df["All Features R2"],

    width,

    label="All Features"

)

plt.bar(

    x+width/2,

    results_df["Causal Features R2"],

    width,

    label="Causal Features"

)

plt.xticks(

    x,

    results_df["Model"],

    rotation=15

)

plt.ylabel("R² Score")

plt.title("Model Performance Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/figures_ml/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8,5))

plt.bar(

    results_df["Model"],

    results_df["Performance Retained (%)"]

)

plt.ylim(60,100)

plt.ylabel("Performance Retained (%)")

plt.title("Predictive Performance Retained Using Only Causal Features")

plt.tight_layout()

plt.savefig(
    "figures/figures_ml/performance_retained.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

prediction_df = pd.DataFrame({

    "Actual": y_test.values,

    "Linear_All": pred_lr_all,
    "Linear_Causal": pred_lr_causal,

    "RF_All": pred_rf_all,
    "RF_Causal": pred_rf_causal,

    "XGB_All": pred_xgb_all,
    "XGB_Causal": pred_xgb_causal,

    "LGBM_All": pred_lgb_all,
    "LGBM_Causal": pred_lgb_causal

})

prediction_df.to_csv(

    REPORT_DIR / "model_predictions.csv",

    index=False

)

print("="*60)
print("Prediction file saved!")
print("="*60)


# ============================================================
# SAVE TRAINED MODELS
# ============================================================

import joblib

joblib.dump(rf_all, MODEL_DIR / "rf_all.pkl")
joblib.dump(rf_causal, MODEL_DIR / "rf_causal.pkl")

joblib.dump(xgb_all, MODEL_DIR / "xgb_all.pkl")
joblib.dump(xgb_causal, MODEL_DIR / "xgb_causal.pkl")

joblib.dump(lgb_all, MODEL_DIR / "lgb_all.pkl")
joblib.dump(lgb_causal, MODEL_DIR / "lgb_causal.pkl")

print("="*60)
print("Models Saved Successfully!")
print("="*60)