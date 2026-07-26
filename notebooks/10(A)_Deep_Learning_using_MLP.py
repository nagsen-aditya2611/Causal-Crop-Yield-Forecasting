import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings("ignore")

print("Libraries Imported Successfully!")

# ============================================================
# OUTPUT FOLDERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "figures" / "figures_dl(a)"

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

all_features = df.select_dtypes(include=[np.number]).columns.tolist()

all_features.remove(target)

# Remove leakage feature
leakage_features = [

    "production"

]

for col in leakage_features:

    if col in all_features:

        all_features.remove(col)

X_all = df[all_features]

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

print("="*70)
print("TRAIN TEST SPLIT")
print("="*70)

print("Training Samples :", len(X_all_train))
print("Testing Samples :", len(X_all_test))

# ============================================================
# FEATURE SCALING
# ============================================================

scaler_all = StandardScaler()

X_train_all = scaler_all.fit_transform(X_all_train)
X_test_all = scaler_all.transform(X_all_test)

scaler_causal = StandardScaler()

X_train_causal = scaler_causal.fit_transform(X_causal_train)
X_test_causal = scaler_causal.transform(X_causal_test)

print("Feature Scaling Completed Successfully!")


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

results = []

def evaluate_model(model,
                   X_train,
                   X_test,
                   y_train,
                   y_test,
                   model_name,
                   feature_set):

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
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
# MLP MODEL
# ============================================================

mlp = MLPRegressor(

    hidden_layer_sizes=(128,64,32),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    max_iter=500,

    early_stopping=True,

    validation_fraction=0.1,

    random_state=42

)


# ============================================================
# MLP
# ALL FEATURES
# ============================================================

mlp_all, pred_mlp_all = evaluate_model(

    mlp,

    X_train_all,
    X_test_all,

    y_train,
    y_test,

    "MLP",
    "All Features"

)

# ============================================================
# MLP
# CAUSAL FEATURES
# ============================================================

mlp = MLPRegressor(

    hidden_layer_sizes=(128,64,32),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    max_iter=500,

    early_stopping=True,

    validation_fraction=0.1,

    random_state=42

)

mlp_causal, pred_mlp_causal = evaluate_model(

    mlp,

    X_train_causal,
    X_test_causal,

    y_train,
    y_test,

    "MLP",
    "Causal Features"

)

# ============================================================
# MLP PERFORMANCE SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.pivot(

    index="Model",

    columns="Feature_Set",

    values="R2"

).reset_index()

results_df.columns = [

    "Model",

    "All Features R2",

    "Causal Features R2"

]

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

    REPORT_DIR / "mlp_model_comparison.csv",

    index=False

)

# ============================================================
# R² COMPARISON
# ============================================================

plt.figure(figsize=(8,6))

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

    results_df["Model"]

)

plt.ylabel("R² Score")

plt.title("MLP Performance Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "mlp_model_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(

    mlp_all.loss_curve_,

    linewidth=2

)

plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.title("MLP Training Loss Curve")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "mlp_loss_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7,7))

plt.scatter(

    y_test,

    pred_mlp_all,

    alpha=0.6

)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    "r--",

    linewidth=2

)

plt.xlabel("Actual Yield")

plt.ylabel("Predicted Yield")

plt.title("MLP : Actual vs Predicted")

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "mlp_actual_vs_predicted.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual": y_test.values,

    "MLP_All": pred_mlp_all,

    "MLP_Causal": pred_mlp_causal

})

prediction_df.to_csv(

    REPORT_DIR / "mlp_predictions.csv",

    index=False

)

print("Predictions saved successfully!")

# ============================================================
# SAVE MODELS
# ============================================================

import joblib

joblib.dump(

    mlp_all,

    MODEL_DIR / "mlp_all_features.pkl"

)

joblib.dump(

    mlp_causal,

    MODEL_DIR / "mlp_causal_features.pkl"

)

print("Models saved successfully!")

# ============================================================
# SAVE MLP RESULTS
# ============================================================

results_df.to_csv(

    REPORT_DIR / "mlp_results.csv",

    index=False

)

print("="*60)
print("MLP Results Saved Successfully!")
print("="*60)