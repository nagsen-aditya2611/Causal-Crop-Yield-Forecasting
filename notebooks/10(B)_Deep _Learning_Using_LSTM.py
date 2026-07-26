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

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

print("TensorFlow Version :", tf.__version__)
print("Libraries Imported Successfully!")

# ============================================================
# OUTPUT FOLDERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "figures" / "figures_deep_learning_using_LSTM"

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
# CAUSAL FEATURES
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
# RESHAPE DATA FOR LSTM
# ============================================================

# LSTM requires input shape:
# (samples, timesteps, features)

X_train_all_lstm = X_train_all.reshape(

    X_train_all.shape[0],
    1,
    X_train_all.shape[1]

)

X_test_all_lstm = X_test_all.reshape(

    X_test_all.shape[0],
    1,
    X_test_all.shape[1]

)

X_train_causal_lstm = X_train_causal.reshape(

    X_train_causal.shape[0],
    1,
    X_train_causal.shape[1]

)

X_test_causal_lstm = X_test_causal.reshape(

    X_test_causal.shape[0],
    1,
    X_test_causal.shape[1]

)

print("="*70)
print("LSTM INPUT SHAPES")
print("="*70)

print("All Features Train :", X_train_all_lstm.shape)
print("All Features Test  :", X_test_all_lstm.shape)

print()

print("Causal Features Train :", X_train_causal_lstm.shape)
print("Causal Features Test  :", X_test_causal_lstm.shape)


# ============================================================
# LSTM MODEL FUNCTION
# ============================================================

def build_lstm_model(input_shape):

    model = Sequential([

        LSTM(
            64,
            activation="tanh",
            input_shape=input_shape
        ),

        Dropout(0.20),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1
        )

    ])

    model.compile(

        optimizer="adam",

        loss="mse",

        metrics=["mae"]

    )

    return model

print("LSTM Model Function Created Successfully!")

# ============================================================
# EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=20,

    restore_best_weights=True,

    verbose=1

)

print("Early Stopping Ready!")


# ============================================================
# LSTM
# ALL FEATURES
# ============================================================

lstm_all = build_lstm_model(

    (X_train_all_lstm.shape[1],
     X_train_all_lstm.shape[2])

)

history_all = lstm_all.fit(

    X_train_all_lstm,

    y_train,

    validation_split=0.20,

    epochs=200,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)

print("Training Completed!")


# ============================================================
# EVALUATE LSTM
# ALL FEATURES
# ============================================================

pred_lstm_all = lstm_all.predict(X_test_all_lstm)

pred_lstm_all = pred_lstm_all.flatten()

mae_all = mean_absolute_error(

    y_test,

    pred_lstm_all

)

rmse_all = np.sqrt(

    mean_squared_error(

        y_test,

        pred_lstm_all

    )

)

r2_all = r2_score(

    y_test,

    pred_lstm_all

)

print("="*60)
print("LSTM")
print("All Features")
print("="*60)

print(f"MAE  : {mae_all:.4f}")
print(f"RMSE : {rmse_all:.4f}")
print(f"R²   : {r2_all:.4f}")

# ============================================================
# LSTM
# CAUSAL FEATURES
# ============================================================

lstm_causal = build_lstm_model(

    (
        X_train_causal_lstm.shape[1],
        X_train_causal_lstm.shape[2]
    )

)

history_causal = lstm_causal.fit(

    X_train_causal_lstm,

    y_train,

    validation_split=0.20,

    epochs=200,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)

print("Causal LSTM Training Completed!")


# ============================================================
# EVALUATE LSTM
# CAUSAL FEATURES
# ============================================================

pred_lstm_causal = lstm_causal.predict(

    X_test_causal_lstm

)

pred_lstm_causal = pred_lstm_causal.flatten()

mae_causal = mean_absolute_error(

    y_test,
    pred_lstm_causal

)

rmse_causal = np.sqrt(

    mean_squared_error(

        y_test,
        pred_lstm_causal

    )

)

r2_causal = r2_score(

    y_test,
    pred_lstm_causal

)

print("="*60)
print("LSTM")
print("Causal Features")
print("="*60)

print(f"MAE  : {mae_causal:.4f}")
print(f"RMSE : {rmse_causal:.4f}")
print(f"R²   : {r2_causal:.4f}")

# ============================================================
# SAVE MODELS
# ============================================================

lstm_all.save(

    MODEL_DIR / "lstm_all_features.keras"

)

lstm_causal.save(

    MODEL_DIR / "lstm_causal_features.keras"

)

print("Models Saved Successfully!")

plt.figure(figsize=(8,5))

plt.plot(
    history_all.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_all.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training Loss (All Features)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "lstm_all_loss.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8,5))

plt.plot(
    history_causal.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_causal.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training Loss (Causal Features)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "lstm_causal_loss.png",
    dpi=300
)

plt.show()

comparison = pd.DataFrame({

    "Model": [

        "LSTM (All Features)",
        "LSTM (Causal Features)"

    ],

    "MAE": [

        mae_all,
        mae_causal

    ],

    "RMSE": [

        rmse_all,
        rmse_causal

    ],

    "R2": [

        r2_all,
        r2_causal

    ]

})

print(comparison)

comparison.to_csv(

    REPORT_DIR / "lstm_comparison.csv",

    index=False

)


# ============================================================
# ACTUAL VS PREDICTED
# ALL FEATURES
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,
    pred_lstm_all,

    alpha=0.6

)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    color="red",
    linestyle="--"

)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.title("LSTM: Actual vs Predicted (All Features)")

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "actual_vs_predicted_all.png",

    dpi=300

)

plt.show()


# ============================================================
# ACTUAL VS PREDICTED
# CAUSAL FEATURES
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,
    pred_lstm_causal,

    alpha=0.6

)

plt.plot(

    [y_test.min(), y_test.max()],

    [y_test.min(), y_test.max()],

    color="red",
    linestyle="--"

)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")

plt.title("LSTM: Actual vs Predicted (Causal Features)")

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "actual_vs_predicted_causal.png",

    dpi=300

)

plt.show()

residuals_all = y_test - pred_lstm_all

plt.figure(figsize=(8,5))

plt.scatter(

    pred_lstm_all,
    residuals_all,

    alpha=0.6

)

plt.axhline(

    0,

    color="red",

    linestyle="--"

)

plt.xlabel("Predicted Yield")

plt.ylabel("Residual")

plt.title("Residual Plot (All Features)")

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "residual_all.png",

    dpi=300

)

plt.show()


residuals_causal = y_test - pred_lstm_causal

plt.figure(figsize=(8,5))

plt.scatter(

    pred_lstm_causal,
    residuals_causal,

    alpha=0.6

)

plt.axhline(

    0,

    color="red",

    linestyle="--"

)

plt.xlabel("Predicted Yield")

plt.ylabel("Residual")

plt.title("Residual Plot (Causal Features)")

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "residual_causal.png",

    dpi=300

)

plt.show()

results = pd.DataFrame({

    "Model":[

        "LSTM All Features",
        "LSTM Causal Features"

    ],

    "MAE":[

        mae_all,
        mae_causal

    ],

    "RMSE":[

        rmse_all,
        rmse_causal

    ],

    "R2":[

        r2_all,
        r2_causal

    ]

})

results.to_csv(

    REPORT_DIR / "lstm_results.csv",

    index=False

)

# ============================================================
# SAVE LSTM PREDICTIONS
# ============================================================

lstm_predictions = pd.DataFrame({

    "Actual": y_test.values,

    "LSTM_All": pred_lstm_all,

    "LSTM_Causal": pred_lstm_causal

})

lstm_predictions.to_csv(

    REPORT_DIR / "lstm_predictions.csv",

    index=False

)

print("="*60)
print("LSTM Predictions Saved Successfully!")
print("="*60)

print(results)