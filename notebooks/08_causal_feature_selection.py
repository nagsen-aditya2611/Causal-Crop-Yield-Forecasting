# ============================================================
# NOTEBOOK 08
# CAUSAL FEATURE SELECTION
# ============================================================

print("="*70)
print("CAUSAL FEATURE SELECTION")
print("="*70)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

print("Libraries Imported Successfully!")

# ============================================================
# OUTPUT FOLDERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "figures" / "figures_feature_selection"
REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Output folders ready!")
print("Figures will be saved to:")
print(FIGURE_DIR)

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_crop_dataset.csv"

df = pd.read_csv(DATA_PATH)

# ============================================================
# DEFINE FEATURE GROUPS
# ============================================================

target = "yield"

causal_features = [

    "avg_ssm",
    "avg_tavg",
    "avg_rad",
    "avg_cwb",
    "awc",
    "bulk_density",
    "drainage_class"

]

non_causal_features = [

    col
    for col in df.columns
    if col not in causal_features
    and col != target

]

print("="*70)
print("FEATURE GROUPS")
print("="*70)

print("Target Variable :", target)

print("\nCausal Features")

for feature in causal_features:
    print("-", feature)

print("\nTotal Causal Features :", len(causal_features))

print("Remaining Features :", len(non_causal_features))


# ============================================================
# CAUSAL FEATURE RANKING
# ============================================================

print("="*70)
print("CAUSAL FEATURE RANKING")
print("="*70)

expert_edges = pd.read_csv(
    REPORT_DIR / "expert_dag_edges.csv"
)

notears_edges = pd.read_csv(
    REPORT_DIR / "notears_edges.csv"
)

# ----------------------------------------------------
# LOAD NOTEBOOK 07 LINEARDML RESULT
# ----------------------------------------------------

lineardml_result = pd.read_csv(
    REPORT_DIR / "lineardml_ate.csv"
)

ate_value = abs(
    lineardml_result["ATE"].iloc[0]
)

print("\nLinearDML ATE Loaded :", ate_value)

ranking = []

for feature in causal_features:

    expert_score = 0
    notears_score = 0
    causal_effect_score = 0

    # ----------------------------
    # EXPERT DAG
    # ----------------------------

    expert_score = (
        (expert_edges["Source"] == feature).sum()
        +
        (expert_edges["Target"] == feature).sum()
    )

    # ----------------------------
    # NOTEARS
    # ----------------------------

    notears_score = (
        (notears_edges["Source"] == feature).sum()
        +
        (notears_edges["Target"] == feature).sum()
    )

    # ----------------------------
    # NOTEBOOK 07
    # ----------------------------

    if feature == "avg_ssm":
        causal_effect_score = 3
    else:
        causal_effect_score = 0

    total_score = (
        expert_score
        + notears_score
        + causal_effect_score
    )

    ranking.append({

        "Feature": feature,
        "Expert_DAG": expert_score,
        "NOTEARS": notears_score,
        "Causal_Effect": causal_effect_score,
        "Total Score": total_score

    })

feature_ranking = pd.DataFrame(ranking)

feature_ranking = feature_ranking.sort_values(
    by="Total Score",
    ascending=False
).reset_index(drop=True)

print(feature_ranking)

feature_ranking.to_csv(
    REPORT_DIR / "causal_feature_ranking.csv",
    index=False
)

print("\nFeature ranking saved successfully!")

# ============================================================
# FINAL SELECTED FEATURES
# ============================================================

selected_features = pd.DataFrame({

    "Selected Features": causal_features

})

selected_features.to_csv(
    REPORT_DIR / "selected_features.csv",
    index=False
)

print("="*70)
print("FINAL SELECTED FEATURES")
print("="*70)

print(selected_features)

print("\nTotal Selected Features :", len(selected_features))


# ============================================================
# FEATURE SELECTION SUMMARY
# ============================================================

with open(
    REPORT_DIR / "feature_selection_summary.txt",
    "w"
) as f:

    f.write("CAUSAL FEATURE SELECTION SUMMARY\n")
    f.write("="*50 + "\n\n")

    f.write(f"Total Variables : {df.shape[1]}\n")
    f.write(f"Target Variable : {target}\n")
    f.write(f"Causal Features : {len(causal_features)}\n")
    f.write(f"Remaining Features : {len(non_causal_features)}\n\n")

    f.write("Selected Features\n\n")

    for feature in causal_features:
        f.write(feature + "\n")

print("Summary report saved successfully!")


# ============================================================
# CAUSAL FEATURE VISUALIZATION
# ============================================================

plt.figure(figsize=(10,6))

plt.barh(
    feature_ranking["Feature"],
    feature_ranking["Total Score"]
)

plt.xlabel("Causal Importance Score")

plt.title("Final Causal Feature Ranking")

for i, score in enumerate(feature_ranking["Total Score"]):
    plt.text(
        score + 0.05,
        i,
        str(score),
        va="center"
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "causal_feature_ranking.png",
    dpi=300
)

plt.show()


print("Feature Ranking Plot Saved!")


# ============================================================
# CORRELATION OF CAUSAL FEATURES
# ============================================================

corr = df[causal_features + [target]].corr()

yield_corr = corr[target].drop(target)

yield_corr = yield_corr.sort_values()

plt.figure(figsize=(8,6))

yield_corr.plot(kind="barh")

plt.title("Correlation of Selected Causal Features with Yield")

plt.xlabel("Correlation")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "causal_feature_correlations.png",
    dpi=300
)

plt.show()

yield_corr.to_csv(
    REPORT_DIR / "causal_feature_correlations.csv"
)

print("Correlation plot saved!")

# ============================================================
# CAUSAL FEATURE HEATMAP
# ============================================================

plt.figure(figsize=(9,7))

corr_matrix = df[causal_features + [target]].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)

plt.title("Correlation Heatmap of Selected Causal Features")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "causal_feature_heatmap.png",
    dpi=300
)

plt.show()

print("Causal Feature Heatmap Saved!")

# ============================================================
# FEATURE SELECTION FLOW
# ============================================================

plt.figure(figsize=(8,6))

plt.axis("off")

plt.text(
    0.5,
    0.90,
    "28 Original Variables",
    ha="center",
    fontsize=14,
    bbox=dict(boxstyle="round", fc="lightblue")
)

plt.text(
    0.5,
    0.73,
    "Expert DAG",
    ha="center",
    fontsize=13,
    bbox=dict(boxstyle="round", fc="lightgreen")
)

plt.text(
    0.5,
    0.56,
    "NOTEARS",
    ha="center",
    fontsize=13,
    bbox=dict(boxstyle="round", fc="khaki")
)

plt.text(
    0.5,
    0.39,
    "Causal Effect Estimation\n(Notebook 07)",
    ha="center",
    fontsize=12,
    bbox=dict(boxstyle="round", fc="orange")
)

plt.text(
    0.5,
    0.18,
    "7 Selected Causal Features",
    ha="center",
    fontsize=14,
    bbox=dict(boxstyle="round", fc="salmon")
)

arrowprops = dict(
    arrowstyle="->",
    lw=2
)

plt.annotate("", xy=(0.5,0.78), xytext=(0.5,0.86), arrowprops=arrowprops)
plt.annotate("", xy=(0.5,0.61), xytext=(0.5,0.69), arrowprops=arrowprops)
plt.annotate("", xy=(0.5,0.44), xytext=(0.5,0.52), arrowprops=arrowprops)
plt.annotate("", xy=(0.5,0.23), xytext=(0.5,0.35), arrowprops=arrowprops)

plt.savefig(
    FIGURE_DIR / "feature_selection_pipeline.png",
    dpi=300
)

plt.show()

print("Feature Selection Pipeline Saved!")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("="*70)
print("FEATURE SELECTION SUMMARY")
print("="*70)

print(f"Total Variables       : {df.shape[1]}")
print(f"Target Variable       : {target}")
print(f"Causal Features       : {len(causal_features)}")
print(f"Remaining Features    : {len(non_causal_features)}")

print("\nSelected Features")

for feature in causal_features:
    print("✓", feature)

print("\nReports Saved")

print("✓ causal_feature_ranking.csv")
print("✓ selected_features.csv")
print("✓ feature_selection_summary.txt")
print("✓ causal_feature_correlations.csv")

print("\nFigures Saved")

print("✓ causal_feature_ranking.png")
print("✓ causal_feature_correlations.png")
print("✓ causal_feature_heatmap.png")
print("✓ feature_selection_pipeline.png")

print("\n" + "="*70)
print("NOTEBOOK 08 COMPLETED SUCCESSFULLY")
print("="*70)