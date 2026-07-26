# ============================================================
# NOTEBOOK 07
# CAUSAL EFFECT ESTIMATION
# ============================================================

import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dowhy import CausalModel

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

print("=" * 70)
print("CAUSAL EFFECT ESTIMATION")
print("=" * 70)

print("Libraries Imported Successfully!")

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FIGURE_DIR = Path("figures/figures_causal_effect")

REPORT_DIR = Path("reports")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Output folders ready!")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/processed/final_crop_dataset.csv"
)

print("\nDataset Loaded Successfully!")

print("Shape :", df.shape)

print(df.head())

# ============================================================
# DEFINE CAUSAL VARIABLES
# ============================================================

treatment = "avg_ssm"

outcome = "yield"

confounders = [

    "avg_tavg",

    "avg_rad",

    "avg_cwb",

    "awc",

    "bulk_density",

    "drainage_class",

    "harvest_year"

]

print("\nTreatment :", treatment)

print("Outcome   :", outcome)

print("\nConfounders")

for var in confounders:

    print(" -", var)

# ============================================================
# BUILD CAUSAL MODEL
# ============================================================

causal_model = CausalModel(

    data=df,

    treatment=treatment,

    outcome=outcome,

    common_causes=confounders

)

print("\nDoWhy Causal Model Created Successfully!")

# ============================================================
# IDENTIFY CAUSAL ESTIMAND
# ============================================================

identified_estimand = causal_model.identify_effect(
    proceed_when_unidentifiable=True
)

print("\n" + "="*70)
print("IDENTIFIED CAUSAL ESTIMAND")
print("="*70)

print(identified_estimand)

# ============================================================
# ESTIMATE CAUSAL EFFECT
# LINEAR REGRESSION ADJUSTMENT
# ============================================================

causal_estimate = causal_model.estimate_effect(

    identified_estimand,

    method_name="backdoor.linear_regression"

)

print("\n" + "="*70)
print("AVERAGE TREATMENT EFFECT (LINEAR REGRESSION)")
print("="*70)

print(causal_estimate)

print("\nEstimated ATE :")

print(causal_estimate.value)

# ============================================================
# SAVE ATE RESULT
# ============================================================

ate_result = pd.DataFrame({

    "Method": ["Linear Regression"],

    "ATE": [causal_estimate.value]

})

ate_result.to_csv(

    REPORT_DIR / "linear_regression_ate.csv",

    index=False

)

print("\nATE Saved Successfully!")

# ============================================================
# CREATE BINARY TREATMENT
# ============================================================

median_ssm = df["avg_ssm"].median()

df["treatment_binary"] = (

    df["avg_ssm"] >= median_ssm

).astype(int)

print("="*70)

print("BINARY TREATMENT CREATED")

print("="*70)

print(df["treatment_binary"].value_counts())


# ============================================================
# PROPENSITY SCORE MODEL
# ============================================================

from sklearn.linear_model import LogisticRegression

X = df[confounders]

T = df["treatment_binary"]

ps_model = LogisticRegression(

    max_iter=1000

)

ps_model.fit(X, T)

df["propensity_score"] = ps_model.predict_proba(X)[:,1]

print("\nPropensity Scores Estimated Successfully!")

print(df["propensity_score"].describe())

# ============================================================
# SAVE PROPENSITY SCORES
# ============================================================

df[

    ["propensity_score"]

].to_csv(

    REPORT_DIR / "propensity_scores.csv",

    index=False

)

print("Propensity Scores Saved!")

# ============================================================
# PROPENSITY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(

    df["propensity_score"],

    bins=30,

    edgecolor="black"

)

plt.xlabel("Propensity Score")

plt.ylabel("Frequency")

plt.title("Distribution of Propensity Scores")

plt.tight_layout()

plt.savefig(

    FIGURE_DIR /

    "propensity_score_distribution.png",

    dpi=300

)

plt.close()

print("Figure Saved!")

from sklearn.neighbors import NearestNeighbors
# ============================================================
# SPLIT TREATED AND CONTROL GROUPS
# ============================================================

treated = df[df["treatment_binary"] == 1].copy()

control = df[df["treatment_binary"] == 0].copy()

print("=" * 70)
print("MATCHING DATA")
print("=" * 70)

print("Treated Units :", len(treated))
print("Control Units :", len(control))


# ============================================================
# NEAREST NEIGHBOR MATCHING
# ============================================================

nn = NearestNeighbors(
    n_neighbors=1
)

nn.fit(control[["propensity_score"]])

distances, indices = nn.kneighbors(
    treated[["propensity_score"]]
)

matched_control = control.iloc[indices.flatten()].copy()

matched_data = pd.concat(
    [treated, matched_control],
    axis=0
)

print("\nMatching Completed Successfully!")

print("Matched Dataset Shape :", matched_data.shape)

# ============================================================
# PSM ATE
# ============================================================

treated_mean = treated[outcome].mean()

matched_control_mean = matched_control[outcome].mean()

psm_ate = treated_mean - matched_control_mean

print("=" * 70)
print("PROPENSITY SCORE MATCHING")
print("=" * 70)

print("Mean Yield (Treated) :", treated_mean)

print("Mean Yield (Matched Control) :", matched_control_mean)

print("\nEstimated PSM ATE :", psm_ate)

# ============================================================
# SAVE PSM RESULT
# ============================================================

psm_result = pd.DataFrame({

    "Method": ["Propensity Score Matching"],

    "ATE": [psm_ate]

})

psm_result.to_csv(

    REPORT_DIR / "psm_ate.csv",

    index=False

)

print("\nPSM Result Saved Successfully!")

# ============================================================
# LOVE PLOT (STANDARDIZED MEAN DIFFERENCE)
# ============================================================

print("="*70)
print("LOVE PLOT")
print("="*70)

covariates = confounders

before = []
after = []

matched = matched_data.copy()

for col in covariates:

    # Before Matching
    treated_before = df[df["treatment_binary"] == 1][col]
    control_before = df[df["treatment_binary"] == 0][col]

    smd_before = (
        treated_before.mean() - control_before.mean()
    ) / np.sqrt(
        (treated_before.var() + control_before.var()) / 2
    )

    before.append(abs(smd_before))

    # After Matching
    treated_after = matched[matched["treatment_binary"] == 1][col]
    control_after = matched[matched["treatment_binary"] == 0][col]

    smd_after = (
        treated_after.mean() - control_after.mean()
    ) / np.sqrt(
        (treated_after.var() + control_after.var()) / 2
    )

    after.append(abs(smd_after))

love = pd.DataFrame({

    "Covariate": covariates,

    "Before": before,

    "After": after

})

plt.figure(figsize=(8,6))

plt.scatter(

    love["Before"],

    love["Covariate"],

    color="red",

    label="Before Matching",

    s=70

)

plt.scatter(

    love["After"],

    love["Covariate"],

    color="blue",

    label="After Matching",

    s=70

)

plt.axvline(

    0.1,

    color="black",

    linestyle="--",

    label="SMD = 0.1"

)

plt.xlabel("Absolute Standardized Mean Difference")

plt.ylabel("Covariates")

plt.title("Love Plot")

plt.legend()

plt.tight_layout()

plt.savefig(

    FIGURE_DIR / "love_plot.png",

    dpi=300

)

plt.close()

print("Love Plot Saved Successfully!")
love.to_csv(

    REPORT_DIR / "love_plot_values.csv",

    index=False

)

print("Love Plot Values Saved!")

# ============================================================
# STABILIZED IPW
# ============================================================

p_treated = df["treatment_binary"].mean()

df["ipw_weight"] = np.where(

    df["treatment_binary"] == 1,

    p_treated / df["propensity_score"],

    (1 - p_treated) / (1 - df["propensity_score"])

)

print("="*70)
print("STABILIZED IPW WEIGHTS")
print("="*70)

print(df["ipw_weight"].describe())



# ============================================================
# ESTIMATE IPW ATE
# ============================================================

treated = df[df["treatment_binary"] == 1]

control = df[df["treatment_binary"] == 0]

treated_mean = np.average(

    treated[outcome],

    weights=treated["ipw_weight"]

)

control_mean = np.average(

    control[outcome],

    weights=control["ipw_weight"]

)

ipw_ate = treated_mean - control_mean

print("=" * 70)
print("INVERSE PROBABILITY WEIGHTING")
print("=" * 70)

print("Weighted Treated Mean :", treated_mean)

print("Weighted Control Mean :", control_mean)

print()

print("Estimated IPW ATE :", ipw_ate)

# ============================================================
# SAVE IPW RESULT
# ============================================================

ipw_result = pd.DataFrame({

    "Method": ["IPW"],

    "ATE": [ipw_ate]

})

ipw_result.to_csv(

    REPORT_DIR / "ipw_ate.csv",

    index=False

)

print("IPW Result Saved!")

# ============================================================
# OUTCOME MODEL FOR AIPW
# ============================================================

from sklearn.linear_model import LinearRegression

outcome_model = LinearRegression()

outcome_model.fit(

    df[[treatment] + confounders],

    df[outcome]

)

print("Outcome Model Trained!")

# ============================================================
# PREDICTED OUTCOMES
# ============================================================

X1 = df[[treatment] + confounders].copy()

X0 = df[[treatment] + confounders].copy()

X1[treatment] = df[treatment].max()

X0[treatment] = df[treatment].min()

mu1 = outcome_model.predict(X1)

mu0 = outcome_model.predict(X0)

# ============================================================
# AUGMENTED IPW
# ============================================================

T = df["treatment_binary"]

Y = df[outcome]

e = df["propensity_score"]

aipw = (

    mu1

    - mu0

    + T * (Y - mu1) / e

    - (1 - T) * (Y - mu0) / (1 - e)

)

aipw_ate = np.mean(aipw)

print("=" * 70)
print("AUGMENTED INVERSE PROBABILITY WEIGHTING")
print("=" * 70)

print("Estimated AIPW ATE :", aipw_ate)

# ============================================================
# SAVE AIPW RESULT
# ============================================================

aipw_result = pd.DataFrame({

    "Method": ["AIPW"],

    "ATE": [aipw_ate]

})

aipw_result.to_csv(

    REPORT_DIR / "aipw_ate.csv",

    index=False

)

print("AIPW Result Saved!")

from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# PREPARE DATA FOR LINEARDML
# ============================================================

Y = df[outcome].values

T = df[treatment].values

X = df[confounders]

print("=" * 70)
print("LINEARDML DATA")
print("=" * 70)

print("Outcome :", Y.shape)
print("Treatment :", T.shape)
print("Confounders :", X.shape)

# ============================================================
# LINEARDML
# ============================================================

dml = LinearDML(

    model_y=RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    model_t=RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    random_state=42

)

dml.fit(

    Y,

    T,

    X=X

)

print("\nLinearDML Trained Successfully!")

# ============================================================
# LINEARDML ATE
# ============================================================

treatment_effect = dml.effect(X)

linear_dml_ate = treatment_effect.mean()

print("=" * 70)
print("LINEARDML")
print("=" * 70)

print("Estimated ATE :", linear_dml_ate)

# ============================================================
# SAVE LINEARDML
# ============================================================

pd.DataFrame({

    "Method":["LinearDML"],

    "ATE":[linear_dml_ate]

}).to_csv(

    REPORT_DIR / "lineardml_ate.csv",

    index=False

)

print("LinearDML Result Saved!")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Method":[

        "Linear Regression",

        "PSM",

        "IPW",

        "LinearDML"

    ],

    "ATE":[

        causal_estimate.value,

        psm_ate,

        ipw_ate,

        linear_dml_ate

    ]

})

print("="*70)
print("ATE COMPARISON")
print("="*70)

print(comparison)

comparison.to_csv(

    REPORT_DIR / "all_ate_results.csv",

    index=False

)

# ============================================================
# ATE COMPARISON PLOT
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Method"],

    comparison["ATE"]

)

plt.ylabel("Average Treatment Effect")

plt.title("Comparison of Causal Estimators")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(

    FIGURE_DIR /

    "ate_comparison.png",

    dpi=300

)

plt.close()

print("ATE Comparison Plot Saved!")

# ============================================================
# REFUTATION TEST 1
# RANDOM COMMON CAUSE
# ============================================================

print("="*70)
print("REFUTATION TEST : RANDOM COMMON CAUSE")
print("="*70)

random_refuter = causal_model.refute_estimate(
    identified_estimand,
    causal_estimate,
    method_name="random_common_cause"
)

print(random_refuter)

# ============================================================
# REFUTATION TEST 2
# PLACEBO TREATMENT
# ============================================================

print("="*70)
print("REFUTATION TEST : PLACEBO")
print("="*70)

placebo_refuter = causal_model.refute_estimate(
    identified_estimand,
    causal_estimate,
    method_name="placebo_treatment_refuter"
)

print(placebo_refuter)

# ============================================================
# REFUTATION TEST 3
# DATA SUBSET
# ============================================================

print("="*70)
print("REFUTATION TEST : DATA SUBSET")
print("="*70)

subset_refuter = causal_model.refute_estimate(
    identified_estimand,
    causal_estimate,
    method_name="data_subset_refuter"
)

print(subset_refuter)

# ============================================================
# SAVE REFUTATION RESULTS
# ============================================================

with open(REPORT_DIR / "refutation_results.txt","w") as f:

    f.write("Random Common Cause\n")
    f.write(str(random_refuter))
    f.write("\n\n")

    f.write("Placebo Test\n")
    f.write(str(placebo_refuter))
    f.write("\n\n")

    f.write("Data Subset Test\n")
    f.write(str(subset_refuter))

print("Refutation Results Saved!")

# ============================================================
# INDIVIDUAL TREATMENT EFFECTS
# ============================================================

ite = dml.effect(X)

print("="*70)
print("INDIVIDUAL TREATMENT EFFECTS")
print("="*70)

print("Mean :", np.mean(ite))
print("Std  :", np.std(ite))
print("Min  :", np.min(ite))
print("Max  :", np.max(ite))

# ============================================================
# ITE HISTOGRAM
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    ite,
    bins=30,
    edgecolor="black"
)

plt.xlabel("Individual Treatment Effect")

plt.ylabel("Frequency")

plt.title("Distribution of Individual Treatment Effects")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "ite_distribution.png",
    dpi=300
)

plt.close()

print("ITE Distribution Saved!")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Method":[

        "Linear Regression",

        "PSM",

        "IPW",

        "LinearDML"

    ],

    "ATE":[

        causal_estimate.value,

        psm_ate,

        ipw_ate,

        linear_dml_ate

    ]

})

print("="*70)
print("FINAL COMPARISON")
print("="*70)

print(comparison)

comparison.to_csv(
    REPORT_DIR / "final_ate_comparison.csv",
    index=False
)

# ============================================================
# ATE COMPARISON
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(
    comparison["Method"],
    comparison["ATE"]
)

plt.ylabel("Average Treatment Effect")

plt.title("Comparison of Causal Estimation Methods")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "ate_comparison.png",
    dpi=300
)

plt.close()

print("ATE Comparison Plot Saved!")

# ============================================================
# NOTEBOOK COMPLETED
# ============================================================

print("\n" + "="*70)
print("NOTEBOOK 07 COMPLETED SUCCESSFULLY")
print("="*70)

print("\nMethods Implemented")

print("✓ Linear Regression")
print("✓ Propensity Score Matching")
print("✓ Inverse Probability Weighting")
print("✓ LinearDML")
print("✓ Refutation Tests")
print("✓ Individual Treatment Effects")
print("✓ Final Comparison")

print("\nAll outputs saved successfully!")

