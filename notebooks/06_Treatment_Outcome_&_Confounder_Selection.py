# ============================================================
# NOTEBOOK 06
# TREATMENT, OUTCOME & CONFOUNDER SELECTION
# ============================================================

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 70)
print("TREATMENT, OUTCOME & CONFOUNDER SELECTION")
print("=" * 70)

print("Libraries Imported Successfully!")

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Output folder ready!")

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
# TREATMENT VARIABLE
# ============================================================

treatment = "avg_ssm"

print("\nTreatment Variable")

print("----------------------------")

print(treatment)

# ============================================================
# OUTCOME VARIABLE
# ============================================================

outcome = "yield"

print("\nOutcome Variable")

print("----------------------------")

print(outcome)

# ============================================================
# CONFOUNDERS
# ============================================================

confounders = [

    "avg_tavg",

    "avg_rad",

    "avg_cwb",

    "awc",

    "bulk_density",

    "drainage_class",

    "harvest_year"

]

print("\nConfounders")

print("----------------------------")

for var in confounders:

    print(var)

# ============================================================
# MEDIATORS
# ============================================================

mediators = [

    "avg_ndvi",

    "avg_fpar"

]

print("\nMediators")

print("----------------------------")

for var in mediators:

    print(var)

# ============================================================
# SAVE VARIABLES
# ============================================================

variable_summary = pd.DataFrame({

    "Treatment": [treatment],

    "Outcome": [outcome],

    "Confounders": [", ".join(confounders)],

    "Mediators": [", ".join(mediators)]

})

variable_summary.to_csv(

    REPORT_DIR / "treatment_outcome_summary.csv",

    index=False

)

print("\nVariable Summary Saved Successfully!")

