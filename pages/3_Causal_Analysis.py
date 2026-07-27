import streamlit as st
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("🌱 Causal Analysis")

st.markdown(
"""
This page summarizes the causal inference pipeline used to identify
the drivers of crop yield before building predictive models.
"""
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = PROJECT_ROOT / "figures"

# ==========================================================
# LOAD RESULTS
# ==========================================================

variable_roles = pd.read_csv(
    REPORT_DIR / "variable_roles.csv"
)

selected_features = pd.read_csv(
    REPORT_DIR / "selected_features.csv"
)

final_ate = pd.read_csv(
    REPORT_DIR / "final_ate_comparison.csv"
)

st.subheader("📋 Variable Roles")

st.dataframe(
    variable_roles,
    use_container_width=True
)

st.subheader("🌾 Selected Causal Features")

st.dataframe(
    selected_features,
    use_container_width=True
)

st.subheader("📊 Average Treatment Effect")

st.dataframe(
    final_ate,
    use_container_width=True
)

st.subheader("📈 Treatment Effect Comparison")

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(

    final_ate["Method"],

    final_ate["ATE"]

)

ax.set_ylabel("Estimated ATE")

plt.xticks(rotation=25)

st.pyplot(fig)

st.divider()

st.subheader("🔄 Causal Analysis Pipeline")

st.markdown("""
1. Dataset Preparation

↓

2. Domain Knowledge (Expert DAG)

↓

3. Causal Discovery (NOTEARS)

↓

4. Treatment & Outcome Selection

↓

5. ATE Estimation

↓

6. Causal Feature Selection

↓

7. Machine Learning

↓

8. Model Comparison
""")

st.divider()

st.subheader("🧠 Methodology")

st.markdown("""

### Step 1
Expert agricultural knowledge was used to define an initial causal graph.

### Step 2
A causal discovery algorithm (NOTEARS) learned additional causal relationships directly from the data.

### Step 3
Average Treatment Effects (ATE) were estimated using multiple causal inference methods.

### Step 4
Only the important causal drivers were selected for downstream prediction models.

### Step 5
Machine Learning and Deep Learning models were trained using:

- All Features
- Selected Causal Features

The predictive performances were then compared.
""")

st.divider()

st.subheader("📌 Key Findings")

best_model = "XGBoost"

best_r2 = 0.904

st.success(f"""
✔ Best Predictive Model

**{best_model}**

R² = **{best_r2:.3f}**
""")

st.info("""
Important observations

• Causal feature selection retained almost all predictive power.

• Models trained using causal features remained highly competitive.

• The proposed pipeline improves interpretability while maintaining prediction accuracy.

• Causal reasoning provides better scientific explanations than purely correlation-based models.
""")

st.divider()

st.subheader("🎓 Research Contribution")

st.markdown("""

This project demonstrates that:

- Causal Inference can identify meaningful agricultural drivers.

- Machine Learning models can maintain high predictive accuracy using only causal variables.

- Deep Learning and ensemble models both benefit from causal feature engineering.

- Explainability is improved without sacrificing performance.

""")

