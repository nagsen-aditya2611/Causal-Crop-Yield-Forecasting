import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🔍 SHAP Explainability")

st.markdown("""
SHAP (SHapley Additive exPlanations) provides interpretable explanations
for machine learning predictions by quantifying the contribution of each feature.
""")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = PROJECT_ROOT / "figures" / "figures_shap"

importance = pd.read_csv(
    REPORT_DIR / "rf_shap_importance.csv"
)

st.subheader("📊 SHAP Feature Importance")

st.dataframe(
    importance,
    use_container_width=True
)

st.subheader("🌳 SHAP Summary Plot")

st.image(
    str(FIGURE_DIR / "rf_summary_plot.png"),
    use_container_width=True
)

st.subheader("📈 SHAP Bar Plot")

st.image(
    str(FIGURE_DIR / "rf_bar_plot.png"),
    use_container_width=True
)


st.subheader("🏆 Top 10 Important Features")

st.dataframe(
    importance.head(10),
    use_container_width=True
)

st.subheader("💡 Interpretation")

top = importance.iloc[0]["Feature"]

st.success(f"""
The Random Forest model identified **{top}**
as the strongest contributor to crop yield prediction.

Features with larger SHAP values exert greater influence on the model's output,
whereas lower-ranked variables contribute comparatively less.

This confirms that the prediction model is driven primarily by
agronomically meaningful variables rather than arbitrary correlations.
""")


st.subheader("📌 Key Insights")

st.markdown("""

- SHAP improves transparency by explaining model predictions.

- The highest-ranked variables align with domain knowledge.

- Feature importance supports the causal feature selection strategy.

- Explainability increases confidence in deploying predictive models for agricultural decision support.

""")

