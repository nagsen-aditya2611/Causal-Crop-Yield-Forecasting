import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

st.title("🤖 Model Performance")

st.markdown("""
Comparison of Machine Learning and Deep Learning models trained using

- All Features
- Selected Causal Features
""")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

comparison = pd.read_csv(
    REPORT_DIR / "model_comparison.csv"
)

st.subheader("🏆 Model Leaderboard")

st.dataframe(
    comparison,
    use_container_width=True
)

st.subheader("📊 R² Comparison")

comparison_sorted = comparison.sort_values(
    "All Features R2",
    ascending=False
)

fig, ax = plt.subplots(figsize=(9,5))

bars = ax.bar(

    comparison_sorted["Model"],

    comparison_sorted["All Features R2"]

)

ax.set_ylabel("R²")

ax.set_ylim(0.6,1.0)

for bar in bars:

    h = bar.get_height()

    ax.text(

        bar.get_x()+bar.get_width()/2,

        h+0.01,

        f"{h:.3f}",

        ha="center"

    )

st.pyplot(fig)

st.subheader("🌱 Performance Retained")

fig, ax = plt.subplots(figsize=(9,5))

bars = ax.bar(

    comparison_sorted["Model"],

    comparison_sorted["Performance Retained (%)"]

)

ax.set_ylabel("%")

ax.set_ylim(60,100)

for bar in bars:

    h = bar.get_height()

    ax.text(

        bar.get_x()+bar.get_width()/2,

        h+0.5,

        f"{h:.1f}%",

        ha="center"

    )

st.pyplot(fig)

best = comparison.sort_values(
    "All Features R2",
    ascending=False
).iloc[0]

st.success(f"""
### 🏆 Best Model

**{best['Model']}**

R² : **{best['All Features R2']:.3f}**

Performance Retained : **{best['Performance Retained (%)']:.1f}%**
""")

st.subheader("📌 Observations")

st.markdown("""

- Ensemble models consistently outperformed simpler models.

- XGBoost achieved the highest prediction accuracy.

- Causal feature selection preserved more than 90% of predictive performance for most ensemble models.

- Deep Learning models remained competitive despite using fewer causal variables.

- The results demonstrate that interpretability can be improved without substantially reducing prediction accuracy.

""")