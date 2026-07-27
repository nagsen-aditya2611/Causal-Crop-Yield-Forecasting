import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About This Project")

st.markdown("""
## 🌾 Causal Crop Yield Forecasting

This project demonstrates how **Causal Inference**, **Machine Learning**, and **Explainable AI**
can be integrated to build an interpretable crop yield prediction system.

Instead of relying only on correlation, the workflow identifies the variables that
causally influence crop yield and evaluates how those causal features affect predictive
performance.
""")

st.divider()

st.header("Project Pipeline")

st.markdown("""
- Dataset Understanding (CY-Bench)
- Data Cleaning & Preprocessing
- Exploratory Data Analysis
- Expert Causal DAG Construction
- Causal Discovery (NOTEARS)
- Treatment Effect Estimation
- Refutation Analysis
- Causal Feature Selection
- Machine Learning Models
- Deep Learning Models
- SHAP Explainability
- Interactive Streamlit Dashboard
""")

st.divider()

st.header("Technologies Used")

st.markdown("""
### Programming
- Python
- Streamlit

### Data Analysis
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM

### Causal Inference
- DoWhy
- EconML
- Causal Discovery (NOTEARS)

### Explainability
- SHAP
""")

st.divider()

st.header("Key Features")

st.markdown("""
✅ Interactive dataset exploration

✅ Exploratory data analysis dashboards

✅ Causal graph visualization

✅ Treatment effect estimation

✅ Machine learning model comparison

✅ SHAP feature importance visualization

✅ Interactive crop yield prediction

✅ PDF prediction report generation
""")

st.divider()

st.header("Author")

st.markdown("""
GitHub:
https://github.com/nagsen-aditya2611
""")

st.success("Thank you for exploring the project! 🌾")