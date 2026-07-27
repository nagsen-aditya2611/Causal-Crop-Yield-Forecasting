import streamlit as st

st.set_page_config(
    page_title="Causal Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Sidebar
# ==========================

st.sidebar.title("🌾 Crop Yield Forecasting")

st.sidebar.markdown("""

**Technologies Used**

- Causal Inference
- Machine Learning
- Deep Learning
- SHAP Explainability
- Streamlit
""")

# ==========================
# Main Page
# ==========================

st.title("🌾 Causal Crop Yield Forecasting")

st.subheader(
    "Integrating Causal Inference, Machine Learning and Deep Learning for Agricultural Decision Support"
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", "CY-Bench")

with col2:
    st.metric("Samples", "1,324")

with col3:
    st.metric("Models", "6")

with col4:
    st.metric("Best R²", "0.904")

st.markdown("---")

st.header("Project Objective")

st.write("""
This project proposes a **causal machine learning framework** for crop yield prediction.

Unlike conventional predictive models, this framework first identifies
**causal drivers of crop yield**, estimates treatment effects using
causal inference techniques, and finally compares predictive models
trained using:

- All available features
- Selected causal features

The objective is to evaluate whether causal feature selection can produce
interpretable models without sacrificing predictive performance.
""")

st.markdown("---")

st.header("Project Workflow")

st.markdown("""
1. Dataset Understanding

2. Exploratory Data Analysis

3. Expert DAG Construction

4. Causal Discovery (NOTEARS)

5. Treatment Effect Estimation

6. Refutation Analysis

7. Feature Selection

8. Machine Learning Models

9. Deep Learning Models

10. SHAP Explainability

11. Model Comparison

12. Interactive Dashboard
""")

st.success("Use the pages in the left sidebar to explore each stage of the project.")