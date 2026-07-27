import streamlit as st
import pandas as pd
import numpy as np
import joblib


from pathlib import Path

from io import BytesIO

from reportlab.lib import colors

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.enums import TA_CENTER

from reportlab.platypus import (

    SimpleDocTemplate,

    Paragraph,

    Spacer,

    Table,

    TableStyle

)

from reportlab.lib.units import inch


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AI-Powered Crop Yield Prediction")

st.markdown("""
Predict crop yield using multiple Machine Learning and Deep Learning models
trained on the CY-Bench dataset.
""")

st.divider()

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_crop_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

df = load_dataset()

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    return {

        "Random Forest": joblib.load(MODEL_DIR / "rf_all.pkl"),

        "XGBoost": joblib.load(MODEL_DIR / "xgb_all.pkl"),

        "LightGBM": joblib.load(MODEL_DIR / "lgb_all.pkl"),

        "MLP": joblib.load(MODEL_DIR / "mlp_all_features.pkl")

    }

models = load_models()

# ==========================================================
# LOAD SCALERS
# ==========================================================

@st.cache_resource
def load_scalers():

    return {

        "MLP": joblib.load(MODEL_DIR / "mlp_scaler_all.pkl")

    }

scalers = load_scalers()

st.success("✅ Dataset, models and scalers loaded successfully.")


# ==========================================================
# MODEL SELECTION
# ==========================================================

st.subheader("⚙ Prediction Settings")

col1, col2 = st.columns(2)

with col1:

    selected_model = st.selectbox(

    "Select Prediction Model",

    [

        "Random Forest",

        "XGBoost",

        "LightGBM",

        "MLP"

    ]

)

# ==========================================================
# MODEL INFORMATION
# ==========================================================

model_info = {

    "Random Forest":
    {
        "emoji":"🌲",
        "desc":"Ensemble of decision trees. Stable and robust for tabular agricultural data.",
        "speed":"Fast",
        "accuracy":"High"
    },

    "XGBoost":
    {
        "emoji":"🚀",
        "desc":"Gradient boosting algorithm. Highest predictive performance in this project.",
        "speed":"Fast",
        "accuracy":"Very High"
    },

    "LightGBM":
    {
        "emoji":"⚡",
        "desc":"Lightweight gradient boosting model designed for speed and efficiency.",
        "speed":"Very Fast",
        "accuracy":"High"
    },

    "MLP":
    {
        "emoji":"🧠",
        "desc":"Feed-forward Neural Network capable of learning nonlinear relationships.",
        "speed":"Medium",
        "accuracy":"Good"
    }

}

info = model_info[selected_model]

st.info(
f"""
### {info['emoji']} {selected_model}

**Description**

{info['desc']}

**Inference Speed:** {info['speed']}

**Expected Accuracy:** {info['accuracy']}
"""
)

with col2:

    prediction_mode = st.radio(

        "Prediction Mode",

        [

            "🌱 Quick Prediction",

            "🔬 Advanced Prediction"

        ]

    )

st.divider()

# ==========================================================
# FEATURE LISTS
# ==========================================================

crop_features = [

    "harvest_year",

    "harvest_area",

    "crop_area",

    "crop_area_percentage",

    "sos",

    "eos"

]

soil_features = [

    "awc",

    "bulk_density",

    "drainage_class"

]

weather_features = [

    "avg_tmin",

    "avg_tmax",

    "avg_tavg",

    "avg_rad",

    "avg_et0",

    "avg_vpd",

    "avg_cwb"

]

vegetation_features = [

    "avg_ssm",

    "avg_rsm",

    "avg_ndvi",

    "avg_fpar"

]

location_features = [

    "latitude",

    "longitude",

    "region_area"

]

# ==========================================================
# USER INPUT
# ==========================================================

user_input = {}

def add_inputs(feature_list):

    for feature in feature_list:

        user_input[feature] = st.number_input(

            feature,

            value=float(df[feature].median())

        )

if prediction_mode == "🌱 Quick Prediction":

    st.subheader("🌱 Quick Prediction")

    quick_features = [

        "harvest_year",

        "harvest_area",

        "awc",

        "drainage_class",

        "avg_tavg",

        "avg_ssm",

        "avg_ndvi",

        "avg_fpar"

    ]

    add_inputs(quick_features)

else:

    st.subheader("🔬 Advanced Prediction")

    with st.expander("🌾 Crop Information", expanded=True):

        add_inputs(crop_features)

    with st.expander("🧱 Soil Information"):

        add_inputs(soil_features)

    with st.expander("🌦 Weather Information"):

        add_inputs(weather_features)

    with st.expander("🌿 Vegetation Information"):

        add_inputs(vegetation_features)

    with st.expander("📍 Region Information"):

        add_inputs(location_features)



# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

feature_order = [

    "harvest_year",
    "harvest_area",
    "sos",
    "eos",
    "awc",
    "bulk_density",
    "drainage_class",
    "latitude",
    "longitude",
    "region_area",
    "crop_area",
    "crop_area_percentage",
    "avg_tmin",
    "avg_tmax",
    "avg_tavg",
    "avg_rad",
    "avg_et0",
    "avg_vpd",
    "avg_cwb",
    "avg_ssm",
    "avg_rsm",
    "avg_ndvi",
    "avg_fpar"

]

# Fill remaining features with dataset median
for feature in feature_order:

    if feature not in user_input:

        user_input[feature] = float(df[feature].median())

input_df = pd.DataFrame(

    [user_input],

    columns=feature_order

)

def create_prediction_report(
    model_name,
    prediction,
    category,
    comparison_df
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading = styles["Heading2"]

    normal = styles["BodyText"]

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "AI Crop Yield Prediction Report",
            title_style
        )
    )

    story.append(Spacer(1,0.30*inch))

    # --------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------

    summary = [

        ["Selected Model", model_name],

        ["Predicted Yield", f"{prediction:.3f}"],

        ["Yield Category", category]

    ]

    table = Table(summary,colWidths=[170,250])

    table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.darkgreen),

            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

            ("GRID",(0,0),(-1,-1),1,colors.grey),

            ("BACKGROUND",(0,1),(0,-1),colors.beige),

            ("BACKGROUND",(1,1),(1,-1),colors.whitesmoke),

            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

            ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),

            ("BOTTOMPADDING",(0,0),(-1,0),10),

            ("TOPPADDING",(0,0),(-1,-1),8),

            ("BOTTOMPADDING",(0,0),(-1,-1),8)

        ])

    )

    story.append(table)

    story.append(Spacer(1,0.35*inch))

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Model Comparison",
            heading
        )
    )

    comparison = [["Model","Predicted Yield"]]

    for _,row in comparison_df.iterrows():

        comparison.append(

            [

                row["Model"],

                str(row["Predicted Yield"])

            ]

        )

    comp_table = Table(comparison,colWidths=[220,180])

    comp_table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.darkblue),

            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

            ("GRID",(0,0),(-1,-1),1,colors.grey),

            ("BACKGROUND",(0,1),(-1,-1),colors.whitesmoke),

            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

            ("BOTTOMPADDING",(0,0),(-1,0),10)

        ])

    )

    story.append(comp_table)

    story.append(Spacer(1,0.35*inch))

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Prediction Interpretation",
            heading
        )
    )

    story.append(

        Paragraph(

        f"""
        The selected model <b>{model_name}</b> predicts a crop yield of
        <b>{prediction:.3f}</b>. This prediction falls into the
        <b>{category}</b> category.

        The prediction is generated using crop information,
        soil properties, weather variables,
        vegetation indices and regional characteristics.

        XGBoost demonstrated the best overall performance during model
        evaluation and is recommended for production deployment.
        """,

        normal

        )

    )

    doc.build(story)

    buffer.seek(0)

    return buffer

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict = st.button(
    "🌾 Predict Yield",
    use_container_width=True
)

if not predict:
    st.stop()

st.success("Prediction Completed Successfully!")

# ==========================================================
# SIDEBAR DATASET SUMMARY
# ==========================================================

st.sidebar.title("📊 Dataset Summary")

st.sidebar.metric(
    "Samples",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Features",
    23
)


st.sidebar.metric(
    "Crop",
    df["crop_name"].iloc[0]
)

st.sidebar.metric(
    "Country",
    df["country_code"].iloc[0]
)

st.sidebar.metric(
    "Target",
    "Yield"
)

st.sidebar.divider()

st.sidebar.markdown(
"""
### Dataset

**Source:** CY-Bench

**Crop:** Wheat

**Region:** India

**Task:** Crop Yield Prediction
"""
)

# ==========================================================
# SELECTED MODEL PREDICTION
# ==========================================================

model = models[selected_model]

if selected_model in [

    "Random Forest",

    "XGBoost",

    "LightGBM"

]:

    prediction = float(

        model.predict(input_df)[0]

    )

elif selected_model == "MLP":

    X = scalers["MLP"].transform(input_df)

    prediction = float(

        model.predict(X)[0]

    )



# ==========================================================
# ALL MODEL COMPARISON
# ==========================================================

all_predictions = {}

# Random Forest
all_predictions["Random Forest"] = float(

    models["Random Forest"].predict(input_df)[0]

)

# XGBoost
all_predictions["XGBoost"] = float(

    models["XGBoost"].predict(input_df)[0]

)

# LightGBM
all_predictions["LightGBM"] = float(

    models["LightGBM"].predict(input_df)[0]

)

# MLP
X_mlp = scalers["MLP"].transform(input_df)

all_predictions["MLP"] = float(

    models["MLP"].predict(X_mlp)[0]

)



# ==========================================================
# DASHBOARD METRICS
# ==========================================================

dataset_mean = df["yield"].mean()
dataset_min = df["yield"].min()
dataset_max = df["yield"].max()

difference = prediction - dataset_mean
percent = (difference / dataset_mean) * 100

if prediction < dataset_mean * 0.85:
    category = "🔴 Low Yield"

elif prediction < dataset_mean * 1.15:
    category = "🟡 Medium Yield"

else:
    category = "🟢 High Yield"

st.divider()

st.subheader("📊 Prediction Dashboard")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🌾 Predicted Yield",
    f"{prediction:.3f}"
)

c2.metric(
    "📈 Dataset Average",
    f"{dataset_mean:.3f}"
)

c3.metric(
    "📊 Difference",
    f"{percent:.1f}%"
)

c4.metric(
    "🏷 Category",
    category
)

# ==========================================================
# YIELD POSITION
# ==========================================================

st.subheader("🌾 Yield Position")

progress = (prediction - dataset_min) / (dataset_max - dataset_min)

progress = max(0, min(progress, 1))

st.progress(progress)

st.caption(
    f"Historical Yield Range : {dataset_min:.2f} → {dataset_max:.2f}"
)

# ==========================================================
# PREDICTION SUMMARY
# ==========================================================

st.divider()

st.subheader("📝 Prediction Summary")

st.write(f"**Selected Model :** {selected_model}")

if prediction > dataset_mean:

    st.success(
        "Predicted yield is ABOVE the historical average."
    )

else:

    st.warning(
        "Predicted yield is BELOW the historical average."
    )

st.info("""
Prediction is based on:

• Crop characteristics

• Soil properties

• Weather variables

• Vegetation indices

• Regional information
""")

# ==========================================================
# PREDICTION CONFIDENCE
# ==========================================================

st.divider()

st.subheader("🎯 Prediction Confidence")

yield_std = df["yield"].std()

distance = abs(prediction - dataset_mean)

if distance <= 0.5 * yield_std:

    confidence = "🟢 High"

    score = 0.90

    explanation = (
        "Prediction is close to the historical distribution "
        "and is expected to be reliable."
    )

elif distance <= 1.5 * yield_std:

    confidence = "🟡 Medium"

    score = 0.70

    explanation = (
        "Prediction lies moderately away from the historical average."
    )

else:

    confidence = "🔴 Low"

    score = 0.45

    explanation = (
        "Prediction is relatively uncommon in the historical dataset."
    )

st.metric(
    "Confidence",
    confidence
)

st.progress(score)

st.caption(explanation)

# ==========================================================
# AI RECOMMENDATION
# ==========================================================

st.subheader("🤖 AI Recommendation")

if category == "🟢 High Yield":

    st.success("""
Current environmental conditions indicate excellent production potential.

Recommended Actions

• Maintain irrigation schedule

• Continue fertilizer management

• Regular pest monitoring

• Harvest at optimum maturity
""")

elif category == "🟡 Medium Yield":

    st.warning("""
Yield is close to the historical average.

Recommended Actions

• Monitor rainfall

• Monitor soil moisture

• Optimize fertilizer application

• Watch vegetation health
""")

else:

    st.error("""
Yield is lower than expected.

Recommended Actions

• Improve irrigation

• Review fertilizer management

• Check soil health

• Inspect crop stress and diseases
""")


# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.divider()

st.subheader("📊 Compare All Models")

comparison_df = pd.DataFrame({

    "Model": list(all_predictions.keys()),

    "Predicted Yield": [

        round(v, 3)

        for v in all_predictions.values()

    ]

})

comparison_df["Selected"] = comparison_df["Model"].apply(

    lambda x: "✅" if x == selected_model else ""

)

st.dataframe(

    comparison_df,

    use_container_width=True,

    hide_index=True

)

# ==========================================================
# BAR CHART
# ==========================================================

st.subheader("📈 Predicted Yield by Model")

chart_df = comparison_df.set_index("Model")

st.bar_chart(

    chart_df["Predicted Yield"]

)

# ==========================================================
# BEST PERFORMING MODEL
# ==========================================================

st.subheader("🏆 Recommended Production Model")

st.success("""
### ⭐ XGBoost

During model evaluation on the test dataset, **XGBoost achieved the strongest balance of**

• Highest R²

• Lowest RMSE

• Lowest MAE

Therefore it is recommended as the primary production model for crop yield prediction.
""")


st.divider()

pdf = create_prediction_report(

    selected_model,

    prediction,

    category,

    comparison_df

)

st.download_button(

    label="📄 Download Prediction Report",

    data=pdf,

    file_name="Crop_Yield_Prediction_Report.pdf",

    mime="application/pdf"

)