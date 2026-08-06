# ==========================================================
# Streamlit APP: Online Calculator (Run: streamlit run app.py)
# ==========================================================


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib


MODEL_PATH = "XGBoost.pkl"

DATA_PATH = "HCC.csv"



@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )



@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )



model = load_model()

data = load_data()



# ==========================================================
# Feature names:
# Radiomics (9) + Clinical (5)
# ==========================================================


feature_names = [

    # radiomics (9)

    "original_shape_Elongation_A",

    "log.sigma.1.0.mm.3D_glcm_Correlation_A",

    "lbp.3D.k_gldm_DependenceEntropy_A",

    "original_glcm_Autocorrelation",

    "original_gldm_LargeDependenceHighGrayLevelEmphasis",

    "log.sigma.1.0.mm.3D_firstorder_Median",

    "log.sigma.1.0.mm.3D_glrlm_ShortRunHighGrayLevelEmphasis",

    "wavelet.HHL_ngtdm_Complexity",

    "exponential_glcm_InverseVariance",


    # clinical (5)

    "Alcohol",

    "AFP",

    "AST_ALT",

    "Ascites",

    "ECOG_PS",

]



# Check columns

missing_cols = [

    c for c in feature_names

    if c not in data.columns

]


if missing_cols:

    st.error(
        f"HCC.csv is missing columns: {missing_cols}"
    )

    st.stop()



X_all = data[feature_names]


stats = X_all.agg(

    [
        "min",
        "max",
        "median"
    ]

)



# ==========================================================
# Helper function:
# identify binary variables
# ==========================================================


def is_binary_01(s: pd.Series) -> bool:

    vals = sorted(

        pd.Series(
            s.dropna().unique()
        ).tolist()

    )

    return (

        len(vals) == 2

        and vals == [0,1]

    )



# ==========================================================
# SHAP TreeExplainer
# ==========================================================


@st.cache_resource
def build_shap_explainer_tree(_model):

    return shap.TreeExplainer(
        _model
    )



explainer = build_shap_explainer_tree(
    model
)



# ==========================================================
# Streamlit interface
# ==========================================================


st.title(
    "HCC Treatment Response Prediction Calculator\n"
    "(Radiomics + Clinical Model)"
)



st.markdown(
"""
This online calculator applies a machine learning model to predict 
**treatment response (responder vs non-responder)** in patients with 
hepatocellular carcinoma (HCC).

After entering radiomics features (Z-score standardized values) and clinical 
variables (Alcohol history, AFP, AST/ALT ratio, Ascites, and ECOG-PS), the model 
provides the predicted probability of **objective response** and generates an 
individual-level SHAP explanation plot.

> ⚠️ This tool is intended for research and educational purposes only and 
should not replace clinical decision-making.
"""
)



st.subheader(
    "1. Input Variables"
)



st.markdown(
    "**(1) Radiomics Features (Z-score standardized values)**"
)



input_values = {}



# ==========================================================
# Radiomics input
# ==========================================================


radiomics_cols = [

    "original_shape_Elongation_A",

    "log.sigma.1.0.mm.3D_glcm_Correlation_A",

    "lbp.3D.k_gldm_DependenceEntropy_A",

    "original_glcm_Autocorrelation",

    "original_gldm_LargeDependenceHighGrayLevelEmphasis",

    "log.sigma.1.0.mm.3D_firstorder_Median",

    "log.sigma.1.0.mm.3D_glrlm_ShortRunHighGrayLevelEmphasis",

    "wavelet.HHL_ngtdm_Complexity",

    "exponential_glcm_InverseVariance",

]



for col in radiomics_cols:


    input_values[col] = st.number_input(

        label=f"{col} (Z-score standardized feature)",

        min_value=float(
            stats.loc["min", col]
        ),

        max_value=float(
            stats.loc["max", col]
        ),

        value=float(
            stats.loc["median", col]
        ),

        step=0.01,

        format="%.4f",

    )



st.markdown("---")



st.markdown(
    "**(2) Clinical Variables**"
)



# ==========================================================
# Alcohol
# ==========================================================


if is_binary_01(data["Alcohol"]):


    input_values["Alcohol"] = st.selectbox(

        "Alcohol history (0=No / 1=Yes)",

        options=[0,1],

        format_func=lambda x:

        "0 = No alcohol history"

        if x == 0

        else

        "1 = Alcohol history"

    )


else:


    input_values["Alcohol"] = st.number_input(

        "Alcohol (numeric encoding)",

        min_value=float(
            stats.loc["min","Alcohol"]
        ),

        max_value=float(
            stats.loc["max","Alcohol"]
        ),

        value=float(
            stats.loc["median","Alcohol"]
        ),

        step=1.0,

    )



# ==========================================================
# AFP
# ==========================================================


input_values["AFP"] = st.number_input(

    "AFP (ng/mL)",

    min_value=float(
        stats.loc["min","AFP"]
    ),

    max_value=float(
        stats.loc["max","AFP"]
    ),

    value=float(
        stats.loc["median","AFP"]
    ),

    step=1.0,

)



# ==========================================================
# AST_ALT
# ==========================================================


if is_binary_01(data["AST_ALT"]):


    input_values["AST_ALT"] = st.selectbox(

        "AST_ALT (0=Low / 1=High)",

        options=[0,1],

    )


else:


    input_values["AST_ALT"] = st.number_input(

        "AST/ALT ratio",

        min_value=float(
            stats.loc["min","AST_ALT"]
        ),

        max_value=float(
            stats.loc["max","AST_ALT"]
        ),

        value=float(
            stats.loc["median","AST_ALT"]
        ),

        step=0.01,

        format="%.4f",

    )

# ==========================================================
# Ascites
# ==========================================================


if is_binary_01(data["Ascites"]):


    input_values["Ascites"] = st.selectbox(

        "Ascites (0=Absent / 1=Present)",

        options=[0,1],

    )


else:


    asc_opts = sorted(

        pd.Series(
            data["Ascites"].dropna().unique()
        ).tolist()

    )


    input_values["Ascites"] = st.selectbox(

        "Ascites (classification/encoding)",

        options=asc_opts,

    )



# ==========================================================
# ECOG Performance Status
# ==========================================================


ecog_opts = sorted(

    pd.Series(
        data["ECOG_PS"].dropna().unique()
    ).tolist()

)


input_values["ECOG_PS"] = st.selectbox(

    "ECOG Performance Status (ECOG-PS)",

    options=ecog_opts

)



# ==========================================================
# Construct model input
# ==========================================================


feature_values = [

    input_values[col]

    for col in feature_names

]


features_df = pd.DataFrame(

    [feature_values],

    columns=feature_names

)



st.markdown("---")



st.subheader(

    "2. Model Prediction"

)



# ==========================================================
# Prediction
# ==========================================================


if st.button("Run Prediction"):


    predicted_class = int(

        model.predict(features_df)[0]

    )


    predicted_proba = model.predict_proba(

        features_df

    )[0]



    prob_non_response = float(

        predicted_proba[0]

    )


    prob_response = float(

        predicted_proba[1]

    )



    st.write(

        f"**Predicted class (group): {predicted_class}**"

    )


    st.write(

        "- group=0: Non-responder "

        "(e.g., stable disease or progressive disease)"

    )


    st.write(

        "- group=1: Responder "

        "(e.g., complete response or partial response)"

    )



    st.write(

        "**Predicted probability**"

    )


    st.write(

        f"- Probability of non-response: "

        f"{prob_non_response*100:.1f}%"

    )


    st.write(

        f"- Probability of objective response: "

        f"{prob_response*100:.1f}%"

    )



    if predicted_class == 1:


        st.success(

            f"The model predicts a relatively high probability "

            f"of objective response (CR/PR).\n\n"

            f"Probability of response ≈ "

            f"{prob_response*100:.1f}%.\n\n"

            "For research purposes only. Clinical decisions "

            "should incorporate multidisciplinary evaluation."

        )


    else:


        st.warning(

            f"The model predicts a relatively low probability "

            f"of objective response (CR/PR).\n\n"

            f"Probability of response ≈ "

            f"{prob_response*100:.1f}%.\n\n"

            "For research purposes only. Clinical interpretation "

            "should incorporate follow-up information and clinical judgment."

        )



    st.markdown("---")



    st.subheader(

        "3. Model Interpretation (SHAP Force Plot)"

    )



    # ======================================================
    # SHAP calculation
    # ======================================================


    with st.spinner(

        "Calculating SHAP values and generating plot..."

    ):


        shap_values = explainer.shap_values(

            features_df

        )



        # Compatible with different SHAP versions

        if isinstance(shap_values, list):


            shap_values_sample = shap_values[1][0]


            expected_value = explainer.expected_value[1]


        else:


            shap_values_sample = shap_values[0]


            expected_value = explainer.expected_value



        plt.figure(

            figsize=(10,3)

        )


        shap.force_plot(

            expected_value,

            shap_values_sample,

            features_df,

            matplotlib=True,

            show=False,

        )



        plt.tight_layout()



        plt.savefig(

            "shap_force_plot_hcc.png",

            bbox_inches="tight",

            dpi=300

        )


        plt.close()



    st.image(

        "shap_force_plot_hcc.png",

        caption=(

            "Individual SHAP force plot "

            "(feature contributions to response probability)"

        )

    )
