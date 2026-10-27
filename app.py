
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

HF_REPO_ID = "godgoku0007/crop-yield-model"
HF_MODEL_FILENAME = "crop_yield_model.joblib"


@st.cache_resource(show_spinner="Loading crop yield model...")
def load_artifact():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        repo_type="model"
    )
    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Unexpected model file format.")

    if "feature_columns" not in artifact:
        raise ValueError("Model is missing feature_columns.")

    return artifact


st.title("🌾 Crop Yield Prediction System")
st.write("Enter crop and farm details to predict crop yield.")
st.caption("University Machine Learning Project")

try:
    artifact = load_artifact()
except Exception as exc:
    st.error("Could not load the model from Hugging Face.")
    st.code(str(exc))
    st.info(
        "Check that your Hugging Face repository is public "
        "and contains crop_yield_model.joblib."
    )
    st.stop()

model = artifact["model"]
features = list(artifact["feature_columns"])
numeric_features = set(artifact.get("numeric_features", []))
target_name = artifact.get("target_column", "Crop Yield")
model_name = artifact.get("model_name", "Trained model")

st.success(f"Model loaded: {model_name}")
st.subheader("Enter farm details")

with st.form("crop_yield_form"):
    values = {}
    left, right = st.columns(2)

    for i, feature in enumerate(features):
        col = left if i % 2 == 0 else right
        label = str(feature).replace("_", " ").title()

        with col:
            if feature in numeric_features:
                values[feature] = st.number_input(
                    label,
                    value=0.0,
                    step=1.0,
                    key=f"num_{feature}"
                )
            else:
                values[feature] = st.text_input(
                    label,
                    placeholder=f"Enter {label}",
                    key=f"cat_{feature}"
                ).strip()

    submitted = st.form_submit_button(
        "Predict Crop Yield",
        type="primary",
        use_container_width=True
    )

if submitted:
    missing = [
        f for f in features
        if f not in numeric_features and not values.get(f, "").strip()
    ]

    if missing:
        st.warning("Please fill in: " + ", ".join(missing))
    else:
        try:
            input_df = pd.DataFrame([values], columns=features)

            for feature in numeric_features:
                input_df[feature] = pd.to_numeric(
                    input_df[feature], errors="raise"
                )

            numeric_values = input_df[list(numeric_features)].to_numpy(
                dtype=float
            )
            if numeric_values.size and not np.isfinite(numeric_values).all():
                raise ValueError("Enter valid finite numeric values.")

            prediction = float(model.predict(input_df)[0])

            if not math.isfinite(prediction):
                raise ValueError("Invalid prediction returned by the model.")

            st.divider()
            st.subheader("Prediction Result")
            st.metric(
                label=f"Estimated {target_name}",
                value=f"{prediction:,.3f}"
            )
            st.success("Prediction completed successfully.")

            with st.expander("Review entered details"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as exc:
            st.error("Prediction failed.")
            st.code(str(exc))

st.divider()
st.caption("Educational use only. Results are estimates.")
