from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_yield_model.joblib"

# -------------------- Load the exported Colab artifact --------------------
@st.cache_resource(show_spinner="Loading the trained crop-yield model...")
def load_artifact(path_string: str):
    path = Path(path_string)
    if not path.is_file():
        return None
    # Only load a model file you created yourself in your Colab notebook.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            "The file is not in the expected format. Download "
            "crop_yield_model.joblib again from the final Colab cell."
        )
    required = ["feature_columns", "numeric_features"]
    missing_keys = [key for key in required if key not in artifact]
    if missing_keys:
        raise ValueError(
            "The model export is missing: " + ", ".join(missing_keys)
            + ". Please export the model again from Colab."
        )
    return artifact

st.title("🌾 Crop Yield Prediction System")
st.write(
    "Enter farm and crop details to estimate yield using your trained "
    "machine-learning model."
)
st.caption("University project · Machine learning regression")

if not MODEL_PATH.exists():
    st.error("Model file not found: crop_yield_model.joblib")
    st.markdown(
        "Upload **`crop_yield_model.joblib`** into the same project folder "
        "as **`app.py`**, then restart the app."
    )
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Use the model file exported by your own Colab notebook. "
        "If this is a version mismatch, rerun the export in the same "
        "environment as the app's scikit-learn version."
    )
    st.stop()

if artifact is None:
    st.error("Model file is unavailable. Check the filename and folder.")
    st.stop()

model = artifact["model"]
features = list(artifact["feature_columns"])
numeric_features = set(artifact.get("numeric_features", []))
target_name = artifact.get("target_column", "Crop Yield")
model_name = artifact.get("model_name", "Trained model")

if not features:
    st.error("The model export contains no input feature names.")
    st.stop()

st.success(f"Model loaded successfully: {model_name}")
st.caption(f"Prediction target: {target_name}")

# -------------------- Input form --------------------
st.subheader("Farm details")
st.info(
    "Enter values in the same units and category format used by your "
    "training dataset. Predictions are estimates, not guarantees."
)

with st.form("crop_yield_form", clear_on_submit=False):
    input_values = {}
    left_col, right_col = st.columns(2)

    for index, feature in enumerate(features):
        column = left_col if index % 2 == 0 else right_col
        label = str(feature).replace("_", " ").strip().title()
        with column:
            if feature in numeric_features:
                input_values[feature] = st.number_input(
                    label,
                    value=0.0,
                    step=1.0,
                    format="%.4f",
                    key=f"numeric_{feature}",
                    help=f"Use the same units as the training dataset for {feature}.",
                )
            else:
                input_values[feature] = st.text_input(
                    label,
                    value="",
                    placeholder=f"Enter {label}",
                    key=f"categorical_{feature}",
                    help="Use a category/value represented in your training data.",
                ).strip()

    submitted = st.form_submit_button(
        "Predict Crop Yield", type="primary", use_container_width=True
    )

# -------------------- Prediction --------------------
if submitted:
    missing_categories = [
        str(feature)
        for feature in features
        if feature not in numeric_features
        and not str(input_values.get(feature, "")).strip()
    ]

    if missing_categories:
        st.warning("Please fill in: " + ", ".join(missing_categories))
    else:
        input_df = pd.DataFrame([input_values], columns=features)

        try:
            for feature in numeric_features:
                input_df[feature] = pd.to_numeric(
                    input_df[feature], errors="raise"
                )

            numeric_values = input_df[list(numeric_features)].to_numpy(dtype=float) if numeric_features else np.array([])
            if numeric_values.size and not np.isfinite(numeric_values).all():
                raise ValueError("Numeric inputs must be finite numbers.")

            with st.spinner("Calculating predicted yield..."):
                prediction = float(model.predict(input_df)[0])

            if not math.isfinite(prediction):
                raise ValueError("The model returned an invalid numeric prediction.")

            st.divider()
            st.subheader("Prediction result")
            st.metric(label=f"Estimated {target_name}", value=f"{prediction:,.3f}")
            st.success("Prediction completed.")
            st.caption(
                "The result is expressed in the same units as the target "
                "column in the training dataset. Check your dataset documentation "
                "to confirm the unit."
            )

            with st.expander("Review input details"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error("Prediction failed. Check the entered values and model file.")
            st.code(str(exc))
            st.write(
                "Use the same feature units and familiar category names as in "
                "the training dataset. If the error persists, share this error text."
            )

st.divider()
st.caption(
    "Educational use only. Prediction quality depends on the training dataset "
    "and how well new farm conditions match it."
)
