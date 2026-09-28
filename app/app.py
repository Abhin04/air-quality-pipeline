from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
except ImportError:
    shap = None


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "air_quality_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "final_aqi_random_forest.pkl"

FEATURE_COLUMNS = [
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_7",
    "aqi_rolling_mean_3",
    "aqi_rolling_mean_7",
    "aqi_rolling_std_7",
]

POLLUTANT_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mumbai AQI Prediction",
    page_icon="🌫️",
    layout="wide",
)


# ---------------------------------------------------------
# Data/model loading
# ---------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the project data or model: {exc}")
    st.stop()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Satisfactory"
    if aqi_value <= 200:
        return "Moderate"
    if aqi_value <= 300:
        return "Poor"
    if aqi_value <= 400:
        return "Very Poor"
    return "Severe"


def calculate_psi(reference, current, bins=10):
    """Calculate a simple Population Stability Index."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / max(reference_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    reference_pct = np.clip(reference_pct, 1e-6, None)
    current_pct = np.clip(current_pct, 1e-6, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def drift_label(psi):
    if pd.isna(psi):
        return "Unavailable"
    if psi < 0.10:
        return "Low"
    if psi < 0.25:
        return "Moderate"
    return "High"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🌫️ Mumbai Air Quality Prediction Dashboard")

st.markdown(
    """
    **Air Quality Prediction & Monitoring System**

    This dashboard presents historical AQI insights, next-day AQI
    predictions, model explainability using SHAP, and exploratory
    distribution-shift checks.
    """
)

st.divider()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Dashboard")

section = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "Historical Backtest",
        "Next-Day Prediction",
        "Model Explainability",
        "Drift Check",
    ],
)


# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

if section == "Overview":

    st.header("📊 AQI Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average AQI", f"{df['aqi'].mean():.2f}")
    col2.metric("Maximum AQI", f"{df['aqi'].max():.0f}")
    col3.metric("Minimum AQI", f"{df['aqi'].min():.0f}")
    col4.metric("Observations", f"{len(df):,}")

    st.subheader("AQI Category Distribution")

    category_counts = (
        df["aqi"]
        .apply(aqi_category)
        .value_counts()
        .reindex(
            ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
            fill_value=0,
        )
    )

    st.bar_chart(category_counts)

    st.subheader("Historical AQI")

    historical = df.set_index("date")[["aqi"]]
    st.line_chart(historical)

    st.subheader("Pollutant Trends")

    selected_pollutants = st.multiselect(
        "Select pollutants",
        POLLUTANT_COLUMNS,
        default=["pm25", "pm10", "no2"],
    )

    if selected_pollutants:
        pollutant_data = df.set_index("date")[selected_pollutants]
        st.line_chart(pollutant_data)

    st.subheader("Summary Statistics")

    summary_columns = ["aqi"] + POLLUTANT_COLUMNS
    st.dataframe(
        df[summary_columns].describe().round(2),
        use_container_width=True,
    )


# ---------------------------------------------------------
# Historical Backtest
# ---------------------------------------------------------

elif section == "Historical Backtest":

    st.header("🧪 Historical Next-Day Prediction Backtest")

    st.write(
        "Select a historical date. The model uses only the features "
        "available on that date to predict the following day's AQI, "
        "then compares the prediction with the actual next-day AQI."
    )

    st.info(
        "This is a reproducible historical test of the forecasting "
        "workflow. It does not use the selected day's target_aqi as "
        "an input feature."
    )

    # A row can only be tested when a following-day observation exists.
    valid_indices = df.index[df["target_aqi"].notna()].tolist()

    if not valid_indices:
        st.error("No rows with a next-day target are available for backtesting.")
        st.stop()

    valid_dates = df.loc[valid_indices, "date"]

    selected_date = st.date_input(
        "Select prediction date",
        value=valid_dates.iloc[-1].date(),
        min_value=valid_dates.iloc[0].date(),
        max_value=valid_dates.iloc[-1].date(),
    )

    selected_timestamp = pd.Timestamp(selected_date)

    matching_rows = df.index[df["date"] == selected_timestamp].tolist()

    if not matching_rows:
        st.warning(
            "No observation exists for the selected date. "
            "Please select a date present in the dataset."
        )
        st.stop()

    row_index = matching_rows[0]

    if row_index >= len(df) - 1 or pd.isna(df.loc[row_index, "target_aqi"]):
        st.warning(
            "A following-day observation is not available for this date. "
            "Please choose an earlier date."
        )
        st.stop()

    selected_row = df.loc[row_index]
    next_row = df.loc[row_index + 1]

    # Use exactly the same 19 features as the deployed model.
    input_data = selected_row[FEATURE_COLUMNS].to_frame().T
    prediction = float(model.predict(input_data[FEATURE_COLUMNS])[0])

    actual_aqi = float(selected_row["target_aqi"])
    absolute_error = abs(prediction - actual_aqi)

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        f"Predicted AQI for {next_row['date'].strftime('%d %b %Y')}",
        f"{prediction:.2f}",
    )

    col2.metric(
        "Actual Next-Day AQI",
        f"{actual_aqi:.0f}",
    )

    col3.metric(
        "Absolute Error",
        f"{absolute_error:.2f}",
    )

    st.write(
    f"**Prediction date:** {selected_timestamp.strftime('%d %b %Y')}  \n"
    f"**Predicted date:** {next_row['date'].strftime('%d %b %Y')}"
)

    st.subheader("Predicted vs Actual")

    comparison_df = pd.DataFrame(
        {
            "AQI": [prediction, actual_aqi],
        },
        index=["Predicted", "Actual"],
    )

    st.bar_chart(comparison_df)

    st.subheader("Input Information Used by the Model")

    display_columns = [
        "date",
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
        "aqi_lag_1",
        "aqi_lag_3",
        "aqi_lag_7",
        "aqi_rolling_mean_3",
        "aqi_rolling_mean_7",
        "aqi_rolling_std_7",
    ]

    st.dataframe(
        selected_row[display_columns].to_frame().T,
        use_container_width=True,
    )

    st.subheader("AQI Categories")

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Category",
        aqi_category(prediction),
    )

    col2.metric(
        "Actual Category",
        aqi_category(actual_aqi),
    )

    st.caption(
        "The actual next-day AQI is shown only for evaluation. "
        "The model prediction is generated from the selected day's "
        "feature values and does not use the next day's AQI."
    )


# ---------------------------------------------------------
# Next-Day Prediction
# ---------------------------------------------------------

elif section == "Next-Day Prediction":

    st.header("🔮 Next-Day AQI Prediction")

    st.write(
        "Enter the current AQI, pollutant concentrations, and historical "
        "AQI features used by the trained Random Forest model."
    )

    latest = df.iloc[-1]

    with st.form("prediction_form"):

        st.subheader("Current Air Quality")

        col1, col2, col3 = st.columns(3)

        aqi = col1.number_input(
            "AQI",
            value=float(latest["aqi"]),
            min_value=0.0,
        )

        pm25 = col2.number_input(
            "PM2.5",
            value=float(latest["pm25"]),
            min_value=0.0,
        )

        pm10 = col3.number_input(
            "PM10",
            value=float(latest["pm10"]),
            min_value=0.0,
        )

        col4, col5, col6 = st.columns(3)

        no2 = col4.number_input(
            "NO₂",
            value=float(latest["no2"]),
            min_value=0.0,
        )

        so2 = col5.number_input(
            "SO₂",
            value=float(latest["so2"]),
            min_value=0.0,
        )

        co = col6.number_input(
            "CO",
            value=float(latest["co"]),
            min_value=0.0,
        )

        o3 = st.number_input(
            "O₃",
            value=float(latest["o3"]),
            min_value=0.0,
        )

        st.subheader("Historical AQI Features")

        col1, col2, col3 = st.columns(3)

        aqi_lag_1 = col1.number_input(
            "AQI Lag 1",
            value=float(latest["aqi_lag_1"]),
        )

        aqi_lag_3 = col2.number_input(
            "AQI Lag 3",
            value=float(latest["aqi_lag_3"]),
        )

        aqi_lag_7 = col3.number_input(
            "AQI Lag 7",
            value=float(latest["aqi_lag_7"]),
        )

        col4, col5, col6 = st.columns(3)

        rolling_mean_3 = col4.number_input(
            "AQI Rolling Mean 3",
            value=float(latest["aqi_rolling_mean_3"]),
        )

        rolling_mean_7 = col5.number_input(
            "AQI Rolling Mean 7",
            value=float(latest["aqi_rolling_mean_7"]),
        )

        rolling_std_7 = col6.number_input(
            "AQI Rolling Std 7",
            value=float(latest["aqi_rolling_std_7"]),
        )

        st.subheader("Calendar Features")

        col1, col2, col3, col4 = st.columns(4)

        year = col1.number_input(
            "Year",
            value=int(latest["year"]),
            step=1,
        )

        month = col2.number_input(
            "Month",
            value=int(latest["month"]),
            min_value=1,
            max_value=12,
            step=1,
        )

        day = col3.number_input(
            "Day",
            value=int(latest["day"]),
            min_value=1,
            max_value=31,
            step=1,
        )

        day_of_week = col4.number_input(
            "Day of Week",
            value=int(latest["day_of_week"]),
            min_value=0,
            max_value=6,
            step=1,
        )

        col1, col2 = st.columns(2)

        day_of_year = col1.number_input(
            "Day of Year",
            value=int(latest["day_of_year"]),
            min_value=1,
            max_value=366,
            step=1,
        )

        is_weekend = col2.selectbox(
            "Is Weekend?",
            options=[0, 1],
            index=int(latest["is_weekend"]),
        )

        submitted = st.form_submit_button(
            "Predict Next-Day AQI",
            type="primary",
        )

    if submitted:

        input_data = pd.DataFrame(
            [
                {
                    "aqi": aqi,
                    "pm25": pm25,
                    "pm10": pm10,
                    "no2": no2,
                    "so2": so2,
                    "co": co,
                    "o3": o3,
                    "year": year,
                    "month": month,
                    "day": day,
                    "day_of_week": day_of_week,
                    "day_of_year": day_of_year,
                    "is_weekend": is_weekend,
                    "aqi_lag_1": aqi_lag_1,
                    "aqi_lag_3": aqi_lag_3,
                    "aqi_lag_7": aqi_lag_7,
                    "aqi_rolling_mean_3": rolling_mean_3,
                    "aqi_rolling_mean_7": rolling_mean_7,
                    "aqi_rolling_std_7": rolling_std_7,
                }
            ]
        )

        prediction = float(model.predict(input_data[FEATURE_COLUMNS])[0])

        st.success("Prediction generated successfully.")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Next-Day AQI",
            f"{prediction:.2f}",
        )

        col2.metric(
            "AQI Category",
            aqi_category(prediction),
        )

        st.info(
            "This is a machine-learning estimate and should not be "
            "treated as an official regulatory AQI measurement."
        )


# ---------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------

elif section == "Model Explainability":

    st.header("🔍 Model Explainability")

    st.write(
        "SHAP is used to examine which features contribute most strongly "
        "to the Random Forest model's predictions."
    )

    if shap is None:
        st.warning(
            "SHAP is not installed in the current environment. "
            "Install it with: pip install shap"
        )
        st.stop()

    explain_df = df[FEATURE_COLUMNS].copy()

    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(explain_df)

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    shap_importance = (
        pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Mean Absolute SHAP": mean_abs_shap,
            }
        )
        .sort_values("Mean Absolute SHAP", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("Global Feature Importance")

    st.bar_chart(
        shap_importance.set_index("Feature").head(10)
    )

    st.dataframe(
        shap_importance.round(4),
        use_container_width=True,
    )

    st.subheader("Interpretation")

    top_feature = shap_importance.iloc[0]

    st.info(
        f"{top_feature['Feature']} has the largest mean absolute SHAP "
        f"value ({top_feature['Mean Absolute SHAP']:.4f}) in this analysis."
    )

    st.caption(
        "SHAP importance describes model behavior and should not be "
        "interpreted as proof of causation."
    )


# ---------------------------------------------------------
# Drift Check
# ---------------------------------------------------------

elif section == "Drift Check":

    st.header("📉 Exploratory Data Drift Check")

    st.write(
        "This section compares the earlier portion of the available "
        "dataset with the later portion using the Population Stability "
        "Index (PSI)."
    )

    st.warning(
        "This is an exploratory distribution-shift check, not a formal "
        "production monitoring system or a claim that the model has "
        "become unreliable."
    )

    split_index = int(len(df) * 0.8)

    reference_df = df.iloc[:split_index]
    current_df = df.iloc[split_index:]

    st.write(
        f"Reference observations: {len(reference_df):,}  |  "
        f"Recent observations: {len(current_df):,}"
    )

    drift_features = ["aqi"] + POLLUTANT_COLUMNS

    drift_results = []

    for feature in drift_features:
        psi_value = calculate_psi(
            reference_df[feature],
            current_df[feature],
        )

        drift_results.append(
            {
                "Feature": feature,
                "PSI": psi_value,
                "Drift Level": drift_label(psi_value),
            }
        )

    drift_df = pd.DataFrame(drift_results)

    st.dataframe(
        drift_df.round({"PSI": 4}),
        use_container_width=True,
    )

    st.subheader("PSI Visualization")

    st.bar_chart(
        drift_df.set_index("Feature")["PSI"]
    )

    st.caption(
        "PSI below 0.10 is treated here as low distribution shift, "
        "0.10–0.25 as moderate, and above 0.25 as high. "
        "These thresholds are used as practical screening indicators."
    )