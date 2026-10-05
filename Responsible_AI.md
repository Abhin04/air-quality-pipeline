# Responsible AI Report

## 1. Purpose

This project predicts next-day AQI for Mumbai using historical
air-quality observations and a Random Forest regression model.

## 2. Data

The model uses historical AQI and pollutant measurements together
with temporal and lag-based features.

## 3. Privacy

The project does not use personally identifiable information,
individual-level health records, or user accounts.

## 4. Fairness

No demographic attributes are present in the dataset.

Weekend/weekday performance was examined as an exploratory
temporal grouping, not as a protected demographic attribute.

## 5. Model Explainability

SHAP was used to examine global feature importance.

LIME was used to demonstrate a local prediction explanation.

Feature importance should not be interpreted as proof of causality.

## 6. Data Drift

A Population Stability Index (PSI) comparison was implemented
between earlier and later portions of the dataset.

The drift analysis is exploratory and should not be interpreted
as proof that the model remains valid under all future conditions.

## 7. Limitations

- The model is trained on historical Mumbai observations.
- External live data may differ in measurement, aggregation,
  or units.
- Predictions should not be interpreted as health advice.
- Model performance may change when environmental conditions
  differ from the training period.

## 8. Reproducibility

The project includes the preprocessing workflow, model,
FastAPI service, Docker configuration, tests, CI workflow,
and Streamlit dashboard.