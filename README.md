# 🚲 BikePulse AI

An end-to-end Python AI/ML portfolio project that predicts hourly bike rental demand from real online data. Includes reproducible data wrangling, classical machine learning, a PyTorch deep neural network, anomaly detection, uncertainty intervals, and an interactive Streamlit dashboard.

## Quick start

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m bikepulse.train --epochs 80
streamlit run app.py
```

The first training run downloads the dataset from UCI. Internet is required for installation and the first download. Later runs reuse `data/hour.csv`. Training writes experiment artifacts locally. The dashboard reads these artifacts and does not retrain on each interaction.

## What is included

| Area | Implementation | Libraries |
|---|---|---|
| Online data | HTTPS download, cached CSV, SHA-256 provenance | requests, pathlib, zipfile |
| Data wrangling | Type coercion, invalid-row removal, deduplication, cyclic features | pandas, NumPy |
| Machine learning | Mean baseline, Ridge, random forest | scikit-learn |
| Deep learning | 16 → 64 → 32 → 1 multilayer neural network, Adam, validation checkpointing | PyTorch |
| AI features | Isolation forest anomalies, calibrated prediction intervals, evidence-grounded query assistant | scikit-learn, NumPy |
| Dashboard | Demand charts, model comparisons, feature importance, downloads, questions | Streamlit, Plotly |
| Quality | Schema and leakage tests, GitHub Actions | pytest |

The local question assistant routes supported questions to measured results. It is an intent-based analytics feature, **not an LLM**. No API key or paid service is needed. This project uses relevant libraries rather than attempting to install every Python library.

## Reproducible methodology

Hourly observations are ordered by timestamp and divided into 60% training, 15% validation, 10% interval calibration, and 15% final testing. Imputation and normalization fit only on training. The validation set chooses the model and neural checkpoint; the test set is reserved for reporting. `casual` and `registered` are deliberately excluded because their sum equals the target `cnt`. Feature importance describes association, not causation.

The selected model's absolute calibration residuals produce nominal 90% prediction intervals. Temporal dependence and distribution shifts violate the exchangeability assumption behind formal coverage guarantees. The report therefore includes measured test coverage rather than claiming guaranteed coverage. Isolation forest flags unusual weather/calendar combinations, not proven errors.

This is historical supervised prediction for 2011–2012, not a live rental service or a future time-series deployment. Weather inputs must be known or forecast at prediction time. A production system would require fresh data, drift monitoring, rolling evaluation, model persistence, and operational validation.

## Outputs

- `data/provenance.json`: source, dataset license, and downloaded CSV checksum.
- `artifacts/metrics.json`: split dates, validation/test metrics, selected model, coverage.
- `artifacts/predictions.csv`: held-out predictions, intervals, anomalies.
- `artifacts/importance.csv`: random forest feature importances.
- `artifacts/training_history.csv`: neural validation learning curve.
- `reports/`: checked-in results from the verified example run.

## Tests

```bash
python -m pytest -q
```

## Dataset credit

Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894. Dataset license: **CC BY 4.0**. Source: https://archive.ics.uci.edu/dataset/275. Uses hourly rental counts and weather/calendar variables. Source files are downloaded at runtime and are not bundled in this repository.

Project code is released under the MIT license; the dataset retains its separate attribution license.

## Verified example run

Trained for 80 epochs on 17,379 online hourly records. The validation-selected neural network achieved test MAE **64.49 rentals/hour**, RMSE **84.12**, and R² **0.847**. Observed test interval coverage was **90.7%**. See `reports/metrics.json` for all model comparisons and exact split dates. Results may vary slightly by hardware. To reproduce the tested dependency versions, install `requirements-lock.txt` instead of `requirements.txt`.
