# 🌲 Universal Indian Forest Phenology Predictor — v5

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/Design-100%25_Data--Driven-brightgreen" />
  <img src="https://img.shields.io/badge/Forest%20Types-Universal-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A **fully data-driven Streamlit application** for extracting and predicting phenological events —
**Start of Season (SOS)**, **Peak of Season (POS)**, and **End of Season (EOS)** — from any
uploaded NDVI time series and NASA POWER meteorological data.

**v5 — Zero Hardcoding:** Every threshold, cadence, amplitude limit, feature ranking, and model
default is derived exclusively from the uploaded data. No forest-type presets, no lookup tables,
no hardcoded DOY ranges. Works for any Indian forest type or any ecosystem worldwide.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/universal_Indian_forest_phenology_v5.py
```

---

## Core Design: 100% Data-Governed

| Parameter | v4 (hardcoded) | v5 (data-driven) |
|-----------|---------------|------------------|
| NDVI cadence | assumed 16d | median of observed date diffs |
| Max gap threshold | fixed 60d | 8× detected cadence |
| Trough min distance | fixed 145d | 40% of autocorrelation cycle estimate |
| MIN_AMPLITUDE | fixed 0.02 | 5% of data P5–P95 range |
| Feature priority | hard list per event | pure Pearson/Spearman from data |
| Feature window | fixed 15d | user-configurable slider |
| Prediction defaults | zero | training data means |
| Data characterization | absent | auto-generated from uploads |
| Coefficient export | not available | CSV download |

---

## Bug Fixes (v5)

- **Fix 1:** Season year = trough start year (not POS year) — eliminates duplicate-year collision
- **Fix 2:** POS = raw NDVI peak (not smoothed) — exact match with observed maximum
- **Fix 3 (v4):** Gap-tolerant cycle extraction (50% tolerance for high-amplitude seasons)
- **Fix 4 (v3):** SG window capped at 31 steps — prevents over-smoothing
- **Fix 5 (v2):** Plateau trough filter 85% ceiling — fixes evergreen forests

---

## Features

- **📊 Data Overview** — auto-characterizes your NDVI and met data (cadence, range, evergreen index)
- **🔬 Training** — extracts phenology, fits models, shows equations and feature tables
- **📈 Correlations** — data-ranked feature heatmaps and year-by-year met plots
- **🔮 Predict** — input pre-filled from training data; LOO-validated predictions
- **📖 Guide** — full methodology documentation

---

## Citation

```
Sharma, S. (2025). Universal Indian Forest Phenology Predictor v5 [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```

---

## License
MIT License
