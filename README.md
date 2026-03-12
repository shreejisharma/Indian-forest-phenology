# 🌲 Universal Indian Forest Phenology Predictor — v5

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Design-100%25_Data--Driven-brightgreen" />
  <img src="https://img.shields.io/badge/Forest%20Types-Universal-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A **fully data-driven Streamlit application** for extracting and predicting phenological events — **Start of Season (SOS)**, **Peak of Season (POS)**, and **End of Season (EOS)** — across all Indian forest types. Upload your NDVI time series and NASA POWER meteorological data; every threshold, feature, and model coefficient is derived from your data — no hardcoded presets.

**[▶ Open Live App](https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/)**

---

## Overview

| Item | Detail |
|---|---|
| **Works with** | All Indian forest types — Tropical Dry/Moist Deciduous, Wet Evergreen, Shola, Thorn, Mangrove, NE India, Himalayan, Alpine — **no forest-type selection needed** |
| **Phenological events** | SOS · POS · EOS · LOS (Length of Season) |
| **NDVI input** | Any CSV with `Date` + `NDVI` columns (MODIS MOD13Q1, Sentinel-2, or other) |
| **Meteorological input** | NASA POWER daily export (headers auto-detected and skipped) |
| **Models** | Ridge · LOESS · Polynomial (deg 2/3) · Gaussian Process — all LOO cross-validated |
| **Feature selection** | Pearson \|r\| ≥ 0.40 filter → collinearity removal → incremental LOO R² check |
| **Minimum data** | 3 growing seasons (≥ 5 recommended for reliable R²) |

---

## What's New in v5 — 100% Data-Driven

| Parameter | v4 (hardcoded) | v5 (data-driven) |
|-----------|---------------|------------------|
| NDVI cadence | assumed 16d | median of observed date differences |
| Max gap threshold | fixed 60d | 8× detected cadence |
| Trough min distance | fixed 145d | 40% of autocorrelation cycle estimate |
| MIN_AMPLITUDE | fixed 0.02 | 5% of data P5–P95 range |
| Feature priority | hard list per event | pure Pearson/Spearman ranking from data |
| Feature window | fixed 15d | user-adjustable sidebar slider |
| Prediction defaults | zero | training data means |
| Data characterization | absent | auto-generated from uploads |
| Model coefficient export | not available | CSV download |

---

## Bug Fixes (v5)

- **Fix 1:** Season year = trough start year (not POS year) — eliminates duplicate-year collision
- **Fix 2:** POS = raw NDVI peak (not smoothed) — exact match with observed maximum
- **Fix 3 (v4):** Gap-tolerant cycle extraction (50% tolerance for high-amplitude seasons)
- **Fix 4 (v3):** SG window capped at 31 steps — prevents over-smoothing across seasons
- **Fix 5 (v2):** Plateau trough filter 85% ceiling — fixes missing seasons in evergreen forests

---

## Features

### 📊 Data Overview Tab *(new in v5)*
- Auto-characterizes uploaded NDVI: cadence, dynamic range, evergreen index
- Lists all detected met parameters with their statistics from your file
- Summary plot — all values derived from your uploaded data

### 🔬 Training Tab
- Fully data-adaptive phenology extraction (cadence, amplitude, cycle length from data)
- Model performance cards: LOO R², MAE, selected features
- Fitted equations with all coefficients shown
- Feature role table — IN MODEL · Not selected · Below threshold
- Observed vs Predicted scatter plots
- **Download phenology table (CSV)**
- **Download model coefficients (CSV)**

### 📈 Correlations Tab
- Feature correlation bar chart + heatmap (data-ranked, no preset priority)
- Year-by-year NDVI + Meteorology plots

### 🔮 Predict Tab
- Inputs pre-filled with training data means (data-derived defaults)
- Value range hints from training data shown in tooltips
- LOO-validated predictions with ecological order enforcement (SOS < POS < EOS)
- Download predictions as CSV

### 📖 Technical Guide
- Full methodology documentation
- Data-driven vs hardcoded parameter comparison table
- R² interpretation guide for research use

---

## Data Requirements

### NDVI CSV
```
date,NDVI
2017-01-09,0.48
2017-02-08,0.39
```
Column names are auto-detected. Supports MODIS 8-day, Sentinel-2 10-day, or irregularly spaced composites. Multi-site CSVs supported with a `site_key` column.

### NASA POWER Meteorological CSV
Download from [NASA POWER Data Access](https://power.larc.nasa.gov/data-access-viewer/).  
Select **Daily** temporal resolution and **point** geometry. Recommended parameters:

| Parameter | Variable | Role |
|---|---|---|
| Mean temperature 2m | `T2M` | GDD, VPD |
| Min temperature 2m | `T2M_MIN` | SOS trigger |
| Max temperature 2m | `T2M_MAX` | Heat stress |
| Precipitation | `PRECTOTCORR` | Monsoon trigger |
| Relative humidity | `RH2M` | Moisture proxy |
| Surface soil wetness | `GWETTOP` | Leaf flush |
| Root zone soil wetness | `GWETROOT` | Drought resistance |
| Wind speed 2m | `WS2M` | Senescence (EOS) |
| Incoming solar radiation | `ALLSKY_SFC_SW_DWN` | POS timing |

The app **automatically derives**: `GDD_5`, `GDD_10`, `GDD_cum`, `DTR`, `VPD`, `SPEI_proxy`, `log_precip`, `MSI`, `T2M_RANGE`

---

## Local Installation

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
pip install -r requirements.txt
streamlit run app/universal_Indian_forest_phenology_v5.py
```

**One-click launchers (no terminal needed):**
- 🪟 Windows — double-click `run_app.bat`
- 🍎 macOS — double-click `Run Phenology App.command`
- 🐧 Linux — double-click `run_app.sh`

---

## Repository Structure

```
Indian-forest-phenology/
├── app/
│   └── universal_Indian_forest_phenology_v5.py   ← main app (v5, data-driven)
├── data/
│   └── ndvi/                                      ← 30+ sample NDVI files
├── docs/
│   └── user_guide.md
├── scripts/
│   ├── gee_extract_modis_ndvi.js
│   └── gee_extract_sentinel2_ndvi.js
├── run_app.bat                                    ← Windows one-click launcher
├── run_app.sh                                     ← macOS/Linux one-click launcher
├── requirements.txt
└── README.md
```

---

## Methodology

### Phenology Extraction (v5 — Data-Driven)
1. **Cadence detection** — median of observed date differences
2. **Gap identification** — gaps > 8× cadence preserved as NaN
3. **Interpolation** — adaptive-frequency grid, within-segment only
4. **SG smoothing** — per-segment, window ≤ 31 steps (≈155 days)
5. **Cycle length** — autocorrelation of smoothed series
6. **Trough detection** — min distance = 40% of cycle length
7. **MIN_AMPLITUDE** — 5% of data P5–P95 range
8. **SOS / EOS** — first/last crossing of user% × per-cycle amplitude
9. **POS** — raw NDVI maximum between SOS and EOS
10. **Season year** — trough start year (not POS year)

### Regression Model
1. Met features computed in user-specified window before each event
2. Pearson r + Spearman ρ composite ≥ 0.40 filter
3. Collinearity filter: |r| > 0.85 → drop weaker feature
4. Forward selection: add if LOO R² improves ≥ 0.03
5. Ridge / LOESS / Polynomial / GPR with Leave-One-Out CV

---

## Citation

```
Sharma, S. (2025). Universal Indian Forest Phenology Predictor v5 [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
