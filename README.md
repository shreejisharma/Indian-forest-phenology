# 🌲 Universal Indian Forest Phenology Predictor — v3

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
| **Models** | Ridge · LOESS · Polynomial (deg 2/3) · Gaussian Process — **all fitted simultaneously**, best auto-selected per event by LOO R² |
| **Feature selection** | Pearson \|r\| ≥ 0.40 filter → collinearity removal (threshold scaled to dataset size) → incremental LOO R² check |
| **Minimum data** | 3 growing seasons (≥ 5 recommended for reliable R²) |

---

## What's New in v3 — Auto-Best Model Selection

| Capability | v5 (previous) | v3 (this release) |
|---|---|---|
| Models fitted | One user-selected model | **All models fitted simultaneously** |
| Model selection | User radio button only | **Auto-selects best by LOO R²** per event |
| User preference | Hard override | Honoured within 0.02 R² of best |
| Equation display | Plain monospace box | **Rich styled box** with label · equation · stats · all-model comparison |
| Model badges | None | Colour-coded inline badges (Ridge · LOESS · Poly · GPR) |
| Prediction tab | Plain event headers | Per-event **coloured header + model badge** |
| Feature selection collinearity | Fixed 0.85 threshold | **Scaled to dataset size** (0.97 for n ≤ 10, 0.90 for n ≤ 20, 0.85 for n > 20) |
| Max features default | 1 | **3** — matches multi-feature equations |
| Welcome screen | Basic text | Styled cards with feature badges |
| CSS | Basic | Enhanced — `.eq-box` sub-classes, `.model-badge`, `.pred-event-header`, `.upload-panel` |

---

## What Was Already in v5 (Carried Forward Unchanged)

| Parameter | v4 (hardcoded) | v5 / v3 (data-driven) |
|---|---|---|
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

## Bug Fixes (v3)

- **Fix 1 (v3):** Max-features sidebar default was `1` — forced single-feature equations regardless of data quality. Changed to `3`.
- **Fix 2 (v3):** Collinearity threshold `0.85` too tight for small datasets — with n ≤ 6 seasons, nearly all feature pairs have \|r\| > 0.85 by chance, causing only the first feature to be selected. Threshold now scales with n.
- **Fix 3 (v5):** Season year = trough start year (not POS year) — eliminates duplicate-year collision
- **Fix 4 (v5):** POS = raw NDVI peak (not smoothed) — exact match with observed maximum
- **Fix 5 (v4):** Gap-tolerant cycle extraction (50% tolerance for high-amplitude seasons)
- **Fix 6 (v3/v4):** SG window capped at 31 steps — prevents over-smoothing across seasons
- **Fix 7 (v2):** Plateau trough filter 85% ceiling — fixes missing seasons in evergreen forests

---

## Features

### 📊 Data Overview Tab
- Auto-characterizes uploaded NDVI: cadence, dynamic range, evergreen index
- Lists all detected met parameters with their statistics from your file
- Summary plot — all values derived from your uploaded data

### 🔬 Season Extraction & Models Tab
- Fully data-adaptive phenology extraction (cadence, amplitude, cycle length from data)
- **All models (Ridge · LOESS · Poly-2 · Poly-3 · GPR) fitted simultaneously** — best auto-selected per event
- Model performance cards: LOO R², MAE, selected features, colour-coded model badge, all-model R² comparison line
- **Rich equation boxes** — label · equation · stats line · all-models comparison in one styled block
- All-model comparison table under each event tab
- Feature role table — IN MODEL · Redundant · Below threshold
- Observed vs Predicted scatter plots
- Climate window coverage audit per event
- **Download phenology table (CSV)**
- **Download model coefficients (CSV)**

### 📈 Climate Correlations Tab
- Feature correlation bar chart + heatmap (data-ranked, no preset priority)
- Year-by-year NDVI + Meteorology plots

### 🎯 Driver Sensitivity Tab
- Dominant driver card per event with coloured border
- Sensitivity heatmap and driver dominance bar chart
- Radar chart of factor influence across events

### 🔮 Predict Tab
- Per-event coloured header showing best model + R²
- Inputs pre-filled with training data means
- Value range hints from training data shown in tooltips
- LOO-validated predictions with ecological order enforcement (SOS < POS < EOS)
- Download predictions as CSV

### 📖 User Guide Tab
- Full methodology documentation
- Auto-model selection table (Ridge / LOESS / Poly / GPR — when each wins)
- R² interpretation guide for research use
- Data format specifications

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

The app **automatically derives**: `GDD_5`, `GDD_10`, `GDD_cum`, `DTR`, `VPD`, `SPEI_proxy`, `log_precip`, `MSI`

> ⚠️ **Important:** Upload a **continuous daily** meteorological file — not one sampled at NDVI cadence (5/16-day). The app needs dense daily records to compute meaningful climate averages in each pre-event window.

---

## Local Installation

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
pip install -r requirements.txt
streamlit run app/forest_phenology_v3.py
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
│   └── forest_phenology_v3.py                    ← main app (v3, auto-best model)
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

### Phenology Extraction (Data-Driven)
1. **Cadence detection** — median of observed date differences
2. **Gap identification** — gaps > 8× cadence preserved as NaN
3. **Interpolation** — adaptive-frequency grid, within-segment only
4. **SG smoothing** — per-segment, window ≤ 31 steps (≈ 155 days)
5. **Cycle length** — autocorrelation of smoothed series
6. **Trough detection** — min distance = 40% of cycle length
7. **MIN_AMPLITUDE** — 5% of data P5–P95 range
8. **SOS / EOS** — first/last crossing of user% × per-cycle amplitude
9. **POS** — raw NDVI maximum between SOS and EOS
10. **Season year** — trough start year (not POS year)

### Regression Model (v3 — Auto-Best Selection)
1. Met features computed in user-specified window before each event
2. Pearson r + Spearman ρ composite ≥ 0.40 filter
3. Collinearity filter: threshold scaled to dataset size (0.97 / 0.90 / 0.85)
4. Forward selection: add feature if LOO R² improves ≥ 0.03
5. **All models fitted**: Ridge · LOESS · Polynomial deg-2 · Polynomial deg-3 · GPR
6. **Best model auto-selected** per event by LOO R²
7. User preferred model honoured when within 0.02 R² of automatic best

| Model | When it typically wins |
|---|---|
| **Ridge** | Small datasets (n < 6), stable linear relationships |
| **LOESS** | Nonlinear single-driver responses |
| **Polynomial deg-2** | Unimodal / curved responses |
| **Polynomial deg-3** | Higher-order curvature, needs n > 5 |
| **GPR** | Complex nonlinear patterns, needs n ≥ 5 |

---

## Citation

```
Sharma, S. (2025). Universal Indian Forest Phenology Predictor v3 [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
