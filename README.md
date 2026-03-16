# 🌲 Universal Indian Forest Phenology Assessment

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/statsmodels-optional-9cf" />
  <img src="https://img.shields.io/badge/Design-100%25_Data--Driven-brightgreen" />
  <img src="https://img.shields.io/badge/Forest%20Types-Universal-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A **fully data-driven Streamlit application** for extracting and predicting phenological events —
**Start of Season (SOS)**, **Peak of Season (POS)**, and **End of Season (EOS)** — across all Indian
forest types. Upload your NDVI time series and meteorological data; every threshold, feature, and
model coefficient is derived entirely from your data — no hardcoded presets, no forest-type
selection required.

**[▶ Open Live App](https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/)**
&nbsp;·&nbsp;
**[📦 View Source](https://github.com/shreejisharma/Indian-forest-phenology/blob/main/app/Universal_Indian_Forest_Phenology_Assessment.py)**

---

## Overview

| Item | Detail |
|---|---|
| **Works with** | All Indian forest types — Tropical Dry/Moist Deciduous, Wet Evergreen, Shola, Thorn, Mangrove, NE India, Himalayan, Alpine — **no forest-type selection needed** |
| **Phenological events** | SOS · POS · EOS · LOS (Length of Season) |
| **NDVI input** | Any CSV with `Date` + `NDVI` columns (MODIS MOD13Q1, Sentinel-2, Landsat, or custom) |
| **Meteorological input** | NASA POWER daily export or custom CSV (headers auto-detected) |
| **Prediction engine** | Ridge · LOESS · Polynomial (deg 2/3) · Gaussian Process — **all fitted simultaneously**, best auto-selected per event by LOO R² |
| **Feature selection** | Pearson \|r\| ≥ 0.40 filter → collinearity removal (threshold scaled to dataset size) → forward LOO R² selection |
| **Minimum data** | 3 complete growing seasons (≥ 5 recommended for reliable statistics) |

---

## What's New in v3

### Prediction Engine — Auto-Best Model Selection (ported from v5.3)

| Capability | Previous | v3 |
|---|---|---|
| Models fitted per run | One (user-selected) | **All 5 models simultaneously** |
| Model selection | User radio button | **Auto-selects best by LOO R²** per event |
| User preference | Hard override | Honoured when within **0.02 R²** of automatic best |
| LOO cross-validation | Separate per model | **Unified `loo_cv()` helper** — Ridge, LOESS (statsmodels), Poly, GPR |
| LOESS implementation | Pure-numpy fallback only | **statsmodels lowess** with PCA projection for multi-feature input; numpy fallback if statsmodels absent |
| GPR kernel | `RBF + WhiteKernel` | **`ConstantKernel × RBF + WhiteKernel`** (more stable, matches v5.3) |
| Ridge alpha search | Fixed ALPHAS list | **`RidgeCV` with `np.logspace(−3, 4, 30)`** |

### UI & Display Improvements

| Element | Previous | v3 |
|---|---|---|
| Equation display | Plain monospace box | **Rich styled box** — `eq-label` · `eq-main` · `eq-meta` · `eq-models` sub-sections |
| Model identification | Text only | **Colour-coded inline badges** — Ridge (green) · LOESS (blue) · Poly-2 (orange) · Poly-3 (red-orange) · GPR (purple) |
| Prediction event headers | Plain text | **Per-event coloured bordered panel** with model badge + R² |
| Welcome screen | Plain text list | **Styled upload cards** with feature-item pill badges |
| All-model comparison | Not shown | **Table under each event tab** showing every model's LOO R² |

### Data Quality Diagnostics (New in v3)

| Check | What it detects |
|---|---|
| **Partial-year met coverage** | Years where met data only covers Jan–Apr (DOY < 240) — these years silently drop out of model training |
| **Avg rows per window** | Warns when a climate window contains fewer than 4 rows (unreliable feature averages) |
| **Recommended window size** | Calculates and displays `6 × cadence` as the minimum recommended climate window |
| **n ≤ 3 full diagnosis** | Explains the three root causes — partial met data, Spearman ρ artefact, overfitting — with fix instructions |

### Feature Selection Improvements

| Parameter | Previous | v3 |
|---|---|---|
| Collinearity threshold | Fixed `0.85` | **Scaled to dataset size**: `0.97` (n ≤ 10) · `0.90` (n ≤ 20) · `0.85` (n > 20) |
| Max features default | `1` (always single-feature) | **`3`** — allows multi-variable equations when data supports it |
| Min rows per climate window | `max(1, window × 0.15)` | **`max(3, window × 0.15)`** — prevents features based on 1–2 observations |

---

## What Was Already in v5 (Carried Forward Unchanged)

| Parameter | Hardcoded (old) | Data-driven (v5 / v3) |
|---|---|---|
| NDVI cadence | assumed 16d | median of observed date differences |
| Max gap threshold | fixed 60d | 8× detected cadence |
| Trough min distance | fixed 145d | 40% of autocorrelation cycle estimate |
| MIN_AMPLITUDE | fixed 0.02 | 5% of data P5–P95 range |
| Feature priority | hard list per event | pure Pearson/Spearman ranking from data |
| Season year assignment | POS year | trough start year (fixes duplicate-year collision) |
| POS date | smoothed peak | raw NDVI maximum between SOS and EOS |
| SG smoothing window | fixed | per-segment, capped at 31 steps |
| Trough ceiling | none | 85% of global amplitude (fixes evergreen false troughs) |

---

## Bug Fixes in v3

| Fix | Description |
|---|---|
| **Partial-year met detection** | New per-year DOY coverage audit — detects years with seasonal data gaps (e.g. Jan–Apr only) that silently reduce training sample size |
| **Min rows per window** | Raised from `max(1, …)` to `max(3, …)` — prevents training features built from 1–2 data points |
| **Collinearity threshold scaling** | Was fixed at 0.85 — with n ≤ 6, nearly all feature pairs have \|r\| > 0.85 by chance, forcing single-feature models even when multi-feature would be appropriate |
| **Max features default** | Was `1` — meant multi-feature equations were never selected regardless of data quality |
| **n ≤ 3 error detail** | Old message was generic; new message explains all three root causes (partial met data · Spearman artefact · LOO overfitting) with a concrete fix |
| **LOESS with multi-feature** | Old fallback LOESS was univariate only; v3 uses PCA projection so LOESS competes fairly against Ridge/Poly when multiple features are selected |
| **GPR kernel stability** | `ConstantKernel × RBF + WhiteKernel` replaces plain `RBF + WhiteKernel` — reduces kernel optimisation failures on small datasets |

---

## Features

### 📊 Data Summary Tab
- Auto-characterises uploaded NDVI: cadence, dynamic range, P5/P95, evergreen index
- Lists all detected meteorological parameters with mean, std, min, max from your file
- Summary plot — NDVI distribution + met parameter bar chart + data stats panel

### 🔬 Season Extraction & Models Tab
- Fully data-adaptive phenology extraction (cadence, amplitude, cycle length all from data)
- **All 5 models fitted simultaneously** — Ridge · LOESS · Poly-2 · Poly-3 · GPR
- **Best model auto-selected per event** by LOO R²; user preference honoured within 0.02 R²
- Model performance cards with colour-coded badge, LOO R², MAE, features, all-model comparison
- **Rich equation boxes** — styled label · equation · stats · all-models line in one block
- All-model comparison table under each event tab (LOO R² for every model)
- Feature role table — IN MODEL · Redundant · Below threshold · Did not improve accuracy
- Observed vs Predicted scatter (best model, per event)
- **Climate window coverage audit** — per-event table with seasons covered, avg rows/window, data quality rating
- **Partial-year met detection** — flags years missing growing-season data
- **Download phenology table (CSV)**
- **Download model coefficients (CSV)**

### 📈 Climate Correlations Tab
- Feature correlation bar chart + Pearson r heatmap (data-ranked, p-value annotated)
- Full correlation table per event — Pearson r, Spearman ρ, composite score
- Year-by-year NDVI + Meteorology overlay plots (auto-detected air and soil parameters)

### 🎯 Driver Sensitivity Tab
- Dominant driver card per event with coloured border and direction annotation
- Sensitivity heatmap — days shifted per 1σ increase in each variable
- Driver dominance bar chart — ranked features per event
- Radar chart of factor influence magnitude across events
- Cross-event dominance summary panel
- Download full sensitivity table as CSV

### 🔮 Predict Tab
- Per-event coloured header showing **which model was auto-selected** and its R²
- Input fields pre-filled with training data means; tooltips show observed range
- Ecological order enforcement — SOS < POS < EOS (auto-corrected if violated)
- Season length, green-up phase, and senescence phase calculated
- Download predictions as CSV

### 📖 User Guide Tab
- Full methodology documentation
- Auto-model selection table (when each model wins)
- R² interpretation guide
- Data format requirements and NASA POWER download instructions

---

## Data Requirements

### NDVI CSV
```
date,NDVI
2017-01-09,0.48
2017-02-08,0.39
```
Column names are auto-detected (`date`, `Date`, `time`, `datetime` for dates; `ndvi`, `NDVI`, `evi`, `value` for NDVI). Supports MODIS 8-day, Sentinel-2 10-day, Landsat, or irregularly spaced composites. Multi-site CSVs supported with a `site_key` or `site_label` column.

### NASA POWER Meteorological CSV

Download from [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/).
Select **Daily** temporal resolution and **Point** geometry for your site coordinates.

Recommended parameters:

| Parameter | Variable | Role |
|---|---|---|
| Mean temperature 2 m | `T2M` | GDD, VPD |
| Min temperature 2 m | `T2M_MIN` | SOS trigger |
| Max temperature 2 m | `T2M_MAX` | Heat stress |
| Precipitation | `PRECTOTCORR` | Monsoon trigger |
| Relative humidity | `RH2M` | Moisture proxy |
| Surface soil wetness | `GWETTOP` | Leaf flush |
| Root zone soil wetness | `GWETROOT` | Drought resistance |
| Wind speed 2 m | `WS2M` | Senescence (EOS) |
| Incoming solar radiation | `ALLSKY_SFC_SW_DWN` | POS timing |

The app **automatically derives**: `GDD_5`, `GDD_10`, `GDD_cum`, `DTR` (diurnal temp range), `VPD`, `SPEI_proxy`, `log_precip`, `MSI` (moisture stress index)

> ⚠️ **Critical:** Upload a **continuous full-year daily** meteorological file — one row per day for every day of the year, for every year in your NDVI series.
>
> A file that only covers part of the year (e.g. Jan–Apr for alternate years) will silently drop those years from model training, reducing your effective sample size. The app now detects and warns about this pattern — look for the **"Partial-year meteorological data detected"** banner in the Season Extraction tab.
>
> Do **not** use a file sampled at NDVI cadence (every 5 or 16 days) — the app needs dense daily data to compute meaningful climate averages in each pre-event window.

---

## Sidebar Controls Reference

| Control | Default | Description |
|---|---|---|
| **Start / End month** | Jun / May | Calendar bounds for growing season search |
| **Minimum season length** | 100 days | Cycles shorter than this are ignored |
| **SOS threshold** | 10% | NDVI must rise to this % of amplitude to trigger SOS |
| **EOS threshold** | 10% | NDVI must fall below this % of amplitude to trigger EOS |
| **Preferred model** | Ridge | Your model preference (honoured within 0.02 R² of best) |
| **Climate window** | 15 days | Days before each event to average met variables. With 5-day met cadence, increase to ≥ 30 days |
| **Max climate variables** | 3 | Maximum features per model. Reduce to 1–2 if R² looks suspiciously high with < 6 seasons |

---

## Local Installation

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
pip install -r requirements.txt
streamlit run app/Universal_Indian_Forest_Phenology_Assessment.py
```

For LOESS with full multi-feature support (recommended):

```bash
pip install statsmodels
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
│   └── Universal_Indian_Forest_Phenology_Assessment.py   ← main application
├── data/
│   └── ndvi/                           ← sample NDVI files
├── docs/
│   └── user_guide.md
├── scripts/
│   ├── gee_extract_modis_ndvi.js
│   └── gee_extract_sentinel2_ndvi.js
├── run_app.bat                         ← Windows one-click launcher
├── run_app.sh                          ← macOS/Linux one-click launcher
├── requirements.txt
└── README.md
```

---

## Methodology

### Phenology Extraction (Fully Data-Driven)

1. **Cadence detection** — median of observed date differences
2. **Gap identification** — gaps > 8× cadence preserved as NaN
3. **Adaptive interpolation** — grid frequency = observed cadence, within-segment only
4. **SG smoothing** — per-segment Savitzky-Golay, window ≤ 31 steps (≈ 155 days at 5-day grid)
5. **Cycle length** — autocorrelation peak of smoothed series
6. **Trough detection** — min distance = 40% of detected cycle length; boundary-aware 3-pass search
7. **MIN_AMPLITUDE** — 5% of data P5–P95 range (data-derived floor)
8. **SOS / EOS threshold** — user% × per-cycle amplitude (local vmin/vmax per season)
9. **POS** — raw NDVI maximum between SOS and EOS dates
10. **Season year** — trough start year (not POS year)

### Regression Model Pipeline (v3 — Auto-Best Selection)

1. Met features computed over user-specified window before each event date (accumulation vs mean detected from column name)
2. Pearson r + Spearman ρ composite ≥ 0.40 filter (threshold relaxed for n ≤ 4)
3. Collinearity filter: threshold scales with n — `0.97` (n ≤ 10) · `0.90` (n ≤ 20) · `0.85` (n > 20)
4. Forward LOO R² selection: add feature if improvement ≥ 0.03 (≥ 0.08 for n ≤ 5)
5. **All 5 models fitted simultaneously**:
   - **Ridge** — `RidgeCV` with 30 log-spaced alphas from 10⁻³ to 10⁴
   - **LOESS** — statsmodels lowess + PCA projection for multi-feature; numpy fallback
   - **Poly-2 / Poly-3** — `PolynomialFeatures` + `StandardScaler` + `RidgeCV`
   - **GPR** — `ConstantKernel × RBF + WhiteKernel`, 5 restarts
6. **Best model selected** per event by LOO R² (`loo_cv()` — unified helper)
7. User preferred model honoured when within **0.02 R²** of automatic best

| Model | When it typically wins |
|---|---|
| **Ridge** | Small datasets (n < 6), stable linear relationships |
| **LOESS** | Nonlinear single-driver responses, moderate n |
| **Polynomial deg-2** | Unimodal / optimum-response curves |
| **Polynomial deg-3** | Higher-order curvature; needs n > 5 |
| **GPR** | Complex nonlinear patterns; needs n ≥ 5 |

### Data Quality Checks (v3)

The app runs the following checks automatically after loading your met file:

| Check | Flag condition | Fix |
|---|---|---|
| Met paired with NDVI | ≥ 90% of met dates match NDVI dates | Upload continuous daily met file |
| Large met gaps | Any gap > 60 days | Fill gaps or reduce climate window |
| **Partial-year coverage** | Any year with max DOY < 240 | Upload full Jan–Dec met for every year |
| Seasons missing climate data | Event window has 0 rows | Widen climate window or fix met file |
| **Avg rows per window** | Mean < 4 rows | Increase climate window (≥ 6× cadence) |
| n ≤ 3 effective seasons | After dropping years with no window data | Collect more complete years |

---

## Interpreting R² Values

| LOO R² | Interpretation |
|---|---|
| > 0.80 | Strong — climate is a reliable predictor of this event |
| 0.50 – 0.80 | Good |
| 0.30 – 0.50 | Moderate — some predictive signal present |
| < 0.30 | Weak — more seasons or better climate drivers needed |

> **Important — small-n caution:** With n ≤ 3 seasons, Pearson r and Spearman ρ are mathematically constrained — any monotonically ordered set of 3 values gives \|r\| = 1.0 regardless of the true relationship. The app detects this condition and displays a detailed diagnostic banner explaining the cause and how to fix it.

---

## Frequently Asked Questions

**Why do all features show Spearman ρ = ±1.000?**
This is a mathematical artefact of having only 3 effective training seasons. With n = 3, any set of values that are monotonically ordered (all increasing or all decreasing) produces a Pearson r and Spearman ρ of exactly ±1.0. It does not mean every climate variable is a perfect predictor. You need more complete years of met data to get meaningful correlations.

**Why does the EOS model only have 3 seasons when I uploaded 6 years of data?**
Your meteorological file likely only has data for part of the year in some years (e.g. Jan–Apr for alternate years). Years without data in the EOS window (typically Sep–Nov) cannot contribute to training. Look for the "Partial-year meteorological data detected" warning and upload a complete full-year file.

**Why does LOESS show R² = −1.0?**
With n = 3, LOO for LOESS trains on 2 points and predicts the 3rd. Two points define a line, and if the 3rd point is far from that line the prediction is poor — R² goes negative. This is expected with very small samples. Ridge regression is more robust at n = 3.

**What is the recommended Climate Window setting?**
Set it to at least **6 × your met cadence**. For 5-day met data, use ≥ 30 days. For daily met data, 15–30 days is typically sufficient. The app calculates and displays the recommended minimum in the cadence warning banner.

**How many years of data do I need?**

| Years available | What the tool can do |
|---|---|
| 1 | Shows NDVI chart and phenology dates only |
| 2 | Fits basic models — treat results as exploratory |
| 3 – 4 | Models available with caution — indicative only |
| 5 – 9 | Reliable models and predictions |
| 10+ | Best results — strong statistical reliability |

---

## Citation

```
Sharma, S. (2025). Universal Indian Forest Phenology Assessment [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
