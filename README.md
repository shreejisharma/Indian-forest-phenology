# 🌲 Universal Indian Forest Phenology Assessment

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-Gemini%20Free-8E44AD?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/statsmodels-optional-9cf" />
  <img src="https://img.shields.io/badge/Design-100%25_Data--Driven-brightgreen" />
  <img src="https://img.shields.io/badge/Forest%20Types-Universal-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <a href="https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/▶%20Open%20Live%20App-Click%20Here-2E7D32?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</p>

A **fully data-driven Streamlit application** for extracting and predicting phenological events —
**Start of Season (SOS)**, **Peak of Season (POS)**, and **End of Season (EOS)** — across all Indian
forest types. Upload your NDVI time series and meteorological data; every threshold, feature, and
model coefficient is derived entirely from your data — no hardcoded presets, no forest-type
selection required. Includes a built-in **🤖 AI Assistant** powered by Google Gemini (free tier)
and a **🛰️ Multi-Sensor Comparison** tab for Landsat, Sentinel-2, and MODIS.

---

## Overview

| Item | Detail |
|---|---|
| **Works with** | All Indian forest types — Tropical Dry/Moist Deciduous, Wet Evergreen, Shola, Thorn, Mangrove, NE India, Himalayan, Alpine — **no forest-type selection needed** |
| **Phenological events** | SOS · POS · EOS · LOS (Length of Season) |
| **NDVI input** | Any CSV with `Date` + `NDVI` columns (MODIS MOD13Q1, Sentinel-2, Landsat, or custom) |
| **Meteorological input** | NASA POWER daily export or custom CSV (headers auto-detected) |
| **Prediction engine** | Ridge · LOESS · Polynomial (deg 2/3) · Gaussian Process — **all fitted simultaneously**, best auto-selected per event by LOO R² |
| **Feature selection** | Pearson \|r\| ≥ 0.40 filter → collinearity removal → forward LOO R² selection |
| **Minimum data** | 3 complete growing seasons (≥ 5 recommended for reliable statistics) |
| **AI Assistant** | Google Gemini (free tier) — ask questions about your results in plain language |
| **Multi-sensor** | Compare SOS/POS/EOS across Landsat, Sentinel-2, and MODIS with inter-sensor agreement stats |

---

## Application Tabs

### 📊 Tab 1 — Data Quality
- KPI cards: year range, mean NDVI, sensor cadence, evergreen index
- Auto-characterises uploaded NDVI: cadence, dynamic range, P5/P95
- Lists all detected meteorological parameters with mean, std, min, max
- Summary plot — NDVI distribution + met parameter bar chart + data stats panel
- Derived variable banner showing auto-computed features (GDD, log-rainfall, VPD, etc.)
- Climate data coverage warnings — large gaps, partial-year detection, paired-with-NDVI warning

### 🌿 Tab 2 — Season Extraction & Models
- Fully data-adaptive phenology extraction (cadence, amplitude, cycle length all from data)
- All 5 models fitted simultaneously — Ridge · LOESS · Poly-2 · Poly-3 · GPR
- Best model auto-selected per event by LOO R²; user preference honoured within 0.02 R²
- Model performance cards with colour-coded badge, LOO R², MAE, features, all-model comparison
- Feature role table — IN MODEL · Redundant · Below threshold · Did not improve accuracy
- Observed vs Predicted scatter (best model, per event)
- Climate window coverage audit — per-event table with seasons covered, avg rows/window, data quality rating
- Partial-year met detection — flags years missing growing-season data
- Download phenology table (CSV) and model coefficients (CSV)

### 🏆 Tab 3 — Model Results
- Rich equation boxes showing the fitted formula per event
- All-model comparison table (LOO R² for every model per event)
- Colour-coded model badges — Ridge (green) · LOESS (blue) · Poly-2 (orange) · Poly-3 (red-orange) · GPR (purple)
- Confidence level per event: HIGH (R² > 0.6) · MEDIUM · LOW

### 🎯 Tab 4 — Climate Drivers
- Feature correlation bar chart + Pearson r heatmap (data-ranked, p-value annotated with \* / \*\*)
- Full correlation table per event — Pearson r, Spearman ρ, Composite score
- Year-by-year NDVI + Meteorology overlay plots
- Climate Driver Sensitivity Analysis — days shifted per 1σ change in each variable
- Driver dominance bar chart and radar chart of factor influence across events

### 🔮 Tab 5 — Predict
- Per-event coloured header showing which model was auto-selected and its R²
- Input fields pre-filled with training data means; tooltips show observed ranges
- Ecological order enforcement — SOS < POS < EOS (auto-corrected if violated)
- Season length, green-up phase (SOS → POS), and senescence phase (POS → EOS) displayed
- Download predictions as CSV

### 📖 Tab 6 — User Guide
- Full methodology documentation
- Auto-model selection table (when each model wins)
- R² interpretation guide
- Data format requirements and NASA POWER download instructions

### 🤖 Tab 7 — AI Assistant
- Powered by **Google Gemini (free tier)**
- Context-aware: automatically receives your phenology results, model performance, NDVI info, and met info
- Ask questions in plain language: *"Why is my EOS R² low?"*, *"What does GDD_cum mean?"*, *"Which climate driver matters most?"*
- Requires `GEMINI_API_KEY` in `.streamlit/secrets.toml` (see setup below)

### 🛰️ Tab 8 — Sensor Compare
- Upload 1–3 separate NDVI CSVs (one per sensor: Landsat, Sentinel-2, MODIS)
- Single file: shows that sensor's extracted phenological events
- Two or three files: unlocks side-by-side comparison charts and inter-sensor agreement stats (Bias, RMSE, Pearson r)
- Uses the same sidebar season window and threshold settings as the main analysis

---

## Data Requirements

### NDVI CSV
```
Date,NDVI
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
> Do **not** use a file sampled at NDVI cadence — the app needs dense daily data to compute meaningful climate averages in each pre-event window.
> Years with data only for part of the year (e.g. Jan–Apr) will be silently dropped from model training. The app detects and warns about this — look for the **"Partial-year meteorological data detected"** banner.

---

## Sidebar Controls Reference

| Control | Default | Description |
|---|---|---|
| **Start / End month** | Jun / May | Calendar bounds for growing season search |
| **Minimum season length** | 100 days | Cycles shorter than this are ignored |
| **SOS threshold** | 10% | NDVI must rise to this % of amplitude to trigger SOS |
| **EOS threshold** | 10% | NDVI must fall below this % of amplitude to trigger EOS |
| **Preferred model** | Ridge | Your model preference (honoured within 0.02 R² of best) |
| **Climate window** | 15 days | Days before each event to average met variables |
| **Max climate variables** | 3 | Maximum features per model |
| **Split NDVI plot every N years** | 8 | Long datasets split into panels of this many years |

---

## Methodology

### Phenology Extraction (Fully Data-Driven)

1. **Cadence detection** — median of observed date differences
2. **Gap identification** — gaps > 8× cadence preserved as NaN
3. **5-day interpolation grid** — always interpolated to a 5-day grid regardless of input cadence
4. **SG smoothing** — per-segment Savitzky-Golay, window ≤ 31 steps (≈ 155 days at 5-day grid)
5. **Cycle length** — autocorrelation peak of smoothed series
6. **Trough detection** — min distance = 40% of detected cycle length; boundary-aware 3-pass search
7. **MIN_AMPLITUDE** — 5% of data P5–P95 range (data-derived floor)
8. **SOS / EOS threshold** — user % × per-cycle amplitude (local vmin/vmax per season)
9. **POS** — raw NDVI maximum between SOS and EOS dates
10. **Cross-year EOS** — EOS allowed to extend into the following year up to the next trough

### Regression Model Pipeline

1. Met features computed over user-specified window before each event date
2. Pearson r + Spearman ρ composite ≥ 0.40 filter
3. Collinearity filter (threshold scaled to dataset size)
4. Forward LOO R² selection: add feature if improvement ≥ 0.03 (≥ 0.08 for n ≤ 5)
5. **All 5 models fitted simultaneously**:
   - **Ridge** — `RidgeCV` with 30 log-spaced alphas from 10⁻³ to 10⁴
   - **LOESS** — statsmodels lowess + PCA projection for multi-feature; numpy fallback
   - **Poly-2 / Poly-3** — `PolynomialFeatures` + `StandardScaler` + `RidgeCV`
   - **GPR** — `ConstantKernel × RBF + WhiteKernel`, 5 restarts
6. **Best model selected** per event by LOO R²
7. User preferred model honoured when within **0.02 R²** of automatic best

| Model | When it typically wins |
|---|---|
| **Ridge** | Small datasets (n < 6), stable linear relationships |
| **LOESS** | Nonlinear single-driver responses, moderate n |
| **Polynomial deg-2** | Unimodal / optimum-response curves |
| **Polynomial deg-3** | Higher-order curvature; needs n > 5 |
| **GPR** | Complex nonlinear patterns; needs n ≥ 5 |

---

## Local Installation

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
pip install streamlit pandas numpy scipy scikit-learn matplotlib statsmodels google-generativeai
streamlit run Universal_Indian_Forest_Phenology_Assessment.py
```

For LOESS with full multi-feature support (recommended):
```bash
pip install statsmodels
```

---

## AI Assistant Setup

The 🤖 AI Assistant tab requires a free Google Gemini API key.

1. Go to [aistudio.google.com](https://aistudio.google.com) and sign in with Google
2. Click **Get API Key** → **Create API key** → copy it
3. Create `.streamlit/secrets.toml` in the project folder:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
4. Place `ai_assistant_gemini_free.py` in the same folder as the main app

> ⚠️ Never push `secrets.toml` to GitHub. It is excluded by `.gitignore` in this repo.
> A safe placeholder template is provided as `.streamlit/secrets.toml.example`.

---

## Interpreting R² Values

| LOO R² | Interpretation |
|---|---|
| > 0.80 | Strong — climate is a reliable predictor of this event |
| 0.50 – 0.80 | Good |
| 0.30 – 0.50 | Moderate — some predictive signal present |
| < 0.30 | Weak — more seasons or better climate drivers needed |

> **Small-n caution:** With n ≤ 3 seasons, Pearson r and Spearman ρ are mathematically constrained — any 3 monotonically ordered values give \|r\| = 1.0 regardless of the true relationship. The app detects this and displays a diagnostic warning with fix instructions.

---

## How Many Years of Data Do I Need?

| Years available | What the tool can do |
|---|---|
| 1 | Shows NDVI chart and phenology dates only |
| 2 | Fits basic models — treat results as exploratory |
| 3 – 4 | Models available with caution — indicative only |
| 5 – 9 | Reliable models and predictions |
| 10+ | Best results — strong statistical reliability |

---

## Repository Structure

```
Indian-forest-phenology/
├── Universal_Indian_Forest_Phenology_Assessment.py   ← main application
├── ai_assistant_gemini_free.py                       ← Gemini AI tab module
├── .streamlit/
│   ├── secrets.toml          ← your real API key (NOT pushed to GitHub)
│   └── secrets.toml.example  ← safe placeholder for others
├── .gitignore
└── README.md
```

---

## Frequently Asked Questions

**Why do all features show Spearman ρ = ±1.000?**
This is a mathematical artefact of having only 3 effective training seasons. With n = 3, any monotonically ordered values produce ρ = ±1.0 exactly. It does not mean every variable is a perfect predictor — you need more complete years of met data.

**Why does my EOS model only have 3 seasons when I uploaded 6 years?**
Your meteorological file likely only covers part of the year in some years (e.g. Jan–Apr only). Years without data in the EOS window (typically Sep–Nov) are excluded from training. Look for the "Partial-year meteorological data detected" warning.

**What is the recommended Climate Window setting?**
At least 6× your met cadence. For daily met data: 15–30 days. The app calculates and displays the recommended minimum in the cadence warning banner.

**Why does LOESS show R² = −1.0?**
With n = 3, LOO for LOESS trains on 2 points and predicts the 3rd. If the 3rd point is far from that line, R² goes negative. Ridge regression is more robust at very small n.

---

## Citation

```
Sharma, S. (2025). Universal Indian Forest Phenology Assessment [Software].
GitHub. https://github.com/shreejisharma/Indian-forest-phenology
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
