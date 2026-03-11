# 🌲 Universal Forest Phenology Assessment — v5

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Design-100%25_Data--Driven-brightgreen" />
  <img src="https://img.shields.io/badge/Forest%20Types-Universal-2E7D32" />
  <img src="https://img.shields.io/badge/Satellite-MODIS%20%7C%20Sentinel--2-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A **100% data-driven Streamlit application** for extracting and predicting phenological events —
**Start of Season (SOS)**, **Peak of Season (POS)**, and **End of Season (EOS)** — from any uploaded
NDVI time series and NASA POWER meteorological data. Works for **any Indian forest type** with
zero hardcoded presets — every threshold, cadence, amplitude floor, and feature ranking is derived
exclusively from your data.

**[▶ Open Live App](https://universal-forest-phenology-assessment.streamlit.app)**

---

## Overview

| Item | Detail |
|------|--------|
| **Forest types** | Universal — any Indian forest (or any ecosystem worldwide) |
| **Phenological events** | SOS · POS · EOS · LOS (Length of Season) |
| **NDVI input** | Any CSV with `Date` + `NDVI` columns (MODIS MOD13Q1, Sentinel-2, or other) |
| **Meteorological input** | NASA POWER daily export (headers auto-detected and skipped) |
| **Model** | Ridge / LOESS / Polynomial / Gaussian Process — LOO cross-validated |
| **Feature selection** | 100% data-driven — pure Pearson \|r\| + Spearman \|ρ\| composite ranking |
| **Minimum data** | 3 growing seasons (≥ 5 recommended for reliable R²) |
| **Version** | v5 — zero hardcoding; all parameters computed from uploaded data |

---

## Features

### 📊 Data Overview Tab
- Auto-detected observation cadence, dominant cycle period, global NDVI amplitude
- Every extraction parameter shown with its derivation formula
- Detected meteorological parameters (raw + derived)
- Growing window summary

### 🔬 Training & Models Tab
- Automatic NDVI 5-day interpolation + per-segment Savitzky-Golay smoothing
- Valley-anchored amplitude threshold phenology extraction
- All 7 bug fixes applied (v2–v7): plateau filter, SG window, gap tolerance, season window alignment, raw POS
- Model performance cards (LOO R², MAE, number of seasons)
- Fitted equations in tabbed layout (SOS / POS / EOS)
- Feature role table — colour-coded: ✅ IN MODEL · ➖ Correlated not selected · ⬜ Below threshold
- Observed vs Predicted scatter plots
- Download phenology table as CSV

### 📈 Correlations & Met Tab
- Feature heatmaps: Pearson r + Spearman ρ across SOS / POS / EOS
- Year-by-year meteorological + NDVI panels for each growing season
- Detailed per-event correlation tables with significance

### 🔮 Predict Tab
- Event-scoped input fields pre-filled from training data means
- Ecological order enforcement (SOS < POS < EOS with automatic correction)
- LOS, green-up lag, and senescence lag computed from predictions
- Download predictions as CSV

### 📖 Technical Guide Tab
- Full methodology documentation
- Bug-fix log (v2–v7) with before/after comparison
- Threshold sensitivity guide
- Citation

---

## v5 Core Design — 100% Data-Governed

**No forest-type dropdown. No lookup tables. No hardcoded numbers.**
You provide only the growing-window start and end month — everything else is computed from your data.

| Parameter | Old (hardcoded) | v5 (data-driven) |
|-----------|----------------|------------------|
| NDVI cadence | 16 days assumed | `median(observed date diffs)` |
| Max interpolation gap | 60 days fixed | `8 × detected cadence` |
| Trough min separation | 145 days fixed | `40% of autocorrelation cycle period` |
| SG smoother window | 31 steps fixed | `42% of detected cycle period` |
| Minimum amplitude | 0.02 fixed | `5% of P5–P95 NDVI range` |
| Amplitude-gap threshold | 0.10 fixed | `10% of global amplitude` |
| Feature priority | hardcoded list per event | pure Pearson + Spearman from data |
| Correlation gate | 0.40 fixed | sidebar slider (0.20–0.70) |
| Min season length | 150 days fixed | `35% of detected cycle` (or user override) |

---

## Bug Fixes (v2 → v7)

| Fix | Description |
|-----|------------|
| **v2** | Plateau trough filter: 85% adaptive ceiling; disabled for low-amplitude evergreen forests |
| **v3** | SG window data-derived (≤ 42% of cycle); `MIN_AMPLITUDE` = 5% of P5–P95 NDVI range |
| **v4** | Gap tolerance scales with amplitude — 50% tolerant for strong signals, 20% strict for weak |
| **v5** | Head/tail segment extraction with amplitude-aware gap checks |
| **v6** | `season_start` derived from POS date — eliminates blank windows in cross-year configs |
| **v6b** | `POS_Date` = raw NDVI maximum (not smoothed peak) |
| **v7** | Deduplicate by `Season_Start` (not calendar `Year`) — no window ever left blank |

---

## Data Requirements

### NDVI CSV
```
Date,NDVI
2016-01-01,0.42
2016-01-17,0.45
2016-02-02,0.51
```
Column names are auto-detected (case-insensitive). Supports MODIS 16-day, Sentinel-2 monthly,
or any irregular composite. Multi-site: add a `site_key` column — app shows a site selector.

### NASA POWER Meteorological CSV
Download from [NASA POWER Data Access](https://power.larc.nasa.gov/data-access-viewer/).
Select **Daily** temporal resolution and **Point** geometry. Recommended parameters:

| Parameter | Variable | Role |
|-----------|----------|------|
| Mean temperature 2m | `T2M` | GDD, VPD |
| Min temperature 2m | `T2M_MIN` | SOS trigger |
| Max temperature 2m | `T2M_MAX` | Heat stress |
| Precipitation | `PRECTOTCORR` | Monsoon trigger |
| Relative humidity | `RH2M` | Moisture proxy |
| Surface soil wetness | `GWETTOP` | Leaf flush |
| Root zone soil wetness | `GWETROOT` | Drought resistance |
| Wind speed 2m | `WS2M` | Senescence (EOS) |
| Incoming solar radiation | `ALLSKY_SFC_SW_DWN` | POS timing |

The app **automatically derives**: `GDD_5`, `GDD_10`, `GDD_cum`, `DTR`, `VPD`, `SPEI_proxy`,
`log_precip`, `MSI`, `T2M_RANGE`.

---

## Sample Data Included

20+ Indian forest sites ready to use:

| Site | Location | Forest Type |
|------|----------|-------------|
| IIT Tirupati | Andhra Pradesh | Tropical Dry Deciduous |
| Mukurthi | Nilgiris, Tamil Nadu | Shola / Southern Montane |
| Agumbe | Western Ghats, Karnataka | Tropical Wet Evergreen |
| Silent Valley | Kerala | Tropical Wet Evergreen |
| Sundarbans | West Bengal | Mangrove |
| Bhitarkanika | Odisha | Mangrove |
| Kaziranga | Assam | NE Moist Evergreen |
| Cherrapunji | Meghalaya | NE Moist Evergreen |
| Bastar | Chhattisgarh | Tropical Moist Deciduous |
| Mudumalai | Tamil Nadu | Tropical Dry Deciduous |
| Valley of Flowers | Uttarakhand | Alpine Meadow |
| Great Himalaya | Himachal Pradesh | Montane Temperate |
| Jaisalmer | Rajasthan | Tropical Thorn Scrub |
| Warangal | Telangana | Khair-Hardwickia Forest |

---

## Installation

```bash
git clone https://github.com/your-username/universal-forest-phenology-assessment.git
cd universal-forest-phenology-assessment
pip install -r requirements.txt
streamlit run app/universal_Indian_forest_phenology_v5.py
```

---

## Repository Structure

```
universal-forest-phenology-assessment/
├── app/
│   └── universal_Indian_forest_phenology_v5.py   ← main app (single file)
├── data/
│   └── ndvi/                                      ← 20+ sample NDVI CSVs
├── docs/
│   └── user_guide.md                              ← detailed methodology
├── scripts/
│   ├── gee_extract_modis_ndvi.js                  ← GEE: MODIS NDVI extraction
│   └── gee_extract_sentinel2_ndvi.js              ← GEE: Sentinel-2 NDVI extraction
├── requirements.txt
└── README.md
```

---

## Methodology

### Phenology Extraction — Valley-Anchored Amplitude Method
1. Gaps > `8 × cadence` preserved as NaN (not interpolated)
2. NDVI resampled to 5-day grid by linear interpolation within valid segments
3. Per-segment Savitzky-Golay smoothing (window ≤ 42% of dominant cycle period)
4. Valley (trough) detection — minimum separation = 40% of autocorrelation cycle period
5. Amplitude `A = NDVI_max − NDVI_min` computed from raw 5-day values per cycle
6. Threshold = `NDVI_min + threshold% × A`
7. SOS = first crossing on ascending limb
8. EOS = last crossing on descending limb
9. POS = **raw NDVI maximum** between SOS and EOS (not smoothed)

### Regression Model
1. 15-day pre-event meteorological windows computed per season per event
2. Features ranked by composite score = `max(|Pearson r|, |Spearman ρ|)` — no preset list
3. Correlation gate: composite score ≥ threshold (default 0.40, adjustable via sidebar)
4. Collinearity filter: `|r| > 0.85` between candidates → weaker one dropped
5. Incremental LOO R² check: feature added only if it improves LOO R² by ≥ 0.03
6. Ridge Regression / LOESS / Polynomial / Gaussian Process — user-selectable

### Threshold Sensitivity
| SOS/EOS % | Effect |
|-----------|--------|
| 5% | Very sensitive — earliest SOS / latest EOS |
| **10%** | **Scientific default** |
| 15–20% | Core growing period only |
| 25–30% | Conservative — peak season only |

---

## Model Performance (R² LOO)

| R² | Interpretation |
|----|---------------|
| > 0.80 | Strong predictability |
| 0.50–0.80 | Good |
| 0.30–0.50 | Moderate — acceptable for short records |
| < 0.30 | Weak — collect more years of data |

---

## Citation

```
Sharma, S. (2025). Universal Forest Phenology Assessment v5 [Software].
GitHub. https://github.com/your-username/universal-forest-phenology-assessment
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
