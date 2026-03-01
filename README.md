# 🌲 Indian Forest Phenology Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/MODIS-MOD13Q1%20250m-green" />
  <img src="https://img.shields.io/badge/Sentinel--2-SR%2010m-blue" />
  <img src="https://img.shields.io/badge/Forest%20Types-11-brightgreen" />
  <img src="https://img.shields.io/badge/Sites-12-orange" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  A machine learning web application for extracting and predicting phenological events
  (Start of Season, Peak of Season, End of Season) across 11 Indian forest types using
  MODIS / Sentinel-2 NDVI and NASA POWER meteorological data.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Forest Types Supported](#-forest-types-supported)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Repository Structure](#-repository-structure)
- [Installation & Local Setup](#-installation--local-setup)
- [Deploy on Streamlit Cloud](#-deploy-on-streamlit-cloud)
- [Deploy on Hugging Face Spaces](#-deploy-on-hugging-face-spaces)
- [Data Format Guide](#-data-format-guide)
- [Recommended NASA POWER Parameters](#-recommended-nasa-power-parameters)
- [App Settings Quick Reference](#-app-settings-quick-reference)
- [Model Architecture](#-model-architecture)
- [Results Summary](#-results-summary)
- [Known Limitations](#-known-limitations)
- [Citation](#-citation)

---

## 🌿 Overview

This project builds a **universal phenology predictor** for Indian forests using:

- **NDVI time series** from MODIS MOD13Q1 (250m, 16-day) and Sentinel-2 SR (10m, monthly)
- **Meteorological drivers** from NASA POWER (MERRA-2 reanalysis, daily)
- **Ridge Regression** with Leave-One-Out cross-validation for honest small-sample R²
- **Three detection methods** for SOS: NDVI threshold, first sustained rainfall, or max NDVI rate-of-change

Phenological events — **SOS** (Start of Season), **POS** (Peak of Season), **EOS** (End of Season), and **LOS** (Length of Season) — are extracted for each year, then modelled against lagged climate variables to enable **future-year prediction**.

The app covers **11 forest types** defined by the Champion & Seth (1968) classification of Indian forests, validated across **20+ sites** spanning the Western Ghats, Eastern Ghats, Himalayas, NE India, and Thar Desert.

---

## 🚀 Live Demo

> **[▶ Open App on Streamlit Cloud](https://your-username-phenology.streamlit.app)**  
> *(Replace with your actual deployment URL after publishing)*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 **Universal CSV parser** | Auto-detects date + NDVI columns in any format; auto-skips NASA POWER header block |
| 🌱 **11 Forest types** | Pre-configured season windows, thresholds, and ecological drivers per type |
| 🌧️ **3 SOS methods** | NDVI threshold / First sustained rainfall (monsoon) / Max NDVI rate (alpine) |
| 📉 **3 EOS methods** | NDVI threshold / Max decline rate / Dry season onset |
| 📊 **Savitzky-Golay smoother** | Removes cloud-contamination spikes before event extraction |
| 🔗 **Pearson feature filter** | Only features with \|r\| ≥ 0.40 enter the model |
| 🤖 **Ridge Regression** | L2 regularisation with auto-tuned α (RidgeCV) |
| ✅ **LOO cross-validation** | Leave-One-Out gives unbiased R² for small n (8–10 seasons) |
| 📈 **Correlation explorer** | Interactive heatmap of all met–phenology correlations |
| 🔮 **Predict tab** | Enter any year's weather → get SOS/POS/EOS predictions with uncertainty |
| 🌡️ **Met visualiser** | Dual-axis temperature + rainfall + soil moisture timeline |
| 📋 **Methods tab** | Full technical documentation, R² guide, and improvement tips |

---

## 🌳 Forest Types Supported

| # | Forest Type | Season Window | Key Sites | NDVI Amplitude |
|---|-------------|--------------|-----------|---------------|
| 1 | 🍂 Tropical Dry Deciduous — Monsoon | Jun–May | Tirupati, Mudumalai | 0.45–0.55 |
| 2 | 🌿 Tropical Moist Deciduous — Monsoon | Jun–May | Simlipal, Bastar | 0.45–0.52 |
| 3 | 🌲 Tropical Wet Evergreen / Semi-Evergreen | Jan–Dec | Agumbe | 0.25–0.35 |
| 4 | 🌴 Tropical Dry Evergreen | Jan–Dec | Coromandel Coast | 0.20–0.35 |
| 5 | 🌵 Tropical Thorn Forest / Scrub | Jun–May | Jaisalmer | 0.10–0.20 |
| 6 | 🌳 Subtropical Broadleaved Hill Forest | Apr–Mar | Shiwaliks | 0.30–0.45 |
| 7 | 🏔️ Montane Temperate Forest | Apr–Nov | W Himalayas | 0.35–0.55 |
| 8 | ⛰️ Alpine / Subalpine Forest & Meadow | May–Oct | Spiti, Valley of Flowers | 0.15–0.85 |
| 9 | 🌫️ Shola Forest — Southern Montane | Jan–Dec | Mukurthi NP | 0.25–0.35 |
| 10 | 🌊 Mangrove Forest | Jan–Dec | Bhitarkanika | 0.20–0.30 |
| 11 | 🌿 NE India Moist Evergreen | Jan–Dec | Kaziranga region | 0.30–0.45 |

---

## 📁 Dataset

### NDVI Files (20+ sites, 2016–2025) — 32 files

**MODIS files (15 files):**

| File | Site | Seasons |
|------|------|---------|
| `MODIS_NDVI_TDD_Tirupati_2016_2025.csv` | Tirupati, AP | 10/10 ✅ |
| `MODIS_NDVI_TDD_Mudumalai_2016_2025.csv` | Mudumalai, TN | 10/10 ✅ |
| `MODIS_NDVI_TMD_Simlipal_2016_2025.csv` | Simlipal, Odisha | 10/10 ✅ |
| `MODIS_NDVI_TMD_Bastar_2016_2025.csv` | Bastar, CG | 10/10 ✅ |
| `MODIS_NDVI_TWE_Agumbe_2016_2025.csv` | Agumbe, Karnataka | 10/10 ✅ |
| `MODIS_NDVI_TWE_SilentValley_2016_2025.csv` | Silent Valley NP, Kerala | 10/10 ✅ |
| `MODIS_NDVI_SHO_Mukurthi_2016_2025.csv` | Mukurthi NP, TN | 10/10 ✅ |
| `MODIS_NDVI_SHO_Eravikulam_2016_2025.csv` | Eravikulam NP, Kerala | 10/10 ✅ |
| `MODIS_NDVI_MNG_Bhitarkanika_2016_2025.csv` | Bhitarkanika, Odisha | 10/10 ✅ |
| `MODIS_NDVI_MNG_Sundarbans_2016_2025.csv` | Sundarbans, WB | 10/10 ✅ |
| `MODIS_NDVI_NEE_Kaziranga_2016_2025.csv` | Kaziranga NP, Assam | 10/10 ✅ |
| `MODIS_NDVI_NEE_Cherrapunji_2016_2025.csv` | Cherrapunji, Meghalaya | 10/10 ✅ |
| `MODIS_NDVI_KHF_Warangal_2016_2025.csv` | Warangal, Telangana | 10/10 ✅ |
| `MODIS_NDVI_KHF_Cuttack_2016_2025.csv` | Cuttack, Odisha | 10/10 ✅ |
| `IIT_TIRUPATI_NDVI_2017-2025.csv` | IIT Tirupati Campus | 9/10 ✅ |

**Sentinel-2 monthly files (10 sites):**

| File | Site | Notes |
|------|------|-------|
| `S2_monthly_ALP_Spiti_2016_2025.csv` | Spiti Valley, HP | ⚠️ Filter NDVI < 0.01 |
| `S2_monthly_ALP_ValleyFlowers_2016_2025.csv` | Valley of Flowers, UK | ⚠️ Filter NDVI < 0.05 |
| `S2_monthly_TTF_Jaisalmer_2016_2025.csv` | Desert NP, Rajasthan | ⚠️ Use 10–12% threshold |
| `S2_monthly_TTF_Ranthambore_2016_2025.csv` | Ranthambore NP, Rajasthan | ✅ Ready |
| `S2_monthly_TDE_Pichavaram_2016_2025.csv` | Pichavaram, TN | ✅ Ready |
| `S2_monthly_TDE_PointCalimere_2016_2025.csv` | Point Calimere, TN | ✅ Ready |
| `S2_monthly_SBH_Rajaji_2016_2025.csv` | Rajaji NP, Uttarakhand | ✅ Ready |
| `S2_monthly_SBH_Manas_2016_2025.csv` | Manas NP, Assam | ✅ Ready |
| `S2_monthly_RBC_Hisar_2016_2025.csv` | Hisar, Haryana | ✅ Ready |
| `S2_monthly_RBC_Ludhiana_2016_2025.csv` | Ludhiana, Punjab | ✅ Ready |

**Fusion files — MODIS + Sentinel-2 combined (4 sites, best quality):**

| File | Site | Notes |
|------|------|-------|
| `FUSION_fused_MNG_Bhitarkanika_2016_2025.csv` | Bhitarkanika NP, Odisha | ✅ Better than MODIS-only |
| `FUSION_fused_MNG_Sundarbans_2016_2025.csv` | Sundarbans, WB | ✅ Better than MODIS-only |
| `FUSION_fused_MTF_Kedarnath_2016_2025.csv` | Kedarnath WS, Uttarakhand | ✅ Ready |
| `FUSION_fused_MTF_GreatHimal_2016_2025.csv` | Great Himalayan NP, HP | ✅ Ready |

> ⚠️ Alpine files (Spiti, Valley of Flowers) require pre-filtering to remove snow-covered months (NDVI < 0.01). See [data/ndvi/README.md](data/ndvi/README.md).

### Meteorology Files

| File | Site | Period | Source |
|------|------|--------|--------|
| `NASA_POWER_Tirupati_2017_2025.csv` | IIT Tirupati | Apr 2017–Mar 2025 | NASA POWER MERRA-2 |
| `NASA_POWER_Kaziranga_2017_2025.csv` | Kaziranga NP, Assam | 2017–2025 | NASA POWER MERRA-2 |

---

## ⚙️ How It Works

```
┌──────────────┐     ┌──────────────────┐
│  NDVI CSV    │     │  NASA POWER      │
│  (MODIS/S2)  │     │  Met CSV         │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       ▼                      ▼
┌──────────────────────────────────────────┐
│  1. PARSE & VALIDATE                     │
│     Auto-detect columns, skip headers    │
│     Flag missing years / bad values      │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  2. SMOOTH NDVI                          │
│     Savitzky-Golay filter (window 5–7,   │
│     poly order 3) — removes cloud spikes │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  3. EXTRACT PHENOLOGY EVENTS             │
│     SOS — threshold / rainfall / deriv   │
│     POS — annual NDVI maximum            │
│     EOS — threshold / drought / deriv    │
│     LOS — EOS_DOY − SOS_DOY             │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  4. BUILD TRAINING FEATURES              │
│     Pre-event windows (15–60 day lags)   │
│     Derive GDD, VPD, SPEI_proxy, DTR     │
│     Log-transform precipitation          │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  5. FEATURE SELECTION                    │
│     Pearson |r| ≥ 0.40 filter            │
│     Ecological priority order per event  │
│     Collinearity check (|r| < 0.85)      │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  6. FIT RIDGE REGRESSION                 │
│     Single feature (n < 10 seasons rule) │
│     α auto-tuned via RidgeCV             │
│     Leave-One-Out cross-validation       │
│     Outputs: R², MAE, equation           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  7. PREDICT & VISUALISE                  │
│     Enter new-year climate data          │
│     Predict SOS / POS / EOS dates        │
│     Correlation heatmaps, trend plots    │
└──────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
Indian-forest-phenology/
│
├── app/
│   └── universal_Indian_forest_phenology_assesment.py ← Main Streamlit app (1,726 lines)
│
├── data/
│   ├── ndvi/
│   │   ├── README.md               ← Column formats, per-site filter guide
│   │   ├── MODIS_NDVI_TDD_Tirupati_2016_2025.csv
│   │   ├── MODIS_NDVI_TDD_Mudumalai_2016_2025.csv
│   │   ├── MODIS_NDVI_TMD_Simlipal_2016_2025.csv
│   │   ├── MODIS_NDVI_TMD_Bastar_2016_2025.csv
│   │   ├── MODIS_NDVI_TWE_Agumbe_2016_2025.csv
│   │   ├── MODIS_NDVI_TWE_SilentValley_2016_2025.csv
│   │   ├── MODIS_NDVI_SHO_Mukurthi_2016_2025.csv
│   │   ├── MODIS_NDVI_SHO_Eravikulam_2016_2025.csv
│   │   ├── MODIS_NDVI_MNG_Bhitarkanika_2016_2025.csv
│   │   ├── MODIS_NDVI_MNG_Sundarbans_2016_2025.csv
│   │   ├── MODIS_NDVI_NEE_Kaziranga_2016_2025.csv
│   │   ├── MODIS_NDVI_NEE_Cherrapunji_2016_2025.csv
│   │   ├── MODIS_NDVI_KHF_Warangal_2016_2025.csv
│   │   ├── MODIS_NDVI_KHF_Cuttack_2016_2025.csv
│   │   ├── IIT_TIRUPATI_NDVI_2017-2025.csv
│   │   ├── S2_monthly_ALP_Spiti_2016_2025.csv
│   │   ├── S2_monthly_ALP_ValleyFlowers_2016_2025.csv
│   │   ├── S2_monthly_TTF_Jaisalmer_2016_2025.csv
│   │   ├── S2_monthly_TTF_Ranthambore_2016_2025.csv
│   │   ├── S2_monthly_TDE_Pichavaram_2016_2025.csv
│   │   ├── S2_monthly_TDE_PointCalimere_2016_2025.csv
│   │   ├── S2_monthly_SBH_Rajaji_2016_2025.csv
│   │   ├── S2_monthly_SBH_Manas_2016_2025.csv
│   │   ├── S2_monthly_RBC_Hisar_2016_2025.csv
│   │   ├── S2_monthly_RBC_Ludhiana_2016_2025.csv
│   │   ├── FUSION_fused_MNG_Bhitarkanika_2016_2025.csv
│   │   ├── FUSION_fused_MNG_Sundarbans_2016_2025.csv
│   │   ├── FUSION_fused_MTF_Kedarnath_2016_2025.csv
│   │   └── FUSION_fused_MTF_GreatHimal_2016_2025.csv
│   │
│   └── meteorology/
│       ├── README.md               ← Download instructions + parameter guide
│       ├── NASA_POWER_Tirupati_2017_2025.csv
│       └── NASA_POWER_Kaziranga_2017_2025.csv
│
├── scripts/
│   ├── gee_extract_modis_ndvi.js   ← GEE script for MODIS extraction
│   └── gee_extract_sentinel2_ndvi.js ← GEE script for Sentinel-2 extraction
│
├── docs/
│   └── user_guide.md               ← Full user guide with site-by-site settings
│
├── .streamlit/
│   └── config.toml                 ← Theme + server config for Streamlit Cloud
│
├── requirements.txt                ← Python dependencies
├── .gitignore
└── README.md                       ← This file
```

---

## 💻 Installation & Local Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/universal_Indian_forest_phenology_assesment.py
```

The app opens at `http://localhost:8501` in your browser.

---

## ☁️ Deploy on Streamlit Cloud

**Free, no credit card required. Takes ~3 minutes.**

1. **Fork or push this repo** to your GitHub account.

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. Click **"New app"**:
   - Repository: `shreejisharma/Indian-forest-phenology`
   - Branch: `main`
   - Main file path: `app/universal_Indian_forest_phenology_assesment.py`

4. Click **"Deploy"** — Streamlit Cloud installs `requirements.txt` automatically.

5. Your app is live at:  
   `https://shreejisharma-indian-forest-phenology.streamlit.app`

> **Tip:** Rename the URL in Streamlit Cloud dashboard to something like  
> `https://india-forest-phenology.streamlit.app`

---

## 🤗 Deploy on Hugging Face Spaces

**Alternative free hosting with more compute.**

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **"Create new Space"**

2. Settings:
   - SDK: **Streamlit**
   - Hardware: **CPU Basic (free)**
   - Visibility: **Public**

3. In the Space's **Files** tab, upload:
   - `app/universal_Indian_forest_phenology_assesment.py` → rename to `app.py` when using Hugging Face (it requires app.py at root)
   - `requirements.txt`
   - All files from `data/ndvi/` and `data/meteorology/`

4. Add a `README.md` at the root with:
   ```yaml
   ---
   title: Indian Forest Phenology Predictor
   emoji: 🌲
   colorFrom: green
   colorTo: blue
   sdk: streamlit
   sdk_version: 1.32.0
   app_file: app.py
   pinned: false
   ---
   ```

5. The Space builds automatically. Live in ~2 minutes.

---

## 📄 Data Format Guide

### NDVI CSV (required)

The app auto-detects the date and NDVI columns — column names do not need to match exactly.

```csv
date,NDVI
2016-01-01,0.423
2016-01-17,0.448
2016-02-02,0.461
```

**Accepted date column names:** `date`, `dates`, `time`, `datetime`, `Date`, `DATE`  
**Accepted NDVI column names:** `ndvi`, `NDVI`, `ndvi_value`, `value`, `evi`  
**Accepted date formats:** `YYYY-MM-DD`, `DD-MM-YYYY`, `MM/DD/YYYY`, `YYYYMMDD`

### NASA POWER Met CSV (required)

Download from [power.larc.nasa.gov/data-access-viewer/](https://power.larc.nasa.gov/data-access-viewer/).  
The app **automatically skips** the multi-line header block — upload the raw downloaded file.

```
-BEGIN HEADER-
NASA/POWER Source Native Resolution Daily Data
...
-END HEADER-
YEAR,DOY,T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,GWETTOP,GWETROOT,WS2M
2017,1,18.42,12.10,26.80,0.00,62.4,0.31,0.55,2.1
2017,2,19.10,13.20,27.40,0.00,58.1,0.30,0.54,1.9
```

---

## 🌡️ Recommended NASA POWER Parameters

When downloading from NASA POWER, select these parameters for best model performance:

| Parameter | Description | Critical for |
|-----------|-------------|-------------|
| `T2M` | Mean temperature at 2m (°C) | All models |
| `T2M_MIN` | Min temperature at 2m (°C) | SOS (pre-monsoon warmth) |
| `T2M_MAX` | Max temperature at 2m (°C) | Heat stress, DTR |
| `PRECTOTCORR` | Precipitation corrected (mm/day) | SOS monsoon trigger |
| `RH2M` | Relative humidity at 2m (%) | Evergreen SOS |
| `GWETTOP` | Surface soil wetness (0–1) | SOS confirmation |
| `GWETROOT` | Root zone soil wetness (0–1) | EOS (drought onset) |
| `ALLSKY_SFC_SW_DWN` | Solar radiation (MJ/m²/day) | **POS** — very important |

---

## ⚙️ App Settings Quick Reference

| Site | Forest Type to Select | Window | SOS Threshold | Expected Peak |
|------|-----------------------|--------|---------------|---------------|
| Tirupati | Tropical Dry Deciduous — Monsoon | Jun–May | 20–25% | October |
| Mudumalai | Tropical Dry Deciduous — Monsoon | Jun–May | 20–25% | November |
| Simlipal | Tropical Moist Deciduous — Monsoon | Jun–May | 20–25% | September |
| Bastar | Tropical Moist Deciduous — Monsoon | Jun–May | 20–25% | October |
| Agumbe | Tropical Wet Evergreen / Semi-Evergreen | Jan–Dec | 15–18% | October |
| Mukurthi | Shola Forest — Southern Montane | Jan–Dec | 15–18% | January |
| Bhitarkanika | Mangrove Forest | Jan–Dec | 15–18% | September |
| Warangal | Kharif / Summer Crop | Jun–Oct | 20–25% | September |
| Spiti *(filtered)* | Alpine / Subalpine | May–Oct | 25–30% | July–Aug |
| Valley of Flowers *(filtered)* | Alpine / Subalpine | May–Oct | 25–30% | Aug–Sep |
| Jaisalmer | Tropical Thorn Forest / Scrub | Jun–May | **10–12%** | October |

---

## 🤖 Model Architecture

```
Feature Selection
─────────────────
• All met parameters + derived features computed per site-season
• Pearson |r| ≥ 0.40 threshold (eliminates noise features)
• Ecological priority order per event (T2M_MIN → PRECTOTCORR for SOS)
• Collinearity filter: if two features |r| > 0.85, keep higher-priority

Ridge Regression
────────────────
• L2 regularisation: prevents overfitting with n = 8–10 seasons
• α tuned via RidgeCV on log-spaced grid [0.01 … 5000]
• Single best feature per event (n < 10 rule — avoids spurious multi-variate fits)
• StandardScaler applied inside Pipeline

Cross-Validation
────────────────
• Leave-One-Out (LOO) — gives unbiased R² for small samples
• Reported R² = LOO R² (not training R²)
• Also reports MAE in days

R² Interpretation
─────────────────
  > 0.70  →  Strong — reliable prediction
  0.40–0.70  →  Moderate — adequate for trend analysis
  0.10–0.40  →  Weak — relative comparisons only
  < 0  →  No valid feature found; prediction = mean DOY
```

---

## 📊 Results Summary

Preliminary results across 12 sites (2016–2025, LOO cross-validated):

| Site | Event | Best Predictor | R² (LOO) | MAE (days) |
|------|-------|---------------|----------|------------|
| Tirupati | SOS | T2M_MIN + PRECTOTCORR | ~0.65 | ~8 |
| Tirupati | POS | GDD_cum | ~0.72 | ~6 |
| Mudumalai | SOS | PRECTOTCORR | ~0.58 | ~10 |
| Simlipal | SOS | PRECTOTCORR | ~0.55 | ~12 |
| Bastar | POS | GDD_cum | ~0.68 | ~7 |
| Agumbe | SOS | RH2M | ~0.42 | ~14 |
| Mukurthi | SOS | ALLSKY_SFC_SW_DWN | ~0.45 | ~12 |
| Bhitarkanika | POS | PRECTOTCORR | ~0.38 | ~15 |
| Warangal | SOS | PRECTOTCORR | ~0.78 | ~5 |
| Valley of Flowers | SOS | T2M_MIN | ~0.62 | ~9 |

> Results are indicative. Actual R² depends on site-specific settings, season window selection, and threshold values. See app → Methods tab for full technical details.

---

## ⚠️ Known Limitations

1. **Small sample size (n = 8–10 seasons)** — LOO R² is reported, but all models should be interpreted cautiously. 15+ seasons strongly recommended for publication.

2. **Kaziranga** — Now included as `MODIS_NDVI_NEE_Kaziranga_2016_2025.csv` using MODIS MOD13Q1 250m, which can see through monsoon clouds via 16-day compositing. Use forest type **NE India Moist Evergreen** with a **Jun–May** window.

3. **Alpine files (2016–2017)** — Spiti and Valley of Flowers have incomplete Jun–Sep data before 2018. Pre-filter before loading (see [data/ndvi/README.md](data/ndvi/README.md)).

4. **Jaisalmer thorn scrub** — Very low NDVI amplitude (0.19). Standard SOS thresholds will fail. Use 10–12% threshold in app settings.

5. **Missing solar radiation** — The Tirupati met file does not include `ALLSKY_SFC_SW_DWN`. This limits POS model quality. Re-download from NASA POWER with solar radiation included.

---

## 📖 Citation

If you use this code or dataset in your work, please cite:

```bibtex
@software{indian_forest_phenology_2025,
  title     = {Indian Forest Phenology Predictor},
  author    = {Shreya Sharma},
  email     = {shreeji500sharma@gmail.com},
  year      = {2025},
  url       = {https://github.com/shreejisharma/Indian-forest-phenology},
  note      = {Streamlit application for phenology extraction and prediction
               across 11 Indian forest types using MODIS/Sentinel-2 NDVI
               and NASA POWER meteorological data}
}
```

Forest type classification follows:
> Champion, H.G. & Seth, S.K. (1968). *A Revised Survey of the Forest Types of India*. Manager of Publications, Delhi.

NDVI data sources:
> Didan, K. (2021). MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. NASA EOSDIS Land Processes DAAC.

Meteorological data:
> Stackhouse, P.W. et al. (2019). NASA/POWER CERES/MERRA-2 Daily 0.5° × 0.625°. NASA Langley Research Center.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.  
Data files are for academic research use only.

---

<p align="center">
  Built with ❤️ using Streamlit · MODIS · NASA POWER · scikit-learn
</p>
