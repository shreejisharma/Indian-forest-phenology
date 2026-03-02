# Indian Forest Phenology Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/MODIS-MOD13Q1%20250m-green" />
  <img src="https://img.shields.io/badge/Sentinel--2-SR%2010m-blue" />
  <img src="https://img.shields.io/badge/Forest%20Types-11-brightgreen" />
  <img src="https://img.shields.io/badge/Sites-20%2B-orange" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A Streamlit web application for extracting and predicting phenological events — Start of Season (SOS), Peak of Season (POS), and End of Season (EOS) — across 11 Indian forest types, using MODIS and Sentinel-2 NDVI time series paired with NASA POWER meteorological data.

**[▶ Open Live App](https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/)**

---

## Overview

Indian forest phenology is strongly driven by the southwest monsoon, yet the relationship between climate variables and phenological timing varies significantly across forest types — from teak-dominated dry deciduous forests of the Deccan Plateau to alpine meadows in Spiti Valley. Existing phenology tools are largely designed for temperate ecosystems and do not accommodate India's diverse monsoon-driven vegetation cycles.

This project addresses that gap by building a **universal phenology predictor** that:

- Extracts SOS, POS, and EOS dates from 8–10 years of NDVI time series (2016–2025) using ecologically appropriate methods for each forest type
- Fits Ridge Regression models with Leave-One-Out cross-validation to relate phenological events to lagged meteorological predictors
- Provides a web interface for uploading site data, inspecting model outputs, and generating predictions for future years

The tool covers **11 forest types** from the Champion & Seth (1968) classification, validated at **20+ sites** across the Western Ghats, Eastern Ghats, Himalayas, NE India, and the Thar Desert.

---

## Live Demo

**[https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/](https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/)**

Upload any NDVI CSV and NASA POWER met file to run the full pipeline in your browser — no local installation required.

---

## Table of Contents

- [Features](#features)
- [Forest Types](#forest-types)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Deployment](#deployment)
- [Data Format](#data-format)
- [Results](#results)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Features

| Feature | Description |
|---------|-------------|
| Universal CSV parser | Detects date and NDVI columns automatically; skips NASA POWER header block |
| 11 forest type configurations | Season windows, NDVI thresholds, and ecological driver priorities pre-set per type |
| Three SOS detection methods | NDVI threshold / first sustained rainfall (monsoon forests) / maximum rate-of-change (alpine) |
| Three EOS detection methods | NDVI threshold / maximum decline rate / dry season onset |
| Savitzky-Golay smoothing | Removes cloud-contamination spikes prior to event extraction |
| Feature selection | Pearson \|r\| ≥ 0.40 filter with ecological priority ordering and collinearity check |
| Ridge Regression | L2 regularisation with RidgeCV alpha tuning on log-spaced grid |
| Leave-One-Out cross-validation | Unbiased R² and MAE estimates for small sample sizes (n = 8–10 seasons) |
| Correlation explorer | Feature–event correlation tables for all meteorological predictors |
| Prediction interface | Enter climate values for any future year to obtain SOS/POS/EOS date estimates |

---

## Forest Types

| # | Forest Type | Season Window | Key Sites | Typical NDVI Amplitude |
|---|-------------|--------------|-----------|------------------------|
| 1 | Tropical Dry Deciduous | Jun–May | Tirupati, Mudumalai | 0.45–0.55 |
| 2 | Tropical Moist Deciduous | Jun–May | Simlipal, Bastar | 0.45–0.52 |
| 3 | Tropical Wet Evergreen / Semi-Evergreen | Jan–Dec | Agumbe, Silent Valley | 0.25–0.35 |
| 4 | Tropical Dry Evergreen | Jan–Dec | Pichavaram, Point Calimere | 0.20–0.35 |
| 5 | Tropical Thorn Forest / Scrub | Jun–May | Jaisalmer, Ranthambore | 0.10–0.20 |
| 6 | Subtropical Broadleaved Hill Forest | Apr–Mar | Rajaji NP, Manas NP | 0.30–0.45 |
| 7 | Montane Temperate Forest | Apr–Nov | Kedarnath, Great Himalayan NP | 0.35–0.55 |
| 8 | Alpine / Subalpine | May–Oct | Spiti Valley, Valley of Flowers | 0.15–0.85 |
| 9 | Shola Forest — Southern Montane | Jan–Dec | Mukurthi NP, Eravikulam | 0.25–0.35 |
| 10 | Mangrove Forest | Jan–Dec | Bhitarkanika, Sundarbans | 0.20–0.30 |
| 11 | NE India Moist Evergreen | Jan–Dec | Kaziranga NP, Cherrapunji | 0.30–0.45 |

---

## Dataset

### NDVI Files — 32 files, 20+ sites, 2016–2025

**MODIS MOD13Q1 250m (15 files)**

| File | Site | Usable Seasons |
|------|------|---------------|
| `MODIS_NDVI_TDD_Tirupati_2016_2025.csv` | Tirupati, AP | 10/10 |
| `MODIS_NDVI_TDD_Mudumalai_2016_2025.csv` | Mudumalai, TN | 10/10 |
| `MODIS_NDVI_TMD_Simlipal_2016_2025.csv` | Simlipal, Odisha | 10/10 |
| `MODIS_NDVI_TMD_Bastar_2016_2025.csv` | Bastar, CG | 10/10 |
| `MODIS_NDVI_TWE_Agumbe_2016_2025.csv` | Agumbe, Karnataka | 10/10 |
| `MODIS_NDVI_TWE_SilentValley_2016_2025.csv` | Silent Valley NP, Kerala | 10/10 |
| `MODIS_NDVI_SHO_Mukurthi_2016_2025.csv` | Mukurthi NP, TN | 10/10 |
| `MODIS_NDVI_SHO_Eravikulam_2016_2025.csv` | Eravikulam NP, Kerala | 10/10 |
| `MODIS_NDVI_MNG_Bhitarkanika_2016_2025.csv` | Bhitarkanika NP, Odisha | 10/10 |
| `MODIS_NDVI_MNG_Sundarbans_2016_2025.csv` | Sundarbans, WB | 10/10 |
| `MODIS_NDVI_NEE_Kaziranga_2016_2025.csv` | Kaziranga NP, Assam | 10/10 |
| `MODIS_NDVI_NEE_Cherrapunji_2016_2025.csv` | Cherrapunji, Meghalaya | 10/10 |
| `MODIS_NDVI_KHF_Warangal_2016_2025.csv` | Warangal, Telangana | 10/10 |
| `MODIS_NDVI_KHF_Cuttack_2016_2025.csv` | Cuttack, Odisha | 10/10 |
| `IIT_TIRUPATI_NDVI_2017-2025.csv` | IIT Tirupati Campus | 9/10 |

**Sentinel-2 SR Harmonized 10m — monthly composites (10 files)**

| File | Site | Notes |
|------|------|-------|
| `S2_monthly_ALP_Spiti_2016_2025.csv` | Spiti Valley, HP | Remove NDVI < 0.01 (snow) before loading |
| `S2_monthly_ALP_ValleyFlowers_2016_2025.csv` | Valley of Flowers, UK | Remove NDVI < 0.05; use 2018–2025 only |
| `S2_monthly_TTF_Jaisalmer_2016_2025.csv` | Desert NP, Rajasthan | Set SOS threshold 10–12% (low amplitude site) |
| `S2_monthly_TTF_Ranthambore_2016_2025.csv` | Ranthambore NP, Rajasthan | Load directly |
| `S2_monthly_TDE_Pichavaram_2016_2025.csv` | Pichavaram, TN | Load directly |
| `S2_monthly_TDE_PointCalimere_2016_2025.csv` | Point Calimere, TN | Load directly |
| `S2_monthly_SBH_Rajaji_2016_2025.csv` | Rajaji NP, Uttarakhand | Load directly |
| `S2_monthly_SBH_Manas_2016_2025.csv` | Manas NP, Assam | Load directly |
| `S2_monthly_RBC_Hisar_2016_2025.csv` | Hisar, Haryana | Load directly |
| `S2_monthly_RBC_Ludhiana_2016_2025.csv` | Ludhiana, Punjab | Load directly |

**MODIS + Sentinel-2 Fused — monthly (4 files, recommended for cloud-prone sites)**

| File | Site |
|------|------|
| `FUSION_fused_MNG_Bhitarkanika_2016_2025.csv` | Bhitarkanika NP, Odisha |
| `FUSION_fused_MNG_Sundarbans_2016_2025.csv` | Sundarbans, WB |
| `FUSION_fused_MTF_Kedarnath_2016_2025.csv` | Kedarnath WS, Uttarakhand |
| `FUSION_fused_MTF_GreatHimal_2016_2025.csv` | Great Himalayan NP, HP |

### Meteorological Files

Daily NASA POWER MERRA-2 reanalysis data. Two site files are included; download links for all remaining sites are in [`data/meteorology/README.md`](data/meteorology/README.md).

| File | Site | Period |
|------|------|--------|
| `NASA_POWER_Tirupati_2017_2025.csv` | IIT Tirupati, AP | 2017–2025 |
| `NASA_POWER_Kaziranga_2017_2025.csv` | Kaziranga NP, Assam | 2017–2025 |

---

## Methodology

```
NDVI CSV  +  NASA POWER Met CSV
         │
         ▼
1. Parse & validate
   — detect date/NDVI columns, skip header blocks, flag missing years
         │
         ▼
2. Smooth NDVI
   — Savitzky-Golay filter (adaptive window, poly order 3)
         │
         ▼
3. Extract phenological events
   — SOS: NDVI threshold / first sustained rainfall / max rate-of-change
   — POS: annual NDVI maximum (constrained by season window)
   — EOS: NDVI threshold / max decline rate / dry season onset
         │
         ▼
4. Build feature matrix
   — 15-day pre-event meteorological windows
   — Derived variables: GDD (base 5°C, 10°C), VPD, SPEI proxy, DTR, log(precip)
         │
         ▼
5. Feature selection
   — Pearson |r| ≥ 0.40 filter
   — Ecological priority order per event type
   — Collinearity removal (|r| > 0.85 between predictors)
         │
         ▼
6. Ridge Regression + LOO cross-validation
   — Alpha tuned via RidgeCV on [0.01 … 5000]
   — Leave-One-Out R² and MAE reported
         │
         ▼
7. Predict & visualise
   — Enter climate values → predict SOS / POS / EOS for any year
```

---

## Repository Structure

```
Indian-forest-phenology/
│
├── app/
│   └── universal_Indian_forest_phenology_assesment.py
│
├── data/
│   ├── ndvi/
│   │   ├── README.md
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
│       ├── README.md
│       ├── NASA_POWER_Tirupati_2017_2025.csv
│       └── NASA_POWER_Kaziranga_2017_2025.csv
│
├── scripts/
│   ├── gee_extract_modis_ndvi.js
│   └── gee_extract_sentinel2_ndvi.js
│
├── docs/
│   └── user_guide.md
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Requirements
- Python 3.9+
- pip

### Local setup

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/universal_Indian_forest_phenology_assesment.py
```

The app will open at `http://localhost:8501`.

---

## Deployment

### Streamlit Community Cloud

1. Fork this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and set:
   - Repository: `shreejisharma/Indian-forest-phenology`
   - Branch: `main`
   - Main file path: `app/universal_Indian_forest_phenology_assesment.py`
4. Click **Deploy**.

### Hugging Face Spaces

1. Create a new Space (SDK: Streamlit, CPU Basic).
2. Upload `app/universal_Indian_forest_phenology_assesment.py` renamed to `app.py`, `requirements.txt`, and the `data/` folder.
3. Add the standard Hugging Face `README.md` YAML header with `sdk: streamlit` and `app_file: app.py`.

---

## Data Format

### NDVI CSV

The app detects date and NDVI columns by name — extra columns are ignored.

```
date,NDVI
2016-01-01,0.423
2016-01-17,0.448
2016-02-02,0.461
```

Accepted date column names: `date`, `dates`, `time`, `datetime`  
Accepted NDVI column names: `ndvi`, `NDVI`, `ndvi_value`, `evi`

### NASA POWER Meteorological CSV

Download from [power.larc.nasa.gov/data-access-viewer/](https://power.larc.nasa.gov/data-access-viewer/) — the multi-line header block is skipped automatically.

Recommended parameters: `T2M`, `T2M_MIN`, `T2M_MAX`, `PRECTOTCORR`, `RH2M`, `GWETTOP`, `GWETROOT`, `ALLSKY_SFC_SW_DWN`, `WS2M`

### App settings by site

| Site | Forest Type | Window | SOS Threshold |
|------|-------------|--------|---------------|
| Tirupati, Mudumalai | Tropical Dry Deciduous | Jun–May | 20–25% |
| Simlipal, Bastar | Tropical Moist Deciduous | Jun–May | 20–25% |
| Agumbe, Silent Valley | Tropical Wet Evergreen | Jan–Dec | 15–18% |
| Mukurthi, Eravikulam | Shola Forest | Jan–Dec | 15–18% |
| Bhitarkanika, Sundarbans | Mangrove Forest | Jan–Dec | 15–18% |
| Kaziranga, Cherrapunji | NE India Moist Evergreen | Jun–May | 20% |
| Warangal, Cuttack | Kharif / Summer Crop | Jun–Oct | 20–25% |
| Hisar, Ludhiana | Rabi / Winter Crop | Nov–Apr | 20–25% |
| Spiti (filtered) | Alpine / Subalpine | May–Oct | 25–30% |
| Valley of Flowers (filtered) | Alpine / Subalpine | May–Oct | 25–30% |
| Jaisalmer | Tropical Thorn Scrub | Jun–May | 10–12% |

---

## Results

Preliminary LOO cross-validated results across 12 sites (2016–2025):

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

R² values are LOO cross-validated. Actual performance depends on site-specific settings, season window, and threshold selection. See the app's Methods tab for full details.

---

## Limitations

- **Sample size.** With 8–10 seasons per site, LOO R² estimates have wide confidence intervals. Results should be interpreted as exploratory; 15+ seasons are recommended before drawing firm conclusions.
- **Alpine sites (2016–2017).** Spiti Valley and Valley of Flowers lack usable Jun–Sep Sentinel-2 data before 2018 due to persistent snow cover. These years should be excluded before loading.
- **Jaisalmer.** NDVI amplitude is only ~0.19 across the season. Standard SOS thresholds will not detect the brief post-monsoon green flush; use 10–12% in the app settings.
- **Meteorological data coverage.** Only two NASA POWER files are included in this repository. Download links for all remaining sites are provided in [`data/meteorology/README.md`](data/meteorology/README.md).

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@software{sharma2025phenology,
  author  = {Sharma, Shreya},
  title   = {Indian Forest Phenology Predictor},
  year    = {2025},
  url     = {https://github.com/shreejisharma/Indian-forest-phenology}
}
```

Forest type classification:
> Champion, H.G. & Seth, S.K. (1968). *A Revised Survey of the Forest Types of India*. Manager of Publications, Delhi.

NDVI data:
> Didan, K. (2021). MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. NASA EOSDIS Land Processes DAAC.

Meteorological data:
> Stackhouse, P.W. et al. (2019). NASA/POWER CERES/MERRA-2 Daily 0.5° × 0.625°. NASA Langley Research Center.

---

## License

MIT — see [LICENSE](LICENSE) for details. Data files are for academic research use only.
