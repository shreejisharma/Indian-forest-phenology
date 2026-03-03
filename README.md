# Indian Forest Phenology Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/MODIS-MOD13Q1%20250m-green" />
  <img src="https://img.shields.io/badge/Sentinel--2-SR%2010m-blue" />
  <img src="https://img.shields.io/badge/Forest%20Types-11-brightgreen" />
  <img src="https://img.shields.io/badge/Sites-20%2B-orange" />
  <img src="https://img.shields.io/badge/Version-3.0-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A Streamlit web application for extracting and predicting phenological events — Start of Season (SOS), Peak of Season (POS), and End of Season (EOS) — across 11 Indian forest types, using MODIS and Sentinel-2 NDVI time series paired with NASA POWER meteorological data.

**[▶ Open Live App](https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app/)**

---

## What's New in v3

> **Only one file changed:** `app/universal_Indian_forest_phenology_assesment.py`
> No data files, scripts, requirements.txt, or other files were modified.
> To update GitHub, replace only that one file.

### Change 1 — Forest Type Sidebar: richer guidance for users

**Problem:** Users uploading data for a new site had no clear guidance on which of the 11 forest types to select.

**What changed (sidebar section, ~lines 1090–1160):**

| Old | New |
|-----|-----|
| Plain dropdown — type name only | Dropdown labels now include region hint e.g. `"🍂 Tropical Dry Deciduous [Tirupati, Deccan, Eastern Ghats]"` |
| Info box: rainfall, states, species, key drivers | Info card expanded — adds **primary monsoon trigger** (SW / NE / Both / Snowmelt) and **NDVI seasonality amplitude** (High / Medium / Low) |
| No selection guidance | New collapsible **"❓ Not sure which forest type?"** panel with a location→type table + 3-question decision tree (deciduousness → rainfall → elevation) |
| No explanation of why those met parameters matter | New green **"🔬 Why these met parameters?"** box — ecological sentence per type explaining the role of each key driver |

Example — new info box for Tropical Dry Deciduous:
```
☔ Primary trigger: SW Monsoon (Jun–Sep)
📊 NDVI seasonality: High amplitude
🔬 Why these met parameters?
   PRECTOTCORR (first monsoon rain → SOS) · GWETTOP (surface soil → leaf flush)
   · T2M_MIN (pre-monsoon warmth → bud break timing)
```

---

### Change 2 — Correlations Tab (Tab 2): end-to-end pipeline explanation added

**Problem:** Users could see the Pearson r charts but did not understand that ALL their uploaded met parameters were being used, or that the Pearson values shown directly select features for the Ridge model.

**What changed (Tab 2 header section, ~lines 1370–1430):**

| Old | New |
|-----|-----|
| One line: "\|r\| ≥ 0.40 required" | 5-step explainer box: CSV → raw params → derived features → Pearson r → Ridge model |
| No list of what was detected | Two-column display: **raw parameters from CSV** + **derived features auto-computed** |
| No per-parameter explanation | Expandable table: every detected parameter with its phenological role |
| Generic caption | Expanded caption: "coloured bars = entered Ridge regression · scatter plots ARE the relationships used in equations shown in Tab 1" |

The 5-step box shown at top of Tab 2:
```
Step 1 — Your NASA POWER CSV → ALL numeric columns used
Step 2 — Derived features computed: GDD_5, GDD_10, GDD_cum, DTR, VPD, SPEI_proxy, log_precip, MSI
Step 3 — Pearson r computed between every feature and SOS/POS/EOS DOY
Step 4 — |r| ≥ 0.40: coloured bars = entered model · grey = excluded
Step 5 — Ridge Regression fitted · scatter plots = the actual model relationships
```

---

### Change 3 — Forest Guide Tab (Tab 4): decision guide added before expanders

**Problem:** Users had to open all 11 expanders individually to find their forest type.

**What changed (Tab 4 header, ~lines 1711–1770):**

| Old | New |
|-----|-----|
| Heading then 11 collapsed expanders | 3-column decision guide at the top |
| No selection guidance | Q1 (rainfall & deciduousness) · Q2 (elevation) · Q3 (primary monsoon) tables |
| No explanation of what the forest type setting controls | Yellow box: forest type sets season window, NDVI threshold %, SOS method, and met driver priorities — Pearson r is computed fresh from user data |

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

- [What's New in v3](#whats-new-in-v3)
- [Features](#features)
- [How the Model Uses Your Met Parameters](#how-the-model-uses-your-met-parameters)
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
| Forest type decision guide *(v3)* | Location table + 3-question tree to help users select the correct type |
| Three SOS detection methods | NDVI threshold / first sustained rainfall (monsoon forests) / maximum rate-of-change (alpine) |
| Three EOS detection methods | NDVI threshold / maximum decline rate / dry season onset |
| Savitzky-Golay smoothing | Removes cloud-contamination spikes prior to event extraction |
| Feature selection | Pearson \|r\| ≥ 0.40 filter with ecological priority ordering and collinearity check |
| Ridge Regression | L2 regularisation with RidgeCV alpha tuning on log-spaced grid |
| Leave-One-Out cross-validation | Unbiased R² and MAE estimates for small sample sizes (n = 8–10 seasons) |
| Met parameter pipeline explainer *(v3)* | Step-by-step display showing how all raw + derived features feed the model |
| Correlation explorer | Feature–event correlation tables for all meteorological predictors |
| Prediction interface | Enter climate values for any future year to obtain SOS/POS/EOS date estimates |

---

## How the Model Uses Your Met Parameters

**Every numeric column in your NASA POWER CSV enters the pipeline.** Here is the exact flow:

```
Your NASA POWER CSV
│
├─ Raw columns detected automatically:
│    T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, RH2M,
│    GWETTOP, GWETROOT, GWETPROF, WS2M, ALLSKY_SFC_SW_DWN …
│
├─ Derived features computed automatically:
│    GDD_5        = max(T2M_avg − 5, 0)          daily GDD, base 5°C
│    GDD_10       = max(T2M_avg − 10, 0)         daily GDD, base 10°C
│    GDD_cum      = cumulative GDD_10 since season start
│    DTR          = T2M_MAX − T2M_MIN             diurnal temperature range
│    VPD          = vapour pressure deficit
│    SPEI_proxy   = PRECTOTCORR − PET             drought/wetness index
│    log_precip   = log(1 + PRECTOTCORR)
│    MSI          = moisture stress index
│
├─ Pearson r computed — every feature vs SOS/POS/EOS DOY
│    |r| ≥ 0.40 → coloured bar in Tab 2 → enters Ridge model
│    |r| < 0.40 → grey bar → excluded
│
└─ Ridge Regression fitted on selected feature(s)
     Scatter plots in Tab 2 = the exact relationships powering the model
```

**What each raw parameter contributes:**

| Parameter | Phenological role |
|-----------|------------------|
| `T2M` | Growing degree accumulation; controls POS timing |
| `T2M_MIN` | Pre-monsoon warmth threshold; drives SOS bud-break |
| `T2M_MAX` | Heat stress; linked to canopy maturity and EOS |
| `PRECTOTCORR` | Primary monsoon signal; first rainfall triggers SOS in deciduous forests |
| `RH2M` | Monsoon arrival proxy; SOS driver in wet evergreen forests |
| `GWETTOP` | Immediate surface moisture (0–5 cm); co-driver of leaf flush |
| `GWETROOT` | Deep root-zone moisture; delays EOS through dry season |
| `GWETPROF` | Full soil column moisture; linked to water use efficiency |
| `WS2M` | Desiccating wind speed; NE monsoon winds trigger senescence |
| `ALLSKY_SFC_SW_DWN` | Incoming solar radiation; dominant POS driver at montane/alpine sites |

---

## Forest Types

| # | Forest Type | Window | Primary Trigger | Key Sites | NDVI Amplitude |
|---|-------------|--------|-----------------|-----------|----------------|
| 1 | 🍂 Tropical Dry Deciduous | Jun–May | SW Monsoon | Tirupati, Mudumalai | 0.45–0.55 |
| 2 | 🌿 Tropical Moist Deciduous | Jun–May | SW Monsoon | Simlipal, Bastar | 0.45–0.52 |
| 3 | 🌲 Tropical Wet Evergreen | Jan–Dec | SW Monsoon | Agumbe, Silent Valley | 0.25–0.35 |
| 4 | 🌴 Tropical Dry Evergreen | Jan–Dec | NE Monsoon | Pichavaram, Point Calimere | 0.20–0.35 |
| 5 | 🌵 Tropical Thorn / Scrub | Jun–May | SW Monsoon | Jaisalmer, Ranthambore | 0.10–0.20 |
| 6 | 🌳 Subtropical Hill Forest | Apr–Mar | SW Monsoon | Rajaji NP, Manas NP | 0.30–0.45 |
| 7 | 🏔️ Montane Temperate | Apr–Nov | Snowmelt + SW | Kedarnath, Great Himalayan NP | 0.35–0.55 |
| 8 | ⛰️ Alpine / Subalpine | May–Oct | Snowmelt | Spiti Valley, Valley of Flowers | 0.15–0.85 |
| 9 | 🌫️ Shola — Southern Montane | Jan–Dec | SW + NE Monsoon | Mukurthi NP, Eravikulam | 0.25–0.35 |
| 10 | 🌊 Mangrove Forest | Jan–Dec | SW Monsoon | Bhitarkanika, Sundarbans | 0.20–0.30 |
| 11 | 🌿 NE India Moist Evergreen | Jan–Dec | SW Monsoon | Kaziranga NP, Cherrapunji | 0.30–0.45 |

### Quick location → type reference

| Your site | Select this type |
|-----------|-----------------|
| Tirupati, Eastern Ghats, Deccan | 🍂 Tropical Dry Deciduous |
| Sal forests — MP, CG, Odisha, Jharkhand | 🌿 Tropical Moist Deciduous |
| Western Ghats — Kerala, Karnataka | 🌲 Tropical Wet Evergreen |
| Tamil Nadu Coromandel coast | 🌴 Tropical Dry Evergreen |
| Rajasthan, Gujarat, semi-arid Deccan | 🌵 Tropical Thorn / Scrub |
| Himalayan foothills 500–1500m | 🌳 Subtropical Hill Forest |
| W Himalayas 1500–3000m | 🏔️ Montane Temperate |
| Himalayas > 3000m, Ladakh, Spiti | ⛰️ Alpine / Subalpine |
| Nilgiris, Munnar, Kodaikanal > 1500m | 🌫️ Shola Montane |
| Sundarbans, Bhitarkanika, Pichavaram | 🌊 Mangrove |
| Assam, Meghalaya, Manipur, Arunachal | 🌿 NE India Moist Evergreen |

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

### Meteorological Files

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
   — detect date/NDVI columns, skip header blocks
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
4. Build feature matrix from ALL uploaded met columns
   — 15-day pre-event windows (mean for temp/humidity, sum for precip/GDD)
   — Raw: T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, RH2M, GWETTOP,
          GWETROOT, GWETPROF, WS2M, ALLSKY_SFC_SW_DWN
   — Derived (auto): GDD_5, GDD_10, GDD_cum, DTR, VPD, SPEI_proxy,
                     log_precip, MSI
         │
         ▼
5. Feature selection (Pearson r screening)
   — Pearson |r| ≥ 0.40 filter — coloured bars in Tab 2
   — Ecological priority order per event (SOS / POS / EOS)
   — Collinearity removal (|r| > 0.85 between predictors)
         │
         ▼
6. Ridge Regression + LOO cross-validation
   — Alpha tuned via RidgeCV on [0.01 … 5000]
   — Leave-One-Out R² and MAE reported
         │
         ▼
7. Predict & visualise
   — Enter climate values → SOS / POS / EOS prediction for any year
```

---

## Repository Structure

```
Indian-forest-phenology/
│
├── app/
│   └── universal_Indian_forest_phenology_assesment.py   ← only file changed in v3
│
├── data/
│   ├── ndvi/          (32 NDVI CSV files — unchanged)
│   └── meteorology/   (2 met CSV files + README — unchanged)
│
├── scripts/
│   ├── gee_extract_modis_ndvi.js        (unchanged)
│   └── gee_extract_sentinel2_ndvi.js    (unchanged)
│
├── docs/
│   └── user_guide.md                    (unchanged)
│
├── .streamlit/config.toml               (unchanged)
├── requirements.txt                     (unchanged)
├── .gitignore                           (unchanged)
└── README.md                            ← updated in v3
```

---

## Installation

```bash
git clone https://github.com/shreejisharma/Indian-forest-phenology.git
cd Indian-forest-phenology
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/universal_Indian_forest_phenology_assesment.py
```

---

## Deployment

### Streamlit Community Cloud

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Set main file path: `app/universal_Indian_forest_phenology_assesment.py`
4. Click **Deploy**.

---

## Data Format

### NDVI CSV

```
date,NDVI
2016-01-01,0.423
2016-01-17,0.448
```

### NASA POWER Meteorological CSV

Download from [power.larc.nasa.gov](https://power.larc.nasa.gov/data-access-viewer/) — header block auto-skipped.  
Recommended parameters: `T2M`, `T2M_MIN`, `T2M_MAX`, `PRECTOTCORR`, `RH2M`, `GWETTOP`, `GWETROOT`, `ALLSKY_SFC_SW_DWN`, `WS2M`

---

## Results

LOO cross-validated R² across key sites:

| Site | SOS R² | POS R² | EOS R² | Notes |
|------|--------|--------|--------|-------|
| Tirupati (MODIS) | 0.23 | 0.50 | **0.69** | Best EOS predictor: T2M_MIN pre-EOS 30d |
| Mudumalai | ~0.58 | — | — | PRECTOTCORR dominant |
| Simlipal | ~0.55 | — | — | PRECTOTCORR dominant |
| Bastar | — | ~0.68 | — | GDD_cum dominant |
| Warangal | ~0.78 | — | — | Kharif crop; PRECTOTCORR |
| Valley of Flowers | ~0.62 | — | — | T2M_MIN dominant (alpine) |

> All R² values are **LOO cross-validated** — not in-sample. A training R² of 0.98 with n=8 and 2 predictors produces LOO R² ≈ −0.76 due to overfitting; this is why LOO R² is the only reported metric.

---

## Limitations

- **Sample size.** 8–10 seasons → LOO R² estimates have wide confidence intervals. 15+ seasons recommended for publication.
- **SOS predictability.** SW monsoon onset date is not in NASA POWER. Enter IMD onset DOY in Tab 1 to significantly improve SOS models.
- **Alpine sites.** Spiti and Valley of Flowers lack usable Sentinel-2 data before 2018 (persistent snow). Exclude those years before loading.
- **Jaisalmer.** NDVI amplitude ~0.19 — use 10–12% SOS threshold instead of the default 25–30%.

---

## Citation

```bibtex
@software{sharma2025phenology,
  author  = {Sharma, Shreya},
  title   = {Indian Forest Phenology Predictor},
  year    = {2025},
  url     = {https://github.com/shreejisharma/Indian-forest-phenology}
}
```

> Champion, H.G. & Seth, S.K. (1968). *A Revised Survey of the Forest Types of India*. Manager of Publications, Delhi.

> Didan, K. (2021). MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. NASA EOSDIS Land Processes DAAC.

> Stackhouse, P.W. et al. (2019). NASA/POWER CERES/MERRA-2 Daily 0.5° × 0.625°. NASA Langley Research Center.

---

## License

MIT — see [LICENSE](LICENSE) for details. Data files are for academic research use only.
