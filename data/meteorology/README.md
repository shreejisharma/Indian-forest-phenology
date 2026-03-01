# Meteorology Data Files

NASA POWER daily meteorological data for all study sites.  
Source: [power.larc.nasa.gov](https://power.larc.nasa.gov/data-access-viewer/) — MERRA-2 reanalysis

---

## Files Included in This Repository

| File | Site | Coordinates | Period |
|------|------|-------------|--------|
| `NASA_POWER_Tirupati_2017_2025.csv` | IIT Tirupati / Tirupati Forest | 13.71°N, 79.58°E | Apr 2017 – Mar 2025 |
| `NASA_POWER_Kaziranga_2017_2025.csv` | Kaziranga NP, Assam | 26.58°N, 93.17°E | 2017 – 2025 |

---

## Download Met Data for All 12 Sites

The remaining 10 sites are not included due to file size. Use the direct API links below — each download takes ~30 seconds.

| NDVI File | Site | Coordinates | One-Click Download |
|-----------|------|-------------|-------------------|
| `MODIS_NDVI_TDD_Tirupati_*` | Tirupati, AP | 13.71°N 79.58°E | ✅ included |
| `MODIS_NDVI_TDD_Mudumalai_*` | Mudumalai, TN | 11.57°N 76.63°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=76.63&latitude=11.57&start=20170101&end=20241231&format=CSV) |
| `MODIS_NDVI_TMD_Simlipal_*` | Simlipal, Odisha | 21.65°N 86.50°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=86.50&latitude=21.65&start=20170101&end=20241231&format=CSV) |
| `MODIS_NDVI_TMD_Bastar_*` | Bastar, CG | 19.00°N 80.75°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=80.75&latitude=19.00&start=20170101&end=20241231&format=CSV) |
| `MODIS_NDVI_TWE_Agumbe_*` | Agumbe, Karnataka | 13.51°N 75.10°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=75.10&latitude=13.51&start=20170101&end=20241231&format=CSV) |
| `MODIS_NDVI_SHO_Mukurthi_*` | Mukurthi, TN | 11.22°N 76.52°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=76.52&latitude=11.22&start=20170101&end=20241231&format=CSV) |
| `MODIS_NDVI_MNG_Bhitarkanika_*` | Bhitarkanika, Odisha | 20.75°N 86.88°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=86.88&latitude=20.75&start=20170101&end=20241231&format=CSV) |
| `MODIS_NDVI_KHF_Warangal_*` | Warangal, Telangana | 18.00°N 79.58°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=79.58&latitude=18.00&start=20170101&end=20241231&format=CSV) |
| `S2_monthly_ALP_Spiti_*` | Spiti Valley, HP | 31.97°N 78.07°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=78.07&latitude=31.97&start=20170101&end=20241231&format=CSV) |
| `S2_monthly_ALP_ValleyFlowers_*` | Valley of Flowers, UK | 30.73°N 79.60°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=79.60&latitude=30.73&start=20170101&end=20241231&format=CSV) |
| `S2_monthly_TTF_Jaisalmer_*` | Desert NP, Rajasthan | 27.10°N 70.90°E | [Download CSV](https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,GWETTOP,GWETROOT,ALLSKY_SFC_SW_DWN,WS2M&community=AG&longitude=70.90&latitude=27.10&start=20170101&end=20241231&format=CSV) |

> Click the **Download CSV** link → file opens in browser → Save As → upload to app alongside the matching NDVI file.

---

## How to Download Manually from NASA POWER

1. Go to [power.larc.nasa.gov/data-access-viewer/](https://power.larc.nasa.gov/data-access-viewer/)
2. Select **Temporal Average: Daily**
3. Select **Community: Agroclimatology (AG)**
4. Enter your site **Latitude** and **Longitude** (see table above)
5. Set **date range: 2017-01-01 to 2024-12-31**
6. Select these parameters: `T2M`, `T2M_MIN`, `T2M_MAX`, `PRECTOTCORR`, `RH2M`, `GWETTOP`, `GWETROOT`, `ALLSKY_SFC_SW_DWN`, `WS2M`
7. Download as **CSV**
8. Upload directly — the app **auto-skips** the NASA POWER multi-line header block

---

## Parameters Used

| Parameter | Unit | Used for |
|-----------|------|----------|
| `T2M` | °C | Mean temperature |
| `T2M_MIN` | °C | Pre-monsoon warming → primary SOS driver |
| `T2M_MAX` | °C | Heat stress, DTR calculation |
| `PRECTOTCORR` | mm/day | Monsoon onset → SOS trigger |
| `RH2M` | % | Relative humidity → evergreen SOS |
| `GWETTOP` | 0–1 | Surface soil moisture |
| `GWETROOT` | 0–1 | Root zone soil moisture → EOS driver |
| `ALLSKY_SFC_SW_DWN` | MJ/m²/day | Solar radiation → **critical for POS** |
| `WS2M` | m/s | Wind speed (desiccation — low priority) |

### Derived Features (computed automatically by the app)

| Feature | Formula | Used for |
|---------|---------|---------|
| `GDD_5` | max(T2M − 5, 0) | Growing degree days base 5°C |
| `GDD_10` | max(T2M − 10, 0) | Growing degree days base 10°C |
| `GDD_cum` | cumulative GDD from season start | POS timing |
| `DTR` | T2M_MAX − T2M_MIN | Diurnal temperature range |
| `VPD` | vapour pressure deficit | Drought stress |
| `SPEI_proxy` | PRECTOTCORR − estimated PET | Drought index |
| `log_precip` | log(PRECTOTCORR + 0.1) | Log-transformed rainfall |
