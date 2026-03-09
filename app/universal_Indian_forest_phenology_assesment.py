"""
🌲 UNIVERSAL INDIAN FOREST PHENOLOGY PREDICTOR 🌲
===================================================

Upload:
  1. NDVI CSV  — any CSV with Date + NDVI columns
  2. NASA POWER Met CSV — daily export, headers auto-skipped

Predicts SOS / POS / EOS / LOS for 11 Indian forest types using:
  • Pearson |r| >= 0.40 feature filter
  • Multi-feature Ridge (up to n−2 features; collinearity-filtered; all significant drivers shown)
  • Ridge Regression with auto-tuned alpha (RidgeCV)
  • Leave-One-Out cross-validation (honest R² for small samples)
  • Threshold-based phenology extraction (robust for multi-hump NDVI)
  • Optional IMD monsoon onset date as SOS predictor

pip install streamlit pandas numpy scipy scikit-learn matplotlib
streamlit run app/universal_Indian_forest_phenology_assesment.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from io import StringIO
import warnings
warnings.filterwarnings('ignore')


# ─── LOESS / LOWESS implementation (no external dependency) ──
def _loess_predict(x_train, y_train, x_new, frac=0.75):
    """
    Locally-weighted scatterplot smoothing (LOESS/LOWESS).
    Uses tricube kernel weights around each prediction point.
    Works with 1-D x arrays.
    """
    n = len(x_train)
    k = max(2, int(np.ceil(frac * n)))
    result = np.zeros(len(x_new))
    for i, xp in enumerate(x_new):
        dists = np.abs(x_train - xp)
        idx   = np.argsort(dists)[:k]
        d_max = dists[idx[-1]] + 1e-10
        u     = dists[idx] / d_max
        w     = np.maximum(0, (1 - u**3)**3)   # tricube
        if w.sum() < 1e-12:
            result[i] = np.mean(y_train)
        else:
            # Weighted least-squares line
            Xw = np.column_stack([np.ones(k), x_train[idx]]) * w[:, None]
            Yw = y_train[idx] * w
            try:
                coef, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
                result[i] = coef[0] + coef[1] * xp
            except Exception:
                result[i] = np.average(y_train[idx], weights=w)
    return result



st.set_page_config(
    page_title="🌲 Indian Forest Phenology",
    page_icon="🌲",
    layout="wide"
)

st.markdown("""
<style>
.main-header{font-size:2.3rem;color:#1B5E20;font-weight:bold;text-align:center;padding:18px 0 6px}
.sub-header{text-align:center;color:#388E3C;font-size:1.0rem;margin-bottom:18px;font-style:italic}
.metric-box{background:linear-gradient(135deg,#E8F5E9,#C8E6C9);padding:22px;border-radius:14px;
            text-align:center;box-shadow:0 3px 8px rgba(0,0,0,.10);margin:4px;border:1px solid #A5D6A7}
.metric-box h3{color:#1B5E20;margin:0 0 4px;font-size:0.92rem;font-weight:600}
.metric-box h1{color:#1B5E20;margin:0;font-size:2.0rem}
.metric-box p{color:#388E3C;margin:4px 0 0;font-size:0.80rem}
.info-box{background:#FFF9C4;padding:14px 18px;border-radius:10px;border-left:5px solid #F9A825;margin:10px 0}
.warn-box{background:#FFE0B2;padding:14px 18px;border-radius:10px;border-left:5px solid #FF9800;margin:8px 0}
.good-box{background:#C8E6C9;padding:14px 18px;border-radius:10px;border-left:5px solid #4CAF50;margin:8px 0}
.eq-box{background:#F3E5F5;padding:13px 16px;border-radius:10px;border-left:5px solid #9C27B0;
        font-family:monospace;font-size:0.83rem;margin:8px 0;word-break:break-all}
.fix-box{background:#E3F2FD;padding:14px 18px;border-radius:10px;border-left:5px solid #1976D2;margin:8px 0}
.upload-hint{background:#F1F8E9;padding:16px 20px;border-radius:12px;border:2px dashed #81C784;margin:16px 0}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────
MIN_CORR_THRESHOLD = 0.40
ALPHAS = [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000]

# Ecological feature priority per event — determines which features are PREFERRED
# when multiple features have similar |r|. Also controls collinearity-filtered selection order.
# GDD_cum excluded from SOS/EOS (it's a running cumulative — spuriously tracks time-of-year)
EVENT_PRIORITIES = {
    # SOS — monsoon forests: minimum temperature controls pre-monsoon warmth,
    # then first rainfall triggers leaf flush. GDD_cum excluded (spurious).
    'SOS': ['T2M_MIN', 'PRECTOTCORR', 'log_precip', 'RH2M', 'GWETTOP',
            'GWETROOT', 'SPEI_proxy', 'VPD', 'T2M', 'GDD_5', 'GDD_10',
            'WS2M', 'DTR', 'T2M_MAX'],

    # POS — peak canopy driven by accumulated heat+radiation after monsoon establishes.
    # GDD_cum (snapshot at event date) is legitimate here.
    'POS': ['PRECTOTCORR', 'log_precip', 'GDD_cum', 'GWETROOT', 'GWETTOP',
            'RH2M', 'GDD_5', 'GDD_10', 'T2M', 'T2M_MAX',
            'ALLSKY_SFC_SW_DWN', 'SPEI_proxy', 'VPD'],

    # EOS — senescence triggered by: soil moisture decline, drying winds, cold nights,
    # increased diurnal range, VPD stress. WS2M added (wind-driven desiccation).
    # GDD_cum excluded (still tracks time-of-year in post-monsoon period).
    'EOS': ['T2M_MIN', 'WS2M', 'DTR', 'T2M_RANGE', 'VPD',
            'GWETTOP', 'GWETROOT', 'MSI', 'SPEI_proxy',
            'T2M', 'log_precip', 'PRECTOTCORR', 'RH2M'],
}

# Features explicitly EXCLUDED from each event (even if |r| is high — ecologically spurious)
EVENT_EXCLUDE = {
    'SOS': {'GDD_cum'},           # Running cumulative = time-of-year proxy
    'POS': set(),                  # GDD_cum is legitimate for POS
    'EOS': {'GDD_cum'},           # Running cumulative = time-of-year proxy
}
SM_NAME = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─── FOREST SEASON CONFIGURATIONS — 11 forest types ──────────
SEASON_CONFIGS = {
    "Tropical Dry Deciduous — Monsoon (Jun-May)": {
        "start_month":6,"end_month":5,"min_days":150,"sos_base":None,
        "pos_constrain_end":12,"threshold_pct":0.30,"sos_method":"rainfall","icon":"🍂",
        "regions":"Tirupati, Deccan Plateau, Eastern Ghats, Vindhyas",
        "species":"Teak, Sal, Axlewood, Bamboo","states":"AP, Telangana, Karnataka, MP, UP, Bihar",
        "rainfall":"800-1200 mm","ndvi_peak":"Oct-Nov","key_drivers":"PRECTOTCORR, GWETTOP, T2M_MIN",
        "desc":"Largest forest type (35% India). SW Monsoon drives leaf flush Jun-Jul. Leaf fall Mar-Apr."},
    "Tropical Moist Deciduous — Monsoon (Jun-May)": {
        "start_month":6,"end_month":5,"min_days":150,"sos_base":None,
        "pos_constrain_end":12,"threshold_pct":0.25,"sos_method":"rainfall","icon":"🌿",
        "regions":"NE India, Western Ghats E slopes, Odisha, Central India",
        "species":"Sal (Shorea robusta), Teak, Laurel, Rosewood, Bamboo",
        "states":"Assam, Odisha, Jharkhand, MP, Chhattisgarh, NE States",
        "rainfall":"1000-2000 mm","ndvi_peak":"Sep-Oct","key_drivers":"RH2M, PRECTOTCORR, T2M, GWETROOT",
        "desc":"Sal-dominant moist forests. SOS Jun-Jul (monsoon flush)."},
    "Tropical Wet Evergreen / Semi-Evergreen (Jan-Dec)": {
        "start_month":1,"end_month":12,"min_days":200,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.20,"sos_method":"threshold","icon":"🌲",
        "regions":"Western Ghats, NE India, Andaman & Nicobar",
        "species":"Dipterocarp, Mesua, Calophyllum, Bamboo, Canes",
        "states":"Kerala, Karnataka, Tamil Nadu (W Ghats), Assam, Meghalaya",
        "rainfall":">2500 mm","ndvi_peak":"Oct-Dec","key_drivers":"RH2M, ALLSKY_SFC_SW_DWN, GWETROOT",
        "desc":"Nearly year-round green. Highest NDVI in India (0.65-0.82)."},
    "Tropical Dry Evergreen (Jan-Dec)": {
        "start_month":1,"end_month":12,"min_days":200,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.20,"sos_method":"threshold","icon":"🌴",
        "regions":"Tamil Nadu Coromandel Coast","species":"Manilkara, Memecylon, Diospyros, Eugenia",
        "states":"Tamil Nadu coastal strip","rainfall":"~1000 mm (NE monsoon)","ndvi_peak":"Dec-Jan",
        "key_drivers":"PRECTOTCORR (Oct-Dec), T2M",
        "desc":"Driven by NE monsoon. SOS Oct-Nov — opposite to most Indian forests."},
    "Tropical Thorn Forest / Scrub (Jun-May)": {
        "start_month":6,"end_month":5,"min_days":100,"sos_base":None,
        "pos_constrain_end":12,"threshold_pct":0.25,"sos_method":"rainfall","icon":"🌵",
        "regions":"Rajasthan, Gujarat, semi-arid Deccan",
        "species":"Khejri, Babul (Acacia), Euphorbia, Ziziphus",
        "states":"Rajasthan, Gujarat, Haryana, parts of MP & AP",
        "rainfall":"<700 mm","ndvi_peak":"Aug-Sep","key_drivers":"PRECTOTCORR, GWETTOP",
        "desc":"Very short growing season driven by monsoon pulse. Low base NDVI."},
    "Subtropical Broadleaved Hill Forest (Apr-Mar)": {
        "start_month":4,"end_month":3,"min_days":150,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.25,"sos_method":"threshold","icon":"🌳",
        "regions":"Himalayan foothills 500-1500m — Shiwaliks, NE hills",
        "species":"Oak (Quercus), Chestnut, Alder, Rhododendron, Sal",
        "states":"Uttarakhand foothills, Himachal, NE hill states",
        "rainfall":"1500-2500 mm","ndvi_peak":"Aug-Sep","key_drivers":"T2M_MIN, GDD_10, PRECTOTCORR",
        "desc":"Temperature and monsoon co-drive phenology. Leaf flush Apr-May."},
    "Montane Temperate Forest (Apr-Nov)": {
        "start_month":4,"end_month":11,"min_days":120,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.30,"sos_method":"derivative","icon":"🏔️",
        "regions":"Western Himalayas 1500-3000m",
        "species":"Oak, Chir Pine, Deodar, Rhododendron, Maple, Birch",
        "states":"Uttarakhand, Himachal Pradesh, J&K, Sikkim",
        "rainfall":"1000-2500 mm","ndvi_peak":"Jul-Aug","key_drivers":"T2M_MIN, GDD_10, ALLSKY_SFC_SW_DWN",
        "desc":"Snowmelt triggers SOS Apr-May. Short growing season Apr-Nov."},
    "Alpine / Subalpine Forest and Meadow (May-Oct)": {
        "start_month":5,"end_month":10,"min_days":80,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.35,"sos_method":"derivative","icon":"⛰️",
        "regions":"Himalaya >3000m — Ladakh, Spiti, Uttarakhand alpine",
        "species":"Juniper, Silver Fir, Spruce, Alpine meadows (Bugyals)",
        "states":"Ladakh, Spiti, Uttarakhand >3000m, Arunachal alpine",
        "rainfall":"300-800 mm (mostly snow)","ndvi_peak":"Jul-Aug","key_drivers":"T2M_MIN, ALLSKY_SFC_SW_DWN, GDD_5",
        "desc":"Very short ~5 month season. Snowmelt controls SOS. High temperature sensitivity."},
    "Shola Forest — Southern Montane (Jan-Dec)": {
        "start_month":1,"end_month":12,"min_days":200,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.15,"sos_method":"threshold","icon":"🌫️",
        "regions":"Nilgiris, Anamalais, Palani — Western Ghats >1500m",
        "species":"Michelia, Syzygium, Rhododendron, Elaeocarpus",
        "states":"Nilgiris/Kodaikanal (Tamil Nadu), Munnar (Kerala), Coorg (Karnataka)",
        "rainfall":"2000-5000 mm (two monsoons)","ndvi_peak":"Oct-Dec",
        "key_drivers":"RH2M, ALLSKY_SFC_SW_DWN, GWETROOT",
        "desc":"Receives both SW and NE monsoons. Nearly evergreen — very weak seasonality."},
    "Mangrove Forest (Jan-Dec)": {
        "start_month":1,"end_month":12,"min_days":200,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.15,"sos_method":"threshold","icon":"🌊",
        "regions":"Sundarbans, Bhitarkanika, Pichavaram, Andaman & Nicobar",
        "species":"Rhizophora, Avicennia, Bruguiera, Sonneratia",
        "states":"West Bengal, Odisha, Tamil Nadu, Andaman & Nicobar",
        "rainfall":">1500 mm","ndvi_peak":"Sep-Oct","key_drivers":"PRECTOTCORR, RH2M, T2M",
        "desc":"Tidal ecosystem — nearly evergreen. Monsoon drives canopy flush."},
    "NE India Moist Evergreen (Jan-Dec)": {
        "start_month":1,"end_month":12,"min_days":200,"sos_base":None,
        "pos_constrain_end":None,"threshold_pct":0.20,"sos_method":"threshold","icon":"🌿",
        "regions":"Assam, Meghalaya, Nagaland, Manipur, Mizoram, Arunachal",
        "species":"Dipterocarp, Bamboo, Cane, Orchids — highest NDVI in India",
        "states":"All NE states + sub-Himalayan W Bengal",
        "rainfall":"2000-4000 mm","ndvi_peak":"Aug-Sep","key_drivers":"PRECTOTCORR, RH2M, T2M",
        "desc":"Highest mean NDVI in India (0.69-0.72). Two peaks: pre-monsoon + monsoon."},
}

# ─── PARSERS ─────────────────────────────────────────────────
def parse_nasa_power(uploaded_file):
    try:
        raw = uploaded_file.read().decode('utf-8', errors='replace')
        lines = raw.splitlines()
        skip_to, in_hdr = 0, False
        for i, ln in enumerate(lines):
            s = ln.strip().upper()
            if '-BEGIN HEADER-' in s: in_hdr = True
            if in_hdr and '-END HEADER-' in s: skip_to = i + 1; break
        if skip_to == 0:
            for i, ln in enumerate(lines):
                up = ln.strip().upper()
                if up.startswith('YEAR') or up.startswith('LON'): skip_to = i; break
        df = pd.read_csv(StringIO('\n'.join(lines[skip_to:])))
        df.columns = [c.strip() for c in df.columns]
        df.replace([-999,-999.0,-99,-99.0,-9999,-9999.0], np.nan, inplace=True)
        if 'Date' not in df.columns:
            if {'YEAR','DOY'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(
                    df['YEAR'].astype(str) + df['DOY'].astype(str).str.zfill(3),
                    format='%Y%j', errors='coerce')
            elif {'YEAR','MO','DY'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(
                    df['YEAR'].astype(str)+'-'+df['MO'].astype(str).str.zfill(2)+'-'+
                    df['DY'].astype(str).str.zfill(2), errors='coerce')
            else:
                return None, [], 'Cannot build Date — need YEAR+DOY or YEAR+MO+DY'
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        excl = {'YEAR','MO','DY','DOY','LON','LAT','ELEV','Date'}
        params = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
        return df, params, None
    except Exception as e:
        return None, [], str(e)


def _parse_date_robust(series, doy_series=None):
    """
    Robust date parser. Handles:
      YYYY-MM-DD  (ISO)
      DD-MM-YYYY  (GEE S2 monthly export — e.g. 01-02-2018 = Feb 1, NOT Jan 2)
      MM/DD/YYYY, DD/MM/YYYY, and other common formats.

    Strategy:
      1. If 'doy_series' is provided (GEE exports include a 'doy' column), validate
         which parsing matches — 100% reliable disambiguation.
      2. Without DOY: compare month-spread of dayfirst vs default — pick richer spread.
      3. Fall through to best-of-formats by count.
    """
    if len(series) == 0:
        return pd.to_datetime(series, errors='coerce')

    parsed_default  = pd.to_datetime(series, errors='coerce')
    parsed_dayfirst = pd.to_datetime(series, dayfirst=True, errors='coerce')

    # Strategy 1: Use DOY reference column if available
    if doy_series is not None:
        doy_ref      = pd.to_numeric(doy_series, errors='coerce')
        default_doy  = parsed_default.apply(
            lambda d: d.timetuple().tm_yday if pd.notna(d) else np.nan)
        dayfirst_doy = parsed_dayfirst.apply(
            lambda d: d.timetuple().tm_yday if pd.notna(d) else np.nan)
        if (dayfirst_doy == doy_ref).sum() > (default_doy == doy_ref).sum():
            return parsed_dayfirst
        return parsed_default

    # Strategy 2: richer month distribution wins
    n_m_default  = parsed_default.dropna().dt.month.nunique()
    n_m_dayfirst = parsed_dayfirst.dropna().dt.month.nunique()
    if n_m_dayfirst > n_m_default:
        return parsed_dayfirst

    # Strategy 3: explicit format — best parse-count
    if parsed_default.notna().sum() / max(len(series), 1) > 0.85:
        return parsed_default
    best, best_n = parsed_default, parsed_default.notna().sum()
    for fmt in ['%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y', '%m-%d-%Y',
                '%Y/%m/%d', '%d-%b-%Y', '%d %b %Y']:
        try:
            attempt = pd.to_datetime(series, format=fmt, errors='coerce')
            if attempt.notna().sum() > best_n:
                best, best_n = attempt, attempt.notna().sum()
        except Exception:
            continue
    return best


def parse_ndvi(uploaded_file):
    """
    Universal NDVI parser. Handles:
      • Any CSV with a date column and an NDVI column
      • Multi-site GEE exports — filters by site_key if multiple sites present
      • DD-MM-YYYY date format (GEE S2 monthly exports)
      • Low-frequency data (monthly, 16-day) — works even with sparse obs
    """
    try:
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        df = pd.read_csv(StringIO(raw_bytes.decode('utf-8', errors='replace')))
        df.columns = [c.strip() for c in df.columns]

        # ── Find date column ─────────────────────────────────
        date_col = next((c for c in df.columns
                         if c.lower() in ['date','dates','time','datetime']), None)
        if not date_col:
            return None, "No Date column found. Expected column named: date / dates / time / datetime"

        # ── Find NDVI column ─────────────────────────────────
        ndvi_col = next((c for c in df.columns
                         if c.lower() in ['ndvi','ndvi_value','value','index','evi']), None)
        if not ndvi_col:
            return None, "No NDVI column found. Expected column named: ndvi / ndvi_value / value / evi"

        # ── Multi-site filter: if site_key column exists and has >1 site, let user pick ──
        site_info = None
        if 'site_key' in df.columns and df['site_key'].nunique() > 1:
            site_info = sorted(df['site_key'].dropna().unique().tolist())
        elif 'site_label' in df.columns and df['site_label'].nunique() > 1:
            site_info = sorted(df['site_label'].dropna().unique().tolist())

        if site_info:
            # Return special tuple so main() can show a selectbox
            return ('MULTI_SITE', site_info, df, date_col, ndvi_col), None

        # ── Parse dates robustly (use 'doy' column to disambiguate if present) ──
        doy_col = df['doy'] if 'doy' in df.columns else None
        df = df.rename(columns={date_col: 'Date', ndvi_col: 'NDVI'})
        df['Date'] = _parse_date_robust(df['Date'].astype(str), doy_series=doy_col)
        df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce')
        result = (df.dropna(subset=['Date', 'NDVI'])[['Date', 'NDVI']]
                    .sort_values('Date').reset_index(drop=True))
        if len(result) == 0:
            return None, "No valid rows after date/NDVI parsing. Check your date format."
        return result, None

    except Exception as e:
        return None, str(e)


def _filter_ndvi_site(df, date_col, ndvi_col, site_key):
    """Filter a multi-site dataframe to one site and return clean NDVI df."""
    key_col = 'site_key' if 'site_key' in df.columns else 'site_label'
    sub = df[df[key_col] == site_key].copy()
    sub = sub.rename(columns={date_col: 'Date', ndvi_col: 'NDVI'})
    doy_col = sub['doy'] if 'doy' in sub.columns else None
    sub['Date'] = _parse_date_robust(sub['Date'].astype(str), doy_series=doy_col)
    sub['NDVI'] = pd.to_numeric(sub['NDVI'], errors='coerce')
    return (sub.dropna(subset=['Date', 'NDVI'])[['Date', 'NDVI']]
               .sort_values('Date').reset_index(drop=True))


# ─── DERIVED MET FEATURES ────────────────────────────────────
def add_derived_features(met_df, season_start_month=4):
    """Add derived meteorological features. GDD accumulation resets at season_start_month."""
    df = met_df.copy(); cols = df.columns.tolist()
    tmin = next((c for c in ['T2M_MIN','TMIN'] if c in cols), None)
    tmax = next((c for c in ['T2M_MAX','TMAX'] if c in cols), None)
    tmn  = next((c for c in ['T2M','TMEAN','TEMP'] if c in cols), None)
    rh   = next((c for c in ['RH2M','RH','RHUM'] if c in cols), None)
    prec = next((c for c in ['PRECTOTCORR','PRECTOT','PRECIP','RAIN'] if c in cols), None)
    sm   = next((c for c in ['GWETTOP','GWETROOT','GWETPROF'] if c in cols), None)
    tavg = None
    if tmin and tmax: tavg = (df[tmax]+df[tmin])/2.0; df['DTR'] = df[tmax]-df[tmin]
    elif tmn: tavg = df[tmn]
    if tavg is not None:
        df['GDD_10'] = np.maximum(tavg-10, 0)
        df['GDD_5']  = np.maximum(tavg-5,  0)
        # GDD_cum resets at season start month (not Jan 1) — avoids spurious year-long accumulation
        def _season_cumsum(series, dates, sm):
            """Cumulative sum resetting at season_start_month each year."""
            out = series.copy() * 0.0
            # assign a "season year" label: if month >= sm → season_yr = year, else year-1
            season_yr = dates.apply(lambda d: d.year if d.month >= sm else d.year - 1)
            for sy, grp_idx in series.groupby(season_yr).groups.items():
                out.loc[grp_idx] = series.loc[grp_idx].cumsum().values
            return out
        df['T2M_avg_tmp'] = tavg.values
        df['GDD_cum'] = _season_cumsum(df['GDD_10'], df['Date'], season_start_month)
        df.drop(columns='T2M_avg_tmp', inplace=True)
    if prec: df['log_precip'] = np.log1p(np.maximum(df[prec].fillna(0), 0))
    if tavg is not None and rh:
        es = 0.6108*np.exp((17.27*tavg)/(tavg+237.3))
        df['VPD'] = np.maximum(es*(1-df[rh]/100.0), 0)
    if prec and sm: df['MSI'] = df[prec]/(df[sm].replace(0, np.nan)+1e-6)
    if prec and tavg is not None:
        pet = 0.0023*(tavg+17.8)*np.maximum(tavg, 0)**0.5
        df['SPEI_proxy'] = df[prec].fillna(0)-pet.fillna(0)
    return df


# ─── TRAINING FEATURES ───────────────────────────────────────
# ─── TRAINING FEATURES ───────────────────────────────────────
def make_training_features(pheno_df, met_df, params, window=15, monsoon_onset_doys=None):
    """
    Build one record per (year × event) with met features from the 15-day window before event.

    Feature handling:
      • GDD_cum  → VALUE at event date (it's already cumulative — summing it is meaningless)
      • GDD_5/GDD_10/PREC/RAIN/log_precip/SPEI → SUM over window (true accumulation)
      • All others (T2M, RH2M, VPD, GWETTOP, etc.) → MEAN over window
    """
    # Cumulative features: take value AT event date, not sum over window
    SNAPSHOT_FEATURES = {'GDD_cum', 'GDD_CUM'}
    # Accumulation features: sum over window
    ACCUM_KEYWORDS = ['PREC', 'RAIN', 'GDD_5', 'GDD_10', 'LOG_P', 'SPEI']

    records = []
    for _, row in pheno_df.iterrows():
        for event in ['SOS', 'POS', 'EOS']:
            evt_dt = row[f'{event}_Date']
            if pd.isna(evt_dt): continue
            rec = {'Year': row['Year'], 'Event': event,
                   'Target_DOY': row.get(f'{event}_Target', row[f'{event}_DOY']),
                   'Season_Start': row.get('Season_Start', pd.NaT),
                   'LOS_Days': row.get('LOS_Days', np.nan),
                   'Peak_NDVI': row.get('Peak_NDVI', np.nan)}
            mask = ((met_df['Date'] >= evt_dt - timedelta(days=window)) &
                    (met_df['Date'] <= evt_dt))
            wdf = met_df[mask]
            if len(wdf) < max(1, window * 0.15): continue  # relaxed for monthly NDVI
            for p in params:
                if p not in met_df.columns: continue
                if p.upper() in SNAPSHOT_FEATURES:
                    # Take value at event date (or nearest available)
                    snap = met_df[met_df['Date'] <= evt_dt][p].dropna()
                    rec[p] = float(snap.iloc[-1]) if len(snap) > 0 else np.nan
                elif any(k in p.upper() for k in ACCUM_KEYWORDS):
                    # True accumulation: sum over window
                    rec[p] = float(wdf[p].sum())
                else:
                    # Instantaneous: mean over window
                    rec[p] = float(wdf[p].mean())
            if monsoon_onset_doys:
                rec['Monsoon_Onset_DOY'] = monsoon_onset_doys.get(int(row['Year']), np.nan)
            records.append(rec)
    return pd.DataFrame(records)


# ─── PHENOLOGY EXTRACTION ────────────────────────────────────
def _find_troughs(ndvi_values, min_distance=10):
    """
    Find all local minima (troughs / inter-season valleys) in a smoothed NDVI array.

    A trough at index i satisfies:
        ndvi[i] == min(ndvi[i-min_distance : i+min_distance+1])
    and is a true local minimum (lower than its immediate neighbours).

    Returns sorted list of trough indices.
    """
    n = len(ndvi_values)
    troughs = []
    for i in range(min_distance, n - min_distance):
        window = ndvi_values[max(0, i - min_distance): i + min_distance + 1]
        if ndvi_values[i] == np.min(window):
            # Confirm it is strictly lower than at least one neighbour on each side
            if ndvi_values[i] <= ndvi_values[i - 1] and ndvi_values[i] <= ndvi_values[i + 1]:
                troughs.append(i)
    # Remove duplicates that are closer than min_distance (keep the lower one)
    if not troughs:
        return troughs
    merged = [troughs[0]]
    for t in troughs[1:]:
        if t - merged[-1] < min_distance:
            # Keep the one with the lower NDVI value
            if ndvi_values[t] < ndvi_values[merged[-1]]:
                merged[-1] = t
        else:
            merged.append(t)
    return merged


def extract_phenology(ndvi_df, season_type, threshold_override=None,
                      eos_threshold_override=None,
                      met_df_for_sos=None, sos_method="threshold", rain_thresh=8.0, roll_days=7,
                      eos_method="threshold", eos_rain_thresh=3.0, eos_roll_days=14,
                      cfg=None):
    """
    NDVI Amplitude-Based Phenology Extraction — Trough-Anchored Method.

    Problem with simple window-min approach:
      Using np.min(season_window) as base picks up the current season's trough,
      which may sit INSIDE the previous growing cycle — causing SOS to be detected
      too early (in the wrong cycle), as visible in the 2024-25 season.

    Correct approach — valley-anchored thresholds:
      1. Interpolate NDVI to 5-day grid  →  pd.Series.interpolate()
      2. Apply Savitzky–Golay smoothing  →  savgol_filter(ndvi, window, 2)
      3. Detect all inter-season troughs (local minima) across the full time series
      4. For each pair of adjacent troughs  [trough_i … trough_{i+1}]  that bracket
         one growing cycle:
           • ndvi_min  = value at trough_i          ← BASE from PREVIOUS cycle valley
           • ndvi_max  = peak NDVI between the two troughs
           • A         = ndvi_max − ndvi_min         ← seasonal amplitude
           • SOS threshold = ndvi_min + thr_pct × A
           • EOS threshold = ndvi_min + eos_thr_pct × A  (same base)
           • SOS = first index AFTER trough_i  where NDVI ≥ SOS threshold
           • EOS = last  index BEFORE trough_{i+1} where NDVI ≥ EOS threshold
           • POS = argmax(NDVI[SOS : EOS+1])

    This guarantees:
      - SOS is always in the RISING limb of the correct cycle
      - EOS is always in the FALLING limb before the next valley
      - Thresholds are independently computed for each cycle (no global bias)
    """
    try:
        if cfg is None:
            cfg = SEASON_CONFIGS.get(season_type, {
                "start_month": 6, "end_month": 5, "min_days": 150,
                "threshold_pct": 0.10,
            })
        sm    = cfg["start_month"]
        em    = cfg["end_month"]
        min_d = cfg.get("min_days", 150)

        thr_pct     = threshold_override     if threshold_override     is not None else cfg.get("threshold_pct", 0.10)
        eos_thr_pct = eos_threshold_override if eos_threshold_override is not None else thr_pct

        # ── Step 1: 5-day regular interpolation ──────────────
        ndvi_raw = ndvi_df[["Date", "NDVI"]].copy().set_index("Date").sort_index()
        ndvi_raw = ndvi_raw[~ndvi_raw.index.duplicated(keep='first')]

        full_range = pd.date_range(
            start=ndvi_raw.index.min(),
            end=ndvi_raw.index.max(),
            freq="5D")
        ndvi_5d = ndvi_raw.reindex(ndvi_raw.index.union(full_range))
        ndvi_5d = ndvi_5d.interpolate(method="time")
        ndvi_5d = ndvi_5d.reindex(full_range)
        ndvi_5d.columns = ["NDVI"]

        # ── Step 2: Savitzky–Golay smoothing ─────────────────
        n = len(ndvi_5d)
        wl_target = max(7, int(n * 0.05))
        wl = wl_target if wl_target % 2 == 1 else wl_target + 1
        wl = min(wl, n - 1 if n > 1 else 1)
        if wl % 2 == 0: wl = max(7, wl - 1)
        poly = min(2, wl - 1)
        if wl < 5:
            return None, f"Too few interpolated NDVI points ({n}) — check date range"

        sm_vals = savgol_filter(ndvi_5d["NDVI"].ffill().bfill(), wl, poly)
        ndvi_5d["Sm"] = sm_vals
        t_all = ndvi_5d.index  # full DatetimeIndex

        # ── Step 3: Find all inter-season troughs ─────────────
        # min_distance ~ 1 year / 2 on a 5-day grid  ≈  36 steps
        min_dist = max(10, int((365 / 5) * 0.4))   # ~29 steps ≈ ~145 days
        trough_indices = _find_troughs(sm_vals, min_distance=min_dist)

        # Fallback: if fewer than 2 troughs found, use season window boundaries
        if len(trough_indices) < 2:
            # Original simple method as fallback
            rows = []
            for yr in range(ndvi_5d.index.year.min(), ndvi_5d.index.year.max() + 1):
                try:
                    s = f"{yr}-{sm:02d}-01"
                    e = f"{yr+1}-{em:02d}-28" if sm > em else f"{yr}-{em:02d}-28"
                    sub_vals = ndvi_5d.loc[s:e, "Sm"].values
                    sub_t    = ndvi_5d.loc[s:e, "Sm"].index
                    if len(sub_vals) < 10: continue
                    ndvi_min = np.min(sub_vals); ndvi_max = np.max(sub_vals)
                    A = ndvi_max - ndvi_min
                    if A < 1e-6: continue
                    sos_thr = ndvi_min + thr_pct * A
                    eos_thr = ndvi_min + eos_thr_pct * A
                    sc = np.where(sub_vals >= sos_thr)[0]
                    ec = np.where(sub_vals >= eos_thr)[0]
                    if not len(sc) or not len(ec): continue
                    si, ei = int(sc[0]), int(ec[-1])
                    if ei <= si: continue
                    pi = si + int(np.argmax(sub_vals[si:ei+1]))
                    sos = sub_t[si]; pos = sub_t[pi]; eos = sub_t[ei]
                    ss = pd.Timestamp(s)
                    rows.append({"Year":yr,"SOS_Date":sos,"SOS_DOY":sos.dayofyear,
                                 "SOS_Target":(sos-ss).days,"SOS_Method":"amplitude_threshold",
                                 "POS_Date":pos,"POS_DOY":pos.dayofyear,"POS_Target":(pos-ss).days,
                                 "EOS_Date":eos,"EOS_DOY":eos.dayofyear,"EOS_Target":(eos-ss).days,
                                 "EOS_Method":"amplitude_threshold","LOS_Days":(eos-sos).days,
                                 "Season_Start":ss,"Peak_NDVI":float(sub_vals[pi]),
                                 "Amplitude":float(A),"Threshold_SOS":float(sos_thr),
                                 "Threshold_EOS":float(eos_thr),"Season":season_type})
                except Exception: continue
            if not rows: return None, "No complete seasons found — check season window / threshold"
            return pd.DataFrame(rows), None

        # ── Step 4: Extract one phenology cycle per trough pair ──
        # Add sentinel endpoints so the first and last cycles are also captured
        # (prepend index 0 if not already a trough, append n-1)
        all_troughs = list(trough_indices)
        if all_troughs[0] > min_dist:
            all_troughs = [0] + all_troughs
        if all_troughs[-1] < n - min_dist:
            all_troughs = all_troughs + [n - 1]

        rows = []
        for i in range(len(all_troughs) - 1):
            try:
                ti  = all_troughs[i]       # index of left  trough (season start valley)
                ti1 = all_troughs[i + 1]   # index of right trough (season end   valley)

                if ti1 - ti < max(10, min_d // 5):
                    continue               # too short a cycle

                cycle_vals = sm_vals[ti:ti1 + 1]   # NDVI values in this cycle
                cycle_t    = t_all[ti:ti1 + 1]      # corresponding dates

                # ── Base value = NDVI at the LEFT trough (previous cycle valley) ──
                ndvi_min = float(sm_vals[ti])        # valley before this season

                # ── Peak = maximum within the cycle ──
                ndvi_max = float(np.max(cycle_vals))
                A = ndvi_max - ndvi_min

                if A < 1e-6:
                    continue

                # ── Per-cycle amplitude thresholds ──
                sos_threshold = ndvi_min + thr_pct     * A
                eos_threshold = ndvi_min + eos_thr_pct * A

                # ── SOS: first point AFTER left trough rising above sos_threshold ──
                # Search from index 1 onward (skip the trough itself)
                sos_cands = np.where(cycle_vals[1:] >= sos_threshold)[0] + 1
                if len(sos_cands) == 0:
                    continue
                sos_idx_local = int(sos_cands[0])

                # ── EOS: last point BEFORE right trough falling below eos_threshold ──
                # Search up to the last point (skip the right trough itself)
                eos_cands = np.where(cycle_vals[:-1] >= eos_threshold)[0]
                if len(eos_cands) == 0:
                    continue
                eos_idx_local = int(eos_cands[-1])

                if eos_idx_local <= sos_idx_local:
                    continue

                # ── POS: argmax between SOS and EOS ──
                pos_idx_local = sos_idx_local + int(
                    np.argmax(cycle_vals[sos_idx_local:eos_idx_local + 1]))

                sos = cycle_t[sos_idx_local]
                pos = cycle_t[pos_idx_local]
                eos = cycle_t[eos_idx_local]

                # Assign year = calendar year of POS (most meaningful label)
                yr = pos.year

                # Season start anchor for relative-DOY calculation
                # Use start of the calendar year containing the left trough
                season_start = pd.Timestamp(f"{t_all[ti].year}-{sm:02d}-01")

                rows.append({
                    "Year":          yr,
                    "SOS_Date":      sos,  "SOS_DOY":  sos.dayofyear,
                    "SOS_Target":    (sos - season_start).days,
                    "SOS_Method":    "valley_amplitude_threshold",
                    "POS_Date":      pos,  "POS_DOY":  pos.dayofyear,
                    "POS_Target":    (pos - season_start).days,
                    "EOS_Date":      eos,  "EOS_DOY":  eos.dayofyear,
                    "EOS_Target":    (eos - season_start).days,
                    "EOS_Method":    "valley_amplitude_threshold",
                    "LOS_Days":      (eos - sos).days,
                    "Season_Start":  season_start,
                    "Peak_NDVI":     float(cycle_vals[pos_idx_local]),
                    "Amplitude":     float(A),
                    "Base_NDVI":     float(ndvi_min),
                    "Threshold_SOS": float(sos_threshold),
                    "Threshold_EOS": float(eos_threshold),
                    "Trough_Date":   t_all[ti],
                    "Season":        season_type,
                })
            except Exception:
                continue

        if not rows:
            return None, "No complete seasons found — check season window / threshold"

        df_out = pd.DataFrame(rows).drop_duplicates(subset="Year", keep="first")
        return df_out.sort_values("Year").reset_index(drop=True), None

    except Exception as e:
        return None, str(e)


# ─── FEATURE SELECTION & CORRELATIONS ────────────────────────
# ─── FEATURE SELECTION & CORRELATIONS ────────────────────────
def select_best_feature(X, y, event):
    """Keep for fallback — returns single best feature."""
    pkeys = EVENT_PRIORITIES.get(event, [])
    def priority(col):
        cu=col.upper()
        for i,k in enumerate(pkeys):
            if k in cu: return i
        return len(pkeys)
    best_feat, best_r = None, 0.0
    for col in X.columns:
        vals=X[col].dropna()
        if vals.std()<1e-8 or len(vals)<3: continue
        idx=vals.index.intersection(y.dropna().index)
        if len(idx)<3: continue
        try: r,_=pearsonr(vals[idx].astype(float), y[idx].astype(float))
        except Exception: continue
        abs_r=abs(r)
        if abs_r < MIN_CORR_THRESHOLD: continue
        if (abs_r > best_r+0.05) or (abs_r >= best_r-0.05 and priority(col)<priority(best_feat or '')):
            best_feat=col; best_r=abs_r
    return best_feat, best_r


def _loo_r2_quick(X_vals, y_vals, alpha=0.01):
    """Fast LOO R² for small arrays (used in feature selection)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    loo = LeaveOneOut(); preds = []
    sc = StandardScaler()
    for tr, te in loo.split(X_vals):
        Xtr = sc.fit_transform(X_vals[tr]); Xte = sc.transform(X_vals[te])
        m = Ridge(alpha=alpha); m.fit(Xtr, y_vals[tr])
        preds.append(float(m.predict(Xte)[0]))
    preds = np.array(preds)
    ss_res = np.sum((y_vals-preds)**2)
    ss_tot = np.sum((y_vals-y_vals.mean())**2) + 1e-12
    return float(np.clip(1-ss_res/ss_tot, -1, 1))


def select_multi_features(X, y, event, max_features=5, min_r=MIN_CORR_THRESHOLD):
    """
    Select features for multi-feature Ridge with LOO CV:
      1. Exclude EVENT_EXCLUDE features (ecologically spurious)
      2. Keep only |Pearson r| >= min_r with target
      3. Remove collinear pairs (|r_pair| > 0.85) — keep higher |r| with target
      4. Incremental LOO R² check: only add feature k if it IMPROVES LOO R² by ≥ 0.03
         (prevents overfitting with small n, e.g. n=4 with 2 correlated features)
      5. Cap at min(max_features, n-2) for regularisation safety
    """
    excluded = EVENT_EXCLUDE.get(event, set())
    usable = []
    for col in X.columns:
        if col in excluded: continue
        vals = X[col].dropna()
        if vals.std() < 1e-8 or len(vals) < 3: continue
        idx = vals.index.intersection(y.dropna().index)
        if len(idx) < 3: continue
        try:
            r, _ = pearsonr(vals[idx].astype(float), y[idx].astype(float))
        except Exception: continue
        if abs(r) >= min_r:
            usable.append((col, abs(r)))
    if not usable: return []
    usable.sort(key=lambda x: -x[1])

    # Greedy collinearity removal (pairwise |r| > 0.85 → drop lower-correlated one)
    collinear_filtered = []
    for feat, r_feat in usable:
        collinear = False
        for sel in collinear_filtered:
            xi = X[feat].fillna(X[feat].median())
            xj = X[sel].fillna(X[sel].median())
            idx2 = xi.index.intersection(xj.index)
            if len(idx2) < 3: continue
            try:
                r_pair, _ = pearsonr(xi[idx2].astype(float), xj[idx2].astype(float))
            except Exception: continue
            if abs(r_pair) > 0.85:
                collinear = True; break
        if not collinear:
            collinear_filtered.append(feat)

    n_obs = len(y.dropna())
    max_safe = max(2, n_obs - 2)
    candidates = collinear_filtered[:min(max_features, max_safe)]

    if len(candidates) <= 1:
        return candidates

    # Incremental LOO R² selection: only add feature if it genuinely helps
    y_vals = y.values.astype(float)
    selected = [candidates[0]]  # always include best feature
    best_r2 = _loo_r2_quick(X[selected].fillna(X[selected[0]].median()).values.reshape(-1,1), y_vals)

    for feat in candidates[1:]:
        trial = selected + [feat]
        Xt = X[trial].fillna(X[trial].median()).values
        try:
            trial_r2 = _loo_r2_quick(Xt, y_vals)
        except Exception:
            continue
        if trial_r2 > best_r2 + 0.03:   # must improve LOO R² by at least 3%
            selected.append(feat)
            best_r2 = trial_r2

    return selected


def get_all_correlations(X, y):
    rows = []
    for col in X.columns:
        vals=X[col].dropna()
        if vals.std()<1e-8 or len(vals)<3: continue
        idx=vals.index.intersection(y.dropna().index)
        if len(idx)<3: continue
        try:
            r,p_val=pearsonr(vals[idx].astype(float), y[idx].astype(float))
            rows.append({'Feature':col,'Pearson_r':round(r,3),'|r|':round(abs(r),3),
                         'p_value':round(p_val,3),'Usable':'✅' if abs(r)>=MIN_CORR_THRESHOLD else '❌'})
        except Exception: continue
    return pd.DataFrame(rows).sort_values('|r|', ascending=False)


# ─── LOO RIDGE ───────────────────────────────────────────────
def loo_ridge(X_vals, y_vals, alpha):
    loo=LeaveOneOut(); preds=[]
    for tr,te in loo.split(X_vals):
        pipe=Pipeline([('sc',StandardScaler()),('r',Ridge(alpha=alpha))])
        pipe.fit(X_vals[tr], y_vals[tr]); preds.append(float(pipe.predict(X_vals[te])[0]))
    preds=np.array(preds)
    ss_res=np.sum((y_vals-preds)**2); ss_tot=np.sum((y_vals-y_vals.mean())**2)+1e-12
    return float(np.clip(1-ss_res/ss_tot,-1.0,1.0)), float(mean_absolute_error(y_vals,preds))


def fit_loess(X_vals_1d, y_vals, frac=0.75):
    """LOESS/LOWESS fit with LOO cross-validation."""
    n = len(y_vals)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        Xtr = X_vals_1d[mask].astype(float); ytr = y_vals[mask].astype(float)
        preds[i] = _loess_predict(Xtr, ytr, np.array([X_vals_1d[i]]), frac=frac)[0]
    ss_res = np.sum((y_vals - preds)**2)
    ss_tot = np.sum((y_vals - y_vals.mean())**2) + 1e-12
    r2  = float(np.clip(1 - ss_res / ss_tot, -1.0, 1.0))
    mae = float(mean_absolute_error(y_vals, preds))
    return r2, mae


def loo_poly(X_vals, y_vals, degree=2):
    """Polynomial regression LOO cross-validation."""
    loo = LeaveOneOut(); preds = []
    for tr, te in loo.split(X_vals):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('pf', PolynomialFeatures(degree=degree, include_bias=False)),
            ('r',  Ridge(alpha=1.0))
        ])
        pipe.fit(X_vals[tr], y_vals[tr])
        preds.append(float(pipe.predict(X_vals[te])[0]))
    preds = np.array(preds)
    ss_res = np.sum((y_vals - preds)**2)
    ss_tot = np.sum((y_vals - y_vals.mean())**2) + 1e-12
    r2  = float(np.clip(1 - ss_res / ss_tot, -1.0, 1.0))
    mae = float(mean_absolute_error(y_vals, preds))
    return r2, mae


def fit_event_model(X_all, y, event, model_key="ridge"):
    """Multi-feature Ridge / LOESS / Polynomial regression with LOO CV.
    model_key: 'ridge' | 'loess' | 'poly2' | 'poly3'
    """
    yt = y.values; n = len(yt)

    features = select_multi_features(X_all, y, event, max_features=5)

    if not features:
        md = float(yt.mean())
        return {'mode':'mean','features':[],'r2':0.0,
                'mae':float(np.mean(np.abs(yt-md))),
                'alpha':None,'coef':[],'intercept':md,'best_r':0.0,'mean_doy':md,'n':n,'pipe':None,
                'model_key': model_key}

    Xf = X_all[features].fillna(X_all[features].median())
    Xv = Xf.values

    best_single_r = 0.0
    for f in features:
        try:
            r_val, _ = pearsonr(Xf[f].astype(float), y.astype(float))
            if abs(r_val) > best_single_r: best_single_r = abs(r_val)
        except Exception: pass

    # ── LOESS (single feature only — uses best feature) ──────
    if model_key == "loess":
        feat = features[0]
        x1d  = Xf[feat].values.astype(float)
        r2, mae = fit_loess(x1d, yt.astype(float))
        # Store training data for prediction interpolation
        sort_idx = np.argsort(x1d)
        loess_data = np.column_stack([x1d[sort_idx], yt.astype(float)[sort_idx]])
        return {'mode': 'loess', 'features': [feat], 'r2': r2, 'mae': mae,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'mean_doy': float(yt.mean()), 'n': n,
                'pipe': None, 'loess_data': loess_data, 'model_key': model_key,
                'x_train': x1d, 'y_train': yt.astype(float)}

    # ── Polynomial regression ─────────────────────────────────
    if model_key in ("poly2", "poly3"):
        degree = 2 if model_key == "poly2" else 3
        r2, mae = loo_poly(Xv, yt, degree=degree)
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('pf', PolynomialFeatures(degree=degree, include_bias=False)),
            ('r',  Ridge(alpha=1.0))
        ])
        pipe.fit(Xv, yt)
        return {'mode': model_key, 'features': features, 'r2': r2, 'mae': mae,
                'alpha': 1.0, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'mean_doy': float(yt.mean()), 'n': n,
                'pipe': pipe, 'model_key': model_key}

    # ── Default: Ridge Regression ─────────────────────────────
    rcv = RidgeCV(alphas=ALPHAS, cv=LeaveOneOut())
    rcv.fit(StandardScaler().fit_transform(Xv), yt)
    best_alpha = float(rcv.alpha_)

    pipe = Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=best_alpha))])
    pipe.fit(Xv, yt)
    r2, mae = loo_ridge(Xv, yt, best_alpha)

    sc = pipe.named_steps['sc']; ridge = pipe.named_steps['r']
    coef_unstd = list(ridge.coef_ / sc.scale_)
    intercept_unstd = float(ridge.intercept_ - np.dot(ridge.coef_ / sc.scale_, sc.mean_))

    return {'mode':'ridge','features':features,'r2':r2,'mae':mae,
            'alpha':best_alpha,'coef':coef_unstd,'intercept':intercept_unstd,
            'best_r':best_single_r,'n':n,'pipe':pipe, 'model_key': model_key}



# ─── UNIVERSAL PREDICTOR ─────────────────────────────────────
class UniversalPredictor:
    def __init__(self):
        self._fits={}; self.r2={}; self.mae={}; self.n_seasons={}; self.corr_tables={}

    def train(self, train_df, all_params, model_key="ridge"):
        meta={'Year','Event','Target_DOY','LOS_Days','Peak_NDVI'}
        feat_cols=[c for c in train_df.columns if c not in meta
                   and pd.api.types.is_numeric_dtype(train_df[c]) and train_df[c].std()>1e-8]
        for event in ['SOS','POS','EOS']:
            sub=train_df[train_df['Event']==event].copy(); self.n_seasons[event]=len(sub)
            if len(sub)<3: continue
            X=sub[feat_cols].fillna(sub[feat_cols].median()); y=sub['Target_DOY']
            self.corr_tables[event]=get_all_correlations(X, y)
            fit=fit_event_model(X, y, event, model_key=model_key)
            self._fits[event]=fit; self.r2[event]=fit['r2']; self.mae[event]=fit['mae']

    def predict(self, inputs, event, year=2026, season_start_month=6):
        if event not in self._fits: return None
        fit = self._fits[event]
        if fit['mode'] == 'mean':
            rel_days = int(round(fit['mean_doy']))
        elif fit['mode'] == 'loess':
            feat    = fit['features'][0]
            x_new   = float(inputs.get(feat, 0.0))
            x_train = fit.get('x_train')
            y_train = fit.get('y_train')
            if x_train is not None and len(x_train) >= 2:
                pred = _loess_predict(x_train, y_train, np.array([x_new]), frac=0.75)[0]
                rel_days = int(np.clip(round(float(pred)), 0, 500))
            else:
                rel_days = int(round(fit['mean_doy']))
        else:
            vals = np.array([[inputs.get(f, 0.0) for f in fit['features']]])
            rel_days = int(np.clip(round(float(fit['pipe'].predict(vals)[0])), 0, 500))
        season_start = datetime(year, season_start_month, 1)
        date = season_start + timedelta(days=rel_days)
        doy = date.timetuple().tm_yday
        return {'doy': doy, 'date': date, 'rel_days': rel_days,
                'r2': self.r2[event], 'mae': self.mae[event], 'event': event}


    def equation_str(self, event, season_start_month=6):
        """Returns only the regression equation line — used in Predict tab & short displays."""
        if event not in self._fits: return "Need ≥ 3 seasons"
        fit = self._fits[event]
        sm_name = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        start_label = sm_name.get(season_start_month, 'Jun')
        target_label = f"{event}_days_from_{start_label}1"

        if fit['mode'] == 'mean':
            return (f"{target_label} ≈ {fit['mean_doy']:.0f}  "
                    f"[No feature |r|≥{MIN_CORR_THRESHOLD} — mean prediction only]")

        intercept = fit.get('intercept', 0.0)
        terms = [f"{intercept:.3f}"]
        for feat, coef in zip(fit['features'], fit['coef']):
            sign = '+' if coef >= 0 else '-'
            terms.append(f"{sign} {abs(coef):.5f} × {feat}")

        eq_str = f"{target_label}  =  " + "  ".join(terms)
        eq_str += (f"\n    [Ridge α={fit['alpha']}, {len(fit['features'])} feature(s), "
                   f"R²(LOO)={fit['r2']:.3f}, MAE=±{fit['mae']:.1f} d]")
        return eq_str

    def corr_table_for_display(self, event):
        """
        Returns a clean DataFrame for the Training tab feature table.
        Columns: Feature | Pearson r | |r| | Role
        Role values: IN MODEL / Excluded (spurious) / Not selected
        p-values are intentionally omitted — shown only in the Correlations heatmap.
        """
        if event not in self._fits: return pd.DataFrame()
        fit = self._fits[event]
        ct  = self.corr_tables.get(event)
        if ct is None or len(ct) == 0: return pd.DataFrame()

        in_model = set(fit['features'])
        rows = []
        for _, row in ct.iterrows():
            feat = row['Feature']
            usable = row['Usable'] == '✅'
            if feat in in_model:
                role = '✅  In model'
            elif feat in EVENT_EXCLUDE.get(event, set()):
                role = '⛔  Excluded — ecologically spurious'
            elif usable:
                role = '➖  Correlated but not selected (collinear or did not improve LOO R²)'
            else:
                role = '⬜  Below |r| threshold'
            rows.append({
                'Feature':   feat,
                'Pearson r': row['Pearson_r'],
                '|r|':       row['|r|'],
                'Role':      role,
            })
        return pd.DataFrame(rows)


# ─── PLOTS ───────────────────────────────────────────────────
def plot_ndvi_phenology(ndvi_raw, pheno_df):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    dates = pd.to_datetime(ndvi_raw['Date'])
    ax.scatter(dates, ndvi_raw['NDVI'], color='#A5D6A7', s=18, alpha=0.55,
               label='NDVI (raw obs)', zorder=3)

    # Smoothed line (same 5-day grid used in extraction)
    ndvi_s = (ndvi_raw.set_index('Date')['NDVI']
              .resample('5D').interpolate(method='time'))
    n = len(ndvi_s)
    wl_target = max(7, int(n * 0.05))
    wl = wl_target if wl_target % 2 == 1 else wl_target + 1
    wl = min(wl, n - 1 if n > 1 else 1)
    if wl % 2 == 0: wl = max(7, wl - 1)
    poly = min(2, wl - 1)
    if wl >= 5:
        sm_arr = savgol_filter(ndvi_s.ffill().bfill().values, wl, poly)
        ax.plot(ndvi_s.index, sm_arr, color='#1B5E20', lw=2.2,
                label=f'Smoothed (SG w={wl})', zorder=5)

    # ── Draw per-cycle threshold band (Base NDVI → threshold) ──
    # Shown as a horizontal span per season so the valley-anchoring is visible
    trough_plotted = False
    for _, row in pheno_df.iterrows():
        base  = row.get('Base_NDVI',  np.nan)
        t_sos = row.get('Threshold_SOS', np.nan)
        t_eos = row.get('Threshold_EOS', np.nan)
        trough_d = row.get('Trough_Date', pd.NaT)
        sos_d = row.get('SOS_Date', pd.NaT)
        eos_d = row.get('EOS_Date', pd.NaT)

        if pd.notna(sos_d) and pd.notna(eos_d):
            # Shaded band: from base to SOS threshold
            if not np.isnan(base) and not np.isnan(t_sos):
                ax.axhspan(base, t_sos,
                           xmin=max(0, (pd.Timestamp(sos_d) - ndvi_s.index[0]).days /
                                    max(1, (ndvi_s.index[-1] - ndvi_s.index[0]).days)),
                           xmax=min(1, (pd.Timestamp(eos_d) - ndvi_s.index[0]).days /
                                    max(1, (ndvi_s.index[-1] - ndvi_s.index[0]).days)),
                           alpha=0.07, color='#F9A825',
                           label='Base→Threshold band' if not trough_plotted else '')
                trough_plotted = True

        # Trough marker (valley anchor)
        if pd.notna(trough_d) and not np.isnan(base):
            ax.plot(trough_d, base, 'v', color='#6A1B9A', ms=7, zorder=6,
                    label='Valley anchor (base NDVI)' if _ == pheno_df.index[0] else '')

    # SOS / POS / EOS vertical lines
    ev_colors = {'SOS': '#43A047', 'POS': '#1565C0', 'EOS': '#E65100'}
    ev_labels_map = {
        'SOS': 'SOS — Green-up start',
        'POS': 'POS — Peak greenness',
        'EOS': 'EOS — Senescence end'
    }
    plotted = set()
    for _, row in pheno_df.iterrows():
        for ev, col in ev_colors.items():
            d = row.get(f'{ev}_Date')
            if pd.notna(d):
                lbl = ev_labels_map[ev] if ev not in plotted else ''
                ax.axvline(d, color=col, lw=1.4, alpha=0.55, ls='--', label=lbl)
                plotted.add(ev)

    handles = [
        Line2D([0],[0], color='#A5D6A7', marker='o', ms=5, lw=0, label='NDVI (raw obs)'),
        Line2D([0],[0], color='#1B5E20', lw=2.2, label=f'Smoothed (SG w={wl if wl>=5 else "n/a"})'),
        Line2D([0],[0], color='#6A1B9A', marker='v', ms=7, lw=0, label='Valley anchor (base NDVI)'),
        Line2D([0],[0], color='#43A047', lw=1.5, ls='--', label='SOS — Green-up start'),
        Line2D([0],[0], color='#1565C0', lw=1.5, ls='--', label='POS — Peak greenness'),
        Line2D([0],[0], color='#E65100', lw=1.5, ls='--', label='EOS — Senescence end'),
    ]
    ax.set_title('NDVI Time Series + Smoothed Signal + Phenology Events\n'
                 '(Valley-anchored amplitude threshold — base = previous cycle trough)',
                 fontsize=11, fontweight='bold', color='#1B5E20')
    ax.set_xlabel('Date'); ax.set_ylabel('NDVI')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.legend(handles=handles, ncol=3, fontsize=8.0, loc='upper left', framealpha=0.88)
    ax.grid(True, alpha=0.22, ls='--'); ax.set_facecolor('#FAFFF8')
    fig.tight_layout()
    return fig


def plot_pheno_trends(pheno_df):
    """Plot trends using actual calendar dates (month-day) for SOS/POS/EOS, and LOS in days."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    fig.patch.set_facecolor('#F7FBF5')

    ev_cfg = [
        ('SOS', 'SOS_Date', 'SOS_DOY',  '#43A047', 'SOS — Green-up start'),
        ('POS', 'POS_Date', 'POS_DOY',  '#1565C0', 'POS — Peak greenness'),
        ('EOS', 'EOS_Date', 'EOS_Target','#E65100', 'EOS — Senescence end'),
        ('LOS', None,       'LOS_Days',  '#795548', 'LOS — Season Length (days)'),
    ]

    for ax, (ev, date_col, doy_col, clr, lbl) in zip(axes, ev_cfg):
        yrs = pheno_df['Year'].values

        if ev == 'LOS':
            vals = pheno_df['LOS_Days'].values
            ax.bar(yrs, vals, color=clr, alpha=0.55, width=0.7, edgecolor='white')
            ax.plot(yrs, vals, 'o-', color=clr, ms=6, lw=2,
                    markeredgecolor='white', markeredgewidth=1.2)
            ax.set_ylabel('Days')
            y_min = max(0, vals.min() - 30)
            ax.set_ylim(y_min, vals.max() + 30)
        else:
            # Use season-relative days (Target) for EOS, raw DOY for SOS/POS
            if doy_col in pheno_df.columns:
                vals = pheno_df[doy_col].values.astype(float)
            else:
                vals = pheno_df[f'{ev}_DOY'].values.astype(float)

            ax.bar(yrs, vals, color=clr, alpha=0.55, width=0.7, edgecolor='white')
            ax.plot(yrs, vals, 'o-', color=clr, ms=6, lw=2,
                    markeredgecolor='white', markeredgewidth=1.2)

            # Y-axis: show month-day labels from actual dates if available
            if date_col and date_col in pheno_df.columns and ev != 'EOS':
                # Convert DOY to month-day ticks
                unique_vals = np.linspace(vals.min(), vals.max(), 5).astype(int)
                tick_labels = []
                for doy in unique_vals:
                    try:
                        tick_labels.append(
                            (pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(doy)-1)).strftime('%b %d'))
                    except Exception:
                        tick_labels.append(str(doy))
                ax.set_yticks(unique_vals)
                ax.set_yticklabels(tick_labels, fontsize=8)
                ax.set_ylabel('Date (approx.)')
            elif ev == 'EOS':
                # EOS uses season-relative days — label as "days since season start"
                ax.set_ylabel('Days since season start')
                # Add secondary annotation showing approx month
                sm = pheno_df.get('Season', pd.Series()).iloc[0] if len(pheno_df) else ''
                ax.set_ylabel('EOS (season-relative days)')
            else:
                ax.set_ylabel('DOY')

        if len(yrs) >= 2:
            m, b = np.polyfit(yrs, vals, 1)
            ax.plot(yrs, m*yrs+b, '--', color='#263238', lw=1.8, label=f'Trend: {m:+.1f} d/yr')
            ax.legend(fontsize=8.5, framealpha=0.85)

        ax.set_title(lbl, fontsize=10, fontweight='bold', color='#1B4332')
        ax.set_xlabel('Year', fontsize=9)
        ax.grid(True, alpha=0.22, ls='--')
        ax.set_facecolor('#FAFFF8')
        ax.tick_params(labelsize=8.5)

    fig.suptitle('Phenological Trends Across Years', fontsize=13, fontweight='bold',
                 color='#1B4332', y=1.02)
    fig.tight_layout()
    return fig


def plot_obs_vs_pred(predictor, train_df):
    events=[ev for ev in ['SOS','POS','EOS']
            if ev in predictor._fits and predictor._fits[ev]['mode']=='ridge']
    if not events: return None
    fig, axes = plt.subplots(1, len(events), figsize=(5*len(events), 4.5), squeeze=False)
    clrs={'SOS':'#4CAF50','POS':'#1565C0','EOS':'#FF6F00'}
    for ax, ev in zip(axes[0], events):
        fit = predictor._fits[ev]
        sub = train_df[train_df['Event'] == ev].copy()
        feats = fit['features']
        # Use all features the model was trained on
        available = [f for f in feats if f in sub.columns]
        if not available: continue
        Xf = sub[available].fillna(sub[available].median())
        # If not all features available, refit temporarily (shouldn't happen normally)
        if len(available) < len(feats):
            # Predict with available subset — just show what we have
            pass
        try:
            pred = fit['pipe'].predict(Xf.values)
        except ValueError:
            # Feature count mismatch — skip gracefully
            ax.text(0.5, 0.5, f'{ev}: feature mismatch\n(re-train to fix)',
                    ha='center', va='center', fontsize=9, transform=ax.transAxes)
            continue
        obs = sub['Target_DOY'].values
        ax.scatter(obs, pred, color=clrs[ev], s=80, edgecolors='white', lw=1.5, zorder=3, alpha=0.9)
        lims = [min(obs.min(), pred.min())-8, max(obs.max(), pred.max())+8]
        ax.plot(lims, lims, 'k--', lw=1.2); ax.set_xlim(lims); ax.set_ylim(lims)
        feat_label = ' + '.join(available)
        ax.set_title(f'{ev}   R²(LOO)={predictor.r2.get(ev,0):.3f}   MAE={predictor.mae.get(ev,0):.1f} d\n{feat_label}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Observed (days from season start)')
        ax.set_ylabel('Predicted (days from season start)')
        ax.grid(True, alpha=0.25); ax.set_facecolor('#FAFFF8')
    fig.suptitle('Observed vs Predicted — Ridge Regression (training fit)', fontsize=11, fontweight='bold')
    fig.tight_layout(); return fig


def plot_correlation_summary(predictor, pheno_df):
    """
    3-panel correlation figure:
      Left   — Grouped horizontal bars (SOS/POS/EOS side-by-side, top-12 features)
      Middle — Clean Pearson-r heatmap with significance stars (** p<0.05, * p<0.10)
      Right  — Scatter plots: best feature vs DOY for each event
    """
    events    = ['SOS', 'POS', 'EOS']
    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#BF360C'}
    ev_markers= {'SOS': 'o',       'POS': 's',       'EOS': '^'}

    # ── Build unified feature list (union across events, top 12 by max |r|) ──
    feat_r = {}
    for ev in events:
        ct = predictor.corr_tables.get(ev)
        if ct is None: continue
        for _, row in ct.iterrows():
            f = row['Feature']
            feat_r[f] = max(feat_r.get(f, 0), abs(row['Pearson_r']))
    all_feats = sorted(feat_r, key=lambda f: -feat_r[f])[:14]

    # ── r matrix & p-value matrix ─────────────────────────────
    r_mat = pd.DataFrame(0.0, index=all_feats, columns=events)
    p_mat = pd.DataFrame(1.0, index=all_feats, columns=events)
    best  = {}
    tdf   = st.session_state.get('train_df')

    for ev in events:
        ct = predictor.corr_tables.get(ev)
        if ct is None: continue
        # Pick best feature for scatter: skip EVENT_EXCLUDE features
        # (they are ecologically spurious and excluded from model)
        _excluded_for_ev = EVENT_EXCLUDE.get(ev, set())
        for _, _row in ct.iterrows():
            if _row['Feature'] not in _excluded_for_ev:
                best[ev] = _row['Feature']
                break
        for _, row in ct.iterrows():
            f = row['Feature']
            if f not in r_mat.index: continue
            # Use values DIRECTLY from corr_tables — same source as Training tab equations
            # (previously p_mat was recomputed from train_df which gave different n and results)
            r_mat.loc[f, ev] = row['Pearson_r']
            if 'p_value' in row and not pd.isna(row['p_value']):
                p_mat.loc[f, ev] = row['p_value']

    n_feats = len(all_feats)
    n_ev    = 3

    fig = plt.figure(figsize=(17, max(6, n_feats * 0.52 + 2.5)))
    fig.patch.set_facecolor('#F8FBF7')
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.1, 1.5], wspace=0.36)

    # ══ LEFT: Grouped horizontal bars ════════════════════════
    ax_bar = fig.add_subplot(gs[0])
    bar_h  = 0.22
    y_pos  = np.arange(n_feats)
    offsets= np.array([-1, 0, 1]) * bar_h

    for i, ev in enumerate(events):
        vals = r_mat[ev].values
        bar_colors = [ev_colors[ev] if abs(v) >= MIN_CORR_THRESHOLD else '#CFCFCF' for v in vals]
        ax_bar.barh(y_pos + offsets[i], vals, height=bar_h * 0.82,
                    color=bar_colors, edgecolor='white', lw=0.4,
                    label=ev, alpha=0.88)

    ax_bar.axvline(0, color='#37474F', lw=1.0)
    for thresh in [MIN_CORR_THRESHOLD, -MIN_CORR_THRESHOLD]:
        ax_bar.axvline(thresh, color='#555', lw=1.1, ls='--', alpha=0.55)
    ax_bar.axvspan( MIN_CORR_THRESHOLD, 1.05, alpha=0.035, color='#4CAF50')
    ax_bar.axvspan(-1.05, -MIN_CORR_THRESHOLD, alpha=0.035, color='#4CAF50')
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(all_feats, fontsize=9.5)
    ax_bar.set_xlim(-1.05, 1.05)
    ax_bar.set_xlabel('Pearson r  (with event DOY)', fontsize=10, fontweight='bold')
    ax_bar.set_title('Feature Correlations — SOS / POS / EOS\n'
                     'Coloured = |r| ≥ 0.40 (usable)  ·  Grey = below threshold',
                     fontsize=9.5, fontweight='bold', color='#1B4332', pad=7)
    ax_bar.grid(True, axis='x', alpha=0.20, ls='--')
    ax_bar.set_facecolor('#FAFFF8')
    ax_bar.legend(title='Event', fontsize=9, title_fontsize=9,
                  loc='lower right', framealpha=0.92,
                  handles=[plt.matplotlib.patches.Patch(color=ev_colors[e], label=e) for e in events])

    # ══ MIDDLE: Clean heatmap with significance stars ════════
    ax_hm = fig.add_subplot(gs[1])
    mat   = r_mat.values.astype(float)
    im    = ax_hm.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1,
                         interpolation='nearest')

    ax_hm.set_xticks(range(n_ev))
    ax_hm.set_xticklabels(events, fontsize=11, fontweight='bold')
    ax_hm.set_yticks(range(n_feats))
    ax_hm.set_yticklabels(all_feats, fontsize=9)

    # Bold border on usable cells; stars for significance
    for i in range(n_feats):
        for j in range(n_ev):
            v  = mat[i, j]
            pv = p_mat.iloc[i, j]
            star = '**' if pv < 0.05 else ('*' if pv < 0.10 else '')
            txt_color = 'white' if abs(v) > 0.60 else '#1A1A1A'
            ax_hm.text(j, i, f'{v:.2f}{star}', ha='center', va='center',
                       fontsize=8.5, fontweight='bold', color=txt_color)
            if abs(v) >= MIN_CORR_THRESHOLD:
                rect = plt.matplotlib.patches.FancyBboxPatch(
                    (j-0.48, i-0.48), 0.96, 0.96,
                    boxstyle='round,pad=0.02', linewidth=1.8,
                    edgecolor='#1B4332', facecolor='none')
                ax_hm.add_patch(rect)

    cb = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Pearson r', fontsize=9)
    ax_hm.set_title('Pearson r Heatmap\n** p<0.05  * p<0.10  (bold border = usable)',
                    fontsize=9.5, fontweight='bold', color='#1B4332', pad=7)
    ax_hm.tick_params(axis='x', bottom=False)
    ax_hm.spines[:].set_visible(False)

    # Alternating row shading for readability
    for i in range(n_feats):
        if i % 2 == 0:
            ax_hm.axhspan(i-0.5, i+0.5, color='#F0F0F0', alpha=0.25, zorder=0)

    # ══ RIGHT: Scatter plots — best feature per event ════════
    inner = gs[2].subgridspec(n_ev, 1, hspace=0.60)
    for i, ev in enumerate(events):
        ax_s  = fig.add_subplot(inner[i])
        feat  = best.get(ev)
        ct    = predictor.corr_tables.get(ev)
        ax_s.set_facecolor('#FAFFF8')
        ax_s.grid(True, alpha=0.22, ls='--')
        ax_s.tick_params(labelsize=7.5)
        if feat is None or ct is None or tdf is None:
            ax_s.text(0.5, 0.5, 'No data', ha='center', va='center',
                      fontsize=8, transform=ax_s.transAxes)
            continue
        r_val = ct.iloc[0]['Pearson_r']
        # train_df stores target as 'Target_DOY' with 'Event' column to filter
        target_col = 'Target_DOY'
        if tdf is not None and feat in tdf.columns and target_col in tdf.columns:
            sub = tdf[tdf['Event'] == ev][[feat, target_col]].dropna() if 'Event' in tdf.columns else tdf[[feat, target_col]].dropna()
        else:
            sub = pd.DataFrame()
        if len(sub) < 3:
            ax_s.text(0.5, 0.5, 'Too few points', ha='center', va='center',
                      fontsize=8, transform=ax_s.transAxes); continue
        x, y = sub[feat].values, sub[target_col].values
        ax_s.scatter(x, y, color=ev_colors[ev], marker=ev_markers[ev],
                     s=60, alpha=0.88, edgecolors='white', lw=0.8, zorder=4)
        if len(x) > 1:
            z  = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax_s.plot(xr, np.polyval(z, xr), color=ev_colors[ev],
                      lw=1.8, ls='--', alpha=0.80)
        ax_s.set_title(f'{ev}  ←  {feat}\nr = {r_val:.3f}',
                       fontsize=8.5, fontweight='bold', color=ev_colors[ev], pad=4)
        ax_s.set_xlabel(feat, fontsize=8)
        ax_s.set_ylabel(f'{ev} (days from season start)', fontsize=7.5)

    fig.suptitle('Feature Correlations with Phenological Events',
                 fontsize=13, fontweight='bold', color='#1B4332', y=1.005)
    fig.tight_layout()
    return fig


def plot_met_with_ndvi(met_df, ndvi_df, raw_params, season_cfg=None,
                       pheno_df=None, predictor=None):
    """
    Year-by-year figures: for each growing season, plot 2 sub-panels:
      Top    — NDVI + Air params (RH2M, T2M_MAX, T2M_MIN, PRECTOTCORR)
      Bottom — NDVI + Soil params (GWETTOP, GWETROOT, GWETPROF, WS2M)

    If pheno_df and predictor are supplied, vertical dashed lines for SOS/POS/EOS
    are drawn on both panels, with an annotation showing:
      • the event date
      • the model feature name and its 15-day pre-event mean value
      • the equation term used for prediction
    This directly links the seasonal climate context to the model equations.
    Returns a list of (title, fig) tuples — one per season.
    """
    AIR_PARAMS = [
        ('RH2M',        '#212121', 'RH2M (%)'),
        ('T2M_MAX',     '#E53935', 'Max T (°C)'),
        ('T2M_MIN',     '#1E88E5', 'Min T (°C)'),
        ('PRECTOTCORR', '#8E24AA', 'Rain (mm)'),
    ]
    SOIL_PARAMS = [
        ('GWETTOP',  '#FB8C00', 'Surface Soil Wetness'),
        ('GWETROOT', '#6A1B9A', 'Root Soil Wetness'),
        ('GWETPROF', '#795548', 'Profile Soil Moisture'),
        ('WS2M',     '#546E7A', 'Wind Speed (m/s)'),
    ]

    # ── Build daily NDVI ──────────────────────────────────────
    ndvi_daily = (ndvi_df.set_index('Date')['NDVI']
                  .resample('D').interpolate(method='time'))

    # ── Determine seasons ─────────────────────────────────────
    if season_cfg:
        sm = season_cfg.get('start_month', 6)
        em = season_cfg.get('end_month', 5)
    else:
        sm, em = 6, 5  # default Jun-May

    years = sorted(met_df['Date'].dt.year.unique())
    figs = []

    for yr in years:
        if sm <= em:
            s = pd.Timestamp(f"{yr}-{sm:02d}-01")
            e = pd.Timestamp(f"{yr}-{em:02d}-28")
        else:
            s = pd.Timestamp(f"{yr}-{sm:02d}-01")
            e = pd.Timestamp(f"{yr+1}-{em:02d}-28")

        df_met = met_df[(met_df['Date'] >= s) & (met_df['Date'] <= e)].copy()
        if len(df_met) < 30: continue
        ndvi_seg = ndvi_daily.reindex(df_met['Date'].values, method='nearest', tolerance='5D')

        season_label = f"{yr}–{yr+1}" if sm > em else str(yr)
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
        fig.patch.set_facecolor('#FAFFF8')
        fig.suptitle(f"NDVI + Meteorology — Growing Season {season_label}",
                     fontsize=16, fontweight='bold', y=0.98)

        def _draw_panel(ax, param_list, panel_title, bar_params=('PRECTOTCORR','PRECTOT','RAIN')):
            """Draw NDVI on left axis + each met param on successive right axes."""
            # NDVI left axis
            ax.fill_between(df_met['Date'], ndvi_seg, alpha=0.15, color='#2E7D32')
            ax.plot(df_met['Date'], ndvi_seg, color='#2E7D32', lw=2.5,
                    label='NDVI', zorder=5)
            ax.set_ylabel('NDVI', color='#2E7D32', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='y', labelcolor='#2E7D32', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.30)
            ax.set_facecolor('#FAFFF8')
            ax.set_title(panel_title, fontsize=13, fontweight='bold', pad=18, loc='center')

            twin_axes = []
            present = [(p, c, l) for p, c, l in param_list if p in df_met.columns]
            for i, (var, col, lab) in enumerate(present):
                axr = ax.twinx()
                offset = 1.0 + 0.10 * i
                if i > 0:
                    axr.spines['right'].set_position(('axes', offset))
                    axr.spines['right'].set_visible(True)

                if var in bar_params:
                    axr.bar(df_met['Date'], df_met[var], color=col, alpha=0.50,
                            width=1.0, label=lab, zorder=3)
                else:
                    axr.plot(df_met['Date'], df_met[var], color=col, lw=1.6,
                             alpha=0.85, label=lab)

                axr.set_ylabel(lab, color=col, fontsize=9, rotation=270, labelpad=18)
                axr.tick_params(axis='y', labelcolor=col, labelsize=8)
                axr.spines['right'].set_color(col)
                axr.yaxis.label.set_color(col)
                twin_axes.append(axr)

            # Legend: combine NDVI + all met params
            h, l = ax.get_legend_handles_labels()
            for axr in twin_axes:
                hh, ll = axr.get_legend_handles_labels()
                h.extend(hh); l.extend(ll)
            return h, l

        # ── Top panel: Air ────────────────────────────────────
        h_air, l_air = _draw_panel(ax_top, AIR_PARAMS,
                                    "Air Parameters  (RH, Tmax, Tmin, Precip)")
        fig.legend(h_air, l_air, loc='upper center', bbox_to_anchor=(0.5, 0.97),
                   ncol=len(h_air), fontsize=9, framealpha=0.95,
                   title='Air Parameters', title_fontsize=9)

        # ── Bottom panel: Soil ────────────────────────────────
        h_soil, l_soil = _draw_panel(ax_bot, SOIL_PARAMS,
                                      "Soil Parameters  (GWETs, Wind Speed)")
        fig.legend(h_soil, l_soil, loc='center left', bbox_to_anchor=(0.01, 0.28),
                   ncol=1, fontsize=9, framealpha=0.95,
                   title='Soil Parameters', title_fontsize=9)

        # ── Draw SOS / POS / EOS vertical lines if pheno_df provided ──
        EV_STYLE = {
            'SOS': ('#2E7D32', '🌱', 'SOS'),
            'POS': ('#1565C0', '🌿', 'POS'),
            'EOS': ('#E65100', '🍂', 'EOS'),
        }
        if pheno_df is not None:
            # Find the row for this year
            yr_row = pheno_df[pheno_df['Year'] == yr] if 'Year' in pheno_df.columns else pd.DataFrame()
            if len(yr_row) == 0 and sm > em:
                # Cross-year season: try yr-1 as the season start year
                yr_row = pheno_df[pheno_df['Year'] == yr - 1] if 'Year' in pheno_df.columns else pd.DataFrame()

            if len(yr_row) > 0:
                row = yr_row.iloc[0]
                for ev, (col, icon, label) in EV_STYLE.items():
                    ev_date = row.get(f'{ev}_Date', pd.NaT)
                    if pd.isna(ev_date): continue
                    if not (s <= ev_date <= e): continue

                    # Draw vertical line on both panels
                    for ax in [ax_top, ax_bot]:
                        ax.axvline(ev_date, color=col, lw=2.0, ls='--', alpha=0.85, zorder=10)

                    # Build annotation: feature name + 15-day pre-event mean
                    annot_lines = [f"{icon} {label}  {ev_date.strftime('%b %d')}"]
                    if predictor is not None:
                        fit = predictor._fits.get(ev, {})
                        feats = fit.get('features', [])
                        for f_name in feats[:2]:  # show up to 2 features
                            if f_name in met_df.columns:
                                win = met_df[(met_df['Date'] >= ev_date - timedelta(days=15)) &
                                             (met_df['Date'] <= ev_date)]
                                val = win[f_name].mean() if len(win) > 0 else np.nan
                                if not np.isnan(val):
                                    unit = '°C' if 'T2M' in f_name else ('%' if 'RH' in f_name else '')
                                    annot_lines.append(f"{f_name}: {val:.1f}{unit}")

                    annot_txt = '\n'.join(annot_lines)
                    # Place label just above NDVI area (y=0.92 in axes coords)
                    ax_top.text(ev_date, 0.96, annot_txt,
                                transform=ax_top.get_xaxis_transform(),
                                fontsize=7.5, color=col, fontweight='bold',
                                ha='center', va='top',
                                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=col, alpha=0.85),
                                zorder=15)

        # ── X-axis ────────────────────────────────────────────
        ax_bot.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=45, ha='right')

        fig.tight_layout(rect=[0.10, 0.0, 1.0, 0.93])
        figs.append((season_label, fig))

    return figs


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown('<p class="main-header">🌲 Universal Vegetation Phenology Predictor</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload NDVI + NASA POWER Met · '
                'NDVI Amplitude Threshold · Ridge / LOESS / Polynomial · LOO CV · SOS / POS / EOS / LOS</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <b>📋 How to use:</b> Upload your <b>NDVI CSV</b> and <b>NASA POWER Met CSV</b> →
    App interpolates NDVI at 5-day intervals, applies SG smoothing, extracts phenology via amplitude-based threshold, fits models &amp; shows correlations →
    Use <b>Predict</b> tab to forecast events for any year<br>
    <b>Model:</b> Pearson |r|≥{MIN_CORR_THRESHOLD} feature filter &nbsp;·&nbsp;
    Single best feature per event &nbsp;·&nbsp; Ridge α auto-tuned &nbsp;·&nbsp; LOO cross-validation
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────
    st.sidebar.markdown("## 📤 Upload Your Data")
    ndvi_file = st.sidebar.file_uploader(
        "1️⃣  NDVI CSV",  type=['csv'],
        help="Any CSV with columns: date + ndvi  (date/dates/time/datetime + ndvi/ndvi_value/value/evi)")
    met_file = st.sidebar.file_uploader(
        "2️⃣  NASA POWER Met CSV", type=['csv'],
        help="Daily export from power.larc.nasa.gov — header block auto-skipped, all parameters auto-detected")

    # ── CACHE-BUST: clear stale predictor/train_df when files change ──
    # Build a fingerprint from current file names + sizes
    _ndvi_fp = f"{ndvi_file.name}:{ndvi_file.size}" if ndvi_file else ""
    _met_fp  = f"{met_file.name}:{met_file.size}"   if met_file  else ""
    _cur_fp  = f"{_ndvi_fp}|{_met_fp}"
    if st.session_state.get('_file_fingerprint') != _cur_fp:
        # New files — wipe all derived state so Predict tab never shows
        # features from a previous training run
        for _k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params','ndvi_df']:
            st.session_state[_k] = None
        st.session_state['_file_fingerprint'] = _cur_fp

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📅 Season Window Configuration")

    # Simple season window — user sets start/end month manually
    sm_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_options = list(sm_names.keys())
    month_labels  = [sm_names[m] for m in month_options]

    col_sm, col_em = st.sidebar.columns(2)
    start_month_sel = col_sm.selectbox(
        "Season start", options=month_options,
        index=5,  # default June
        format_func=lambda m: sm_names[m],
        help="Month when the growing season begins")
    end_month_sel = col_em.selectbox(
        "Season end", options=month_options,
        index=4,  # default May
        format_func=lambda m: sm_names[m],
        help="Month when the growing season ends")
    min_days_sel = st.sidebar.slider(
        "Minimum days in season window", 60, 300, 150, 10,
        help="Seasons with fewer interpolated days are skipped")

    # Build a minimal cfg dict so downstream code keeps working
    season_type = f"Custom ({sm_names[start_month_sel]}–{sm_names[end_month_sel]})"
    cfg = {
        "start_month": start_month_sel,
        "end_month":   end_month_sel,
        "min_days":    min_days_sel,
        "sos_base":    None,
        "pos_constrain_end": None,
        "threshold_pct": 0.10,
        "sos_method":  "threshold",
        "icon": "🌿",
    }

    # NDVI threshold — single slider for amplitude-based threshold
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ NDVI Amplitude Threshold")
    st.sidebar.caption(
        "Threshold is applied **per season** as: NDVI_min + threshold% × (NDVI_max − NDVI_min). "
        "SOS = first date NDVI rises above this; EOS = last date before dormancy.")
    sos_threshold_pct = st.sidebar.slider(
        "SOS threshold (% of NDVI amplitude)", 5, 30, 10, 5,
        help="First date NDVI rises above: base + threshold% × amplitude. Default: 10%"
    ) / 100.0
    eos_threshold_pct = st.sidebar.slider(
        "EOS threshold (% of NDVI amplitude)", 5, 30, 10, 5,
        help="Last date NDVI stays above: base + threshold% × amplitude before next dormancy. Default: 10%"
    ) / 100.0
    st.sidebar.caption(
        f"SOS: **{int(sos_threshold_pct*100)}%** | EOS: **{int(eos_threshold_pct*100)}%**"
    )
    threshold_pct_override = sos_threshold_pct

    # Fixed method values (amplitude threshold only — other methods removed)
    sos_method_sel       = "threshold"
    eos_method_sel       = "threshold"
    rain_thresh_sel      = 8.0
    roll_days_sel        = 7
    eos_rain_thresh_sel  = 3.0
    eos_roll_days_sel    = 14

    # Regression model selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Regression Model")
    model_options = {
        "Ridge Regression (Linear)":    "ridge",
        "LOESS / LOWESS Smoothing":     "loess",
        "Polynomial Regression (deg 2)":"poly2",
        "Polynomial Regression (deg 3)":"poly3",
    }
    regression_model_sel = st.sidebar.radio(
        "Select fitting model",
        options=list(model_options.keys()),
        index=0,
        help="Ridge = L2-regularised linear regression (default). LOESS = locally weighted smoothing. Polynomial = non-linear fit."
    )
    regression_model_key = model_options[regression_model_sel]

    # ── GATE ─────────────────────────────────────────────────
    if not (ndvi_file and met_file):
        st.markdown("""
<div class="upload-hint">
<b>👈 Upload both files in the sidebar to begin analysis</b><br><br>
<b>NDVI CSV format example:</b>
<pre>Date,NDVI
2016-01-01,0.42
2016-01-17,0.45
2016-02-02,0.48</pre>
<b>NASA POWER Met CSV:</b> download from
<a href="https://power.larc.nasa.gov/data-access-viewer/" target="_blank">power.larc.nasa.gov</a>
→ Daily → Point → your site coordinates → select parameters → download CSV.<br>
The header block is skipped automatically. All met parameters are auto-detected.<br><br>
<b>Recommended NASA POWER parameters:</b>
T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, RH2M, GWETTOP, GWETROOT, ALLSKY_SFC_SW_DWN
</div>
        """, unsafe_allow_html=True)
        return

    # ── PARSE ────────────────────────────────────────────────
    with st.spinner("📂 Parsing files…"):
        ndvi_result, ndvi_err = parse_ndvi(ndvi_file)
        if ndvi_result is None:
            st.error(f"❌ NDVI: {ndvi_err}"); return

        # ── Handle multi-site CSV ─────────────────────────────
        if isinstance(ndvi_result, tuple) and ndvi_result[0] == 'MULTI_SITE':
            _, site_list, raw_df, date_col, ndvi_col = ndvi_result
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🗺️ Multi-Site CSV Detected")
            st.sidebar.info(
                f"Your CSV contains **{len(site_list)} sites**. Select one to analyse:")
            chosen_site = st.sidebar.selectbox(
                "Select site", site_list, key="site_select",
                help="Your file has multiple sites — choose one to run analysis on.")
            ndvi_df = _filter_ndvi_site(raw_df, date_col, ndvi_col, chosen_site)
            if len(ndvi_df) == 0:
                st.error(f"❌ No valid NDVI rows for site '{chosen_site}' after parsing."); return
            st.sidebar.success(f"✅ Site: **{chosen_site}** — {len(ndvi_df)} obs loaded")
        else:
            ndvi_df = ndvi_result

        met_file.seek(0)
        met_df, raw_params, met_err = parse_nasa_power(met_file)
        if met_df is None: st.error(f"❌ Met: {met_err}"); return

    met_df     = add_derived_features(met_df, season_start_month=cfg['start_month'])
    all_params = [c for c in met_df.columns
                  if c not in {'Date','YEAR','MO','DY','DOY','LON','LAT','ELEV'}
                  and pd.api.types.is_numeric_dtype(met_df[c])]
    derived = [p for p in all_params if p not in raw_params]

    st.sidebar.success(f"✅ NDVI: {len(ndvi_df)} rows  |  {ndvi_df['Date'].dt.year.nunique()} years")
    st.sidebar.success(f"✅ Met: {len(raw_params)} raw parameters detected")
    if derived: st.sidebar.info(
        f"🔧 {len(derived)} derived features computed from your met data:\n\n"
        f"**GDD_5 / GDD_10** — daily growing degree days (base 5°C / 10°C)\n\n"
        f"**GDD_cum** — cumulative GDD, resets at season start month (not Jan 1)\n\n"
        f"**DTR** — diurnal temperature range (Tmax−Tmin)\n\n"
        f"**VPD** — vapour pressure deficit\n\n"
        f"**SPEI_proxy** — rainfall minus PET (drought index)\n\n"
        f"**log_precip** — log-transformed rainfall\n\n"
        f"**MSI** — moisture stress index (precip/soil wetness)"
    )

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab5 = st.tabs([
        "🔬 Training & Models", "📊 Correlations & Met", "🔮 Predict",
        "📖 Technical Guide"
    ])
    icons = {'SOS':'🌱','POS':'🌿','EOS':'🍂'}

    # ══ TAB 1 — TRAINING ══════════════════════════════════════
    with tab1:
        st.markdown("### 🔬 Phenology Extraction & Model Training")

        with st.spinner("Extracting phenology from NDVI (5-day interpolation + SG smoothing + amplitude threshold)…"):
            pheno_df, pheno_err = extract_phenology(
                ndvi_df, season_type,
                threshold_override=threshold_pct_override,
                eos_threshold_override=eos_threshold_pct,
                cfg=cfg)

        if pheno_df is None: st.error(f"❌ Phenology extraction: {pheno_err}"); return
        n_seasons = len(pheno_df)
        if n_seasons < 3:
            st.error(f"❌ Only {n_seasons} season(s) detected. Minimum 3 required. Upload ≥3 years of NDVI.")
            return

        st.success(f"✅ **{n_seasons} growing seasons** extracted")

        c_left, c_right = st.columns([2,1])
        with c_left: st.pyplot(plot_ndvi_phenology(ndvi_df, pheno_df))
        with c_right:
            st.markdown("**Extracted dates:**")
            # Build a clean display table with actual calendar dates
            display_rows = []
            for _, row in pheno_df.iterrows():
                sos_d = row.get('SOS_Date')
                pos_d = row.get('POS_Date')
                eos_d = row.get('EOS_Date')
                display_rows.append({
                    'Year':     int(row['Year']),
                    'SOS':      pd.Timestamp(sos_d).strftime('%b %d') if pd.notna(sos_d) else '—',
                    'SOS_DOY':  int(row.get('SOS_DOY', 0)),
                    'POS':      pd.Timestamp(pos_d).strftime('%b %d') if pd.notna(pos_d) else '—',
                    'POS_DOY':  int(row.get('POS_DOY', 0)),
                    'EOS':      pd.Timestamp(eos_d).strftime('%b %d %Y') if pd.notna(eos_d) else '—',
                    'EOS_DOY':  int(row.get('EOS_DOY', 0)),
                    'LOS (d)':  int(row.get('LOS_Days', 0)),
                    'Peak NDVI': round(float(row.get('Peak_NDVI', 0)), 3),
                })
            dc_df = pd.DataFrame(display_rows)
            st.dataframe(dc_df, use_container_width=True, height=300)

        fig_t = plot_pheno_trends(pheno_df)
        if fig_t: st.pyplot(fig_t)

        # Method explanation box
        st.markdown(
            '<div class="info-box"><b>📐 Method:</b> Phenological metrics were extracted using an '
            '<b>amplitude-based threshold method</b>. NDVI was first interpolated to a regular '
            '<b>5-day grid</b>, then smoothed with the <b>Savitzky–Golay filter</b>. '
            'The seasonal NDVI amplitude A = NDVI_max − NDVI_min was computed per season. '
            'SOS = first date NDVI ≥ NDVI_min + threshold% × A; '
            'EOS = last date NDVI ≥ same threshold; '
            'POS = maximum NDVI between SOS and EOS. '
            'Thresholds are computed independently for each growing season.</div>',
            unsafe_allow_html=True)

        # Train
        model_label = regression_model_sel
        with st.spinner(f"Building training features and fitting {model_label}…"):
            train_df = make_training_features(pheno_df, met_df, all_params)
            predictor = UniversalPredictor()
            predictor.train(train_df, all_params, model_key=regression_model_key)

        st.session_state.update({
            'pheno_df':pheno_df, 'met_df':met_df, 'train_df':train_df,
            'predictor':predictor, 'all_params':all_params,
            'raw_params':raw_params, 'ndvi_df':ndvi_df,
            'regression_model_key': regression_model_key})

        # Performance cards
        st.markdown("---")
        st.markdown(f"### 📊 Model Performance  (LOO Cross-Validated R²)  —  {model_label}")
        c1, c2, c3 = st.columns(3)

        def _card(col, ev):
            if ev not in predictor._fits:
                col.markdown(f'<div class="metric-box"><h3>{icons[ev]} {ev}</h3>'
                             f'<h1 style="color:#607D8B">—</h1><p>Need ≥ 3 seasons</p></div>',
                             unsafe_allow_html=True); return
            fit=predictor._fits[ev]; r2=predictor.r2.get(ev,0); mae=predictor.mae.get(ev,0); n=predictor.n_seasons.get(ev,0)
            if fit['mode']=='mean':
                col.markdown(f'<div class="metric-box"><h3>{icons[ev]} {ev}</h3>'
                             f'<h1 style="color:#607D8B">0.0%</h1>'
                             f'<p>No feature |r|≥{MIN_CORR_THRESHOLD}<br>Prediction = mean DOY ≈ {fit["mean_doy"]:.0f}<br>MAE = ±{mae:.1f} d</p></div>',
                             unsafe_allow_html=True)
            else:
                clr='#1B5E20' if r2>0.6 else '#E65100' if r2>0.3 else '#B71C1C'
                feats = fit.get('features', [])
                feat_str = ', '.join(feats) if feats else '—'
                n_feats = len(feats)
                mkey = fit.get('model_key', 'ridge')
                model_tag = {'ridge':'Ridge','loess':'LOESS','poly2':'Poly-2','poly3':'Poly-3'}.get(mkey, mkey)
                col.markdown(f'<div class="metric-box"><h3>{icons[ev]} {ev}</h3>'
                             f'<h1 style="color:{clr}">{r2*100:.1f}%</h1>'
                             f'<p>{model_tag} | LOO | {n} seasons<br>'
                             f'<b>{n_feats} feature{"s" if n_feats!=1 else ""}:</b> {feat_str}<br>'
                             f'Best |r|={fit["best_r"]:.3f} | MAE = ±{mae:.1f} d</p></div>',
                             unsafe_allow_html=True)

        _card(c1,'SOS'); _card(c2,'POS'); _card(c3,'EOS')

        if n_seasons < 7:
            st.markdown(f'<div class="warn-box">⚠️ <b>{n_seasons} seasons</b> — small training set. '
                        f'R² 0.3–0.6 is realistic. Accuracy increases substantially with ≥10 seasons.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="good-box">✅ Good number of seasons — model estimates are reliable.</div>',
                        unsafe_allow_html=True)

        # ── Fitted Equations + Feature Tables ────────────────────
        st.markdown("---")
        st.markdown("### 📐 Fitted Equations")
        st.caption("Equations are fitted exclusively to your uploaded data — no hard-coded coefficients.")

        ev_tab_sos, ev_tab_pos, ev_tab_eos = st.tabs(
            [f"{icons['SOS']} SOS — Green-up",
             f"{icons['POS']} POS — Peak",
             f"{icons['EOS']} EOS — Senescence"])

        for ui_tab, ev in zip([ev_tab_sos, ev_tab_pos, ev_tab_eos], ['SOS', 'POS', 'EOS']):
            with ui_tab:
                sm_val = cfg['start_month']
                raw_eq = predictor.equation_str(ev, season_start_month=sm_val)
                eq_html = raw_eq.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
                st.markdown(f'<div class="eq-box">{eq_html}</div>', unsafe_allow_html=True)

                ct_display = predictor.corr_table_for_display(ev)
                if not ct_display.empty:
                    st.markdown(
                        f"**Meteorological features ranked by |Pearson r| with {ev} timing**  "
                        f"· Only |r| ≥ {MIN_CORR_THRESHOLD} features enter the model  "
                        f"· p-values and significance stars: see the **Correlations** tab")

                    def _style_role(val):
                        if val.startswith('✅'):  return 'background-color:#C8E6C9;color:#1B5E20;font-weight:600'
                        if val.startswith('⛔'):  return 'background-color:#FFCCBC;color:#BF360C;font-weight:600'
                        if val.startswith('➖'):  return 'color:#555'
                        return 'color:#999'

                    styled = (ct_display.style
                              .background_gradient(subset=['Pearson r'], cmap='RdYlGn', vmin=-1, vmax=1)
                              .background_gradient(subset=['|r|'], cmap='Greens', vmin=0, vmax=1)
                              .applymap(_style_role, subset=['Role'])
                              .format({'Pearson r': '{:+.3f}', '|r|': '{:.3f}'})
                              .set_properties(**{'font-size': '0.84rem'}))
                    st.dataframe(styled, use_container_width=True, hide_index=True)

        # Obs vs Pred
        fig_s = plot_obs_vs_pred(predictor, train_df)
        if fig_s:
            st.markdown("---")
            st.markdown("### 📉 Observed vs Predicted DOY")
            st.pyplot(fig_s)

        st.markdown("---")
        dl=[c for c in ['Year','SOS_DOY','SOS_Method','POS_DOY','EOS_DOY','LOS_Days','Peak_NDVI','Amplitude']
            if c in pheno_df.columns]
        st.download_button("📥 Download Phenology Table (CSV)", pheno_df[dl].to_csv(index=False),
                           "phenology_extracted.csv", "text/csv")

    # ══ TAB 2 — CORRELATIONS ═══════════════════════════════════
    with tab2:
        st.markdown("### 📊 Feature Correlations & Meteorological Parameters")

        st.markdown("""
<div style='background:#E3F2FD;padding:16px 20px;border-radius:12px;border-left:5px solid #1565C0;margin-bottom:16px'>
<b>🔗 How your uploaded meteorological parameters drive the model — end-to-end flow:</b><br><br>
<b>Step 1 — Your NASA POWER CSV</b> is parsed and ALL numeric columns are used
(T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, RH2M, GWETTOP, GWETROOT, GWETPROF, WS2M, ALLSKY_SFC_SW_DWN …)<br><br>
<b>Step 2 — Derived features are computed automatically</b> from those raw columns:
<code>GDD_5, GDD_10, GDD_cum, DTR (= T2M_MAX − T2M_MIN), VPD, SPEI_proxy, log_precip, MSI</code><br><br>
<b>Step 3 — For each phenological event (SOS / POS / EOS)</b>, the app computes the
<b>Pearson r</b> between every feature and the event timing (DOY). This is what you see in the
bar chart and heatmap below.<br><br>
<b>Step 4 — Feature selection:</b> only features with <b>|Pearson r| ≥ 0.40</b> are allowed into the model.
Coloured bars = entered the model · Grey bars = below threshold, excluded.<br><br>
<b>Step 5 — Ridge Regression</b> is fitted using the selected feature(s).
The scatter plots on the right show the best single predictor vs observed event DOY —
<b>this is exactly the relationship the model is using to make predictions.</b>
</div>
        """, unsafe_allow_html=True)

        raw_params_ss2 = st.session_state.get('raw_params', [])
        met_df_check   = st.session_state.get('met_df')
        if raw_params_ss2 and met_df_check is not None:
            all_met2 = [c for c in met_df_check.columns
                       if c not in {'Date','YEAR','MO','DY','DOY','LON','LAT','ELEV'}
                       and pd.api.types.is_numeric_dtype(met_df_check[c])]
            derived_ss2 = [p for p in all_met2 if p not in raw_params_ss2]

            col_r2, col_d2 = st.columns(2)
            with col_r2:
                st.markdown(f"**📡 {len(raw_params_ss2)} raw parameters from your CSV:**")
                st.code("  ".join(raw_params_ss2), language=None)
            with col_d2:
                st.markdown(f"**🔧 {len(derived_ss2)} derived features computed automatically:**")
                st.code("  ".join(derived_ss2) if derived_ss2 else "(none — add T2M_MIN/MAX for DTR, VPD etc.)", language=None)

            _param_desc = {
                "T2M":              "Mean air temp at 2m (°C) — growing degree accumulation, POS timing",
                "T2M_MIN":          "Min air temp at 2m (°C) — pre-monsoon warmth, SOS bud-break trigger",
                "T2M_MAX":          "Max air temp at 2m (°C) — heat stress, canopy maturity",
                "PRECTOTCORR":      "Corrected precipitation (mm/day) — monsoon trigger for SOS; soil recharge",
                "RH2M":             "Relative humidity at 2m (%) — monsoon arrival signal, evergreen SOS",
                "GWETTOP":          "Surface soil wetness 0–5cm (0–1) — immediate moisture for leaf flush",
                "GWETROOT":         "Root zone soil wetness (0–1) — deep reserves; delays EOS in dry season",
                "GWETPROF":         "Profile soil moisture (0–1) — full column water; links to WUE",
                "WS2M":             "Wind speed at 2m (m/s) — NE monsoon drying winds drive senescence",
                "ALLSKY_SFC_SW_DWN":"Incoming solar radiation (MJ/m²/day) — POS driver at high altitudes",
                "GDD_5":            "Growing degree days base 5°C (derived) — cold-adapted species phenology",
                "GDD_10":           "Growing degree days base 10°C (derived) — tropical/subtropical phenology",
                "GDD_cum":          "Cumulative GDD since season start (derived) — heat accumulation for POS",
                "DTR":              "Diurnal Temperature Range (derived) — proxy for clear-sky / dryness",
                "VPD":              "Vapour Pressure Deficit (derived) — atmospheric dryness stress",
                "SPEI_proxy":       "Rainfall minus PET (derived) — drought/wetness index",
                "log_precip":       "log(1+PRECTOTCORR) (derived) — linearises skewed rainfall distribution",
                "MSI":              "Moisture Stress Index (derived) — precip/soil wetness ratio",
            }
            present_params2 = [p for p in all_met2 if p in _param_desc]
            if present_params2:
                with st.expander(f"📋 What does each parameter mean in phenology context? ({len(present_params2)} detected)", expanded=False):
                    rows2 = [{"Parameter": p, "Description": _param_desc[p]} for p in present_params2]
                    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            f'<div class="fix-box"><b>Reading the charts below:</b> Pearson |r| ≥ {MIN_CORR_THRESHOLD} required to enter the model. '
            f'<b>Coloured bars</b> = used in Ridge regression (entered model) · <b>grey</b> = excluded. '
            f'The scatter plots on the right show the actual met value vs observed DOY — '
            f'this IS the relationship powering each model equation in Tab 1.</div>',
            unsafe_allow_html=True)

        predictor = st.session_state.get('predictor')
        pheno_df_ss = st.session_state.get('pheno_df')
        if predictor is None:
            st.info("Train models first — upload files and visit the 🔬 Training tab.")
        else:
            # ── Rich correlation summary (bars + heatmap + scatters) ──
            fig_corr = plot_correlation_summary(predictor, pheno_df_ss)
            if fig_corr:
                st.pyplot(fig_corr, use_container_width=True)

            # ── Per-event detailed correlation tables ──────────────
            st.markdown("---")
            st.markdown("#### 📋 Detailed Correlation Tables")
            st.caption(
                "Pearson r and significance stars are identical to the heatmap above — "
                "both read from the same computed source.  "
                "** p < 0.05  ·  * p < 0.10  ·  (blank) p ≥ 0.10")
            icons_ev = {'SOS': '🌱', 'POS': '🌿', 'EOS': '🍂'}
            c1, c2, c3 = st.columns(3)
            for col_st, ev in zip([c1, c2, c3], ['SOS', 'POS', 'EOS']):
                with col_st:
                    st.markdown(f"**{icons_ev[ev]} {ev}**")
                    ct = predictor.corr_tables.get(ev)
                    if ct is not None and len(ct):
                        # Build display version: r, |r|, significance (no raw p number)
                        def _sig(p):
                            if p < 0.05:  return '**'
                            if p < 0.10:  return '*'
                            return ''
                        disp = ct[['Feature','Pearson_r','|r|']].copy()
                        disp['Sig'] = ct['p_value'].apply(_sig)
                        disp = disp.rename(columns={'Pearson_r': 'Pearson r'})
                        styled_ct = (disp.style
                                     .background_gradient(subset=['Pearson r'], cmap='RdYlGn', vmin=-1, vmax=1)
                                     .background_gradient(subset=['|r|'], cmap='Greens', vmin=0, vmax=1)
                                     .format({'Pearson r': '{:+.3f}', '|r|': '{:.3f}'})
                                     .set_properties(**{'font-size': '0.82rem'}))
                        st.dataframe(styled_ct, use_container_width=True,
                                     hide_index=True, height=320)
                    else:
                        st.info("No correlation data.")

        # ── Met parameters with NDVI overlay ──────────────────────
        st.markdown("---")
        st.markdown("#### 📈 NDVI + Meteorology — Year-by-Year Growing Seasons")
        st.caption("Each season: Top panel = Air parameters (RH, Tmax, Tmin, Precip) · Bottom panel = Soil parameters (GWETTOP, GWETROOT, GWETPROF, Wind)")
        met_df_ss  = st.session_state.get('met_df')
        raw_params_ss = st.session_state.get('raw_params', [])
        ndvi_df_ss = st.session_state.get('ndvi_df')
        if met_df_ss is not None and raw_params_ss and ndvi_df_ss is not None:
            figs_list = plot_met_with_ndvi(met_df_ss, ndvi_df_ss, raw_params_ss,
                                           season_cfg=cfg,
                                           pheno_df=pheno_df_ss,
                                           predictor=st.session_state.get('predictor'))
            if figs_list:
                for season_label, fig_m in figs_list:
                    st.markdown(f"**Growing Season {season_label}**")
                    st.pyplot(fig_m, use_container_width=True)
                    plt.close(fig_m)
            else:
                st.info("No complete seasons found in the data.")
        else:
            st.info("Upload files and train models to see the meteorological overview.")

    # ══ TAB 3 — PREDICT ════════════════════════════════════════
    with tab3:
        st.markdown("### 🔮 Predict SOS / POS / EOS / LOS for Any Year")
        predictor=st.session_state.get('predictor'); train_df=st.session_state.get('train_df')
        if predictor is None: st.info("Train models first (Tab 1)."); return

        pred_sm = cfg['start_month']

        # ── Build per-event feature inputs ────────────────────
        # Each event uses its OWN features — show them grouped by event
        # so the user knows exactly which value to enter for which prediction.
        ev_feats = {}
        for ev in ['SOS','POS','EOS']:
            fit = predictor._fits.get(ev, {})
            feats = [f for f in fit.get('features', []) if fit.get('mode') == 'ridge']
            if feats:
                ev_feats[ev] = feats

        if not ev_feats:
            st.warning("No usable features found. Try more seasons or a different forest type."); return

        st.markdown("""
<div class="info-box">
<b>📋 How prediction works:</b> Each phenological event (SOS / POS / EOS) has its own
regression equation with its own meteorological driver(s). Enter the expected
<b>15-day pre-event conditions</b> for each event separately below.
Defaults are pre-filled from training data means — change them to forecast future scenarios.
</div>""", unsafe_allow_html=True)

        ev_labels_full = {'SOS': '🌱 SOS — Green-up start', 
                          'POS': '🌿 POS — Peak greenness',
                          'EOS': '🍂 EOS — Senescence end'}
        ev_colors_hex  = {'SOS': '#E8F5E9', 'POS': '#E3F2FD', 'EOS': '#FFF3E0'}
        ev_border_hex  = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#E65100'}

        # ev_inputs[ev][feature] = value  — scoped per event so shared feature names
        # (e.g. log_precip used by both SOS and EOS) never overwrite each other.
        ev_inputs = {ev: {} for ev in ['SOS', 'POS', 'EOS']}
        # Compute typical event date range from training data (to guide user)
        pheno_ss = st.session_state.get('pheno_df')
        mo_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                    7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

        for ev, feats in ev_feats.items():
            fit = predictor._fits[ev]
            r2  = predictor.r2.get(ev, 0)
            mae = predictor.mae.get(ev, 0)
            eq_short = predictor.equation_str(ev, season_start_month=pred_sm).split('\n')[0]

            # Compute typical event month from historical data
            ev_month_hint = ""
            if pheno_ss is not None and f'{ev}_Date' in pheno_ss.columns:
                ev_dates = pheno_ss[f'{ev}_Date'].dropna()
                if len(ev_dates) > 0:
                    median_month = int(ev_dates.dt.month.median())
                    median_day   = int(ev_dates.dt.day.median())
                    ev_month_hint = (f"  |  Historically occurs ~{mo_names[median_month]} {median_day} "
                                     f"(±{mae:.0f} d MAE)  →  enter 15-day average ending ~{mo_names[median_month]} {median_day}")

            st.markdown(
                f"<div style='background:{ev_colors_hex[ev]};padding:14px 18px;border-radius:10px;"
                f"border-left:5px solid {ev_border_hex[ev]};margin:10px 0'>"
                f"<b>{ev_labels_full[ev]}</b>&nbsp;&nbsp;"
                f"<span style='font-size:0.82rem;color:#555'>R²(LOO)={r2:.3f} | MAE=±{mae:.1f}d"
                f"{ev_month_hint}</span><br>"
                f"<code style='font-size:0.78rem'>{eq_short}</code>"
                f"</div>", unsafe_allow_html=True)

            ncols = max(1, len(feats))
            col_list = st.columns(min(ncols, 4))
            for idx, f in enumerate(feats):
                default = 0.0
                if train_df is not None and f in train_df.columns:
                    ev_sub = train_df[train_df['Event'] == ev]
                    vals = ev_sub[f].dropna() if len(ev_sub) > 0 else train_df[f].dropna()
                    if len(vals): default = float(vals.mean())
                is_sum = any(k in f.upper() for k in ['PREC','GDD','LOG_P','SPEI'])
                # Build context-aware help text
                timing_tip = ""
                if pheno_ss is not None and f'{ev}_Date' in pheno_ss.columns:
                    ev_dates = pheno_ss[f'{ev}_Date'].dropna()
                    if len(ev_dates) > 0:
                        median_month = int(ev_dates.dt.month.median())
                        timing_tip = (f" Sample this from the 15 days before expected {ev} "
                                      f"(typically ~{mo_names[median_month]}).")
                with col_list[idx % len(col_list)]:
                    ev_inputs[ev][f] = st.number_input(
                        f"{f}  [{ev}]", value=round(default, 3), format="%.3f",
                        key=f"inp_{ev}_{f}",
                        help=(f"{'15-day sum' if is_sum else '15-day mean'} of {f} "
                              f"before expected {ev}.{timing_tip} "
                              f"Training mean: {default:.3f}"))

        st.markdown("---")
        pred_year=st.number_input("Prediction year", 2024, 2040, 2026)

        if st.button("🚀 Predict Phenology Events", type="primary"):
            results={}
            for ev in ['SOS','POS','EOS']:
                # Pass only this event's feature values — fixes shared-feature-name overwrite bug
                res=predictor.predict(ev_inputs.get(ev, {}), ev, pred_year, season_start_month=pred_sm)
                if res: results[ev]=res

            if results:
                # ── Enforce ecological ordering: SOS < POS < EOS ──────────
                cfg_pred = cfg
                pheno_ss = st.session_state.get('pheno_df')
                order_warnings = []

                # Use rel_days (season-relative) for ordering checks — cross-year safe
                if 'SOS' in results and 'POS' in results:
                    if results['POS']['rel_days'] <= results['SOS']['rel_days']:
                        if pheno_ss is not None and 'POS_Target' in pheno_ss.columns:
                            corrected_rel = int(round(pheno_ss['POS_Target'].mean()))
                        else:
                            corrected_rel = results['SOS']['rel_days'] + 90
                        corrected_rel = max(corrected_rel, results['SOS']['rel_days'] + 14)
                        new_date = datetime(pred_year, pred_sm, 1) + timedelta(days=corrected_rel)
                        results['POS']['rel_days'] = corrected_rel
                        results['POS']['doy'] = new_date.timetuple().tm_yday
                        results['POS']['date'] = new_date
                        order_warnings.append(f"⚠️ **POS** predicted before SOS — corrected to mean training POS (~{new_date.strftime('%b %d')})")

                if 'POS' in results and 'EOS' in results:
                    if results['EOS']['rel_days'] <= results['POS']['rel_days']:
                        if pheno_ss is not None and 'EOS_Target' in pheno_ss.columns:
                            corrected_rel = int(round(pheno_ss['EOS_Target'].mean()))
                        else:
                            corrected_rel = results['POS']['rel_days'] + 90
                        corrected_rel = max(corrected_rel, results['POS']['rel_days'] + 14)
                        new_date = datetime(pred_year, pred_sm, 1) + timedelta(days=corrected_rel)
                        results['EOS']['rel_days'] = corrected_rel
                        results['EOS']['doy'] = new_date.timetuple().tm_yday
                        results['EOS']['date'] = new_date
                        order_warnings.append(f"⚠️ **EOS** predicted before POS — corrected to mean training EOS (~{new_date.strftime('%b %d')})")

                if order_warnings:
                    st.markdown('<div class="warn-box">'
                                '<b>Ecological order correction applied:</b><br>'
                                + '<br>'.join(order_warnings) +
                                '<br><small>This happens when the model extrapolates outside its training range. '
                                'Consider entering more realistic met input values.</small></div>',
                                unsafe_allow_html=True)

                cols=st.columns(len(results))
                for col,(ev,res) in zip(cols, results.items()):
                    actual_yr = res['date'].year
                    yr_label = f"{actual_yr}" if actual_yr != pred_year else str(pred_year)
                    col.markdown(f'<div class="metric-box"><h3>{icons[ev]} {ev}</h3>'
                                 f'<h1>{res["date"].strftime("%b %d")}</h1>'
                                 f'<p>Day {res["doy"]} of {yr_label}<br>'
                                 f'R²(LOO)={res["r2"]:.3f}<br>MAE=±{res["mae"]:.1f} d</p></div>',
                                 unsafe_allow_html=True)

                if 'SOS' in results and 'EOS' in results:
                    sd=results['SOS']['date']; ed=results['EOS']['date']
                    # LOS: handle cross-year seasons (e.g. Jun-May window)
                    if ed >= sd:
                        los = (ed - sd).days
                    else:
                        los = (ed + pd.DateOffset(years=1) - sd).days
                    st.markdown("---")
                    c_los1, c_los2, c_los3 = st.columns(3)
                    c_los1.metric("📏 Length of Season (LOS)", f"{los} days")
                    c_los2.metric("🌱 SOS → 🌿 POS (green-up lag)",
                                  f"{(results['POS']['date']-results['SOS']['date']).days} days"
                                  if 'POS' in results else "—")
                    c_los3.metric("🌿 POS → 🍂 EOS (senescence lag)",
                                  f"{(results['EOS']['date']-results['POS']['date']).days} days"
                                  if 'POS' in results else "—")

                out=pd.DataFrame({'Event':list(results.keys()),
                    'Date':[r['date'].strftime('%Y-%m-%d') for r in results.values()],
                    'DOY':[r['doy'] for r in results.values()],
                    'R²_LOO':[round(r['r2'],3) for r in results.values()],
                    'MAE_days':[round(r['mae'],1) for r in results.values()]})
                st.dataframe(out, use_container_width=True)
                st.download_button("📥 Download Predictions", out.to_csv(index=False),
                                   "predictions.csv","text/csv")

                st.markdown("---")
                st.markdown("**Equations used for this prediction:**")
                for ev in list(results.keys()):
                    raw_eq = predictor.equation_str(ev, season_start_month=cfg['start_month'])
                    eq_line = raw_eq.split('\n')[0]   # show equation line only — clean, no feature table
                    st.markdown(f'<div class="eq-box"><b>{icons[ev]} {ev}:</b>&nbsp;&nbsp;{eq_line}</div>',
                                unsafe_allow_html=True)

    # ══ TAB 5 — TECHNICAL GUIDE ════════════════════════════════
    with tab5:
        st.markdown(f"""
### 📖 Technical Guide

#### Phenology Extraction Method

Phenological metrics are extracted using the **NDVI Amplitude-Based Threshold Method**:

1. **5-day interpolation** — raw NDVI observations are interpolated to a regular 5-day grid to fill temporal gaps.
2. **Savitzky–Golay smoothing** — adaptive window (≈5% of series length, min 7 points) reduces noise.
3. **Per-season amplitude** — for each growing season independently: A = NDVI_max − NDVI_min.
4. **Threshold** = NDVI_min + threshold% × A  (default 10%, adjustable 5–30%).
5. **SOS** = first 5-day step when smoothed NDVI ≥ threshold (after dormancy minimum).
6. **EOS** = last 5-day step when smoothed NDVI ≥ threshold (before next dormancy minimum).
7. **POS** = maximum smoothed NDVI value between SOS and EOS.

Thresholds are computed **independently per season** — inter-annual amplitude variability does not bias detection.

#### Model Architecture

| Component | Detail |
|-----------|--------|
| Feature selection | Pearson |r| ≥ {MIN_CORR_THRESHOLD} required |
| Regression options | **Ridge** (L2, RidgeCV), **LOESS** (locally weighted), **Polynomial deg 2/3** |
| Cross-validation | **Leave-One-Out** (unbiased for small samples) |
| Phenology extraction | NDVI amplitude threshold (per-season) |

#### Available Regression Models

| Model | Best for | Notes |
|-------|---------|-------|
| Ridge Regression | Most cases — reliable, regularised | α auto-tuned via RidgeCV LOO |
| LOESS / LOWESS | Non-linear, unimodal relationships | Uses single best feature; frac=0.75 |
| Polynomial deg 2 | Curved relationships | Ridge-regularised polynomial |
| Polynomial deg 3 | Complex curves | Ridge-regularised; may overfit with few seasons |

#### R² Guide

| R² | Meaning |
|----|---------|
| >0.70 | Strong signal — reliable prediction |
| 0.40–0.70 | Moderate — adequate for trend analysis |
| 0.10–0.40 | Weak — relative comparisons only |
| 0.0 | No feature found — prediction = mean DOY |
| <0 | Below mean baseline — increase seasons |

#### Threshold Sensitivity

The 5–30% range covers the full ecologically relevant spectrum:
- **5–10%** — sensitive; picks up early green-up / late senescence; best for sparse or high-noise NDVI
- **10–15%** — standard recommended range (this method's scientific default is 10%)
- **15–30%** — conservative; detects only peak growing period

#### Data Requirements

| Seasons | Capability |
|---------|-----------|
| 3–4 | Minimum — high uncertainty |
| 5–8 | Adequate — publishable with caveats |
| 9–14 | Good — reliable LOO R² |
| 15+ | Publication-quality |

#### Recommended NASA POWER Parameters

Download daily data from [power.larc.nasa.gov](https://power.larc.nasa.gov/data-access-viewer/):
T2M, T2M_MIN, T2M_MAX, PRECTOTCORR, RH2M, GWETTOP, GWETROOT, ALLSKY_SFC_SW_DWN

#### Scientific Description (for thesis / report)

> *"Phenological metrics were extracted from the smoothed NDVI time series using an amplitude-based threshold method. Prior to extraction, the NDVI time series was interpolated to a regular 5-day grid and smoothed using the Savitzky–Golay filter to reduce sensor noise. The seasonal NDVI amplitude was calculated as the difference between the maximum and minimum NDVI values within each growing cycle. The Start of Season (SOS) was defined as the first time step when NDVI exceeded 10% of the seasonal amplitude above the base value (NDVI_min). Similarly, the End of Season (EOS) was defined as the last time step when NDVI remained above this threshold. The Peak of Season (POS) corresponded to the maximum NDVI value occurring between SOS and EOS. Thresholds were computed independently for each growing season to account for inter-annual amplitude variability."*
        """)


if __name__ == "__main__":
    for k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params','ndvi_df']:
        if k not in st.session_state: st.session_state[k] = None
    main()
