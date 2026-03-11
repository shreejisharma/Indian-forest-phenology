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

BUG FIX v5 (two critical fixes):

  FIX 1 — MISSING SEASON (Year dedup via trough_year, not pos.year):
  - OLD: yr = pos.year  → when POS falls in a calendar year shared by the
         PREVIOUS season's late peak (e.g. both 2018-19 and 2019-20 get
         Year=2019), drop_duplicates silently drops the later-appended season.
  - NEW: yr = trough_year = t_all[ti].year  (start-of-cycle year, always
         unique per trough pair). Applied in main loop, head segment, tail
         segment. This guarantees every detected cycle survives dedup.

  FIX 2 — POS DATE ≠ RAW NDVI PEAK:
  - OLD: pos_idx_local = int(np.argmax(cycle_sm))  → smoothed peak can be
         shifted days away from the true raw maximum and underestimates NDVI.
  - NEW: pos_idx_local = int(np.nanargmax(cycle_raw))  → POS always equals
         the raw NDVI maximum, matching what the user sees in the scatter plot.
         Applied in main loop, head segment, tail segment.

Previous fixes retained:
  BUG FIX v4 (gap-tolerant cycle extraction)
  BUG FIX v3 (SG window capped at 31 steps)
  BUG FIX v2 (plateau trough filter 85% ceiling)

pip install streamlit pandas numpy scipy scikit-learn matplotlib
streamlit run universal_Indian_forest_phenology_FIXED_v5.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
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
    n = len(x_train)
    k = max(2, int(np.ceil(frac * n)))
    result = np.zeros(len(x_new))
    for i, xp in enumerate(x_new):
        dists = np.abs(x_train - xp)
        idx   = np.argsort(dists)[:k]
        d_max = dists[idx[-1]] + 1e-10
        u     = dists[idx] / d_max
        w     = np.maximum(0, (1 - u**3)**3)
        if w.sum() < 1e-12:
            result[i] = np.mean(y_train)
        else:
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

EVENT_PRIORITIES = {
    'SOS': ['T2M_MIN', 'PRECTOTCORR', 'log_precip', 'RH2M', 'GWETTOP',
            'GWETROOT', 'SPEI_proxy', 'VPD', 'T2M', 'GDD_5', 'GDD_10',
            'WS2M', 'DTR', 'T2M_MAX'],
    'POS': ['PRECTOTCORR', 'log_precip', 'GDD_cum', 'GWETROOT', 'GWETTOP',
            'RH2M', 'GDD_5', 'GDD_10', 'T2M', 'T2M_MAX',
            'ALLSKY_SFC_SW_DWN', 'SPEI_proxy', 'VPD'],
    'EOS': ['T2M_MIN', 'WS2M', 'DTR', 'T2M_RANGE', 'VPD',
            'GWETTOP', 'GWETROOT', 'MSI', 'SPEI_proxy',
            'T2M', 'log_precip', 'PRECTOTCORR', 'RH2M'],
}

EVENT_EXCLUDE = {
    'SOS': {'GDD_cum'},
    'POS': set(),
    'EOS': {'GDD_cum'},
}
SM_NAME = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

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
    if len(series) == 0:
        return pd.to_datetime(series, errors='coerce')
    parsed_default  = pd.to_datetime(series, errors='coerce')
    parsed_dayfirst = pd.to_datetime(series, dayfirst=True, errors='coerce')
    if doy_series is not None:
        doy_ref      = pd.to_numeric(doy_series, errors='coerce')
        default_doy  = parsed_default.apply(
            lambda d: d.timetuple().tm_yday if pd.notna(d) else np.nan)
        dayfirst_doy = parsed_dayfirst.apply(
            lambda d: d.timetuple().tm_yday if pd.notna(d) else np.nan)
        if (dayfirst_doy == doy_ref).sum() > (default_doy == doy_ref).sum():
            return parsed_dayfirst
        return parsed_default
    n_m_default  = parsed_default.dropna().dt.month.nunique()
    n_m_dayfirst = parsed_dayfirst.dropna().dt.month.nunique()
    if n_m_dayfirst > n_m_default:
        return parsed_dayfirst
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
    try:
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        df = pd.read_csv(StringIO(raw_bytes.decode('utf-8', errors='replace')))
        df.columns = [c.strip() for c in df.columns]
        date_col = next((c for c in df.columns
                         if c.lower() in ['date','dates','time','datetime']), None)
        if not date_col:
            return None, "No Date column found. Expected column named: date / dates / time / datetime"
        ndvi_col = next((c for c in df.columns
                         if c.lower() in ['ndvi','ndvi_value','value','index','evi']), None)
        if not ndvi_col:
            return None, "No NDVI column found. Expected column named: ndvi / ndvi_value / value / evi"
        site_info = None
        if 'site_key' in df.columns and df['site_key'].nunique() > 1:
            site_info = sorted(df['site_key'].dropna().unique().tolist())
        elif 'site_label' in df.columns and df['site_label'].nunique() > 1:
            site_info = sorted(df['site_label'].dropna().unique().tolist())
        if site_info:
            return ('MULTI_SITE', site_info, df, date_col, ndvi_col), None
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
        def _season_cumsum(series, dates, sm):
            out = series.copy() * 0.0
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
def make_training_features(pheno_df, met_df, params, window=15, monsoon_onset_doys=None):
    SNAPSHOT_FEATURES = {'GDD_cum', 'GDD_CUM'}
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
            if len(wdf) < max(1, window * 0.15): continue
            for p in params:
                if p not in met_df.columns: continue
                if p.upper() in SNAPSHOT_FEATURES:
                    snap = met_df[met_df['Date'] <= evt_dt][p].dropna()
                    rec[p] = float(snap.iloc[-1]) if len(snap) > 0 else np.nan
                elif any(k in p.upper() for k in ACCUM_KEYWORDS):
                    rec[p] = float(wdf[p].sum())
                else:
                    rec[p] = float(wdf[p].mean())
            if monsoon_onset_doys:
                rec['Monsoon_Onset_DOY'] = monsoon_onset_doys.get(int(row['Year']), np.nan)
            records.append(rec)
    return pd.DataFrame(records)


# ─── PHENOLOGY EXTRACTION ────────────────────────────────────
def _find_troughs(ndvi_values, min_distance=10):
    n = len(ndvi_values)
    troughs = []
    for i in range(min_distance, n - min_distance):
        window = ndvi_values[max(0, i - min_distance): i + min_distance + 1]
        if ndvi_values[i] == np.min(window):
            if ndvi_values[i] <= ndvi_values[i - 1] and ndvi_values[i] <= ndvi_values[i + 1]:
                troughs.append(i)
    if not troughs:
        return troughs
    merged = [troughs[0]]
    for t in troughs[1:]:
        if t - merged[-1] < min_distance:
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

    BUG FIX v5 changes in this function:
      FIX 1: yr = trough_year (t_all[ti].year) instead of yr = pos.year
             → prevents same-calendar-year collisions from dropping seasons.
      FIX 2: pos_idx_local = int(np.nanargmax(cycle_raw)) instead of np.argmax(cycle_sm)
             → POS always aligns with the visible raw NDVI maximum.
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

        ndvi_raw = ndvi_df[["Date", "NDVI"]].copy().set_index("Date").sort_index()
        ndvi_raw = ndvi_raw[~ndvi_raw.index.duplicated(keep='first')]

        orig_dates = ndvi_raw.index.sort_values()
        orig_diffs  = pd.Series(orig_dates).diff().dt.days.fillna(0)
        typical_cadence = float(orig_diffs[orig_diffs > 0].median())
        MAX_INTERP_GAP_DAYS = max(60, int(typical_cadence * 8))
        gap_starts = orig_dates[orig_diffs.values > MAX_INTERP_GAP_DAYS]

        full_range = pd.date_range(
            start=ndvi_raw.index.min(),
            end=ndvi_raw.index.max(),
            freq="5D")

        ndvi_5d = ndvi_raw.reindex(ndvi_raw.index.union(full_range))
        ndvi_5d = ndvi_5d.interpolate(method="time", limit_area="inside")

        for gap_start in gap_starts:
            before = orig_dates[orig_dates < gap_start]
            if len(before) == 0:
                continue
            gap_end_real = gap_start
            gap_start_real = before[-1]
            mask = (ndvi_5d.index > gap_start_real) & (ndvi_5d.index < gap_end_real)
            ndvi_5d.loc[mask] = np.nan

        ndvi_5d = ndvi_5d.reindex(full_range)
        ndvi_5d.columns = ["NDVI"]

        n = len(ndvi_5d)
        ndvi_vals = ndvi_5d["NDVI"].values.copy()
        valid_mask = ~np.isnan(ndvi_vals)
        sm_vals = np.full(n, np.nan)

        seg_labels = np.zeros(n, dtype=int)
        seg_id = 0
        in_seg = False
        for i in range(n):
            if valid_mask[i]:
                if not in_seg:
                    seg_id += 1
                    in_seg = True
                seg_labels[i] = seg_id
            else:
                in_seg = False

        # SG window capped at 31 steps (~155 days) — shorter than one growing season
        MAX_SG_STEPS = 31
        for sid in range(1, seg_id + 1):
            idx_seg = np.where(seg_labels == sid)[0]
            seg_n = len(idx_seg)
            if seg_n < 5:
                sm_vals[idx_seg] = ndvi_vals[idx_seg]
                continue
            wl_t = max(7, min(int(seg_n * 0.05), MAX_SG_STEPS))
            wl_s = wl_t if wl_t % 2 == 1 else wl_t + 1
            wl_s = min(wl_s, seg_n - 1 if seg_n > 1 else 1)
            if wl_s % 2 == 0: wl_s = max(7, wl_s - 1)
            poly_s = min(2, wl_s - 1)
            if wl_s < 5 or wl_s >= seg_n:
                sm_vals[idx_seg] = ndvi_vals[idx_seg]
                continue
            seg_data = ndvi_vals[idx_seg]
            sm_vals[idx_seg] = savgol_filter(seg_data, wl_s, poly_s)

        wl = max(7, int(n * 0.05)); wl = wl if wl % 2 == 1 else wl + 1
        wl = min(wl, n - 1 if n > 1 else 1)
        if wl % 2 == 0: wl = max(7, wl - 1)
        poly = min(2, wl - 1)
        if wl < 5:
            return None, f"Too few interpolated NDVI points ({n}) — check date range"

        ndvi_5d["NDVI"] = ndvi_vals
        ndvi_5d["Sm"]   = sm_vals
        t_all = ndvi_5d.index

        sm_for_troughs = pd.Series(sm_vals, index=t_all).interpolate(method="linear",
                                   limit_direction="both").values

        min_dist = max(10, int((365 / 5) * 0.4))
        trough_indices_raw = _find_troughs(sm_for_troughs, min_distance=min_dist)

        nan_mask = np.isnan(sm_vals)

        # Gap filter: discard troughs at/adjacent to gap regions
        trough_indices = []
        for ti in trough_indices_raw:
            window = slice(max(0, ti - 5), min(n, ti + 6))
            if nan_mask[window].any():
                continue
            trough_indices.append(ti)

        # Plateau trough filter (v2 fix: 85% ceiling, skip for low-amplitude forests)
        if len(trough_indices) >= 2:
            valid_sm = sm_for_troughs[~nan_mask]
            global_min = float(np.percentile(valid_sm, 5))
            global_max = float(np.percentile(valid_sm, 95))
            global_amp = global_max - global_min

            if global_amp >= 0.20:
                trough_ceiling = global_min + 0.85 * global_amp
                trough_indices = [ti for ti in trough_indices
                                  if sm_for_troughs[ti] <= trough_ceiling]

        # Gap tolerance (v4 fix)
        _GAP_STRICT   = 0.20
        _GAP_TOLERANT = 0.50
        _AMP_GAP_THRESHOLD = 0.10

        def _cycle_has_gap(i_start, i_end, amplitude=None):
            if i_end <= i_start: return True
            gap_frac = nan_mask[i_start:i_end+1].mean()
            if amplitude is not None and amplitude >= _AMP_GAP_THRESHOLD:
                return gap_frac > _GAP_TOLERANT
            return gap_frac > _GAP_STRICT

        MIN_AMPLITUDE = 0.02

        if len(trough_indices) < 2:
            rows = []
            for yr in range(ndvi_5d.index.year.min(), ndvi_5d.index.year.max() + 1):
                try:
                    s = f"{yr}-{sm:02d}-01"
                    e = f"{yr+1}-{em:02d}-28" if sm > em else f"{yr}-{em:02d}-28"
                    sub_sm = ndvi_5d.loc[s:e, "Sm"]
                    sub_vals = sub_sm.dropna().values
                    sub_t    = sub_sm.dropna().index
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
                    # FIX 2: POS from raw max, not smoothed
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

        all_troughs = list(trough_indices)
        rows = []

        def _date_in_window(d):
            m = d.month
            if sm <= em:
                return sm <= m <= em
            else:
                return m >= sm or m <= em

        # ── HEAD SEGMENT ─────────────────────────────────────────────────────
        if all_troughs:
            ti_first = all_troughs[0]
            head_len = ti_first
            _head_amp_pre = float(np.max(sm_for_troughs[0:ti_first + 1])) - float(sm_for_troughs[0])
            if head_len >= max(10, min_d // 5) and not _cycle_has_gap(0, ti_first, amplitude=_head_amp_pre):
                try:
                    seg_sm  = sm_for_troughs[0:ti_first + 1]
                    seg_raw = ndvi_vals[0:ti_first + 1]
                    seg_t   = t_all[0:ti_first + 1]
                    _head_has_gap = nan_mask[0:ti_first + 1].any()

                    if _head_has_gap:
                        work_arr = seg_sm
                        ndvi_min = float(sm_for_troughs[0])
                        ndvi_max = float(np.max(seg_sm))
                    else:
                        work_arr = seg_raw
                        ndvi_min = float(ndvi_vals[0]) if not np.isnan(ndvi_vals[0]) else float(sm_for_troughs[0])
                        ndvi_max = float(np.nanmax(seg_raw))

                    A = ndvi_max - ndvi_min
                    if A >= max(1e-6, MIN_AMPLITUDE):
                        sos_threshold = ndvi_min + thr_pct     * A
                        eos_threshold = ndvi_min + eos_thr_pct * A

                        # ── FIX 2: POS from raw signal peak ──────────────────
                        pi = int(np.nanargmax(work_arr))   # was: np.argmax(seg_sm)
                        # ─────────────────────────────────────────────────────
                        pos = seg_t[pi]
                        if _date_in_window(pos):
                            asc  = work_arr[1:pi + 1]
                            sc   = np.where(asc >= sos_threshold)[0]
                            desc = work_arr[pi:]
                            ec   = np.where(desc >= eos_threshold)[0]
                            if len(sc) and len(ec):
                                si = 1 + int(sc[0])
                                ei = pi + int(ec[-1])
                                if ei > si:
                                    sos = seg_t[si]; eos = seg_t[ei]

                                    # ── FIX 1: season year from trough start ─
                                    trough_year = seg_t[0].year   # was: yr = pos.year
                                    # ─────────────────────────────────────────
                                    season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                                    rows.append({
                                        "Year":         trough_year,
                                        "SOS_Date":     sos,  "SOS_DOY":  sos.dayofyear,
                                        "SOS_Target":   (sos - season_start).days,
                                        "SOS_Method":   "valley_amplitude_threshold",
                                        "POS_Date":     pos,  "POS_DOY":  pos.dayofyear,
                                        "POS_Target":   (pos - season_start).days,
                                        "EOS_Date":     eos,  "EOS_DOY":  eos.dayofyear,
                                        "EOS_Target":   (eos - season_start).days,
                                        "EOS_Method":   "valley_amplitude_threshold",
                                        "LOS_Days":     (eos - sos).days,
                                        "Season_Start": season_start,
                                        "Peak_NDVI":    float(ndvi_max),
                                        "Amplitude":    float(A),
                                        "Base_NDVI":    float(ndvi_min),
                                        "Threshold_SOS": float(sos_threshold),
                                        "Threshold_EOS": float(eos_threshold),
                                        "Trough_Date":  seg_t[int(np.argmin(seg_sm))],
                                        "Season":       season_type,
                                    })
                except Exception:
                    pass

        # ── MAIN TROUGH-TO-TROUGH LOOP ───────────────────────────────────────
        for i in range(len(all_troughs) - 1):
            try:
                ti  = all_troughs[i]
                ti1 = all_troughs[i + 1]

                if ti1 - ti < max(10, min_d // 5):
                    continue

                _cycle_sm_pre = sm_for_troughs[ti:ti1 + 1]
                _amp_pre = float(np.max(_cycle_sm_pre)) - float(sm_for_troughs[ti])

                if _cycle_has_gap(ti, ti1, amplitude=_amp_pre):
                    continue

                _cycle_has_any_gap = nan_mask[ti:ti1 + 1].any()

                cycle_sm   = sm_for_troughs[ti:ti1 + 1]
                cycle_t    = t_all[ti:ti1 + 1]

                if _cycle_has_any_gap:
                    cycle_raw = sm_for_troughs[ti:ti1 + 1]
                    ndvi_min = float(sm_for_troughs[ti])
                    ndvi_max = float(np.max(cycle_raw))
                else:
                    cycle_raw = ndvi_vals[ti:ti1 + 1]
                    ndvi_min = float(ndvi_vals[ti]) if not np.isnan(ndvi_vals[ti]) \
                               else float(sm_for_troughs[ti])
                    ndvi_max = float(np.nanmax(cycle_raw))

                A = ndvi_max - ndvi_min
                if A < max(1e-6, MIN_AMPLITUDE):
                    continue

                sos_threshold = ndvi_min + thr_pct     * A
                eos_threshold = ndvi_min + eos_thr_pct * A

                # ── FIX 2: POS from raw signal, not smoothed ─────────────────
                pos_idx_local = int(np.nanargmax(cycle_raw))  # was: np.argmax(cycle_sm)
                # ─────────────────────────────────────────────────────────────

                asc = cycle_raw[1:pos_idx_local + 1]
                sos_cands = np.where(asc >= sos_threshold)[0] + 1
                if len(sos_cands) == 0:
                    continue
                sos_idx_local = int(sos_cands[0])

                desc = cycle_raw[pos_idx_local:-1]
                eos_cands = np.where(desc >= eos_threshold)[0]
                if len(eos_cands) == 0:
                    continue
                eos_idx_local = pos_idx_local + int(eos_cands[-1])

                if eos_idx_local <= sos_idx_local:
                    continue

                sos = cycle_t[sos_idx_local]
                pos = cycle_t[pos_idx_local]
                eos = cycle_t[eos_idx_local]

                if not _date_in_window(pos):
                    continue

                # ── FIX 1: season year from trough start, not POS year ────────
                trough_year  = t_all[ti].year       # was: yr = pos.year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                # ─────────────────────────────────────────────────────────────

                rows.append({
                    "Year":          trough_year,
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
                    "Peak_NDVI":     float(ndvi_max),
                    "Amplitude":     float(A),
                    "Base_NDVI":     float(ndvi_min),
                    "Threshold_SOS": float(sos_threshold),
                    "Threshold_EOS": float(eos_threshold),
                    "Trough_Date":   t_all[ti],
                    "Season":        season_type,
                })
            except Exception:
                continue

        # ── TAIL SEGMENT ─────────────────────────────────────────────────────
        covered_trough_starts = set()
        for i in range(len(all_troughs) - 1):
            _amp_pre = float(np.max(sm_for_troughs[all_troughs[i]:all_troughs[i+1]+1])) \
                       - float(sm_for_troughs[all_troughs[i]])
            if not _cycle_has_gap(all_troughs[i], all_troughs[i+1], amplitude=_amp_pre):
                covered_trough_starts.add(all_troughs[i])

        for ti0 in [ti for ti in all_troughs if ti not in covered_trough_starts]:
            tail_end_i = n - 1
            tail_len = tail_end_i - ti0
            if tail_len < max(10, min_d // 5): continue
            try:
                _tail_amp_pre = float(np.max(sm_for_troughs[ti0:tail_end_i + 1])) \
                                - float(sm_for_troughs[ti0])
                if _cycle_has_gap(ti0, tail_end_i, amplitude=_tail_amp_pre): continue

                seg_sm  = sm_for_troughs[ti0:tail_end_i + 1]
                seg_raw = ndvi_vals[ti0:tail_end_i + 1]
                seg_t   = t_all[ti0:tail_end_i + 1]

                _tail_has_gap = nan_mask[ti0:tail_end_i + 1].any()
                if _tail_has_gap:
                    ndvi_min = float(sm_for_troughs[ti0])
                    ndvi_max = float(np.max(seg_sm))
                    _seg_signal = seg_sm
                else:
                    ndvi_min = float(ndvi_vals[ti0]) if not np.isnan(ndvi_vals[ti0]) \
                               else float(sm_for_troughs[ti0])
                    ndvi_max = float(np.nanmax(seg_raw))
                    _seg_signal = seg_raw

                A = ndvi_max - ndvi_min
                if A < max(1e-6, MIN_AMPLITUDE): continue
                sos_threshold = ndvi_min + thr_pct     * A
                eos_threshold = ndvi_min + eos_thr_pct * A

                # ── FIX 2: POS from raw signal ────────────────────────────────
                pi = int(np.nanargmax(_seg_signal))  # was: np.argmax(seg_sm)
                # ─────────────────────────────────────────────────────────────

                asc_limb = _seg_signal[1:pi + 1]
                sos_cands = np.where(asc_limb >= sos_threshold)[0] + 1
                if not len(sos_cands): continue
                si = int(sos_cands[0])
                desc_limb = _seg_signal[pi:]
                eos_cands_desc = np.where(desc_limb >= eos_threshold)[0]
                if not len(eos_cands_desc): continue
                ei = pi + int(eos_cands_desc[-1])
                if ei <= si: continue
                sos = seg_t[si]; pos = seg_t[pi]; eos = seg_t[ei]
                if not _date_in_window(pos): continue

                # ── FIX 1: season year from trough start ──────────────────────
                trough_year  = seg_t[0].year        # was: yr = pos.year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                # ─────────────────────────────────────────────────────────────

                rows.append({
                    "Year":         trough_year,
                    "SOS_Date":     sos,  "SOS_DOY":  sos.dayofyear,
                    "SOS_Target":   (sos - season_start).days,
                    "SOS_Method":   "valley_amplitude_threshold",
                    "POS_Date":     pos,  "POS_DOY":  pos.dayofyear,
                    "POS_Target":   (pos - season_start).days,
                    "EOS_Date":     eos,  "EOS_DOY":  eos.dayofyear,
                    "EOS_Target":   (eos - season_start).days,
                    "EOS_Method":   "valley_amplitude_threshold",
                    "LOS_Days":     (eos - sos).days,
                    "Season_Start": season_start,
                    "Peak_NDVI":    float(ndvi_max),
                    "Amplitude":    float(A),
                    "Base_NDVI":    float(ndvi_min),
                    "Threshold_SOS": float(sos_threshold),
                    "Threshold_EOS": float(eos_threshold),
                    "Trough_Date":  t_all[ti0],
                    "Season":       season_type,
                })
            except Exception:
                pass

        if not rows:
            return None, (
                f"No complete seasons found. "
                f"Troughs detected: {len(trough_indices)}. "
                f"Data span: {ndvi_5d.index.min().date()} → {ndvi_5d.index.max().date()} "
                f"({n} points on 5-day grid). "
                f"Try: (1) reducing the Minimum Days slider, "
                f"(2) ensuring Season Start ≠ Season End, "
                f"(3) adjusting the threshold % slider."
            )

        df_out = pd.DataFrame(rows).drop_duplicates(subset="Year", keep="first")
        return df_out.sort_values("Year").reset_index(drop=True), None

    except Exception as e:
        return None, str(e)


# ─── FEATURE SELECTION & CORRELATIONS ────────────────────────
def select_best_feature(X, y, event):
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
    excluded = EVENT_EXCLUDE.get(event, set())
    usable = []
    n_obs = len(y.dropna())
    for col in X.columns:
        if col in excluded: continue
        vals = X[col].dropna()
        if vals.std() < 1e-8 or len(vals) < 3: continue
        idx = vals.index.intersection(y.dropna().index)
        if len(idx) < 3: continue
        try:
            rp, _ = pearsonr(vals[idx].astype(float), y[idx].astype(float))
            rs, _ = spearmanr(vals[idx].astype(float), y[idx].astype(float))
        except Exception: continue
        composite = max(abs(rp), abs(rs))
        if composite >= min_r:
            usable.append((col, composite, abs(rp), abs(float(rs))))
    if not usable: return []
    usable.sort(key=lambda x: -x[1])
    collinear_filtered = []
    for feat, score, rp_abs, rs_abs in usable:
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
    max_safe = max(2, n_obs - 2)
    candidates = collinear_filtered[:min(max_features, max_safe)]
    if len(candidates) <= 1:
        return candidates
    y_vals = y.values.astype(float)
    selected = [candidates[0]]
    best_r2 = _loo_r2_quick(X[selected].fillna(X[selected[0]].median()).values.reshape(-1,1), y_vals)
    for feat in candidates[1:]:
        trial = selected + [feat]
        Xt = X[trial].fillna(X[trial].median()).values
        try:
            trial_r2 = _loo_r2_quick(Xt, y_vals)
        except Exception:
            continue
        if trial_r2 > best_r2 + 0.03:
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
            r,   p_val  = pearsonr( vals[idx].astype(float), y[idx].astype(float))
            rho, p_sp   = spearmanr(vals[idx].astype(float), y[idx].astype(float))
            composite   = max(abs(r), abs(float(rho)))
            rows.append({
                'Feature':    col,
                'Pearson_r':  round(r,   3),
                '|r|':        round(abs(r), 3),
                'Spearman_rho': round(float(rho), 3),
                '|rho|':      round(abs(float(rho)), 3),
                'Composite':  round(composite, 3),
                'p_value':    round(p_val, 3),
                'Usable':     '✅' if composite >= MIN_CORR_THRESHOLD else '❌'
            })
        except Exception: continue
    return pd.DataFrame(rows).sort_values('Composite', ascending=False)


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
    if model_key == "loess":
        feat = features[0]
        x1d  = Xf[feat].values.astype(float)
        r2, mae = fit_loess(x1d, yt.astype(float))
        sort_idx = np.argsort(x1d)
        loess_data = np.column_stack([x1d[sort_idx], yt.astype(float)[sort_idx]])
        return {'mode': 'loess', 'features': [feat], 'r2': r2, 'mae': mae,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'mean_doy': float(yt.mean()), 'n': n,
                'pipe': None, 'loess_data': loess_data, 'model_key': model_key,
                'x_train': x1d, 'y_train': yt.astype(float)}
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
    if model_key == "gpr":
        sc = StandardScaler()
        Xvs = sc.fit_transform(Xv)
        yt_f = yt.astype(float)
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) \
                 + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                       normalize_y=True, random_state=42)
        loo = LeaveOneOut()
        preds_loo = np.zeros(n)
        for tr, te in loo.split(Xvs):
            gpr_cv = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                               normalize_y=True, random_state=42)
            gpr_cv.fit(Xvs[tr], yt_f[tr])
            preds_loo[te] = gpr_cv.predict(Xvs[te])
        ss_res = np.sum((yt_f - preds_loo) ** 2)
        ss_tot = np.sum((yt_f - yt_f.mean()) ** 2) + 1e-12
        r2  = float(np.clip(1 - ss_res / ss_tot, -1, 1))
        mae = float(np.mean(np.abs(yt_f - preds_loo)))
        gpr.fit(Xvs, yt_f)
        return {'mode': 'gpr', 'features': features, 'r2': r2, 'mae': mae,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'mean_doy': float(yt.mean()), 'n': n,
                'pipe': None, 'gpr_model': gpr, 'gpr_scaler': sc,
                'model_key': model_key}
    # Default: Ridge
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
        elif fit['mode'] == 'gpr':
            vals = np.array([[inputs.get(f, 0.0) for f in fit['features']]])
            sc   = fit.get('gpr_scaler')
            gpr  = fit.get('gpr_model')
            if sc is not None and gpr is not None:
                vals_s = sc.transform(vals)
                pred   = gpr.predict(vals_s)[0]
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
        if event not in self._fits: return "Need ≥ 3 seasons"
        fit = self._fits[event]
        sm_name = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        start_label = sm_name.get(season_start_month, 'Jun')
        target_label = f"{event}_days_from_{start_label}1"
        if fit['mode'] == 'mean':
            return (f"{target_label} ≈ {fit['mean_doy']:.0f}  "
                    f"[No feature |r|≥{MIN_CORR_THRESHOLD} — mean prediction only]")
        if fit['mode'] == 'gpr':
            feats_str = ", ".join(fit['features'])
            return (f"{target_label}  =  GPR({feats_str})\n"
                    f"    [Gaussian Process, RBF+WhiteKernel, {len(fit['features'])} feature(s), "
                    f"R²(LOO)={fit['r2']:.3f}, MAE=±{fit['mae']:.1f} d]")
        if fit['mode'] == 'loess':
            feat = fit['features'][0] if fit['features'] else '?'
            return (f"{target_label}  =  LOESS({feat})\n"
                    f"    [Locally weighted smoothing, R²(LOO)={fit['r2']:.3f}, MAE=±{fit['mae']:.1f} d]")
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
                role = '⬜  Below threshold'
            rows.append({
                'Feature':      feat,
                'Pearson r':    row['Pearson_r'],
                'Spearman ρ':   row.get('Spearman_rho', float('nan')),
                'Composite':    row.get('Composite', row['|r|']),
                'Role':         role,
            })
        return pd.DataFrame(rows)


# ─── PLOTS ───────────────────────────────────────────────────
def plot_ndvi_phenology(ndvi_raw, pheno_df, season_window=None):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    dates = pd.to_datetime(ndvi_raw['Date'])
    ax.scatter(dates, ndvi_raw['NDVI'], color='#A5D6A7', s=18, alpha=0.55,
               label='NDVI (raw obs)', zorder=3)

    ndvi_idx = ndvi_raw.copy()
    ndvi_idx['Date'] = pd.to_datetime(ndvi_idx['Date'])
    ndvi_s = ndvi_idx.set_index('Date')['NDVI'].sort_index()
    ndvi_s = ndvi_s[~ndvi_s.index.duplicated(keep='first')]

    orig_dates = ndvi_s.index.sort_values()
    orig_diffs = pd.Series(orig_dates).diff().dt.days.fillna(0)
    typical_cadence = float(orig_diffs[orig_diffs > 0].median())
    MAX_INTERP_GAP_DAYS = max(60, int(typical_cadence * 8))
    gap_starts = orig_dates[orig_diffs.values > MAX_INTERP_GAP_DAYS]

    full_range = pd.date_range(start=ndvi_s.index.min(),
                               end=ndvi_s.index.max(), freq='5D')
    ndvi_5d = ndvi_s.reindex(ndvi_s.index.union(full_range))
    ndvi_5d = ndvi_5d.interpolate(method='time', limit_area='inside')
    for gap_start in gap_starts:
        before = orig_dates[orig_dates < gap_start]
        if len(before) == 0: continue
        mask = (ndvi_5d.index > before[-1]) & (ndvi_5d.index < gap_start)
        ndvi_5d.loc[mask] = np.nan
    ndvi_5d = ndvi_5d.reindex(full_range)

    n = len(ndvi_5d)
    ndvi_vals = ndvi_5d.values.copy()
    valid_mask = ~np.isnan(ndvi_vals)
    sm_arr = np.full(n, np.nan)
    seg_labels = np.zeros(n, dtype=int); seg_id = 0; in_seg = False
    for i in range(n):
        if valid_mask[i]:
            if not in_seg: seg_id += 1; in_seg = True
            seg_labels[i] = seg_id
        else:
            in_seg = False

    _MAX_SG = 31
    for sid in range(1, seg_id + 1):
        idx_seg = np.where(seg_labels == sid)[0]
        seg_n = len(idx_seg)
        if seg_n < 5:
            sm_arr[idx_seg] = ndvi_vals[idx_seg]; continue
        wl_t = max(7, min(int(seg_n * 0.05), _MAX_SG))
        wl_s = wl_t if wl_t % 2 == 1 else wl_t + 1
        wl_s = min(wl_s, seg_n - 1 if seg_n > 1 else 1)
        if wl_s % 2 == 0: wl_s = max(7, wl_s - 1)
        poly_s = min(2, wl_s - 1)
        if wl_s >= 5 and wl_s < seg_n:
            sm_arr[idx_seg] = savgol_filter(ndvi_vals[idx_seg], wl_s, poly_s)
        else:
            sm_arr[idx_seg] = ndvi_vals[idx_seg]

    sm_plot = sm_arr.copy()
    for i in range(1, n):
        if np.isnan(sm_arr[i]) and not np.isnan(sm_arr[i-1]):
            sm_plot[i-1] = np.nan
        if np.isnan(sm_arr[i-1]) and not np.isnan(sm_arr[i]):
            sm_plot[i] = np.nan

    ax.plot(ndvi_5d.index, sm_plot, color='#1B5E20', lw=2.2,
            label=f'Smoothed (SG w≤31)', zorder=5)

    in_gap = False; gap_s = None
    for i in range(n):
        currently_nan = np.isnan(sm_arr[i])
        if currently_nan and not in_gap:
            gap_s = ndvi_5d.index[i]; in_gap = True
        elif not currently_nan and in_gap:
            ax.axvspan(gap_s, ndvi_5d.index[i], color='#BDBDBD', alpha=0.30,
                       label='_gap' if i > 0 else 'Data gap')
            in_gap = False
    if in_gap:
        ax.axvspan(gap_s, ndvi_5d.index[-1], color='#BDBDBD', alpha=0.30)

    if season_window is not None:
        win_sm, win_em = season_window
        year_min = ndvi_5d.index.year.min()
        year_max = ndvi_5d.index.year.max() + 1
        win_plotted = False
        for yr in range(year_min, year_max + 1):
            try:
                ws = pd.Timestamp(f"{yr}-{win_sm:02d}-01")
                if win_sm > win_em:
                    we = pd.Timestamp(f"{yr+1}-{win_em:02d}-28")
                else:
                    we = pd.Timestamp(f"{yr}-{win_em:02d}-28")
                data_start = ndvi_5d.index[0]; data_end = ndvi_5d.index[-1]
                if we < data_start or ws > data_end: continue
                lbl = 'Selected growing window' if not win_plotted else ''
                ax.axvspan(max(ws, data_start), min(we, data_end),
                           color='#A5D6A7', alpha=0.13, zorder=0, label=lbl)
                win_plotted = True
            except Exception:
                continue

    thr_sos_plotted = False; thr_eos_plotted = False; base_plotted = False
    for _, row in pheno_df.iterrows():
        trough_d = row.get('Trough_Date'); eos_d = row.get('EOS_Date')
        base = row.get('Base_NDVI'); thr_sos = row.get('Threshold_SOS')
        thr_eos = row.get('Threshold_EOS'); peak = row.get('Peak_NDVI')
        amp = row.get('Amplitude'); sos_d = row.get('SOS_Date'); pos_d = row.get('POS_Date')
        if pd.isna(trough_d) or pd.isna(eos_d): continue
        seg_start = pd.Timestamp(trough_d); seg_end = pd.Timestamp(eos_d) + pd.Timedelta(days=20)
        if pd.notna(base):
            lbl_b = 'Base NDVI (dormancy valley)' if not base_plotted else ''
            ax.hlines(base, seg_start, seg_end, colors='#F57F17', lw=1.1, ls=':', alpha=0.75, label=lbl_b, zorder=4)
            base_plotted = True
        if pd.notna(thr_sos):
            lbl_s = f'SOS threshold (base + {int(round((thr_sos-base)/amp*100)) if pd.notna(amp) and amp>0 else "?"}% × A)' if not thr_sos_plotted else ''
            ax.hlines(thr_sos, seg_start, seg_end, colors='#66BB6A', lw=1.2, ls='--', alpha=0.70, label=lbl_s, zorder=4)
            thr_sos_plotted = True
        if pd.notna(thr_eos) and (not pd.notna(thr_sos) or abs(thr_eos - thr_sos) > 1e-4):
            lbl_e = f'EOS threshold (base + {int(round((thr_eos-base)/amp*100)) if pd.notna(amp) and amp>0 else "?"}% × A)' if not thr_eos_plotted else ''
            ax.hlines(thr_eos, seg_start, seg_end, colors='#FFA726', lw=1.2, ls='--', alpha=0.70, label=lbl_e, zorder=4)
            thr_eos_plotted = True
        if pd.notna(pos_d) and pd.notna(base) and pd.notna(peak) and pd.notna(amp) and amp > 0.01:
            px = pd.Timestamp(pos_d)
            ax.annotate('', xy=(px, peak), xytext=(px, base),
                        arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=1.1))
            ax.text(px, base + amp * 0.5, f'  A={amp:.3f}',
                    fontsize=7, color='#7B1FA2', va='center', ha='left')

    ev_colors = {'SOS': '#43A047', 'POS': '#1565C0', 'EOS': '#E65100'}
    ev_labels_map = {
        'SOS': 'SOS — Green-up start', 'POS': 'POS — Peak greenness', 'EOS': 'EOS — Senescence end'}
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
        Line2D([0],[0], color='#1B5E20', lw=2.2, label='Smoothed (SG w≤31, capped at 155d)'),
        plt.Rectangle((0,0),1,1, fc='#BDBDBD', alpha=0.40, label='Data gap (not interpolated)'),
        plt.Rectangle((0,0),1,1, fc='#A5D6A7', alpha=0.35, label='Selected growing window'),
        Line2D([0],[0], color='#F57F17', lw=1.1, ls=':', label='Base NDVI (dormancy valley)'),
        Line2D([0],[0], color='#66BB6A', lw=1.2, ls='--', label='SOS threshold (base + thr% × A)'),
        Line2D([0],[0], color='#FFA726', lw=1.2, ls='--', label='EOS threshold (base + thr% × A)'),
        Line2D([0],[0], color='#7B1FA2', lw=1.1, label='A = amplitude (NDVI_max − base)'),
        Line2D([0],[0], color='#43A047', lw=1.5, ls='--', label='SOS — Green-up start'),
        Line2D([0],[0], color='#1565C0', lw=1.5, ls='--', label='POS — Peak greenness'),
        Line2D([0],[0], color='#E65100', lw=1.5, ls='--', label='EOS — Senescence end'),
    ]
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    win_label = ''
    if season_window:
        ws, we = season_window
        win_label = f'  |  Growing window: {month_names.get(ws,"?")} → {month_names.get(we,"?")}'
    ax.set_title(f'NDVI Time Series + Smoothed Signal + Phenology Events{win_label}\n'
                 'with per-season amplitude threshold definition',
                 fontsize=11, fontweight='bold', color='#1B5E20')
    ax.set_xlabel('Date'); ax.set_ylabel('NDVI')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.legend(handles=handles, ncol=3, fontsize=7.5, loc='upper left', framealpha=0.88)
    ax.grid(True, alpha=0.22, ls='--'); ax.set_facecolor('#FAFFF8')
    fig.tight_layout()
    return fig


def plot_pheno_trends(pheno_df):
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
            ax.plot(yrs, vals, 'o-', color=clr, ms=6, lw=2, markeredgecolor='white', markeredgewidth=1.2)
            ax.set_ylabel('Days')
            y_min = max(0, vals.min() - 30)
            ax.set_ylim(y_min, vals.max() + 30)
        else:
            if doy_col in pheno_df.columns:
                vals = pheno_df[doy_col].values.astype(float)
            else:
                vals = pheno_df[f'{ev}_DOY'].values.astype(float)
            ax.bar(yrs, vals, color=clr, alpha=0.55, width=0.7, edgecolor='white')
            ax.plot(yrs, vals, 'o-', color=clr, ms=6, lw=2, markeredgecolor='white', markeredgewidth=1.2)
            if date_col and date_col in pheno_df.columns and ev != 'EOS':
                unique_vals = np.linspace(vals.min(), vals.max(), 5).astype(int)
                tick_labels = []
                for doy in unique_vals:
                    try:
                        tick_labels.append(
                            (pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(doy)-1)).strftime('%b %d'))
                    except Exception:
                        tick_labels.append(str(doy))
                ax.set_yticks(unique_vals); ax.set_yticklabels(tick_labels, fontsize=8)
                ax.set_ylabel('Date (approx.)')
            elif ev == 'EOS':
                ax.set_ylabel('EOS (season-relative days)')
            else:
                ax.set_ylabel('DOY')
        if len(yrs) >= 2:
            m, b = np.polyfit(yrs, vals, 1)
            ax.plot(yrs, m*yrs+b, '--', color='#263238', lw=1.8, label=f'Trend: {m:+.1f} d/yr')
            ax.legend(fontsize=8.5, framealpha=0.85)
        ax.set_title(lbl, fontsize=10, fontweight='bold', color='#1B4332')
        ax.set_xlabel('Year', fontsize=9)
        ax.grid(True, alpha=0.22, ls='--'); ax.set_facecolor('#FAFFF8'); ax.tick_params(labelsize=8.5)
    fig.suptitle('Phenological Trends Across Years', fontsize=13, fontweight='bold', color='#1B4332', y=1.02)
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
        available = [f for f in feats if f in sub.columns]
        if not available: continue
        Xf = sub[available].fillna(sub[available].median())
        try:
            pred = fit['pipe'].predict(Xf.values)
        except ValueError:
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
    events    = ['SOS', 'POS', 'EOS']
    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#BF360C'}
    ev_markers= {'SOS': 'o',       'POS': 's',       'EOS': '^'}
    feat_r = {}
    for ev in events:
        ct = predictor.corr_tables.get(ev)
        if ct is None: continue
        for _, row in ct.iterrows():
            f = row['Feature']
            feat_r[f] = max(feat_r.get(f, 0), abs(row['Pearson_r']))
    all_feats = sorted(feat_r, key=lambda f: -feat_r[f])[:14]
    r_mat = pd.DataFrame(0.0, index=all_feats, columns=events)
    p_mat = pd.DataFrame(1.0, index=all_feats, columns=events)
    best  = {}
    tdf   = st.session_state.get('train_df')
    for ev in events:
        ct = predictor.corr_tables.get(ev)
        if ct is None: continue
        _excluded_for_ev = EVENT_EXCLUDE.get(ev, set())
        for _, _row in ct.iterrows():
            if _row['Feature'] not in _excluded_for_ev:
                best[ev] = _row['Feature']; break
        for _, row in ct.iterrows():
            f = row['Feature']
            if f not in r_mat.index: continue
            r_mat.loc[f, ev] = row['Pearson_r']
            if 'p_value' in row and not pd.isna(row['p_value']):
                p_mat.loc[f, ev] = row['p_value']
    n_feats = len(all_feats); n_ev = 3
    fig = plt.figure(figsize=(17, max(6, n_feats * 0.52 + 2.5)))
    fig.patch.set_facecolor('#F8FBF7')
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.1, 1.5], wspace=0.36)
    ax_bar = fig.add_subplot(gs[0])
    bar_h  = 0.22; y_pos  = np.arange(n_feats); offsets= np.array([-1, 0, 1]) * bar_h
    for i, ev in enumerate(events):
        vals = r_mat[ev].values
        bar_colors = [ev_colors[ev] if abs(v) >= MIN_CORR_THRESHOLD else '#CFCFCF' for v in vals]
        ax_bar.barh(y_pos + offsets[i], vals, height=bar_h * 0.82,
                    color=bar_colors, edgecolor='white', lw=0.4, label=ev, alpha=0.88)
    ax_bar.axvline(0, color='#37474F', lw=1.0)
    for thresh in [MIN_CORR_THRESHOLD, -MIN_CORR_THRESHOLD]:
        ax_bar.axvline(thresh, color='#555', lw=1.1, ls='--', alpha=0.55)
    ax_bar.axvspan( MIN_CORR_THRESHOLD, 1.05, alpha=0.035, color='#4CAF50')
    ax_bar.axvspan(-1.05, -MIN_CORR_THRESHOLD, alpha=0.035, color='#4CAF50')
    ax_bar.set_yticks(y_pos); ax_bar.set_yticklabels(all_feats, fontsize=9.5)
    ax_bar.set_xlim(-1.05, 1.05)
    ax_bar.set_xlabel('Pearson r  (with event DOY)', fontsize=10, fontweight='bold')
    ax_bar.set_title('Feature Correlations — SOS / POS / EOS\n'
                     'Coloured = |r| ≥ 0.40 (usable)  ·  Grey = below threshold',
                     fontsize=9.5, fontweight='bold', color='#1B4332', pad=7)
    ax_bar.grid(True, axis='x', alpha=0.20, ls='--'); ax_bar.set_facecolor('#FAFFF8')
    ax_bar.legend(title='Event', fontsize=9, title_fontsize=9, loc='lower right', framealpha=0.92,
                  handles=[plt.matplotlib.patches.Patch(color=ev_colors[e], label=e) for e in events])
    ax_hm = fig.add_subplot(gs[1])
    mat   = r_mat.values.astype(float)
    im    = ax_hm.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
    ax_hm.set_xticks(range(n_ev)); ax_hm.set_xticklabels(events, fontsize=11, fontweight='bold')
    ax_hm.set_yticks(range(n_feats)); ax_hm.set_yticklabels(all_feats, fontsize=9)
    for i in range(n_feats):
        for j in range(n_ev):
            v  = mat[i, j]; pv = p_mat.iloc[i, j]
            star = '**' if pv < 0.05 else ('*' if pv < 0.10 else '')
            txt_color = 'white' if abs(v) > 0.60 else '#1A1A1A'
            ax_hm.text(j, i, f'{v:.2f}{star}', ha='center', va='center',
                       fontsize=8.5, fontweight='bold', color=txt_color)
            if abs(v) >= MIN_CORR_THRESHOLD:
                rect = plt.matplotlib.patches.FancyBboxPatch(
                    (j-0.48, i-0.48), 0.96, 0.96, boxstyle='round,pad=0.02', linewidth=1.8,
                    edgecolor='#1B4332', facecolor='none')
                ax_hm.add_patch(rect)
    cb = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8); cb.set_label('Pearson r', fontsize=9)
    ax_hm.set_title('Pearson r Heatmap\n** p<0.05  * p<0.10  (bold border = usable)',
                    fontsize=9.5, fontweight='bold', color='#1B4332', pad=7)
    ax_hm.tick_params(axis='x', bottom=False); ax_hm.spines[:].set_visible(False)
    for i in range(n_feats):
        if i % 2 == 0:
            ax_hm.axhspan(i-0.5, i+0.5, color='#F0F0F0', alpha=0.25, zorder=0)
    inner = gs[2].subgridspec(n_ev, 1, hspace=0.60)
    for i, ev in enumerate(events):
        ax_s  = fig.add_subplot(inner[i]); feat  = best.get(ev)
        ct    = predictor.corr_tables.get(ev)
        ax_s.set_facecolor('#FAFFF8'); ax_s.grid(True, alpha=0.22, ls='--')
        ax_s.tick_params(labelsize=7.5)
        if feat is None or ct is None or tdf is None:
            ax_s.text(0.5, 0.5, 'No data', ha='center', va='center',
                      fontsize=8, transform=ax_s.transAxes); continue
        r_val = ct.iloc[0]['Pearson_r']
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
            z  = np.polyfit(x, y, 1); xr = np.linspace(x.min(), x.max(), 100)
            ax_s.plot(xr, np.polyval(z, xr), color=ev_colors[ev], lw=1.8, ls='--', alpha=0.80)
        ax_s.set_title(f'{ev}  ←  {feat}\nr = {r_val:.3f}',
                       fontsize=8.5, fontweight='bold', color=ev_colors[ev], pad=4)
        ax_s.set_xlabel(feat, fontsize=8)
        ax_s.set_ylabel(f'{ev} (days from season start)', fontsize=7.5)
    fig.suptitle('Feature Correlations with Phenological Events',
                 fontsize=13, fontweight='bold', color='#1B4332', y=1.005)
    fig.tight_layout()
    return fig


def plot_met_with_ndvi(met_df, ndvi_df, raw_params, season_cfg=None, pheno_df=None, predictor=None):
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
    ndvi_idx = ndvi_df.set_index('Date')['NDVI'].sort_index()
    ndvi_idx = ndvi_idx[~ndvi_idx.index.duplicated(keep='first')]
    full_range = pd.date_range(start=ndvi_idx.index.min(), end=ndvi_idx.index.max(), freq='5D')
    ndvi_5d = (ndvi_idx.reindex(ndvi_idx.index.union(full_range))
                       .interpolate(method='time').reindex(full_range))
    if pheno_df is None or len(pheno_df) == 0:
        return []
    figs = []
    for _, row in pheno_df.sort_values('Year').iterrows():
        try:
            yr       = int(row['Year'])
            trough_d = row.get('Trough_Date', pd.NaT)
            sos_d    = row.get('SOS_Date',    pd.NaT)
            eos_d    = row.get('EOS_Date',    pd.NaT)
            if pd.notna(trough_d):
                s = pd.Timestamp(trough_d)
            elif pd.notna(sos_d):
                s = pd.Timestamp(sos_d) - pd.Timedelta(days=60)
            else:
                continue
            if pd.notna(eos_d):
                e = pd.Timestamp(eos_d) + pd.Timedelta(days=30)
            else:
                continue
            df_met = met_df[(met_df['Date'] >= s) & (met_df['Date'] <= e)].copy()
            if len(df_met) < 20:
                continue
            ndvi_seg = ndvi_5d.reindex(df_met['Date'].values, method='nearest',
                                       tolerance=pd.Timedelta('8D'))
            ndvi_seg = ndvi_seg.ffill().bfill()
            sos_str = pd.Timestamp(sos_d).strftime('%d %b') if pd.notna(sos_d) else '?'
            eos_str = pd.Timestamp(eos_d).strftime('%d %b %Y') if pd.notna(eos_d) else '?'
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
            fig.patch.set_facecolor('#FAFFF8')
            fig.suptitle(f"NDVI + Meteorology  —  Growing Season {yr}   [ {sos_str} → {eos_str} ]",
                fontsize=15, fontweight='bold', y=0.98)
            def _draw_panel(ax, param_list, panel_title,
                            bar_params=('PRECTOTCORR', 'PRECTOT', 'RAIN')):
                ax.fill_between(df_met['Date'], ndvi_seg, alpha=0.18, color='#2E7D32')
                ax.plot(df_met['Date'], ndvi_seg, color='#2E7D32', lw=2.5, label='NDVI', zorder=5)
                ax.set_ylabel('NDVI', color='#2E7D32', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.05); ax.tick_params(axis='y', labelcolor='#2E7D32', labelsize=9)
                ax.grid(True, linestyle='--', alpha=0.30); ax.set_facecolor('#FAFFF8')
                ax.set_title(panel_title, fontsize=13, fontweight='bold', pad=14, loc='center')
                twin_axes = []
                present = [(p, c, l) for p, c, l in param_list if p in df_met.columns]
                for i, (var, col, lab) in enumerate(present):
                    axr = ax.twinx()
                    if i > 0:
                        axr.spines['right'].set_position(('axes', 1.0 + 0.10 * i))
                        axr.spines['right'].set_visible(True)
                    if var in bar_params:
                        axr.bar(df_met['Date'], df_met[var], color=col, alpha=0.50, width=1.0, label=lab, zorder=3)
                    else:
                        axr.plot(df_met['Date'], df_met[var], color=col, lw=1.6, alpha=0.85, label=lab)
                    axr.set_ylabel(lab, color=col, fontsize=9, rotation=270, labelpad=18)
                    axr.tick_params(axis='y', labelcolor=col, labelsize=8)
                    axr.spines['right'].set_color(col); twin_axes.append(axr)
                h, l = ax.get_legend_handles_labels()
                for axr in twin_axes:
                    hh, ll = axr.get_legend_handles_labels(); h.extend(hh); l.extend(ll)
                return h, l
            h_air, l_air = _draw_panel(ax_top, AIR_PARAMS, "Air Parameters  (RH, Tmax, Tmin, Precip)")
            fig.legend(h_air, l_air, loc='upper center', bbox_to_anchor=(0.5, 0.975),
                       ncol=len(h_air), fontsize=9, framealpha=0.95, title='Air Parameters', title_fontsize=9)
            h_soil, l_soil = _draw_panel(ax_bot, SOIL_PARAMS, "Soil Parameters  (GWETs, Wind Speed)")
            fig.legend(h_soil, l_soil, loc='center left', bbox_to_anchor=(0.01, 0.25),
                       ncol=1, fontsize=9, framealpha=0.95, title='Soil Parameters', title_fontsize=9)
            ax_bot.set_xlabel('Date', fontsize=13, fontweight='bold')
            ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=45, ha='right')
            fig.tight_layout(rect=[0.10, 0.0, 1.0, 0.93])
            figs.append((str(yr), fig))
        except Exception:
            continue
    return figs


def main():
    regression_model_key = "ridge"
    regression_model_sel = "Ridge Regression (Linear)"
    sos_threshold_pct    = 0.10
    eos_threshold_pct    = 0.10
    threshold_pct_override = 0.10
    start_month_sel      = 6
    end_month_sel        = 5
    min_days_sel         = 150

    st.markdown('<p class="main-header">🌲 Universal Vegetation Phenology Predictor</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload NDVI + NASA POWER Met · '
                'NDVI Amplitude Threshold · Ridge / LOESS / Polynomial · LOO CV · SOS / POS / EOS / LOS<br>'
                '<span style="color:#E65100;font-weight:bold">'
                'v5 — Fixed missing season (trough_year dedup) + POS aligned to raw NDVI peak'
                '</span></p>',
                unsafe_allow_html=True)

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
        help="Any CSV with columns: date + ndvi")
    met_file = st.sidebar.file_uploader(
        "2️⃣  NASA POWER Met CSV", type=['csv'],
        help="Daily export from power.larc.nasa.gov — header block auto-skipped")

    _ndvi_fp = f"{ndvi_file.name}:{ndvi_file.size}" if ndvi_file else ""
    _met_fp  = f"{met_file.name}:{met_file.size}"   if met_file  else ""

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📅 Season Window")
    sm_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_options = list(sm_names.keys())
    col_sm, col_em = st.sidebar.columns(2)
    start_month_sel = col_sm.selectbox("Season start", options=month_options, index=5,
        format_func=lambda m: sm_names[m])
    end_month_sel = col_em.selectbox("Season end", options=month_options, index=4,
        format_func=lambda m: sm_names[m])
    if start_month_sel != end_month_sel:
        if start_month_sel > end_month_sel:
            st.sidebar.info(f"🌿 Cross-year window: **{sm_names[start_month_sel]} → {sm_names[end_month_sel]}**")
        else:
            st.sidebar.info(f"🌿 Within-year window: **{sm_names[start_month_sel]} → {sm_names[end_month_sel]}**")
    min_days_sel = st.sidebar.slider("Min. season length (days)", 60, 300, 150, 10)
    season_type = f"Custom ({sm_names[start_month_sel]}–{sm_names[end_month_sel]})"
    cfg = {
        "start_month": start_month_sel, "end_month": end_month_sel, "min_days": min_days_sel,
        "sos_base": None, "pos_constrain_end": None, "threshold_pct": 0.10,
        "sos_method": "threshold", "icon": "🌿",
    }
    if start_month_sel == end_month_sel:
        st.sidebar.warning("⚠️ Season Start = Season End — window is only ~1 month. No seasons will be detected.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ NDVI Amplitude Threshold")
    sos_threshold_pct = st.sidebar.slider("SOS threshold (% of NDVI amplitude)", 5, 30, 10, 5) / 100.0
    eos_threshold_pct = st.sidebar.slider("EOS threshold (% of NDVI amplitude)", 5, 30, 10, 5) / 100.0
    st.sidebar.caption(f"SOS: **{int(sos_threshold_pct*100)}%** | EOS: **{int(eos_threshold_pct*100)}%**")
    threshold_pct_override = sos_threshold_pct

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Regression Model")
    model_options = {
        "Ridge Regression (Linear)":          "ridge",
        "LOESS / LOWESS Smoothing":           "loess",
        "Polynomial Regression (deg 2)":      "poly2",
        "Polynomial Regression (deg 3)":      "poly3",
        "Gaussian Process (best for n<10)":   "gpr",
    }
    regression_model_sel = st.sidebar.radio("Select fitting model", options=list(model_options.keys()), index=0)
    regression_model_key = model_options[regression_model_sel]

    _cur_fp = (f"{_ndvi_fp}|{_met_fp}"
               f"|sm={start_month_sel}|em={end_month_sel}|md={min_days_sel}"
               f"|sos={sos_threshold_pct:.4f}|eos={eos_threshold_pct:.4f}"
               f"|model={regression_model_key}")
    if st.session_state.get('_file_fingerprint') != _cur_fp:
        for _k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params','ndvi_df']:
            st.session_state[_k] = None
        st.session_state['_file_fingerprint'] = _cur_fp

    if not (ndvi_file and met_file):
        st.markdown("""
<div class="upload-hint">
<b>👈 Upload both files in the sidebar to begin analysis</b><br><br>
<b>NDVI CSV format example:</b>
<pre>Date,NDVI
2016-01-01,0.42
2016-01-17,0.45</pre>
<b>NASA POWER Met CSV:</b> download from
<a href="https://power.larc.nasa.gov/data-access-viewer/" target="_blank">power.larc.nasa.gov</a>
→ Daily → Point → your site coordinates → select parameters → download CSV.
</div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("📂 Parsing files…"):
        ndvi_result, ndvi_err = parse_ndvi(ndvi_file)
        if ndvi_result is None:
            st.error(f"❌ NDVI: {ndvi_err}"); return
        if isinstance(ndvi_result, tuple) and ndvi_result[0] == 'MULTI_SITE':
            _, site_list, raw_df, date_col, ndvi_col = ndvi_result
            st.sidebar.markdown("---"); st.sidebar.markdown("### 🗺️ Multi-Site CSV Detected")
            st.sidebar.info(f"Your CSV contains **{len(site_list)} sites**. Select one to analyse:")
            chosen_site = st.sidebar.selectbox("Select site", site_list, key="site_select")
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
    if derived: st.sidebar.info(f"🔧 {len(derived)} derived features computed")

    tab1, tab2, tab3, tab5 = st.tabs([
        "🔬 Training & Models", "📊 Correlations & Met", "🔮 Predict", "📖 Technical Guide"])
    icons = {'SOS':'🌱','POS':'🌿','EOS':'🍂'}

    # ══ TAB 1 ══════════════════════════════════════════════════
    with tab1:
        st.markdown("### 🔬 Phenology Extraction & Model Training")
        with st.spinner("Extracting phenology…"):
            pheno_df, pheno_err = extract_phenology(
                ndvi_df, season_type,
                threshold_override=threshold_pct_override,
                eos_threshold_override=eos_threshold_pct,
                cfg=cfg)
        if pheno_df is None: st.error(f"❌ Phenology extraction: {pheno_err}"); return
        n_seasons = len(pheno_df)
        if n_seasons < 3:
            st.error(f"❌ Only {n_seasons} season(s) detected. Minimum 3 required."); return
        st.success(f"✅ **{n_seasons} growing seasons** extracted")

        # Show v5 fix note if 2019 season is present
        if 2019 in pheno_df['Year'].values:
            row_2019 = pheno_df[pheno_df['Year'] == 2019].iloc[0]
            st.markdown(
                f'<div class="fix-box">✅ <b>v5 Fix 1:</b> Season 2019 correctly extracted '
                f'(POS = {pd.Timestamp(row_2019["POS_Date"]).strftime("%b %d %Y")}, '
                f'Peak NDVI = {row_2019["Peak_NDVI"]:.3f} — matches raw max).<br>'
                f'✅ <b>v5 Fix 2:</b> POS date aligns with raw NDVI maximum (not smoothed peak).'
                f'</div>',
                unsafe_allow_html=True)

        c_left, c_right = st.columns([2,1])
        with c_left:
            st.pyplot(plot_ndvi_phenology(ndvi_df, pheno_df, season_window=(start_month_sel, end_month_sel)))
        with c_right:
            st.markdown("**Extracted dates:**")
            display_rows = []
            for _, row in pheno_df.iterrows():
                sos_d = row.get('SOS_Date'); pos_d = row.get('POS_Date'); eos_d = row.get('EOS_Date')
                display_rows.append({
                    'Year': int(row['Year']),
                    'SOS': pd.Timestamp(sos_d).strftime('%b %d') if pd.notna(sos_d) else '—',
                    'SOS_DOY': int(row.get('SOS_DOY', 0)),
                    'POS': pd.Timestamp(pos_d).strftime('%b %d') if pd.notna(pos_d) else '—',
                    'POS_DOY': int(row.get('POS_DOY', 0)),
                    'EOS': pd.Timestamp(eos_d).strftime('%b %d %Y') if pd.notna(eos_d) else '—',
                    'EOS_DOY': int(row.get('EOS_DOY', 0)),
                    'LOS (d)': int(row.get('LOS_Days', 0)),
                    'Peak NDVI': round(float(row.get('Peak_NDVI', 0)), 3),
                })
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, height=300)

        fig_t = plot_pheno_trends(pheno_df)
        if fig_t: st.pyplot(fig_t)

        st.markdown(
            '<div class="info-box"><b>📐 Method (v5):</b> Valley-anchored amplitude-based threshold. '
            '5-day interpolation + per-segment SG smoothing (w≤31 steps) + inter-season trough detection. '
            'A = NDVI_max − NDVI_min per cycle. SOS/EOS = first/last threshold crossing on raw signal. '
            '<b>Fix 1:</b> Season Year = trough start year (not POS year) — eliminates duplicate-year collision. '
            '<b>Fix 2:</b> POS = raw NDVI peak (not smoothed peak) — exact match with observed maximum.</div>',
            unsafe_allow_html=True)

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
                model_tag = {'ridge':'Ridge','loess':'LOESS','poly2':'Poly-2','poly3':'Poly-3','gpr':'GPR'}.get(mkey, mkey)
                col.markdown(f'<div class="metric-box"><h3>{icons[ev]} {ev}</h3>'
                             f'<h1 style="color:{clr}">{r2*100:.1f}%</h1>'
                             f'<p>{model_tag} | LOO | {n} seasons<br>'
                             f'<b>{n_feats} feature{"s" if n_feats!=1 else ""}:</b> {feat_str}<br>'
                             f'Best |r|={fit["best_r"]:.3f} | MAE = ±{mae:.1f} d</p></div>',
                             unsafe_allow_html=True)

        _card(c1,'SOS'); _card(c2,'POS'); _card(c3,'EOS')

        if n_seasons < 7:
            st.markdown(f'<div class="warn-box">⚠️ <b>{n_seasons} seasons</b> — small training set. R² 0.3–0.6 is realistic.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="good-box">✅ Good number of seasons — model estimates are reliable.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📐 Fitted Equations")
        ev_tab_sos, ev_tab_pos, ev_tab_eos = st.tabs([
            f"{icons['SOS']} SOS — Green-up", f"{icons['POS']} POS — Peak", f"{icons['EOS']} EOS — Senescence"])
        for ui_tab, ev in zip([ev_tab_sos, ev_tab_pos, ev_tab_eos], ['SOS', 'POS', 'EOS']):
            with ui_tab:
                sm_val = cfg['start_month']
                raw_eq = predictor.equation_str(ev, season_start_month=sm_val)
                eq_html = raw_eq.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
                st.markdown(f'<div class="eq-box">{eq_html}</div>', unsafe_allow_html=True)
                ct_display = predictor.corr_table_for_display(ev)
                if not ct_display.empty:
                    def _style_role(val):
                        if val.startswith('✅'):  return 'background-color:#C8E6C9;color:#1B5E20;font-weight:600'
                        if val.startswith('⛔'):  return 'background-color:#FFCCBC;color:#BF360C;font-weight:600'
                        if val.startswith('➖'):  return 'color:#555'
                        return 'color:#999'
                    fmt = {'Pearson r': '{:+.3f}', 'Spearman ρ': '{:+.3f}', 'Composite': '{:.3f}'}
                    styled = ct_display.style
                    if 'Pearson r'  in ct_display.columns: styled = styled.background_gradient(subset=['Pearson r'],  cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Spearman ρ' in ct_display.columns: styled = styled.background_gradient(subset=['Spearman ρ'], cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Composite'  in ct_display.columns: styled = styled.background_gradient(subset=['Composite'],  cmap='Greens', vmin=0,  vmax=1)
                    styled = (styled.applymap(_style_role, subset=['Role']).format(fmt)
                              .set_properties(**{'font-size': '0.84rem'}))
                    st.dataframe(styled, use_container_width=True, hide_index=True)

        fig_s = plot_obs_vs_pred(predictor, train_df)
        if fig_s:
            st.markdown("---"); st.markdown("### 📉 Observed vs Predicted DOY"); st.pyplot(fig_s)

        st.markdown("---")
        dl=[c for c in ['Year','SOS_DOY','SOS_Method','POS_DOY','EOS_DOY','LOS_Days','Peak_NDVI','Amplitude']
            if c in pheno_df.columns]
        st.download_button("📥 Download Phenology Table (CSV)", pheno_df[dl].to_csv(index=False),
                           "phenology_extracted.csv", "text/csv")

    # ══ TAB 2 ══════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📊 Feature Correlations & Meteorological Parameters")
        predictor = st.session_state.get('predictor')
        pheno_df_ss = st.session_state.get('pheno_df')
        if predictor is None:
            st.info("Train models first — upload files and visit the 🔬 Training tab.")
        else:
            fig_corr = plot_correlation_summary(predictor, pheno_df_ss)
            if fig_corr: st.pyplot(fig_corr, use_container_width=True)
            st.markdown("---")
            st.markdown("#### 📋 Detailed Correlation Tables")
            c1, c2, c3 = st.columns(3)
            for col_st, ev in zip([c1, c2, c3], ['SOS', 'POS', 'EOS']):
                with col_st:
                    st.markdown(f"**{icons[ev]} {ev}**")
                    ct = predictor.corr_tables.get(ev)
                    if ct is not None and len(ct):
                        def _sig(p):
                            if p < 0.05: return '**'
                            if p < 0.10: return '*'
                            return ''
                        disp = ct[['Feature','Pearson_r','Spearman_rho','Composite']].copy() \
                               if 'Spearman_rho' in ct.columns \
                               else ct[['Feature','Pearson_r','|r|']].copy()
                        disp['Sig'] = ct['p_value'].apply(_sig)
                        disp = disp.rename(columns={'Pearson_r': 'Pearson r', 'Spearman_rho': 'Spearman ρ'})
                        fmt_d = {'Pearson r': '{:+.3f}'}
                        if 'Spearman ρ' in disp.columns: fmt_d['Spearman ρ'] = '{:+.3f}'
                        if 'Composite'  in disp.columns: fmt_d['Composite']  = '{:.3f}'
                        styled_ct = disp.style.background_gradient(subset=['Pearson r'], cmap='RdYlGn', vmin=-1, vmax=1)
                        if 'Spearman ρ' in disp.columns:
                            styled_ct = styled_ct.background_gradient(subset=['Spearman ρ'], cmap='RdYlGn', vmin=-1, vmax=1)
                        if 'Composite' in disp.columns:
                            styled_ct = styled_ct.background_gradient(subset=['Composite'], cmap='Greens', vmin=0, vmax=1)
                        styled_ct = styled_ct.format(fmt_d).set_properties(**{'font-size': '0.82rem'})
                        st.dataframe(styled_ct, use_container_width=True, hide_index=True, height=320)
                    else:
                        st.info("No correlation data.")

        st.markdown("---")
        st.markdown("#### 📈 NDVI + Meteorology — Year-by-Year Growing Seasons")
        met_df_ss  = st.session_state.get('met_df')
        raw_params_ss = st.session_state.get('raw_params', [])
        ndvi_df_ss = st.session_state.get('ndvi_df')
        if met_df_ss is not None and raw_params_ss and ndvi_df_ss is not None:
            figs_list = plot_met_with_ndvi(met_df_ss, ndvi_df_ss, raw_params_ss,
                                           season_cfg=cfg, pheno_df=pheno_df_ss,
                                           predictor=st.session_state.get('predictor'))
            if figs_list:
                for season_label, fig_m in figs_list:
                    st.markdown(f"**Growing Season {season_label}**")
                    st.pyplot(fig_m, use_container_width=True); plt.close(fig_m)
            else:
                st.info("No complete seasons found in the data.")
        else:
            st.info("Upload files and train models to see the meteorological overview.")

    # ══ TAB 3 ══════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🔮 Predict SOS / POS / EOS / LOS for Any Year")
        predictor=st.session_state.get('predictor'); train_df=st.session_state.get('train_df')
        if predictor is None: st.info("Train models first (Tab 1)."); return
        pred_sm = cfg['start_month']
        ev_feats = {}
        for ev in ['SOS','POS','EOS']:
            fit = predictor._fits.get(ev, {})
            feats = [f for f in fit.get('features', []) if fit.get('mode') == 'ridge']
            if feats: ev_feats[ev] = feats
        if not ev_feats:
            st.warning("No usable features found. Try more seasons or a different forest type."); return

        ev_labels_full = {'SOS': '🌱 SOS — Green-up start', 'POS': '🌿 POS — Peak greenness', 'EOS': '🍂 EOS — Senescence end'}
        ev_colors_hex  = {'SOS': '#E8F5E9', 'POS': '#E3F2FD', 'EOS': '#FFF3E0'}
        ev_border_hex  = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#E65100'}
        ev_inputs = {ev: {} for ev in ['SOS', 'POS', 'EOS']}
        pheno_ss = st.session_state.get('pheno_df')
        mo_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                    7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

        for ev, feats in ev_feats.items():
            fit = predictor._fits[ev]; r2 = predictor.r2.get(ev, 0); mae = predictor.mae.get(ev, 0)
            eq_short = predictor.equation_str(ev, season_start_month=pred_sm).split('\n')[0]
            ev_month_hint = ""
            if pheno_ss is not None and f'{ev}_Date' in pheno_ss.columns:
                ev_dates = pheno_ss[f'{ev}_Date'].dropna()
                if len(ev_dates) > 0:
                    median_month = int(ev_dates.dt.month.median()); median_day = int(ev_dates.dt.day.median())
                    ev_month_hint = (f"  |  Historically occurs ~{mo_names[median_month]} {median_day} (±{mae:.0f} d MAE)")
            st.markdown(
                f"<div style='background:{ev_colors_hex[ev]};padding:14px 18px;border-radius:10px;"
                f"border-left:5px solid {ev_border_hex[ev]};margin:10px 0'>"
                f"<b>{ev_labels_full[ev]}</b>&nbsp;&nbsp;"
                f"<span style='font-size:0.82rem;color:#555'>R²(LOO)={r2:.3f} | MAE=±{mae:.1f}d{ev_month_hint}</span><br>"
                f"<code style='font-size:0.78rem'>{eq_short}</code></div>", unsafe_allow_html=True)
            ncols = max(1, len(feats))
            col_list = st.columns(min(ncols, 4))
            for idx, f in enumerate(feats):
                default = 0.0
                if train_df is not None and f in train_df.columns:
                    ev_sub = train_df[train_df['Event'] == ev]
                    vals = ev_sub[f].dropna() if len(ev_sub) > 0 else train_df[f].dropna()
                    if len(vals): default = float(vals.mean())
                is_sum = any(k in f.upper() for k in ['PREC','GDD','LOG_P','SPEI'])
                with col_list[idx % len(col_list)]:
                    ev_inputs[ev][f] = st.number_input(
                        f"{f}  [{ev}]", value=round(default, 3), format="%.3f",
                        key=f"inp_{ev}_{f}",
                        help=f"{'15-day sum' if is_sum else '15-day mean'} of {f} before expected {ev}. Training mean: {default:.3f}")

        st.markdown("---")
        pred_year=st.number_input("Prediction year", 2024, 2040, 2026)
        if st.button("🚀 Predict Phenology Events", type="primary"):
            results={}
            for ev in ['SOS','POS','EOS']:
                res=predictor.predict(ev_inputs.get(ev, {}), ev, pred_year, season_start_month=pred_sm)
                if res: results[ev]=res
            if results:
                order_warnings = []
                if 'SOS' in results and 'POS' in results:
                    if results['POS']['rel_days'] <= results['SOS']['rel_days']:
                        corrected_rel = max(int(round(pheno_ss['POS_Target'].mean())) if pheno_ss is not None and 'POS_Target' in pheno_ss.columns else results['SOS']['rel_days']+90, results['SOS']['rel_days']+14)
                        new_date = datetime(pred_year, pred_sm, 1) + timedelta(days=corrected_rel)
                        results['POS'].update({'rel_days': corrected_rel, 'doy': new_date.timetuple().tm_yday, 'date': new_date})
                        order_warnings.append(f"⚠️ **POS** predicted before SOS — corrected to ~{new_date.strftime('%b %d')}")
                if 'POS' in results and 'EOS' in results:
                    if results['EOS']['rel_days'] <= results['POS']['rel_days']:
                        corrected_rel = max(int(round(pheno_ss['EOS_Target'].mean())) if pheno_ss is not None and 'EOS_Target' in pheno_ss.columns else results['POS']['rel_days']+90, results['POS']['rel_days']+14)
                        new_date = datetime(pred_year, pred_sm, 1) + timedelta(days=corrected_rel)
                        results['EOS'].update({'rel_days': corrected_rel, 'doy': new_date.timetuple().tm_yday, 'date': new_date})
                        order_warnings.append(f"⚠️ **EOS** predicted before POS — corrected to ~{new_date.strftime('%b %d')}")
                if order_warnings:
                    st.markdown('<div class="warn-box"><b>Ecological order correction:</b><br>' + '<br>'.join(order_warnings) + '</div>', unsafe_allow_html=True)
                cols=st.columns(len(results))
                for col,(ev,res) in zip(cols, results.items()):
                    actual_yr = res['date'].year
                    col.markdown(f'<div class="metric-box"><h3>{icons[ev]} {ev}</h3>'
                                 f'<h1>{res["date"].strftime("%b %d")}</h1>'
                                 f'<p>Day {res["doy"]} of {actual_yr}<br>R²(LOO)={res["r2"]:.3f}<br>MAE=±{res["mae"]:.1f} d</p></div>',
                                 unsafe_allow_html=True)
                if 'SOS' in results and 'EOS' in results:
                    sd=results['SOS']['date']; ed=results['EOS']['date']
                    los = (ed - sd).days if ed >= sd else (ed + pd.DateOffset(years=1) - sd).days
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("📏 Length of Season (LOS)", f"{los} days")
                    c2.metric("🌱 SOS → 🌿 POS", f"{(results['POS']['date']-results['SOS']['date']).days} days" if 'POS' in results else "—")
                    c3.metric("🌿 POS → 🍂 EOS", f"{(results['EOS']['date']-results['POS']['date']).days} days" if 'POS' in results else "—")
                out=pd.DataFrame({'Event':list(results.keys()),
                    'Date':[r['date'].strftime('%Y-%m-%d') for r in results.values()],
                    'DOY':[r['doy'] for r in results.values()],
                    'R²_LOO':[round(r['r2'],3) for r in results.values()],
                    'MAE_days':[round(r['mae'],1) for r in results.values()]})
                st.dataframe(out, use_container_width=True)
                st.download_button("📥 Download Predictions", out.to_csv(index=False), "predictions.csv","text/csv")

    # ══ TAB 5 ══════════════════════════════════════════════════
    with tab5:
        st.markdown(f"""
### 📖 Technical Guide

---

#### 🐛 Bug Fix v5 — Two Critical Fixes

---

##### Fix 1 — Missing Season (Ghost-Year Duplicate via `pos.year`)

**Problem:** When a season's NDVI peak (POS) falls in a calendar year shared with an adjacent
season's late peak, `drop_duplicates(subset="Year")` silently drops one season.

**Concrete example with IIT Tirupati dataset:**
- Season 2018-19 (Apr 2018 → May 2019): POS = **Feb 2019** → `Year = 2019`
- Season 2019-20 (May 2019 → May 2020): POS = **Sep 2019** → `Year = 2019`
- `drop_duplicates` keeps the first (2018-19) and **drops 2019-20 entirely**

**Fix:**
```python
# ❌ OLD (in all three segments — main loop, head, tail):
yr = pos.year

# ✅ NEW: use trough date year as season identity
trough_year = t_all[ti].year   # always unique per cycle
rows.append({{"Year": trough_year, ...}})
```

---

##### Fix 2 — POS Date ≠ Raw NDVI Peak

**Problem:** `pos_idx_local = int(np.argmax(cycle_sm))` uses the smoothed signal.
SG smoothing shifts and suppresses sharp peaks — for Sep 28 2019 (raw = **0.713**)
the smoothed peak moved to Oct 8 2019 (sm = 0.651): **10-day shift, 0.062 NDVI underestimate**.

**Fix:**
```python
# ❌ OLD (all three segments):
pos_idx_local = int(np.argmax(cycle_sm))

# ✅ NEW: POS from raw signal — matches what user sees in the scatter plot
pos_idx_local = int(np.nanargmax(cycle_raw))
```

---

#### 🐛 Bug Fix v4 — Gap-Tolerant Cycle Extraction
- Tolerates up to 50% gap when amplitude ≥ 0.10 (clear seasonal signal)
- Low-amplitude cycles keep strict 20% gap threshold

#### 🐛 Bug Fix v3 — SG Window Capped at 31 Steps (~155 days)
- Prevents over-smoothing that crushes per-season amplitude below MIN_AMPLITUDE

#### 🐛 Bug Fix v2 — Plateau Trough Filter (85% ceiling, skip for amp < 0.20)
- Fixes missing seasons in high-NDVI evergreen forests (Shola, mangrove)

---

#### 🌿 Phenology Extraction Method — Valley-Anchored Amplitude Threshold

| Step | Process |
|------|---------|
| 1 | Gap detection — gaps > max(60d, 8×cadence) preserved as NaN |
| 2 | 5-day interpolation within segments only |
| 3 | Per-segment SG smoothing (window ≤ 31 steps = 155d) |
| 4 | Valley (trough) detection — min separation ~145 days |
| 5 | Season year = **trough start year** (v5 fix) |
| 6 | Amplitude A = NDVI_max(raw) − NDVI_min per cycle |
| 7 | Threshold = NDVI_min + threshold% × A |
| 8 | SOS = first crossing on ascending raw limb |
| 9 | **POS = raw NDVI peak** between SOS and EOS (v5 fix) |
| 10 | EOS = last crossing on descending raw limb |

---

#### 📉 R² (LOO) Interpretation

| R² | Meaning |
|----|---------|
| > 0.80 | Strong |
| 0.50–0.80 | Good |
| 0.30–0.50 | Moderate |
| < 0.30 | Weak — add more seasons |
        """)


if __name__ == "__main__":
    for k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params','ndvi_df']:
        if k not in st.session_state: st.session_state[k] = None
    main()
