"""
Forest Phenology Analyser
=========================
Upload NDVI and meteorological data to extract seasonal phenology events (SOS, POS, EOS, LOS),
identify climate drivers, train predictive models, and forecast future phenology dates.

Supports any Indian forest type — fully data-driven, no manual configuration required.

Requirements:
    pip install streamlit pandas numpy scipy scikit-learn matplotlib

Run:
    streamlit run universal_Indian_forest_phenology_v5.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
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


# ─── LOESS (no external dependency) ─────────────────────────
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


# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Forest Phenology Analyser",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
/* ── Typography & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App Header ── */
.app-header {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
    padding: 32px 40px 24px; border-radius: 16px; margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(27,94,32,0.25);
}
.app-header h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.5px; }
.app-header p  { color: #C8E6C9; font-size: 0.92rem; margin: 0; line-height: 1.6; }
.app-header .badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    color: #fff; font-size: 0.78rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; margin: 4px 4px 0 0;
}

/* ── Cards ── */
.metric-card {
    background: #fff; padding: 20px 16px; border-radius: 14px;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #E8F5E9; margin: 4px;
}
.metric-card .label { color: #616161; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.metric-card .value { color: #1B5E20; font-size: 1.85rem; font-weight: 700; margin: 0; }
.metric-card .sub   { color: #757575; font-size: 0.76rem; margin-top: 4px; }

/* ── Info Banners ── */
.banner-info  { background: #E3F2FD; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #1976D2; margin: 10px 0; font-size: 0.88rem; }
.banner-warn  { background: #FFF8E1; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #F9A825; margin: 10px 0; font-size: 0.88rem; }
.banner-good  { background: #E8F5E9; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #43A047; margin: 10px 0; font-size: 0.88rem; }
.banner-error { background: #FFEBEE; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #E53935; margin: 10px 0; font-size: 0.88rem; }

/* ── Equation Box ── */
.eq-box { background: #F8F9FA; padding: 14px 16px; border-radius: 10px;
    border-left: 4px solid #7B1FA2; font-family: 'Courier New', monospace;
    font-size: 0.83rem; margin: 8px 0; word-break: break-all; color: #212121; }

/* ── Upload Panel ── */
.upload-panel { background: #F9FBF9; padding: 24px 28px; border-radius: 14px;
    border: 2px dashed #A5D6A7; margin: 20px 0; }
.upload-panel h3 { color: #1B5E20; margin-bottom: 12px; font-size: 1.05rem; }
.upload-panel code { background: #E8F5E9; padding: 2px 6px; border-radius: 4px;
    font-size: 0.82rem; }

/* ── Section Heading ── */
.section-title { font-size: 1.15rem; font-weight: 700; color: #1B5E20;
    margin: 24px 0 12px; padding-bottom: 6px; border-bottom: 2px solid #C8E6C9; }

/* ── Guide Definition Term ── */
.term { background: #E8F5E9; padding: 2px 8px; border-radius: 5px;
    font-weight: 600; color: #1B5E20; font-size: 0.88rem; }

/* ── Stat table ── */
.stat-card { background: #FAFFFE; padding: 12px 16px; border-radius: 10px;
    border: 1px solid #C8E6C9; margin: 4px 0; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS (truly universal, not forest-type specific) ───
MIN_CORR_THRESHOLD = 0.40   # minimum |r| for a feature to be used
ALPHAS = [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000]

# Keywords that identify ACCUMULATION features (use sum over window, not mean)
# Covers: standard NASA POWER names, PPT, custom rain/precip names, GDD variants
ACCUM_KEYWORDS = [
    'PREC', 'RAIN', 'PPT', 'PRECIP', 'RAINFALL',         # precipitation variants
    'GDD_5', 'GDD_10', 'GDD',                             # growing degree days
    'LOG_P', 'LOG_PREC',                                   # log precipitation
    'SPEI', 'SPEI_PROXY',                                  # water balance
    'PET', 'ET',                                           # evapotranspiration
    'CDD', 'HDD',                                          # heating/cooling degree days
]

# Keywords that identify SNAPSHOT features (use last value before event, not mean/sum)
# These are cumulative-from-season-start metrics — taking a window mean is wrong
SNAPSHOT_KEYWORDS = ['GDD_CUM', 'CPPT', 'CT2M', 'CUMUL', 'ACCUM', 'CUM_']
SNAPSHOT_FEATURES = {'GDD_cum', 'GDD_CUM', 'CPPT', 'CT2M'}  # exact names


# ═══════════════════════════════════════════════════════════════
# DATA-ADAPTIVE UTILITIES
# ═══════════════════════════════════════════════════════════════

def detect_ndvi_cadence(ndvi_df):
    """
    Detect typical observation cadence from the data itself.
    Returns (median_cadence_days, max_gap_days, interp_freq_days).
    No hardcoded assumptions.
    """
    dates = pd.to_datetime(ndvi_df['Date']).sort_values()
    diffs = dates.diff().dt.days.dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 16, 64, 5
    median_cad = float(diffs.median())
    max_gap    = max(60, int(median_cad * 8))   # 8× typical cadence = gap
    interp_freq = max(1, min(8, int(median_cad // 2)))  # half cadence, 1-8d
    return median_cad, max_gap, interp_freq


def detect_seasonality(ndvi_series_5d):
    """
    Estimate dominant cycle length from autocorrelation — data-driven.
    Returns estimated cycle length in 5-day steps.
    """
    vals = ndvi_series_5d.dropna().values
    if len(vals) < 24:
        return 73  # ~365 days at 5d cadence fallback
    n = len(vals)
    v = vals - vals.mean()
    max_lag = min(n // 2, 110)   # up to ~550 days
    acf = []
    for lag in range(10, max_lag):
        if lag >= len(v): break
        r = np.corrcoef(v[:n-lag], v[lag:])[0, 1]
        acf.append((lag, r))
    if not acf:
        return 73
    acf_arr = np.array(acf)
    # Find first significant positive peak after lag>20 (to skip noise)
    for i in range(1, len(acf_arr) - 1):
        if (acf_arr[i, 1] > acf_arr[i-1, 1] and
                acf_arr[i, 1] > acf_arr[i+1, 1] and
                acf_arr[i, 1] > 0.3 and
                acf_arr[i, 0] > 20):
            return int(acf_arr[i, 0])
    return 73


def compute_data_driven_min_amplitude(ndvi_vals_clean):
    """
    Set minimum detectable amplitude from the actual NDVI variance.
    Uses 5th–95th percentile range: if data has little variation, threshold is low.
    Never hardcodes a fixed value like 0.05.
    """
    p5  = float(np.percentile(ndvi_vals_clean, 5))
    p95 = float(np.percentile(ndvi_vals_clean, 95))
    data_range = p95 - p5
    # Minimum amplitude = 5% of the data's own dynamic range (but ≥0.01)
    return max(0.01, data_range * 0.05)


def characterize_ndvi_data(ndvi_df):
    """
    Returns a dict of data-derived NDVI characteristics shown in the UI.
    Zero hardcoding — purely from the uploaded NDVI series.
    """
    vals = ndvi_df['NDVI'].dropna().values
    years = pd.to_datetime(ndvi_df['Date']).dt.year.unique()
    cadence, max_gap, _ = detect_ndvi_cadence(ndvi_df)
    return {
        'n_obs':       len(vals),
        'n_years':     len(years),
        'year_range':  f"{years.min()}–{years.max()}",
        'ndvi_mean':   round(float(np.mean(vals)), 3),
        'ndvi_std':    round(float(np.std(vals)),  3),
        'ndvi_min':    round(float(np.min(vals)),  3),
        'ndvi_max':    round(float(np.max(vals)),  3),
        'ndvi_p5':     round(float(np.percentile(vals, 5)),  3),
        'ndvi_p95':    round(float(np.percentile(vals, 95)), 3),
        'data_range':  round(float(np.percentile(vals, 95) - np.percentile(vals, 5)), 3),
        'cadence_d':   round(cadence, 1),
        'max_gap_d':   max_gap,
        'evergreen_index': round(float(np.percentile(vals, 5) / (np.percentile(vals, 95) + 1e-6)), 3),
    }


def characterize_met_data(met_df, raw_params):
    """
    Summarize available met parameters from the data — no preset lists.
    """
    info = {}
    for p in raw_params:
        if p not in met_df.columns: continue
        col = met_df[p].dropna()
        if len(col) == 0: continue
        info[p] = {
            'mean': round(float(col.mean()), 3),
            'std':  round(float(col.std()),  3),
            'min':  round(float(col.min()),  3),
            'max':  round(float(col.max()),  3),
        }
    return info


# ═══════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════

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
                if up.startswith('YEAR') or up.startswith('LON') or up.startswith('DATE'): skip_to = i; break
        df = pd.read_csv(StringIO('\n'.join(lines[skip_to:])))
        df.columns = [c.strip() for c in df.columns]
        df.replace([-999, -999.0, -99, -99.0, -9999, -9999.0], np.nan, inplace=True)
        if 'Date' not in df.columns:
            # Try YEAR+DOY first
            if {'YEAR', 'DOY'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(
                    df['YEAR'].astype(str) + df['DOY'].astype(str).str.zfill(3),
                    format='%Y%j', errors='coerce')
            # Try YEAR+MO+DY
            elif {'YEAR', 'MO', 'DY'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(
                    df['YEAR'].astype(str) + '-' + df['MO'].astype(str).str.zfill(2) + '-' +
                    df['DY'].astype(str).str.zfill(2), errors='coerce')
            # Try common date column names
            else:
                date_col = next((c for c in df.columns
                                 if c.lower() in ['date', 'datetime', 'time', 'dates']), None)
                if date_col:
                    df['Date'] = _parse_date_robust(df[date_col].astype(str))
                    if date_col != 'Date':
                        df = df.drop(columns=[date_col])
                else:
                    return None, [], 'Cannot build Date — need YEAR+DOY, YEAR+MO+DY, or a Date column'
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        excl = {'YEAR', 'MO', 'DY', 'DOY', 'LON', 'LAT', 'ELEV', 'Date'}
        params = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
        return df, params, None
    except Exception as e:
        return None, [], str(e)


def _parse_date_robust(series, doy_series=None):
    if len(series) == 0:
        return pd.to_datetime(series, errors='coerce')

    # Try explicit formats first — catches dd-mm-yy, dd-mm-yyyy etc.
    explicit_formats = [
        '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d',
        '%d-%m-%y', '%d/%m/%y', '%m/%d/%Y', '%m-%d-%Y',
        '%d-%b-%Y', '%d %b %Y', '%d-%b-%y',
    ]
    best_explicit, best_explicit_n = None, 0
    for fmt in explicit_formats:
        try:
            attempt = pd.to_datetime(series, format=fmt, errors='coerce')
            n_ok = attempt.notna().sum()
            if n_ok > best_explicit_n:
                best_explicit, best_explicit_n = attempt, n_ok
        except Exception:
            continue

    parsed_default  = pd.to_datetime(series, errors='coerce')
    parsed_dayfirst = pd.to_datetime(series, dayfirst=True, errors='coerce')

    if doy_series is not None:
        doy_ref      = pd.to_numeric(doy_series, errors='coerce')
        default_doy  = parsed_default.dt.dayofyear.where(parsed_default.notna(), np.nan)
        dayfirst_doy = parsed_dayfirst.dt.dayofyear.where(parsed_dayfirst.notna(), np.nan)
        explicit_doy = best_explicit.dt.dayofyear.where(best_explicit.notna(), np.nan) if best_explicit is not None else pd.Series([np.nan]*len(series))
        scores = {
            'default':  (default_doy == doy_ref).sum(),
            'dayfirst': (dayfirst_doy == doy_ref).sum(),
            'explicit': (explicit_doy == doy_ref).sum(),
        }
        best_key = max(scores, key=scores.get)
        return {'default': parsed_default, 'dayfirst': parsed_dayfirst, 'explicit': best_explicit or parsed_default}[best_key]

    # Choose between candidates: prefer the one with most realistic year range
    # (close to current decade) and most unique months
    candidates = [parsed_default, parsed_dayfirst]
    if best_explicit is not None and best_explicit_n >= len(series) * 0.85:
        candidates.append(best_explicit)

    def _score(s):
        if s is None or s.notna().sum() == 0:
            return -999
        yr_median = s.dropna().dt.year.median()
        yr_plausible = 1 if 1980 <= yr_median <= 2040 else 0
        n_months = s.dropna().dt.month.nunique()
        n_valid  = s.notna().sum()
        return yr_plausible * 1000 + n_months * 10 + n_valid

    best = max(candidates, key=_score)
    return best


def parse_ndvi(uploaded_file):
    try:
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        df = pd.read_csv(StringIO(raw_bytes.decode('utf-8', errors='replace')))
        df.columns = [c.strip() for c in df.columns]
        date_col = next((c for c in df.columns
                         if c.lower() in ['date', 'dates', 'time', 'datetime']), None)
        if not date_col:
            return None, "No Date column found. Expected: date / dates / time / datetime"
        ndvi_col = next((c for c in df.columns
                         if c.lower() in ['ndvi', 'ndvi_value', 'value', 'index', 'evi']), None)
        if not ndvi_col:
            return None, "No NDVI column found. Expected: ndvi / ndvi_value / value / evi"
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
            return None, "No valid rows after parsing. Check date format."
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


# ═══════════════════════════════════════════════════════════════
# DERIVED MET FEATURES (data-agnostic computation)
# ═══════════════════════════════════════════════════════════════

def _season_cumsum(series, dates, sm_):
    """
    Vectorized seasonal cumulative sum — works with any pandas/datetime64 dtype.
    BUG FIX: replaces .apply(lambda d: d.year ...) which fails on pandas 2.x
    with datetime64[us] dtype (AttributeError: 'NaTType' has no attribute 'year').
    """
    out = series.copy() * 0.0
    dt = pd.to_datetime(dates)
    season_yr = np.where(dt.dt.month >= sm_, dt.dt.year, dt.dt.year - 1)
    season_yr_s = pd.Series(season_yr, index=series.index)
    for sy, grp_idx in series.groupby(season_yr_s).groups.items():
        out.loc[grp_idx] = series.loc[grp_idx].cumsum().values
    return out


def _detect_column(cols, *keyword_groups):
    """
    Universal column detector — finds the best matching column from a list
    using keyword matching (case-insensitive substring).

    keyword_groups: ordered list of keyword tuples from most-specific to least.
    Returns the first column that matches any keyword in the first matching group.

    Example:
        _detect_column(cols, ['T2M_MIN','TMIN','TEMP_MIN'], ['MIN_T','TMIN'])
    """
    cols_upper = {c: c.upper() for c in cols}
    for keywords in keyword_groups:
        for c in cols:
            cu = cols_upper[c]
            for kw in keywords:
                if kw.upper() == cu or kw.upper() in cu:
                    return c
    return None


def add_derived_features(met_df, season_start_month=1):
    """
    UNIVERSAL feature derivation — works with ANY meteorological column names.

    Strategy:
      1. Use fuzzy keyword matching (not hardcoded exact names) to find
         temperature, humidity, precipitation, soil moisture, radiation columns.
      2. Derive GDD, DTR, VPD, log_precip, SPEI_proxy, MSI from whatever is found.
      3. Never overwrite columns already present in the uploaded file.
      4. All raw columns (regardless of name) are passed through untouched as features.

    This means the app works with:
      - NASA POWER standard names (T2M, RH2M, PRECTOTCORR, GWETTOP ...)
      - Custom exports (PPT, RH, RAD, CT2M, CPPT ...)
      - Any other naming convention
    """
    df = met_df.copy()
    cols = df.columns.tolist()

    # ── Temperature columns (fuzzy keyword match) ──────────────────────────────
    # Order: exact standard names first, then common aliases, then generic keywords
    tmin = _detect_column(cols,
        ['T2M_MIN', 'TMIN', 'TEMP_MIN'],
        ['MIN_T', 'TMIN', 'MINTEMP', 'T_MIN', 'TEMPMIN'],
        ['MIN'])   # last resort — only if column name contains 'MIN'

    tmax = _detect_column(cols,
        ['T2M_MAX', 'TMAX', 'TEMP_MAX'],
        ['MAX_T', 'TMAX', 'MAXTEMP', 'T_MAX', 'TEMPMAX'],
        ['MAX'])

    tmn = _detect_column(cols,
        ['T2M', 'TMEAN', 'TAVG', 'TEMP_MEAN', 'T_MEAN'],
        ['TEMP', 'TEMPERATURE', 'AIR_T', 'TAIR'],
        ['T2M'])

    # Resolve ambiguity: if tmin/tmax found from same column as tmn, clear tmn
    if tmn and (tmn == tmin or tmn == tmax):
        tmn = None

    # ── Humidity ───────────────────────────────────────────────────────────────
    rh = _detect_column(cols,
        ['RH2M', 'RH', 'RHUM', 'REL_HUM', 'RELATIVE_HUMIDITY'],
        ['HUMID', 'RH_', 'HR'],
        ['RH'])

    # ── Precipitation ──────────────────────────────────────────────────────────
    prec = _detect_column(cols,
        ['PRECTOTCORR', 'PRECTOT', 'PRECIP', 'PRECIPITATION'],
        ['RAIN', 'PPT', 'RAINFALL', 'PREC', 'PR_'],
        ['PPT', 'RAIN', 'PREC'])

    # Avoid using cumulative precipitation column as raw precip
    # (cumulative columns inflate GDD-like derivations)
    if prec:
        cu = prec.upper()
        if any(k in cu for k in ['CUM', 'CPPT', 'CUMUL', 'ACCUM', 'TOTAL_P']):
            # Try to find a non-cumulative alternative
            alt = _detect_column([c for c in cols if c != prec],
                ['PPT', 'RAIN', 'PRECIP', 'PREC'],
                ['PPT', 'RAIN'])
            prec = alt  # may be None — that's OK

    # ── Soil moisture ──────────────────────────────────────────────────────────
    sm = _detect_column(cols,
        ['GWETTOP', 'GWETROOT', 'GWETPROF', 'SOIL_MOISTURE', 'SM_TOP'],
        ['SOIL_W', 'SOILW', 'SM_', 'VSM', 'SWC'],
        ['GWET', 'SOIL'])

    # ── Solar radiation ────────────────────────────────────────────────────────
    rad = _detect_column(cols,
        ['ALLSKY_SFC_SW_DWN', 'SRAD', 'RAD', 'SOLAR', 'INSOL', 'RADIATION'],
        ['SW_DWN', 'SHORTWAVE', 'SOLRAD', 'RS', 'RADSOL'],
        ['RAD', 'SOL'])

    # ── Compute tavg (average temperature) ────────────────────────────────────
    tavg = None
    if tmin and tmax and tmin != tmax:
        tavg = (df[tmax] + df[tmin]) / 2.0
        if 'DTR' not in cols:
            df['DTR'] = df[tmax] - df[tmin]
    elif tmn:
        tavg = df[tmn]
    elif tmin and not tmax:
        tavg = df[tmin]  # fallback — only min temp available
    elif tmax and not tmin:
        tavg = df[tmax]  # fallback — only max temp available

    # ── Growing Degree Days ────────────────────────────────────────────────────
    if tavg is not None:
        if 'GDD_10' not in cols:
            df['GDD_10'] = np.maximum(tavg - 10, 0)
        if 'GDD_5' not in cols:
            df['GDD_5']  = np.maximum(tavg - 5,  0)
        if 'GDD_cum' not in cols:
            df['GDD_cum'] = _season_cumsum(
                np.maximum(tavg - 10, 0).rename('GDD_10_tmp'),
                df['Date'], season_start_month)

    # ── Log precipitation ──────────────────────────────────────────────────────
    if prec and 'log_precip' not in cols:
        df['log_precip'] = np.log1p(np.maximum(df[prec].fillna(0), 0))

    # ── Vapour Pressure Deficit ────────────────────────────────────────────────
    if tavg is not None and rh and 'VPD' not in cols:
        es = 0.6108 * np.exp((17.27 * tavg) / (tavg + 237.3))
        df['VPD'] = np.maximum(es * (1 - df[rh] / 100.0), 0)

    # ── Moisture Stress Index ──────────────────────────────────────────────────
    if prec and sm and 'MSI' not in cols:
        df['MSI'] = df[prec] / (df[sm].replace(0, np.nan) + 1e-6)

    # ── SPEI proxy (water balance) ─────────────────────────────────────────────
    if prec and tavg is not None and 'SPEI_proxy' not in cols:
        pet = 0.0023 * (tavg + 17.8) * np.maximum(tavg, 0) ** 0.5
        df['SPEI_proxy'] = df[prec].fillna(0) - pet.fillna(0)

    # ── Standardise solar radiation column name ────────────────────────────────
    # If file has RAD/SRAD but not the standard name, add alias so downstream
    # correlation code always has a consistent handle to reference
    if rad and rad not in ('ALLSKY_SFC_SW_DWN',) and 'ALLSKY_SFC_SW_DWN' not in cols:
        df['ALLSKY_SFC_SW_DWN'] = df[rad]

    return df


# ═══════════════════════════════════════════════════════════════
# TRAINING FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════

def make_training_features(pheno_df, met_df, params, window=15):
    """
    Build per-event training features using a rolling window before each event date.
    Window size (days) is user-configurable and not hardcoded.
    Accumulation vs mean is determined by the parameter name keywords — data-aware.
    """
    records = []
    for _, row in pheno_df.iterrows():
        for event in ['SOS', 'POS', 'EOS']:
            evt_dt = row[f'{event}_Date']
            if pd.isna(evt_dt):
                continue
            rec = {
                'Year':         row['Year'],
                'Event':        event,
                'Target_DOY':   row.get(f'{event}_Target', row[f'{event}_DOY']),
                'Season_Start': row.get('Season_Start', pd.NaT),
                'LOS_Days':     row.get('LOS_Days', np.nan),
                'Peak_NDVI':    row.get('Peak_NDVI', np.nan),
            }
            mask = ((met_df['Date'] >= evt_dt - timedelta(days=window)) &
                    (met_df['Date'] <= evt_dt))
            wdf = met_df[mask]
            if len(wdf) < max(1, window * 0.15):
                continue
            for p in params:
                if p not in met_df.columns:
                    continue
                p_upper = p.upper()
                # Snapshot: cumulative-from-season-start — take last value before event
                is_snapshot = (p_upper in SNAPSHOT_FEATURES or
                               any(k in p_upper for k in SNAPSHOT_KEYWORDS))
                # Accumulation: sum over window (precipitation, GDD, etc.)
                is_accum = (not is_snapshot and
                            any(k in p_upper for k in ACCUM_KEYWORDS))
                if is_snapshot:
                    snap = met_df[met_df['Date'] <= evt_dt][p].dropna()
                    rec[p] = float(snap.iloc[-1]) if len(snap) > 0 else np.nan
                elif is_accum:
                    rec[p] = float(wdf[p].sum())
                else:
                    rec[p] = float(wdf[p].mean())
            records.append(rec)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# PHENOLOGY EXTRACTION — FULLY DATA-DRIVEN
# ═══════════════════════════════════════════════════════════════

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


def _find_troughs_boundary(v, min_distance):
    """
    FIX v6 — Boundary-aware trough finder.

    The standard _find_troughs() misses real troughs that fall within min_dist
    steps of the data start or end (e.g. Apr 2003 trough when data starts Feb 2003).
    This function:
      1. Runs the standard search (internal troughs).
      2. Checks the first 1.5×min_dist steps for a boundary trough at the START.
      3. Checks the last  1.5×min_dist steps for a boundary trough at the END.
    A boundary trough is accepted only if it is genuinely lower than its surrounding
    values and sufficiently far from the nearest interior trough.
    """
    n = len(v)
    # Step 1: standard interior troughs
    troughs = []
    for i in range(1, n - 1):
        w = v[max(0, i - min_distance): i + min_distance + 1]
        if v[i] == np.min(w) and v[i] <= v[i - 1] and v[i] <= v[i + 1]:
            troughs.append(i)
    if troughs:
        merged = [troughs[0]]
        for t in troughs[1:]:
            if t - merged[-1] < min_distance:
                if v[t] < v[merged[-1]]: merged[-1] = t
            else:
                merged.append(t)
    else:
        merged = []

    # Step 2: boundary trough at START
    search_end = min(int(min_distance * 1.5), n // 3)
    if search_end > 1:
        start_win = v[0:search_end]
        bmi = int(np.argmin(start_win))
        next_t = merged[0] if merged else n
        # Accept: not at very first step, genuinely low, far enough from next interior trough
        if bmi > 0 and next_t - bmi >= min_distance:
            # Must be lower than the midpoint between start and next trough
            midpoint_val = float(np.mean(v[bmi:next_t])) if next_t < n else float(np.mean(v[bmi:]))
            if v[bmi] < midpoint_val:
                merged.insert(0, bmi)

    # Step 3: boundary trough at END
    search_start = max(n - int(min_distance * 1.5), 2 * n // 3)
    if search_start < n - 1:
        end_win = v[search_start:]
        bmi = search_start + int(np.argmin(end_win))
        prev_t = merged[-1] if merged else -1
        if bmi - prev_t >= min_distance and bmi < n - 1:
            midpoint_val = float(np.mean(v[prev_t:bmi])) if prev_t >= 0 else float(np.mean(v[0:bmi]))
            if v[bmi] < midpoint_val:
                merged.append(bmi)

    return merged


def extract_phenology(ndvi_df, cfg, sos_threshold_pct, eos_threshold_pct):
    """
    Fully data-driven phenology extraction.

    All parameters are either user-set or derived from the NDVI data itself:
      - Cadence → interpolation frequency (from data)
      - Max gap → from data cadence (not hardcoded 60d)
      - SG window → from segment length, capped at 31 steps (FIX v3)
      - Trough min distance → from autocorrelation estimate of cycle length
      - MIN_AMPLITUDE → from data's own dynamic range (FIX v5 data-driven)
      - Gap tolerance → scaled by detected amplitude (FIX v4)
      - Trough ceiling → 85% of amplitude (FIX v2)
      - Season year → trough start year (FIX 1)
      - POS → raw NDVI peak (FIX 2)
    """
    try:
        sm    = cfg["start_month"]
        em    = cfg["end_month"]
        min_d = cfg.get("min_days", 100)

        thr_pct     = sos_threshold_pct
        eos_thr_pct = eos_threshold_pct

        # ── Step 1: Build time-indexed raw NDVI ──────────────
        ndvi_raw = ndvi_df[["Date", "NDVI"]].copy().set_index("Date").sort_index()
        ndvi_raw = ndvi_raw[~ndvi_raw.index.duplicated(keep='first')]

        # ── Step 2: Data-derived cadence & gap parameters ────
        orig_dates  = ndvi_raw.index.sort_values()
        orig_diffs  = pd.Series(orig_dates).diff().dt.days.fillna(0)
        pos_diffs   = orig_diffs[orig_diffs > 0]
        typical_cad = float(pos_diffs.median()) if len(pos_diffs) > 0 else 16.0
        MAX_INTERP_GAP = max(60, int(typical_cad * 8))   # data-driven
        gap_starts  = orig_dates[orig_diffs.values > MAX_INTERP_GAP]

        # ── Step 3: 5-day grid interpolation ─────────────────
        interp_freq = max(1, min(8, round(typical_cad)))   # FIX v6: was int(cad//2) → gave 2d grid for 5d data
        full_range  = pd.date_range(
            start=ndvi_raw.index.min(),
            end=ndvi_raw.index.max(),
            freq=f"{interp_freq}D")

        ndvi_5d = ndvi_raw.reindex(ndvi_raw.index.union(full_range))
        ndvi_5d = ndvi_5d.interpolate(method="time", limit_area="inside")

        for gap_start in gap_starts:
            before = orig_dates[orig_dates < gap_start]
            if len(before) == 0:
                continue
            mask = (ndvi_5d.index > before[-1]) & (ndvi_5d.index < gap_start)
            ndvi_5d.loc[mask] = np.nan

        ndvi_5d = ndvi_5d.reindex(full_range)
        ndvi_5d.columns = ["NDVI"]

        n         = len(ndvi_5d)
        ndvi_vals = ndvi_5d["NDVI"].values.copy()
        valid_mask = ~np.isnan(ndvi_vals)

        # ── Step 4: Data-driven MIN_AMPLITUDE ────────────────
        valid_vals  = ndvi_vals[valid_mask]
        MIN_AMPLITUDE = compute_data_driven_min_amplitude(valid_vals) if len(valid_vals) > 5 else 0.02

        # ── Step 5: Per-segment SG smoothing (window ≤ 31) ───
        MAX_SG_STEPS = 31
        sm_vals  = np.full(n, np.nan)
        seg_labels = np.zeros(n, dtype=int)
        seg_id, in_seg = 0, False
        for i in range(n):
            if valid_mask[i]:
                if not in_seg:
                    seg_id += 1
                    in_seg = True
                seg_labels[i] = seg_id
            else:
                in_seg = False

        for sid in range(1, seg_id + 1):
            idx_seg = np.where(seg_labels == sid)[0]
            seg_n   = len(idx_seg)
            if seg_n < 5:
                sm_vals[idx_seg] = ndvi_vals[idx_seg]
                continue
            wl_t = max(7, min(int(seg_n * 0.05), MAX_SG_STEPS))
            wl_s = wl_t if wl_t % 2 == 1 else wl_t + 1
            wl_s = min(wl_s, seg_n - 1 if seg_n > 1 else 1)
            if wl_s % 2 == 0:
                wl_s = max(7, wl_s - 1)
            poly_s = min(2, wl_s - 1)
            if wl_s >= 5 and wl_s < seg_n:
                sm_vals[idx_seg] = savgol_filter(ndvi_vals[idx_seg], wl_s, poly_s)
            else:
                sm_vals[idx_seg] = ndvi_vals[idx_seg]

        t_all = ndvi_5d.index
        sm_for_troughs = pd.Series(sm_vals, index=t_all).interpolate(
            method="linear", limit_direction="both").values

        # ── Step 6: Data-driven trough min-distance ───────────
        # Estimate from autocorrelation of smoothed series; fallback = 365d/2 / freq
        try:
            cycle_steps = detect_seasonality(pd.Series(sm_for_troughs, index=t_all))
        except Exception:
            cycle_steps = int(365 / interp_freq)
        min_dist = max(10, int(cycle_steps * 0.4))   # 40% of cycle length

        trough_raw = _find_troughs_boundary(sm_for_troughs, min_dist)

        # Gap filter: discard troughs at/adjacent to NaN
        trough_indices = []
        for ti in trough_raw:
            window_sl = slice(max(0, ti - 5), min(n, ti + 6))
            if np.isnan(sm_vals[window_sl]).any():
                continue
            trough_indices.append(ti)

        # Plateau trough filter (v2: 85% ceiling, skip if amp < 0.20)
        if len(trough_indices) >= 2:
            valid_sm   = sm_for_troughs[~np.isnan(sm_vals)]
            global_min = float(np.percentile(valid_sm, 5))
            global_max = float(np.percentile(valid_sm, 95))
            global_amp = global_max - global_min
            if global_amp >= 0.20:
                trough_ceil = global_min + 0.85 * global_amp
                trough_indices = [ti for ti in trough_indices
                                  if sm_for_troughs[ti] <= trough_ceil]

        # Gap-tolerance parameters (v4)
        _GAP_STRICT   = 0.20
        _GAP_TOLERANT = 0.50
        _AMP_GAP_THR  = 0.10

        def _cycle_has_gap(i_start, i_end, amplitude=None):
            if i_end <= i_start:
                return True
            gap_frac = np.isnan(sm_vals[i_start:i_end + 1]).mean()
            if amplitude is not None and amplitude >= _AMP_GAP_THR:
                return gap_frac > _GAP_TOLERANT
            return gap_frac > _GAP_STRICT

        rows = []

        def _date_in_window(d):
            m = d.month
            if sm <= em:
                return sm <= m <= em
            return m >= sm or m <= em

        # ── HEAD SEGMENT ─────────────────────────────────────
        # FIX v6: Only use head segment if data START looks like a genuine trough.
        # If data starts mid-season (NDVI already elevated), the head segment
        # creates a false season merging the tail of an invisible prior season
        # with the start of the real first season → wrong SOS/POS dates.
        _valid_sm_vals = sm_for_troughs[~np.isnan(sm_vals)]
        _global_min    = float(np.percentile(_valid_sm_vals, 5))  if len(_valid_sm_vals) > 0 else 0
        _global_amp    = (float(np.percentile(_valid_sm_vals, 95)) - _global_min) if len(_valid_sm_vals) > 0 else 1
        _head_trough_ceiling = _global_min + 0.25 * _global_amp  # start must be in bottom 25% of range
        _head_start_looks_like_trough = (float(sm_for_troughs[0]) <= _head_trough_ceiling)

        if trough_indices and _head_start_looks_like_trough:
            ti_first = trough_indices[0]
            head_len = ti_first
            _amp_pre = (float(np.max(sm_for_troughs[0:ti_first + 1])) -
                        float(sm_for_troughs[0]))
            if (head_len >= max(10, min_d // interp_freq) and
                    not _cycle_has_gap(0, ti_first, amplitude=_amp_pre)):
                try:
                    seg_sm  = sm_for_troughs[0:ti_first + 1]
                    seg_raw = ndvi_vals[0:ti_first + 1]
                    seg_t   = t_all[0:ti_first + 1]
                    _head_gap = np.isnan(sm_vals[0:ti_first + 1]).any()
                    work_arr  = seg_sm if _head_gap else seg_raw
                    ndvi_min  = float(sm_for_troughs[0]) if _head_gap else float(
                        ndvi_vals[0]) if not np.isnan(ndvi_vals[0]) else float(sm_for_troughs[0])
                    ndvi_max  = float(np.nanmax(work_arr))
                    A = ndvi_max - ndvi_min
                    if A >= MIN_AMPLITUDE:
                        sos_thr = ndvi_min + thr_pct     * A
                        eos_thr = ndvi_min + eos_thr_pct * A
                        # FIX 2: POS = raw peak
                        pi  = int(np.nanargmax(work_arr))
                        pos = seg_t[pi]
                        if _date_in_window(pos):
                            asc = work_arr[1:pi + 1]
                            sc  = np.where(asc >= sos_thr)[0]
                            desc = work_arr[pi:]
                            # EOS: first crossing below threshold; fallback = descending minimum
                            ec_below = np.where(desc < eos_thr)[0]
                            if len(ec_below):
                                ei = pi + max(0, int(ec_below[0]) - 1)
                            else:
                                ei = pi + int(np.nanargmin(desc))  # best estimate
                            if len(sc) and ei > 0:
                                si = 1 + int(sc[0])
                                if ei > si:
                                    sos = seg_t[si]; eos = seg_t[ei]
                                    data_end_dt = t_all[-1]
                                    if eos > data_end_dt:
                                        eos = data_end_dt
                                    if (eos - sos).days >= 365:
                                        eos = sos + pd.Timedelta(days=364)
                                    if eos > sos:
                                        trough_year  = seg_t[0].year
                                        season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                                        rows.append(_make_row(
                                            trough_year, season_start, sos, pos, eos,
                                            ndvi_max, A, ndvi_min, sos_thr, eos_thr,
                                            seg_t[int(np.argmin(seg_sm))], sm, em))
                except Exception:
                    pass

        # ── MAIN LOOP ─────────────────────────────────────────
        for i in range(len(trough_indices) - 1):
            try:
                ti  = trough_indices[i]
                ti1 = trough_indices[i + 1]
                if ti1 - ti < max(10, min_d // interp_freq):
                    continue
                _amp_pre = (float(np.max(sm_for_troughs[ti:ti1 + 1])) -
                            float(sm_for_troughs[ti]))
                if _cycle_has_gap(ti, ti1, amplitude=_amp_pre):
                    continue
                _has_gap  = np.isnan(sm_vals[ti:ti1 + 1]).any()
                cycle_raw = sm_for_troughs[ti:ti1 + 1] if _has_gap else ndvi_vals[ti:ti1 + 1]
                cycle_t   = t_all[ti:ti1 + 1]
                ndvi_min  = float(sm_for_troughs[ti]) if _has_gap else (
                    float(ndvi_vals[ti]) if not np.isnan(ndvi_vals[ti]) else float(sm_for_troughs[ti]))
                ndvi_max  = float(np.nanmax(cycle_raw))
                A = ndvi_max - ndvi_min
                if A < MIN_AMPLITUDE:
                    continue
                sos_thr = ndvi_min + thr_pct     * A
                eos_thr = ndvi_min + eos_thr_pct * A
                # FIX 2
                pos_idx = int(np.nanargmax(cycle_raw))
                asc     = cycle_raw[1:pos_idx + 1]
                sc      = np.where(asc >= sos_thr)[0] + 1
                if not len(sc): continue
                si = int(sc[0])
                desc = cycle_raw[pos_idx:-1]
                # EOS: first crossing below threshold; fallback = descending minimum
                ec_below = np.where(desc < eos_thr)[0]
                if len(ec_below):
                    ei = pos_idx + max(0, int(ec_below[0]) - 1)
                else:
                    ei = pos_idx + int(np.nanargmin(desc))  # best estimate
                if ei <= si: continue
                sos = cycle_t[si]; pos = cycle_t[pos_idx]; eos = cycle_t[ei]
                data_end_dt = t_all[-1]
                if eos > data_end_dt:
                    eos = data_end_dt
                if (eos - sos).days >= 365:
                    eos = sos + pd.Timedelta(days=364)
                if eos <= sos: continue
                if not _date_in_window(pos): continue
                trough_year  = t_all[ti].year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                rows.append(_make_row(
                    trough_year, season_start, sos, pos, eos,
                    ndvi_max, A, ndvi_min, sos_thr, eos_thr, t_all[ti], sm, em))
            except Exception:
                continue

        # ── TAIL SEGMENT ─────────────────────────────────────
        covered = set()
        for i in range(len(trough_indices) - 1):
            _a = (float(np.max(sm_for_troughs[trough_indices[i]:trough_indices[i+1]+1])) -
                  float(sm_for_troughs[trough_indices[i]]))
            if not _cycle_has_gap(trough_indices[i], trough_indices[i+1], amplitude=_a):
                covered.add(trough_indices[i])

        for ti0 in [ti for ti in trough_indices if ti not in covered]:
            tail_end = n - 1
            tail_len = tail_end - ti0
            if tail_len < max(10, min_d // interp_freq):
                continue
            try:
                _a = (float(np.max(sm_for_troughs[ti0:tail_end + 1])) -
                      float(sm_for_troughs[ti0]))
                if _cycle_has_gap(ti0, tail_end, amplitude=_a):
                    continue
                _has_gap = np.isnan(sm_vals[ti0:tail_end + 1]).any()
                seg_raw  = sm_for_troughs[ti0:tail_end + 1] if _has_gap else ndvi_vals[ti0:tail_end + 1]
                seg_t    = t_all[ti0:tail_end + 1]
                seg_sm   = sm_for_troughs[ti0:tail_end + 1]
                ndvi_min = float(sm_for_troughs[ti0]) if _has_gap else (
                    float(ndvi_vals[ti0]) if not np.isnan(ndvi_vals[ti0]) else float(sm_for_troughs[ti0]))
                ndvi_max = float(np.nanmax(seg_raw))
                A = ndvi_max - ndvi_min
                if A < MIN_AMPLITUDE: continue
                sos_thr = ndvi_min + thr_pct     * A
                eos_thr = ndvi_min + eos_thr_pct * A
                pi  = int(np.nanargmax(seg_raw))  # FIX 2
                asc = seg_raw[1:pi + 1]
                sc  = np.where(asc >= sos_thr)[0] + 1
                if not len(sc): continue
                si = int(sc[0])
                desc = seg_raw[pi:]
                # EOS: first crossing below threshold; fallback = descending minimum
                ec_below = np.where(desc < eos_thr)[0]
                if len(ec_below):
                    ei = pi + max(0, int(ec_below[0]) - 1)
                else:
                    ei = pi + int(np.nanargmin(desc))  # best estimate
                if ei <= si: continue
                sos = seg_t[si]; pos = seg_t[pi]; eos = seg_t[ei]
                data_end_dt = t_all[-1]
                if eos > data_end_dt:
                    eos = data_end_dt
                if (eos - sos).days >= 365:
                    eos = sos + pd.Timedelta(days=364)
                if eos <= sos: continue
                eos_is_at_data_end = (eos >= data_end_dt - pd.Timedelta(days=interp_freq*2))
                ndvi_at_data_end = float(ndvi_vals[-1]) if not np.isnan(ndvi_vals[-1]) else float(sm_for_troughs[-1])
                if eos_is_at_data_end and ndvi_at_data_end >= eos_thr:
                    continue  # Incomplete season — EOS not observed within data
                if not _date_in_window(pos): continue
                trough_year  = seg_t[0].year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                rows.append(_make_row(
                    trough_year, season_start, sos, pos, eos,
                    ndvi_max, A, ndvi_min, sos_thr, eos_thr, t_all[ti0], sm, em))
            except Exception:
                pass

        if not rows:
            return None, (
                f"No complete seasons detected. "
                f"Troughs found: {len(trough_indices)}. "
                f"Data: {ndvi_5d.index.min().date()} → {ndvi_5d.index.max().date()} "
                f"({n} pts, {interp_freq}d grid). "
                f"MIN_AMPLITUDE (data-derived) = {MIN_AMPLITUDE:.3f}. "
                f"Try: reduce Min Days slider, check season window, adjust threshold %."
            )

        df_out = pd.DataFrame(rows).drop_duplicates(subset="Year", keep="first")
        return df_out.sort_values("Year").reset_index(drop=True), None

    except Exception as e:
        return None, str(e)


def _make_row(year, season_start, sos, pos, eos, ndvi_max, A, ndvi_min,
              sos_thr, eos_thr, trough_date, sm, em):
    return {
        "Year":          year,
        "SOS_Date":      sos,  "SOS_DOY":  sos.dayofyear,
        "SOS_Target":    (sos - season_start).days,
        "POS_Date":      pos,  "POS_DOY":  pos.dayofyear,
        "POS_Target":    (pos - season_start).days,
        "EOS_Date":      eos,  "EOS_DOY":  eos.dayofyear,
        "EOS_Target":    (eos - season_start).days,
        "LOS_Days":      (eos - sos).days,
        "Season_Start":  season_start,
        "Peak_NDVI":     float(ndvi_max),
        "Amplitude":     float(A),
        "Base_NDVI":     float(ndvi_min),
        "Threshold_SOS": float(sos_thr),
        "Threshold_EOS": float(eos_thr),
        "Trough_Date":   trough_date,
    }


# ═══════════════════════════════════════════════════════════════
# FEATURE SELECTION — PURELY DATA-DRIVEN
# ═══════════════════════════════════════════════════════════════

def get_all_correlations(X, y):
    """Compute Pearson r + Spearman ρ for every feature vs target.
    Works with as few as 2 paired observations (returns r=1 or -1 for n=2,
    which is mathematically correct — user is warned via data quality badge)."""
    rows = []
    for col in X.columns:
        vals = X[col].dropna()
        if vals.std() < 1e-8 or len(vals) < 2:
            continue
        idx = vals.index.intersection(y.dropna().index)
        if len(idx) < 2:
            continue
        try:
            r,   p_val = pearsonr( vals[idx].astype(float), y[idx].astype(float))
            # spearmanr needs n>=3 for a meaningful p-value; graceful fallback for n=2
            if len(idx) >= 3:
                rho, p_sp = spearmanr(vals[idx].astype(float), y[idx].astype(float))
            else:
                rho, p_sp = float(r), 1.0   # n=2: Pearson and Spearman are identical
            composite = max(abs(r), abs(float(rho)))
            rows.append({
                'Feature':      col,
                'Pearson_r':    round(r,   3),
                '|r|':          round(abs(r), 3),
                'Spearman_rho': round(float(rho), 3),
                '|rho|':        round(abs(float(rho)), 3),
                'Composite':    round(composite, 3),
                'p_value':      round(p_val, 3),
                'Usable':       '✅' if composite >= MIN_CORR_THRESHOLD else '❌'
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values('Composite', ascending=False)


def _loo_r2_quick(X_vals, y_vals, alpha=0.01):
    """LOO R² — gracefully degrades for very small samples.
    n=2: LOO trains on 1 point → Ridge prediction = intercept only → returns 0.0
    n=1: returns 0.0 immediately (cannot evaluate)
    """
    n = len(y_vals)
    if n < 2:
        return 0.0
    if n == 2:
        # With 2 observations, LOO trains on 1 → always predicts mean → R²=0
        # Return actual Pearson r² as a proxy instead
        try:
            r, _ = pearsonr(X_vals[:, 0] if X_vals.ndim > 1 else X_vals, y_vals)
            return float(r ** 2)
        except Exception:
            return 0.0
    loo   = LeaveOneOut()
    preds = []
    sc    = StandardScaler()
    for tr, te in loo.split(X_vals):
        if len(tr) < 1:
            continue
        try:
            Xtr = sc.fit_transform(X_vals[tr])
            Xte = sc.transform(X_vals[te])
            m   = Ridge(alpha=alpha)
            m.fit(Xtr, y_vals[tr])
            preds.append(float(m.predict(Xte)[0]))
        except Exception:
            preds.append(float(y_vals[tr].mean()))
    if not preds:
        return 0.0
    preds  = np.array(preds)
    ss_res = np.sum((y_vals - preds) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2) + 1e-12
    return float(np.clip(1 - ss_res / ss_tot, -1, 1))


def select_multi_features(X, y, max_features=5, min_r=MIN_CORR_THRESHOLD,
                          user_max_features=None):
    """
    Data-driven feature selection with optional user override for max features.
    Collinearity threshold is relaxed for small n (<=5) to avoid discarding
    genuinely different variables that happen to correlate in small samples.
    """
    n_obs = len(y.dropna())

    if n_obs <= 3:
        effective_min_r = min(min_r, 0.10)
    elif n_obs <= 5:
        effective_min_r = min(min_r, 0.25)
    else:
        effective_min_r = min_r

    # With n<=5, virtually all monotonic pairs get |r|>0.85 by chance.
    # Use 0.97 so only genuinely redundant (near-duplicate) features are dropped.
    collinear_thr = 0.97 if n_obs <= 5 else 0.85

    usable = []
    for col in X.columns:
        vals = X[col].dropna()
        if vals.std() < 1e-8 or len(vals) < 2:
            continue
        idx = vals.index.intersection(y.dropna().index)
        if len(idx) < 2:
            continue
        try:
            rp, _ = pearsonr(vals[idx].astype(float), y[idx].astype(float))
            rs = rp
            if len(idx) >= 3:
                rs, _ = spearmanr(vals[idx].astype(float), y[idx].astype(float))
        except Exception:
            continue
        composite = max(abs(rp), abs(float(rs)))
        if composite >= effective_min_r:
            usable.append((col, composite))

    if not usable:
        return []
    usable.sort(key=lambda x: -x[1])

    collinear_filtered = []
    for feat, score in usable:
        collinear = False
        for sel in collinear_filtered:
            xi   = X[feat].fillna(X[feat].median())
            xj   = X[sel].fillna(X[sel].median())
            idx2 = xi.index.intersection(xj.index)
            if len(idx2) < 2:
                continue
            try:
                r_pair, _ = pearsonr(xi[idx2].astype(float), xj[idx2].astype(float))
            except Exception:
                continue
            if abs(r_pair) > collinear_thr:
                collinear = True
                break
        if not collinear:
            collinear_filtered.append(feat)

    # Effective max features: user override takes priority
    if user_max_features is not None:
        effective_max = user_max_features
    elif n_obs <= 3:
        effective_max = 1
    elif n_obs <= 5:
        effective_max = min(max_features, 2)
    else:
        effective_max = max_features

    max_safe   = max(1, n_obs - 1)
    candidates = collinear_filtered[:min(effective_max, max_safe)]
    if len(candidates) <= 1:
        return candidates

    y_vals = y.values.astype(float)
    selected = [candidates[0]]
    best_r2  = _loo_r2_quick(
        X[selected].fillna(X[selected[0]].median()).values.reshape(-1, 1), y_vals)

    # Stricter improvement requirement for small datasets
    improvement_thr = 0.08 if n_obs <= 5 else 0.03

    for feat in candidates[1:]:
        trial = selected + [feat]
        Xt    = X[trial].fillna(X[trial].median()).values
        try:
            trial_r2 = _loo_r2_quick(Xt, y_vals)
        except Exception:
            continue
        if trial_r2 > best_r2 + improvement_thr:
            selected.append(feat)
            best_r2 = trial_r2
    return selected



# ═══════════════════════════════════════════════════════════════
# MODEL FITTING
# ═══════════════════════════════════════════════════════════════

def loo_ridge(X_vals, y_vals, alpha):
    """LOO ridge — gracefully handles very small samples.
    n=2: returns direct fit R² (LOO would train on 1 point, meaningless).
    n=1: returns 0, inf.
    """
    n = len(y_vals)
    if n < 2:
        return 0.0, float('inf')
    if n == 2:
        # Direct fit on 2 points — R² = 1 (perfect), MAE from in-sample
        pipe = Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=alpha))])
        pipe.fit(X_vals, y_vals)
        preds  = pipe.predict(X_vals)
        mae    = float(mean_absolute_error(y_vals, preds))
        try:
            r, _ = pearsonr(X_vals[:, 0], y_vals)
            r2   = float(r ** 2)
        except Exception:
            r2 = 0.0
        return r2, mae
    loo   = LeaveOneOut()
    preds = []
    for tr, te in loo.split(X_vals):
        pipe = Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=alpha))])
        try:
            pipe.fit(X_vals[tr], y_vals[tr])
            preds.append(float(pipe.predict(X_vals[te])[0]))
        except Exception:
            preds.append(float(y_vals[tr].mean()))
    preds  = np.array(preds)
    ss_res = np.sum((y_vals - preds) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2) + 1e-12
    return float(np.clip(1 - ss_res / ss_tot, -1, 1)), float(mean_absolute_error(y_vals, preds))


def fit_loess(X_vals_1d, y_vals, frac=0.75):
    n     = len(y_vals)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        preds[i] = _loess_predict(
            X_vals_1d[mask].astype(float),
            y_vals[mask].astype(float),
            np.array([X_vals_1d[i]]), frac=frac)[0]
    ss_res = np.sum((y_vals - preds) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2) + 1e-12
    return (float(np.clip(1 - ss_res / ss_tot, -1, 1)),
            float(mean_absolute_error(y_vals, preds)))


def loo_poly(X_vals, y_vals, degree=2):
    loo   = LeaveOneOut()
    preds = []
    for tr, te in loo.split(X_vals):
        pipe = Pipeline([('sc', StandardScaler()),
                         ('pf', PolynomialFeatures(degree=degree, include_bias=False)),
                         ('r',  Ridge(alpha=1.0))])
        pipe.fit(X_vals[tr], y_vals[tr])
        preds.append(float(pipe.predict(X_vals[te])[0]))
    preds  = np.array(preds)
    ss_res = np.sum((y_vals - preds) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2) + 1e-12
    return (float(np.clip(1 - ss_res / ss_tot, -1, 1)),
            float(mean_absolute_error(y_vals, preds)))


def fit_event_model(X_all, y, model_key="ridge", user_max_features=None):
    """
    Fit a model for one phenological event.
    For small data (n < 5), automatically falls back to the most stable model:
      n=2-3 → single-feature Ridge (most stable)
      n=4-5 → single-feature Ridge or LOESS if user picks it
      n≥6   → all models available
    """
    yt = y.values; n = len(yt)

    # ── Auto-adapt model for very small datasets ─────────────────────────────
    effective_model_key = model_key
    if n <= 3:
        # poly2/poly3/gpr all need more points than this — force ridge
        effective_model_key = "ridge"
    elif n <= 5:
        # poly3 and GPR are unreliable — prefer ridge or loess
        if model_key in ("poly3", "gpr"):
            effective_model_key = "ridge"

    # ── Feature selection (adapts max_features to n automatically) ───────────
    # Lower the correlation threshold slightly for small data to avoid no-feature fallback
    adaptive_min_r = MIN_CORR_THRESHOLD
    if n <= 4:
        adaptive_min_r = max(0.25, MIN_CORR_THRESHOLD - 0.15)  # 0.25 floor
    elif n <= 6:
        adaptive_min_r = max(0.30, MIN_CORR_THRESHOLD - 0.10)

    features = select_multi_features(X_all, y, max_features=5, min_r=adaptive_min_r,
                                     user_max_features=user_max_features)

    if not features:
        md = float(yt.mean())
        return {'mode': 'mean', 'features': [], 'r2': 0.0,
                'mae': float(np.mean(np.abs(yt - md))),
                'alpha': None, 'coef': [], 'intercept': md,
                'best_r': 0.0, 'mean_doy': md, 'n': n, 'pipe': None,
                'model_key': effective_model_key,
                'adaptive_min_r': adaptive_min_r}

    Xf = X_all[features].fillna(X_all[features].median())
    Xv = Xf.values

    best_single_r = 0.0
    for f in features:
        try:
            r_val, _ = pearsonr(Xf[f].astype(float), y.astype(float))
            if abs(r_val) > best_single_r:
                best_single_r = abs(r_val)
        except Exception:
            pass

    if effective_model_key == "loess":
        feat = features[0]
        x1d  = Xf[feat].values.astype(float)
        r2, mae = fit_loess(x1d, yt.astype(float))
        return {'mode': 'loess', 'features': [feat], 'r2': r2, 'mae': mae,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'mean_doy': float(yt.mean()), 'n': n,
                'pipe': None, 'model_key': effective_model_key,
                'x_train': x1d, 'y_train': yt.astype(float)}

    if effective_model_key in ("poly2", "poly3"):
        degree = 2 if effective_model_key == "poly2" else 3
        r2, mae = loo_poly(Xv, yt, degree=degree)
        pipe = Pipeline([('sc', StandardScaler()),
                         ('pf', PolynomialFeatures(degree=degree, include_bias=False)),
                         ('r',  Ridge(alpha=1.0))])
        pipe.fit(Xv, yt)
        return {'mode': effective_model_key, 'features': features, 'r2': r2, 'mae': mae,
                'alpha': 1.0, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'mean_doy': float(yt.mean()), 'n': n,
                'pipe': pipe, 'model_key': effective_model_key}

    if effective_model_key == "gpr":
        sc   = StandardScaler()
        Xvs  = sc.fit_transform(Xv)
        yt_f = yt.astype(float)
        kernel = (RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) +
                  WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3)))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                       normalize_y=True, random_state=42)
        loo_cv = LeaveOneOut()
        preds  = np.zeros(n)
        for tr, te in loo_cv.split(Xvs):
            g = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                         normalize_y=True, random_state=42)
            g.fit(Xvs[tr], yt_f[tr])
            preds[te] = g.predict(Xvs[te])
        ss_res = np.sum((yt_f - preds) ** 2)
        ss_tot = np.sum((yt_f - yt_f.mean()) ** 2) + 1e-12
        r2  = float(np.clip(1 - ss_res / ss_tot, -1, 1))
        mae = float(np.mean(np.abs(yt_f - preds)))
        gpr.fit(Xvs, yt_f)
        return {'mode': 'gpr', 'features': features, 'r2': r2, 'mae': mae,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'mean_doy': float(yt.mean()), 'n': n,
                'pipe': None, 'gpr_model': gpr, 'gpr_scaler': sc,
                'model_key': effective_model_key}

    # Default: Ridge with LOO-tuned alpha
    rcv = RidgeCV(alphas=ALPHAS, cv=LeaveOneOut())
    rcv.fit(StandardScaler().fit_transform(Xv), yt)
    best_alpha = float(rcv.alpha_)
    pipe = Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=best_alpha))])
    pipe.fit(Xv, yt)
    r2, mae = loo_ridge(Xv, yt, best_alpha)
    sc    = pipe.named_steps['sc']
    ridge = pipe.named_steps['r']
    coef_unstd = list(ridge.coef_ / sc.scale_)
    intercept_unstd = float(ridge.intercept_ - np.dot(ridge.coef_ / sc.scale_, sc.mean_))
    return {'mode': 'ridge', 'features': features, 'r2': r2, 'mae': mae,
            'alpha': best_alpha, 'coef': coef_unstd, 'intercept': intercept_unstd,
            'best_r': best_single_r, 'n': n, 'pipe': pipe, 'model_key': model_key}


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL PREDICTOR CLASS
# ═══════════════════════════════════════════════════════════════

class UniversalPredictor:
    def __init__(self):
        self._fits      = {}
        self.r2         = {}
        self.mae        = {}
        self.n_seasons  = {}
        self.corr_tables = {}

    def train(self, train_df, all_params, model_key="ridge", user_max_features=None):
        meta      = {'Year', 'Event', 'Target_DOY', 'LOS_Days', 'Peak_NDVI', 'Season_Start'}
        feat_cols = [c for c in train_df.columns
                     if c not in meta
                     and pd.api.types.is_numeric_dtype(train_df[c])
                     and train_df[c].std() > 1e-8]
        for event in ['SOS', 'POS', 'EOS']:
            sub = train_df[train_df['Event'] == event].copy()
            self.n_seasons[event] = len(sub)
            if len(sub) < 2:
                continue
            X   = sub[feat_cols].fillna(sub[feat_cols].median())
            y   = sub['Target_DOY']
            self.corr_tables[event] = get_all_correlations(X, y)
            fit = fit_event_model(X, y, model_key=model_key,
                                  user_max_features=user_max_features)
            self._fits[event] = fit
            self.r2[event]    = fit['r2']
            self.mae[event]   = fit['mae']

    def predict(self, inputs, event, year=2026, season_start_month=6):
        if event not in self._fits:
            return None
        fit = self._fits[event]
        if fit['mode'] == 'mean':
            rel_days = int(round(fit['mean_doy']))
        elif fit['mode'] == 'loess':
            feat    = fit['features'][0]
            x_new   = float(inputs.get(feat, 0.0))
            x_train = fit.get('x_train')
            y_train = fit.get('y_train')
            if x_train is not None and len(x_train) >= 2:
                pred     = _loess_predict(x_train, y_train, np.array([x_new]), frac=0.75)[0]
                rel_days = int(np.clip(round(float(pred)), 0, 500))
            else:
                rel_days = int(round(fit['mean_doy']))
        elif fit['mode'] == 'gpr':
            vals = np.array([[inputs.get(f, 0.0) for f in fit['features']]])
            sc   = fit.get('gpr_scaler')
            gpr  = fit.get('gpr_model')
            if sc is not None and gpr is not None:
                pred     = gpr.predict(sc.transform(vals))[0]
                rel_days = int(np.clip(round(float(pred)), 0, 500))
            else:
                rel_days = int(round(fit['mean_doy']))
        else:
            vals     = np.array([[inputs.get(f, 0.0) for f in fit['features']]])
            rel_days = int(np.clip(round(float(fit['pipe'].predict(vals)[0])), 0, 500))

        season_start = datetime(year, season_start_month, 1)
        date = season_start + timedelta(days=rel_days)
        doy  = date.timetuple().tm_yday
        return {'doy': doy, 'date': date, 'rel_days': rel_days,
                'r2': self.r2[event], 'mae': self.mae[event], 'event': event}

    def equation_str(self, event, season_start_month=6):
        if event not in self._fits:
            return f"Need ≥ 2 seasons with met data to fit a model (currently {self.n_seasons.get(event, 0)})"
        fit = self._fits[event]
        mo  = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        lbl = f"{event}_days_from_{mo.get(season_start_month,'Jan')}1"
        if fit['mode'] == 'mean':
            return (f"{lbl} ≈ {fit['mean_doy']:.0f}  "
                    f"[No feature |r|≥{MIN_CORR_THRESHOLD} — mean only]")
        if fit['mode'] == 'gpr':
            return (f"{lbl}  =  GPR({', '.join(fit['features'])})\n"
                    f"    [Gaussian Process, RBF+WhiteKernel, R²(LOO)={fit['r2']:.3f}, MAE=±{fit['mae']:.1f} d]")
        if fit['mode'] == 'loess':
            return (f"{lbl}  =  LOESS({fit['features'][0]})\n"
                    f"    [Locally-weighted, R²(LOO)={fit['r2']:.3f}, MAE=±{fit['mae']:.1f} d]")
        terms = [f"{fit.get('intercept', 0.0):.3f}"]
        for feat, coef in zip(fit['features'], fit['coef']):
            s = '+' if coef >= 0 else '-'
            terms.append(f"{s} {abs(coef):.5f} × {feat}")
        return (f"{lbl}  =  " + "  ".join(terms) +
                f"\n    [Ridge α={fit['alpha']}, {len(fit['features'])} feature(s), "
                f"R²(LOO)={fit['r2']:.3f}, MAE=±{fit['mae']:.1f} d]")

    def corr_table_for_display(self, event):
        if event not in self._fits:
            return pd.DataFrame()
        fit = self._fits[event]
        ct  = self.corr_tables.get(event)
        if ct is None or len(ct) == 0:
            return pd.DataFrame()
        in_model   = set(fit['features'])
        # Reconstruct which features were collinear vs dropped by LOO
        # by checking collinearity between correlated features and the selected ones
        selected_first = fit['features'][0] if fit['features'] else None
        rows = []
        for _, row in ct.iterrows():
            feat   = row['Feature']
            usable = row['Usable'] == '✅'
            if feat in in_model:
                role = '✅  In model'
            elif usable:
                # Check if collinear with selected feature
                is_collinear = False
                collinear_with = None
                if selected_first and selected_first in ct['Feature'].values:
                    try:
                        sel_r = ct[ct['Feature'] == selected_first]['Pearson_r'].values[0]
                        feat_r = row['Pearson_r']
                        # If both have same-sign high |r| with target → likely collinear
                        if abs(feat_r) > 0.85 and abs(sel_r) > 0.85 and (feat_r * sel_r > 0 or abs(abs(feat_r) - abs(sel_r)) < 0.15):
                            is_collinear = True
                            collinear_with = selected_first
                    except Exception:
                        pass
                if is_collinear and collinear_with:
                    role = f'➖  Redundant — highly similar to {collinear_with}'
                else:
                    role = '➖  Did not improve model accuracy — not added'
            else:
                role = '⬜  Below correlation threshold'
            rows.append({'Feature':    feat,
                         'Pearson r':  row['Pearson_r'],
                         'Spearman ρ': row.get('Spearman_rho', float('nan')),
                         'Composite':  row.get('Composite', row['|r|']),
                         'Role':       role})
        return pd.DataFrame(rows)

    def export_coefficients(self, season_start_month=6):
        """Export all model coefficients as a DataFrame for download."""
        records = []
        mo = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
              7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        for event in ['SOS', 'POS', 'EOS']:
            if event not in self._fits:
                continue
            fit = self._fits[event]
            if fit['mode'] == 'mean':
                records.append({'Event': event, 'Feature': 'INTERCEPT',
                                'Coefficient': fit['mean_doy'],
                                'Model': 'mean', 'R2_LOO': 0.0, 'MAE_days': fit['mae']})
            elif fit['mode'] == 'ridge' and fit['coef']:
                for feat, coef in zip(fit['features'], fit['coef']):
                    records.append({'Event': event, 'Feature': feat, 'Coefficient': round(coef, 6),
                                    'Model': 'Ridge', 'Alpha': fit['alpha'],
                                    'R2_LOO': round(fit['r2'], 4), 'MAE_days': round(fit['mae'], 2)})
                records.append({'Event': event, 'Feature': 'INTERCEPT',
                                'Coefficient': round(fit.get('intercept', 0), 4),
                                'Model': 'Ridge', 'Alpha': fit['alpha'],
                                'R2_LOO': round(fit['r2'], 4), 'MAE_days': round(fit['mae'], 2)})
            else:
                records.append({'Event': event, 'Feature': str(fit['features']),
                                'Coefficient': float('nan'), 'Model': fit['mode'],
                                'R2_LOO': round(fit['r2'], 4), 'MAE_days': round(fit['mae'], 2)})
        return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_ndvi_phenology(ndvi_raw, pheno_df, season_window=None, interp_freq=5):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    dates   = pd.to_datetime(ndvi_raw['Date'])
    ax.scatter(dates, ndvi_raw['NDVI'], color='#A5D6A7', s=18, alpha=0.55,
               label='NDVI (raw obs)', zorder=3)

    ndvi_s = ndvi_raw.set_index('Date')['NDVI'].sort_index()
    ndvi_s = ndvi_s[~ndvi_s.index.duplicated(keep='first')]
    orig_dates = ndvi_s.index.sort_values()
    orig_diffs = pd.Series(orig_dates).diff().dt.days.fillna(0)
    pos_diffs  = orig_diffs[orig_diffs > 0]
    typical_cad = float(pos_diffs.median()) if len(pos_diffs) > 0 else 16.0
    MAX_GAP     = max(60, int(typical_cad * 8))
    gap_starts  = orig_dates[orig_diffs.values > MAX_GAP]
    full_range  = pd.date_range(start=ndvi_s.index.min(), end=ndvi_s.index.max(),
                                freq=f'{interp_freq}D')
    ndvi_5d = ndvi_s.reindex(ndvi_s.index.union(full_range)).interpolate(
        method='time', limit_area='inside')
    for g in gap_starts:
        before = orig_dates[orig_dates < g]
        if len(before) == 0: continue
        mask = (ndvi_5d.index > before[-1]) & (ndvi_5d.index < g)
        ndvi_5d.loc[mask] = np.nan
    ndvi_5d = ndvi_5d.reindex(full_range)

    n, ndvi_vals = len(ndvi_5d), ndvi_5d.values.copy()
    valid_mask = ~np.isnan(ndvi_vals)
    sm_arr = np.full(n, np.nan)
    seg_labels = np.zeros(n, dtype=int); seg_id, in_seg = 0, False
    for i in range(n):
        if valid_mask[i]:
            if not in_seg: seg_id += 1; in_seg = True
            seg_labels[i] = seg_id
        else:
            in_seg = False

    _MAX_SG = 31
    for sid in range(1, seg_id + 1):
        idx_seg = np.where(seg_labels == sid)[0]; seg_n = len(idx_seg)
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

    ax.plot(ndvi_5d.index, sm_arr, color='#1B5E20', lw=2.2,
            label='Smoothed (SG, data-adaptive w≤31)', zorder=5)

    # Gap shading
    in_gap = False; gap_s = None
    for i in range(n):
        nan_now = np.isnan(sm_arr[i])
        if nan_now and not in_gap: gap_s = ndvi_5d.index[i]; in_gap = True
        elif not nan_now and in_gap:
            ax.axvspan(gap_s, ndvi_5d.index[i], color='#BDBDBD', alpha=0.30, label='Data gap')
            in_gap = False
    if in_gap: ax.axvspan(gap_s, ndvi_5d.index[-1], color='#BDBDBD', alpha=0.30)

    # Season window shading
    if season_window:
        ws_m, we_m = season_window
        y_min = ndvi_5d.index.year.min(); y_max = ndvi_5d.index.year.max() + 1
        plotted = False
        for yr in range(y_min, y_max + 1):
            try:
                ws = pd.Timestamp(f"{yr}-{ws_m:02d}-01")
                we = pd.Timestamp(f"{yr+1}-{we_m:02d}-28") if ws_m > we_m \
                    else pd.Timestamp(f"{yr}-{we_m:02d}-28")
                ds, de = ndvi_5d.index[0], ndvi_5d.index[-1]
                if we < ds or ws > de: continue
                lbl = 'Selected season window' if not plotted else ''
                ax.axvspan(max(ws, ds), min(we, de), color='#A5D6A7', alpha=0.12, zorder=0, label=lbl)
                plotted = True
            except Exception:
                pass

    # Per-season threshold & amplitude annotation
    thr_sos_p = thr_eos_p = base_p = False
    for _, row in pheno_df.iterrows():
        td = row.get('Trough_Date'); ed = row.get('EOS_Date')
        base = row.get('Base_NDVI'); thr_s = row.get('Threshold_SOS')
        thr_e = row.get('Threshold_EOS'); pk = row.get('Peak_NDVI')
        amp = row.get('Amplitude'); sd = row.get('SOS_Date'); pd_ = row.get('POS_Date')
        if pd.isna(td) or pd.isna(ed): continue
        seg_st = pd.Timestamp(td); seg_en = pd.Timestamp(ed) + pd.Timedelta(days=20)
        if pd.notna(base):
            ax.hlines(base, seg_st, seg_en, colors='#F57F17', lw=1.1, ls=':', alpha=0.75,
                      label='Base NDVI' if not base_p else '', zorder=4)
            base_p = True
        if pd.notna(thr_s):
            ax.hlines(thr_s, seg_st, seg_en, colors='#66BB6A', lw=1.2, ls='--', alpha=0.70,
                      label='SOS threshold' if not thr_sos_p else '', zorder=4)
            thr_sos_p = True
        if pd.notna(thr_e):
            ax.hlines(thr_e, seg_st, seg_en, colors='#FFA726', lw=1.2, ls='--', alpha=0.70,
                      label='EOS threshold' if not thr_eos_p else '', zorder=4)
            thr_eos_p = True
        if pd.notna(pd_) and pd.notna(amp) and amp > 0.01 and pd.notna(base):
            px = pd.Timestamp(pd_)
            ax.annotate('', xy=(px, pk), xytext=(px, base),
                        arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=1.1))
            ax.text(px, base + amp * 0.5, f'  A={amp:.3f}', fontsize=7, color='#7B1FA2',
                    va='center', ha='left')

    ev_colors = {'SOS': '#43A047', 'POS': '#1565C0', 'EOS': '#E65100'}
    plotted_ev = set()
    for _, row in pheno_df.iterrows():
        for ev, col in ev_colors.items():
            d = row.get(f'{ev}_Date')
            if pd.notna(d):
                ax.axvline(d, color=col, lw=1.4, alpha=0.55, ls='--',
                           label=f'{ev}' if ev not in plotted_ev else '')
                plotted_ev.add(ev)

    mo = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
          7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    win_str = (f"  |  Window: {mo.get(season_window[0], '?')} → {mo.get(season_window[1], '?')}"
               if season_window else '')
    ax.set_title(f'NDVI Time Series — Data-Driven Phenology Extraction{win_str}\n'
                 'Amplitude thresholds calibrated per-cycle from data',
                 fontsize=11, fontweight='bold', color='#1B5E20')
    ax.set_xlabel('Date'); ax.set_ylabel('NDVI')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.legend(ncol=4, fontsize=7.5, loc='upper left', framealpha=0.88)
    ax.grid(True, alpha=0.22, ls='--'); ax.set_facecolor('#FAFFF8')
    fig.tight_layout()
    return fig


def plot_pheno_trends(pheno_df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    fig.patch.set_facecolor('#F7FBF5')
    ev_cfg = [
        ('SOS', 'SOS_Date', 'SOS_DOY',    '#43A047', 'SOS — Green-up start'),
        ('POS', 'POS_Date', 'POS_DOY',    '#1565C0', 'POS — Peak greenness'),
        ('EOS', 'EOS_Date', 'EOS_Target', '#E65100', 'EOS — Senescence end'),
        ('LOS', None,       'LOS_Days',   '#795548', 'LOS — Season length (days)'),
    ]
    for ax, (ev, date_col, doy_col, clr, lbl) in zip(axes, ev_cfg):
        yrs  = pheno_df['Year'].values
        vals = pheno_df[doy_col].values.astype(float) if doy_col in pheno_df.columns else np.zeros(len(yrs))
        ax.bar(yrs, vals, color=clr, alpha=0.45, width=0.7, edgecolor='white')
        ax.plot(yrs, vals, 'o-', color=clr, ms=6, lw=2, markeredgecolor='white', markeredgewidth=1.2)
        if ev == 'LOS':
            ax.set_ylabel('Days')
        elif ev != 'EOS' and date_col and date_col in pheno_df.columns:
            unique_v = np.linspace(vals.min(), vals.max(), 5).astype(int)
            tick_l = []
            for doy in unique_v:
                try:
                    tick_l.append((pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(doy) - 1)).strftime('%b %d'))
                except Exception:
                    tick_l.append(str(doy))
            ax.set_yticks(unique_v); ax.set_yticklabels(tick_l, fontsize=8); ax.set_ylabel('Date (approx.)')
        else:
            ax.set_ylabel('Season-relative days' if ev == 'EOS' else 'DOY')
        if len(yrs) >= 2:
            m, b = np.polyfit(yrs, vals, 1)
            ax.plot(yrs, m * yrs + b, '--', color='#263238', lw=1.8, label=f'Trend: {m:+.1f} d/yr')
            ax.legend(fontsize=8.5, framealpha=0.85)
        ax.set_title(lbl, fontsize=10, fontweight='bold', color='#1B4332')
        ax.set_xlabel('Year', fontsize=9)
        ax.grid(True, alpha=0.22, ls='--'); ax.set_facecolor('#FAFFF8'); ax.tick_params(labelsize=8.5)
    fig.suptitle('Phenological Trends — Data-Derived from Uploaded NDVI',
                 fontsize=13, fontweight='bold', color='#1B4332', y=1.02)
    fig.tight_layout()
    return fig


def plot_obs_vs_pred(predictor, train_df):
    events = [ev for ev in ['SOS', 'POS', 'EOS']
              if ev in predictor._fits and predictor._fits[ev].get('pipe') is not None
              and predictor._fits[ev]['mode'] in ('ridge', 'poly2', 'poly3')]
    if not events:
        return None
    fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 4.5), squeeze=False)
    clrs = {'SOS': '#4CAF50', 'POS': '#1565C0', 'EOS': '#FF6F00'}
    for ax, ev in zip(axes[0], events):
        fit  = predictor._fits[ev]
        sub  = train_df[train_df['Event'] == ev].copy()
        feats = [f for f in fit['features'] if f in sub.columns]
        if not feats: continue
        Xf   = sub[feats].fillna(sub[feats].median())
        try:
            pred = fit['pipe'].predict(Xf.values)
        except Exception:
            continue
        obs  = sub['Target_DOY'].values
        ax.scatter(obs, pred, color=clrs[ev], s=80, edgecolors='white', lw=1.5, zorder=3, alpha=0.9)
        lims = [min(obs.min(), pred.min()) - 8, max(obs.max(), pred.max()) + 8]
        ax.plot(lims, lims, 'k--', lw=1.2); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f'{ev}   R²(LOO)={predictor.r2.get(ev, 0):.3f}   MAE={predictor.mae.get(ev, 0):.1f} d\n'
                     f'{" + ".join(feats)}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Observed (days from season start)')
        ax.set_ylabel('Predicted (days from season start)')
        ax.grid(True, alpha=0.25); ax.set_facecolor('#FAFFF8')
    fig.suptitle('Observed vs Predicted (training fit — data-driven features)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_correlation_summary(predictor):
    events    = ['SOS', 'POS', 'EOS']
    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#BF360C'}
    feat_r    = {}
    for ev in events:
        ct = predictor.corr_tables.get(ev)
        if ct is None: continue
        for _, row in ct.iterrows():
            f = row['Feature']
            feat_r[f] = max(feat_r.get(f, 0), abs(row['Pearson_r']))
    all_feats = sorted(feat_r, key=lambda f: -feat_r[f])[:14]
    r_mat = pd.DataFrame(0.0, index=all_feats, columns=events)
    p_mat = pd.DataFrame(1.0, index=all_feats, columns=events)
    for ev in events:
        ct = predictor.corr_tables.get(ev)
        if ct is None: continue
        for _, row in ct.iterrows():
            f = row['Feature']
            if f not in r_mat.index: continue
            r_mat.loc[f, ev] = row['Pearson_r']
            if 'p_value' in row and not pd.isna(row['p_value']):
                p_mat.loc[f, ev] = row['p_value']

    n_feats = len(all_feats)
    fig = plt.figure(figsize=(16, max(6, n_feats * 0.52 + 2.5)))
    fig.patch.set_facecolor('#F8FBF7')
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.1], wspace=0.36)
    ax_bar = fig.add_subplot(gs[0])

    bar_h = 0.22; y_pos = np.arange(n_feats); offs = np.array([-1, 0, 1]) * bar_h
    for i, ev in enumerate(events):
        vals      = r_mat[ev].values
        bar_clrs  = [ev_colors[ev] if abs(v) >= MIN_CORR_THRESHOLD else '#CFCFCF' for v in vals]
        ax_bar.barh(y_pos + offs[i], vals, height=bar_h * 0.82,
                    color=bar_clrs, edgecolor='white', lw=0.4, label=ev, alpha=0.88)
    ax_bar.axvline(0, color='#37474F', lw=1.0)
    for thr in [MIN_CORR_THRESHOLD, -MIN_CORR_THRESHOLD]:
        ax_bar.axvline(thr, color='#555', lw=1.1, ls='--', alpha=0.55)
    ax_bar.axvspan( MIN_CORR_THRESHOLD, 1.05, alpha=0.035, color='#4CAF50')
    ax_bar.axvspan(-1.05, -MIN_CORR_THRESHOLD, alpha=0.035, color='#4CAF50')
    ax_bar.set_yticks(y_pos); ax_bar.set_yticklabels(all_feats, fontsize=9.5)
    ax_bar.set_xlim(-1.05, 1.05)
    ax_bar.set_xlabel('Pearson r  (with event DOY)', fontsize=10, fontweight='bold')
    ax_bar.set_title(f'Feature Correlations — SOS / POS / EOS\n'
                     f'Coloured = |r| ≥ {MIN_CORR_THRESHOLD} (usable)  ·  Grey = below threshold',
                     fontsize=9.5, fontweight='bold', color='#1B4332', pad=7)
    ax_bar.grid(True, axis='x', alpha=0.20, ls='--'); ax_bar.set_facecolor('#FAFFF8')
    ax_bar.legend(title='Event', fontsize=9, title_fontsize=9, loc='lower right', framealpha=0.92,
                  handles=[plt.matplotlib.patches.Patch(color=ev_colors[e], label=e) for e in events])

    ax_hm = fig.add_subplot(gs[1])
    mat = r_mat.values.astype(float)
    im  = ax_hm.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
    ax_hm.set_xticks(range(3)); ax_hm.set_xticklabels(events, fontsize=11, fontweight='bold')
    ax_hm.set_yticks(range(n_feats)); ax_hm.set_yticklabels(all_feats, fontsize=9)
    for i in range(n_feats):
        for j in range(3):
            v  = mat[i, j]; pv = p_mat.iloc[i, j]
            star = '**' if pv < 0.05 else ('*' if pv < 0.10 else '')
            tc   = 'white' if abs(v) > 0.60 else '#1A1A1A'
            ax_hm.text(j, i, f'{v:.2f}{star}', ha='center', va='center',
                       fontsize=8.5, fontweight='bold', color=tc)
            if abs(v) >= MIN_CORR_THRESHOLD:
                rect = plt.matplotlib.patches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96, boxstyle='round,pad=0.02',
                    linewidth=1.8, edgecolor='#1B4332', facecolor='none')
                ax_hm.add_patch(rect)
    cb = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8); cb.set_label('Pearson r', fontsize=9)
    ax_hm.set_title('Pearson r Heatmap\n** p<0.05  * p<0.10', fontsize=9.5,
                    fontweight='bold', color='#1B4332', pad=7)
    ax_hm.spines[:].set_visible(False)
    for i in range(n_feats):
        if i % 2 == 0:
            ax_hm.axhspan(i - 0.5, i + 0.5, color='#F0F0F0', alpha=0.25, zorder=0)
    fig.suptitle('Data-Driven Feature Correlations with Phenological Events',
                 fontsize=13, fontweight='bold', color='#1B4332', y=1.005)
    fig.tight_layout()
    return fig


def plot_data_summary(ndvi_info, met_info):
    """Data characterization panel — purely from uploaded files."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#F8FBF7')

    # NDVI distribution
    ax = axes[0]
    p5, p95 = ndvi_info['ndvi_p5'], ndvi_info['ndvi_p95']
    mean_v, std_v = ndvi_info['ndvi_mean'], ndvi_info['ndvi_std']
    x = np.linspace(p5 - 0.1, p95 + 0.1, 200)
    y = np.exp(-0.5 * ((x - mean_v) / (std_v + 1e-6)) ** 2)
    ax.fill_between(x, y, alpha=0.35, color='#2E7D32')
    ax.axvline(mean_v, color='#1B5E20', lw=2.0, label=f'Mean = {mean_v:.3f}')
    ax.axvline(p5,  color='#F57F17', lw=1.5, ls='--', label=f'P5  = {p5:.3f}')
    ax.axvline(p95, color='#E53935', lw=1.5, ls='--', label=f'P95 = {p95:.3f}')
    ax.set_title(f'NDVI Distribution\n(data range = {ndvi_info["data_range"]:.3f})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('NDVI'); ax.legend(fontsize=8.5); ax.set_facecolor('#FAFFF8')
    ax.grid(True, alpha=0.22)

    # Met parameters heatmap (top params by absolute mean)
    ax2 = axes[1]
    if met_info:
        top_params = sorted(met_info.keys(), key=lambda p: abs(met_info[p]['mean']), reverse=True)[:10]
        means = [met_info[p]['mean'] for p in top_params]
        stds  = [met_info[p]['std']  for p in top_params]
        y_pos = np.arange(len(top_params))
        ax2.barh(y_pos, means, xerr=stds, color='#1976D2', alpha=0.70, ecolor='#0D47A1',
                 capsize=3, edgecolor='white')
        ax2.set_yticks(y_pos); ax2.set_yticklabels(top_params, fontsize=9)
        ax2.set_title('Met Parameters\n(mean ± std from uploaded data)',
                      fontsize=10, fontweight='bold')
        ax2.set_xlabel('Value'); ax2.set_facecolor('#FAFFF8')
        ax2.grid(True, alpha=0.20, axis='x')
    else:
        ax2.text(0.5, 0.5, 'No met parameters', ha='center', va='center', transform=ax2.transAxes)

    # Data coverage & key stats
    ax3 = axes[2]
    ax3.axis('off')
    stats_text = (
        f"NDVI Data Summary\n"
        f"{'─'*30}\n"
        f"Observations:   {ndvi_info['n_obs']}\n"
        f"Years covered:  {ndvi_info['year_range']}\n"
        f"Cadence:        {ndvi_info['cadence_d']:.1f} days\n"
        f"Max gap thresh: {ndvi_info['max_gap_d']} days\n"
        f"NDVI mean:      {ndvi_info['ndvi_mean']:.3f}\n"
        f"NDVI std:       {ndvi_info['ndvi_std']:.3f}\n"
        f"Dynamic range:  {ndvi_info['data_range']:.3f}\n"
        f"Evergreen idx:  {ndvi_info['evergreen_index']:.3f}\n"
        f"  (P5/P95; 1.0=constant, 0=seasonal)\n"
        f"{'─'*30}\n"
        f"Met parameters: {len(met_info)}"
    )
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=9.5,
             va='top', ha='left', fontfamily='monospace',
             bbox=dict(facecolor='#E8F5E9', edgecolor='#A5D6A7', boxstyle='round,pad=0.8'))
    ax3.set_title('Data Characterization', fontsize=10, fontweight='bold')

    fig.suptitle('Uploaded Data — Automatic Characterization (No Hardcoded Assumptions)',
                 fontsize=12, fontweight='bold', color='#1B4332')
    fig.tight_layout()
    return fig


def plot_met_with_ndvi(met_df, ndvi_df, raw_params, pheno_df, interp_freq=5):
    ndvi_s = ndvi_df.set_index('Date')['NDVI'].sort_index()
    ndvi_s = ndvi_s[~ndvi_s.index.duplicated(keep='first')]
    full_r = pd.date_range(start=ndvi_s.index.min(), end=ndvi_s.index.max(), freq=f'{interp_freq}D')
    ndvi_5d = ndvi_s.reindex(ndvi_s.index.union(full_r)).interpolate(method='time').reindex(full_r)
    if pheno_df is None or len(pheno_df) == 0:
        return []

    # Auto-assign colors to whatever params exist
    ALL_COLS = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#546E7A',
                '#F9A825','#6A1B9A','#795548','#212121']
    param_colors = {p: ALL_COLS[i % len(ALL_COLS)] for i, p in enumerate(raw_params)}

    # Split params into air/soil/other based on names
    air_keys  = ['T2M', 'RH2M', 'PREC', 'ALLSKY']
    soil_keys = ['GWET', 'WS2M', 'MSI']
    air_params  = [(p, param_colors[p], p) for p in raw_params
                   if any(k in p.upper() for k in air_keys) and p in met_df.columns]
    soil_params = [(p, param_colors[p], p) for p in raw_params
                   if any(k in p.upper() for k in soil_keys) and p in met_df.columns]
    if not air_params and not soil_params:
        air_params = [(p, param_colors[p], p) for p in raw_params[:3] if p in met_df.columns]

    figs = []
    for _, row in pheno_df.sort_values('Year').iterrows():
        try:
            trough_d = row.get('Trough_Date', pd.NaT)
            sos_d    = row.get('SOS_Date',    pd.NaT)
            eos_d    = row.get('EOS_Date',    pd.NaT)
            s = (pd.Timestamp(trough_d) if pd.notna(trough_d) else
                 pd.Timestamp(sos_d) - pd.Timedelta(days=60) if pd.notna(sos_d) else None)
            if s is None or pd.isna(eos_d): continue
            e = pd.Timestamp(eos_d) + pd.Timedelta(days=30)
            df_met  = met_df[(met_df['Date'] >= s) & (met_df['Date'] <= e)].copy()
            if len(df_met) < 10: continue
            ndvi_seg = ndvi_5d.reindex(df_met['Date'].values, method='nearest',
                                       tolerance=pd.Timedelta('8D')).ffill().bfill()
            yr      = int(row['Year'])
            sos_str = pd.Timestamp(sos_d).strftime('%d %b') if pd.notna(sos_d) else '?'
            eos_str = pd.Timestamp(eos_d).strftime('%d %b %Y') if pd.notna(eos_d) else '?'

            n_panels = 2 if (air_params and soil_params) else 1
            fig, axes_p = plt.subplots(n_panels, 1, figsize=(16, 5.5 * n_panels), sharex=True)
            if n_panels == 1:
                axes_p = [axes_p]
            fig.patch.set_facecolor('#FAFFF8')
            fig.suptitle(f"NDVI + Met — Season {yr}  [ {sos_str} → {eos_str} ]",
                         fontsize=14, fontweight='bold', y=0.99)

            def _draw_panel(ax, param_list, title,
                            bar_keys=('PRECTOTCORR', 'PRECTOT', 'RAIN')):
                ax.fill_between(df_met['Date'], ndvi_seg, alpha=0.18, color='#2E7D32')
                ax.plot(df_met['Date'], ndvi_seg, color='#2E7D32', lw=2.5, label='NDVI')
                ax.set_ylabel('NDVI', color='#2E7D32', fontsize=11, fontweight='bold')
                ax.set_ylim(0, 1.05); ax.tick_params(axis='y', labelcolor='#2E7D32')
                ax.grid(True, linestyle='--', alpha=0.28); ax.set_facecolor('#FAFFF8')
                ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
                twin_axes = []
                present = [(p, c, l) for p, c, l in param_list if p in df_met.columns][:4]
                for i, (var, col, lab) in enumerate(present):
                    axr = ax.twinx()
                    if i > 0:
                        axr.spines['right'].set_position(('axes', 1.0 + 0.09 * i))
                        axr.spines['right'].set_visible(True)
                    if var in bar_keys:
                        axr.bar(df_met['Date'], df_met[var], color=col, alpha=0.50, width=1.0, label=lab)
                    else:
                        axr.plot(df_met['Date'], df_met[var], color=col, lw=1.6, alpha=0.85, label=lab)
                    axr.set_ylabel(lab, color=col, fontsize=8, rotation=270, labelpad=18)
                    axr.tick_params(axis='y', labelcolor=col, labelsize=8)
                    axr.spines['right'].set_color(col)
                    twin_axes.append(axr)
                return ax, twin_axes

            if n_panels == 2:
                _draw_panel(axes_p[0], air_params, "Air Parameters (auto-detected)")
                _draw_panel(axes_p[1], soil_params, "Soil/Wind Parameters (auto-detected)")
            else:
                _draw_panel(axes_p[0], (air_params or soil_params), "Met Parameters (auto-detected)")

            axes_p[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
            axes_p[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes_p[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(axes_p[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            fig.tight_layout(rect=[0.10, 0.0, 1.0, 0.95])
            figs.append((str(yr), fig))
        except Exception:
            continue
    return figs


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    # ── APP HEADER ────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>🌿 Forest Phenology Analyser</h1>
        <p>Upload your NDVI and meteorological data to automatically extract seasonal events,
        train predictive models, and forecast future phenology for any Indian forest type.</p>
        <span class="badge">🌱 SOS — Start of Season</span>
        <span class="badge">🌿 POS — Peak of Season</span>
        <span class="badge">🍂 EOS — End of Season</span>
        <span class="badge">📏 LOS — Length of Season</span>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────
    st.sidebar.markdown("## 📂 Upload Data")
    ndvi_file = st.sidebar.file_uploader(
        "NDVI File (CSV)",  type=['csv'], key="ndvi_uploader",
        help="A CSV file with a date column and an NDVI column. Any date format is accepted.")
    met_file = st.sidebar.file_uploader(
        "Meteorological File (CSV)", type=['csv'], key="met_uploader",
        help="Daily meteorological data CSV — from NASA POWER or your own source.")

    _fp_ndvi = f"{ndvi_file.name}:{ndvi_file.size}" if ndvi_file else ""
    _fp_met  = f"{met_file.name}:{met_file.size}"   if met_file  else ""

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📅 Growing Season Window")
    st.sidebar.caption("Set the calendar period that defines one growing season for your study site.")
    sm_names  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_opts = list(sm_names.keys())
    col_sm, col_em = st.sidebar.columns(2)
    start_m = col_sm.selectbox("Start month", options=month_opts, index=5,
                               format_func=lambda m: sm_names[m])
    end_m   = col_em.selectbox("End month",   options=month_opts, index=4,
                               format_func=lambda m: sm_names[m])
    if start_m != end_m:
        if start_m > end_m:
            st.sidebar.info(f"Cross-year window: **{sm_names[start_m]} → {sm_names[end_m]}**")
        else:
            st.sidebar.info(f"Within-year window: **{sm_names[start_m]} → {sm_names[end_m]}**")
    if start_m == end_m:
        st.sidebar.warning("⚠️ Start and end month are the same — no seasons will be detected.")

    min_days = st.sidebar.slider(
        "Minimum season length (days)", 30, 300, 100, 10,
        help="Seasons shorter than this value are ignored. Increase if short noise cycles appear.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Detection Sensitivity")
    st.sidebar.caption(
        "These thresholds define when green-up starts (SOS) and ends (EOS), "
        "expressed as a percentage of each season's NDVI amplitude.")
    sos_thr = st.sidebar.slider("SOS threshold  (% of amplitude)", 5, 40, 10, 5) / 100.0
    eos_thr = st.sidebar.slider("EOS threshold  (% of amplitude)", 5, 40, 10, 5) / 100.0
    st.sidebar.caption(
        f"Current: SOS at **{int(sos_thr*100)}%** · EOS at **{int(eos_thr*100)}%** "
        "of each season's NDVI swing. Higher % = stricter (later SOS, earlier EOS).")
    st.sidebar.caption(
        "ℹ️ Changing the SOS threshold can shift when each season 'starts', "
        "which changes the climate window used to predict all events including EOS. "
        "This is expected — SOS and EOS share the same season baseline.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📈 Prediction Model")
    st.sidebar.caption("Model used to relate meteorological conditions to phenological event dates.")
    model_opts = {
        "Ridge Regression":             "ridge",
        "LOESS Smoothing":              "loess",
        "Polynomial Regression (Deg 2)":"poly2",
        "Polynomial Regression (Deg 3)":"poly3",
        "Gaussian Process":             "gpr",
    }
    model_sel = st.sidebar.radio("Model type", list(model_opts.keys()), index=0)
    model_key = model_opts[model_sel]

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🗓️ Climate Window")
    feat_window = st.sidebar.slider(
        "Days before event to average climate", 7, 60, 15, 1,
        help="How many days of meteorological data before each event date are averaged to form predictors.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔢 Maximum Features in Model")
    max_features_override = st.sidebar.slider(
        "Max climate variables per model", 1, 4, 1, 1,
        key="max_feat_slider",
        help=(
            "Maximum number of climate variables allowed in each model. "
            "With few seasons (< 5), using more than 1–2 features causes overfitting — "
            "the model memorises the training data but fails to predict new seasons. "
            "Increase only if you have ≥ 6 seasons and want to test multi-variable models."
        ))
    if max_features_override >= 2:
        st.sidebar.markdown(
            '<div style="background:#FFF8E1;padding:8px 12px;border-radius:8px;'
            'border-left:3px solid #F9A825;font-size:0.80rem;margin-top:4px">'
            '⚠️ Using 2+ features with fewer than 6 seasons can overfit. '
            'LOO R² may appear high but predictions may be unreliable.</div>',
            unsafe_allow_html=True)

    cfg = {"start_month": start_m, "end_month": end_m, "min_days": min_days}

    _fp = (f"{_fp_ndvi}|{_fp_met}|sm={start_m}|em={end_m}|md={min_days}"
           f"|sos={sos_thr:.3f}|eos={eos_thr:.3f}|model={model_key}|win={feat_window}"
           f"|mxf={max_features_override}")
    if st.session_state.get('_fp') != _fp:
        for k in ['predictor','pheno_df','met_df','train_df','all_params',
                  'raw_params','ndvi_df','ndvi_info','met_info','interp_freq']:
            st.session_state[k] = None
        st.session_state['_fp'] = _fp

    # ── WELCOME SCREEN ────────────────────────────────────────
    if not (ndvi_file and met_file):
        st.markdown("""
<div class="upload-panel">
<h3>👈 Upload your two data files using the sidebar to begin</h3>

<b>File 1 — NDVI CSV</b><br>
Any CSV with a date column and an NDVI column. Column names are detected automatically.
<br><br>
<code>Date, NDVI</code><br>
<code>2016-01-01, 0.42</code><br>
<code>2016-01-17, 0.45</code>
<br><br>

<b>File 2 — Meteorological CSV</b><br>
Daily climate data for your study site. Download free from
<a href="https://power.larc.nasa.gov/data-access-viewer/" target="_blank">NASA POWER</a>
(Daily → Point → your coordinates → CSV), or use your own file.
Parameters such as temperature, rainfall, humidity, and radiation are detected automatically.
<br><br>

<b>What this tool does:</b><br>
• Detects your forest's seasonal green-up and senescence from the NDVI curve<br>
• Identifies the <b>Start (SOS)</b>, <b>Peak (POS)</b>, and <b>End (EOS)</b> of each growing season<br>
• Finds which climate variables best explain year-to-year variation in those dates<br>
• Trains a model so you can predict future phenology dates from forecast weather<br>
</div>
        """, unsafe_allow_html=True)
        return

    # ── PARSE FILES ───────────────────────────────────────────
    with st.spinner("Reading files…"):
        ndvi_result, ndvi_err = parse_ndvi(ndvi_file)
        if ndvi_result is None:
            st.error(f"Could not read NDVI file: {ndvi_err}"); return

        if isinstance(ndvi_result, tuple) and ndvi_result[0] == 'MULTI_SITE':
            _, site_list, raw_df, date_col, ndvi_col = ndvi_result
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🗺️ Multiple Sites Detected")
            chosen_site = st.sidebar.selectbox("Select site to analyse", site_list, key="site_sel")
            ndvi_df = _filter_ndvi_site(raw_df, date_col, ndvi_col, chosen_site)
            if len(ndvi_df) == 0:
                st.error(f"No valid rows for site '{chosen_site}'."); return
            st.sidebar.success(f"✅ Site: **{chosen_site}** — {len(ndvi_df)} observations")
        else:
            ndvi_df = ndvi_result

        met_file.seek(0)
        met_df, raw_params, met_err = parse_nasa_power(met_file)
        if met_df is None:
            st.error(f"Could not read met file: {met_err}"); return

    # ── DERIVED FEATURES ──────────────────────────────────────
    met_df     = add_derived_features(met_df, season_start_month=start_m)
    all_params = [c for c in met_df.columns
                  if c not in {'Date','YEAR','MO','DY','DOY','LON','LAT','ELEV'}
                  and pd.api.types.is_numeric_dtype(met_df[c])]
    derived    = [p for p in all_params if p not in raw_params]

    # ── SIDEBAR SUCCESS ───────────────────────────────────────
    _, _, interp_freq = detect_ndvi_cadence(ndvi_df)
    ndvi_info  = characterize_ndvi_data(ndvi_df)
    met_info   = characterize_met_data(met_df, raw_params)

    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ NDVI loaded — {ndvi_info['n_obs']} observations · {ndvi_info['n_years']} years")
    st.sidebar.success(f"✅ Met loaded — {len(raw_params)} climate parameters")
    if derived:
        st.sidebar.info(f"+ {len(derived)} derived features computed automatically")

    # ── TABS ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Summary",
        "🔬 Season Extraction & Models",
        "📈 Climate Correlations",
        "🔮 Predict",
        "📖 User Guide"])

    icons = {'SOS': '🌱', 'POS': '🌿', 'EOS': '🍂'}

    # ══════════════════════════════════════════════════════════
    # TAB 1 — DATA SUMMARY
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<p class="section-title">Your Uploaded Data</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f'<div class="metric-card"><div class="label">📅 Year Range</div>'
            f'<div class="value">{ndvi_info["year_range"]}</div>'
            f'<div class="sub">{ndvi_info["n_obs"]} NDVI observations</div></div>',
            unsafe_allow_html=True)
        c2.markdown(
            f'<div class="metric-card"><div class="label">🌿 Mean NDVI</div>'
            f'<div class="value">{ndvi_info["ndvi_mean"]}</div>'
            f'<div class="sub">std = {ndvi_info["ndvi_std"]}</div></div>',
            unsafe_allow_html=True)
        c3.markdown(
            f'<div class="metric-card"><div class="label">📏 NDVI Range</div>'
            f'<div class="value">{ndvi_info["data_range"]}</div>'
            f'<div class="sub">P5 = {ndvi_info["ndvi_p5"]}  ·  P95 = {ndvi_info["ndvi_p95"]}</div></div>',
            unsafe_allow_html=True)
        c4.markdown(
            f'<div class="metric-card"><div class="label">🔁 Observation Cadence</div>'
            f'<div class="value">{ndvi_info["cadence_d"]:.0f} days</div>'
            f'<div class="sub">Auto-detected from file</div></div>',
            unsafe_allow_html=True)

        st.markdown("")
        fig_ds = plot_data_summary(ndvi_info, met_info)
        st.pyplot(fig_ds, use_container_width=True)

        st.markdown('<p class="section-title">Climate Parameters Available</p>', unsafe_allow_html=True)
        if met_info:
            met_summary_df = pd.DataFrame([
                {'Parameter': p, 'Mean': round(v['mean'], 3), 'Std Dev': round(v['std'], 3),
                 'Min': round(v['min'], 3), 'Max': round(v['max'], 3)}
                for p, v in met_info.items()
            ])
            st.dataframe(
                met_summary_df.style.background_gradient(subset=['Mean'], cmap='Greens'),
                use_container_width=True, hide_index=True)
        if derived:
            st.markdown(
                f'<div class="banner-info">ℹ️ In addition to the {len(raw_params)} parameters in your file, '
                f'<b>{len(derived)} derived variables</b> were computed automatically '
                f'(e.g. Growing Degree Days, log-rainfall, seasonal cumulative values). '
                f'These are included in the model feature pool.</div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — SEASON EXTRACTION & MODELS
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<p class="section-title">Extracting Seasonal Events from NDVI</p>',
                    unsafe_allow_html=True)

        with st.spinner("Analysing NDVI time series…"):
            pheno_df, pheno_err = extract_phenology(ndvi_df, cfg, sos_thr, eos_thr)

        if pheno_df is None:
            st.error(f"Could not detect seasons: {pheno_err}"); return
        n_seasons = len(pheno_df)

        # Dataset size guidance
        if n_seasons == 0:
            st.error("No complete seasons found. Try reducing the minimum season length, "
                     "or adjust the growing season window months."); return
        elif n_seasons == 1:
            st.warning("Only **1 season** was extracted. The NDVI chart is shown below, "
                       "but model training requires at least 2 seasons. "
                       "Add more years of data to enable predictions.")
        elif n_seasons == 2:
            st.warning("**2 seasons** extracted — model results are exploratory only. "
                       "With just 2 data points, statistical reliability is limited. "
                       "We recommend collecting at least 5 years of data.")
        elif n_seasons <= 4:
            st.info(f"**{n_seasons} seasons** extracted — small dataset. "
                    f"Results are indicative. At least 5 seasons are recommended for reliable predictions.")
        elif n_seasons < 7:
            st.info(f"**{n_seasons} seasons** extracted — usable dataset. "
                    "Predictions are available. More years will further improve accuracy.")
        else:
            st.success(f"✅ **{n_seasons} growing seasons** extracted — good dataset for modelling.")

        # Phenology chart + table
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.pyplot(plot_ndvi_phenology(ndvi_df, pheno_df,
                                          season_window=(start_m, end_m),
                                          interp_freq=interp_freq))
        with c_right:
            st.markdown("**Extracted season dates**")
            disp = []
            for _, row in pheno_df.iterrows():
                sd = row.get('SOS_Date'); pd_ = row.get('POS_Date'); ed = row.get('EOS_Date')
                disp.append({
                    'Year':       int(row['Year']),
                    'SOS':        pd.Timestamp(sd).strftime('%b %d') if pd.notna(sd) else '—',
                    'DOY':        int(row.get('SOS_DOY', 0)),
                    'POS':        pd.Timestamp(pd_).strftime('%b %d') if pd.notna(pd_) else '—',
                    'DOY.1':      int(row.get('POS_DOY', 0)),
                    'EOS':        pd.Timestamp(ed).strftime('%b %d %Y') if pd.notna(ed) else '—',
                    'LOS (days)': int(row.get('LOS_Days', 0)),
                    'Peak NDVI':  round(float(row.get('Peak_NDVI', 0)), 3),
                })
            st.dataframe(pd.DataFrame(disp), use_container_width=True, height=300)

        fig_t = plot_pheno_trends(pheno_df)
        st.pyplot(fig_t)

        # ── MODEL TRAINING ────────────────────────────────────
        st.markdown('<p class="section-title">Predictive Model Training</p>', unsafe_allow_html=True)

        with st.spinner(f"Training {model_sel} models…"):
            train_df  = make_training_features(pheno_df, met_df, all_params, window=feat_window)
            predictor = UniversalPredictor()
            predictor.train(train_df, all_params, model_key=model_key,
                            user_max_features=max_features_override)

        st.session_state.update({
            'pheno_df': pheno_df, 'met_df': met_df, 'train_df': train_df,
            'predictor': predictor, 'all_params': all_params,
            'raw_params': raw_params, 'ndvi_df': ndvi_df,
            'ndvi_info': ndvi_info, 'met_info': met_info,
            'interp_freq': interp_freq,
        })

        st.markdown(
            f"The table below shows how well each model predicts the event date "
            f"(validated by leaving one season out at a time — **Leave-One-Out cross-validation**).")

        c1, c2, c3 = st.columns(3)

        def _card(col, ev):
            ev_full = {'SOS':'Start of Season','POS':'Peak of Season','EOS':'End of Season'}
            n_ev = predictor.n_seasons.get(ev, 0)
            if ev not in predictor._fits:
                msg = ("Need ≥ 2 seasons to train a model." if n_ev < 2
                       else "No climate variable was correlated strongly enough.")
                col.markdown(
                    f'<div class="metric-card"><div class="label">{icons[ev]} {ev} — {ev_full[ev]}</div>'
                    f'<div class="value" style="color:#9E9E9E;font-size:1.3rem">Not fitted</div>'
                    f'<div class="sub">{msg}<br>{n_ev} season(s) available</div></div>',
                    unsafe_allow_html=True); return
            fit  = predictor._fits[ev]
            r2   = predictor.r2.get(ev, 0)
            mae  = predictor.mae.get(ev, 0)
            n    = predictor.n_seasons.get(ev, 0)
            if fit['mode'] == 'mean':
                col.markdown(
                    f'<div class="metric-card"><div class="label">{icons[ev]} {ev} — {ev_full[ev]}</div>'
                    f'<div class="value" style="color:#9E9E9E">Mean only</div>'
                    f'<div class="sub">No climate variable met the correlation threshold.<br>'
                    f'Prediction = historical average ≈ DOY {fit["mean_doy"]:.0f}<br>'
                    f'Typical error: ±{mae:.1f} days</div></div>', unsafe_allow_html=True)
            else:
                clr = '#1B5E20' if r2 > 0.6 else '#E65100' if r2 > 0.3 else '#B71C1C'
                feats = fit.get('features', [])
                mtag  = {'ridge':'Ridge','loess':'LOESS','poly2':'Poly-2',
                         'poly3':'Poly-3','gpr':'Gaussian Process'}.get(fit.get('model_key','ridge'), '—')
                col.markdown(
                    f'<div class="metric-card"><div class="label">{icons[ev]} {ev} — {ev_full[ev]}</div>'
                    f'<div class="value" style="color:{clr}">{r2*100:.1f}%</div>'
                    f'<div class="sub">Accuracy (R²) · {mtag} · {n} seasons<br>'
                    f'Driver(s): <b>{", ".join(feats) or "—"}</b><br>'
                    f'Typical error: ±{mae:.1f} days</div></div>',
                    unsafe_allow_html=True)

        _card(c1, 'SOS'); _card(c2, 'POS'); _card(c3, 'EOS')

        if n_seasons < 4:
            st.markdown(
                '<div class="banner-warn">⚠️ With fewer than 4 seasons, accuracy scores (R²) '
                'should be treated as indicative only. Collect more years of data for reliable predictions.</div>',
                unsafe_allow_html=True)
        elif n_seasons < 7:
            st.markdown(
                f'<div class="banner-info">ℹ️ With {n_seasons} seasons, an R² of 0.3–0.6 is typical. '
                f'More years will improve model reliability.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="banner-good">✅ {n_seasons} seasons — sufficient data for reliable modelling.</div>',
                unsafe_allow_html=True)

        # Fitted equations
        st.markdown('<p class="section-title">Model Equations</p>', unsafe_allow_html=True)

        # Explain why only 1 feature when n is small
        if n_seasons <= 3 and max_features_override == 1:
            st.markdown(
                f'<div class="banner-info">ℹ️ <b>Why only 1 climate variable?</b> — '
                f'With {n_seasons} seasons, adding more variables causes the model to memorise '
                f'the training data perfectly but fail on new seasons (overfitting). '
                f'To try 2 variables, increase <b>Maximum Features in Model</b> in the sidebar — '
                f'but treat the results with caution.</div>',
                unsafe_allow_html=True)
        elif n_seasons <= 5 and max_features_override == 1:
            st.markdown(
                f'<div class="banner-info">ℹ️ With {n_seasons} seasons, the model uses 1 variable by default '
                f'to avoid overfitting. You can try 2 variables using the sidebar slider.</div>',
                unsafe_allow_html=True)

        st.caption(
            "Each equation shows the relationship between the selected climate variable(s) "
            "and the event date, expressed as days from the start of the growing season.")
        t_sos, t_pos, t_eos = st.tabs(
            [f"{icons['SOS']} Start of Season (SOS)",
             f"{icons['POS']} Peak of Season (POS)",
             f"{icons['EOS']} End of Season (EOS)"])
        for ui_tab, ev in zip([t_sos, t_pos, t_eos], ['SOS', 'POS', 'EOS']):
            with ui_tab:
                eq   = predictor.equation_str(ev, season_start_month=start_m)
                eq_h = eq.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
                st.markdown(f'<div class="eq-box">{eq_h}</div>', unsafe_allow_html=True)
                ct   = predictor.corr_table_for_display(ev)
                if not ct.empty:
                    def _sr(val):
                        if val.startswith('✅'): return 'background-color:#C8E6C9;color:#1B5E20;font-weight:600'
                        if val.startswith('➖') and 'Redundant' in val: return 'color:#9E9E9E;font-style:italic'
                        if val.startswith('➖'): return 'color:#555'
                        return 'color:#bbb'
                    fmt  = {'Pearson r': '{:+.3f}', 'Spearman ρ': '{:+.3f}', 'Composite': '{:.3f}'}
                    sty  = ct.style
                    if 'Pearson r'  in ct.columns: sty = sty.background_gradient(subset=['Pearson r'],  cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Spearman ρ' in ct.columns: sty = sty.background_gradient(subset=['Spearman ρ'], cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Composite'  in ct.columns: sty = sty.background_gradient(subset=['Composite'],  cmap='Greens', vmin=0,  vmax=1)
                    sty = sty.applymap(_sr, subset=['Role']).format(fmt).set_properties(**{'font-size': '0.84rem'})
                    st.dataframe(sty, use_container_width=True, hide_index=True)

        fig_s = plot_obs_vs_pred(predictor, train_df)
        if fig_s:
            st.markdown('<p class="section-title">Observed vs Predicted</p>', unsafe_allow_html=True)
            st.caption("Each point is one historical season. The closer points are to the diagonal line, "
                       "the more accurate the model.")
            st.pyplot(fig_s)

        # Downloads
        st.markdown("---")
        dl_cols = [c for c in ['Year','SOS_DOY','POS_DOY','EOS_DOY','LOS_Days',
                                'Peak_NDVI','Amplitude','Base_NDVI','SOS_Date',
                                'POS_Date','EOS_Date'] if c in pheno_df.columns]
        coef_df = predictor.export_coefficients(season_start_month=start_m)
        col_d1, col_d2 = st.columns(2)
        col_d1.download_button("📥 Download Phenology Table (CSV)",
                               pheno_df[dl_cols].to_csv(index=False),
                               "phenology_table.csv", "text/csv")
        col_d2.download_button("📥 Download Model Coefficients (CSV)",
                               coef_df.to_csv(index=False),
                               "model_coefficients.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # TAB 3 — CLIMATE CORRELATIONS
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<p class="section-title">Which Climate Variables Drive Each Seasonal Event?</p>',
                    unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        pheno_df_ss  = st.session_state.get('pheno_df')
        if predictor_ss is None:
            st.info("Complete the Season Extraction step first (🔬 Season Extraction & Models tab)."); return

        fig_c = plot_correlation_summary(predictor_ss)
        if fig_c:
            st.pyplot(fig_c, use_container_width=True)

        st.markdown('<p class="section-title">Full Correlation Table</p>', unsafe_allow_html=True)
        st.caption(
            "Pearson r and Spearman ρ measure the linear and rank correlation between each "
            "climate variable and the event date. Values range from −1 to +1. "
            "A positive value means higher climate values → later event. "
            "Variables marked ✅ were used in the model.")
        c1, c2, c3 = st.columns(3)
        for col_st, ev in zip([c1, c2, c3], ['SOS', 'POS', 'EOS']):
            with col_st:
                ev_full = {'SOS':'Start of Season','POS':'Peak','EOS':'End of Season'}
                st.markdown(f"**{icons[ev]} {ev_full[ev]}**")
                ct = predictor_ss.corr_tables.get(ev)
                if ct is not None and len(ct):
                    disp = ct[['Feature','Pearson_r',
                               'Spearman_rho' if 'Spearman_rho' in ct.columns else '|r|',
                               'Composite']].copy()
                    disp = disp.rename(columns={'Pearson_r':'Pearson r','Spearman_rho':'Spearman ρ'})
                    sty  = disp.style.background_gradient(subset=['Pearson r'], cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Spearman ρ' in disp.columns:
                        sty = sty.background_gradient(subset=['Spearman ρ'], cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Composite' in disp.columns:
                        sty = sty.background_gradient(subset=['Composite'], cmap='Greens', vmin=0, vmax=1)
                    sty = sty.format({'Pearson r':'{:+.3f}','Spearman ρ':'{:+.3f}','Composite':'{:.3f}'}
                                     if 'Spearman ρ' in disp.columns else
                                     {'Pearson r':'{:+.3f}','Composite':'{:.3f}'})
                    st.dataframe(sty, use_container_width=True, hide_index=True, height=320)
                else:
                    st.info("No correlation data available.")

        st.markdown('<p class="section-title">NDVI and Climate — Year by Year</p>',
                    unsafe_allow_html=True)
        st.caption("Each chart shows NDVI (green fill) alongside your climate variables "
                   "for one growing season.")
        _met  = st.session_state.get('met_df')
        _ndvi = st.session_state.get('ndvi_df')
        _rp   = st.session_state.get('raw_params', [])
        _if   = st.session_state.get('interp_freq', 5)
        if _met is not None and _ndvi is not None:
            figs_l = plot_met_with_ndvi(_met, _ndvi, _rp, pheno_df_ss, interp_freq=_if)
            if figs_l:
                for s_lbl, f_m in figs_l:
                    st.markdown(f"**Season {s_lbl}**")
                    st.pyplot(f_m, use_container_width=True); plt.close(f_m)
            else:
                st.info("No complete seasons with overlapping climate data found.")

    # ══════════════════════════════════════════════════════════
    # TAB 4 — PREDICT
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<p class="section-title">Predict Phenology Dates for Any Year</p>',
                    unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        train_df_ss  = st.session_state.get('train_df')
        pheno_ss     = st.session_state.get('pheno_df')
        if predictor_ss is None:
            st.info("Complete the Season Extraction step first (🔬 Season Extraction & Models tab)."); return

        st.markdown(
            '<div class="banner-info">Enter the expected climate conditions for your target year. '
            'Values are pre-filled with historical averages from your training data as a starting point. '
            'Adjust them to match forecast or scenario conditions.</div>',
            unsafe_allow_html=True)

        mo = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
              7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        ev_colors_hex  = {'SOS': '#E8F5E9', 'POS': '#E3F2FD', 'EOS': '#FFF3E0'}
        ev_border_hex  = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#E65100'}
        ev_labels_full = {'SOS': '🌱 Start of Season (SOS)',
                          'POS': '🌿 Peak of Season (POS)',
                          'EOS': '🍂 End of Season (EOS)'}
        ev_inputs = {ev: {} for ev in ['SOS', 'POS', 'EOS']}

        any_model = False
        for ev in ['SOS', 'POS', 'EOS']:
            fit = predictor_ss._fits.get(ev, {})
            if not fit or fit.get('mode') not in ('ridge','loess','poly2','poly3','gpr'):
                continue
            feats = fit.get('features', [])
            if not feats:
                continue
            any_model = True
            r2  = predictor_ss.r2.get(ev, 0)
            mae = predictor_ss.mae.get(ev, 0)

            hist_hint = ""
            if pheno_ss is not None and f'{ev}_Date' in pheno_ss.columns:
                ev_dates = pheno_ss[f'{ev}_Date'].dropna()
                if len(ev_dates) > 0:
                    med_m = int(ev_dates.dt.month.median())
                    med_d = int(ev_dates.dt.day.median())
                    hist_hint = f" · Historically around {mo[med_m]} {med_d} (±{mae:.0f} days)"

            st.markdown(
                f"<div style='background:{ev_colors_hex[ev]};padding:14px 18px;border-radius:10px;"
                f"border-left:4px solid {ev_border_hex[ev]};margin:10px 0'>"
                f"<b>{ev_labels_full[ev]}</b>"
                f"<span style='font-size:0.82rem;color:#666'>&nbsp;&nbsp;"
                f"Model accuracy (R²) = {r2:.0%}{hist_hint}</span></div>",
                unsafe_allow_html=True)

            col_list = st.columns(min(len(feats), 4))
            for idx, f in enumerate(feats):
                default = 0.0
                if train_df_ss is not None and f in train_df_ss.columns:
                    ev_sub = train_df_ss[train_df_ss['Event'] == ev]
                    vals   = ev_sub[f].dropna() if len(ev_sub) > 0 else train_df_ss[f].dropna()
                    if len(vals):
                        default = float(vals.mean())
                is_sum = any(k in f.upper() for k in ACCUM_KEYWORDS)
                vmin = vmax = None
                if train_df_ss is not None and f in train_df_ss.columns:
                    col_vals = train_df_ss[f].dropna()
                    if len(col_vals) >= 2:
                        vmin = float(col_vals.min())
                        vmax = float(col_vals.max())
                with col_list[idx % len(col_list)]:
                    ev_inputs[ev][f] = st.number_input(
                        f"{f}  [{ev}]", value=round(default, 3), format="%.3f",
                        key=f"inp_{ev}_{f}",
                        help=(f"{'Total (sum)' if is_sum else f'{feat_window}-day average'} of {f} "
                              f"before the expected {ev} date.\n"
                              f"Historical mean: {default:.3f}"
                              + (f" | range in data: [{vmin:.2f} – {vmax:.2f}]" if vmin is not None else "")))

        if not any_model:
            st.markdown(
                '<div class="banner-warn">⚠️ No predictive models could be fitted — '
                'not enough seasons or no climate variable showed sufficient correlation. '
                'Upload more years of data to enable predictions.</div>',
                unsafe_allow_html=True)
            return

        st.markdown("---")
        pred_year = st.number_input("Year to predict for", 2020, 2050, 2026)

        if st.button("▶  Run Prediction", type="primary"):
            results = {}
            for ev in ['SOS', 'POS', 'EOS']:
                res = predictor_ss.predict(ev_inputs.get(ev, {}), ev, pred_year,
                                           season_start_month=start_m)
                if res:
                    results[ev] = res

            if results:
                order_warns = []
                if 'SOS' in results and 'POS' in results:
                    if results['POS']['rel_days'] <= results['SOS']['rel_days']:
                        fb = (int(round(pheno_ss['POS_Target'].mean()))
                              if pheno_ss is not None and 'POS_Target' in pheno_ss.columns
                              else results['SOS']['rel_days'] + 90)
                        corrected = max(fb, results['SOS']['rel_days'] + 14)
                        nd = datetime(pred_year, start_m, 1) + timedelta(days=corrected)
                        results['POS'].update({'rel_days': corrected,
                                               'doy': nd.timetuple().tm_yday, 'date': nd})
                        order_warns.append(f"Peak was predicted before start — adjusted to ~{nd.strftime('%b %d')}")
                if 'POS' in results and 'EOS' in results:
                    if results['EOS']['rel_days'] <= results['POS']['rel_days']:
                        fb = (int(round(pheno_ss['EOS_Target'].mean()))
                              if pheno_ss is not None and 'EOS_Target' in pheno_ss.columns
                              else results['POS']['rel_days'] + 90)
                        corrected = max(fb, results['POS']['rel_days'] + 14)
                        nd = datetime(pred_year, start_m, 1) + timedelta(days=corrected)
                        results['EOS'].update({'rel_days': corrected,
                                               'doy': nd.timetuple().tm_yday, 'date': nd})
                        order_warns.append(f"End was predicted before peak — adjusted to ~{nd.strftime('%b %d')}")
                if order_warns:
                    st.markdown(
                        '<div class="banner-warn"><b>Note:</b> ' +
                        ' · '.join(order_warns) + '</div>', unsafe_allow_html=True)

                cols = st.columns(len(results))
                for col, (ev, res) in zip(cols, results.items()):
                    ev_full = {'SOS':'Start of Season','POS':'Peak of Season','EOS':'End of Season'}
                    col.markdown(
                        f'<div class="metric-card"><div class="label">{icons[ev]} {ev_full[ev]}</div>'
                        f'<div class="value">{res["date"].strftime("%b %d")}</div>'
                        f'<div class="sub">Day {res["doy"]} of {res["date"].year}<br>'
                        f'Model accuracy (R²) = {res["r2"]:.0%}<br>'
                        f'Typical error: ±{res["mae"]:.0f} days</div></div>',
                        unsafe_allow_html=True)

                if 'SOS' in results and 'EOS' in results:
                    sd = results['SOS']['date']; ed = results['EOS']['date']
                    los = (ed - sd).days if ed >= sd else (ed - sd).days + 365
                    st.markdown("---")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("📏 Season Length", f"{los} days")
                    if 'POS' in results:
                        mc2.metric("Green-up phase (SOS → POS)",
                                   f"{(results['POS']['date']-results['SOS']['date']).days} days")
                        mc3.metric("Senescence phase (POS → EOS)",
                                   f"{(results['EOS']['date']-results['POS']['date']).days} days")

                out = pd.DataFrame({
                    'Event':         list(results.keys()),
                    'Predicted Date':[r['date'].strftime('%Y-%m-%d') for r in results.values()],
                    'Day of Year':   [r['doy'] for r in results.values()],
                    'R² (accuracy)': [round(r['r2'], 3) for r in results.values()],
                    'Typical error (days)': [round(r['mae'], 1) for r in results.values()],
                })
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Predictions (CSV)", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # TAB 5 — USER GUIDE
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<p class="section-title">User Guide</p>', unsafe_allow_html=True)

        st.markdown("""
This tool analyses the timing of seasonal changes in forest vegetation using satellite NDVI data
and daily climate records. It is designed for researchers and ecologists studying **forest phenology**
across any region of India — no configuration for forest type is required.

---

### 📘 Key Terms

**NDVI — Normalised Difference Vegetation Index**
A satellite-derived measure of vegetation greenness, ranging from 0 (bare soil) to 1 (dense canopy).
The seasonal rise and fall of NDVI is used to track the timing of green-up and senescence.

**SOS — Start of Season** 🌱
The date when the forest begins its seasonal green-up. Defined as the date when NDVI first rises above
a threshold percentage of its annual swing (amplitude).

**POS — Peak of Season** 🌿
The date of maximum greenness in the growing season. Corresponds to the highest NDVI value between SOS and EOS.

**EOS — End of Season** 🍂
The date when NDVI falls back below the same threshold on the descending limb. Marks the end of the active growing period.

**LOS — Length of Season** 📏
The number of days between SOS and EOS. A longer LOS means an extended period of active canopy cover.

**DOY — Day of Year**
The number of the day within a calendar year (1 = 1 January, 365 = 31 December). Used to express event dates numerically for analysis and plotting.

**Amplitude**
The difference between the peak NDVI and the baseline NDVI for a given season. Larger amplitude indicates a stronger seasonal signal — typical of deciduous forests.

---

### ⚙️ Controls Explained

| Control | What it does |
|---|---|
| **Growing Season Window** | Sets the calendar period searched for each season. Set Start to the month of lowest NDVI (trough) and End to one month before that the following year. |
| **Minimum Season Length** | Seasons shorter than this are ignored. Increase if the NDVI chart shows spurious short cycles. Decrease if true seasons are being missed. |
| **SOS / EOS Threshold** | The percentage of each season's NDVI amplitude at which green-up is considered to have started or ended. 10% is a common default. Lower values give earlier SOS / later EOS. |
| **Climate Window** | How many days before each event date to average the climate data when building the predictive model. |
| **Maximum Features in Model** | Maximum number of climate variables the model can use. Default is 1 for small datasets (< 5 seasons) to prevent overfitting. Increase to 2–3 only when you have ≥ 6 seasons. |
| **Prediction Model** | The statistical method used to link climate variables to event dates. Ridge Regression is recommended for small datasets. |

---

### 📊 Understanding Model Accuracy (R²)

R² measures what fraction of the year-to-year variation in an event date is explained by climate.

| R² | Interpretation |
|---|---|
| > 0.80 | Strong — climate is a good predictor of this event |
| 0.50 – 0.80 | Good |
| 0.30 – 0.50 | Moderate — some predictive signal present |
| < 0.30 | Weak — more years of data or different predictors may help |

The error figure shown (±X days) is the typical difference between the model's prediction and the actual
historical date. This is evaluated using **Leave-One-Out cross-validation**: each season is predicted
using all other seasons, giving an honest estimate of out-of-sample accuracy.

---

### 📂 Data Format Requirements

**NDVI file:**
- Any CSV with a date column and an NDVI column
- Column names are detected automatically (e.g. `date`, `Date`, `DATE`, `ndvi`, `NDVI`)
- Dates can be in any standard format: `YYYY-MM-DD`, `DD-MM-YY`, `MM/DD/YYYY`, etc.
- Typical sources: MODIS MOD13A2 / MYD13A2, Sentinel-2, Landsat time series

**Meteorological file:**
- Daily CSV from [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/) or your own source
- Column names such as `T2M` (temperature), `PRECTOTCORR` (rainfall), `RH2M` (humidity), `ALLSKY_SFC_SW_DWN` (radiation) are recognised automatically
- Derived variables (Growing Degree Days, log-rainfall, seasonal cumulative values) are computed automatically from what is available

---

### 📋 How Many Years of Data Are Needed?

| Years available | What the tool can do |
|---|---|
| 1 year | Shows NDVI chart and season dates only |
| 2 years | Fits basic models — treat results as exploratory |
| 3 – 4 years | Models available with caution — indicative only |
| 5 – 9 years | Reliable models and predictions |
| 10+ years | Best results — strong statistical reliability |

---

### 📥 Outputs Available

- **Phenology Table** — SOS, POS, EOS dates and Day of Year for each season (CSV download)
- **Model Coefficients** — Fitted equation parameters for reproducible reporting (CSV download)
- **Prediction Results** — Forecast dates for a target year (CSV download)
- **Correlation Table** — Ranked climate variables with Pearson r and Spearman ρ (displayed in Climate Correlations tab)

        """)


if __name__ == "__main__":
    for k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params',
              'ndvi_df','ndvi_info','met_info','interp_freq','_fp']:
        if k not in st.session_state:
            st.session_state[k] = None
    main()

if __name__ == "__main__":
    for k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params',
              'ndvi_df','ndvi_info','met_info','interp_freq','_fp']:
        if k not in st.session_state:
            st.session_state[k] = None
    main()
