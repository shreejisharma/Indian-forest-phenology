"""
Universal Indian Forest Phenology Assessment  v2.3  (Restyled UI)
====================================================
Changes from v2.0:
  1. SPLIT PLOT  — if NDVI data spans > 8 years the time-series plot is
     automatically split into ≤8-year chunks for readability.
  2. CROSS-YEAR EOS FIX — the season-boundary logic now properly handles
     windows that wrap around the calendar year (e.g. Jun → May).
     EOS is no longer truncated at Dec 31; it is allowed to extend into
     the following year up to the next season's trough.
  3. 5-DAY INTERPOLATION GRID — interpolation is always done on a 5-day
     grid regardless of the raw NDVI cadence (MODIS 16-day, Sentinel 10-day,
     etc.) so that the Savitzky-Golay smoother receives evenly-spaced input.

AI ASSISTANT ADDED (v2.1 + AI):
  - New "🤖 AI Assistant" tab powered by Google Gemini (free tier)
  - Place ai_assistant_gemini_free.py next to this file
  - Add GEMINI_API_KEY to .streamlit/secrets.toml

Requirements:
    pip install streamlit pandas numpy scipy scikit-learn matplotlib statsmodels google-generativeai

Run:
    streamlit run Universal_Indian_Forest_Phenology_Assessment_v2_1_with_AI.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d as _scipy_interp1d
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
import threading
import time
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# ─── KEEP-ALIVE  (prevents Streamlit Cloud free-tier sleep) ──
# Pings this app's own URL every 10 minutes in a background thread.
# Set STREAMLIT_APP_URL in your Streamlit Cloud Secrets as:
#   STREAMLIT_APP_URL = "https://indian-forest-phenology-pnlas9tfyhyoft2vmglxpm.streamlit.app"
def _keep_alive_worker():
    try:
        import streamlit as _st
        url = _st.secrets.get("STREAMLIT_APP_URL", "")
    except Exception:
        url = ""
    if not url or not _REQUESTS_AVAILABLE:
        return
    while True:
        try:
            _requests.get(url, timeout=15)
        except Exception:
            pass
        time.sleep(600)   # ping every 10 minutes

if _REQUESTS_AVAILABLE:
    if "___keep_alive_started" not in __import__("builtins").__dict__:
        __import__("builtins").__dict__["___keep_alive_started"] = True
        _t = threading.Thread(target=_keep_alive_worker, daemon=True)
        _t.start()

# ── Optional statsmodels LOESS ────────────────────────────────
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
    _LOESS_AVAILABLE = True
except ImportError:
    _LOESS_AVAILABLE = False


# ─── LOESS fallback ──────────────────────────────────────────
def _loess_predict_fallback(x_train, y_train, x_new, frac=0.75):
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
    page_title="🌲 Indian Forest Phenology Assessment  v2.3",
    page_icon="🌲",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7FBF8;
}

/* ─── APP HEADER ─── */
.app-header {
    background: linear-gradient(135deg, #0D2016 0%, #1A3828 55%, #0F2D1F 100%);
    padding: 36px 48px 30px;
    border-radius: 20px;
    margin-bottom: 28px;
    border: 1px solid rgba(78,191,96,0.22);
    box-shadow: 0 10px 48px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative; overflow: hidden;
}
.app-header::before {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(78,191,96,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.app-header h1 {
    color: #F0FFF4; font-family: 'Inter', sans-serif;
    font-size: 2.0rem; font-weight: 800; margin: 0 0 8px;
    letter-spacing: -0.5px; line-height: 1.15;
}
.app-header p  { color: #8FC99A; font-size: 0.87rem; margin: 0; line-height: 1.75; }
.app-header .badge {
    display: inline-block; background: rgba(78,191,96,0.14);
    color: #6DD685; font-size: 0.73rem; font-weight: 700;
    padding: 3px 11px; border-radius: 20px; margin: 7px 4px 0 0;
    border: 1px solid rgba(78,191,96,0.28); letter-spacing: 0.35px;
}

/* ─── METRIC / KPI CARDS ─── */
.metric-card {
    background: #FFFFFF;
    padding: 22px 18px 18px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #E0EDE2;
    margin: 4px;
    box-shadow: 0 2px 10px rgba(13,32,22,0.07);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(13,32,22,0.13);
}
.metric-card .label {
    color: #4E7A56;
    font-size: 0.70rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 10px;
}
.metric-card .value {
    color: #0D2016; font-size: 2.0rem; font-weight: 800;
    margin: 0; line-height: 1;
}
.metric-card .sub {
    color: #8A9E8C; font-size: 0.74rem; margin-top: 7px;
}

/* ─── MODEL COMPARISON CARDS ─── */
.model-card {
    background: #FFFFFF;
    border-radius: 16px; padding: 20px 22px;
    border: 2px solid #E0EDE2; margin: 8px 0;
    box-shadow: 0 2px 10px rgba(13,32,22,0.06);
    position: relative;
}
.model-card.best {
    border-color: #2E7D32;
    background: linear-gradient(135deg, #F0FFF4 0%, #FFFFFF 100%);
    box-shadow: 0 4px 22px rgba(46,125,50,0.14);
}
.model-card.best::before {
    content: '✦ BEST MODEL'; position: absolute; top: -1px; right: 16px;
    background: #2E7D32; color: #fff; font-size: 0.64rem; font-weight: 800;
    letter-spacing: 0.9px; padding: 3px 11px; border-radius: 0 0 9px 9px;
}
.model-card .model-name {
    font-size: 1.05rem; font-weight: 700; color: #0D2016; margin-bottom: 6px;
}
.model-card .r2-bar-track {
    background: #EEF5EF; border-radius: 6px; height: 9px; margin: 8px 0 4px;
}
.model-card .r2-bar-fill {
    height: 9px; border-radius: 6px;
    background: linear-gradient(90deg, #4CAF50, #1B5E20);
}
.model-card .r2-bar-fill.negative { background: linear-gradient(90deg, #E53935, #B71C1C); }
.model-card .stats-row { display: flex; gap: 14px; margin-top: 8px; flex-wrap: wrap; }
.model-card .stat-chip {
    background: #F4F9F5; border: 1px solid #D4E8D6;
    border-radius: 8px; padding: 4px 11px;
    font-size: 0.75rem; color: #2E5E35; font-weight: 600;
}
.model-card .stat-chip.warn { background: #FFF8E1; border-color: #F9A825; color: #7A5000; }
.model-card .stat-chip.bad  { background: #FFEBEE; border-color: #E57373; color: #7A1A1A; }

/* ─── SECTION TITLES ─── */
.section-title {
    font-size: 1.08rem; font-weight: 800; color: #0D2016;
    margin: 28px 0 14px; padding-bottom: 9px;
    border-bottom: 2.5px solid #C8E6C9;
    letter-spacing: -0.2px;
}

/* ─── SENSOR STRIP (new) ─── */
.sensor-strip {
    display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 18px 0;
}
.sensor-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    border: 1.5px solid; cursor: default;
    transition: transform 0.12s;
}
.sensor-pill:hover { transform: translateY(-1px); }
.sensor-pill.modis  { background: #E8F5E9; color: #1B5E20; border-color: #A5D6A7; }
.sensor-pill.ls     { background: #FFF3E0; color: #E65100; border-color: #FFCC80; }
.sensor-pill.s2     { background: #E3F2FD; color: #0D47A1; border-color: #90CAF9; }

/* ─── STAT ROW (new horizontal stat strip) ─── */
.stat-strip {
    display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0;
}
.stat-chip-lg {
    background: #F4F9F5; border: 1px solid #D0E8D4;
    border-radius: 10px; padding: 7px 14px;
    font-size: 0.80rem; color: #1A3828; font-weight: 600;
    display: inline-flex; align-items: center; gap: 5px;
}
.stat-chip-lg b { color: #0D2016; font-weight: 800; }

/* ─── RESULT / EQUATION BOXES ─── */
.eq-box {
    background: linear-gradient(135deg, #F5EEF8 0%, #FAFBFF 100%);
    padding: 18px 22px; border-radius: 14px;
    border-left: 4px solid #6A1B9A;
    margin: 12px 0;
    box-shadow: 0 2px 10px rgba(106,27,154,0.08);
}
.eq-label {
    display: block; font-size: 0.70rem; font-weight: 800; color: #6A1B9A;
    text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 10px;
}
.eq-main {
    display: block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.87rem; color: #1A1A2E; font-weight: 500;
    line-height: 1.65; margin-bottom: 8px;
    background: rgba(255,255,255,0.75); padding: 10px 14px;
    border-radius: 8px; border: 1px solid rgba(106,27,154,0.10);
}
.eq-meta  {
    display: block; font-size: 0.77rem; color: #5A5A6A; margin-bottom: 4px;
}
.eq-models {
    display: block; font-size: 0.74rem; color: #7A7A8A;
    background: rgba(106,27,154,0.05); padding: 6px 12px;
    border-radius: 6px; margin-top: 6px;
}

/* ─── RESULT BOX ─── */
.result-box {
    background: linear-gradient(135deg, #EDF7ED 0%, #F9FBFF 100%);
    padding: 18px 22px; border-radius: 14px;
    border-left: 4px solid #2E7D32; margin: 12px 0;
    box-shadow: 0 3px 12px rgba(46,125,50,0.09);
}
.result-box.delay   { background: linear-gradient(135deg, #FFF8E1 0%, #FFFCF4 100%);
    border-left-color: #F9A825; }
.result-box.advance { background: linear-gradient(135deg, #E3F2FD 0%, #F5FBFF 100%);
    border-left-color: #1976D2; }
.result-title {
    font-size: 0.70rem; font-weight: 800; letter-spacing: 0.9px;
    text-transform: uppercase; color: #2E7D32; margin-bottom: 10px; display: block;
}
.result-title.delay   { color: #8B6000; }
.result-title.advance { color: #1565C0; }
.result-main {
    font-size: 1.0rem; font-weight: 600; color: #0D2016;
    line-height: 1.55; margin-bottom: 8px; display: block;
}
.result-meta {
    font-size: 0.79rem; color: #5A7A60; font-family: 'JetBrains Mono', monospace;
    background: rgba(255,255,255,0.7); padding: 7px 12px; border-radius: 7px;
    border: 1px solid rgba(46,125,50,0.14); display: block; margin-top: 8px;
}

/* ─── BANNERS ─── */
.banner-info  { background: #EBF5FB; padding: 13px 18px; border-radius: 10px;
    border-left: 3px solid #1976D2; margin: 10px 0; font-size: 0.86rem; color: #1A3A5C; }
.banner-warn  { background: #FFFBEC; padding: 13px 18px; border-radius: 10px;
    border-left: 3px solid #F9A825; margin: 10px 0; font-size: 0.86rem; color: #5A4000; }
.banner-good  { background: #F0FFF4; padding: 13px 18px; border-radius: 10px;
    border-left: 3px solid #43A047; margin: 10px 0; font-size: 0.86rem; color: #1A3A20; }
.banner-error { background: #FFF0F0; padding: 13px 18px; border-radius: 10px;
    border-left: 3px solid #E53935; margin: 10px 0; font-size: 0.86rem; color: #5A1A1A; }

/* ─── MODEL BADGES ─── */
.model-badge {
    display: inline-block; padding: 2px 9px; border-radius: 10px;
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.3px; margin-left: 6px;
    vertical-align: middle; font-family: 'JetBrains Mono', monospace;
}
.model-badge.ridge { background: #E8F5E9; color: #1B5E20; border: 1px solid #A5D6A7; }
.model-badge.loess { background: #E3F2FD; color: #0D47A1; border: 1px solid #90CAF9; }
.model-badge.poly2 { background: #FFF3E0; color: #E65100; border: 1px solid #FFCC80; }
.model-badge.poly3 { background: #FBE9E7; color: #BF360C; border: 1px solid #FFAB91; }
.model-badge.gpr   { background: #F3E5F5; color: #6A1B9A; border: 1px solid #CE93D8; }
.model-badge.mean  { background: #F5F5F5; color: #757575; border: 1px solid #E0E0E0; }

/* ─── CONF BADGE (new) ─── */
.conf-high   { display:inline-block; background:#E8F5E9; color:#1B5E20; border:1px solid #A5D6A7;
    padding:2px 10px; border-radius:12px; font-size:0.73rem; font-weight:800; }
.conf-med    { display:inline-block; background:#FFF8E1; color:#8B6000; border:1px solid #FFD54F;
    padding:2px 10px; border-radius:12px; font-size:0.73rem; font-weight:800; }
.conf-low    { display:inline-block; background:#FFEBEE; color:#7A1A1A; border:1px solid #EF9A9A;
    padding:2px 10px; border-radius:12px; font-size:0.73rem; font-weight:800; }

/* ─── UPLOAD PANEL ─── */
.upload-panel {
    background: linear-gradient(135deg, #F6FBF7 0%, #EEF7EF 100%);
    padding: 32px 36px; border-radius: 20px;
    border: 2px dashed #A5D6A7; margin: 20px 0;
    box-shadow: 0 2px 18px rgba(27,94,32,0.05);
}
.upload-panel h3 { color: #0D2016; margin-bottom: 16px; font-size: 1.1rem; font-weight: 700; }
.upload-panel code {
    background: #E8F5E9; padding: 2px 7px; border-radius: 5px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.81rem; color: #1B5E20;
}
.upload-panel .up-section {
    background: rgba(255,255,255,0.82); border-radius: 12px;
    padding: 14px 18px; margin: 10px 0; border: 1px solid #C8E6C9;
}
.upload-panel .up-label { font-weight: 700; color: #1B5E20; font-size: 0.88rem; margin-bottom: 4px; }
.upload-panel .feature-item {
    display: inline-block; background: rgba(27,94,32,0.09);
    color: #1B5E20; font-size: 0.75rem; font-weight: 600;
    padding: 3px 9px; border-radius: 12px; margin: 2px 2px 0 0;
    border: 1px solid rgba(27,94,32,0.14);
}

/* ─── SENSOR COMPARE BADGES ─── */
.badge-best { background:#E8F5E9; color:#1B5E20; border:1px solid #A5D6A7;
    border-radius:8px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
.badge-ok   { background:#FFF8E1; color:#7A5000; border:1px solid #FFD080;
    border-radius:8px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
.badge-poor { background:#FFEBEE; color:#7A1A1A; border:1px solid #FFAAAA;
    border-radius:8px; padding:2px 9px; font-size:0.72rem; font-weight:700; }
.model-tbl { width:100%; border-collapse:collapse; font-size:0.86rem; }
.model-tbl th { background:#0D2016; color:#FFFFFF; padding:10px 14px; text-align:left;
    font-size:0.74rem; text-transform:uppercase; letter-spacing:0.6px; }
.model-tbl td { padding:10px 14px; border-bottom:1px solid #EDF2F8; }
.model-tbl tr:nth-child(even) td { background:#F4F9F5; }
.model-tbl tr:hover td { background:#E8F5E9; }

/* ─── MISC ─── */
.term { background: #E8F5E9; padding: 2px 8px; border-radius: 5px;
    font-weight: 600; color: #1B5E20; font-size: 0.87rem; }
.stat-card { background: #FAFFFE; padding: 12px 16px; border-radius: 10px;
    border: 1px solid #C8E6C9; margin: 4px 0; font-size: 0.87rem; }
.pred-event-header {
    padding: 14px 18px; border-radius: 12px;
    margin: 10px 0; border-left-width: 4px; border-left-style: solid;
}
.pred-event-header .ev-title { font-size: 1.0rem; font-weight: 700; margin-bottom: 2px; }
.pred-event-header .ev-meta  { font-size: 0.82rem; color: #555; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────
MIN_CORR_THRESHOLD = 0.40
ALPHAS = [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000]

ACCUM_KEYWORDS = [
    'PREC', 'RAIN', 'PPT', 'PRECIP', 'RAINFALL',
    'GDD_5', 'GDD_10', 'GDD',
    'LOG_P', 'LOG_PREC',
    'SPEI', 'SPEI_PROXY',
    'PET', 'ET',
    'CDD', 'HDD',
]
SNAPSHOT_KEYWORDS = ['GDD_CUM', 'CPPT', 'CT2M', 'CUMUL', 'ACCUM', 'CUM_']
SNAPSHOT_FEATURES = {'GDD_cum', 'GDD_CUM', 'CPPT', 'CT2M'}

# ─── FIXED INTERPOLATION STEP (5 days always) ────────────────
INTERP_STEP_DAYS = 5   # Force 5-day grid regardless of input cadence


# ═══════════════════════════════════════════════════════════════
# DATA-ADAPTIVE UTILITIES
# ═══════════════════════════════════════════════════════════════

def detect_ndvi_cadence(ndvi_df):
    dates = pd.to_datetime(ndvi_df['Date']).sort_values()
    diffs = dates.diff().dt.days.dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 16, 64, INTERP_STEP_DAYS
    median_cad = float(diffs.median())
    max_gap    = max(60, int(median_cad * 8))
    return median_cad, max_gap, INTERP_STEP_DAYS


def detect_seasonality(ndvi_series_5d):
    vals = ndvi_series_5d.dropna().values
    if len(vals) < 24:
        return 73
    n = len(vals)
    v = vals - vals.mean()
    max_lag = min(n // 2, 110)
    acf = []
    for lag in range(10, max_lag):
        if lag >= len(v): break
        r = np.corrcoef(v[:n-lag], v[lag:])[0, 1]
        acf.append((lag, r))
    if not acf:
        return 73
    acf_arr = np.array(acf)
    for i in range(1, len(acf_arr) - 1):
        if (acf_arr[i, 1] > acf_arr[i-1, 1] and
                acf_arr[i, 1] > acf_arr[i+1, 1] and
                acf_arr[i, 1] > 0.3 and
                acf_arr[i, 0] > 20):
            return int(acf_arr[i, 0])
    return 73


def compute_data_driven_min_amplitude(ndvi_vals_clean):
    p5  = float(np.percentile(ndvi_vals_clean, 5))
    p95 = float(np.percentile(ndvi_vals_clean, 95))
    data_range = p95 - p5
    return max(0.01, data_range * 0.05)


def characterize_ndvi_data(ndvi_df):
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


def audit_met_coverage(met_df, ndvi_df, pheno_df, window=15):
    result = {
        'met_cadence_days': None,
        'met_is_paired_with_ndvi': False,
        'met_has_large_gaps': False,
        'met_gap_periods': [],
        'per_event_coverage': {},
        'warnings': [],
    }
    if met_df is None or len(met_df) == 0:
        result['warnings'].append("Meteorological file is empty.")
        return result

    met_dates = pd.to_datetime(met_df['Date']).sort_values()
    diffs = met_dates.diff().dt.days.dropna()
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) > 0:
        result['met_cadence_days'] = float(pos_diffs.median())

    if ndvi_df is not None:
        ndvi_dates_set = set(pd.to_datetime(ndvi_df['Date']).dt.date.tolist())
        met_dates_set  = set(met_dates.dt.date.tolist())
        overlap = len(ndvi_dates_set & met_dates_set)
        total   = len(met_dates_set)
        if total > 0 and overlap / total >= 0.90:
            result['met_is_paired_with_ndvi'] = True
            result['warnings'].append(
                "⚠️ Your meteorological file appears to be sampled at the same dates as your NDVI "
                "rather than being a continuous daily record. This severely limits the app's ability "
                "to compute climate windows before each season event — many windows will be empty or "
                "have very few data points. "
                "For best results, upload a continuous daily meteorological CSV (e.g. from NASA POWER "
                "Daily for your coordinates)."
            )

    gap_mask = diffs > 60
    if gap_mask.any():
        result['met_has_large_gaps'] = True
        gap_end_dates   = met_dates[diffs.index[gap_mask]]
        gap_start_dates = met_dates.shift(1)[diffs.index[gap_mask]]
        for gs, ge in zip(gap_start_dates, gap_end_dates):
            gap_str = f"{pd.Timestamp(gs).strftime('%b %Y')} → {pd.Timestamp(ge).strftime('%b %Y')}"
            result['met_gap_periods'].append(gap_str)
        result['warnings'].append(
            f"⚠️ Large gaps detected in meteorological data: "
            + ", ".join(result['met_gap_periods'])
            + ". Seasons whose event windows fall inside these gaps will be excluded from model training."
        )

    if 'Date' in met_df.columns:
        met_df_dt = met_df.copy()
        met_df_dt['_year'] = pd.to_datetime(met_df_dt['Date']).dt.year
        met_df_dt['_doy']  = pd.to_datetime(met_df_dt['Date']).dt.dayofyear
        year_doy = met_df_dt.groupby('_year')['_doy'].agg(['min', 'max', 'count'])
        partial_years = []
        for yr, row_y in year_doy.iterrows():
            if row_y['max'] < 240:
                partial_years.append(
                    f"{yr} (DOY {int(row_y['min'])}–{int(row_y['max'])}, "
                    f"{int(row_y['count'])} obs)"
                )
        if partial_years:
            result['warnings'].append(
                f"⚠️ <b>Partial-year meteorological data detected</b> — the following years only cover "
                f"the first part of the calendar year and have NO data in the growing-season "
                f"(Jul–Nov) window: <b>{', '.join(partial_years)}</b>. "
                f"These years will be <b>excluded from model training</b>, reducing your effective "
                f"sample size. Upload a complete full-year meteorological file for all years to fix this."
            )

    if pheno_df is not None and len(pheno_df) > 0:
        n_total = len(pheno_df)
        for ev in ['SOS', 'POS', 'EOS']:
            date_col = f'{ev}_Date'
            if date_col not in pheno_df.columns:
                continue
            n_with_data = 0
            n_rows_list = []
            seasons_missing = []
            for _, row in pheno_df.iterrows():
                evt_dt = row[date_col]
                if pd.isna(evt_dt):
                    seasons_missing.append(int(row['Year']))
                    continue
                mask = ((met_df['Date'] >= pd.Timestamp(evt_dt) - timedelta(days=window)) &
                        (met_df['Date'] <= pd.Timestamp(evt_dt)))
                n_rows = len(met_df[mask])
                if n_rows >= 1:
                    n_with_data += 1
                    n_rows_list.append(n_rows)
                else:
                    seasons_missing.append(int(row['Year']))
            result['per_event_coverage'][ev] = {
                'n_seasons_with_data': n_with_data,
                'n_seasons_total':     n_total,
                'seasons_missing':     seasons_missing,
                'coverage_pct':        round(100 * n_with_data / n_total, 0) if n_total > 0 else 0,
                'avg_rows_per_window': round(float(np.mean(n_rows_list)), 1) if n_rows_list else 0,
            }
            if n_with_data < n_total:
                missing_yrs = ", ".join(str(y) for y in seasons_missing)
                result['warnings'].append(
                    f"⚠️ {ev} model: {n_total - n_with_data} season(s) have NO meteorological data "
                    f"in the {window}-day pre-event window (year(s): {missing_yrs}). "
                    f"Only {n_with_data} of {n_total} seasons can be used for training."
                )
            if n_rows_list and np.mean(n_rows_list) < 4:
                result['warnings'].append(
                    f"⚠️ {ev} model: average rows per climate window = "
                    f"{np.mean(n_rows_list):.1f} (window={window} days, "
                    f"met cadence≈{result.get('met_cadence_days', '?'):.0f} days). "
                    f"Feature averages based on fewer than 4 observations are unreliable. "
                    f"Increase the <b>Climate Window</b> slider to at least "
                    f"{int(result.get('met_cadence_days', 5) * 6)} days, or upload daily met data."
                )
    return result


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
                if up.startswith('YEAR') or up.startswith('LON') or up.startswith('DATE'):
                    skip_to = i; break
        df = pd.read_csv(StringIO('\n'.join(lines[skip_to:])))
        df.columns = [c.strip() for c in df.columns]
        df.replace([-999, -999.0, -99, -99.0, -9999, -9999.0], np.nan, inplace=True)
        if 'Date' not in df.columns:
            if {'YEAR', 'DOY'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(
                    df['YEAR'].astype(str) + df['DOY'].astype(str).str.zfill(3),
                    format='%Y%j', errors='coerce')
            elif {'YEAR', 'MO', 'DY'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(
                    df['YEAR'].astype(str) + '-' + df['MO'].astype(str).str.zfill(2) + '-' +
                    df['DY'].astype(str).str.zfill(2), errors='coerce')
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
        explicit_doy = (best_explicit.dt.dayofyear.where(best_explicit.notna(), np.nan)
                        if best_explicit is not None else pd.Series([np.nan]*len(series)))
        scores = {
            'default':  (default_doy == doy_ref).sum(),
            'dayfirst': (dayfirst_doy == doy_ref).sum(),
            'explicit': (explicit_doy == doy_ref).sum(),
        }
        best_key = max(scores, key=scores.get)
        return {'default': parsed_default, 'dayfirst': parsed_dayfirst,
                'explicit': best_explicit or parsed_default}[best_key]
    candidates = [parsed_default, parsed_dayfirst]
    if best_explicit is not None and best_explicit_n >= len(series) * 0.85:
        candidates.append(best_explicit)
    def _score(s):
        if s is None or s.notna().sum() == 0: return -999
        yr_median = s.dropna().dt.year.median()
        yr_plausible = 1 if 1980 <= yr_median <= 2040 else 0
        return yr_plausible * 1000 + s.dropna().dt.month.nunique() * 10 + s.notna().sum()
    return max(candidates, key=_score)


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
# DERIVED MET FEATURES
# ═══════════════════════════════════════════════════════════════

def _season_cumsum(series, dates, sm_):
    out = series.copy() * 0.0
    dt = pd.to_datetime(dates)
    season_yr = np.where(dt.dt.month >= sm_, dt.dt.year, dt.dt.year - 1)
    season_yr_s = pd.Series(season_yr, index=series.index)
    for sy, grp_idx in series.groupby(season_yr_s).groups.items():
        out.loc[grp_idx] = series.loc[grp_idx].cumsum().values
    return out


def _detect_column(cols, *keyword_groups):
    cols_upper = {c: c.upper() for c in cols}
    for keywords in keyword_groups:
        for c in cols:
            cu = cols_upper[c]
            for kw in keywords:
                if kw.upper() == cu or kw.upper() in cu:
                    return c
    return None


def add_derived_features(met_df, season_start_month=1):
    df   = met_df.copy()
    cols = df.columns.tolist()

    tmin = _detect_column(cols, ['T2M_MIN','TMIN','TEMP_MIN'], ['MIN_T','TMIN','MINTEMP','T_MIN','TEMPMIN'], ['MIN'])
    tmax = _detect_column(cols, ['T2M_MAX','TMAX','TEMP_MAX'], ['MAX_T','TMAX','MAXTEMP','T_MAX','TEMPMAX'], ['MAX'])
    tmn  = _detect_column(cols, ['T2M','TMEAN','TAVG','TEMP_MEAN','T_MEAN'], ['TEMP','TEMPERATURE','AIR_T','TAIR'], ['T2M'])
    if tmn and (tmn == tmin or tmn == tmax): tmn = None

    rh   = _detect_column(cols, ['RH2M','RH','RHUM','REL_HUM','RELATIVE_HUMIDITY'], ['HUMID','RH_','HR'], ['RH'])
    prec = _detect_column(cols, ['PRECTOTCORR','PRECTOT','PRECIP','PRECIPITATION'], ['RAIN','PPT','RAINFALL','PREC','PR_'], ['PPT','RAIN','PREC'])
    if prec:
        cu = prec.upper()
        if any(k in cu for k in ['CUM','CPPT','CUMUL','ACCUM','TOTAL_P']):
            alt = _detect_column([c for c in cols if c != prec], ['PPT','RAIN','PRECIP','PREC'], ['PPT','RAIN'])
            prec = alt
    sm  = _detect_column(cols, ['GWETTOP','GWETROOT','GWETPROF','SOIL_MOISTURE','SM_TOP'], ['SOIL_W','SOILW','SM_','VSM','SWC'], ['GWET','SOIL'])
    rad = _detect_column(cols, ['ALLSKY_SFC_SW_DWN','SRAD','RAD','SOLAR','INSOL','RADIATION'], ['SW_DWN','SHORTWAVE','SOLRAD','RS','RADSOL'], ['RAD','SOL'])

    tavg = None
    if tmin and tmax and tmin != tmax:
        tavg = (df[tmax] + df[tmin]) / 2.0
        if 'DTR' not in cols: df['DTR'] = df[tmax] - df[tmin]
    elif tmn:  tavg = df[tmn]
    elif tmin: tavg = df[tmin]
    elif tmax: tavg = df[tmax]

    if tavg is not None:
        if 'GDD_10' not in cols: df['GDD_10'] = np.maximum(tavg - 10, 0)
        if 'GDD_5'  not in cols: df['GDD_5']  = np.maximum(tavg - 5,  0)
        if 'GDD_cum' not in cols:
            df['GDD_cum'] = _season_cumsum(
                np.maximum(tavg - 10, 0).rename('GDD_10_tmp'), df['Date'], season_start_month)

    if prec and 'log_precip' not in cols:
        df['log_precip'] = np.log1p(np.maximum(df[prec].fillna(0), 0))
    if tavg is not None and rh and 'VPD' not in cols:
        es = 0.6108 * np.exp((17.27 * tavg) / (tavg + 237.3))
        df['VPD'] = np.maximum(es * (1 - df[rh] / 100.0), 0)
    if prec and sm and 'MSI' not in cols:
        df['MSI'] = df[prec] / (df[sm].replace(0, np.nan) + 1e-6)
    if prec and tavg is not None and 'SPEI_proxy' not in cols:
        pet = 0.0023 * (tavg + 17.8) * np.maximum(tavg, 0) ** 0.5
        df['SPEI_proxy'] = df[prec].fillna(0) - pet.fillna(0)
    if rad and rad not in ('ALLSKY_SFC_SW_DWN',) and 'ALLSKY_SFC_SW_DWN' not in cols:
        df['ALLSKY_SFC_SW_DWN'] = df[rad]
    return df


# ═══════════════════════════════════════════════════════════════
# TRAINING FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════

def make_training_features(pheno_df, met_df, params, window=15):
    records = []
    for _, row in pheno_df.iterrows():
        for event in ['SOS', 'POS', 'EOS']:
            evt_dt = row[f'{event}_Date']
            if pd.isna(evt_dt): continue
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
            if len(wdf) < max(3, window * 0.15): continue
            for p in params:
                if p not in met_df.columns: continue
                p_upper = p.upper()
                is_snapshot = (p_upper in SNAPSHOT_FEATURES or
                               any(k in p_upper for k in SNAPSHOT_KEYWORDS))
                is_accum = (not is_snapshot and any(k in p_upper for k in ACCUM_KEYWORDS))
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
# PHENOLOGY EXTRACTION  — FIXED FOR CROSS-YEAR SEASONS
# ═══════════════════════════════════════════════════════════════

def _find_troughs(ndvi_values, min_distance=10):
    n = len(ndvi_values)
    troughs = []
    for i in range(min_distance, n - min_distance):
        window = ndvi_values[max(0, i - min_distance): i + min_distance + 1]
        if ndvi_values[i] == np.min(window):
            if ndvi_values[i] <= ndvi_values[i-1] and ndvi_values[i] <= ndvi_values[i+1]:
                troughs.append(i)
    if not troughs: return troughs
    merged = [troughs[0]]
    for t in troughs[1:]:
        if t - merged[-1] < min_distance:
            if ndvi_values[t] < ndvi_values[merged[-1]]: merged[-1] = t
        else:
            merged.append(t)
    return merged


def _find_troughs_boundary(v, min_distance):
    """
    Improved trough finder:
    - Uses a wider local window (2× min_distance) for more stable minima
    - Merges nearby troughs by keeping the true minimum
    - Head/tail boundary troughs require the trough to be clearly below
      the mid-cycle mean (not just below a single midpoint)
    - Filters out shallow troughs that are above P25 of the full series
      when the overall amplitude is large (avoids noise bumps being
      mistaken for season boundaries)
    """
    n = len(v)
    v_clean = v[~np.isnan(v)] if np.any(np.isnan(v)) else v
    global_p25 = float(np.percentile(v_clean, 25)) if len(v_clean) > 4 else -np.inf
    global_p75 = float(np.percentile(v_clean, 75)) if len(v_clean) > 4 else np.inf
    global_amp = global_p75 - global_p25

    troughs = []
    half = min_distance
    for i in range(1, n - 1):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w  = v[lo:hi]
        # Must be the minimum in its neighbourhood AND a local descent
        if v[i] == np.nanmin(w) and v[i] <= v[i - 1] and v[i] <= v[i + 1]:
            # Skip trivially shallow troughs when amplitude is substantial
            if global_amp > 0.08 and v[i] > global_p25 + 0.80 * global_amp:
                continue
            troughs.append(i)

    if not troughs:
        return []

    # Merge troughs that are too close — keep the deepest
    merged = [troughs[0]]
    for t in troughs[1:]:
        if t - merged[-1] < min_distance:
            if v[t] < v[merged[-1]]:
                merged[-1] = t
        else:
            merged.append(t)

    # ── Head boundary trough ────────────────────────────────
    # Only add if the series start looks like it is already past a trough
    search_end = min(int(min_distance * 2.0), n // 4)
    if search_end > 2 and merged:
        head_win = v[0:search_end]
        bmi = int(np.argmin(head_win))
        next_t = merged[0]
        gap_ok  = (next_t - bmi) >= min_distance
        # The candidate must be clearly below the mean of the cycle it precedes
        cycle_mean = float(np.nanmean(v[bmi:next_t])) if next_t > bmi else np.inf
        depth_ok   = v[bmi] < cycle_mean - 0.10 * global_amp
        if bmi > 0 and gap_ok and depth_ok:
            merged.insert(0, bmi)

    # ── Tail boundary trough ────────────────────────────────
    search_start = max(n - int(min_distance * 2.0), 3 * n // 4)
    if search_start < n - 2 and merged:
        tail_win = v[search_start:]
        bmi = search_start + int(np.argmin(tail_win))
        prev_t = merged[-1]
        gap_ok  = (bmi - prev_t) >= min_distance and bmi < n - 2
        cycle_mean = float(np.nanmean(v[prev_t:bmi])) if bmi > prev_t else np.inf
        depth_ok   = v[bmi] < cycle_mean - 0.10 * global_amp
        if gap_ok and depth_ok:
            merged.append(bmi)

    return merged


def _is_peak_in_window(pos_date, sm, em):
    """
    Check if the peak date falls within the season window.
    Handles cross-year windows (e.g. sm=6 (Jun), em=5 (May) means Jun–May next year).
    """
    m = pos_date.month
    if sm <= em:
        return sm <= m <= em
    else:
        return m >= sm or m <= em


def extract_phenology(ndvi_df, cfg, sos_threshold_pct, eos_threshold_pct):
    """
    Extract phenology events from NDVI time series.

    Key fixes (v2.1):
    - Interpolation always uses INTERP_STEP_DAYS (5-day) grid
    - EOS is allowed to extend across the calendar year boundary
    - Cross-year season window handled correctly in _is_peak_in_window
    - EOS cap: min(next_trough_date, data_end + 60 days) instead of hard Dec-31 wall
    """
    try:
        sm    = cfg["start_month"]
        em    = cfg["end_month"]
        min_d = cfg.get("min_days", 100)
        thr_pct     = sos_threshold_pct
        eos_thr_pct = eos_threshold_pct

        ndvi_raw = ndvi_df[["Date", "NDVI"]].copy().set_index("Date").sort_index()
        if ndvi_raw.index.duplicated().any():
            ndvi_raw = ndvi_raw.groupby(ndvi_raw.index)['NDVI'].mean().rename('NDVI').to_frame()

        orig_dates  = ndvi_raw.index.sort_values()
        orig_diffs  = pd.Series(orig_dates).diff().dt.days.fillna(0)
        pos_diffs   = orig_diffs[orig_diffs > 0]
        typical_cad = float(pos_diffs.median()) if len(pos_diffs) > 0 else 16.0
        MAX_INTERP_GAP = max(60, int(typical_cad * 8))
        gap_starts  = orig_dates[orig_diffs.values > MAX_INTERP_GAP]

        # ── FIX 1: Always use 5-day interpolation grid ──────────
        interp_freq = INTERP_STEP_DAYS   # fixed at 5 days

        full_range  = pd.date_range(start=ndvi_raw.index.min(), end=ndvi_raw.index.max(),
                                    freq=f"{interp_freq}D")
        ndvi_5d = ndvi_raw.reindex(ndvi_raw.index.union(full_range))
        ndvi_5d = ndvi_5d.interpolate(method="time", limit_area="inside")
        for gap_start in gap_starts:
            before = orig_dates[orig_dates < gap_start]
            if len(before) == 0: continue
            mask = (ndvi_5d.index > before[-1]) & (ndvi_5d.index < gap_start)
            ndvi_5d.loc[mask] = np.nan
        ndvi_5d = ndvi_5d.reindex(full_range)
        ndvi_5d.columns = ["NDVI"]

        n         = len(ndvi_5d)
        ndvi_vals = ndvi_5d["NDVI"].values.copy()
        valid_mask = ~np.isnan(ndvi_vals)
        valid_vals  = ndvi_vals[valid_mask]
        MIN_AMPLITUDE = compute_data_driven_min_amplitude(valid_vals) if len(valid_vals) > 5 else 0.02

        MAX_SG_STEPS = 31
        sm_vals  = np.full(n, np.nan)
        seg_labels = np.zeros(n, dtype=int)
        seg_id, in_seg = 0, False
        for i in range(n):
            if valid_mask[i]:
                if not in_seg: seg_id += 1; in_seg = True
                seg_labels[i] = seg_id
            else:
                in_seg = False

        # SG smooth segments
        for sid in range(1, seg_id + 1):
            idx_seg = np.where(seg_labels == sid)[0]
            seg_n   = len(idx_seg)
            if seg_n < 5: sm_vals[idx_seg] = ndvi_vals[idx_seg]; continue
            wl_t = max(7, min(int(seg_n * 0.05), MAX_SG_STEPS))
            wl_s = wl_t if wl_t % 2 == 1 else wl_t + 1
            wl_s = min(wl_s, seg_n - 1 if seg_n > 1 else 1)
            if wl_s % 2 == 0: wl_s = max(7, wl_s - 1)
            poly_s = min(2, wl_s - 1)
            if wl_s >= 5 and wl_s < seg_n:
                sm_vals[idx_seg] = savgol_filter(ndvi_vals[idx_seg], wl_s, poly_s)
            else:
                sm_vals[idx_seg] = ndvi_vals[idx_seg]

        t_all = ndvi_5d.index
        sm_for_troughs = pd.Series(sm_vals, index=t_all).interpolate(
            method="linear", limit_direction="both").values

        try:
            cycle_steps = detect_seasonality(pd.Series(sm_for_troughs, index=t_all))
        except Exception:
            cycle_steps = int(365 / interp_freq)
        min_dist = max(10, int(cycle_steps * 0.4))


        trough_raw = _find_troughs_boundary(sm_for_troughs, min_dist)
        trough_indices = []
        for ti in trough_raw:
            window_sl = slice(max(0, ti - 5), min(n, ti + 6))
            if np.isnan(sm_vals[window_sl]).any(): continue
            trough_indices.append(ti)

        if len(trough_indices) >= 2:
            valid_sm   = sm_for_troughs[~np.isnan(sm_vals)]

            global_min = float(np.percentile(valid_sm, 5))
            global_max = float(np.percentile(valid_sm, 95))
            global_amp = global_max - global_min
            if global_amp >= 0.20:
                trough_ceil = global_min + 0.85 * global_amp
                trough_indices = [ti for ti in trough_indices if sm_for_troughs[ti] <= trough_ceil]

        _GAP_STRICT   = 0.20
        _GAP_TOLERANT = 0.50
        _AMP_GAP_THR  = 0.10

        def _cycle_has_gap(i_start, i_end, amplitude=None):
            if i_end <= i_start: return True
            gap_frac = np.isnan(sm_vals[i_start:i_end + 1]).mean()
            if amplitude is not None and amplitude >= _AMP_GAP_THR:
                return gap_frac > _GAP_TOLERANT
            return gap_frac > _GAP_STRICT

        rows = []

        # ── FIX 2: Build EOS upper bound per cycle ─────────────
        # EOS must not exceed the START of the next trough (which marks
        # the beginning of the next season). This replaces the old hard
        # "data_end_dt" cap that was cutting EOS at the end of the array.
        def _eos_upper_bound(cycle_end_trough_idx):
            """Return a Timestamp that is a safe upper bound for EOS."""
            # The next trough is the natural end; add a small buffer.
            return t_all[cycle_end_trough_idx] + pd.Timedelta(days=interp_freq * 2)

        _valid_sm_vals = sm_for_troughs[~np.isnan(sm_vals)]
        _global_min    = float(np.percentile(_valid_sm_vals, 5))  if len(_valid_sm_vals) > 0 else 0
        _global_amp    = (float(np.percentile(_valid_sm_vals, 95)) - _global_min) if len(_valid_sm_vals) > 0 else 1
        _head_trough_ceiling = _global_min + 0.40 * _global_amp
        _head_start_looks_like_trough = (float(sm_for_troughs[0]) <= _head_trough_ceiling)

        # ── helper to extract one season cycle ────────────────
        def _extract_cycle(cycle_raw, cycle_t, ndvi_min, pos_idx_hint=None, eos_upper=None):
            """
            Improved cycle extraction (v2.2):

            Key improvements over v2.1:
            ─────────────────────────────────────────────────────────────
            1. BASE NDVI  — use the lower of (trough value, P10 of cycle)
               so that noise spikes at the trough don't inflate the base
               and push thresholds too high.

            2. SOS — use the LAST point that is still BELOW the threshold
               on the ascending limb, then step forward one point.
               This anchors SOS to the true green-up crossing rather than
               the first noise exceedance.

            3. EOS — use the FIRST point that falls AND STAYS below the
               threshold.  Require at least 2 consecutive sub-threshold
               points so that a single noisy dip doesn't trigger a
               premature EOS.

            4. EOS UPPER BOUND — clamp strictly to eos_upper (next trough
               start) without the -1 offset that was causing early EOS.

            5. Peak validation — reject cycles where the peak is within
               the first or last 5% of the cycle (likely a boundary
               artefact, not a real season).
            ─────────────────────────────────────────────────────────────
            """
            n_c = len(cycle_raw)
            if n_c < 10:
                return None

            ndvi_max = float(np.nanmax(cycle_raw))

            # ── 1. Robust base NDVI ────────────────────────────────────
            # Use the trough value passed in (ndvi_min) as primary base.
            # P10 of the FULL cycle is unreliable because the high-NDVI
            # peak portion drags P10 upward, inflating sos_thr above the
            # ascending limb and causing SOS ≈ POS.
            # Instead, use P10 of only the first 25% of the cycle (the
            # trough-to-green-up segment) which is where the true base lies.
            first_quarter = cycle_raw[:max(3, n_c // 4)]
            p10_base = float(np.nanpercentile(first_quarter, 10))
            base = min(ndvi_min, p10_base)
            base = max(base, 0.0)

            A = ndvi_max - base
            if A < MIN_AMPLITUDE:
                return None

            sos_thr = base + thr_pct     * A
            eos_thr = base + eos_thr_pct * A

            # ── Peak index ─────────────────────────────────────────────
            pi = int(np.nanargmax(cycle_raw)) if pos_idx_hint is None else pos_idx_hint
            pos = cycle_t[pi]

            # Reject boundary peaks (artefacts from head/tail segments)
            margin = max(3, int(n_c * 0.05))
            if pi < margin or pi > n_c - margin:
                return None

            # ── 2. SOS: last-below-then-first-above on ascending limb ─
            # Ascending limb = cycle_raw[0 : pi+1]
            # Find the LAST index on the ascending limb that is still
            # BELOW sos_thr. SOS = the next index (first above).
            asc = cycle_raw[:pi + 1]          # includes peak
            below_mask = asc < sos_thr
            below_indices = np.where(below_mask)[0]
            if len(below_indices) == 0:
                # Entire ascending limb already above threshold — SOS = start
                si = 0
            else:
                si = int(below_indices[-1]) + 1
            if si >= pi:
                # The threshold crossing is never reached before the peak.
                # This means sos_thr is above the ascending limb — the base
                # is inflated or the season has no proper green-up.
                # Reject this cycle rather than produce SOS ≈ POS.
                return None
            sos = cycle_t[min(si, n_c - 1)]

            # ── 3. EOS: tiered detection ───────────────────────────────
            # Descending limb starts right after peak.
            desc = cycle_raw[pi:]
            n_desc = len(desc)
            ei_rel = None

            # Tier 1 (strict): 2 consecutive points below threshold
            for k in range(1, n_desc - 1):
                if desc[k] < eos_thr and desc[min(k + 1, n_desc - 1)] < eos_thr:
                    ei_rel = k
                    break

            # Tier 2 (relaxed): 1 crossing + next 5-step mean still below threshold
            if ei_rel is None:
                for k in range(1, n_desc):
                    if desc[k] < eos_thr:
                        lookahead = desc[k: min(k + 6, n_desc)]
                        if float(np.nanmean(lookahead)) < eos_thr + 0.02 * A:
                            ei_rel = k
                            break

            # Tier 3 (fallback): use trough only if NDVI is genuinely low there;
            # otherwise use first crossing below mid-amplitude
            if ei_rel is None:
                trough_rel = int(np.nanargmin(desc))
                mid_thr = base + 0.5 * A
                if desc[trough_rel] < mid_thr:
                    ei_rel = trough_rel
                else:
                    # find first crossing below mid-amplitude on descending limb
                    for k in range(1, n_desc):
                        if desc[k] < mid_thr:
                            ei_rel = k
                            break
                    if ei_rel is None:
                        ei_rel = trough_rel

            ei = pi + ei_rel
            ei = min(ei, n_c - 1)
            eos_d = cycle_t[ei]

            # ── 4. Clamp to eos_upper (next trough) ───────────────────
            if eos_upper is not None and eos_d > eos_upper:
                upper_idx = np.searchsorted(
                    [t.value for t in cycle_t], eos_upper.value, side='right') - 1
                upper_idx = max(pi, min(upper_idx, n_c - 1))
                ei    = upper_idx
                eos_d = cycle_t[ei]

            # ── 5. Sanity guards ───────────────────────────────────────
            if eos_d <= sos:
                return None
            if (eos_d - sos).days >= 730:
                eos_d = sos + pd.Timedelta(days=729)
            if ei <= si:
                return None

            return sos, pos, eos_d, ndvi_max, A, base, sos_thr, eos_thr

        # ── Head segment (before first trough) ────────────────
        if trough_indices and _head_start_looks_like_trough:
            ti_first = trough_indices[0]
            head_len = ti_first
            _amp_pre = (float(np.max(sm_for_troughs[0:ti_first + 1])) - float(sm_for_troughs[0]))
            if (head_len >= max(10, min_d // interp_freq) and
                    not _cycle_has_gap(0, ti_first, amplitude=_amp_pre)):
                try:
                    seg_sm  = sm_for_troughs[0:ti_first + 1]
                    seg_t   = t_all[0:ti_first + 1]
                    # Always use smooth for detection — raw can have noise spikes
                    work_arr   = seg_sm
                    ndvi_min_h = float(sm_for_troughs[0])
                    # Peak on smooth
                    pi  = int(np.nanargmax(seg_sm))
                    pos = seg_t[pi]
                    if _is_peak_in_window(pos, sm, em):
                        eos_ub = t_all[ti_first] + pd.Timedelta(days=interp_freq * 2)
                        res = _extract_cycle(work_arr, seg_t, ndvi_min_h,
                                             eos_upper=eos_ub)
                        if res:
                            sos_d, pos_d, eos_d, ndvi_max, A, ndvi_min_r, sos_thr, eos_thr = res
                            # For cross-year windows (e.g. Jun→May), label by peak year
                            # so Sep 2006 peak → year 2006, not the trough year (Jan 2006)
                            if sm > em:
                                trough_year = pos_d.year if pos_d.month >= sm else pos_d.year - 1
                            else:
                                trough_year = seg_t[0].year
                            season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                            rows.append(_make_row(
                                trough_year, season_start, sos_d, pos_d, eos_d,
                                ndvi_max, A, ndvi_min_r, sos_thr, eos_thr,
                                seg_t[int(np.argmin(seg_sm))], sm, em))
                except Exception:
                    pass

        # ── Main cycles (trough[i] → trough[i+1]) ─────────────
        for i in range(len(trough_indices) - 1):
            try:
                ti  = trough_indices[i]
                ti1 = trough_indices[i + 1]
                if ti1 - ti < max(10, min_d // interp_freq): continue
                _amp_pre = (float(np.max(sm_for_troughs[ti:ti1 + 1])) - float(sm_for_troughs[ti]))
                if _cycle_has_gap(ti, ti1, amplitude=_amp_pre): continue
                _has_gap  = np.isnan(sm_vals[ti:ti1 + 1]).any()
                cycle_t   = t_all[ti:ti1 + 1]
                # Always use the smooth series for SOS/EOS detection and peak
                # finding. Raw ndvi_vals can have cloud/noise spikes (e.g. a
                # single 0.255 outlier in Aug 2006) that corrupt thresholds and
                # mislead nanargmax. Smooth gives a clean monotonic green-up.
                cycle_raw = sm_for_troughs[ti:ti1 + 1]
                # Robust trough-based base value
                _trough_v = float(sm_for_troughs[ti])
                ndvi_min  = _trough_v

                # Peak from smooth series
                sm_cycle  = sm_for_troughs[ti:ti1 + 1]
                pi = int(np.nanargmax(sm_cycle))
                pos = cycle_t[pi]
                if not _is_peak_in_window(pos, sm, em):
                    continue

                # ── FIX: EOS upper bound = next trough + small buffer ─
                eos_ub = t_all[ti1] + pd.Timedelta(days=interp_freq * 2)

                res = _extract_cycle(cycle_raw, cycle_t, ndvi_min, eos_upper=eos_ub)
                if res is None:
                    continue
                sos_d, pos_d, eos_d, ndvi_max, A, ndvi_min_r, sos_thr, eos_thr = res

                # For cross-year windows (e.g. Jun→May), label by peak year
                # so Sep 2006 peak → year 2006, not the bounding trough year
                if sm > em:
                    trough_year = pos_d.year if pos_d.month >= sm else pos_d.year - 1
                else:
                    trough_year = t_all[ti].year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                rows.append(_make_row(
                    trough_year, season_start, sos_d, pos_d, eos_d,
                    ndvi_max, A, ndvi_min_r, sos_thr, eos_thr, t_all[ti], sm, em))
            except Exception:
                continue

        # ── Tail segment (after last trough, toward end of data) ─
        covered = set()
        for i in range(len(trough_indices) - 1):
            _a = (float(np.max(sm_for_troughs[trough_indices[i]:trough_indices[i+1]+1])) -
                  float(sm_for_troughs[trough_indices[i]]))
            if not _cycle_has_gap(trough_indices[i], trough_indices[i+1], amplitude=_a):
                covered.add(trough_indices[i])

        for ti0 in [ti for ti in trough_indices if ti not in covered]:
            tail_end = n - 1
            tail_len = tail_end - ti0
            if tail_len < max(10, min_d // interp_freq): continue
            try:
                _a = (float(np.max(sm_for_troughs[ti0:tail_end + 1])) - float(sm_for_troughs[ti0]))
                if _cycle_has_gap(ti0, tail_end, amplitude=_a): continue
                _has_gap = np.isnan(sm_vals[ti0:tail_end + 1]).any()
                seg_t    = t_all[ti0:tail_end + 1]
                seg_sm   = sm_for_troughs[ti0:tail_end + 1]
                # Always use smooth for detection — same reason as main cycles
                seg_raw  = seg_sm
                ndvi_min = float(sm_for_troughs[ti0])

                # Peak from smooth
                pi = int(np.nanargmax(seg_sm))
                pos = seg_t[pi]
                if not _is_peak_in_window(pos, sm, em):
                    continue

                data_end_dt = t_all[-1]

                # ── Detect ascending-only tail (data ends before peak) ─
                last_10 = seg_sm[max(0, len(seg_sm) - 10):]
                tail_still_ascending = (
                    len(last_10) >= 3 and
                    float(np.nanmean(last_10[-3:])) > float(np.nanmean(last_10[:3])))
                peak_at_data_end = (pi >= len(seg_sm) - 3)

                if peak_at_data_end and tail_still_ascending:
                    # Incomplete season: data ends on ascending limb.
                    # Extract SOS only; mark POS/EOS as data-end placeholder.
                    _A_partial = float(seg_raw[pi]) - ndvi_min
                    if _A_partial < MIN_AMPLITUDE:
                        continue
                    _sos_thr_p = ndvi_min + sos_threshold_pct * _A_partial
                    _asc_p = seg_raw[:pi + 1]
                    _below_p = np.where(_asc_p < _sos_thr_p)[0]
                    _si_p = int(_below_p[-1]) + 1 if len(_below_p) > 0 else 0
                    _si_p = min(_si_p, len(seg_t) - 1)
                    sos_d   = seg_t[_si_p]
                    pos_d   = seg_t[pi]
                    eos_d   = data_end_dt
                    ndvi_max = float(seg_raw[pi])
                    A        = _A_partial
                    ndvi_min_r = ndvi_min
                    sos_thr  = _sos_thr_p
                    eos_thr  = ndvi_min + eos_threshold_pct * _A_partial
                    if sm > em:
                        trough_year = sos_d.year if sos_d.month >= sm else sos_d.year - 1
                    else:
                        trough_year = seg_t[0].year
                    season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                    rows.append(_make_row(
                        trough_year, season_start, sos_d, pos_d, eos_d,
                        ndvi_max, A, ndvi_min_r, sos_thr, eos_thr, t_all[ti0], sm, em))
                    continue

                res = _extract_cycle(seg_raw, seg_t, ndvi_min, eos_upper=None)
                if res is None:
                    continue
                sos_d, pos_d, eos_d, ndvi_max, A, ndvi_min_r, sos_thr, eos_thr = res

                # Tail-specific guard: reject EOS stuck at data end when NDVI
                # is still above EOS threshold and NOT clearly declining.
                eos_is_at_data_end = (eos_d >= data_end_dt - pd.Timedelta(days=interp_freq * 2))
                ndvi_at_data_end = (float(ndvi_vals[-1]) if not np.isnan(ndvi_vals[-1])
                                    else float(sm_for_troughs[-1]))
                last_5 = sm_for_troughs[max(0, n - 5):]
                clearly_declining = (
                    float(last_5[-1]) < float(last_5[0]) - 0.01 * A
                ) if len(last_5) >= 2 else False
                if (eos_is_at_data_end and
                        ndvi_at_data_end > eos_thr and
                        not clearly_declining):
                    continue

                # For cross-year windows, label by peak year
                if sm > em:
                    trough_year = pos_d.year if pos_d.month >= sm else pos_d.year - 1
                else:
                    trough_year = seg_t[0].year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                rows.append(_make_row(
                    trough_year, season_start, sos_d, pos_d, eos_d,
                    ndvi_max, A, ndvi_min_r, sos_thr, eos_thr, t_all[ti0], sm, em))
            except Exception:
                pass

        if not rows:
            return None, (
                f"No complete seasons detected. Troughs found: {len(trough_indices)}. "
                f"Data: {ndvi_5d.index.min().date()} → {ndvi_5d.index.max().date()} "
                f"({n} pts, {interp_freq}d grid). "
                f"MIN_AMPLITUDE (data-derived) = {MIN_AMPLITUDE:.3f}. "
                f"Try: reduce Min Days slider, check season window, adjust threshold %."
            )

        df_out = pd.DataFrame(rows)
        # When two rows share the same Year (e.g. head + main loop overlap),
        # keep the one with the largest Amplitude (most complete season).
        if len(df_out) > 0:
            df_out = (df_out
                      .sort_values(['Year', 'Amplitude'], ascending=[True, False])
                      .drop_duplicates(subset='Year', keep='first'))
        return df_out.sort_values("Year").reset_index(drop=True), None

    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"


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
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════

def get_all_correlations(X, y):
    rows = []
    for col in X.columns:
        vals = X[col].dropna()
        if vals.std() < 1e-8 or len(vals) < 2: continue
        idx = vals.index.intersection(y.dropna().index)
        if len(idx) < 2: continue
        try:
            r,   p_val = pearsonr(vals[idx].astype(float), y[idx].astype(float))
            if len(idx) >= 3:
                rho, p_sp = spearmanr(vals[idx].astype(float), y[idx].astype(float))
            else:
                rho, p_sp = float(r), 1.0
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
    n = len(y_vals)
    if n < 2: return 0.0
    if n == 2:
        try:
            r, _ = pearsonr(X_vals[:, 0] if X_vals.ndim > 1 else X_vals, y_vals)
            return float(r ** 2)
        except Exception:
            return 0.0
    loo   = LeaveOneOut()
    preds = []
    sc    = StandardScaler()
    for tr, te in loo.split(X_vals):
        if len(tr) < 1: continue
        try:
            Xtr = sc.fit_transform(X_vals[tr])
            Xte = sc.transform(X_vals[te])
            m   = Ridge(alpha=alpha)
            m.fit(Xtr, y_vals[tr])
            preds.append(float(m.predict(Xte)[0]))
        except Exception:
            preds.append(float(y_vals[tr].mean()))
    if not preds: return 0.0
    preds  = np.array(preds)
    ss_res = np.sum((y_vals - preds) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2) + 1e-12
    return float(np.clip(1 - ss_res / ss_tot, -1, 1))


def select_multi_features(X, y, max_features=5, min_r=MIN_CORR_THRESHOLD,
                           user_max_features=None):
    n_obs = len(y.dropna())
    if n_obs <= 3:   effective_min_r = min(min_r, 0.10)
    elif n_obs <= 5: effective_min_r = min(min_r, 0.25)
    else:            effective_min_r = min_r

    if n_obs <= 10:   collinear_thr = 0.97
    elif n_obs <= 20: collinear_thr = 0.90
    else:             collinear_thr = 0.85

    usable = []
    for col in X.columns:
        vals = X[col].dropna()
        if vals.std() < 1e-8 or len(vals) < 2: continue
        idx = vals.index.intersection(y.dropna().index)
        if len(idx) < 2: continue
        try:
            rp, _ = pearsonr(vals[idx].astype(float), y[idx].astype(float))
            rs = rp
            if len(idx) >= 3:
                rs, _ = spearmanr(vals[idx].astype(float), y[idx].astype(float))
        except Exception:
            continue
        # For small n (<=5): use |Pearson r| only for ranking.
        # Spearman with n=3 gives only ±1.0 or ±0.5 (only 3 rank positions),
        # which inflates many features to composite=1.0 and makes alphabetical
        # tie-breaking decide the winner instead of actual correlation strength.
        if n_obs <= 5:
            composite = abs(rp)
        else:
            composite = max(abs(rp), abs(float(rs)))
        if composite >= effective_min_r:
            usable.append((col, composite))

    if not usable: return []
    usable.sort(key=lambda x: -x[1])

    collinear_filtered = []
    for feat, score in usable:
        collinear = False
        for sel in collinear_filtered:
            xi   = X[feat].fillna(X[feat].median())
            xj   = X[sel].fillna(X[sel].median())
            idx2 = xi.index.intersection(xj.index)
            if len(idx2) < 2: continue
            try:
                r_pair, _ = pearsonr(xi[idx2].astype(float), xj[idx2].astype(float))
            except Exception:
                continue
            if abs(r_pair) > collinear_thr:
                collinear = True
                break
        if not collinear:
            collinear_filtered.append(feat)

    if user_max_features is not None: effective_max = user_max_features
    elif n_obs <= 3:  effective_max = 2   # allow 2 features; Ridge handles collinearity
    elif n_obs <= 5:  effective_max = min(max_features, 3)
    else:             effective_max = max_features

    max_safe   = max(1, n_obs - 1)
    candidates = collinear_filtered[:min(effective_max, max_safe)]
    if len(candidates) <= 1: return candidates

    y_vals   = y.values.astype(float)
    selected = [candidates[0]]
    best_r2  = _loo_r2_quick(
        X[selected].fillna(X[selected[0]].median()).values.reshape(-1, 1), y_vals)
    # For small n, LOO R2 is unreliable so use a very small improvement threshold.
    # This allows a 2nd feature to enter even when LOO improvement is marginal.
    improvement_thr = 0.01 if n_obs <= 5 else 0.03

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
# PREDICTION ENGINE  (v5.3 — unchanged from original)
# ═══════════════════════════════════════════════════════════════

def loo_cv(X_vals, y_vals, model_fn):
    n = len(y_vals)
    if n < 2:
        return np.nan, np.nan
    if n == 2:
        try:
            r, _ = pearsonr(X_vals[:, 0] if X_vals.ndim > 1 else X_vals.ravel(), y_vals)
            return float(r ** 2), float(np.mean(np.abs(y_vals - y_vals.mean())))
        except Exception:
            return 0.0, float(np.mean(np.abs(y_vals - y_vals.mean())))
    preds = []
    for i in range(n):
        idx_train = [j for j in range(n) if j != i]
        Xt, yt = X_vals[idx_train], y_vals[idx_train]
        Xv     = X_vals[[i]]
        try:
            m = model_fn()
            m.fit(Xt, yt)
            preds.append(float(m.predict(Xv)[0]))
        except Exception:
            preds.append(float(np.mean(yt)))
    preds  = np.array(preds)
    ss_res = np.sum((y_vals - preds) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2) + 1e-12
    r2  = float(np.clip(1 - ss_res / ss_tot, -1, 1))
    mae = float(np.mean(np.abs(y_vals - preds)))
    return r2, mae


def fit_all_models(X_all, y, preferred_key="ridge", user_max_features=None):
    yt = y.values.astype(float)
    n  = len(yt)

    adaptive_min_r = MIN_CORR_THRESHOLD
    if n <= 4:  adaptive_min_r = max(0.25, MIN_CORR_THRESHOLD - 0.15)
    elif n <= 6: adaptive_min_r = max(0.30, MIN_CORR_THRESHOLD - 0.10)

    features = select_multi_features(X_all, y, max_features=5,
                                     min_r=adaptive_min_r,
                                     user_max_features=user_max_features)

    if not features:
        md = float(yt.mean())
        mean_fit = {
            'mode': 'mean', 'features': [], 'r2': 0.0,
            'mae': float(np.mean(np.abs(yt - md))),
            'alpha': None, 'coef': [], 'intercept': md,
            'best_r': 0.0, 'mean_doy': md, 'n': n,
            'pipe': None, 'model_key': 'mean',
        }
        return {
            'best_name': 'mean', 'best_fit': mean_fit,
            'all_models': {'mean': mean_fit},
            'features': [], 'r2': 0.0, 'mae': mean_fit['mae'], 'n': n,
        }

    Xf = X_all[features].fillna(X_all[features].median())
    Xv = Xf.values
    sc = StandardScaler()
    Xs = sc.fit_transform(Xv)

    best_single_r = 0.0
    for f in features:
        try:
            r_val, _ = pearsonr(Xf[f].astype(float), y.astype(float))
            if abs(r_val) > best_single_r:
                best_single_r = abs(r_val)
        except Exception:
            pass

    all_models = {}

    # Ridge
    try:
        rcv = RidgeCV(alphas=np.logspace(-3, 4, 30), cv=None)
        rcv.fit(Xs, yt)
        best_alpha = float(rcv.alpha_)
        pipe_ridge = Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=best_alpha))])
        pipe_ridge.fit(Xv, yt)
        _sc    = pipe_ridge.named_steps['sc']
        _ridge = pipe_ridge.named_steps['r']
        coef_unstd = list(_ridge.coef_ / _sc.scale_)
        intercept_unstd = float(_ridge.intercept_ - np.dot(_ridge.coef_ / _sc.scale_, _sc.mean_))
        r2_ridge, mae_ridge = loo_cv(
            Xs, yt,
            lambda a=best_alpha: Pipeline([('sc2', StandardScaler()),
                                           ('r2', Ridge(alpha=a))]))
        all_models['Ridge'] = {
            'mode': 'ridge', 'features': features, 'r2': r2_ridge, 'mae': mae_ridge,
            'alpha': best_alpha, 'coef': coef_unstd, 'intercept': intercept_unstd,
            'best_r': best_single_r, 'n': n, 'pipe': pipe_ridge,
            'scaler': sc, 'model_key': 'ridge',
        }
    except Exception:
        pass

    # LOESS
    try:
        if _LOESS_AVAILABLE:
            from sklearn.decomposition import PCA as _PCA
            if Xs.shape[1] == 1:
                Xloess = Xs[:, 0]; _loess_pca = None
            else:
                _pca = _PCA(n_components=1)
                Xloess = _pca.fit_transform(Xs)[:, 0]; _loess_pca = _pca
            frac_val = min(0.75, max(0.25, 6.0 / max(n, 1)))
            loess_preds = []
            for i in range(n):
                idx_tr = [j for j in range(n) if j != i]
                x_tr, y_tr = Xloess[idx_tr], yt[idx_tr]
                x_val = Xloess[i]
                try:
                    sm_arr = _sm_lowess(y_tr, x_tr, frac=frac_val, return_sorted=True)
                    sm_sorted = sm_arr[np.argsort(sm_arr[:, 0])]
                    f_interp = _scipy_interp1d(sm_sorted[:, 0], sm_sorted[:, 1],
                                               bounds_error=False,
                                               fill_value=(sm_sorted[0, 1], sm_sorted[-1, 1]))
                    loess_preds.append(float(f_interp(x_val)))
                except Exception:
                    loess_preds.append(float(np.mean(y_tr)))
            lp     = np.array(loess_preds)
            ss_res = np.sum((yt - lp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
            r2l    = float(np.clip(1 - ss_res / ss_tot, -1, 1))
            mael   = float(np.mean(np.abs(yt - lp)))

            class _LoessWrapper:
                def __init__(self, x_train, y_train, frac, pca_obj, scaler_obj):
                    self.x_train = x_train; self.y_train = y_train
                    self.frac = frac; self.pca_obj = pca_obj; self.sc = scaler_obj
                def predict(self, X_new):
                    Xs_new = self.sc.transform(X_new)
                    xn = (self.pca_obj.transform(Xs_new)[:, 0]
                          if self.pca_obj is not None else Xs_new[:, 0])
                    sm_arr = _sm_lowess(self.y_train, self.x_train, frac=self.frac, return_sorted=True)
                    sm_sorted = sm_arr[np.argsort(sm_arr[:, 0])]
                    f_i = _scipy_interp1d(sm_sorted[:, 0], sm_sorted[:, 1],
                                          bounds_error=False,
                                          fill_value=(sm_sorted[0, 1], sm_sorted[-1, 1]))
                    return f_i(xn)

            all_models['LOESS'] = {
                'mode': 'loess', 'features': features, 'r2': r2l, 'mae': mael,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'n': n,
                'pipe': _LoessWrapper(Xloess, yt, frac_val, _loess_pca, sc),
                'scaler': None, 'model_key': 'loess',
                'x_train': Xloess, 'y_train': yt,
            }
        else:
            feat1 = features[0]
            x1d   = Xf[feat1].values.astype(float)
            x1d_sc = (x1d - x1d.mean()) / (x1d.std() + 1e-9)
            loess_preds_fb = []
            for i in range(n):
                mask = np.ones(n, dtype=bool); mask[i] = False
                loess_preds_fb.append(float(_loess_predict_fallback(
                    x1d_sc[mask], yt[mask], np.array([x1d_sc[i]]), frac=0.75)[0]))
            lp_fb  = np.array(loess_preds_fb)
            ss_res = np.sum((yt - lp_fb) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
            r2l  = float(np.clip(1 - ss_res / ss_tot, -1, 1))
            mael = float(np.mean(np.abs(yt - lp_fb)))
            all_models['LOESS'] = {
                'mode': 'loess', 'features': [feat1], 'r2': r2l, 'mae': mael,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'n': n, 'pipe': None, 'scaler': None,
                'model_key': 'loess', 'x_train': x1d_sc, 'y_train': yt,
            }
    except Exception:
        pass

    for deg in [2, 3]:
        if n <= deg + 2: continue
        try:
            poly_pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('sc2',  StandardScaler()),
                ('r',    RidgeCV(alphas=np.logspace(-3, 4, 20))),
            ])
            poly_pipe.fit(Xv, yt)
            r2p, maep = loo_cv(Xv, yt,
                lambda d=deg: Pipeline([
                    ('poly', PolynomialFeatures(degree=d, include_bias=False)),
                    ('sc2',  StandardScaler()),
                    ('r',    RidgeCV(alphas=np.logspace(-3, 4, 20))),
                ]))
            mname = f'Poly-{deg}'
            all_models[mname] = {
                'mode': f'poly{deg}', 'features': features, 'r2': r2p, 'mae': maep,
                'alpha': 1.0, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'n': n,
                'pipe': poly_pipe, 'scaler': None, 'model_key': f'poly{deg}',
            }
        except Exception:
            pass

    if n >= 5:
        try:
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
            gpr.fit(Xs, yt)
            r2g, maeg = loo_cv(Xs, yt,
                lambda: GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                    normalize_y=True, random_state=42))
            all_models['GPR'] = {
                'mode': 'gpr', 'features': features, 'r2': r2g, 'mae': maeg,
                'alpha': None, 'coef': [], 'intercept': 0.0,
                'best_r': best_single_r, 'n': n,
                'pipe': None, 'scaler': sc,
                'gpr_model': gpr, 'gpr_scaler': sc, 'model_key': 'gpr',
            }
        except Exception:
            pass

    if not all_models:
        md = float(yt.mean())
        mean_fit = {
            'mode': 'mean', 'features': [], 'r2': 0.0,
            'mae': float(np.mean(np.abs(yt - md))),
            'alpha': None, 'coef': [], 'intercept': md,
            'best_r': 0.0, 'mean_doy': md, 'n': n,
            'pipe': None, 'model_key': 'mean',
        }
        return {
            'best_name': 'mean', 'best_fit': mean_fit,
            'all_models': {'mean': mean_fit},
            'features': [], 'r2': 0.0, 'mae': mean_fit['mae'], 'n': n,
        }

    _key_to_name = {
        'ridge': 'Ridge', 'loess': 'LOESS',
        'poly2': 'Poly-2', 'poly3': 'Poly-3', 'gpr': 'GPR',
    }
    preferred_name = _key_to_name.get(preferred_key, 'Ridge')

    def _model_r2(name):
        r2 = all_models[name].get('r2', np.nan)
        return r2 if not np.isnan(r2) else -999.0

    best_name_auto = max(all_models, key=_model_r2)
    best_r2_auto   = _model_r2(best_name_auto)
    if (preferred_name in all_models and
            abs(_model_r2(preferred_name) - best_r2_auto) <= 0.02):
        best_name = preferred_name
    else:
        best_name = best_name_auto

    best_fit = all_models[best_name]
    return {
        'best_name':  best_name,
        'best_fit':   best_fit,
        'all_models': all_models,
        'features':   features,
        'r2':         best_fit['r2'],
        'mae':        best_fit['mae'],
        'n':          n,
    }


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL PREDICTOR
# ═══════════════════════════════════════════════════════════════

class UniversalPredictor:
    def __init__(self):
        self._fits       = {}
        self.r2          = {}
        self.mae         = {}
        self.n_seasons   = {}
        self.corr_tables = {}

    def train(self, train_df, all_params, model_key="ridge", user_max_features=None):
        if train_df is None or train_df.empty or 'Event' not in train_df.columns:
            return
        meta      = {'Year', 'Event', 'Target_DOY', 'LOS_Days', 'Peak_NDVI', 'Season_Start'}
        feat_cols = [c for c in train_df.columns
                     if c not in meta
                     and pd.api.types.is_numeric_dtype(train_df[c])
                     and train_df[c].std() > 1e-8]
        for event in ['SOS', 'POS', 'EOS']:
            sub = train_df[train_df['Event'] == event].copy()
            self.n_seasons[event] = len(sub)
            if len(sub) < 2: continue
            X   = sub[feat_cols].fillna(sub[feat_cols].median())
            y   = sub['Target_DOY']
            self.corr_tables[event] = get_all_correlations(X, y)
            result = fit_all_models(X, y, preferred_key=model_key,
                                    user_max_features=user_max_features)
            self._fits[event]  = result
            self.r2[event]     = result['r2']
            self.mae[event]    = result['mae']

    def predict(self, inputs, event, year=2026, season_start_month=6):
        if event not in self._fits: return None
        result   = self._fits[event]
        best_fit = result['best_fit']
        features = result['features']

        if best_fit['mode'] == 'mean':
            rel_days = int(round(best_fit.get('mean_doy', best_fit.get('intercept', 0))))
        elif best_fit['mode'] == 'loess':
            pipe = best_fit.get('pipe')
            if pipe is not None and hasattr(pipe, 'predict'):
                X_new = np.array([[inputs.get(f, 0.0) for f in features]])
                pred  = pipe.predict(X_new)
                rel_days = int(np.clip(round(float(pred[0])), 0, 500))
            else:
                x_train = best_fit.get('x_train'); y_train = best_fit.get('y_train')
                feat1   = features[0] if features else None
                if x_train is not None and y_train is not None and feat1:
                    x_val    = float(inputs.get(feat1, 0.0))
                    x_val_sc = (x_val - x_train.mean()) / (x_train.std() + 1e-9)
                    pred     = _loess_predict_fallback(x_train, y_train, np.array([x_val_sc]), frac=0.75)[0]
                    rel_days = int(np.clip(round(float(pred)), 0, 500))
                else:
                    rel_days = int(round(float(np.mean(best_fit.get('y_train', [0])))))
        elif best_fit['mode'] == 'gpr':
            gpr = best_fit.get('gpr_model'); gpr_sc = best_fit.get('gpr_scaler')
            if gpr is not None and gpr_sc is not None:
                X_new = np.array([[inputs.get(f, 0.0) for f in features]])
                pred  = gpr.predict(gpr_sc.transform(X_new))[0]
                rel_days = int(np.clip(round(float(pred)), 0, 500))
            else:
                rel_days = int(round(self.mae.get(event, 0)))
        else:
            pipe = best_fit.get('pipe')
            if pipe is not None:
                X_new = np.array([[inputs.get(f, 0.0) for f in features]])
                pred  = pipe.predict(X_new)[0]
                rel_days = int(np.clip(round(float(pred)), 0, 500))
            else:
                rel_days = int(round(self.r2.get(event, 0)))

        season_start = datetime(year, season_start_month, 1)
        date = season_start + timedelta(days=rel_days)
        doy  = date.timetuple().tm_yday
        return {
            'doy': doy, 'date': date, 'rel_days': rel_days,
            'r2': self.r2[event], 'mae': self.mae[event],
            'event': event, 'model': result['best_name'],
            'all_r2': {k: v.get('r2', np.nan) for k, v in result['all_models'].items()},
        }

    def equation_str(self, event, season_start_month=6):
        n_ev = self.n_seasons.get(event, 0)
        if event not in self._fits:
            return f"Need ≥ 2 seasons with met data to fit a model (currently {n_ev})"
        result   = self._fits[event]
        best_fit = result['best_fit']
        best_name = result['best_name']
        all_models = result['all_models']
        mo  = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        lbl = f"{event}_days_from_{mo.get(season_start_month,'Jan')}1"
        model_scores = "  ·  ".join(
            f"{nm}: R²={v.get('r2', np.nan):.3f}"
            for nm, v in all_models.items()
            if not np.isnan(v.get('r2', np.nan))
        )
        if best_fit['mode'] == 'mean':
            return (f"{lbl} ≈ {best_fit.get('mean_doy', best_fit.get('intercept', 0)):.0f}  "
                    f"[No feature |r|≥{MIN_CORR_THRESHOLD} — mean only]")
        if best_fit['mode'] == 'gpr':
            return (f"{lbl}  =  GPR({', '.join(result['features'])})\n"
                    f"    [Best model: GPR  ·  R²(LOO)={best_fit['r2']:.3f}  ·  MAE=±{best_fit['mae']:.1f} d]\n"
                    f"    All models — {model_scores}")
        if best_fit['mode'] == 'loess':
            feat_str = ', '.join(result['features']) if result['features'] else '—'
            return (f"{lbl}  =  LOESS({feat_str})\n"
                    f"    [Best model: LOESS  ·  R²(LOO)={best_fit['r2']:.3f}  ·  MAE=±{best_fit['mae']:.1f} d]\n"
                    f"    All models — {model_scores}")
        if best_fit['mode'] in ('poly2', 'poly3'):
            deg = 2 if best_fit['mode'] == 'poly2' else 3
            return (f"{lbl}  =  Polynomial-{deg}({', '.join(result['features'])})\n"
                    f"    [Best model: Poly-{deg}  ·  R²(LOO)={best_fit['r2']:.3f}  ·  MAE=±{best_fit['mae']:.1f} d]\n"
                    f"    All models — {model_scores}")
        terms = [f"{best_fit.get('intercept', 0.0):.3f}"]
        for feat, coef in zip(result['features'], best_fit.get('coef', [])):
            s = '+' if coef >= 0 else '-'
            terms.append(f"{s} {abs(coef):.5f} × {feat}")
        return (f"{lbl}  =  " + "  ".join(terms) +
                f"\n    [Best model: Ridge  ·  α={best_fit.get('alpha', '?')}  ·  "
                f"{len(result['features'])} feature(s)  ·  "
                f"R²(LOO)={best_fit['r2']:.3f}  ·  MAE=±{best_fit['mae']:.1f} d]\n"
                f"    All models — {model_scores}")

    def corr_table_for_display(self, event):
        if event not in self._fits: return pd.DataFrame()
        result   = self._fits[event]
        best_fit = result['best_fit']
        ct       = self.corr_tables.get(event)
        if ct is None or len(ct) == 0: return pd.DataFrame()
        in_model = set(result['features'])
        selected_first = result['features'][0] if result['features'] else None
        rows = []
        for _, row in ct.iterrows():
            feat   = row['Feature']
            usable = row['Usable'] == '✅'
            if feat in in_model:
                role = '✅  In model'
            elif usable:
                is_collinear = False; collinear_with = None
                if selected_first and selected_first in ct['Feature'].values:
                    try:
                        sel_r  = ct[ct['Feature'] == selected_first]['Pearson_r'].values[0]
                        feat_r = row['Pearson_r']
                        if (abs(feat_r) > 0.85 and abs(sel_r) > 0.85 and
                                (feat_r * sel_r > 0 or abs(abs(feat_r) - abs(sel_r)) < 0.15)):
                            is_collinear = True; collinear_with = selected_first
                    except Exception:
                        pass
                role = (f'➖  Redundant — highly similar to {collinear_with}'
                        if is_collinear and collinear_with
                        else '➖  Did not improve model accuracy — not added')
            else:
                role = '⬜  Below correlation threshold'
            rows.append({'Feature': feat, 'Pearson r': row['Pearson_r'],
                         'Spearman ρ': row.get('Spearman_rho', float('nan')),
                         'Composite': row.get('Composite', row['|r|']), 'Role': role})
        return pd.DataFrame(rows)

    def export_coefficients(self, season_start_month=6):
        records = []
        for event in ['SOS', 'POS', 'EOS']:
            if event not in self._fits: continue
            result   = self._fits[event]
            best_fit = result['best_fit']
            best_name = result['best_name']
            if best_fit['mode'] == 'mean':
                records.append({'Event': event, 'Feature': 'INTERCEPT',
                                'Coefficient': best_fit.get('mean_doy', best_fit.get('intercept', 0)),
                                'Model': 'mean', 'R2_LOO': 0.0, 'MAE_days': round(best_fit['mae'], 2)})
            elif best_fit['mode'] == 'ridge' and best_fit.get('coef'):
                for feat, coef in zip(result['features'], best_fit['coef']):
                    records.append({'Event': event, 'Feature': feat,
                                    'Coefficient': round(coef, 6),
                                    'Model': f'Ridge (best of all)',
                                    'Alpha': best_fit.get('alpha'),
                                    'R2_LOO': round(best_fit['r2'], 4),
                                    'MAE_days': round(best_fit['mae'], 2)})
                records.append({'Event': event, 'Feature': 'INTERCEPT',
                                'Coefficient': round(best_fit.get('intercept', 0), 4),
                                'Model': f'Ridge (best of all)',
                                'Alpha': best_fit.get('alpha'),
                                'R2_LOO': round(best_fit['r2'], 4),
                                'MAE_days': round(best_fit['mae'], 2)})
            else:
                records.append({'Event': event, 'Feature': str(result['features']),
                                'Coefficient': float('nan'),
                                'Model': f'{best_name} (best of all)',
                                'R2_LOO': round(best_fit['r2'], 4),
                                'MAE_days': round(best_fit['mae'], 2)})
            for mname, mfit in result['all_models'].items():
                if mname == best_name: continue
                records.append({'Event': event, 'Feature': '(comparison)',
                                'Coefficient': float('nan'),
                                'Model': f'{mname}',
                                'R2_LOO': round(mfit.get('r2', np.nan), 4) if not np.isnan(mfit.get('r2', np.nan)) else float('nan'),
                                'MAE_days': round(mfit.get('mae', np.nan), 2) if not np.isnan(mfit.get('mae', np.nan)) else float('nan')})
        return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compute_sensitivity_analysis(predictor, train_df):
    result    = {}
    dominants = {}
    feat_stds = {}
    meta_cols = {'Year', 'Event', 'Target_DOY', 'LOS_Days', 'Peak_NDVI', 'Season_Start'}
    feat_cols = [c for c in train_df.columns if c not in meta_cols
                 and pd.api.types.is_numeric_dtype(train_df[c])]
    for f in feat_cols:
        vals = train_df[f].dropna()
        feat_stds[f] = float(vals.std()) if len(vals) > 1 else 1.0
    for ev in ['SOS', 'POS', 'EOS']:
        if ev not in predictor._fits: continue
        result_ev = predictor._fits[ev]
        best_fit  = result_ev['best_fit']
        if best_fit['mode'] != 'ridge' or not best_fit.get('coef') or not result_ev.get('features'):
            continue
        sub      = train_df[train_df['Event'] == ev]['Target_DOY'].dropna()
        mean_tgt = float(sub.mean()) if len(sub) > 0 else 1.0
        ev_result = {}
        for feat, coef in zip(result_ev['features'], best_fit['coef']):
            std_f       = feat_stds.get(feat, 1.0)
            days_shift  = coef * std_f
            pct_of_mean = (days_shift / max(abs(mean_tgt), 1)) * 100
            ev_result[feat] = {
                'days_per_std': round(days_shift, 1),
                'pct_of_mean':  round(pct_of_mean, 2),
                'direction':    'delays' if days_shift > 0 else 'advances',
                'coef':         round(coef, 5),
                'std':          round(std_f, 3),
            }
        result[ev] = ev_result
        if ev_result:
            dom = max(ev_result, key=lambda f: abs(ev_result[f]['days_per_std']))
            dominants[ev] = {
                'feature':      dom,
                'days_per_std': ev_result[dom]['days_per_std'],
                'direction':    ev_result[dom]['direction'],
            }
    return result, dominants


# ═══════════════════════════════════════════════════════════════
# PLOTS — SPLIT NDVI PLOT FOR LONG DATASETS
# ═══════════════════════════════════════════════════════════════

SPLIT_PLOT_THRESHOLD_YEARS = 8   # split into chunks if data spans more than this


def _build_ndvi_interpolated(ndvi_raw, interp_freq=INTERP_STEP_DAYS):
    """Re-create the 5-day smoothed NDVI series used in extraction."""
    ndvi_s = ndvi_raw.set_index('Date')['NDVI'].sort_index()
    if ndvi_s.index.duplicated().any():
        ndvi_s = ndvi_s.groupby(ndvi_s.index).mean()
    orig_dates  = ndvi_s.index.sort_values()
    orig_diffs  = pd.Series(orig_dates).diff().dt.days.fillna(0)
    pos_diffs   = orig_diffs[orig_diffs > 0]
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
    ndvi_5d_grid = ndvi_5d.reindex(full_range)

    n = len(ndvi_5d_grid); vals = ndvi_5d_grid.values.copy()
    valid_mask = ~np.isnan(vals)
    sm_arr = np.full(n, np.nan)
    seg_labels = np.zeros(n, dtype=int); seg_id, in_seg = 0, False
    for i in range(n):
        if valid_mask[i]:
            if not in_seg: seg_id += 1; in_seg = True
            seg_labels[i] = seg_id
        else:
            in_seg = False
    # SG smooth
    for sid in range(1, seg_id + 1):
        idx_seg = np.where(seg_labels == sid)[0]; seg_n = len(idx_seg)
        if seg_n < 5: sm_arr[idx_seg] = vals[idx_seg]; continue
        wl_t = max(7, min(int(seg_n * 0.05), 31))
        wl_s = wl_t if wl_t % 2 == 1 else wl_t + 1
        wl_s = min(wl_s, seg_n - 1 if seg_n > 1 else 1)
        if wl_s % 2 == 0: wl_s = max(7, wl_s - 1)
        poly_s = min(2, wl_s - 1)
        if wl_s >= 5 and wl_s < seg_n:
            sm_arr[idx_seg] = savgol_filter(vals[idx_seg], wl_s, poly_s)
        else:
            sm_arr[idx_seg] = vals[idx_seg]

    return ndvi_5d_grid, pd.Series(sm_arr, index=full_range), full_range


def _draw_ndvi_axes(ax, ndvi_raw, ndvi_5d_grid, sm_series, full_range,
                    pheno_df, season_window, date_start, date_end):
    """
    Draw all NDVI phenology content onto ax, clipped to [date_start, date_end].
    """
    raw_mask = (pd.to_datetime(ndvi_raw['Date']) >= date_start) & \
               (pd.to_datetime(ndvi_raw['Date']) <= date_end)
    dates_raw = pd.to_datetime(ndvi_raw['Date'])[raw_mask]
    ndvi_raw_vals = ndvi_raw['NDVI'][raw_mask]
    ax.scatter(dates_raw, ndvi_raw_vals, color='#A5D6A7', s=18, alpha=0.55,
               label='NDVI (raw obs)', zorder=3)

    sm_clip = sm_series[(full_range >= date_start) & (full_range <= date_end)]
    ax.plot(sm_clip.index, sm_clip.values, color='#1B5E20', lw=2.2,
            label='Smoothed (SG, 5-day grid)', zorder=5)

    sm_vals_clip = sm_clip.values
    idx_clip = sm_clip.index
    in_gap = False; gap_s = None
    for i in range(len(sm_vals_clip)):
        nan_now = np.isnan(sm_vals_clip[i])
        if nan_now and not in_gap: gap_s = idx_clip[i]; in_gap = True
        elif not nan_now and in_gap:
            ax.axvspan(gap_s, idx_clip[i], color='#BDBDBD', alpha=0.30, label='Data gap')
            in_gap = False
    if in_gap: ax.axvspan(gap_s, idx_clip[-1], color='#BDBDBD', alpha=0.30)

    if season_window:
        ws_m, we_m = season_window
        y_min_yr = date_start.year; y_max_yr = date_end.year + 1
        plotted = False
        for yr in range(y_min_yr, y_max_yr + 1):
            try:
                ws = pd.Timestamp(f"{yr}-{ws_m:02d}-01")
                we = (pd.Timestamp(f"{yr+1}-{we_m:02d}-28") if ws_m > we_m
                      else pd.Timestamp(f"{yr}-{we_m:02d}-28"))
                if we < date_start or ws > date_end: continue
                lbl = 'Selected season window' if not plotted else ''
                ax.axvspan(max(ws, date_start), min(we, date_end),
                           color='#A5D6A7', alpha=0.12, zorder=0, label=lbl)
                plotted = True
            except Exception:
                pass

    thr_sos_p = thr_eos_p = base_p = False
    for _, row in pheno_df.iterrows():
        td = row.get('Trough_Date'); ed = row.get('EOS_Date')
        base = row.get('Base_NDVI'); thr_s = row.get('Threshold_SOS')
        thr_e = row.get('Threshold_EOS'); pk = row.get('Peak_NDVI')
        amp = row.get('Amplitude'); sd = row.get('SOS_Date'); pd_ = row.get('POS_Date')
        if pd.isna(td) or pd.isna(ed): continue
        seg_st = pd.Timestamp(td); seg_en = pd.Timestamp(ed) + pd.Timedelta(days=20)
        if seg_en < date_start or seg_st > date_end: continue
        seg_st = max(seg_st, date_start); seg_en = min(seg_en, date_end)
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
            if date_start <= px <= date_end:
                ax.annotate('', xy=(px, pk), xytext=(px, base),
                            arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=1.1))
                ax.text(px, base + amp * 0.5, f'  A={amp:.3f}', fontsize=7, color='#7B1FA2',
                        va='center', ha='left')

    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#C62828'}
    plotted_ev = set()
    for _, row in pheno_df.iterrows():
        for ev, col in ev_colors.items():
            d = row.get(f'{ev}_Date')
            if pd.notna(d) and date_start <= pd.Timestamp(d) <= date_end:
                ax.axvline(d, color=col, lw=1.4, alpha=0.55, ls='--',
                           label=f'{ev}' if ev not in plotted_ev else '')
                plotted_ev.add(ev)

    ax.set_ylabel('NDVI'); ax.set_facecolor('#F7FBF8')
    # Adapt tick density and clamp x-axis so no empty future years are shown.
    ax.set_xlim(date_start, date_end)
    _span_days = max((date_end - date_start).days, 1)
    _span_yrs  = _span_days / 365.25
    _tick_interval = 3 if _span_yrs <= 4 else 6
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=_tick_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.legend(ncol=4, fontsize=7, loc='upper left', framealpha=0.88)
    ax.grid(True, alpha=0.18, color='#C8E6C9', ls='--')


def plot_ndvi_phenology(ndvi_raw, pheno_df, season_window=None,
                        interp_freq=INTERP_STEP_DAYS,
                        split_threshold_years=SPLIT_PLOT_THRESHOLD_YEARS):
    """
    Plot NDVI phenology.
    If the dataset spans > split_threshold_years years, automatically split
    into multiple sub-figures of ≤ split_threshold_years years each.
    Returns a list of (label, fig) tuples. Single figure → list of length 1.
    """
    ndvi_5d_grid, sm_series, full_range = _build_ndvi_interpolated(ndvi_raw, interp_freq)

    dates_all = pd.to_datetime(ndvi_raw['Date'])
    year_min   = dates_all.dt.year.min()
    year_max   = dates_all.dt.year.max()
    n_years    = year_max - year_min + 1

    if n_years <= split_threshold_years:
        chunks = [(year_min, year_max)]
    else:
        chunk_size = split_threshold_years
        chunks = []
        y = year_min
        while y <= year_max:
            chunks.append((y, min(y + chunk_size - 1, year_max)))
            y += chunk_size

    figs = []
    mo = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
          7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    win_str = (f"  |  Window: {mo.get(season_window[0], '?')} → {mo.get(season_window[1], '?')}"
               if season_window else '')

    for (y_start, y_end) in chunks:
        date_start = pd.Timestamp(f"{y_start}-01-01")
        # Bleed window: enough to show EOS of the last season in this panel.
        # For cross-year windows (e.g. Apr→Mar or Jun→May), EOS can fall up to
        # 15 months after the panel's last year starts — use 16 months of bleed.
        if season_window and season_window[0] > season_window[1]:
            # Cross-year window: need up to ~15 months of bleed
            bleed_months = 16
        else:
            bleed_months = 6
        # Cap date_end so x-axis never extends past the actual data end.
        # The old formula used year_max+1 + bleed_months which pushed the axis
        # far into empty future years for datasets ending mid-year (e.g. Mar 2026).
        actual_data_end = pd.Timestamp(dates_all.max())
        date_end = pd.Timestamp(f"{y_end}-12-31") + pd.DateOffset(months=bleed_months)
        date_end = min(date_end, actual_data_end + pd.DateOffset(months=2))

        fig, ax = plt.subplots(figsize=(14, 4.8))
        _draw_ndvi_axes(ax, ndvi_raw, ndvi_5d_grid, sm_series, full_range,
                        pheno_df, season_window, date_start, date_end)

        lbl_range = f"{y_start}" if y_start == y_end else f"{y_start}–{y_end}"
        ax.set_title(
            f'NDVI Time Series — Phenology  ({lbl_range}){win_str}\n'
            f'5-day interpolation grid · Amplitude thresholds calibrated per-cycle from data',
            fontsize=10, fontweight='bold', color='#0D2016')
        ax.set_xlabel('Date')
        fig.tight_layout()
        figs.append((lbl_range, fig))

    return figs


def plot_pheno_trends(pheno_df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    fig.patch.set_facecolor('#F7FBF8')
    ev_cfg = [
        ('SOS', 'SOS_Date', 'SOS_DOY',    '#2E7D32', 'SOS — Green-up start'),
        ('POS', 'POS_Date', 'POS_DOY',    '#1565C0', 'POS — Peak greenness'),
        ('EOS', 'EOS_Date', 'EOS_Target', '#C62828', 'EOS — Senescence end'),
        ('LOS', None,       'LOS_Days',   '#5D4037', 'LOS — Season length (days)'),
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
            ax.plot(yrs, m * yrs + b, '--', color='#0D2016', lw=1.8, label=f'Trend: {m:+.1f} d/yr')
            ax.legend(fontsize=8.5, framealpha=0.85)
        ax.set_title(lbl, fontsize=10, fontweight='bold', color='#0D2016')
        ax.set_xlabel('Year', fontsize=9)
        ax.grid(True, alpha=0.18, color='#C8E6C9', ls='--'); ax.set_facecolor('#F7FBF8'); ax.tick_params(labelsize=8.5)
    fig.suptitle('Phenological Trends — Data-Derived from Uploaded NDVI',
                 fontsize=13, fontweight='bold', color='#0D2016', y=1.02)
    fig.tight_layout()
    return fig


def plot_obs_vs_pred(predictor, train_df):
    events = []
    for ev in ['SOS', 'POS', 'EOS']:
        if ev not in predictor._fits: continue
        result   = predictor._fits[ev]
        best_fit = result['best_fit']
        if best_fit.get('pipe') is not None and best_fit['mode'] in ('ridge', 'poly2', 'poly3'):
            events.append(ev)
    if not events: return None
    fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 4.5), squeeze=False)
    clrs = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#C62828'}
    for ax, ev in zip(axes[0], events):
        result   = predictor._fits[ev]
        best_fit = result['best_fit']
        feats    = result['features']
        sub      = train_df[train_df['Event'] == ev].copy()
        avail    = [f for f in feats if f in sub.columns]
        if not avail: continue
        Xf = sub[avail].fillna(sub[avail].median())
        try: pred = best_fit['pipe'].predict(Xf.values)
        except Exception: continue
        obs  = sub['Target_DOY'].values
        ax.scatter(obs, pred, color=clrs[ev], s=80, edgecolors='white', lw=1.5, zorder=3, alpha=0.9)
        lims = [min(obs.min(), pred.min()) - 8, max(obs.max(), pred.max()) + 8]
        ax.plot(lims, lims, 'k--', lw=1.2); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f'{ev}  [{result["best_name"]}]  R²(LOO)={predictor.r2.get(ev, 0):.3f}  '
                     f'MAE={predictor.mae.get(ev, 0):.1f} d\n{" + ".join(avail)}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Observed (days from season start)')
        ax.set_ylabel('Predicted (days from season start)')
        ax.grid(True, alpha=0.18, color='#C8E6C9'); ax.set_facecolor('#F7FBF8')
    fig.suptitle('Observed vs Predicted (best auto-selected model per event)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_correlation_summary(predictor):
    events    = ['SOS', 'POS', 'EOS']
    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#C62828'}
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
    fig.patch.set_facecolor('#F7FBF8')
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.1], wspace=0.36)
    ax_bar = fig.add_subplot(gs[0])
    bar_h = 0.22; y_pos = np.arange(n_feats); offs = np.array([-1, 0, 1]) * bar_h
    for i, ev in enumerate(events):
        vals     = r_mat[ev].values
        bar_clrs = [ev_colors[ev] if abs(v) >= MIN_CORR_THRESHOLD else '#CFCFCF' for v in vals]
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
                 fontsize=13, fontweight='bold', color='#0D2016', y=1.005)
    fig.tight_layout()
    return fig


def plot_data_summary(ndvi_info, met_info):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#F7FBF8')
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
    ax.set_xlabel('NDVI'); ax.legend(fontsize=8.5); ax.set_facecolor('#F7FBF8')
    ax.grid(True, alpha=0.18, color='#C8E6C9')
    ax2 = axes[1]
    if met_info:
        top_params = sorted(met_info.keys(), key=lambda p: abs(met_info[p]['mean']), reverse=True)[:10]
        means = [met_info[p]['mean'] for p in top_params]
        stds  = [met_info[p]['std']  for p in top_params]
        y_pos = np.arange(len(top_params))
        ax2.barh(y_pos, means, xerr=stds, color='#1565C0', alpha=0.72, ecolor='#0D47A1',
                 capsize=3, edgecolor='white')
        ax2.set_yticks(y_pos); ax2.set_yticklabels(top_params, fontsize=9)
        ax2.set_title('Met Parameters\n(mean ± std from uploaded data)',
                      fontsize=10, fontweight='bold')
        ax2.set_xlabel('Value'); ax2.set_facecolor('#FAFFF8')
        ax2.grid(True, alpha=0.20, axis='x')
    else:
        ax2.text(0.5, 0.5, 'No met parameters', ha='center', va='center', transform=ax2.transAxes)
    ax3 = axes[2]; ax3.axis('off')
    stats_text = (
        f"NDVI Data Summary\n{'─'*30}\n"
        f"Observations:   {ndvi_info['n_obs']}\n"
        f"Years covered:  {ndvi_info['year_range']}\n"
        f"Cadence:        {ndvi_info['cadence_d']:.1f} days\n"
        f"Interp grid:    {INTERP_STEP_DAYS} days (fixed)\n"
        f"Max gap thresh: {ndvi_info['max_gap_d']} days\n"
        f"NDVI mean:      {ndvi_info['ndvi_mean']:.3f}\n"
        f"NDVI std:       {ndvi_info['ndvi_std']:.3f}\n"
        f"Dynamic range:  {ndvi_info['data_range']:.3f}\n"
        f"Evergreen idx:  {ndvi_info['evergreen_index']:.3f}\n"
        f"  (P5/P95; 1.0=constant, 0=seasonal)\n"
        f"{'─'*30}\nMet parameters: {len(met_info)}"
    )
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=9.5,
             va='top', ha='left', fontfamily='monospace',
             bbox=dict(facecolor='#E8F5E9', edgecolor='#A5D6A7', boxstyle='round,pad=0.8'))
    ax3.set_title('Data Characterization', fontsize=10, fontweight='bold')
    fig.suptitle('Uploaded Data — Automatic Characterization (5-day interpolation grid)',
                 fontsize=12, fontweight='bold', color='#0D2016')
    fig.tight_layout()
    return fig


def plot_sensitivity_heatmap(sensitivity, predictor, train_df):
    ev_list   = [ev for ev in ['SOS', 'POS', 'EOS'] if ev in sensitivity]
    all_feats = sorted({f for ev in ev_list for f in sensitivity[ev]},
                       key=lambda f: max(abs(sensitivity[ev].get(f, {}).get('days_per_std', 0))
                                         for ev in ev_list), reverse=True)
    if not ev_list or not all_feats: return None
    n_feats = len(all_feats); n_evs = len(ev_list)
    mat = np.zeros((n_feats, n_evs))
    for j, ev in enumerate(ev_list):
        for i, f in enumerate(all_feats):
            mat[i, j] = sensitivity[ev].get(f, {}).get('days_per_std', 0)
    abs_max = max(abs(mat).max(), 1.0)
    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#C62828'}
    fig = plt.figure(figsize=(16, max(5, n_feats * 0.7 + 3)))
    fig.patch.set_facecolor('#F7FBF8')
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.8], wspace=0.45)
    ax_hm = fig.add_subplot(gs[0])
    im = ax_hm.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax_hm.set_xticks(range(n_evs)); ax_hm.set_xticklabels(ev_list, fontsize=12, fontweight='bold')
    ax_hm.set_yticks(range(n_feats)); ax_hm.set_yticklabels(all_feats, fontsize=10)
    for i in range(n_feats):
        for j in range(n_evs):
            v  = mat[i, j]; tc = 'white' if abs(v) > abs_max * 0.55 else '#1A1A1A'
            sign = '+' if v >= 0 else ''
            ax_hm.text(j, i, f'{sign}{v:.1f}d', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=tc)
    cb = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cb.set_label('Days shifted per 1σ increase', fontsize=8); cb.ax.tick_params(labelsize=8)
    ax_hm.set_title('Sensitivity Heatmap\n(+red = delays event  ·  −blue = advances event)',
                    fontsize=10, fontweight='bold', color='#0D2016', pad=8)
    ax_hm.spines[:].set_visible(False)
    for j, ev in enumerate(ev_list):
        ax_hm.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (j - 0.48, -0.48), 0.96, n_feats - 0.04,
            boxstyle='round,pad=0.02', linewidth=2,
            edgecolor=ev_colors.get(ev, '#555'), facecolor='none', zorder=5))
    ax_bar = fig.add_subplot(gs[1])
    bar_h  = 0.22; y_pos  = np.arange(n_feats)
    offsets = np.linspace(-(n_evs - 1) / 2, (n_evs - 1) / 2, n_evs) * bar_h
    for j, ev in enumerate(ev_list):
        vals     = [mat[i, j] for i in range(n_feats)]
        bar_clrs = [ev_colors.get(ev, '#888') if abs(v) > 0.5 else '#CFCFCF' for v in vals]
        ax_bar.barh(y_pos + offsets[j], vals, height=bar_h * 0.85,
                    color=bar_clrs, edgecolor='white', lw=0.3, label=ev, alpha=0.88)
    ax_bar.axvline(0, color='#37474F', lw=1.0)
    ax_bar.set_yticks(y_pos); ax_bar.set_yticklabels(all_feats, fontsize=10)
    ax_bar.set_xlabel('Days shifted per 1σ increase in feature', fontsize=9, fontweight='bold')
    ax_bar.set_title('Driver Analysis — Effect on Event Timing',
                     fontsize=10, fontweight='bold', color='#0D2016', pad=8)
    ax_bar.grid(True, axis='x', alpha=0.22, ls='--'); ax_bar.set_facecolor('#FAFFF8')
    ax_bar.legend(title='Event', fontsize=9, title_fontsize=9, loc='lower right', framealpha=0.92,
                  handles=[plt.matplotlib.patches.Patch(color=ev_colors[e], label=e) for e in ev_list])
    ax_bar.spines['top'].set_visible(False); ax_bar.spines['right'].set_visible(False)
    fig.suptitle('Climate Driver Sensitivity — How Much Each Variable Shifts Each Season Event',
                 fontsize=12, fontweight='bold', color='#0D2016', y=1.01)
    fig.tight_layout()
    return fig


def plot_driver_dominance_cards(sensitivity, dominants):
    ev_list = [ev for ev in ['SOS', 'POS', 'EOS'] if ev in sensitivity and sensitivity[ev]]
    if not ev_list: return None
    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#C62828'}
    fig, axes = plt.subplots(1, len(ev_list), figsize=(6 * len(ev_list), max(4, len(ev_list) * 1.5 + 2)))
    if len(ev_list) == 1: axes = [axes]
    fig.patch.set_facecolor('#F7FBF8')
    for ax, ev in zip(axes, ev_list):
        ev_sens  = sensitivity[ev]
        ranked   = sorted(ev_sens.items(), key=lambda x: abs(x[1]['days_per_std']), reverse=True)
        feats    = [r[0] for r in ranked]; vals = [r[1]['days_per_std'] for r in ranked]
        colors   = ['#E53935' if v > 0 else '#1E88E5' for v in vals]
        abs_max  = max((abs(v) for v in vals), default=1.0)
        offset   = abs_max * 0.04
        ax.barh(range(len(feats)), vals, color=colors, alpha=0.82, edgecolor='white', lw=0.5, zorder=3)
        for i, (feat, val) in enumerate(zip(feats, vals)):
            sign = '+' if val >= 0 else ''; direct = '→ delays' if val > 0 else '→ advances'
            x_pos = val + offset * (1 if val >= 0 else -1); ha = 'left' if val >= 0 else 'right'
            ax.text(x_pos, i, f'{sign}{val:.1f}d  {direct}', va='center', ha=ha,
                    fontsize=8.5, color='#222222', zorder=4)
        for i, (feat, val) in enumerate(zip(feats[:3], vals[:3])):
            badge_x = -abs_max * 1.28
            ax.text(badge_x, i, f'#{i+1}', va='center', ha='center', fontsize=8, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.25', facecolor=colors[i], edgecolor='none'))
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=10)
        ax.set_xlim(-abs_max * 1.45, abs_max * 1.6); ax.axvline(0, color='#37474F', lw=1.2, zorder=2)
        ax.set_xlabel('Days shifted per 1σ increase', fontsize=9)
        ax.set_title(f'{ev} — Driver Ranking\nDominant: {dominants.get(ev, {}).get("feature", "—")}',
                     fontsize=11, fontweight='bold', color=ev_colors.get(ev, '#333'))
        ax.set_facecolor('#F7FBF8'); ax.grid(True, axis='x', alpha=0.2, ls='--', zorder=1)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Climate Driver Ranking per Phenological Event\n'
                 '(red = delays event  ·  blue = advances event  ·  magnitude = days per 1σ)',
                 fontsize=11, fontweight='bold', color='#0D2016')
    fig.tight_layout()
    return fig


def plot_radar_chart(sensitivity, selected_event='SOS'):
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    ev_list = [ev for ev in ['SOS', 'POS', 'EOS'] if ev in sensitivity and sensitivity[ev]]
    if not ev_list: return None
    ev_colors = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#C62828'}
    ev_labels = {'SOS': 'Start of Season', 'POS': 'Peak of Season', 'EOS': 'End of Season'}
    all_feats = sorted({f for ev in ev_list for f in sensitivity[ev]})
    N = len(all_feats)
    if N < 3: return None
    fig = plt.figure(figsize=(15, 6)); fig.patch.set_facecolor('#F7FBF8')
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.4)
    ax_radar = fig.add_subplot(gs[0], polar=True)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]
    for ev in ev_list:
        vals_radar = [abs(sensitivity[ev].get(f, {}).get('days_per_std', 0)) for f in all_feats]
        vals_radar += vals_radar[:1]
        lw = 2.5 if ev == selected_event else 1.0
        alpha = 0.30 if ev == selected_event else 0.10
        ax_radar.plot(angles, vals_radar, color=ev_colors[ev], linewidth=lw, label=ev)
        ax_radar.fill(angles, vals_radar, color=ev_colors[ev], alpha=alpha)
    ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(all_feats, fontsize=9, color='#444')
    ax_radar.set_facecolor('#FAFFF8'); ax_radar.grid(color='#CCCCCC', linestyle='--', alpha=0.5)
    ax_radar.spines['polar'].set_color('#DDDDDD')
    ax_radar.set_title('Factor Influence Magnitude\n(absolute days per 1σ)',
                       fontsize=11, fontweight='bold', color='#0D2016', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.85,
                    handles=[mpatches.Patch(color=ev_colors[e], label=e) for e in ev_list])
    ax_grid = fig.add_subplot(gs[1])
    ax_grid.set_xlim(0, 1); ax_grid.set_ylim(0, 1); ax_grid.axis('off')
    ax_grid.set_title('Cross-Event Driver Dominance\n(top driver for each event)',
                      fontsize=11, fontweight='bold', color='#0D2016', pad=14)
    n_ev = len(ev_list); cell_h = 0.80 / n_ev; y_start = 0.88
    for i, ev in enumerate(ev_list):
        ev_sens = sensitivity[ev]
        if not ev_sens: continue
        dom_feat = max(ev_sens, key=lambda f: abs(ev_sens[f]['days_per_std']))
        dom_val  = ev_sens[dom_feat]['days_per_std']
        dom_dir  = 'delays' if dom_val > 0 else 'advances'
        sign = '+' if dom_val > 0 else ''
        bar_color = ev_colors[ev]
        y_box = y_start - i * (cell_h + 0.05)
        box = FancyBboxPatch((0.04, y_box - cell_h + 0.01), 0.92, cell_h - 0.01,
                             boxstyle='round,pad=0.02', linewidth=1.5,
                             edgecolor=bar_color, facecolor=bar_color + '18')
        ax_grid.add_patch(box)
        ax_grid.text(0.08, y_box - 0.012, f'{ev}  {ev_labels[ev]}',
                     fontsize=9, color=bar_color, fontweight='bold', va='top')
        ax_grid.text(0.08, y_box - 0.038, dom_feat, fontsize=13, color='#1A1A1A', fontweight='bold', va='top')
        ax_grid.text(0.08, y_box - 0.068,
                     f'↑ 1σ {dom_dir} {ev} by {sign}{dom_val:.1f} days',
                     fontsize=9, color='#555555', va='top')
        ranked_feats = sorted(ev_sens.items(), key=lambda x: abs(x[1]['days_per_std']), reverse=True)
        bar_x = 0.60
        for j, (feat, finfo) in enumerate(ranked_feats[:4]):
            frac = abs(finfo['days_per_std']) / max(abs(v['days_per_std']) for v in ev_sens.values())
            fc   = '#E53935' if finfo['days_per_std'] > 0 else '#1E88E5'
            ax_grid.barh(y_box - 0.025 - j * 0.022, frac * 0.32,
                         left=bar_x, height=0.016, color=fc, alpha=0.75)
            ax_grid.text(bar_x + frac * 0.32 + 0.01, y_box - 0.025 - j * 0.022,
                         feat, fontsize=7, va='center', color='#555')
    fig.suptitle('Radar: Factor Influence on Events  ·  Cross-Event Driver Summary',
                 fontsize=12, fontweight='bold', color='#0D2016')
    fig.tight_layout()
    return fig


def plot_met_with_ndvi(met_df, ndvi_df, raw_params, pheno_df,
                       interp_freq=INTERP_STEP_DAYS):
    ndvi_s = ndvi_df.set_index('Date')['NDVI'].sort_index()
    if ndvi_s.index.duplicated().any():
        ndvi_s = ndvi_s.groupby(ndvi_s.index).mean()
    full_r  = pd.date_range(start=ndvi_s.index.min(), end=ndvi_s.index.max(), freq=f'{interp_freq}D')
    ndvi_5d = ndvi_s.reindex(ndvi_s.index.union(full_r)).interpolate(method='time').reindex(full_r)
    if pheno_df is None or len(pheno_df) == 0: return []
    ALL_COLS = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#546E7A',
                '#F9A825','#6A1B9A','#795548','#212121']
    param_colors = {p: ALL_COLS[i % len(ALL_COLS)] for i, p in enumerate(raw_params)}
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
            trough_d = row.get('Trough_Date', pd.NaT); sos_d = row.get('SOS_Date', pd.NaT)
            eos_d    = row.get('EOS_Date',    pd.NaT)
            s = (pd.Timestamp(trough_d) if pd.notna(trough_d) else
                 pd.Timestamp(sos_d) - pd.Timedelta(days=60) if pd.notna(sos_d) else None)
            if s is None or pd.isna(eos_d): continue
            e = pd.Timestamp(eos_d) + pd.Timedelta(days=30)
            df_met = met_df[(met_df['Date'] >= s) & (met_df['Date'] <= e)].copy()
            if len(df_met) < 10: continue
            ndvi_seg = ndvi_5d.reindex(df_met['Date'].values, method='nearest',
                                       tolerance=pd.Timedelta('8D')).ffill().bfill()
            yr      = int(row['Year'])
            sos_str = pd.Timestamp(sos_d).strftime('%d %b') if pd.notna(sos_d) else '?'
            eos_str = pd.Timestamp(eos_d).strftime('%d %b %Y') if pd.notna(eos_d) else '?'
            n_panels = 2 if (air_params and soil_params) else 1
            fig, axes_p = plt.subplots(n_panels, 1, figsize=(16, 5.5 * n_panels), sharex=True)
            if n_panels == 1: axes_p = [axes_p]
            fig.patch.set_facecolor('#FAFFF8')
            fig.suptitle(f"NDVI + Met — Season {yr}  [ {sos_str} → {eos_str} ]",
                         fontsize=14, fontweight='bold', y=0.99)
            def _draw_panel(ax, param_list, title, bar_keys=('PRECTOTCORR','PRECTOT','RAIN')):
                ax.fill_between(df_met['Date'], ndvi_seg, alpha=0.18, color='#2E7D32')
                ax.plot(df_met['Date'], ndvi_seg, color='#2E7D32', lw=2.5, label='NDVI')
                ax.set_ylabel('NDVI', color='#2E7D32', fontsize=11, fontweight='bold')
                ax.set_ylim(0, 1.05); ax.tick_params(axis='y', labelcolor='#2E7D32')
                ax.grid(True, linestyle='--', alpha=0.28); ax.set_facecolor('#F7FBF8')
                ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
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
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════

def _model_badge_html(model_name):
    key_map = {'Ridge': 'ridge', 'LOESS': 'loess',
               'Poly-2': 'poly2', 'Poly-3': 'poly3', 'GPR': 'gpr', 'mean': 'mean'}
    css_cls = key_map.get(model_name, 'ridge')
    return f'<span class="model-badge {css_cls}">{model_name}</span>'


def _build_eq_box_html(eq_raw, ev, best_name):
    ev_labels = {
        'SOS': '🌱 Start of Season — SOS Model',
        'POS': '🌿 Peak of Season — POS Model',
        'EOS': '🍂 End of Season — EOS Model',
    }
    badge = _model_badge_html(best_name)
    label = f"{ev_labels.get(ev, ev)} {badge}"
    lines = eq_raw.strip().split('\n')
    main_line   = lines[0].strip() if lines else eq_raw
    meta_line   = ''
    models_line = ''
    for ln in lines[1:]:
        ln = ln.strip()
        if ln.startswith('[Best model') or ln.startswith('[No feature'):
            meta_line = ln.strip('[]')
        elif ln.startswith('All models'):
            models_line = ln
    if not meta_line and len(lines) > 1:
        meta_line = lines[1].strip().strip('[]')
    html = (f'<div class="eq-box"><span class="eq-label">{label}</span>'
            f'<span class="eq-main">{main_line}</span>')
    if meta_line:
        html += f'<span class="eq-meta">📊 {meta_line}</span>'
    if models_line:
        html += f'<span class="eq-models">🔬 {models_line}</span>'
    html += '</div>'
    return html


# ═══════════════════════════════════════════════════════════════
DS_SENSORS = {
    "Landsat (30m)":    {"key":"LS",  "color":"#E8A020", "bg":"#FFF9EE", "hex":"#B07800"},
    "Sentinel-2 (10m)": {"key":"S2",  "color":"#2080E0", "bg":"#EEF5FF", "hex":"#0055AA"},
    "MODIS (250-500m)": {"key":"MOD", "color":"#00B896", "bg":"#EEFAF7", "hex":"#006655"},
}

def _parse_ndvi_sensor(uploaded_file):
    """Parse a sensor NDVI CSV: Date + NDVI columns, dedup by mean."""
    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        df = pd.read_csv(StringIO(raw.decode('utf-8', errors='replace')))
        df.columns = [c.strip() for c in df.columns]
        date_col = next((c for c in df.columns if c.lower() in ['date','dates','time','datetime']), None)
        ndvi_col = next((c for c in df.columns if c.lower() in ['ndvi','ndvi_value','value','index','evi']), None)
        if not date_col or not ndvi_col:
            return None, f"Need columns: Date + NDVI (found: {list(df.columns)})"
        df = df.rename(columns={date_col:'Date', ndvi_col:'NDVI'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce')
        df = df.dropna(subset=['Date'])[['Date','NDVI']].sort_values('Date').reset_index(drop=True)
        if df['Date'].duplicated().any():
            df = df.groupby('Date', as_index=False)['NDVI'].mean().sort_values('Date').reset_index(drop=True)
        if len(df) == 0:
            return None, "No valid rows after parsing"
        return df, None
    except Exception as e:
        return None, str(e)


def _extract_pheno_sensor(ndvi_df, start_month=6, end_month=5,
                           sos_thr=0.10, eos_thr=0.10, min_days=90):
    """Wrapper: calls full production extract_phenology for one sensor."""
    if ndvi_df['Date'].duplicated().any():
        ndvi_df = ndvi_df.groupby('Date', as_index=False)['NDVI'].mean().sort_values('Date').reset_index(drop=True)
    cfg = {"start_month": start_month, "end_month": end_month, "min_days": min_days}
    pheno_df, err = extract_phenology(ndvi_df, cfg,
                                      sos_threshold_pct=sos_thr, eos_threshold_pct=eos_thr)
    if pheno_df is None:
        return None, err
    for dc in ['SOS_Date','POS_Date','EOS_Date']:
        if dc in pheno_df.columns:
            pheno_df[dc] = pd.to_datetime(pheno_df[dc], errors='coerce')
    return pheno_df.reset_index(drop=True), None


def fit_sensor_models(pheno_df, met_df, params, event='SOS', window=15):
    """Fit all models for one sensor+event. Returns dict of results."""
    from datetime import timedelta
    date_col = f'{event}_Date'; doy_col = f'{event}_DOY'
    if date_col not in pheno_df.columns: return {}
    rows = pheno_df[[date_col, doy_col]].dropna()
    if len(rows) < 2: return {}

    # Build feature matrix
    records = []
    for _, row in rows.iterrows():
        evt_dt = row[date_col]
        rec = {'y': float(row[doy_col])}
        for p in params:
            if p not in met_df.columns: continue
            mask = ((met_df['Date'] >= pd.Timestamp(evt_dt)-timedelta(days=window)) &
                    (met_df['Date'] <= pd.Timestamp(evt_dt)))
            wdf = met_df[mask]
            if len(wdf) == 0: rec[p] = np.nan; continue
            rec[p] = float(wdf[p].mean())
        records.append(rec)

    df = pd.DataFrame(records).dropna()
    if len(df) < 2: return {}

    y  = df['y'].values
    Xc = [c for c in df.columns if c != 'y']
    if not Xc: return {}
    X  = df[Xc].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    def loo_r2_mae(preds):
        ss_res = np.sum((y-preds)**2)
        ss_tot = np.sum((y-y.mean())**2)+1e-12
        r2  = float(np.clip(1-ss_res/ss_tot,-1,1))
        mae = float(np.mean(np.abs(y-preds)))
        return r2, mae

    results = {}

    # Ridge
    try:
        rcv = RidgeCV(alphas=np.logspace(-3,4,30))
        rcv.fit(Xs, y); a = float(rcv.alpha_)
        preds_r = []
        for i in range(len(y)):
            idx_t = [j for j in range(len(y)) if j!=i]
            sc2 = StandardScaler(); Xtr = sc2.fit_transform(X[idx_t]); Xte = sc2.transform(X[[i]])
            m = Ridge(alpha=a); m.fit(Xtr, y[idx_t])
            preds_r.append(float(m.predict(Xte)[0]))
        pipe = Pipeline([('sc',StandardScaler()),('r',Ridge(alpha=a))])
        pipe.fit(X, y)
        r2, mae = loo_r2_mae(np.array(preds_r))
        results['Ridge'] = {'r2':r2,'mae':mae,'n':len(y),'features':Xc,
                             'coef':list(pipe.named_steps['r'].coef_/pipe.named_steps['sc'].scale_),
                             'intercept':float(pipe.named_steps['r'].intercept_-
                                               np.dot(pipe.named_steps['r'].coef_/pipe.named_steps['sc'].scale_,
                                                      pipe.named_steps['sc'].mean_)),
                             'alpha':a,'pipe':pipe}
    except Exception: pass

    # Poly-2
    if len(y) >= 4:
        try:
            ppipe = Pipeline([('poly',PolynomialFeatures(2,include_bias=False)),
                               ('sc',StandardScaler()),('r',RidgeCV(alphas=np.logspace(-3,4,20)))])
            preds_p = []
            for i in range(len(y)):
                idx_t = [j for j in range(len(y)) if j!=i]
                pp2 = Pipeline([('poly',PolynomialFeatures(2,include_bias=False)),
                                 ('sc',StandardScaler()),('r',RidgeCV(alphas=np.logspace(-3,4,20)))])
                pp2.fit(X[idx_t], y[idx_t])
                preds_p.append(float(pp2.predict(X[[i]])[0]))
            ppipe.fit(X, y)
            r2, mae = loo_r2_mae(np.array(preds_p))
            results['Poly-2'] = {'r2':r2,'mae':mae,'n':len(y),'features':Xc,'pipe':ppipe}
        except Exception: pass

    # GPR
    if len(y) >= 5:
        try:
            kernel = ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(0.1)
            gpr = GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=42)
            gpr.fit(Xs, y)
            preds_g = []
            for i in range(len(y)):
                idx_t = [j for j in range(len(y)) if j!=i]
                sc2 = StandardScaler(); Xtr = sc2.fit_transform(X[idx_t])
                gp2 = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(0.1),
                                               normalize_y=True,random_state=42)
                gp2.fit(Xtr, y[idx_t])
                preds_g.append(float(gp2.predict(sc2.transform(X[[i]]))[0]))
            r2, mae = loo_r2_mae(np.array(preds_g))
            results['GPR'] = {'r2':r2,'mae':mae,'n':len(y),'features':Xc,'gpr':gpr,'sc':sc}
        except Exception: pass

    # Mean baseline
    md = float(y.mean())
    results['Mean Baseline'] = {'r2':0.0,'mae':float(np.mean(np.abs(y-md))),'n':len(y),'features':[]}

    return results

# ═══════════════════════════════════════════════════════════════
# SENSOR AGREEMENT STATISTICS
# ═══════════════════════════════════════════════════════════════
def sensor_agreement(df_a, df_b, event='SOS'):
    """Compute bias, RMSE, MAE, Pearson r between two sensor phenology tables."""
    col = f'{event}_DOY'
    merged = pd.merge(df_a[['Year',col]], df_b[['Year',col]], on='Year', suffixes=('_a','_b')).dropna()
    if len(merged) < 2: return {}
    a = merged[f'{col}_a'].values
    b = merged[f'{col}_b'].values
    bias = float(np.mean(b - a))
    rmse = float(np.sqrt(np.mean((b-a)**2)))
    mae  = float(np.mean(np.abs(b-a)))
    try: r, p = pearsonr(a, b)
    except Exception: r, p = np.nan, np.nan
    return {'bias':round(bias,2),'rmse':round(rmse,2),'mae':round(mae,2),
            'r':round(float(r),4),'p':round(float(p),4),'n':len(merged)}

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
def plot_pheno_comparison(sensor_pheno, sensor_colors):
    """Bar+line chart comparing SOS/POS/EOS DOY across sensors per year."""
    events = ['SOS_DOY','POS_DOY','EOS_DOY']
    ev_labels = ['SOS — Start','POS — Peak','EOS — End']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#F8FAFD')

    for ax, ev, lbl in zip(axes, events, ev_labels):
        for sname, pheno in sensor_pheno.items():
            if ev not in pheno.columns: continue
            clr = sensor_colors[sname]['color']
            yrs = pheno['Year'].values
            vals= pheno[ev].values.astype(float)
            ax.plot(yrs, vals, 'o-', color=clr, lw=2.2, ms=7,
                    label=sname, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
            ax.fill_between(yrs, vals, alpha=0.08, color=clr)

        ax.set_title(lbl, fontsize=11, fontweight='bold', color='#1A2840', pad=10)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Day of Year', fontsize=9)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.grid(True, alpha=0.2, ls='--'); ax.set_facecolor('#FAFCFF')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('Phenological Event Comparison Across Sensors',
                 fontsize=13, fontweight='bold', color='#0A1628', y=1.02)
    fig.tight_layout()
    return fig

def plot_model_radar(model_results_by_sensor, sensor_colors, event='SOS'):
    """R² comparison radar / bar chart across models and sensors."""
    model_names = ['Ridge','Poly-2','GPR','Mean Baseline']
    n_mod = len(model_names)
    sensors = list(model_results_by_sensor.keys())
    n_sen   = len(sensors)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#F8FAFD')

    x      = np.arange(n_mod)
    width  = 0.22
    offset = np.linspace(-(n_sen-1)/2, (n_sen-1)/2, n_sen) * width

    for i, sname in enumerate(sensors):
        vals = []
        for mn in model_names:
            r = model_results_by_sensor[sname].get(event,{}).get(mn,{}).get('r2', np.nan)
            vals.append(float(r) if not np.isnan(r) else 0)
        clr = sensor_colors[sname]['color']
        bars = ax.bar(x + offset[i], vals, width*0.85, label=sname,
                      color=clr, alpha=0.85, edgecolor='white', lw=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='600')

    ax.axhline(0.6, color='#1B5E20', lw=1.2, ls='--', alpha=0.5, label='R²=0.60 (good)')
    ax.axhline(0.3, color='#F9A825', lw=1,   ls='--', alpha=0.5, label='R²=0.30 (moderate)')
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('LOO R²', fontsize=10, fontweight='bold')
    ax.set_title(f'Model Performance Comparison — {event}  (LOO R²)',
                 fontsize=11, fontweight='bold', color='#0A1628', pad=10)
    ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.2, ls='--')
    ax.set_facecolor('#FAFCFF')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def plot_ndvi_overlay(sensor_ndvi, sensor_colors):
    """Plot NDVI time series from all sensors overlaid."""
    fig, ax = plt.subplots(figsize=(16, 4.5))
    fig.patch.set_facecolor('#F8FAFD')

    for sname, df in sensor_ndvi.items():
        clr = sensor_colors[sname]['color']
        # 5-day grid
        ndvi_s = df.set_index('Date')['NDVI'].sort_index()
        # Safety dedup — in case raw df still has duplicate dates
        if ndvi_s.index.duplicated().any():
            ndvi_s = ndvi_s.groupby(ndvi_s.index).mean()
        fr = pd.date_range(ndvi_s.index.min(), ndvi_s.index.max(), freq=f'{INTERP_STEP_DAYS}D')
        combined_idx = ndvi_s.index.union(fr)
        if combined_idx.duplicated().any():
            combined_idx = combined_idx[~combined_idx.duplicated()]
        ndvi_5d = ndvi_s.reindex(combined_idx).interpolate(
            method='time', limit_area='inside').reindex(fr)
        # raw scatter
        ax.scatter(df['Date'], df['NDVI'], color=clr, s=12, alpha=0.35, zorder=2)
        # smoothed line
        ax.plot(ndvi_5d.index, ndvi_5d.values, color=clr, lw=2, label=sname, zorder=3)

    ax.set_ylabel('NDVI', fontsize=10)
    ax.set_xlabel('Date', fontsize=9)
    ax.set_title('NDVI Time Series — All Sensors Overlaid (5-day grid)',
                 fontsize=11, fontweight='bold', color='#0A1628')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.grid(True, alpha=0.18, ls='--'); ax.set_facecolor('#FAFCFF')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def plot_bias_scatter(sensor_pheno, sensor_colors, ref_sensor, comp_sensor, event='SOS'):
    """Scatter plot: ref sensor DOY vs comparison sensor DOY."""
    col = f'{event}_DOY'
    a = sensor_pheno[ref_sensor]; b = sensor_pheno[comp_sensor]
    m = pd.merge(a[['Year',col]], b[['Year',col]], on='Year', suffixes=('_a','_b')).dropna()
    if len(m) < 2: return None
    xa = m[f'{col}_a'].values; xb = m[f'{col}_b'].values

    fig, ax = plt.subplots(figsize=(5, 5))
    ca = sensor_colors[ref_sensor]['color']
    cb = sensor_colors[comp_sensor]['color']
    ax.scatter(xa, xb, c=[ca]*len(xa), s=90, edgecolors='white', lw=1.5, zorder=3)
    for x_, y_, yr in zip(xa, xb, m['Year']):
        ax.annotate(str(yr), (x_,y_), xytext=(4,4), textcoords='offset points', fontsize=8, color='#555')
    lims = [min(xa.min(),xb.min())-8, max(xa.max(),xb.max())+8]
    ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.5, label='1:1 line')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f'{ref_sensor} — {event} DOY', fontsize=9)
    ax.set_ylabel(f'{comp_sensor} — {event} DOY', fontsize=9)
    ax.set_title(f'{event}: {ref_sensor.split(" ")[0]} vs {comp_sensor.split(" ")[0]}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, ls='--')
    ax.set_facecolor('#FAFCFF')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_html_report(sensor_pheno, model_results, agree_stats, config):
    """Generate a professional HTML report."""
    rows_pheno = ""
    for sname, pheno in sensor_pheno.items():
        for _, row in pheno.iterrows():
            rows_pheno += f"""<tr>
                <td>{sname}</td><td>{int(row['Year'])}</td>
                <td>{pd.Timestamp(row['SOS_Date']).strftime('%b %d')}</td>
                <td>{int(row['SOS_DOY'])}</td>
                <td>{pd.Timestamp(row['POS_Date']).strftime('%b %d')}</td>
                <td>{int(row['POS_DOY'])}</td>
                <td>{pd.Timestamp(row['EOS_Date']).strftime('%b %d')}</td>
                <td>{int(row['LOS_Days'])}</td>
                <td>{row['Peak_NDVI']:.4f}</td>
            </tr>"""

    rows_model = ""
    for sname, ev_results in model_results.items():
        for ev, mods in ev_results.items():
            best = max((m for m in mods if m != 'Mean Baseline'),
                       key=lambda m: mods[m].get('r2',-999), default=None)
            for mn, mfit in mods.items():
                r2  = mfit.get('r2', np.nan)
                mae = mfit.get('mae', np.nan)
                star = '⭐ BEST' if mn == best else ''
                _r2s  = f"{r2:.4f}" if not np.isnan(r2)  else "—"
                _maes = f"{mae:.1f}" if not np.isnan(mae) else "—"
                _best_cls = 'class="best-row"' if mn==best else ""
                rows_model += (f'<tr {_best_cls}>'
                    f'<td>{sname}</td><td>{ev}</td><td>{mn}</td>'
                    f'<td>{_r2s}</td>'
                    f'<td>{_maes}</td>'
                    f'<td>{mfit.get("n","—")}</td>'
                    f'<td>{star}</td></tr>')

    agree_rows = ""
    for pair, ev_stats in agree_stats.items():
        for ev, stats in ev_stats.items():
            if not stats: continue
            agree_rows += f"""<tr>
                <td>{pair}</td><td>{ev}</td>
                <td>{stats.get('bias','—')}</td>
                <td>{stats.get('rmse','—')}</td>
                <td>{stats.get('mae','—')}</td>
                <td>{stats.get('r','—')}</td>
                <td>{stats.get('n','—')}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Dual-Sensor Phenology Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1A2840; margin:0; padding:0; background:#F5F8FC; }}
.header {{ background:linear-gradient(135deg,#0A0E1A,#1A3060); color:#fff; padding:40px 48px; }}
.header h1 {{ font-size:2rem; margin:0 0 8px; font-weight:800; }}
.header p {{ color:#90B0D8; font-size:0.94rem; margin:0; }}
.section {{ background:#fff; border-radius:12px; padding:32px 36px; margin:20px 28px; border:1px solid #E0EAF5; }}
.section h2 {{ font-size:1.15rem; font-weight:700; color:#0A1628; border-bottom:2px solid #D0DFF0; padding-bottom:8px; margin-top:0; }}
table {{ width:100%; border-collapse:collapse; font-size:0.88rem; margin-top:12px; }}
th {{ background:#F0F7FF; padding:10px 14px; text-align:left; font-weight:700; font-size:0.80rem; text-transform:uppercase; letter-spacing:0.5px; color:#2A4870; }}
td {{ padding:9px 14px; border-bottom:1px solid #EEF3FA; }}
tr:hover td {{ background:#FAFCFF; }}
tr.best-row td {{ background:#F0FFF4; font-weight:600; }}
.footer {{ text-align:center; padding:24px; color:#8A9BB0; font-size:0.82rem; }}
.chip {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.76rem; font-weight:700; }}
.ls-chip  {{ background:#FFF4E0; color:#B07000; }}
.s2-chip  {{ background:#E0EEFF; color:#0055AA; }}
.mod-chip {{ background:#E0FFF8; color:#006655; }}
</style></head><body>
<div class="header">
<h1>🛰️ Dual-Sensor Phenology Comparison Report</h1>
<p>Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} &nbsp;·&nbsp;
Site window: {config.get('start_m','Jun')} → {config.get('end_m','May')} &nbsp;·&nbsp;
SOS threshold: {config.get('sos_thr',10)}% &nbsp;·&nbsp; EOS threshold: {config.get('eos_thr',10)}%</p>
</div>

<div class="section">
<h2>1. Phenology Dates — All Sensors</h2>
<table>
<tr><th>Sensor</th><th>Year</th><th>SOS Date</th><th>SOS DOY</th><th>POS Date</th><th>POS DOY</th><th>EOS Date</th><th>LOS (days)</th><th>Peak NDVI</th></tr>
{rows_pheno}
</table></div>

<div class="section">
<h2>2. Model Performance (LOO Cross-Validation)</h2>
<table>
<tr><th>Sensor</th><th>Event</th><th>Model</th><th>LOO R²</th><th>MAE (days)</th><th>n seasons</th><th>Status</th></tr>
{rows_model}
</table></div>

<div class="section">
<h2>3. Sensor Agreement Statistics</h2>
<table>
<tr><th>Sensor Pair</th><th>Event</th><th>Bias (days)</th><th>RMSE</th><th>MAE</th><th>Pearson r</th><th>n years</th></tr>
{agree_rows if agree_rows else '<tr><td colspan="7" style="color:#888;text-align:center">Need ≥2 sensors for agreement statistics</td></tr>'}
</table>
<p style="font-size:0.82rem;color:#6A84A0;margin-top:10px">
Bias = mean(Sensor B – Sensor A) · Positive = Sensor B detects event later than Sensor A
</p></div>

<div class="section">
<h2>4. Definition of Phenology Metrics</h2>
<table>
<tr><th>Metric</th><th>Definition</th></tr>
<tr><td><b>SOS</b></td><td>Start of Season — first crossing of (Base_NDVI + SOS_thr% × Amplitude) on ascending limb</td></tr>
<tr><td><b>POS</b></td><td>Peak of Season — date of maximum NDVI in the season cycle</td></tr>
<tr><td><b>EOS</b></td><td>End of Season — first crossing below (Base_NDVI + EOS_thr% × Amplitude) on descending limb</td></tr>
<tr><td><b>LOS</b></td><td>Length of Season — EOS minus SOS in days</td></tr>
<tr><td><b>Amplitude</b></td><td>Peak_NDVI minus Base_NDVI (trough)</td></tr>
<tr><td><b>LOO R²</b></td><td>Leave-One-Out R² — honest out-of-sample model accuracy</td></tr>
<tr><td><b>Bias</b></td><td>Systematic offset between sensors (days). Reflects spatial resolution effect.</td></tr>
</table></div>

<div class="footer">Indian Forest Phenology Assessment &nbsp;·&nbsp; Multi-Sensor Edition &nbsp;·&nbsp; {datetime.now().year}</div>
</body></html>"""
    return html

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════


def _render_sensor_comparison_tab(start_m, end_m, sos_thr, eos_thr, min_days,
                                   sensor_files=None):
    """
    Render the Sensor Comparison tab.
    sensor_files: dict {sensor_name: uploaded_file} from the sidebar.
    Works with 1 sensor (shows results only) or 2–3 (adds comparison stats).
    """
    mo_n = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
            7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    if sensor_files is None:
        sensor_files = {}

    n_loaded = len(sensor_files)

    # ── Header ───────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0A0E1A,#0F1E35);padding:20px 28px 16px;'
        f'border-radius:16px;margin-bottom:18px;border:1px solid rgba(100,180,255,0.18)">'
        f'<div style="color:#E8F4FF;font-size:1.1rem;font-weight:800;margin-bottom:5px">'
        f'🛰️ Multi-Sensor Phenology Comparison</div>'
        f'<div style="color:#7BA8CC;font-size:0.83rem;line-height:1.55">'
        f'Upload sensor NDVI files in the <b>sidebar</b> (👈 left panel) under '
        f'<em>OPTIONAL — Multi-Sensor Comparison</em>. '
        f'One file shows that sensor\'s results. Two or three files add side-by-side '
        f'comparison charts and inter-sensor agreement statistics.</div>'
        f'<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">'
        f'<span style="background:rgba(232,160,32,.18);color:#FFB432;font-size:0.70rem;font-weight:700;padding:2px 10px;border-radius:12px;border:1px solid rgba(232,160,32,.3)">🌍 Landsat 30m</span>'
        f'<span style="background:rgba(32,128,224,.18);color:#60B8FF;font-size:0.70rem;font-weight:700;padding:2px 10px;border-radius:12px;border:1px solid rgba(32,128,224,.3)">🛰️ Sentinel-2 10m</span>'
        f'<span style="background:rgba(0,184,150,.18);color:#00DCB4;font-size:0.70rem;font-weight:700;padding:2px 10px;border-radius:12px;border:1px solid rgba(0,184,150,.3)">📡 MODIS 250–500m</span>'
        f'<span style="background:rgba(160,160,255,.15);color:#B0A8FF;font-size:0.70rem;font-weight:700;padding:2px 10px;border-radius:12px;border:1px solid rgba(160,160,255,.3)">'
        f'{"✅ " + str(n_loaded) + " sensor" + ("s" if n_loaded!=1 else "") + " loaded" if n_loaded else "⬅️ Add sensors in sidebar"}'
        f'</span></div></div>',
        unsafe_allow_html=True)

    # ── Status strip ─────────────────────────────────────────
    if n_loaded == 0:
        st.markdown(
            '<div style="background:#F5F8FF;border:2px dashed #90B8E8;border-radius:14px;'
            'padding:36px 24px;text-align:center;margin:24px 0">'
            '<div style="font-size:2.4rem;margin-bottom:12px">🛰️</div>'
            '<div style="font-size:1.05rem;font-weight:700;color:#1A3860;margin-bottom:8px">'
            'No sensor files uploaded yet</div>'
            '<div style="color:#4A6A90;font-size:0.86rem;line-height:1.6">'
            'Open the <b>sidebar</b> (👈) and upload at least one sensor CSV under<br>'
            '<em>"OPTIONAL — Multi-Sensor Comparison"</em>.<br><br>'
            'Each file needs: <code>Date</code> and <code>NDVI</code> columns.<br>'
            'Duplicate dates are automatically averaged.</div></div>',
            unsafe_allow_html=True)
        return
    elif n_loaded == 1:
        st.markdown(
            '<div class="banner-info">ℹ️ <b>1 sensor loaded</b> — showing single-sensor results. '
            'Add a second or third sensor file in the sidebar to unlock comparison charts and '
            'inter-sensor agreement statistics.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner-good">✅ <b>{n_loaded} sensors loaded</b> — '
            f'showing results and cross-sensor comparison.</div>', unsafe_allow_html=True)

    # ── Optional shared met file ─────────────────────────────
    met_col1, met_col2 = st.columns([3, 1])
    with met_col1:
        met_f = st.file_uploader(
            "📡 Shared meteorological CSV *(optional — enables model fitting per sensor)*",
            type=['csv'], key="ds_met_up",
            help="Same NASA POWER daily CSV used in the main analysis.")
    with met_col2:
        met_window = st.slider("Climate window (days)", 7, 60, 15, 1, key="ds_met_win")

    st.markdown("---")

    # ── Parse & extract phenology ────────────────────────────
    sensor_ndvi  = {}
    sensor_pheno = {}
    errors       = {}

    met_df_ds = None; met_params_ds = []
    if met_f:
        met_df_ds, met_params_ds, met_err = parse_nasa_power(met_f)
        if met_df_ds is None:
            st.markdown(f'<div class="banner-warn">⚠️ Met file error: {met_err}</div>',
                        unsafe_allow_html=True)
        else:
            met_df_ds = add_derived_features(met_df_ds, season_start_month=start_m)
            met_params_ds = [c for c in met_df_ds.columns
                             if c not in {'Date','YEAR','MO','DY','DOY','LON','LAT','ELEV'}
                             and pd.api.types.is_numeric_dtype(met_df_ds[c])]

    with st.spinner("Extracting phenology for all sensors…"):
        for sname, f in sensor_files.items():
            df, err = _parse_ndvi_sensor(f)
            if df is None:
                errors[sname] = err; continue
            sensor_ndvi[sname] = df
            pheno, perr = _extract_pheno_sensor(df, start_month=start_m,
                                                 end_month=end_m, sos_thr=sos_thr,
                                                 eos_thr=eos_thr, min_days=min_days)
            if pheno is None:
                errors[sname] = perr
            else:
                sensor_pheno[sname] = pheno

    for sname, err in errors.items():
        st.markdown(f'<div class="banner-warn">⚠️ <b>{sname}:</b> {err}</div>',
                    unsafe_allow_html=True)

    if not sensor_pheno:
        st.error("No sensors produced valid phenology. Check files and season settings.")
        return

    # ── Model fitting ────────────────────────────────────────
    model_results = {}
    if met_df_ds is not None and met_params_ds:
        with st.spinner("Fitting models per sensor…"):
            for sname, pheno in sensor_pheno.items():
                model_results[sname] = {}
                for ev in ['SOS','POS','EOS']:
                    model_results[sname][ev] = fit_sensor_models(
                        pheno, met_df_ds, met_params_ds, event=ev, window=met_window)

    # ── Sensor agreement ─────────────────────────────────────
    sensor_names = list(sensor_pheno.keys())
    agree_stats  = {}
    if len(sensor_names) >= 2:
        for i in range(len(sensor_names)):
            for j in range(i+1, len(sensor_names)):
                pair = f"{sensor_names[i].split('(')[0].strip()} vs {sensor_names[j].split('(')[0].strip()}"
                agree_stats[pair] = {}
                for ev in ['SOS','POS','EOS']:
                    agree_stats[pair][ev] = sensor_agreement(
                        sensor_pheno[sensor_names[i]], sensor_pheno[sensor_names[j]], ev)

    # ── Sub-tabs inside the comparison tab ───────────────────
    _agree_label = "📐 Sensor Agreement" if len(sensor_pheno) >= 2 else "📐 Agreement (2+ sensors)"
    ds_tabs = st.tabs(["🌿 NDVI Overview", "📅 Season Dates",
                       "🏆 Model Performance", _agree_label, "📋 Export"])

    # ── Sub-tab 1: NDVI Overview ─────────────────────────────
    with ds_tabs[0]:
        st.markdown('<p class="section-title">NDVI Time Series — All Sensors</p>',
                    unsafe_allow_html=True)
        kc = st.columns(len(sensor_ndvi))
        for col, (sname, df) in zip(kc, sensor_ndvi.items()):
            scfg  = DS_SENSORS.get(sname, {'color':'#888','bg':'#EEE'})
            n_obs  = len(df)
            yr_rng = f"{df['Date'].dt.year.min()}–{df['Date'].dt.year.max()}"
            diffs  = df['Date'].sort_values().diff().dt.days.dropna()
            cad    = round(float(diffs[diffs>0].median()), 0) if len(diffs[diffs>0]) > 0 else '?'
            n_seas = len(sensor_pheno.get(sname, pd.DataFrame()))
            col.markdown(
                f"<div style='background:{scfg['bg']};border-top:3px solid {scfg['color']};"
                f"border-radius:12px;padding:14px 16px;text-align:center'>"
                f"<div style='font-size:0.70rem;font-weight:700;color:{scfg['color']};margin-bottom:4px'>"
                f"{sname.split('(')[0].strip().upper()}</div>"
                f"<div style='font-size:1.2rem;font-weight:800;color:#0A1628'>{yr_rng}</div>"
                f"<div style='font-size:0.78rem;color:#6A84A0;margin-top:3px'>"
                f"{n_obs} obs · {cad}d cadence · <b>{n_seas} seasons</b></div>"
                f"</div>", unsafe_allow_html=True)

        if sensor_ndvi:
            fig_ov = plot_ndvi_overlay(sensor_ndvi, DS_SENSORS)
            st.pyplot(fig_ov, use_container_width=True)
            plt.close(fig_ov)

        ref_df = pd.DataFrame({
            'Sensor': ['MODIS (MOD13Q1)','Landsat (8/9)','Sentinel-2 (MSI)'],
            'Spatial resolution': ['250 m','30 m','10 m'],
            'Temporal cadence': ['16 days','16 days','5–12 days'],
            'Primary use': ['Long-term baselines','Detailed spatial','High-res recent'],
        })
        st.markdown('<p class="section-title">Sensor Reference Table</p>', unsafe_allow_html=True)
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

    # ── Sub-tab 2: Phenology Dates ───────────────────────────
    with ds_tabs[1]:
        st.markdown('<p class="section-title">Extracted Season Dates — Per Sensor</p>',
                    unsafe_allow_html=True)
        pt_cols = st.columns(len(sensor_pheno))
        mo_n2   = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        for col, (sname, pheno) in zip(pt_cols, sensor_pheno.items()):
            scfg = DS_SENSORS.get(sname, {'color':'#888','bg':'#FFF'})
            col.markdown(
                f"<div style='background:{scfg['bg']};border-left:3px solid {scfg['color']};"
                f"border-radius:8px;padding:8px 12px;margin-bottom:6px'>"
                f"<b style='color:{scfg['color']}'>{sname}</b></div>",
                unsafe_allow_html=True)
            disp = []
            for _, row in pheno.iterrows():
                disp.append({
                    'Year': int(row['Year']),
                    'SOS':  pd.Timestamp(row['SOS_Date']).strftime('%b %d') if pd.notna(row.get('SOS_Date')) else '—',
                    'POS':  pd.Timestamp(row['POS_Date']).strftime('%b %d') if pd.notna(row.get('POS_Date')) else '—',
                    'EOS':  pd.Timestamp(row['EOS_Date']).strftime('%b %d') if pd.notna(row.get('EOS_Date')) else '—',
                    'LOS':  f"{int(row['LOS_Days'])}d" if pd.notna(row.get('LOS_Days')) else '—',
                    'Peak': round(float(row['Peak_NDVI']),3) if pd.notna(row.get('Peak_NDVI')) else '—',
                })
            col.dataframe(pd.DataFrame(disp), use_container_width=True, hide_index=True)

        if len(sensor_pheno) >= 1:
            fig_cmp = plot_pheno_comparison(sensor_pheno, DS_SENSORS)
            st.pyplot(fig_cmp, use_container_width=True)
            plt.close(fig_cmp)

        st.markdown('<p class="section-title">Trend Summary (slope d/yr)</p>',
                    unsafe_allow_html=True)
        trend_rows = []
        for ev in ['SOS_DOY','POS_DOY','EOS_DOY','LOS_Days']:
            row_d = {'Metric': ev.replace('_DOY',' DOY').replace('_Days',' (days)')}
            for sname, pheno in sensor_pheno.items():
                sshort = sname.split('(')[0].strip()
                if ev not in pheno.columns or len(pheno) < 2:
                    row_d[sshort] = '—'; continue
                yrs  = pheno['Year'].values
                vals = pheno[ev].values.astype(float)
                slope, _ = np.polyfit(yrs, vals, 1)
                mean_v   = float(np.mean(vals))
                row_d[sshort] = f"μ={mean_v:.1f} | trend={slope:+.2f}d/yr"
            trend_rows.append(row_d)
        st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

        comb = []
        for sname, pheno in sensor_pheno.items():
            t = pheno.copy(); t.insert(0,'Sensor',sname); comb.append(t)
        if comb:
            st.download_button("📥 Download Combined Phenology CSV",
                               pd.concat(comb).to_csv(index=False),
                               "phenology_all_sensors.csv", "text/csv", key="ds_dl_pheno")

    # ── Sub-tab 3: Model Performance ────────────────────────
    with ds_tabs[2]:
        st.markdown('<p class="section-title">Model Performance — All Sensors × All Events</p>',
                    unsafe_allow_html=True)
        if not model_results:
            st.markdown(
                '<div class="banner-info">ℹ️ Upload a shared <b>meteorological CSV</b> above ' 
                'to enable model fitting per sensor and see performance comparison.</div>',
                unsafe_allow_html=True)
        else:
            ev_sel = st.radio("Event:", ['SOS','POS','EOS'], horizontal=True,
                              key="ds_ev_sel",
                              format_func=lambda e: {'SOS':'🌱 SOS','POS':'🌿 POS','EOS':'🍂 EOS'}[e])
            fig_mr = plot_model_radar(model_results, DS_SENSORS, event=ev_sel)
            st.pyplot(fig_mr, use_container_width=True)
            plt.close(fig_mr)

            st.markdown(f"**Detailed table — {ev_sel}:**")
            all_mod_names = ['Ridge','Poly-2','GPR','Mean Baseline']
            rows_html = ""
            for sname in sensor_pheno:
                scfg   = DS_SENSORS.get(sname,{'color':'#888','bg':'#FFF'})
                ev_m   = model_results.get(sname,{}).get(ev_sel,{})
                cands  = {m: d.get('r2',-999) for m,d in ev_m.items() if m != 'Mean Baseline'}
                best_m = max(cands, key=cands.get) if cands else None
                for mn in all_mod_names:
                    mfit = ev_m.get(mn)
                    if mfit is None:
                        rows_html += (f"<tr><td><b style='color:{scfg['color']}'>{sname.split('(')[0].strip()}</b></td>"
                                      f"<td>{mn}</td><td colspan='3' style='color:#AAA'>n/a for sample size</td></tr>")
                        continue
                    r2  = mfit.get('r2', np.nan)
                    mae = mfit.get('mae', np.nan)
                    n   = mfit.get('n','—')
                    r2s  = f"{r2:.4f}" if not np.isnan(r2) else "—"
                    maes = f"±{mae:.1f}d" if not np.isnan(mae) else "—"
                    is_b  = (mn == best_m)
                    badge = "<span class='badge-best'>✅ BEST</span>" if is_b else                             "<span class='badge-ok'>Alt</span>" if not np.isnan(r2) and r2 > 0.3 else                             "<span class='badge-poor'>Poor</span>"
                    rbg = scfg['bg'] if is_b else '#fff'
                    r2p = max(0, r2*100) if not np.isnan(r2) else 0
                    bc  = '#1B5E20' if r2 > 0.6 else '#F9A825' if r2 > 0.3 else '#E53935' if r2 > 0 else '#CCC'
                    rows_html += (
                        f"<tr style='background:{rbg}'>"
                        f"<td><b style='color:{scfg['color']}'>{sname.split('(')[0].strip()}</b></td>"
                        f"<td>{mn}</td>"
                        f"<td><div style='display:flex;align-items:center;gap:6px'>"
                        f"<div style='background:#EEE;border-radius:3px;height:5px;width:55px'>"
                        f"<div style='height:5px;border-radius:3px;width:{r2p:.0f}%;background:{bc}'></div></div>"
                        f"<span style='font-family:DM Mono,monospace;font-size:0.82rem'>{r2s}</span></div></td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.82rem'>{maes}</td>"
                        f"<td>{badge}</td></tr>")
            st.markdown(
                f"<div style='border:1px solid #D8E8F4;border-radius:12px;overflow:hidden'>"
                f"<table class='model-tbl'><thead><tr>"
                f"<th>Sensor</th><th>Model</th><th>LOO R²</th><th>MAE</th><th>Status</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table></div>",
                unsafe_allow_html=True)
            st.caption("LOO R² = Leave-One-Out R² · MAE = Mean Absolute Error in days")

            mod_rows = []
            for sname, ev_res in model_results.items():
                for ev, mods in ev_res.items():
                    for mn, mfit in mods.items():
                        mod_rows.append({'Sensor':sname,'Event':ev,'Model':mn,
                                         'LOO_R2':round(mfit.get('r2',np.nan),6),
                                         'MAE_days':round(mfit.get('mae',np.nan),3),
                                         'n_seasons':mfit.get('n','—')})
            st.download_button("📥 Download Model Performance CSV",
                               pd.DataFrame(mod_rows).to_csv(index=False),
                               "model_performance_sensors.csv","text/csv",key="ds_dl_mod")

    # ── Sub-tab 4: Sensor Agreement ──────────────────────────
    with ds_tabs[3]:
        st.markdown('<p class="section-title">Inter-Sensor Agreement Statistics</p>',
                    unsafe_allow_html=True)
        if len(sensor_pheno) < 2:
            st.markdown('<div class="banner-info">ℹ️ Upload at least <b>2 sensors</b> to see agreement statistics.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="banner-info">ℹ️ <b>Bias</b> = mean(Sensor B – Sensor A) in days. ' 
                'Positive = Sensor B detects event later. <b>RMSE</b> = root mean square difference. ' 
                '<b>Pearson r</b> = how well both sensors track the same inter-annual variability.</div>',
                unsafe_allow_html=True)
            agree_rows = []
            for pair, ev_stats in agree_stats.items():
                for ev, stats in ev_stats.items():
                    if not stats: continue
                    agree_rows.append({'Pair':pair,'Event':ev,
                        'Bias (d)':stats.get('bias','—'),
                        'RMSE (d)':stats.get('rmse','—'),
                        'MAE (d)': stats.get('mae','—'),
                        'Pearson r':stats.get('r','—'),
                        'p-value':  stats.get('p','—'),
                        'n years':  stats.get('n','—')})
            if agree_rows:
                adf = pd.DataFrame(agree_rows)
                def _cbias(val):
                    try:
                        v = float(val)
                        if abs(v) <= 5:  return 'background-color:#F0FFF4;color:#1B5E20'
                        if abs(v) <= 15: return 'background-color:#FFFBEC;color:#8B6000'
                        return 'background-color:#FFEEEE;color:#8B1A1A'
                    except: return ''
                _style_fn = adf.style.map if hasattr(adf.style, 'map') else adf.style.applymap
                st.dataframe(_style_fn(_cbias, subset=['Bias (d)']),
                             use_container_width=True, hide_index=True)

            ev_sc = st.selectbox("Scatter plot event:", ['SOS','POS','EOS'], key="ds_sc_ev")
            sc_cols = st.columns(min(len(sensor_names)*(len(sensor_names)-1)//2, 3))
            pairs_list = [(sensor_names[i],sensor_names[j])
                          for i in range(len(sensor_names)) for j in range(i+1,len(sensor_names))]
            for col, (sa, sb) in zip(sc_cols, pairs_list):
                fig_sc = plot_bias_scatter(sensor_pheno, DS_SENSORS, sa, sb, ev_sc)
                if fig_sc:
                    col.pyplot(fig_sc, use_container_width=True)
                    plt.close(fig_sc)

            st.markdown('<p class="section-title">Plain-English Interpretation</p>',
                        unsafe_allow_html=True)
            for pair, ev_stats in agree_stats.items():
                for ev, stats in ev_stats.items():
                    if not stats: continue
                    bias = stats.get('bias', 0); r = stats.get('r', 0); rmse = stats.get('rmse', 0)
                    msg = (f"<b>{pair} / {ev}:</b> Bias = {bias:+.1f}d "
                           + ("(excellent ≤5d) " if abs(bias) <= 5 else
                              "(moderate) " if abs(bias) <= 15 else "(large offset) ")
                           + f"· RMSE = {rmse:.1f}d · r = {r:.3f} "
                           + ("(well correlated)" if r > 0.7 else "(moderate)" if r > 0.4 else "(low correlation)"))
                    clr = '#1B5E20' if abs(bias) <= 5 else '#8B6000' if abs(bias) <= 15 else '#8B1A1A'
                    st.markdown(
                        f"<div style='background:#F8FAFF;border-left:3px solid {clr};"
                        f"border-radius:8px;padding:10px 16px;margin:4px 0;font-size:0.84rem'>"
                        f"{msg}</div>", unsafe_allow_html=True)

    # ── Sub-tab 5: Export ────────────────────────────────────
    with ds_tabs[4]:
        st.markdown('<p class="section-title">Export Results</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="banner-good">✅ Choose your preferred export format below. '
            'All formats use the same extracted phenology data.</div>',
            unsafe_allow_html=True)

        # Build combined CSV
        comb_rows = []
        for sname, pheno in sensor_pheno.items():
            t = pheno.copy(); t.insert(0,'Sensor', sname); comb_rows.append(t)
        combined_csv = pd.concat(comb_rows).to_csv(index=False) if comb_rows else ""

        # Build summary stats CSV
        agree_rows_exp = []
        for pair, ev_stats in agree_stats.items():
            for ev, stats in ev_stats.items():
                if not stats: continue
                agree_rows_exp.append({'Pair':pair,'Event':ev,**stats})
        agree_csv = pd.DataFrame(agree_rows_exp).to_csv(index=False) if agree_rows_exp else ""

        # Build HTML report
        cfg_ds = {'start_m':mo_n[start_m],'end_m':mo_n[end_m],
                  'sos_thr':int(sos_thr*100),'eos_thr':int(eos_thr*100)}
        html_r = generate_html_report(sensor_pheno, model_results, agree_stats, cfg_ds)

        st.markdown("#### 📥 Download Options")
        c1, c2, c3 = st.columns(3)

        # Option 1: Styled HTML report
        with c1:
            st.markdown(
                '<div style="background:#F0FFF4;border:1px solid #A5D6A7;border-radius:12px;'
                'padding:16px 18px;text-align:center;min-height:110px">'
                '<div style="font-size:1.6rem">📄</div>'
                '<div style="font-weight:700;color:#1B5E20;font-size:0.88rem;margin:6px 0 4px">'
                'HTML Report</div>'
                '<div style="color:#4A7A50;font-size:0.76rem">Full styled report with tables, '
                'metric definitions &amp; agreement stats</div></div>',
                unsafe_allow_html=True)
            st.download_button(
                "Download HTML",
                html_r,
                f"Phenology_Report_{datetime.now().strftime('%Y%m%d')}.html",
                "text/html", key="ds_dl_html", use_container_width=True)

        # Option 2: Combined phenology CSV
        with c2:
            st.markdown(
                '<div style="background:#EBF3FD;border:1px solid #90CAF9;border-radius:12px;'
                'padding:16px 18px;text-align:center;min-height:110px">'
                '<div style="font-size:1.6rem">📊</div>'
                '<div style="font-weight:700;color:#1565C0;font-size:0.88rem;margin:6px 0 4px">'
                'Phenology CSV</div>'
                '<div style="color:#2A5A9A;font-size:0.76rem">All sensors, all seasons — '
                'SOS/POS/EOS/LOS/Amplitude in one flat table</div></div>',
                unsafe_allow_html=True)
            if combined_csv:
                st.download_button(
                    "Download CSV",
                    combined_csv,
                    f"Phenology_AllSensors_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv", key="ds_dl_csv2", use_container_width=True)

        # Option 3: Agreement stats CSV (only if 2+ sensors)
        with c3:
            st.markdown(
                '<div style="background:#FFF8E1;border:1px solid #FFD080;border-radius:12px;'
                'padding:16px 18px;text-align:center;min-height:110px">'
                '<div style="font-size:1.6rem">📐</div>'
                '<div style="font-weight:700;color:#8B6000;font-size:0.88rem;margin:6px 0 4px">'
                'Agreement Stats CSV</div>'
                '<div style="color:#6A5000;font-size:0.76rem">Bias, RMSE, MAE, Pearson r '
                'between each sensor pair</div></div>',
                unsafe_allow_html=True)
            if agree_csv and len(sensor_pheno) >= 2:
                st.download_button(
                    "Download CSV",
                    agree_csv,
                    f"SensorAgreement_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv", key="ds_dl_agree", use_container_width=True)
            else:
                st.caption("Available with 2+ sensors")

        # Preview table
        if comb_rows:
            st.markdown("#### 📋 Preview — Combined Phenology Table")
            preview_df = pd.concat(comb_rows)[
                ['Sensor','Year','SOS_Date','POS_Date','EOS_Date','LOS_Days',
                 'Amplitude','Peak_NDVI']].copy()
            for dc in ['SOS_Date','POS_Date','EOS_Date']:
                preview_df[dc] = pd.to_datetime(preview_df[dc]).dt.strftime('%b %d %Y')
            preview_df['Amplitude'] = preview_df['Amplitude'].round(3)
            preview_df['Peak_NDVI'] = preview_df['Peak_NDVI'].round(3)
            st.dataframe(preview_df, use_container_width=True, hide_index=True,
                         height=min(400, 40 + 35 * len(preview_df)))



def main():
    # ── Global matplotlib theme ───────────────────────────────────────
    import matplotlib as _mpl
    _mpl.rcParams.update({
        'figure.facecolor':  '#F7FBF8',
        'axes.facecolor':    '#F7FBF8',
        'axes.edgecolor':    '#D4E8D6',
        'axes.labelcolor':   '#0D2016',
        'axes.titlecolor':   '#0D2016',
        'grid.color':        '#C8E6C9',
        'grid.alpha':        0.45,
        'grid.linewidth':    0.7,
        'xtick.color':       '#0D2016',
        'ytick.color':       '#0D2016',
        'text.color':        '#0D2016',
        'font.family':       'sans-serif',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })
    st.markdown("""
    <div class="app-header">
        <h1>🌲 Indian Forest Phenology Assessment</h1>
        <p>Scientific platform for NDVI-based seasonal event extraction, multi-model prediction,
        OLS statistical verification, and climate driver analysis for Indian forest ecosystems.</p>
        <span class="badge">🌱 SOS</span>
        <span class="badge">🌿 POS</span>
        <span class="badge">🍂 EOS</span>
        <span class="badge">📏 LOS</span>
        <span class="badge">Ridge · LOESS · Poly · GPR</span>
        <span class="badge">OLS Verification</span>
        <span class="badge">LOO Cross-Validation</span>
        <span class="badge">🤖 AI Assistant</span>
    </div>
    """, unsafe_allow_html=True)


    # ── SIDEBAR ───────────────────────────────────────────────
    st.sidebar.markdown("## 📂 Upload Data")

    # ── Section 1: Required for phenology ─────────────────────
    st.sidebar.markdown(
        '<div style="background:#E8F5E9;border-left:3px solid #2E7D32;border-radius:6px;'
        'padding:7px 10px;margin-bottom:6px;font-size:0.80rem;font-weight:700;color:#1B5E20">'
        '🌿 REQUIRED — Main Phenology</div>', unsafe_allow_html=True)
    ndvi_file = st.sidebar.file_uploader(
        "Primary NDVI CSV  *(Date + NDVI columns)*",
        type=['csv'], key="ndvi_uploader",
        help="Any CSV with Date and NDVI columns. Used for all phenology tabs.")
    met_file  = st.sidebar.file_uploader(
        "Meteorological CSV  *(NASA POWER daily)*",
        type=['csv'], key="met_uploader",
        help="Continuous daily climate data from NASA POWER. Needed for model training.")

    # ── Section 2: Optional sensor files ──────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="background:#EEF5FF;border-left:3px solid #1565C0;border-radius:6px;'
        'padding:7px 10px;margin-bottom:4px;font-size:0.80rem;font-weight:700;color:#0D47A1">'
        '🛰️ OPTIONAL — Multi-Sensor Comparison</div>', unsafe_allow_html=True)
    st.sidebar.caption(
        "Upload 1–3 sensor CSV files to see their phenology side by side in **Tab 1 · Sensor Compare**. "
        "Even a single sensor shows its extracted events. Comparison stats appear when you upload 2 or more.")

    _SENSOR_DEFS = {
        "Landsat (30m)":    {"key": "LS",  "color": "#E8A020", "bg": "#FFF9EE"},
        "Sentinel-2 (10m)": {"key": "S2",  "color": "#2080E0", "bg": "#EEF5FF"},
        "MODIS (250-500m)": {"key": "MOD", "color": "#00B896", "bg": "#EEFAF7"},
    }
    _sensor_sidebar_files = {}
    for _sname, _scfg in _SENSOR_DEFS.items():
        _lbl_short = _sname.split("(")[0].strip()
        st.sidebar.markdown(
            f'<div style="background:{_scfg["bg"]};border-left:2px solid {_scfg["color"]};'
            f'border-radius:5px;padding:4px 9px;margin-bottom:2px;'
            f'font-size:0.76rem;font-weight:700;color:{_scfg["color"]}">{_sname}</div>',
            unsafe_allow_html=True)
        _f = st.sidebar.file_uploader(
            f"{_lbl_short} CSV", type=['csv'],
            key=f"sb_sensor_{_scfg['key']}", label_visibility="collapsed")
        if _f:
            _sensor_sidebar_files[_sname] = _f

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
                               format_func=lambda m: sm_names[m], key="start_month_sel")
    end_m   = col_em.selectbox("End month",   options=month_opts, index=4,
                               format_func=lambda m: sm_names[m], key="end_month_sel")
    if start_m != end_m:
        if start_m > end_m:
            st.sidebar.info(f"Cross-year window: **{sm_names[start_m]} → {sm_names[end_m]}** "
                            f"(e.g. Jun 2023 → May 2024)")
        else:
            st.sidebar.info(f"Within-year window: **{sm_names[start_m]} → {sm_names[end_m]}**")
    if start_m == end_m:
        st.sidebar.warning("⚠️ Start and end month are the same — no seasons will be detected.")

    min_days = st.sidebar.slider("Minimum season length (days)", 30, 300, 100, 10,
                                 key="min_days_slider")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Detection Sensitivity")
    sos_thr = st.sidebar.slider("SOS threshold  (% of amplitude)", 5, 40, 10, 5,
                                key="sos_thr_slider") / 100.0
    eos_thr = st.sidebar.slider("EOS threshold  (% of amplitude)", 5, 40, 10, 5,
                                key="eos_thr_slider") / 100.0
    st.sidebar.caption(
        f"Current: SOS at **{int(sos_thr*100)}%** · EOS at **{int(eos_thr*100)}%** of each season's NDVI swing.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📈 Preferred Model")
    st.sidebar.caption(
        "The app automatically fits **all** models and picks the best by LOO R². "
        "Your preference is honoured when it's within 0.02 R² of the best.")
    model_opts = {
        "Ridge Regression (default)":      "ridge",
        "LOESS Smoothing":                  "loess",
        "Polynomial Regression (Deg 2)":    "poly2",
        "Polynomial Regression (Deg 3)":    "poly3",
        "Gaussian Process":                 "gpr",
    }
    model_sel = st.sidebar.radio("Preferred model", list(model_opts.keys()), index=0,
                                 key="model_type_radio")
    model_key = model_opts[model_sel]

    if not _LOESS_AVAILABLE:
        st.sidebar.markdown(
            '<div style="background:#FFF8E1;padding:6px 10px;border-radius:6px;'
            'border-left:3px solid #F9A825;font-size:0.78rem;margin-top:4px">'
            '⚠️ <b>statsmodels</b> not installed — LOESS will use a pure-numpy fallback '
            '(univariate only). Install with: <code>pip install statsmodels</code></div>',
            unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🗓️ Climate Window")
    feat_window = st.sidebar.slider("Days before event to average climate", 7, 60, 15, 1,
                                    key="feat_window_slider")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔢 Maximum Features in Model")
    max_features_override = st.sidebar.slider("Max climate variables per model", 1, 4, 3, 1,
                                              key="max_feat_slider")
    if max_features_override >= 2:
        st.sidebar.markdown(
            '<div style="background:#FFF8E1;padding:8px 12px;border-radius:8px;'
            'border-left:3px solid #F9A825;font-size:0.80rem;margin-top:4px">'
            '⚠️ Using 3+ features with fewer than 6 seasons can overfit. '
            'Reduce to 1–2 if R² looks suspiciously high.</div>',
            unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Plot Settings")
    split_yrs = st.sidebar.slider(
        "Split NDVI plot every N years", 4, 15, SPLIT_PLOT_THRESHOLD_YEARS, 1,
        key="split_yrs_slider",
        help="Long datasets are split into panels of this many years for readability.")

    cfg = {"start_month": start_m, "end_month": end_m, "min_days": min_days}

    _fp = (f"{_fp_ndvi}|{_fp_met}|sm={start_m}|em={end_m}|md={min_days}"
           f"|sos={sos_thr:.3f}|eos={eos_thr:.3f}|model={model_key}|win={feat_window}"
           f"|mxf={max_features_override}|spl={split_yrs}")
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
<div class="up-section">
<div class="up-label">📄 File 1 — NDVI CSV</div>
Any CSV with a date column and an NDVI column. Column names are auto-detected.<br>
Example format: <code>Date, NDVI</code> &nbsp;→&nbsp; <code>2016-01-01, 0.42</code>
</div>
<div class="up-section">
<div class="up-label">🌦️ File 2 — Meteorological CSV</div>
Daily climate data. Download free from
<a href="https://power.larc.nasa.gov/data-access-viewer/" target="_blank">NASA POWER</a>
(Daily → Point → your site coordinates → CSV).
<br><i>⚠️ Use continuous <b>daily</b> records — not NDVI-cadence sampled files.</i>
</div>
<div style="margin-top:14px">
<b>v2.1 + AI improvements:</b><br>
<span class="feature-item">🕐 5-day interpolation grid (always)</span>
<span class="feature-item">📆 Cross-year EOS correctly handled</span>
<span class="feature-item">📊 Auto-split plot for long datasets</span>
<span class="feature-item">🤖 Ridge · LOESS · Poly-2 · Poly-3 · GPR</span>
<span class="feature-item">💬 AI Assistant tab (Gemini free)</span>
</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown("""
<style>
@keyframes teaser-shimmer{0%{background-position:-500px 0}100%{background-position:500px 0}}
@keyframes teaser-pulse{0%,100%{transform:scale(1);opacity:.8}50%{transform:scale(1.18);opacity:1}}
@keyframes teaser-fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.teaser-wrap{
    display:flex;align-items:center;gap:18px;
    background:linear-gradient(135deg,#0A1520 0%,#0F2035 60%,#0A1A2E 100%);
    border:1px solid rgba(100,180,255,.22);border-radius:14px;
    padding:16px 22px;position:relative;overflow:hidden;
    animation:teaser-fadein .45s ease both;
    transition:border-color .2s,box-shadow .2s;
}
.teaser-wrap:hover{border-color:rgba(100,180,255,.45);box-shadow:0 4px 28px rgba(0,0,0,.30);}
.teaser-shimmer{position:absolute;top:0;left:0;right:0;height:2.5px;
    background:linear-gradient(90deg,transparent,#60B8FF 40%,#00DCB4 60%,transparent);
    background-size:500px 100%;animation:teaser-shimmer 2.6s linear infinite;border-radius:2px 2px 0 0;}
.teaser-icon{font-size:2rem;flex-shrink:0;animation:teaser-pulse 3s ease-in-out infinite;line-height:1;}
.teaser-body{flex:1;}
.teaser-title{color:#E8F4FF;font-size:.94rem;font-weight:700;margin:0 0 4px;}
.teaser-sub{color:#7BA8CC;font-size:.80rem;line-height:1.55;margin:0 0 10px;}
.teaser-tags{display:flex;gap:6px;flex-wrap:wrap;}
.teaser-tag{font-size:.68rem;font-weight:700;padding:2px 9px;border-radius:20px;border:1px solid;}
.tt1{background:rgba(232,160,32,.18);color:#FFB432;border-color:rgba(232,160,32,.40);}
.tt2{background:rgba(32,128,224,.18);color:#60B8FF;border-color:rgba(32,128,224,.40);}
.tt3{background:rgba(0,184,150,.18);color:#00DCB4;border-color:rgba(0,184,150,.40);}
.tt4{background:rgba(200,160,255,.18);color:#C090FF;border-color:rgba(200,160,255,.40);}
.teaser-badge{flex-shrink:0;text-align:center;}
.teaser-tab-pill{
    background:linear-gradient(135deg,rgba(30,120,255,.22),rgba(0,180,150,.18));
    border:1px solid rgba(100,180,255,.38);border-radius:22px;
    padding:8px 16px;font-size:.76rem;font-weight:700;
    color:#A8D8FF;white-space:nowrap;line-height:1.5;
}
.teaser-tab-pill span{display:block;font-size:.66rem;color:#7BA8CC;font-weight:400;margin-top:2px;}
</style>
<div class="teaser-wrap">
  <div class="teaser-shimmer"></div>
  <div class="teaser-icon">&#x1F6F0;</div>
  <div class="teaser-body">
    <div class="teaser-title">&#x2728; Multi-Sensor Phenology Comparison &mdash; also available!</div>
    <div class="teaser-sub">
      Upload per-sensor NDVI CSVs to compare SOS, POS &amp; EOS across Landsat, Sentinel-2 and MODIS
      with model benchmarking and inter-sensor agreement stats.
    </div>
    <div class="teaser-tags">
      <span class="teaser-tag tt1">&#x1F30D; Landsat 30m</span>
      <span class="teaser-tag tt2">&#x1F6F0; Sentinel-2 10m</span>
      <span class="teaser-tag tt3">&#x1F4E1; MODIS 250&ndash;500m</span>
      <span class="teaser-tag tt4">&#x1F4D0; Inter-sensor agreement</span>
    </div>
  </div>
  <div class="teaser-badge">
    <div class="teaser-tab-pill">&#x1F6F0; Sensor Compare tab
      <span>last tab in the list &#x2192;</span>
    </div>
  </div>
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
            st.sidebar.markdown("---"); st.sidebar.markdown("### 🗺️ Multiple Sites Detected")
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

    met_df     = add_derived_features(met_df, season_start_month=start_m)
    all_params = [c for c in met_df.columns
                  if c not in {'Date','YEAR','MO','DY','DOY','LON','LAT','ELEV'}
                  and pd.api.types.is_numeric_dtype(met_df[c])]
    derived    = [p for p in all_params if p not in raw_params]

    _, _, interp_freq_val = detect_ndvi_cadence(ndvi_df)
    # interp_freq_val is always INTERP_STEP_DAYS = 5
    ndvi_info  = characterize_ndvi_data(ndvi_df)
    met_info   = characterize_met_data(met_df, raw_params)

    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ NDVI loaded — {ndvi_info['n_obs']} observations · {ndvi_info['n_years']} years")
    st.sidebar.success(f"✅ Met loaded — {len(raw_params)} climate parameters")
    if derived:
        st.sidebar.info(f"+ {len(derived)} derived features computed automatically")

    # ── TABS ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Data Quality",
        "🌿 Season Extraction",
        "🏆 Model Results",
        "🎯 Climate Drivers",
        "🔮 Predict",
        "📖 User Guide",
        "🤖 AI Assistant",
        "🛰️ Sensor Compare"])

    icons = {'SOS': '🌱', 'POS': '🌿', 'EOS': '🍂'}

    # ══════════════════════════════════════════════════════════
    # TAB 1 — DATA SUMMARY
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="label">📅 Year Range</div>'
                    f'<div class="value">{ndvi_info["year_range"]}</div>'
                    f'<div class="sub">{ndvi_info["n_obs"]} NDVI observations</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">🌿 Mean NDVI</div>'
                    f'<div class="value">{ndvi_info["ndvi_mean"]}</div>'
                    f'<div class="sub">σ = {ndvi_info["ndvi_std"]} · Range = {ndvi_info["data_range"]}</div></div>',
                    unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="label">📡 Sensor Cadence</div>'
                    f'<div class="value">{ndvi_info["cadence_d"]:.0f}d</div>'
                    f'<div class="sub">Interpolated to {INTERP_STEP_DAYS}-day grid</div></div>',
                    unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="label">🌲 Evergreen Index</div>'
                    f'<div class="value">{ndvi_info["evergreen_index"]}</div>'
                    f'<div class="sub">0 = fully seasonal · 1 = evergreen</div></div>',
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
            st.dataframe(met_summary_df.style.background_gradient(subset=['Mean'], cmap='Greens'),
                         use_container_width=True, hide_index=True)
        # ── Stat summary strip ──────────────────────────────────────────
        st.markdown(
            f'<div class="stat-strip">'
            f'<span class="stat-chip-lg">📅 <b>{ndvi_info["year_range"]}</b></span>'
            f'<span class="stat-chip-lg">🔢 <b>{ndvi_info["n_obs"]}</b> observations</span>'
            f'<span class="stat-chip-lg">🌿 Mean NDVI <b>{ndvi_info["ndvi_mean"]}</b></span>'
            f'<span class="stat-chip-lg">📡 Cadence <b>{ndvi_info["cadence_d"]:.0f}d</b> → 5d grid</span>'
            f'<span class="stat-chip-lg">🌲 Evergreen idx <b>{ndvi_info["evergreen_index"]}</b></span>'
            f'<span class="stat-chip-lg">🌦️ <b>{len(raw_params)}</b> met variables</span>'
            f'</div>',
            unsafe_allow_html=True)
        if derived:
            st.markdown(
                f'<div class="banner-info">ℹ️ In addition to the {len(raw_params)} parameters in your file, '
                f'<b>{len(derived)} derived variables</b> were computed automatically '
                f'(GDD, log-rainfall, VPD, etc.).</div>', unsafe_allow_html=True)

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

        if n_seasons == 0:
            st.error("No complete seasons found."); return
        elif n_seasons == 1:
            st.warning("Only **1 season** extracted. Model training requires ≥ 2 seasons.")
        elif n_seasons == 2:
            st.warning("**2 seasons** extracted — model results are exploratory only.")
        elif n_seasons <= 4:
            st.info(f"**{n_seasons} seasons** — small dataset. Results are indicative.")
        elif n_seasons < 7:
            st.info(f"**{n_seasons} seasons** — usable dataset.")
        else:
            st.success(f"✅ **{n_seasons} growing seasons** extracted.")

        st.markdown(
            '<div class="banner-info">ℹ️ <b>5-day interpolation grid</b> is used throughout. '
            'Long datasets (>8 years) are split into readable panels. '
            '<b>EOS can now extend across the calendar year end</b> (e.g. Nov → Feb).</div>',
            unsafe_allow_html=True)

        # ── SPLIT NDVI PLOTS ──────────────────────────────────────
        ndvi_figs = plot_ndvi_phenology(
            ndvi_df, pheno_df,
            season_window=(start_m, end_m),
            interp_freq=INTERP_STEP_DAYS,
            split_threshold_years=split_yrs)

        if len(ndvi_figs) == 1:
            st.pyplot(ndvi_figs[0][1], use_container_width=True)
            plt.close(ndvi_figs[0][1])
        else:
            st.info(f"Dataset split into **{len(ndvi_figs)} panels** of ≤{split_yrs} years each.")
            for lbl, fig_p in ndvi_figs:
                st.markdown(f"**Years {lbl}**")
                st.pyplot(fig_p, use_container_width=True)
                plt.close(fig_p)

        # ── SEASON TABLE ──────────────────────────────────────────
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

        st.pyplot(plot_pheno_trends(pheno_df))

        # ── Download season table ─────────────────────────────
        dl_cols = [c for c in ['Year','SOS_DOY','POS_DOY','EOS_DOY','LOS_Days',
                                'Peak_NDVI','Amplitude','Base_NDVI','SOS_Date',
                                'POS_Date','EOS_Date'] if c in pheno_df.columns]
        st.download_button('📥 Download Season Table (CSV)',
                           pheno_df[dl_cols].to_csv(index=False),
                           'phenology_table.csv', 'text/csv',
                           key='dl_pheno_tab2b')

        # ── Train models (runs silently here, results shown in Tab 3) ──
        with st.spinner('Training predictive models…'):
            train_df  = make_training_features(pheno_df, met_df, all_params, window=feat_window)
            predictor = UniversalPredictor()
            if not train_df.empty and 'Event' in train_df.columns:
                predictor.train(train_df, all_params, model_key=model_key,
                                user_max_features=max_features_override)
        # ── Check met coverage warnings ──────────────────────────────
        met_audit = audit_met_coverage(met_df, ndvi_df, pheno_df, window=feat_window)
        if met_audit.get('warnings'):
            st.markdown('<p class="section-title">⚠️ Data Quality Warnings</p>', unsafe_allow_html=True)
        for _w in met_audit.get('warnings', []):
            st.markdown(f'<div class="banner-warn">{_w}</div>', unsafe_allow_html=True)

        # Store state for Tab 3
        st.session_state.update({
            'pheno_df': pheno_df, 'met_df': met_df, 'train_df': train_df,
            'predictor': predictor, 'all_params': all_params,
            'raw_params': raw_params, 'ndvi_df': ndvi_df,
            'ndvi_info': ndvi_info, 'met_info': met_info,
            'interp_freq': INTERP_STEP_DAYS,
        })
        st.success(f'✅ {len(pheno_df)} seasons extracted · Models trained · See **Tab 3** for results.')

    # ══════════════════════════════════════════════════════════
    # TAB 3 — MODEL RESULTS
    # ══════════════════════════════════════════════════════════
    with tab3:
        predictor_t3 = st.session_state.get('predictor')
        pheno_df_t3  = st.session_state.get('pheno_df')
        train_df_t3  = st.session_state.get('train_df')
        met_df_t3    = st.session_state.get('met_df')

        if predictor_t3 is None or pheno_df_t3 is None:
            st.markdown(
                '<div class="banner-info">ℹ️ Complete <b>Tab 2 – Season Extraction</b> first '
                'to see model training results here.</div>', unsafe_allow_html=True)
        else:
            # ── Overview KPI row ──────────────────────────────
            st.markdown('<p class="section-title">Model Performance Overview</p>', unsafe_allow_html=True)
            kpi_cols = st.columns(3)
            ev_icons_t3 = {'SOS':'🌱','POS':'🌿','EOS':'🍂'}
            ev_names_t3 = {'SOS':'Start of Season','POS':'Peak of Season','EOS':'End of Season'}
            ev_clrs_t3  = {'SOS':'#2E7D32','POS':'#1565C0','EOS':'#C62828'}

            for kc, ev in zip(kpi_cols, ['SOS','POS','EOS']):
                if ev not in predictor_t3._fits:
                    kc.markdown(
                        f'<div class="metric-card"><div class="label">{ev_icons_t3[ev]} {ev}</div>'
                        f'<div class="value" style="color:#9E9E9E;font-size:1.3rem">—</div>'
                        f'<div class="sub">Not fitted</div></div>', unsafe_allow_html=True)
                    continue
                r2_t3  = predictor_t3.r2.get(ev, np.nan)
                mae_t3 = predictor_t3.mae.get(ev, np.nan)
                nm_t3  = predictor_t3._fits[ev]['best_name']
                clr_t3 = '#1B5E20' if not np.isnan(r2_t3) and r2_t3 > 0.6 else \
                          '#E65100' if not np.isnan(r2_t3) and r2_t3 > 0.3 else '#B71C1C'
                conf   = 'High' if not np.isnan(r2_t3) and r2_t3 > 0.6 else \
                          'Medium' if not np.isnan(r2_t3) and r2_t3 > 0.3 else 'Low'
                conf_badge_cls = 'conf-high' if conf=='High' else 'conf-med' if conf=='Medium' else 'conf-low'
                kc.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{ev_icons_t3[ev]} {ev} — {ev_names_t3[ev]}</div>'
                    f'<div class="value" style="color:{clr_t3}">{r2_t3*100:.1f}%</div>'
                    f'<div class="sub">LOO R² · Best: <b>{nm_t3}</b><br>'
                    f'MAE: ±{mae_t3:.1f} d &nbsp; '
                    f'<span class="{conf_badge_cls}">{conf} confidence</span>'
                    f'</div></div>',
                    unsafe_allow_html=True)

            # ── Professional Model Comparison Table ───────────
            st.markdown('<p class="section-title">All Models Compared — Side by Side</p>', unsafe_allow_html=True)
            st.markdown(
                '<div class="banner-info">ℹ️ Every model is trained and evaluated using '
                '<b>Leave-One-Out cross-validation (LOO R²)</b>. '
                'The model with the highest LOO R² is selected automatically. '
                'Green = best · Yellow = acceptable · Red = poor fit.</div>',
                unsafe_allow_html=True)

            MODEL_NAMES = ['Ridge', 'LOESS', 'Poly-2', 'Poly-3', 'GPR']
            MODEL_DESC  = {
                'Ridge': 'Regularized linear regression. Best for small datasets.',
                'LOESS': 'Local smoothing regression. Good for nonlinear single drivers.',
                'Poly-2': 'Degree-2 polynomial. Captures curved relationships.',
                'Poly-3': 'Degree-3 polynomial. More complex curves; needs ≥6 seasons.',
                'GPR':   'Gaussian Process. Probabilistic, needs ≥5 seasons.',
            }

            for ev in ['SOS', 'POS', 'EOS']:
                if ev not in predictor_t3._fits: continue
                result_t3 = predictor_t3._fits[ev]
                all_mods  = result_t3['all_models']
                best_nm   = result_t3['best_name']
                feats_t3  = result_t3['features']

                st.markdown(
                    f"<div style='margin:20px 0 8px;padding:12px 20px;border-radius:12px;"
                    f"background:{'#F0FFF4' if ev=='SOS' else '#EBF3FD' if ev=='POS' else '#FFF0F0'};"
                    f"border-left:4px solid {ev_clrs_t3[ev]}'>"
                    f"<b style='font-size:1.05rem;color:{ev_clrs_t3[ev]}'>"
                    f"{ev_icons_t3[ev]} {ev_names_t3[ev]} ({ev})</b>"
                    f"<span style='font-size:0.80rem;color:#666;margin-left:10px'>"
                    f"Drivers used: <b>{', '.join(feats_t3) if feats_t3 else '—'}</b></span></div>",
                    unsafe_allow_html=True)

                # Build comparison table rows
                tbl_rows = []
                for mn in MODEL_NAMES:
                    if mn not in all_mods:
                        tbl_rows.append({'Model': mn, 'LOO R²': '—', 'MAE (days)': '—',
                                         'R² %': 0, 'Status': '—', 'Note': 'n/a for this sample size'})
                        continue
                    mf = all_mods[mn]
                    r2v  = mf.get('r2', np.nan)
                    maev = mf.get('mae', np.nan)
                    is_b = (mn == best_nm)
                    r2s  = f"{r2v:.4f}" if not np.isnan(r2v) else "—"
                    maes = f"±{maev:.1f}" if not np.isnan(maev) else "—"
                    status = '✅  BEST (selected)' if is_b else ('🟡  Alternative' if not np.isnan(r2v) and r2v > 0.3 else '🔴  Poor fit' if not np.isnan(r2v) else '—')
                    tbl_rows.append({'Model': mn, 'LOO R²': r2s, 'MAE (days)': maes,
                                     'R² %': float(r2v) if not np.isnan(r2v) else 0.0,
                                     'Status': status, 'Note': MODEL_DESC.get(mn,'')})

                tbl_df = pd.DataFrame(tbl_rows)

                # Render as styled HTML table
                rows_html = ''
                for _, row in tbl_df.iterrows():
                    is_best_row = '✅' in str(row['Status'])
                    bg = f"background:{ev_clrs_t3[ev]}12;font-weight:700" if is_best_row else ""
                    r2_pct = row['R² %']
                    bar_w  = max(0, min(100, r2_pct * 100)) if r2_pct > 0 else 0
                    bar_col = '#1B5E20' if r2_pct > 0.6 else '#F9A825' if r2_pct > 0.3 else '#E53935' if r2_pct > 0 else '#CCC'
                    rows_html += (
                        f"<tr style='{bg}'>"
                        f"<td style='padding:10px 14px;border-bottom:1px solid #EEE;font-weight:700'>{row['Model']}</td>"
                        f"<td style='padding:10px 14px;border-bottom:1px solid #EEE'>"
                        f"<div style='display:flex;align-items:center;gap:8px'>"
                        f"<div style='background:#F0F0F0;border-radius:4px;height:6px;width:70px'>"
                        f"<div style='height:6px;border-radius:4px;width:{bar_w:.0f}%;background:{bar_col}'></div></div>"
                        f"<span>{row['LOO R²']}</span></div></td>"
                        f"<td style='padding:10px 14px;border-bottom:1px solid #EEE'>{row['MAE (days)']}</td>"
                        f"<td style='padding:10px 14px;border-bottom:1px solid #EEE'>{row['Status']}</td>"
                        f"<td style='padding:10px 14px;border-bottom:1px solid #EEE;font-size:0.78rem;color:#666'>{row['Note']}</td>"
                        f"</tr>")

                st.markdown(
                    f"<div style='border:1px solid #E0E8E2;border-radius:12px;overflow:hidden;margin-bottom:8px'>"
                    f"<table style='width:100%;border-collapse:collapse;font-size:0.88rem'>"
                    f"<thead><tr style='background:#0D2016'>"
                    f"<th style='padding:10px 14px;text-align:left;color:#FFFFFF;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.6px'>Model</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#FFFFFF;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.6px'>LOO R²</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#2E5E35;font-size:0.80rem;text-transform:uppercase;letter-spacing:0.5px'>MAE (days)</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#2E5E35;font-size:0.80rem;text-transform:uppercase;letter-spacing:0.5px'>Status</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#2E5E35;font-size:0.80rem;text-transform:uppercase;letter-spacing:0.5px'>When to use</th>"
                    f"</tr></thead><tbody>{rows_html}</tbody></table></div>",
                    unsafe_allow_html=True)

                # Why this model won
                bf_mode = result_t3['best_fit']['mode']
                r2_best = predictor_t3.r2.get(ev, np.nan)
                why_text = {
                    'ridge': f"Ridge was selected because it handles small datasets (n={predictor_t3.n_seasons.get(ev,0)}) well and gives stable, interpretable coefficients.",
                    'loess': "LOESS was selected because it captured the nonlinear relationship better than Ridge.",
                    'poly2': "Polynomial degree-2 was selected because the relationship is curved rather than linear.",
                    'poly3': "Polynomial degree-3 was selected for its higher flexibility on this dataset.",
                    'gpr':   "Gaussian Process was selected for its probabilistic flexibility on this dataset.",
                    'mean':  "No climate driver met the correlation threshold — using historical mean.",
                }.get(bf_mode, "Best model selected by LOO R².")

                conf_label = 'HIGH' if not np.isnan(r2_best) and r2_best > 0.6 else \
                              'MEDIUM' if not np.isnan(r2_best) and r2_best > 0.3 else 'LOW'
                conf_clr   = '#1B5E20' if conf_label == 'HIGH' else '#E65100' if conf_label == 'MEDIUM' else '#B71C1C'
                st.markdown(
                    f"<div style='background:#fff;border:1px solid #D4E8D6;border-radius:10px;"
                    f"padding:12px 18px;margin:4px 0 16px;display:flex;gap:16px;align-items:flex-start'>"
                    f"<div style='flex:0 0 auto'><span style='background:{conf_clr}22;color:{conf_clr};"
                    f"font-size:0.70rem;font-weight:800;letter-spacing:0.5px;padding:3px 10px;"
                    f"border-radius:8px;border:1px solid {conf_clr}44'>{conf_label} CONFIDENCE</span></div>"
                    f"<div style='font-size:0.84rem;color:#2E3A2F'><b>Why {best_nm} won:</b> {why_text}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

            # ── Feature Selection Explanation ──────────────────
            st.markdown('<p class="section-title">Feature Selection — Why Each Climate Variable Was Chosen or Rejected</p>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="banner-info">ℹ️ For each event, climate variables are ranked by '
                'correlation with event timing. Variables must have |Pearson r| ≥ 0.40 to qualify. '
                'Redundant variables (highly correlated with each other, r > 0.97) are excluded to avoid overfitting.'
                '</div>', unsafe_allow_html=True)

            for ev in ['SOS', 'POS', 'EOS']:
                ct = predictor_t3.corr_tables.get(ev)
                if ct is None or len(ct) == 0: continue
                result_t3 = predictor_t3._fits.get(ev)
                if result_t3 is None: continue
                in_model  = set(result_t3['features'])

                st.markdown(
                    f"<div style='margin:16px 0 6px;padding:10px 18px;border-radius:10px;"
                    f"background:{'#F0FFF4' if ev=='SOS' else '#EBF3FD' if ev=='POS' else '#FFF0F0'};"
                    f"border-left:3px solid {ev_clrs_t3[ev]}'>"
                    f"<b style='color:{ev_clrs_t3[ev]}'>{ev_icons_t3[ev]} {ev} — Feature Selection Log</b></div>",
                    unsafe_allow_html=True)

                # Build feature table
                feat_rows_html = ''
                for _, frow in ct.iterrows():
                    fname  = frow['Feature']
                    pr     = frow['Pearson_r']
                    pval   = frow.get('p_value', np.nan)
                    comp   = frow.get('Composite', abs(pr))
                    stars  = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                    if fname in in_model:
                        icon = '✅'; status = 'IN MODEL'; bg = '#F0FFF4'; fc = '#1B5E20'
                        reason = f"Strongest predictor (|r|={abs(pr):.3f}). Statistically significant {stars if stars else '(borderline)'}."
                    elif comp < 0.40:
                        icon = '⬜'; status = 'BELOW THRESHOLD'; bg = '#FAFAFA'; fc = '#888'
                        reason = f"Correlation too weak (|r|={abs(pr):.3f} < 0.40 threshold). Not informative."
                    else:
                        icon = '➖'; status = 'REDUNDANT / NOT ADDED'; bg = '#FFFBEC'; fc = '#8B6000'
                        reason = f"Correlated with a stronger predictor already in the model, or did not improve LOO accuracy."
                    r_dir = '+' if pr >= 0 else ''
                    feat_rows_html += (
                        f"<tr style='background:{bg}'>"
                        f"<td style='padding:9px 14px;border-bottom:1px solid #EEE'>"
                        f"<span style='font-weight:700;color:{fc}'>{icon} {fname}</span></td>"
                        f"<td style='padding:9px 14px;border-bottom:1px solid #EEE;font-family:monospace'>"
                        f"r = {r_dir}{pr:.3f}{stars}</td>"
                        f"<td style='padding:9px 14px;border-bottom:1px solid #EEE;font-size:0.78rem'>"
                        f"p = {pval:.4f} {stars}</td>"
                        f"<td style='padding:9px 14px;border-bottom:1px solid #EEE'>"
                        f"<span style='background:{bg};color:{fc};font-size:0.73rem;font-weight:700;"
                        f"padding:2px 8px;border-radius:6px;border:1px solid {fc}33'>{status}</span></td>"
                        f"<td style='padding:9px 14px;border-bottom:1px solid #EEE;font-size:0.79rem;color:#555'>{reason}</td>"
                        f"</tr>")

                st.markdown(
                    f"<div style='border:1px solid #E0E8E2;border-radius:12px;overflow:hidden;margin-bottom:12px'>"
                    f"<table style='width:100%;border-collapse:collapse;font-size:0.86rem'>"
                    f"<thead><tr style='background:#0D2016'>"
                    f"<th style='padding:10px 14px;text-align:left;color:#FFFFFF;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.6px'>Feature</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#FFFFFF;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.6px'>Correlation</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#2E5E35;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px'>p-value</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#2E5E35;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px'>Decision</th>"
                    f"<th style='padding:10px 14px;text-align:left;color:#2E5E35;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px'>Reason</th>"
                    f"</tr></thead><tbody>{feat_rows_html}</tbody></table></div>",
                    unsafe_allow_html=True)

            # ── Model Equations (3 formats) ──────────────────
            st.markdown('<p class="section-title">Model Equations — Three Formats</p>',
                        unsafe_allow_html=True)
            mo_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                        7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

            for ev in ['SOS', 'POS', 'EOS']:
                if ev not in predictor_t3._fits: continue
                res_t3   = predictor_t3._fits[ev]
                bf_t3    = res_t3['best_fit']
                feats_t3 = res_t3['features']
                r2_t3    = predictor_t3.r2.get(ev, np.nan)
                mae_t3   = predictor_t3.mae.get(ev, np.nan)
                n_t3     = predictor_t3.n_seasons.get(ev, 0)

                with st.expander(f"{ev_icons_t3[ev]} {ev} — {ev_names_t3[ev]} Equation & Interpretation", expanded=False):
                    # Format 1: Mathematical
                    eq_raw = predictor_t3.equation_str(ev, season_start_month=start_m)
                    eq_h   = eq_raw.replace('\n','<br>').replace('  ','&nbsp;&nbsp;')
                    st.markdown(
                        f'<div class="eq-box"><span class="eq-label">📐 Mathematical Form</span>'
                        f'<span class="eq-main">{eq_h}</span>'
                        f'<span class="eq-meta">Model: {res_t3["best_name"]} · '
                        f'n = {n_t3} seasons · LOO R² = {r2_t3:.4f} · MAE = ±{mae_t3:.1f} days</span>'
                        f'</div>', unsafe_allow_html=True)

                    # Format 2: Plain English
                    if bf_t3['mode'] == 'ridge' and bf_t3.get('coef') and feats_t3:
                        coefs_t3  = bf_t3['coef']
                        intercept = bf_t3.get('intercept', 0)
                        base_doy  = int(round(intercept))
                        try:
                            base_date = (pd.Timestamp(f'2024-01-01') + pd.Timedelta(days=base_doy-1)).strftime('%B %d')
                        except Exception:
                            base_date = f"Day {base_doy}"
                        plain_lines = [f"The {ev_names_t3[ev]} baseline (intercept) is around <b>{base_date}</b> (day {base_doy} from Jan 1)."]

                        def _ev_unit(feat):
                            fu = feat.upper()
                            # Temperature variables
                            if any(k in fu for k in ['T2M','TEMP','TMAX','TMIN','T2MDEW','TS']): return '°C'
                            # Precipitation
                            if any(k in fu for k in ['PREC','PPT','RAIN','CPPT','PRECTOT']): return 'mm/day'
                            # Relative humidity
                            if fu.startswith('RH'): return '%'
                            # Soil moisture / wetness (0–1 scale)
                            if any(k in fu for k in ['GWET','SOIL','MOIST']): return '(fraction 0–1)'
                            # Pressure
                            if fu in ('PS','SLP','PRESSURE'): return 'kPa'
                            # Wind speed
                            if 'WS2M' in fu or fu.startswith('WS'): return 'm/s'
                            # Wind direction
                            if 'WD2M' in fu or fu.startswith('WD'): return '° (direction)'
                            # Specific humidity
                            if 'QV2M' in fu: return 'g/kg'
                            # VPD
                            if 'VPD' in fu: return 'kPa'
                            # Radiation / irradiance / PAR — all MJ/m²/day
                            if any(k in fu for k in ['RAD','ALLSKY','CLRSKY','TOA_SW','PAR','UVA','UVB','SW_DWN','LW_DWN']): return 'MJ/m²/day'
                            # GDD
                            if 'GDD' in fu: return '°C·days'
                            return 'unit'

                        def _ev_description(feat, coef, ev_name):
                            fu    = feat.upper()
                            unit  = _ev_unit(feat)
                            absc  = abs(coef)
                            direction = 'delays' if coef > 0 else 'advances'
                            arrow_sym = '⏩' if coef < 0 else '⏪'
                            # Build a friendly variable description
                            var_desc = feat
                            if 'T2M_MIN' in fu:  var_desc = 'minimum temperature (T2M_MIN)'
                            elif 'T2M_MAX' in fu: var_desc = 'maximum temperature (T2M_MAX)'
                            elif 'T2M_RANGE' in fu: var_desc = 'diurnal temperature range (T2M_RANGE)'
                            elif 'T2M' == fu:     var_desc = 'mean temperature (T2M)'
                            elif 'T2MDEW' in fu:  var_desc = 'dew-point temperature (T2MDEW)'
                            elif 'TS' == fu:      var_desc = 'skin temperature (TS)'
                            elif 'PRECTOTCORR' in fu: var_desc = 'daily rainfall (PRECTOTCORR)'
                            elif 'RH2M' in fu:    var_desc = 'relative humidity (RH2M)'
                            elif 'GWETTOP' in fu: var_desc = 'surface soil wetness (GWETTOP)'
                            elif 'GWETROOT' in fu: var_desc = 'root-zone soil wetness (GWETROOT)'
                            elif 'GWETPROF' in fu: var_desc = 'profile soil moisture (GWETPROF)'
                            elif 'PS' == fu:      var_desc = 'surface pressure (PS)'
                            elif 'WS2M' in fu:    var_desc = 'wind speed at 2 m (WS2M)'
                            elif 'WD2M' in fu:    var_desc = 'wind direction at 2 m (WD2M)'
                            elif 'QV2M' in fu:    var_desc = 'specific humidity (QV2M)'
                            elif 'TOA_SW_DWN' in fu: var_desc = 'top-of-atmosphere irradiance (TOA_SW_DWN)'
                            elif 'CLRSKY_SFC_SW_DWN' in fu: var_desc = 'clear-sky surface irradiance (CLRSKY_SFC_SW_DWN)'
                            elif 'ALLSKY_SFC_SW_DWN' in fu: var_desc = 'all-sky surface irradiance (ALLSKY_SFC_SW_DWN)'
                            elif 'ALLSKY_SFC_PAR_TOT' in fu: var_desc = 'photosynthetically active radiation (PAR)'
                            elif 'ALLSKY_SFC_UVA' in fu: var_desc = 'surface UVA irradiance'
                            elif 'GDD' in fu:     var_desc = 'growing degree days (GDD_cum)'
                            return (
                                f"Every <b>+1 {unit}</b> increase in <b>{var_desc}</b> "
                                f"{arrow_sym} {ev_name} <b>{direction} by {absc:.2f} days</b>."
                            )

                        for feat, coef in zip(feats_t3, coefs_t3):
                            plain_lines.append(_ev_description(feat, coef, ev_names_t3[ev]))

                        # Add a combined summary sentence
                        n_delay   = sum(1 for c in coefs_t3 if c > 0)
                        n_advance = sum(1 for c in coefs_t3 if c < 0)
                        if n_delay > 0 and n_advance > 0:
                            plain_lines.append(f"<i>When all drivers act together, their effects combine linearly.</i>")
                        elif n_advance == len(coefs_t3):
                            plain_lines.append(f"<i>All predictors work in the same direction — higher values consistently advance {ev_names_t3[ev]}.</i>")
                        elif n_delay == len(coefs_t3):
                            plain_lines.append(f"<i>All predictors work in the same direction — higher values consistently delay {ev_names_t3[ev]}.</i>")

                        st.markdown(
                            f'<div class="result-box"><span class="result-title">📖 Plain English Interpretation</span>'
                            f'<span class="result-main">{"<br>".join(plain_lines)}</span></div>',
                            unsafe_allow_html=True)

                    # Format 3: Example calculation (only for Ridge)
                    if bf_t3['mode'] == 'ridge' and bf_t3.get('coef') and feats_t3 and train_df_t3 is not None:
                        ex_inputs = {}
                        ex_lines  = []
                        for feat in feats_t3:
                            if feat in train_df_t3.columns:
                                sub_ev = train_df_t3[train_df_t3['Event']==ev]
                                vals_  = sub_ev[feat].dropna() if len(sub_ev) > 0 else train_df_t3[feat].dropna()
                                ex_val = float(vals_.mean()) if len(vals_) > 0 else 0.0
                            else:
                                ex_val = 0.0
                            ex_inputs[feat] = ex_val
                            ex_lines.append(f"  {feat} = {ex_val:.3f}")
                        pred_res = predictor_t3.predict(ex_inputs, ev, year=2026, season_start_month=start_m)
                        if pred_res:
                            pred_date = pred_res['date'].strftime('%B %d, 2026')
                            pred_doy  = pred_res['doy']
                            st.markdown(
                                f'<div class="eq-box"><span class="eq-label">🧮 Example Calculation (using historical averages)</span>'
                                f'<span class="eq-main">Input values:<br>{"<br>".join(ex_lines)}<br><br>'
                                f'→ Predicted {ev} = <b>{pred_date}</b> (Day {pred_doy})<br>'
                                f'→ Confidence interval: ±{mae_t3:.1f} days</span>'
                                f'<span class="eq-meta">This example uses historical average climate values as inputs.</span>'
                                f'</div>', unsafe_allow_html=True)

            # ── Observed vs Predicted plot ────────────────────
            fig_s_t3 = plot_obs_vs_pred(predictor_t3, train_df_t3)
            if fig_s_t3:
                st.markdown('<p class="section-title">Observed vs Predicted — LOO Validation</p>',
                            unsafe_allow_html=True)
                st.pyplot(fig_s_t3, use_container_width=True)
                plt.close(fig_s_t3)

            # ── Download model coefficients ───────────────────
            coef_df_t3 = predictor_t3.export_coefficients(season_start_month=start_m)
            st.download_button("📥 Download Model Coefficients (CSV)",
                               coef_df_t3.to_csv(index=False),
                               "model_coefficients.csv", "text/csv",
                               key="dl_coef_tab3")


    # TAB 4 — CLIMATE DRIVERS
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<p class="section-title">Climate–Phenology Correlations</p>',
                    unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        pheno_df_ss  = st.session_state.get('pheno_df')
        if predictor_ss is None:
            st.info("Complete the Season Extraction step first."); return
        fig_c = plot_correlation_summary(predictor_ss)
        if fig_c: st.pyplot(fig_c, use_container_width=True)
        small_n_events = [(ev, predictor_ss.n_seasons.get(ev, 0))
                         for ev in ['SOS','POS','EOS']
                         if 0 < predictor_ss.n_seasons.get(ev, 0) <= 3]
        if small_n_events:
            ev_strs = ", ".join(f"{ev} (n={n})" for ev, n in small_n_events)
            st.markdown(
                f'<div class="banner-error">🔴 <b>Correlation values are not meaningful for: {ev_strs}.</b> '
                f'With n≤3, Pearson r is mathematically constrained and cannot be interpreted causally.'
                f'</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Full Correlation Table</p>', unsafe_allow_html=True)
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
        _met  = st.session_state.get('met_df')
        _ndvi = st.session_state.get('ndvi_df')
        _rp   = st.session_state.get('raw_params', [])
        _if   = INTERP_STEP_DAYS
        if _met is not None and _ndvi is not None:
            figs_l = plot_met_with_ndvi(_met, _ndvi, _rp, pheno_df_ss, interp_freq=_if)
            if figs_l:
                for s_lbl, f_m in figs_l:
                    st.markdown(f"**Season {s_lbl}**")
                    st.pyplot(f_m, use_container_width=True); plt.close(f_m)
            else:
                st.info("No complete seasons with overlapping climate data found.")

    # ══════════════════════════════════════════════════════════
    # TAB 4 continued — Driver Sensitivity (appends to tab4)
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<p class="section-title">Climate Driver Sensitivity Analysis</p>',
                    unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        train_df_ss  = st.session_state.get('train_df')
        if predictor_ss is None:
            st.info("Complete the Season Extraction step first.")
        else:
            ridge_events = [ev for ev in ['SOS','POS','EOS']
                            if ev in predictor_ss._fits
                            and predictor_ss._fits[ev]['best_fit'].get('mode') == 'ridge'
                            and predictor_ss._fits[ev]['best_fit'].get('coef')]
            if not ridge_events:
                st.markdown(
                    '<div class="banner-warn">⚠️ Sensitivity analysis requires at least one Ridge '
                    'model as the auto-selected best. Currently the best models for all events '
                    'are non-Ridge types. Switch preferred model to Ridge or upload more data.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="banner-info">ℹ️ Sensitivity shows how much each variable shifts '
                    'each event (days per 1σ), based on fitted Ridge coefficients × observed variability.'
                    '</div>', unsafe_allow_html=True)
                sensitivity, dominants = compute_sensitivity_analysis(predictor_ss, train_df_ss)
                if not sensitivity:
                    st.warning("No sensitivity data available.")
                else:
                    st.markdown('<p class="section-title">Dominant Driver per Event</p>',
                                unsafe_allow_html=True)
                    ev_colors_hex  = {'SOS': '#E8F5E9', 'POS': '#E3F2FD', 'EOS': '#FFF3E0'}
                    ev_border_hex  = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#E65100'}
                    ev_icons       = {'SOS': '🌱', 'POS': '🌿', 'EOS': '🍂'}
                    ev_labels_full = {'SOS': 'Start of Season', 'POS': 'Peak of Season', 'EOS': 'End of Season'}
                    dom_cols = st.columns(len(ridge_events))
                    for col_d, ev in zip(dom_cols, ridge_events):
                        dom = dominants.get(ev)
                        sens_ev = sensitivity.get(ev, {})
                        if dom:
                            d_days = dom['days_per_std']
                            sign   = '+' if d_days > 0 else ''
                            dirstr = 'delays' if d_days > 0 else 'advances'
                            col_d.markdown(
                                f"<div style='background:{ev_colors_hex[ev]};padding:16px;border-radius:10px;"
                                f"border-left:4px solid {ev_border_hex[ev]};margin:6px 0'>"
                                f"<div style='font-size:0.78rem;color:#666;font-weight:600'>"
                                f"{ev_icons[ev]} {ev_labels_full[ev]}</div>"
                                f"<div style='font-size:1.4rem;font-weight:700;color:{ev_border_hex[ev]};margin:4px 0'>"
                                f"{dom['feature']}</div>"
                                f"<div style='font-size:0.85rem;color:#555'>"
                                f"↑ 1σ increase {dirstr} {ev} by <b>{sign}{d_days:.1f} days</b></div>"
                                f"<div style='font-size:0.78rem;color:#888;margin-top:4px'>"
                                f"{len(sens_ev)} variable(s) in model</div></div>",
                                unsafe_allow_html=True)
                    fig_hm = plot_sensitivity_heatmap(sensitivity, predictor_ss, train_df_ss)
                    if fig_hm: st.pyplot(fig_hm, use_container_width=True); plt.close(fig_hm)
                    fig_dc = plot_driver_dominance_cards(sensitivity, dominants)
                    if fig_dc: st.pyplot(fig_dc, use_container_width=True); plt.close(fig_dc)
                    available_evs = [ev for ev in ['SOS', 'POS', 'EOS'] if ev in sensitivity]
                    radar_ev = st.radio(
                        "Highlight event on radar:", available_evs, horizontal=True,
                        key="radar_ev_sel",
                        format_func=lambda e: {'SOS': '🌱 SOS', 'POS': '🌿 POS', 'EOS': '🍂 EOS'}[e])
                    fig_rd = plot_radar_chart(sensitivity, selected_event=radar_ev)
                    if fig_rd: st.pyplot(fig_rd, use_container_width=True); plt.close(fig_rd)
                    else: st.info("Radar chart needs ≥ 3 climate features in the model.")
                    rows = []
                    for ev in ['SOS', 'POS', 'EOS']:
                        if ev not in sensitivity: continue
                        for feat, vals in sensitivity[ev].items():
                            rows.append({
                                'Event': ev, 'Climate Variable': feat,
                                'Coefficient': vals['coef'], 'Feature Std (σ)': vals['std'],
                                'Days per 1σ ↑': vals['days_per_std'],
                                '% of mean target': f"{vals['pct_of_mean']:+.1f}%",
                                'Effect': f"↑ 1σ {vals['direction']} {ev} by {abs(vals['days_per_std']):.1f} days",
                            })
                    if rows:
                        sdf = pd.DataFrame(rows)
                        st.dataframe(sdf, use_container_width=True, hide_index=True)
                        st.download_button("📥 Download Sensitivity Table (CSV)",
                                           sdf.to_csv(index=False), "sensitivity_analysis.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════
    # TAB 5 — PREDICT
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<p class="section-title">Future Season Prediction</p>', unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        train_df_ss  = st.session_state.get('train_df')
        pheno_ss     = st.session_state.get('pheno_df')
        if predictor_ss is None:
            st.info("⬅️ Complete Season Extraction (Tab 2) first."); return

        st.markdown(
            '<div class="banner-info">Enter the expected climate conditions for your target year. '
            'Values are pre-filled with <b>historical averages</b> — change them to reflect '
            'your forecast scenario. Results include confidence levels and historical comparison.'
            '</div>', unsafe_allow_html=True)

        mo = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
              7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        ev_colors_hex  = {'SOS': '#F0FFF4', 'POS': '#EBF3FD', 'EOS': '#FFF0F0'}
        ev_border_hex  = {'SOS': '#2E7D32', 'POS': '#1565C0', 'EOS': '#C62828'}
        ev_labels_full = {'SOS': '🌱 Start of Season (SOS)',
                          'POS': '🌿 Peak of Season (POS)',
                          'EOS': '🍂 End of Season (EOS)'}
        ev_inputs  = {ev: {} for ev in ['SOS', 'POS', 'EOS']}
        ev_defaults= {ev: {} for ev in ['SOS', 'POS', 'EOS']}
        any_model  = False

        # Build full feature list from training data (all available params)
        _meta_cols = {'Year','Event','Target_DOY','LOS_Days','Peak_NDVI','Season_Start'}
        _all_train_feats = []
        if train_df_ss is not None and 'Event' in train_df_ss.columns:
            _all_train_feats = [c for c in train_df_ss.columns
                                if c not in _meta_cols
                                and pd.api.types.is_numeric_dtype(train_df_ss[c])
                                and train_df_ss[c].std() > 1e-8]

        def _get_unit(fname):
            fu = fname.upper()
            if any(k in fu for k in ['T2M','TEMP','TMAX','TMIN','CT2M','DTR']): return '°C'
            if any(k in fu for k in ['PPT','PREC','RAIN','CPPT','LOG_P']): return 'mm'
            if 'RH' in fu: return '%'
            if 'VPD' in fu: return 'kPa'
            if 'RAD' in fu or 'ALLSKY' in fu: return 'MJ/m²'
            if 'WS' in fu: return 'm/s'
            if 'GDD' in fu: return '°C·d'
            if 'SPEI' in fu: return 'index'
            if 'MSI' in fu: return 'index'
            return 'unit'

        def _get_feat_default(feat, ev):
            if train_df_ss is None or feat not in train_df_ss.columns: return 0.0
            ev_sub = train_df_ss[train_df_ss['Event'] == ev]
            vals_  = ev_sub[feat].dropna() if len(ev_sub) > 0 else train_df_ss[feat].dropna()
            return float(vals_.mean()) if len(vals_) > 0 else 0.0

        for ev in ['SOS', 'POS', 'EOS']:
            if ev not in predictor_ss._fits: continue
            result   = predictor_ss._fits[ev]
            best_fit = result['best_fit']
            if best_fit.get('mode') not in ('ridge','loess','poly2','poly3','gpr'): continue
            model_feats = result['features']
            if not model_feats: continue
            any_model = True
            r2  = predictor_ss.r2.get(ev, 0)
            mae = predictor_ss.mae.get(ev, 0)
            best_name = result['best_name']

            hist_date_str = ""
            if pheno_ss is not None and f'{ev}_Date' in pheno_ss.columns:
                ev_dates = pheno_ss[f'{ev}_Date'].dropna()
                if len(ev_dates) > 0:
                    med_m = int(ev_dates.dt.month.median())
                    med_d = int(ev_dates.dt.day.median())
                    hist_date_str = f"{mo[med_m]} {med_d}"

            conf_label = 'HIGH' if r2 > 0.6 else 'MEDIUM' if r2 > 0.3 else 'LOW'
            conf_clr   = '#1B5E20' if conf_label == 'HIGH' else '#E65100' if conf_label == 'MEDIUM' else '#B71C1C'

            st.markdown(
                f"<div style='background:{ev_colors_hex[ev]};padding:14px 20px;border-radius:12px;"
                f"border-left:4px solid {ev_border_hex[ev]};margin:14px 0 6px'>"
                f"<b style='font-size:1.0rem;color:{ev_border_hex[ev]}'>{ev_labels_full[ev]}</b>"
                f"&nbsp;&nbsp;<span style='font-size:0.78rem;color:#666'>"
                f"Model: <b>{best_name}</b> · "
                f"<span style='color:{conf_clr};font-weight:700'>{conf_label} confidence</span> "
                f"(R²={r2:.0%}, ±{mae:.0f}d) · Uses: <b>{', '.join(model_feats)}</b>"
                f"{' · Historical avg: <b>' + hist_date_str + '</b>' if hist_date_str else ''}"
                f"</span></div>",
                unsafe_allow_html=True)

            # ── Section A: Model prediction inputs (used directly) ──
            st.markdown(
                f"<div style='font-size:0.76rem;font-weight:800;color:#1B5E20;"
                f"text-transform:uppercase;letter-spacing:0.6px;margin:8px 0 4px'>"
                f"✅ Model Inputs — These values are used in the prediction equation</div>",
                unsafe_allow_html=True)
            col_list = st.columns(min(len(model_feats), 4))
            for idx, f in enumerate(model_feats):
                default = _get_feat_default(f, ev)
                ev_defaults[ev][f] = default
                is_sum  = any(k in f.upper() for k in ACCUM_KEYWORDS)
                _unit   = _get_unit(f)
                vmin = vmax = None
                if train_df_ss is not None and f in train_df_ss.columns:
                    col_vals = train_df_ss[f].dropna()
                    if len(col_vals) >= 2: vmin, vmax = float(col_vals.min()), float(col_vals.max())
                with col_list[idx % len(col_list)]:
                    ev_inputs[ev][f] = st.number_input(
                        f"🔑 {f} ({_unit})", value=round(default, 3), format="%.3f",
                        key=f"inp_{ev}_{f}",
                        help=((('Total (sum)' if is_sum else str(feat_window)+'-day average')
                               + ' before expected ' + ev + '. Avg: ' + str(round(default,3))
                               + (' | Range: ['+str(round(vmin,2))+' – '+str(round(vmax,2))+']' if vmin is not None else ''))))

            # ── Section B: Additional climate context (not in model, shown for awareness) ──
            other_feats = [f for f in _all_train_feats if f not in model_feats]
            if other_feats:
                with st.expander(
                    f"📋 View all {len(other_feats)} other available climate variables for {ev} "
                    f"(not used in prediction — shown for reference only)", expanded=False):
                    st.markdown(
                        '<div class="banner-info" style="font-size:0.82rem">'
                        'ℹ️ These variables were <b>available</b> but were <b>not selected</b> '
                        'for this model (either below correlation threshold or redundant with '
                        'selected features). They do not affect the prediction — but you can '
                        'review their historical values below.</div>', unsafe_allow_html=True)
                    other_cols = st.columns(min(len(other_feats), 4))
                    for idx2, f2 in enumerate(other_feats):
                        default2 = _get_feat_default(f2, ev)
                        _unit2   = _get_unit(f2)
                        vmin2 = vmax2 = None
                        if train_df_ss is not None and f2 in train_df_ss.columns:
                            cv2 = train_df_ss[f2].dropna()
                            if len(cv2) >= 2: vmin2, vmax2 = float(cv2.min()), float(cv2.max())
                        with other_cols[idx2 % len(other_cols)]:
                            st.number_input(
                                f"📊 {f2} ({_unit2})", value=round(default2, 3), format="%.3f",
                                key=f"ctx_{ev}_{f2}", disabled=True,
                                help=("Avg: " + str(round(default2,3)) + (" | Range:["+str(round(vmin2,2))+"-"+str(round(vmax2,2))+"]" if vmin2 is not None else "") + " | Reference only"))

        if not any_model:
            st.markdown('<div class="banner-warn">⚠️ No predictive models fitted. Upload more data.</div>',
                        unsafe_allow_html=True); return

        st.markdown("---")
        pred_year = st.number_input("Year to predict for", 2020, 2050, 2026, key="pred_year_input")

        if st.button("▶  Run Prediction", type="primary"):
            results = {}
            for ev in ['SOS', 'POS', 'EOS']:
                res = predictor_ss.predict(ev_inputs.get(ev, {}), ev, pred_year,
                                           season_start_month=start_m)
                if res: results[ev] = res

            if results:
                # Order correction
                order_warns = []
                if 'SOS' in results and 'POS' in results:
                    if results['POS']['rel_days'] <= results['SOS']['rel_days']:
                        fb = (int(round(pheno_ss['POS_Target'].mean()))
                              if pheno_ss is not None and 'POS_Target' in pheno_ss.columns
                              else results['SOS']['rel_days'] + 90)
                        corrected = max(fb, results['SOS']['rel_days'] + 14)
                        nd = datetime(pred_year, start_m, 1) + timedelta(days=corrected)
                        results['POS'].update({'rel_days': corrected, 'doy': nd.timetuple().tm_yday, 'date': nd})
                        order_warns.append("Peak adjusted (was predicted before start)")
                if 'POS' in results and 'EOS' in results:
                    if results['EOS']['rel_days'] <= results['POS']['rel_days']:
                        fb = (int(round(pheno_ss['EOS_Target'].mean()))
                              if pheno_ss is not None and 'EOS_Target' in pheno_ss.columns
                              else results['POS']['rel_days'] + 90)
                        corrected = max(fb, results['POS']['rel_days'] + 14)
                        nd = datetime(pred_year, start_m, 1) + timedelta(days=corrected)
                        results['EOS'].update({'rel_days': corrected, 'doy': nd.timetuple().tm_yday, 'date': nd})
                        order_warns.append("End adjusted (was predicted before peak)")
                if order_warns:
                    st.markdown(f'<div class="banner-warn">ℹ️ {" · ".join(order_warns)}.</div>',
                                unsafe_allow_html=True)

                # ── Actionable Prediction Cards ──────────────────
                st.markdown('<p class="section-title">Prediction Results</p>', unsafe_allow_html=True)
                ev_icons_p = {'SOS':'🌱','POS':'🌿','EOS':'🍂'}
                ev_names_p = {'SOS':'Start of Season','POS':'Peak of Season','EOS':'End of Season'}
                ev_clrs_p  = {'SOS':'#2E7D32','POS':'#1565C0','EOS':'#C62828'}

                for ev, res in results.items():
                    r2_r    = res.get('r2', np.nan)
                    mae_r   = res.get('mae', np.nan)
                    model_r = res.get('model', '—')
                    pred_dt = res['date']
                    pred_doy= res['doy']
                    conf_l  = 'HIGH' if not np.isnan(r2_r) and r2_r > 0.6 else                                'MEDIUM' if not np.isnan(r2_r) and r2_r > 0.3 else 'LOW'
                    conf_c  = '#1B5E20' if conf_l == 'HIGH' else '#E65100' if conf_l == 'MEDIUM' else '#B71C1C'
                    ev_clr  = ev_clrs_p[ev]

                    # Historical comparison
                    hist_str = ""; diff_str = ""
                    if pheno_ss is not None and f'{ev}_Date' in pheno_ss.columns:
                        ev_dates = pheno_ss[f'{ev}_Date'].dropna()
                        if len(ev_dates) > 0:
                            med_m = int(ev_dates.dt.month.median())
                            med_d = int(ev_dates.dt.day.median())
                            hist_str = f"{mo[med_m]} {med_d}"
                            try:
                                hist_dt = datetime(pred_year, med_m, med_d)
                                diff_d  = (pred_dt - hist_dt).days
                                diff_str = (f"{abs(diff_d)} days <b>earlier</b> than historical average"
                                           if diff_d < 0 else
                                           f"{diff_d} days <b>later</b> than historical average"
                                           if diff_d > 0 else "same as historical average")
                            except Exception:
                                diff_str = ""

                    # Build "why" explanation from inputs vs defaults
                    why_parts = []
                    if ev in predictor_ss._fits:
                        res_ev = predictor_ss._fits[ev]
                        if res_ev['best_fit']['mode'] == 'ridge' and res_ev['best_fit'].get('coef'):
                            for feat, coef in zip(res_ev['features'], res_ev['best_fit']['coef']):
                                inp_val = ev_inputs[ev].get(feat, 0)
                                def_val = ev_defaults[ev].get(feat, 0)
                                diff_v  = inp_val - def_val
                                if abs(diff_v) < 0.001: continue
                                effect  = coef * diff_v
                                eff_dir = 'later' if effect > 0 else 'earlier'
                                fu = feat.upper()
                                unit = '°C' if any(k in fu for k in ['T2M','TEMP']) else                                        'mm' if any(k in fu for k in ['PPT','PREC','RAIN']) else 'unit'
                                why_parts.append(
                                    f"<b>{feat}</b>: {'+' if diff_v > 0 else ''}{diff_v:.2f} {unit} "
                                    f"vs average → {eff_dir} by <b>{abs(effect):.1f} days</b>")

                    why_html = ("<br>".join(why_parts)) if why_parts else "Using historical average conditions."

                    st.markdown(
                        f"<div style='border:2px solid {ev_clr};border-radius:16px;overflow:hidden;margin:12px 0'>"
                        # Header
                        f"<div style='background:{ev_clr};padding:14px 20px;display:flex;justify-content:space-between;align-items:center'>"
                        f"<span style='color:#fff;font-size:1.05rem;font-weight:800'>{ev_icons_p[ev]} {ev_names_p[ev]} ({ev})</span>"
                        f"<span style='background:rgba(255,255,255,0.25);color:#fff;font-size:0.72rem;"
                        f"font-weight:800;padding:3px 12px;border-radius:20px;letter-spacing:0.5px'>"
                        f"{conf_l} CONFIDENCE</span></div>"
                        # Body
                        f"<div style='padding:18px 22px;background:#fff'>"
                        # Predicted date - big
                        f"<div style='display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;margin-bottom:14px'>"
                        f"<div><div style='font-size:0.72rem;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:0.5px'>Predicted Date</div>"
                        f"<div style='font-size:2rem;font-weight:800;color:{ev_clr};line-height:1.1'>{pred_dt.strftime('%b %d')}</div>"
                        f"<div style='font-size:0.82rem;color:#666'>{pred_dt.strftime('%Y')} · Day {pred_doy}</div></div>"
                        f"<div><div style='font-size:0.72rem;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:0.5px'>Confidence Interval</div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:#333'>±{mae_r:.0f} days</div>"
                        f"<div style='font-size:0.78rem;color:#888'>Typical model error</div></div>"
                        f"<div><div style='font-size:0.72rem;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:0.5px'>Model Accuracy</div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:{conf_c}'>{r2_r*100:.1f}%</div>"
                        f"<div style='font-size:0.78rem;color:#888'>LOO R² · {model_r}</div></div>"
                        + (f"<div><div style='font-size:0.72rem;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:0.5px'>Historical Average</div>"
                           f"<div style='font-size:1.1rem;font-weight:700;color:#444'>{hist_str}</div>"
                           f"<div style='font-size:0.78rem;color:#555'>{diff_str}</div></div>" if hist_str else "")
                        + f"</div>"
                        # Why card
                        + (f"<div style='background:#F8FBFF;border-top:1px solid #E8EEF2;padding:12px 22px'>"
                           f"<div style='font-size:0.72rem;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px'>Why this prediction?</div>"
                           f"<div style='font-size:0.84rem;color:#333'>{why_html}</div></div>" if why_parts else "")
                        + f"</div></div>",
                        unsafe_allow_html=True)

                # ── Season Summary Box ─────────────────────────
                if 'SOS' in results and 'EOS' in results:
                    sd = results['SOS']['date']; ed = results['EOS']['date']
                    los = (ed - sd).days if ed >= sd else (ed - sd).days + 365
                    gu  = (results['POS']['date'] - results['SOS']['date']).days if 'POS' in results else None
                    sen = (results['EOS']['date'] - results['POS']['date']).days if 'POS' in results else None

                    # Historical comparison
                    hist_los = None
                    if pheno_ss is not None and 'LOS_Days' in pheno_ss.columns:
                        hist_los = float(pheno_ss['LOS_Days'].median())

                    diff_los = f" ({'+' if los-hist_los>=0 else ''}{los-hist_los:.0f}d vs historical)" if hist_los else ""

                    st.markdown(
                        f"<div style='background:#F5F8F5;border:1px solid #D4E8D6;border-radius:14px;"
                        f"padding:18px 22px;margin:16px 0'>"
                        f"<div style='font-size:0.80rem;font-weight:800;color:#2E5E35;"
                        f"text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px'>📊 Season Summary for {pred_year}</div>"
                        f"<div style='display:flex;gap:24px;flex-wrap:wrap'>"
                        f"<div><div style='font-size:0.72rem;color:#888;font-weight:600'>Total Season Length</div>"
                        f"<div style='font-size:1.4rem;font-weight:800;color:#1B5E20'>{los} days</div>"
                        f"<div style='font-size:0.78rem;color:#777'>{diff_los}</div></div>"
                        + (f"<div><div style='font-size:0.72rem;color:#888;font-weight:600'>Green-up Phase (SOS→POS)</div>"
                           f"<div style='font-size:1.4rem;font-weight:800;color:#1565C0'>{gu} days</div></div>" if gu else "")
                        + (f"<div><div style='font-size:0.72rem;color:#888;font-weight:600'>Senescence Phase (POS→EOS)</div>"
                           f"<div style='font-size:1.4rem;font-weight:800;color:#C62828'>{sen} days</div></div>" if sen else "")
                        + f"</div></div>",
                        unsafe_allow_html=True)

                # Download
                out = pd.DataFrame({
                    'Event':        list(results.keys()),
                    'Predicted Date': [r['date'].strftime('%Y-%m-%d') for r in results.values()],
                    'Day of Year':  [r['doy'] for r in results.values()],
                    'Model Used':   [r.get('model','—') for r in results.values()],
                    'R² (LOO)':     [round(r['r2'], 3) for r in results.values()],
                    'MAE (days)':   [round(r['mae'], 1) for r in results.values()],
                    'Confidence':   ['HIGH' if r['r2'] > 0.6 else 'MEDIUM' if r['r2'] > 0.3 else 'LOW'
                                     for r in results.values()],
                })
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Predictions (CSV)", out.to_csv(index=False),
                                   "predictions.csv", "text/csv", key="dl_pred_tab5")


    # ══════════════════════════════════════════════════════════
    # TAB 6 — USER GUIDE
    # ══════════════════════════════════════════════════════════
    with tab6:
        st.markdown('<p class="section-title">User Guide — v2.1 + AI</p>', unsafe_allow_html=True)
        st.markdown(f"""
This tool analyses forest vegetation phenology from NDVI + climate data.
No forest-type configuration required — fully data-driven.

---

### 🆕 What's new in v2.1 + AI

| Feature | Detail |
|---|---|
| **5-day grid** | Interpolation always uses a {INTERP_STEP_DAYS}-day grid |
| **8 tabs** | Data Quality → Season Extraction → Model Results → Climate Drivers → Predict → User Guide → AI → Sensor Compare |
| **Cross-year EOS** | End-of-season dates can extend into the next calendar year |
| **Split NDVI plot** | Datasets spanning >8 years split into ≤8-year panels |
| **AI Assistant tab** | Ask any question about your results — powered by Google Gemini (free) |

---

### 🤖 AI Assistant Setup

1. Go to **aistudio.google.com** and sign in with Google
2. Click **Get API Key** → Create API key → copy it
3. Add to `.streamlit/secrets.toml`:  `GEMINI_API_KEY = "your_key_here"`
4. Run `pip install google-generativeai`

---

### 📘 Key Terms

**SOS / POS / EOS / LOS** — Start, Peak, End, Length of Season.
**DOY** — Day of Year (1=Jan 1, 365=Dec 31).
**LOO R²** — Leave-One-Out R²: honest out-of-sample accuracy.
**Amplitude** — Peak NDVI minus baseline NDVI per season.

---

### 🤖 Prediction Engine (Auto-Best)

All models fitted simultaneously per event:

| Model | When it wins |
|---|---|
| **Ridge** | Small datasets (n<6); stable, interpretable |
| **LOESS** | Nonlinear with a single driver |
| **Polynomial (deg 2/3)** | Unimodal / curved responses |
| **Gaussian Process** | Complex nonlinear patterns; needs n≥5 |

---

### 📘 Key Terms

**SOS / POS / EOS / LOS** — Start, Peak, End, Length of Season.
**DOY** — Day of Year (1=Jan 1, 365=Dec 31).
**LOO R²** — Leave-One-Out R²: honest out-of-sample accuracy.
**Amplitude** — Peak NDVI minus baseline NDVI per season.

---

### 📂 Data Format

**NDVI CSV** — Any CSV with date + NDVI columns (auto-detected).

**Met CSV** — Continuous **daily** data from
[NASA POWER](https://power.larc.nasa.gov/data-access-viewer/) (Daily → Point → your site → CSV).

---

### 🔗 Links
- **GitHub:** [Universal_Indian_Forest_Phenology_Assessment.py](https://github.com/shreejisharma/Indian-forest-phenology)
- **Run locally:** `streamlit run Universal_Indian_Forest_Phenology_Assessment_v2.py`
        """)

    # ══════════════════════════════════════════════════════════
    # TAB 7 — AI ASSISTANT
    # ══════════════════════════════════════════════════════════
    with tab7:
        from ai_assistant_gemini_free import render_chat_tab
        try:
            _gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            _gemini_key = ""
        render_chat_tab(
            api_key   = _gemini_key,
            pheno_df  = st.session_state.get("pheno_df"),
            predictor = st.session_state.get("predictor"),
            ndvi_info = st.session_state.get("ndvi_info"),
            met_info  = st.session_state.get("met_info"),
        )

    # ══════════════════════════════════════════════════════════
    # TAB 8 — SENSOR COMPARE
    # ══════════════════════════════════════════════════════════
    with tab8:
        # ── Highlight banner ───────────────────────────────────
        st.markdown("""
<div style="background:linear-gradient(135deg,#0A1520 0%,#0F2035 60%,#0A1A2E 100%);
padding:20px 28px 16px;border-radius:16px;margin-bottom:18px;
border:1px solid rgba(100,180,255,0.22)">
<div style="color:#E8F4FF;font-size:1.1rem;font-weight:800;margin-bottom:6px">
🛰️ Multi-Sensor Phenology Comparison</div>
<div style="color:#7BA8CC;font-size:0.84rem;line-height:1.6;margin-bottom:10px">
Want to compare phenology across <b>Landsat</b>, <b>Sentinel-2</b> and <b>MODIS</b>?
Upload 1–3 sensor CSV files using the sidebar (👈 left panel) under
<em>"OPTIONAL — Multi-Sensor Comparison"</em> and results will appear here.<br>
<b>One file</b> shows that sensor's extracted events.
<b>Two or three files</b> unlock side-by-side comparison charts and
inter-sensor agreement statistics (Bias, RMSE, Pearson r).</div>
<div style="display:flex;gap:8px;flex-wrap:wrap">
<span style="background:rgba(232,160,32,.18);color:#FFB432;font-size:0.71rem;font-weight:700;
padding:2px 10px;border-radius:12px;border:1px solid rgba(232,160,32,.3)">🌍 Landsat 30m</span>
<span style="background:rgba(32,128,224,.18);color:#60B8FF;font-size:0.71rem;font-weight:700;
padding:2px 10px;border-radius:12px;border:1px solid rgba(32,128,224,.3)">🛰️ Sentinel-2 10m</span>
<span style="background:rgba(0,184,150,.18);color:#00DCB4;font-size:0.71rem;font-weight:700;
padding:2px 10px;border-radius:12px;border:1px solid rgba(0,184,150,.3)">📡 MODIS 250–500m</span>
<span style="background:rgba(200,160,255,.15);color:#C090FF;font-size:0.71rem;font-weight:700;
padding:2px 10px;border-radius:12px;border:1px solid rgba(200,160,255,.3)">📐 Inter-sensor agreement</span>
</div></div>
""", unsafe_allow_html=True)

        _render_sensor_comparison_tab(
            start_m      = start_m,
            end_m        = end_m,
            sos_thr      = sos_thr,
            eos_thr      = eos_thr,
            min_days     = min_days,
            sensor_files = _sensor_sidebar_files,
        )



if __name__ == "__main__":
    for k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params',
              'ndvi_df','ndvi_info','met_info','interp_freq','_fp']:
        if k not in st.session_state:
            st.session_state[k] = None
    main()
