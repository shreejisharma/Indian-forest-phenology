"""
Forest Phenology Analyser
=========================
Upload NDVI and meteorological data to extract seasonal phenology events (SOS, POS, EOS, LOS),
identify climate drivers, train predictive models, and forecast future phenology dates.

Supports any Indian forest type — fully data-driven, no manual configuration required.

Requirements:
    pip install streamlit pandas numpy scipy scikit-learn matplotlib statsmodels

Run:
    streamlit run forest_phenology_analyser.py

CHANGELOG — Model Engine Update
--------------------------------
Replaced fit_event_model / loo_ridge / fit_loess / loo_poly / GPR block with the
unified model engine from v5.3 (PhenologyEngine.fit_models pattern):

  • loo_cv_generic()         — single callable-based LOO helper (replaces 4 separate LOO funcs)
  • fit_all_models()         — fits Ridge + LOESS + Poly-2 + Poly-3 + GPR in one pass,
                               each with LOO R² and MAE; best model selected by LOO R²
  • RidgeCV alphas           — 30 log-spaced values from 1e-3 to 1e4 (was 9 fixed values)
  • LOESS                    — statsmodels lowess + PCA projection for multi-feature input
                               (falls back to custom _loess_predict if statsmodels absent)
  • GPR kernel               — ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
                               (was RBF + WhiteKernel only)
  • Model selection          — best_model = argmax(loo_r2) across all fitted models
  • UniversalPredictor.train — calls fit_all_models; stores full results dict per event
  • UniversalPredictor.predict — dispatches to correct model using fit['mode']
  • equation_str             — shows all model types with LOO metrics
  • Small-n safety           — v2's auto-downgrade logic (n<=3→Ridge, n<=5→no poly3/GPR) KEPT
  • Feature selection        — v2's sophisticated multi-feature selector KEPT unchanged
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
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
import io

# Optional statsmodels LOESS
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _LOESS_AVAILABLE = True
except ImportError:
    _LOESS_AVAILABLE = False

# ─── LOESS fallback (no external dependency) ─────────────────
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
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
.metric-card {
    background: #fff; padding: 20px 16px; border-radius: 14px;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #E8F5E9; margin: 4px;
}
.metric-card .label { color: #616161; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.metric-card .value { color: #1B5E20; font-size: 1.85rem; font-weight: 700; margin: 0; }
.metric-card .sub   { color: #757575; font-size: 0.76rem; margin-top: 4px; }
.banner-info  { background: #E3F2FD; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #1976D2; margin: 10px 0; font-size: 0.88rem; }
.banner-warn  { background: #FFF8E1; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #F9A825; margin: 10px 0; font-size: 0.88rem; }
.banner-good  { background: #E8F5E9; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #43A047; margin: 10px 0; font-size: 0.88rem; }
.banner-error { background: #FFEBEE; padding: 14px 18px; border-radius: 10px;
    border-left: 4px solid #E53935; margin: 10px 0; font-size: 0.88rem; }
.eq-box { background: #F8F9FA; padding: 14px 16px; border-radius: 10px;
    border-left: 4px solid #7B1FA2; font-family: 'Courier New', monospace;
    font-size: 0.83rem; margin: 8px 0; word-break: break-all; color: #212121; }
.upload-panel { background: #F9FBF9; padding: 24px 28px; border-radius: 14px;
    border: 2px dashed #A5D6A7; margin: 20px 0; }
.upload-panel h3 { color: #1B5E20; margin-bottom: 12px; font-size: 1.05rem; }
.upload-panel code { background: #E8F5E9; padding: 2px 6px; border-radius: 4px; font-size: 0.82rem; }
.section-title { font-size: 1.15rem; font-weight: 700; color: #1B5E20;
    margin: 24px 0 12px; padding-bottom: 6px; border-bottom: 2px solid #C8E6C9; }
.term { background: #E8F5E9; padding: 2px 8px; border-radius: 5px;
    font-weight: 600; color: #1B5E20; font-size: 0.88rem; }
.stat-card { background: #FAFFFE; padding: 12px 16px; border-radius: 10px;
    border: 1px solid #C8E6C9; margin: 4px 0; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────
MIN_CORR_THRESHOLD = 0.40
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


# ═══════════════════════════════════════════════════════════════
# ── NEW: UNIFIED LOO CV HELPER (from v5.3) ─────────────────────
# ═══════════════════════════════════════════════════════════════

def loo_cv_generic(X: np.ndarray, y: np.ndarray, model_fn):
    """
    Generic Leave-One-Out cross-validation.

    Ported from PhenologyEngine.loo_cv (v5.3).
    Takes a callable model_fn() → fresh unfitted model (same interface as sklearn).
    Falls back to training-mean prediction on any per-fold error.

    Returns (r2, mae).  Gracefully handles n < 3.
    """
    n = len(y)
    if n < 2:
        return 0.0, float(np.std(y)) if len(y) > 0 else float('inf')
    if n == 2:
        # LOO on 2 points trains on 1 — use Pearson r² proxy instead
        try:
            r, _ = pearsonr(X[:, 0] if X.ndim > 1 else X.ravel(), y)
            mae  = float(np.mean(np.abs(y - y.mean())))
            return float(r ** 2), mae
        except Exception:
            return 0.0, float(np.mean(np.abs(y - y.mean())))

    preds = []
    for i in range(n):
        idx_train = [j for j in range(n) if j != i]
        Xt, yt = X[idx_train], y[idx_train]
        Xv     = X[[i]]
        try:
            m = model_fn()
            m.fit(Xt, yt)
            preds.append(float(m.predict(Xv)[0]))
        except Exception:
            preds.append(float(np.mean(yt)))

    preds  = np.array(preds)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2  = float(np.clip(1 - ss_res / (ss_tot + 1e-12), -1, 1))
    mae = float(np.mean(np.abs(y - preds)))
    return r2, mae


# ═══════════════════════════════════════════════════════════════
# ── NEW: FIT ALL MODELS (ported from PhenologyEngine.fit_models v5.3) ──
# ═══════════════════════════════════════════════════════════════

def fit_all_models(X: np.ndarray, y: np.ndarray):
    """
    Fit Ridge, LOESS, Polynomial (deg-2, deg-3), and GPR models with LOO R² / MAE.

    Ported from PhenologyEngine.fit_models (v5.3) with these additions:
      - statsmodels LOESS with PCA projection for multi-feature input
      - Falls back to custom _loess_predict if statsmodels unavailable
      - ConstantKernel * RBF + WhiteKernel for GPR (v5.3 kernel)
      - RidgeCV with 30 log-spaced alphas (1e-3 → 1e4)

    Returns dict keyed by model name:
      {
        'Ridge':      {'model': pipe, 'scaler': sc, 'loo_r2': r2, 'loo_mae': mae,
                       'coefs': [...], 'intercept': float, 'alpha': float},
        'LOESS':      {'model': obj,  'scaler': sc, 'loo_r2': r2, 'loo_mae': mae, ...},
        'Poly_deg2':  {'model': pipe, 'scaler': None, 'loo_r2': r2, ...},
        'Poly_deg3':  ...
        'GPR':        {'model': gpr,  'scaler': sc, 'loo_r2': r2, ...},
      }
    """
    results = {}
    n = len(y)

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # ── Ridge ──────────────────────────────────────────────────
    alphas_v53 = np.logspace(-3, 4, 30)
    ridge      = RidgeCV(alphas=alphas_v53)
    ridge.fit(Xs, y)
    r2_r, mae_r = loo_cv_generic(
        Xs, y,
        lambda: Pipeline([('sc', StandardScaler()),
                          ('r',  Ridge(alpha=float(ridge.alpha_)))])
    )
    # Unstandardised coefficients for equation display
    coef_unstd = list(ridge.coef_ / sc.scale_)
    intercept_unstd = float(ridge.intercept_ - np.dot(ridge.coef_ / sc.scale_, sc.mean_))
    results['Ridge'] = {
        'model':     ridge,
        'scaler':    sc,
        'loo_r2':    r2_r,
        'loo_mae':   mae_r,
        'coefs':     coef_unstd,
        'intercept': intercept_unstd,
        'alpha':     float(ridge.alpha_),
        'mode':      'ridge',
    }

    # ── LOESS ──────────────────────────────────────────────────
    if n >= 4:
        try:
            if _LOESS_AVAILABLE:
                # Multi-feature → project onto first PC
                from sklearn.decomposition import PCA
                if Xs.shape[1] == 1:
                    X_loess = Xs[:, 0]
                    _pca    = None
                else:
                    pca     = PCA(n_components=1)
                    X_loess = pca.fit_transform(Xs)[:, 0]
                    _pca    = pca

                frac_val = min(0.75, max(0.25, 6.0 / n))

                # LOO for statsmodels LOESS
                loess_preds = []
                for i in range(n):
                    idx_tr  = [j for j in range(n) if j != i]
                    x_tr    = X_loess[idx_tr]
                    y_tr    = y[idx_tr]
                    x_val   = X_loess[i]
                    try:
                        sm_sorted = sm_lowess(y_tr, x_tr, frac=frac_val, return_sorted=True)
                        sm_sorted = sm_sorted[np.argsort(sm_sorted[:, 0])]
                        f_interp  = interp1d(sm_sorted[:, 0], sm_sorted[:, 1],
                                             bounds_error=False,
                                             fill_value=(sm_sorted[0, 1], sm_sorted[-1, 1]))
                        loess_preds.append(float(f_interp(x_val)))
                    except Exception:
                        loess_preds.append(float(np.mean(y_tr)))

                lp     = np.array(loess_preds)
                ss_res = np.sum((y - lp) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2_l   = float(np.clip(1 - ss_res / (ss_tot + 1e-12), -1, 1))
                mae_l  = float(np.mean(np.abs(y - lp)))

                # Wrapper object so .predict() works uniformly
                class _SmLoessModel:
                    def __init__(self, x_tr, y_tr, frac, pca_obj):
                        self.x_tr  = x_tr
                        self.y_tr  = y_tr
                        self.frac  = frac
                        self.pca   = pca_obj
                    def predict(self, X_new_scaled):
                        if self.pca is not None:
                            xn = self.pca.transform(X_new_scaled)[:, 0]
                        else:
                            xn = X_new_scaled[:, 0]
                        sm = sm_lowess(self.y_tr, self.x_tr, frac=self.frac, return_sorted=True)
                        sm = sm[np.argsort(sm[:, 0])]
                        f  = interp1d(sm[:, 0], sm[:, 1], bounds_error=False,
                                      fill_value=(sm[0, 1], sm[-1, 1]))
                        return f(xn)

                results['LOESS'] = {
                    'model':   _SmLoessModel(X_loess, y, frac_val, _pca),
                    'scaler':  sc,
                    'loo_r2':  r2_l,
                    'loo_mae': mae_l,
                    'coefs':   None,
                    'intercept': None,
                    'mode':    'loess',
                    'feature_index': 0,      # first feature (or PC1)
                }

            else:
                # Fallback: custom LOESS on first feature only
                x1d    = X[:, 0].astype(float)
                frac_v = min(0.75, max(0.25, 6.0 / n))
                preds_f = np.zeros(n)
                for i in range(n):
                    mask = np.ones(n, dtype=bool); mask[i] = False
                    preds_f[i] = _loess_predict(x1d[mask], y[mask], np.array([x1d[i]]),
                                                frac=frac_v)[0]
                ss_res = np.sum((y - preds_f) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2_l   = float(np.clip(1 - ss_res / (ss_tot + 1e-12), -1, 1))
                mae_l  = float(np.mean(np.abs(y - preds_f)))

                class _FallbackLoess:
                    def __init__(self, x_tr, y_tr, frac):
                        self.x_tr = x_tr; self.y_tr = y_tr; self.frac = frac
                    def predict(self, X_new_scaled):
                        return _loess_predict(self.x_tr, self.y_tr,
                                              X_new_scaled[:, 0].astype(float), self.frac)

                results['LOESS'] = {
                    'model':   _FallbackLoess(x1d, y, frac_v),
                    'scaler':  None,   # custom LOESS uses raw X, not scaled
                    'loo_r2':  r2_l,
                    'loo_mae': mae_l,
                    'coefs':   None,
                    'intercept': None,
                    'mode':    'loess',
                }
        except Exception:
            pass

    # ── Polynomial deg-2 and deg-3 ─────────────────────────────
    for deg in [2, 3]:
        if n < deg + 2:
            continue
        try:
            poly_pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('sc',   StandardScaler()),
                ('r',    RidgeCV(alphas=alphas_v53)),
            ])
            poly_pipe.fit(X, y)
            r2_p, mae_p = loo_cv_generic(
                X, y,
                lambda d=deg: Pipeline([
                    ('poly', PolynomialFeatures(degree=d, include_bias=False)),
                    ('sc',   StandardScaler()),
                    ('r',    RidgeCV(alphas=alphas_v53)),
                ])
            )
            results[f'Poly_deg{deg}'] = {
                'model':     poly_pipe,
                'scaler':    None,   # poly pipe has its own scaler
                'loo_r2':    r2_p,
                'loo_mae':   mae_p,
                'coefs':     None,
                'intercept': None,
                'mode':      f'poly{deg}',
            }
        except Exception:
            pass

    # ── GPR ─────────────────────────────────────────────────────
    if n >= 5:
        try:
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
            gpr    = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gpr.fit(Xs, y)

            r2_g, mae_g = loo_cv_generic(
                Xs, y,
                lambda: GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                    normalize_y=True)
            )
            results['GPR'] = {
                'model':     gpr,
                'scaler':    sc,
                'loo_r2':    r2_g,
                'loo_mae':   mae_g,
                'coefs':     None,
                'intercept': None,
                'mode':      'gpr',
            }
        except Exception:
            pass

    return results


# ═══════════════════════════════════════════════════════════════
# DATA-ADAPTIVE UTILITIES  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════

def detect_ndvi_cadence(ndvi_df):
    dates = pd.to_datetime(ndvi_df['Date']).sort_values()
    diffs = dates.diff().dt.days.dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 16, 64, 5
    median_cad = float(diffs.median())
    max_gap    = max(60, int(median_cad * 8))
    interp_freq = max(1, min(8, int(median_cad // 2)))
    return median_cad, max_gap, interp_freq


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
    return max(0.01, (p95 - p5) * 0.05)


def characterize_ndvi_data(ndvi_df):
    vals  = ndvi_df['NDVI'].dropna().values
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
        'met_cadence_days': None, 'met_is_paired_with_ndvi': False,
        'met_has_large_gaps': False, 'met_gap_periods': [],
        'per_event_coverage': {}, 'warnings': [],
    }
    if met_df is None or len(met_df) == 0:
        result['warnings'].append("Meteorological file is empty.")
        return result
    met_dates = pd.to_datetime(met_df['Date']).sort_values()
    diffs     = met_dates.diff().dt.days.dropna()
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
                "rather than being a continuous daily record. For best results, upload a continuous "
                "daily meteorological CSV (e.g. from NASA POWER Daily for your coordinates).")
    gap_mask = diffs > 60
    if gap_mask.any():
        result['met_has_large_gaps'] = True
        gap_end_dates   = met_dates[diffs.index[gap_mask]]
        gap_start_dates = met_dates.shift(1)[diffs.index[gap_mask]]
        for gs, ge in zip(gap_start_dates, gap_end_dates):
            result['met_gap_periods'].append(
                f"{pd.Timestamp(gs).strftime('%b %Y')} → {pd.Timestamp(ge).strftime('%b %Y')}")
        result['warnings'].append(
            f"⚠️ Large gaps detected in meteorological data: "
            + ", ".join(result['met_gap_periods']))
    if pheno_df is not None and len(pheno_df) > 0:
        n_total = len(pheno_df)
        for ev in ['SOS', 'POS', 'EOS']:
            date_col = f'{ev}_Date'
            if date_col not in pheno_df.columns: continue
            n_with_data = 0; seasons_missing = []
            for _, row in pheno_df.iterrows():
                evt_dt = row[date_col]
                if pd.isna(evt_dt):
                    seasons_missing.append(int(row['Year'])); continue
                mask   = ((met_df['Date'] >= pd.Timestamp(evt_dt) - timedelta(days=window)) &
                          (met_df['Date'] <= pd.Timestamp(evt_dt)))
                if len(met_df[mask]) >= 1:
                    n_with_data += 1
                else:
                    seasons_missing.append(int(row['Year']))
            result['per_event_coverage'][ev] = {
                'n_seasons_with_data': n_with_data, 'n_seasons_total': n_total,
                'seasons_missing': seasons_missing,
                'coverage_pct': round(100 * n_with_data / n_total, 0) if n_total > 0 else 0,
            }
            if n_with_data < n_total:
                missing_yrs = ", ".join(str(y) for y in seasons_missing)
                result['warnings'].append(
                    f"⚠️ {ev} model: {n_total - n_with_data} season(s) have NO met data "
                    f"in the {window}-day pre-event window (year(s): {missing_yrs}). "
                    f"Only {n_with_data} of {n_total} seasons used for training.")
    return result


# ═══════════════════════════════════════════════════════════════
# PARSERS  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════

def parse_nasa_power(uploaded_file):
    try:
        raw   = uploaded_file.read().decode('utf-8', errors='replace')
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
                    if date_col != 'Date': df = df.drop(columns=[date_col])
                else:
                    return None, [], 'Cannot build Date — need YEAR+DOY, YEAR+MO+DY, or a Date column'
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        excl   = {'YEAR', 'MO', 'DY', 'DOY', 'LON', 'LAT', 'ELEV', 'Date'}
        params = [c for c in df.columns
                  if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
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
        yr_median   = s.dropna().dt.year.median()
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
# DERIVED MET FEATURES  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════

def _season_cumsum(series, dates, sm_):
    out = series.copy() * 0.0
    dt  = pd.to_datetime(dates)
    season_yr   = np.where(dt.dt.month >= sm_, dt.dt.year, dt.dt.year - 1)
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
            alt  = _detect_column([c for c in cols if c != prec], ['PPT','RAIN','PRECIP','PREC'], ['PPT','RAIN'])
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
            df['GDD_cum'] = _season_cumsum(np.maximum(tavg - 10, 0).rename('GDD_10_tmp'),
                                           df['Date'], season_start_month)
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
# TRAINING FEATURE BUILDER  (unchanged from v2)
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
            wdf  = met_df[mask]
            if len(wdf) < max(1, window * 0.15): continue
            for p in params:
                if p not in met_df.columns: continue
                p_upper = p.upper()
                is_snapshot = (p_upper in SNAPSHOT_FEATURES or
                               any(k in p_upper for k in SNAPSHOT_KEYWORDS))
                is_accum    = (not is_snapshot and
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
# PHENOLOGY EXTRACTION  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════

def _find_troughs(ndvi_values, min_distance=10):
    n = len(ndvi_values); troughs = []
    for i in range(min_distance, n - min_distance):
        window = ndvi_values[max(0, i - min_distance): i + min_distance + 1]
        if ndvi_values[i] == np.min(window):
            if ndvi_values[i] <= ndvi_values[i - 1] and ndvi_values[i] <= ndvi_values[i + 1]:
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
    n = len(v); troughs = []
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
    search_end = min(int(min_distance * 1.5), n // 3)
    if search_end > 1:
        bmi      = int(np.argmin(v[0:search_end]))
        next_t   = merged[0] if merged else n
        if bmi > 0 and next_t - bmi >= min_distance:
            midpoint_val = float(np.mean(v[bmi:next_t])) if next_t < n else float(np.mean(v[bmi:]))
            if v[bmi] < midpoint_val:
                merged.insert(0, bmi)
    search_start = max(n - int(min_distance * 1.5), 2 * n // 3)
    if search_start < n - 1:
        bmi    = search_start + int(np.argmin(v[search_start:]))
        prev_t = merged[-1] if merged else -1
        if bmi - prev_t >= min_distance and bmi < n - 1:
            midpoint_val = float(np.mean(v[prev_t:bmi])) if prev_t >= 0 else float(np.mean(v[0:bmi]))
            if v[bmi] < midpoint_val:
                merged.append(bmi)
    return merged


def extract_phenology(ndvi_df, cfg, sos_threshold_pct, eos_threshold_pct):
    try:
        sm = cfg["start_month"]; em = cfg["end_month"]; min_d = cfg.get("min_days", 100)
        thr_pct = sos_threshold_pct; eos_thr_pct = eos_threshold_pct
        ndvi_raw = ndvi_df[["Date", "NDVI"]].copy().set_index("Date").sort_index()
        if ndvi_raw.index.duplicated().any():
            ndvi_raw = ndvi_raw.groupby(ndvi_raw.index)['NDVI'].mean().rename('NDVI').to_frame()
        orig_dates  = ndvi_raw.index.sort_values()
        orig_diffs  = pd.Series(orig_dates).diff().dt.days.fillna(0)
        pos_diffs   = orig_diffs[orig_diffs > 0]
        typical_cad = float(pos_diffs.median()) if len(pos_diffs) > 0 else 16.0
        MAX_INTERP_GAP = max(60, int(typical_cad * 8))
        gap_starts  = orig_dates[orig_diffs.values > MAX_INTERP_GAP]
        interp_freq = max(1, min(8, round(typical_cad)))
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
        valid_vals = ndvi_vals[valid_mask]
        MIN_AMPLITUDE = compute_data_driven_min_amplitude(valid_vals) if len(valid_vals) > 5 else 0.02
        MAX_SG_STEPS = 31
        sm_vals = np.full(n, np.nan); seg_labels = np.zeros(n, dtype=int); seg_id, in_seg = 0, False
        for i in range(n):
            if valid_mask[i]:
                if not in_seg: seg_id += 1; in_seg = True
                seg_labels[i] = seg_id
            else:
                in_seg = False
        for sid in range(1, seg_id + 1):
            idx_seg = np.where(seg_labels == sid)[0]; seg_n = len(idx_seg)
            if seg_n < 5:
                sm_vals[idx_seg] = ndvi_vals[idx_seg]; continue
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
        min_dist   = max(10, int(cycle_steps * 0.4))
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
                trough_ceil    = global_min + 0.85 * global_amp
                trough_indices = [ti for ti in trough_indices if sm_for_troughs[ti] <= trough_ceil]
        _GAP_STRICT = 0.20; _GAP_TOLERANT = 0.50; _AMP_GAP_THR = 0.10

        def _cycle_has_gap(i_start, i_end, amplitude=None):
            if i_end <= i_start: return True
            gap_frac = np.isnan(sm_vals[i_start:i_end + 1]).mean()
            if amplitude is not None and amplitude >= _AMP_GAP_THR: return gap_frac > _GAP_TOLERANT
            return gap_frac > _GAP_STRICT

        rows = []

        def _date_in_window(d):
            m = d.month
            if sm <= em: return sm <= m <= em
            return m >= sm or m <= em

        _valid_sm_vals = sm_for_troughs[~np.isnan(sm_vals)]
        _global_min    = float(np.percentile(_valid_sm_vals, 5))  if len(_valid_sm_vals) > 0 else 0
        _global_amp    = (float(np.percentile(_valid_sm_vals, 95)) - _global_min) if len(_valid_sm_vals) > 0 else 1
        _head_trough_ceiling = _global_min + 0.25 * _global_amp
        _head_start_looks_like_trough = (float(sm_for_troughs[0]) <= _head_trough_ceiling)

        if trough_indices and _head_start_looks_like_trough:
            ti_first = trough_indices[0]; head_len = ti_first
            _amp_pre = (float(np.max(sm_for_troughs[0:ti_first + 1])) - float(sm_for_troughs[0]))
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
                    ndvi_max  = float(np.nanmax(work_arr)); A = ndvi_max - ndvi_min
                    if A >= MIN_AMPLITUDE:
                        sos_thr = ndvi_min + thr_pct * A; eos_thr = ndvi_min + eos_thr_pct * A
                        pi  = int(np.nanargmax(work_arr)); pos = seg_t[pi]
                        if _date_in_window(pos):
                            asc = work_arr[1:pi + 1]; sc  = np.where(asc >= sos_thr)[0]
                            desc = work_arr[pi:]
                            ec_below = np.where(desc < eos_thr)[0]
                            ei = pi + max(0, int(ec_below[0]) - 1) if len(ec_below) else pi + int(np.nanargmin(desc))
                            if len(sc) and ei > 0:
                                si = 1 + int(sc[0])
                                if ei > si:
                                    sos = seg_t[si]; eos = seg_t[ei]
                                    if eos > t_all[-1]: eos = t_all[-1]
                                    if (eos - sos).days >= 365: eos = sos + pd.Timedelta(days=364)
                                    if eos > sos:
                                        trough_year  = seg_t[0].year
                                        season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                                        rows.append(_make_row(trough_year, season_start, sos, pos, eos,
                                            ndvi_max, A, ndvi_min, sos_thr, eos_thr,
                                            seg_t[int(np.argmin(seg_sm))], sm, em))
                except Exception:
                    pass

        for i in range(len(trough_indices) - 1):
            try:
                ti  = trough_indices[i]; ti1 = trough_indices[i + 1]
                if ti1 - ti < max(10, min_d // interp_freq): continue
                _amp_pre = (float(np.max(sm_for_troughs[ti:ti1 + 1])) - float(sm_for_troughs[ti]))
                if _cycle_has_gap(ti, ti1, amplitude=_amp_pre): continue
                _has_gap  = np.isnan(sm_vals[ti:ti1 + 1]).any()
                cycle_raw = sm_for_troughs[ti:ti1 + 1] if _has_gap else ndvi_vals[ti:ti1 + 1]
                cycle_t   = t_all[ti:ti1 + 1]
                ndvi_min  = float(sm_for_troughs[ti]) if _has_gap else (
                    float(ndvi_vals[ti]) if not np.isnan(ndvi_vals[ti]) else float(sm_for_troughs[ti]))
                ndvi_max  = float(np.nanmax(cycle_raw)); A = ndvi_max - ndvi_min
                if A < MIN_AMPLITUDE: continue
                sos_thr = ndvi_min + thr_pct * A; eos_thr = ndvi_min + eos_thr_pct * A
                pos_idx = int(np.nanargmax(cycle_raw))
                asc     = cycle_raw[1:pos_idx + 1]; sc = np.where(asc >= sos_thr)[0] + 1
                if not len(sc): continue
                si = int(sc[0]); desc = cycle_raw[pos_idx:-1]
                ec_below = np.where(desc < eos_thr)[0]
                ei = pos_idx + max(0, int(ec_below[0]) - 1) if len(ec_below) else pos_idx + int(np.nanargmin(desc))
                if ei <= si: continue
                sos = cycle_t[si]; pos = cycle_t[pos_idx]; eos = cycle_t[ei]
                if eos > t_all[-1]: eos = t_all[-1]
                if (eos - sos).days >= 365: eos = sos + pd.Timedelta(days=364)
                if eos <= sos: continue
                if not _date_in_window(pos): continue
                trough_year  = t_all[ti].year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                rows.append(_make_row(trough_year, season_start, sos, pos, eos,
                    ndvi_max, A, ndvi_min, sos_thr, eos_thr, t_all[ti], sm, em))
            except Exception:
                continue

        covered = set()
        for i in range(len(trough_indices) - 1):
            _a = (float(np.max(sm_for_troughs[trough_indices[i]:trough_indices[i+1]+1])) -
                  float(sm_for_troughs[trough_indices[i]]))
            if not _cycle_has_gap(trough_indices[i], trough_indices[i+1], amplitude=_a):
                covered.add(trough_indices[i])
        for ti0 in [ti for ti in trough_indices if ti not in covered]:
            tail_end = n - 1; tail_len = tail_end - ti0
            if tail_len < max(10, min_d // interp_freq): continue
            try:
                _a = (float(np.max(sm_for_troughs[ti0:tail_end + 1])) - float(sm_for_troughs[ti0]))
                if _cycle_has_gap(ti0, tail_end, amplitude=_a): continue
                _has_gap = np.isnan(sm_vals[ti0:tail_end + 1]).any()
                seg_raw  = sm_for_troughs[ti0:tail_end + 1] if _has_gap else ndvi_vals[ti0:tail_end + 1]
                seg_t    = t_all[ti0:tail_end + 1]
                seg_sm   = sm_for_troughs[ti0:tail_end + 1]
                ndvi_min = float(sm_for_troughs[ti0]) if _has_gap else (
                    float(ndvi_vals[ti0]) if not np.isnan(ndvi_vals[ti0]) else float(sm_for_troughs[ti0]))
                ndvi_max = float(np.nanmax(seg_raw)); A = ndvi_max - ndvi_min
                if A < MIN_AMPLITUDE: continue
                sos_thr = ndvi_min + thr_pct * A; eos_thr = ndvi_min + eos_thr_pct * A
                pi  = int(np.nanargmax(seg_raw)); asc = seg_raw[1:pi + 1]
                sc  = np.where(asc >= sos_thr)[0] + 1
                if not len(sc): continue
                si = int(sc[0]); desc = seg_raw[pi:]
                ec_below = np.where(desc < eos_thr)[0]
                ei = pi + max(0, int(ec_below[0]) - 1) if len(ec_below) else pi + int(np.nanargmin(desc))
                if ei <= si: continue
                sos = seg_t[si]; pos = seg_t[pi]; eos = seg_t[ei]
                if eos > t_all[-1]: eos = t_all[-1]
                if (eos - sos).days >= 365: eos = sos + pd.Timedelta(days=364)
                if eos <= sos: continue
                eos_is_at_data_end  = (eos >= t_all[-1] - pd.Timedelta(days=interp_freq*2))
                ndvi_at_data_end    = float(ndvi_vals[-1]) if not np.isnan(ndvi_vals[-1]) else float(sm_for_troughs[-1])
                if eos_is_at_data_end and ndvi_at_data_end >= eos_thr: continue
                if not _date_in_window(pos): continue
                trough_year  = seg_t[0].year
                season_start = pd.Timestamp(f"{trough_year}-{sm:02d}-01")
                rows.append(_make_row(trough_year, season_start, sos, pos, eos,
                    ndvi_max, A, ndvi_min, sos_thr, eos_thr, t_all[ti0], sm, em))
            except Exception:
                pass

        if not rows:
            return None, (
                f"No complete seasons detected. Troughs found: {len(trough_indices)}. "
                f"Data: {ndvi_5d.index.min().date()} → {ndvi_5d.index.max().date()} "
                f"({n} pts, {interp_freq}d grid). MIN_AMPLITUDE={MIN_AMPLITUDE:.3f}.")
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
# FEATURE SELECTION  (unchanged from v2 — more sophisticated than v5.3)
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
            rows.append({'Feature': col, 'Pearson_r': round(r, 3), '|r|': round(abs(r), 3),
                         'Spearman_rho': round(float(rho), 3), '|rho|': round(abs(float(rho)), 3),
                         'Composite': round(composite, 3), 'p_value': round(p_val, 3),
                         'Usable': '✅' if composite >= MIN_CORR_THRESHOLD else '❌'})
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
    loo   = LeaveOneOut(); preds = []
    sc    = StandardScaler()
    for tr, te in loo.split(X_vals):
        if len(tr) < 1: continue
        try:
            Xtr = sc.fit_transform(X_vals[tr]); Xte = sc.transform(X_vals[te])
            m   = Ridge(alpha=alpha); m.fit(Xtr, y_vals[tr])
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
    if n_obs <= 3:    effective_min_r = min(min_r, 0.10)
    elif n_obs <= 5:  effective_min_r = min(min_r, 0.25)
    else:             effective_min_r = min_r
    collinear_thr = 0.97 if n_obs <= 5 else 0.85
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
                collinear = True; break
        if not collinear:
            collinear_filtered.append(feat)
    if user_max_features is not None:
        effective_max = user_max_features
    elif n_obs <= 3:  effective_max = 1
    elif n_obs <= 5:  effective_max = min(max_features, 2)
    else:             effective_max = max_features
    max_safe   = max(1, n_obs - 1)
    candidates = collinear_filtered[:min(effective_max, max_safe)]
    if len(candidates) <= 1: return candidates
    y_vals   = y.values.astype(float)
    selected = [candidates[0]]
    best_r2  = _loo_r2_quick(
        X[selected].fillna(X[selected[0]].median()).values.reshape(-1, 1), y_vals)
    improvement_thr = 0.08 if n_obs <= 5 else 0.03
    for feat in candidates[1:]:
        trial = selected + [feat]
        Xt    = X[trial].fillna(X[trial].median()).values
        try:
            trial_r2 = _loo_r2_quick(Xt, y_vals)
        except Exception:
            continue
        if trial_r2 > best_r2 + improvement_thr:
            selected.append(feat); best_r2 = trial_r2
    return selected


# ═══════════════════════════════════════════════════════════════
# ── NEW: UNIVERSAL PREDICTOR (v5.3 engine, v2 feature selection) ──
# ═══════════════════════════════════════════════════════════════

class UniversalPredictor:
    """
    Predictor using v5.3 model engine:
      - fit_all_models() → Ridge + LOESS + Poly-2 + Poly-3 + GPR, all with LOO R²
      - Best model selected by highest LOO R²
      - Small-n safety from v2 retained (n≤3→Ridge only, n≤5→no Poly3/GPR)
      - Feature selection from v2 retained (sophisticated multi-feature selector)
    """
    def __init__(self):
        self._fits       = {}   # event → {'best_name', 'all_models', 'features', ...}
        self.r2          = {}
        self.mae         = {}
        self.n_seasons   = {}
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

            X = sub[feat_cols].fillna(sub[feat_cols].median())
            y = sub['Target_DOY']

            self.corr_tables[event] = get_all_correlations(X, y)

            # ── v2 feature selection (unchanged) ─────────────────────
            n = len(y)
            adaptive_min_r = MIN_CORR_THRESHOLD
            if n <= 4:  adaptive_min_r = max(0.25, MIN_CORR_THRESHOLD - 0.15)
            elif n <= 6: adaptive_min_r = max(0.30, MIN_CORR_THRESHOLD - 0.10)

            features = select_multi_features(X, y, max_features=5,
                                             min_r=adaptive_min_r,
                                             user_max_features=user_max_features)
            if not features:
                # Mean-only fallback
                md = float(y.mean())
                self._fits[event] = {
                    'best_name': 'Mean',
                    'all_models': {},
                    'features': [],
                    'mode': 'mean',
                    'mean_doy': md,
                    'r2': 0.0,
                    'mae': float(np.mean(np.abs(y - md))),
                    'n': n,
                }
                self.r2[event]  = 0.0
                self.mae[event] = float(np.mean(np.abs(y - md)))
                continue

            Xf = X[features].fillna(X[features].median())
            Xv = Xf.values
            yt = y.values.astype(float)

            # ── v2 small-n safety: restrict available models ──────────
            if n <= 3:
                allowed = {'Ridge'}
            elif n <= 5:
                allowed = {'Ridge', 'LOESS', 'Poly_deg2'}
            else:
                allowed = {'Ridge', 'LOESS', 'Poly_deg2', 'Poly_deg3', 'GPR'}

            # ── v5.3 fit_all_models ───────────────────────────────────
            all_mods = fit_all_models(Xv, yt)

            # Filter to allowed models
            all_mods = {k: v for k, v in all_mods.items() if k in allowed}

            if not all_mods:
                md = float(yt.mean())
                self._fits[event] = {
                    'best_name': 'Mean', 'all_models': {}, 'features': features,
                    'mode': 'mean', 'mean_doy': md,
                    'r2': 0.0, 'mae': float(np.mean(np.abs(yt - md))), 'n': n,
                }
                self.r2[event]  = 0.0
                self.mae[event] = float(np.mean(np.abs(yt - md)))
                continue

            # ── v5.3 best-model selection: argmax(loo_r2) ────────────
            best_name = max(all_mods, key=lambda k: all_mods[k]['loo_r2']
                            if not np.isnan(all_mods[k]['loo_r2']) else -999)
            best      = all_mods[best_name]

            # Build unstandardised Ridge coefficients for display
            ridge_res = all_mods.get('Ridge')
            coef_display  = ridge_res['coefs']      if ridge_res else None
            inter_display = ridge_res['intercept']  if ridge_res else None

            self._fits[event] = {
                'best_name':  best_name,
                'all_models': all_mods,
                'features':   features,
                'mode':       best['mode'],
                # Convenience aliases used by predict() and equation_str()
                'best_model': best['model'],
                'best_scaler': best['scaler'],
                'r2':         best['loo_r2'],
                'mae':        best['loo_mae'],
                'n':          n,
                'mean_doy':   float(yt.mean()),
                # Ridge coefficients for equation display (always from Ridge model)
                'coefs':      coef_display,
                'intercept':  inter_display,
                'alpha':      ridge_res['alpha'] if ridge_res else None,
            }
            self.r2[event]  = best['loo_r2']
            self.mae[event] = best['loo_mae']

    def predict(self, inputs: dict, event: str, year: int = 2026, season_start_month: int = 6):
        """
        Predict event DOY and date for given input feature values.
        Dispatches to the best-model mode using v5.3 predict pattern.
        """
        if event not in self._fits:
            return None
        fit = self._fits[event]

        if fit['mode'] == 'mean':
            rel_days = int(round(fit['mean_doy']))

        elif fit['mode'] == 'loess':
            best = fit['all_models'].get('LOESS') or fit['all_models'].get(fit['best_name'])
            model   = best['model']
            scaler  = best['scaler']
            feats   = fit['features']
            vals    = np.array([[inputs.get(f, 0.0) for f in feats]])
            if scaler is not None:
                vals_scaled = scaler.transform(vals)
            else:
                vals_scaled = vals
            try:
                pred     = float(model.predict(vals_scaled)[0])
                rel_days = int(np.clip(round(pred), 0, 500))
            except Exception:
                rel_days = int(round(fit['mean_doy']))

        elif fit['mode'] in ('poly2', 'poly3'):
            key  = {'poly2': 'Poly_deg2', 'poly3': 'Poly_deg3'}[fit['mode']]
            best = fit['all_models'].get(key) or fit['all_models'].get(fit['best_name'])
            vals = np.array([[inputs.get(f, 0.0) for f in fit['features']]])
            try:
                pred     = float(best['model'].predict(vals)[0])
                rel_days = int(np.clip(round(pred), 0, 500))
            except Exception:
                rel_days = int(round(fit['mean_doy']))

        elif fit['mode'] == 'gpr':
            best   = fit['all_models'].get('GPR') or fit['all_models'].get(fit['best_name'])
            scaler = best['scaler']
            vals   = np.array([[inputs.get(f, 0.0) for f in fit['features']]])
            try:
                vals_scaled = scaler.transform(vals) if scaler is not None else vals
                pred        = float(best['model'].predict(vals_scaled)[0])
                rel_days    = int(np.clip(round(pred), 0, 500))
            except Exception:
                rel_days = int(round(fit['mean_doy']))

        else:  # ridge (default)
            best   = fit['all_models'].get('Ridge') or fit['all_models'].get(fit['best_name'])
            scaler = best['scaler']
            vals   = np.array([[inputs.get(f, 0.0) for f in fit['features']]])
            try:
                vals_scaled = scaler.transform(vals) if scaler is not None else vals
                pred        = float(best['model'].predict(vals_scaled)[0])
                rel_days    = int(np.clip(round(pred), 0, 500))
            except Exception:
                rel_days = int(round(fit['mean_doy']))

        season_start = datetime(year, season_start_month, 1)
        date = season_start + timedelta(days=rel_days)
        doy  = date.timetuple().tm_yday
        return {
            'doy': doy, 'date': date, 'rel_days': rel_days,
            'r2':  self.r2.get(event, 0.0),
            'mae': self.mae.get(event, 0.0),
            'event': event,
        }

    def equation_str(self, event: str, season_start_month: int = 6) -> str:
        """
        Human-readable model equation / summary (v5.3 style).
        Shows all fitted models with their LOO metrics, highlights the best.
        """
        if event not in self._fits:
            n = self.n_seasons.get(event, 0)
            return (f"Need ≥ 2 seasons with met data to fit a model (currently {n})")

        fit = self._fits[event]
        mo  = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        lbl = f"{event}_days_from_{mo.get(season_start_month,'Jan')}1"

        if fit['mode'] == 'mean':
            return (f"{lbl} ≈ {fit['mean_doy']:.0f}  "
                    f"[No feature |r|≥{MIN_CORR_THRESHOLD} — historical mean only]")

        # Best-model one-liner
        best_name = fit['best_name']
        r2        = fit['r2']
        mae       = fit['mae']
        feats     = fit['features']
        mode      = fit['mode']

        mode_labels = {
            'ridge':  'Ridge Regression',
            'loess':  'LOESS (locally-weighted)',
            'poly2':  'Polynomial deg-2',
            'poly3':  'Polynomial deg-3',
            'gpr':    'Gaussian Process (RBF kernel)',
        }

        lines = []

        # ── Primary equation (Ridge, if available) ────────────
        if fit.get('coefs') and fit.get('intercept') is not None:
            terms = [f"{fit['intercept']:.3f}"]
            for feat, coef in zip(feats, fit['coefs']):
                s = '+' if coef >= 0 else '-'
                terms.append(f"{s} {abs(coef):.5f} × {feat}")
            lines.append(f"Ridge:  {lbl}  =  " + "  ".join(terms) +
                         f"\n        [α={fit['alpha']:.4f}, "
                         f"LOO R²={all_mods_r2_str(fit, 'Ridge')}, "
                         f"MAE=±{mae_str(fit, 'Ridge')} d]")

        # ── Summary line for other models ─────────────────────
        for mname, mres in fit.get('all_models', {}).items():
            if mname == 'Ridge': continue
            r2_m  = mres['loo_r2']
            mae_m = mres['loo_mae']
            r2_s  = f"{r2_m:.3f}" if not np.isnan(r2_m) else "N/A"
            mae_s = f"{mae_m:.1f}" if not np.isnan(mae_m) else "N/A"
            lines.append(f"{mname}:  {mode_labels.get(mres['mode'], mname)}({', '.join(feats)})  "
                         f"[LOO R²={r2_s}, MAE=±{mae_s} d]")

        # Highlight best model
        lines.append(f"\n★ Best model: {best_name}  "
                     f"[LOO R²={r2:.3f}, MAE=±{mae:.1f} d,  {len(feats)} feature(s)]")
        return "\n".join(lines)

    def corr_table_for_display(self, event: str) -> pd.DataFrame:
        if event not in self._fits:
            return pd.DataFrame()
        fit = self._fits[event]
        ct  = self.corr_tables.get(event)
        if ct is None or len(ct) == 0:
            return pd.DataFrame()
        in_model       = set(fit['features'])
        selected_first = fit['features'][0] if fit['features'] else None
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
                         'Composite': row.get('Composite', row['|r|']),
                         'Role': role})
        return pd.DataFrame(rows)

    def export_coefficients(self, season_start_month: int = 6) -> pd.DataFrame:
        records = []
        for event in ['SOS', 'POS', 'EOS']:
            if event not in self._fits: continue
            fit = self._fits[event]
            if fit['mode'] == 'mean':
                records.append({'Event': event, 'Feature': 'INTERCEPT',
                                 'Coefficient': fit['mean_doy'], 'Model': 'mean',
                                 'Best_Model': 'mean', 'R2_LOO': 0.0, 'MAE_days': fit['mae']})
            else:
                best_name = fit['best_name']
                for mname, mres in fit.get('all_models', {}).items():
                    is_best = (mname == best_name)
                    if mres.get('coefs'):
                        for feat, coef in zip(fit['features'], mres['coefs']):
                            records.append({
                                'Event': event, 'Feature': feat,
                                'Coefficient': round(coef, 6), 'Model': mname,
                                'Best_Model': '★' if is_best else '',
                                'R2_LOO': round(mres['loo_r2'], 4),
                                'MAE_days': round(mres['loo_mae'], 2),
                            })
                    else:
                        records.append({
                            'Event': event, 'Feature': str(fit['features']),
                            'Coefficient': float('nan'), 'Model': mname,
                            'Best_Model': '★' if is_best else '',
                            'R2_LOO': round(mres['loo_r2'], 4),
                            'MAE_days': round(mres['loo_mae'], 2),
                        })
        return pd.DataFrame(records)


# Helper functions used by equation_str
def all_mods_r2_str(fit, mname):
    m = fit.get('all_models', {}).get(mname)
    if m is None: return 'N/A'
    v = m['loo_r2']
    return f"{v:.3f}" if not np.isnan(v) else "N/A"

def mae_str(fit, mname):
    m = fit.get('all_models', {}).get(mname)
    if m is None: return 'N/A'
    v = m['loo_mae']
    return f"{v:.1f}" if not np.isnan(v) else "N/A"


# ═══════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════

def compute_sensitivity_analysis(predictor, train_df):
    result    = {}; dominants = {}; feat_stds = {}
    meta_cols = {'Year', 'Event', 'Target_DOY', 'LOS_Days', 'Peak_NDVI', 'Season_Start'}
    feat_cols = [c for c in train_df.columns
                 if c not in meta_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    for f in feat_cols:
        vals = train_df[f].dropna()
        feat_stds[f] = float(vals.std()) if len(vals) > 1 else 1.0
    for ev in ['SOS', 'POS', 'EOS']:
        if ev not in predictor._fits: continue
        fit = predictor._fits[ev]
        # Only Ridge has interpretable linear coefficients for sensitivity
        ridge_res = fit.get('all_models', {}).get('Ridge')
        if not ridge_res or not ridge_res.get('coefs') or not fit.get('features'): continue
        sub      = train_df[train_df['Event'] == ev]['Target_DOY'].dropna()
        mean_tgt = float(sub.mean()) if len(sub) > 0 else 1.0
        ev_result = {}
        for feat, coef in zip(fit['features'], ridge_res['coefs']):
            std_f      = feat_stds.get(feat, 1.0)
            days_shift = coef * std_f
            pct        = (days_shift / max(abs(mean_tgt), 1)) * 100
            ev_result[feat] = {
                'days_per_std': round(days_shift, 1),
                'pct_of_mean':  round(pct, 2),
                'direction':    'delays' if days_shift > 0 else 'advances',
                'coef':         round(coef, 5),
                'std':          round(std_f, 3),
            }
        result[ev] = ev_result
        if ev_result:
            dom = max(ev_result, key=lambda f: abs(ev_result[f]['days_per_std']))
            dominants[ev] = {
                'feature': dom,
                'days_per_std': ev_result[dom]['days_per_std'],
                'direction': ev_result[dom]['direction'],
            }
    return result, dominants


# ═══════════════════════════════════════════════════════════════
# PLOTS  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════

def plot_ndvi_phenology(ndvi_raw, pheno_df, season_window=None, interp_freq=5):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    dates   = pd.to_datetime(ndvi_raw['Date'])
    ax.scatter(dates, ndvi_raw['NDVI'], color='#A5D6A7', s=18, alpha=0.55,
               label='NDVI (raw obs)', zorder=3)
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
    ndvi_5d = ndvi_5d.reindex(full_range)
    n, ndvi_vals = len(ndvi_5d), ndvi_5d.values.copy()
    valid_mask   = ~np.isnan(ndvi_vals)
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
    in_gap = False; gap_s = None
    for i in range(n):
        nan_now = np.isnan(sm_arr[i])
        if nan_now and not in_gap: gap_s = ndvi_5d.index[i]; in_gap = True
        elif not nan_now and in_gap:
            ax.axvspan(gap_s, ndvi_5d.index[i], color='#BDBDBD', alpha=0.30, label='Data gap')
            in_gap = False
    if in_gap: ax.axvspan(gap_s, ndvi_5d.index[-1], color='#BDBDBD', alpha=0.30)
    if season_window:
        ws_m, we_m = season_window
        y_min = ndvi_5d.index.year.min(); y_max = ndvi_5d.index.year.max() + 1
        plotted = False
        for yr in range(y_min, y_max + 1):
            try:
                ws = pd.Timestamp(f"{yr}-{ws_m:02d}-01")
                we = (pd.Timestamp(f"{yr+1}-{we_m:02d}-28") if ws_m > we_m
                      else pd.Timestamp(f"{yr}-{we_m:02d}-28"))
                ds, de = ndvi_5d.index[0], ndvi_5d.index[-1]
                if we < ds or ws > de: continue
                ax.axvspan(max(ws, ds), min(we, de), color='#A5D6A7', alpha=0.12, zorder=0,
                           label='Selected season window' if not plotted else '')
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
        if pd.notna(base):
            ax.hlines(base, seg_st, seg_en, colors='#F57F17', lw=1.1, ls=':', alpha=0.75,
                      label='Base NDVI' if not base_p else '', zorder=4); base_p = True
        if pd.notna(thr_s):
            ax.hlines(thr_s, seg_st, seg_en, colors='#66BB6A', lw=1.2, ls='--', alpha=0.70,
                      label='SOS threshold' if not thr_sos_p else '', zorder=4); thr_sos_p = True
        if pd.notna(thr_e):
            ax.hlines(thr_e, seg_st, seg_en, colors='#FFA726', lw=1.2, ls='--', alpha=0.70,
                      label='EOS threshold' if not thr_eos_p else '', zorder=4); thr_eos_p = True
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
    win_str = (f"  |  Window: {mo.get(season_window[0],'?')} → {mo.get(season_window[1],'?')}"
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
        vals = (pheno_df[doy_col].values.astype(float)
                if doy_col in pheno_df.columns else np.zeros(len(yrs)))
        ax.bar(yrs, vals, color=clr, alpha=0.45, width=0.7, edgecolor='white')
        ax.plot(yrs, vals, 'o-', color=clr, ms=6, lw=2, markeredgecolor='white', markeredgewidth=1.2)
        if ev == 'LOS':
            ax.set_ylabel('Days')
        elif ev != 'EOS' and date_col and date_col in pheno_df.columns:
            unique_v = np.linspace(vals.min(), vals.max(), 5).astype(int)
            tick_l   = []
            for doy in unique_v:
                try: tick_l.append((pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(doy) - 1)).strftime('%b %d'))
                except: tick_l.append(str(doy))
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
              if ev in predictor._fits
              and predictor._fits[ev].get('mode') not in ('mean',)
              and predictor._fits[ev].get('best_model') is not None]
    if not events:
        return None
    fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 4.5), squeeze=False)
    clrs = {'SOS': '#4CAF50', 'POS': '#1565C0', 'EOS': '#FF6F00'}
    for ax, ev in zip(axes[0], events):
        fit    = predictor._fits[ev]
        sub    = train_df[train_df['Event'] == ev].copy()
        feats  = [f for f in fit['features'] if f in sub.columns]
        if not feats: continue
        Xf     = sub[feats].fillna(sub[feats].median())
        scaler = fit['best_scaler']
        model  = fit['best_model']
        mode   = fit['mode']
        try:
            if mode in ('poly2', 'poly3'):
                pred = model.predict(Xf.values)
            elif scaler is not None:
                pred = model.predict(scaler.transform(Xf.values))
            else:
                pred = model.predict(Xf.values)
        except Exception:
            continue
        obs  = sub['Target_DOY'].values
        ax.scatter(obs, pred, color=clrs[ev], s=80, edgecolors='white', lw=1.5, zorder=3, alpha=0.9)
        lims = [min(obs.min(), pred.min()) - 8, max(obs.max(), pred.max()) + 8]
        ax.plot(lims, lims, 'k--', lw=1.2); ax.set_xlim(lims); ax.set_ylim(lims)
        best_name = fit.get('best_name', mode)
        ax.set_title(f'{ev}  [{best_name}]  R²(LOO)={predictor.r2.get(ev, 0):.3f}'
                     f'  MAE={predictor.mae.get(ev, 0):.1f} d\n{" + ".join(feats)}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Observed (days from season start)')
        ax.set_ylabel('Predicted (days from season start)')
        ax.grid(True, alpha=0.25); ax.set_facecolor('#FAFFF8')
    fig.suptitle('Observed vs Predicted (training fit — best model per event)',
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
                     f'Coloured = |r| ≥ {MIN_CORR_THRESHOLD}  ·  Grey = below threshold',
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#F8FBF7')
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
    ax3 = axes[2]; ax3.axis('off')
    stats_text = (
        f"NDVI Data Summary\n{'─'*30}\n"
        f"Observations:   {ndvi_info['n_obs']}\n"
        f"Years covered:  {ndvi_info['year_range']}\n"
        f"Cadence:        {ndvi_info['cadence_d']:.1f} days\n"
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
    fig.suptitle('Uploaded Data — Automatic Characterization (No Hardcoded Assumptions)',
                 fontsize=12, fontweight='bold', color='#1B4332')
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
    abs_max   = max(abs(mat).max(), 1.0)
    ev_colors = {'SOS': '#43A047', 'POS': '#1565C0', 'EOS': '#E65100'}
    fig = plt.figure(figsize=(16, max(5, n_feats * 0.7 + 3)))
    fig.patch.set_facecolor('#F8FBF7')
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.8], wspace=0.45)
    ax_hm = fig.add_subplot(gs[0])
    im = ax_hm.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax_hm.set_xticks(range(n_evs)); ax_hm.set_xticklabels(ev_list, fontsize=12, fontweight='bold')
    ax_hm.set_yticks(range(n_feats)); ax_hm.set_yticklabels(all_feats, fontsize=10)
    for i in range(n_feats):
        for j in range(n_evs):
            v = mat[i, j]; tc = 'white' if abs(v) > abs_max * 0.55 else '#1A1A1A'
            sign = '+' if v >= 0 else ''
            ax_hm.text(j, i, f'{sign}{v:.1f}d', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=tc)
    cb = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cb.set_label('Days shifted per 1σ increase', fontsize=8); cb.ax.tick_params(labelsize=8)
    ax_hm.set_title('Sensitivity Heatmap\n(+red=delays  ·  −blue=advances)',
                    fontsize=10, fontweight='bold', color='#1B4332', pad=8)
    ax_hm.spines[:].set_visible(False)
    for j, ev in enumerate(ev_list):
        ax_hm.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (j - 0.48, -0.48), 0.96, n_feats - 0.04, boxstyle='round,pad=0.02',
            linewidth=2, edgecolor=ev_colors.get(ev, '#555'), facecolor='none', zorder=5))
    ax_bar = fig.add_subplot(gs[1])
    bar_h = 0.22; y_pos = np.arange(n_feats)
    offsets = np.linspace(-(n_evs - 1) / 2, (n_evs - 1) / 2, n_evs) * bar_h
    for j, ev in enumerate(ev_list):
        vals     = [mat[i, j] for i in range(n_feats)]
        bar_clrs = [ev_colors.get(ev, '#888') if abs(v) > 0.5 else '#CFCFCF' for v in vals]
        ax_bar.barh(y_pos + offsets[j], vals, height=bar_h * 0.85,
                    color=bar_clrs, edgecolor='white', lw=0.3, label=ev, alpha=0.88)
    ax_bar.axvline(0, color='#37474F', lw=1.0)
    ax_bar.set_yticks(y_pos); ax_bar.set_yticklabels(all_feats, fontsize=10)
    ax_bar.set_xlabel('Days shifted per 1σ increase', fontsize=9, fontweight='bold')
    ax_bar.set_title('Driver Analysis', fontsize=10, fontweight='bold', color='#1B4332', pad=8)
    ax_bar.grid(True, axis='x', alpha=0.22, ls='--'); ax_bar.set_facecolor('#FAFFF8')
    ax_bar.legend(title='Event', fontsize=9, title_fontsize=9, loc='lower right', framealpha=0.92,
                  handles=[plt.matplotlib.patches.Patch(color=ev_colors[e], label=e) for e in ev_list])
    ax_bar.spines['top'].set_visible(False); ax_bar.spines['right'].set_visible(False)
    fig.suptitle('Climate Driver Sensitivity', fontsize=12, fontweight='bold', color='#1B4332', y=1.01)
    fig.tight_layout()
    return fig


def plot_driver_dominance_cards(sensitivity, dominants):
    ev_list = [ev for ev in ['SOS', 'POS', 'EOS'] if ev in sensitivity and sensitivity[ev]]
    if not ev_list: return None
    ev_colors = {'SOS': '#43A047', 'POS': '#1565C0', 'EOS': '#E65100'}
    fig, axes = plt.subplots(1, len(ev_list),
                             figsize=(6 * len(ev_list), max(4, len(ev_list) * 1.5 + 2)))
    if len(ev_list) == 1: axes = [axes]
    fig.patch.set_facecolor('#F8FBF7')
    for ax, ev in zip(axes, ev_list):
        ev_sens = sensitivity[ev]
        ranked  = sorted(ev_sens.items(), key=lambda x: abs(x[1]['days_per_std']), reverse=True)
        feats   = [r[0] for r in ranked]; vals = [r[1]['days_per_std'] for r in ranked]
        colors  = ['#E53935' if v > 0 else '#1E88E5' for v in vals]
        abs_max = max((abs(v) for v in vals), default=1.0)
        offset  = abs_max * 0.04
        ax.barh(range(len(feats)), vals, color=colors, alpha=0.82, edgecolor='white', lw=0.5, zorder=3)
        for i, (feat, val) in enumerate(zip(feats, vals)):
            sign   = '+' if val >= 0 else ''
            direct = '→ delays' if val > 0 else '→ advances'
            x_pos  = val + offset * (1 if val >= 0 else -1)
            ax.text(x_pos, i, f'{sign}{val:.1f}d  {direct}',
                    va='center', ha='left' if val >= 0 else 'right',
                    fontsize=8.5, color='#222222', zorder=4)
        for i, (feat, val) in enumerate(zip(feats[:3], vals[:3])):
            badge_x = -abs_max * 1.28
            ax.text(badge_x, i, f'#{i+1}', va='center', ha='center',
                    fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=colors[i], edgecolor='none'))
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=10)
        ax.set_xlim(-abs_max * 1.45, abs_max * 1.6)
        ax.axvline(0, color='#37474F', lw=1.2, zorder=2)
        ax.set_xlabel('Days shifted per 1σ increase', fontsize=9)
        ax.set_title(f'{ev} — Driver Ranking\nDominant: {dominants.get(ev, {}).get("feature", "—")}',
                     fontsize=11, fontweight='bold', color=ev_colors.get(ev, '#333'))
        ax.set_facecolor('#FAFFF8'); ax.grid(True, axis='x', alpha=0.2, ls='--', zorder=1)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Climate Driver Ranking per Phenological Event',
                 fontsize=11, fontweight='bold', color='#1B4332')
    fig.tight_layout()
    return fig


def plot_radar_chart(sensitivity, selected_event='SOS'):
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    ev_list = [ev for ev in ['SOS', 'POS', 'EOS'] if ev in sensitivity and sensitivity[ev]]
    if not ev_list: return None
    ev_colors  = {'SOS': '#43A047', 'POS': '#1565C0', 'EOS': '#E65100'}
    ev_labels  = {'SOS': 'Start of Season', 'POS': 'Peak of Season', 'EOS': 'End of Season'}
    all_feats = sorted({f for ev in ev_list for f in sensitivity[ev]})
    N = len(all_feats)
    if N < 3: return None
    fig = plt.figure(figsize=(15, 6)); fig.patch.set_facecolor('#F8FBF7')
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.4)
    ax_radar = fig.add_subplot(gs[0], polar=True)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]
    for ev in ev_list:
        vals_radar = [abs(sensitivity[ev].get(f, {}).get('days_per_std', 0)) for f in all_feats]
        vals_radar += vals_radar[:1]
        lw = 2.5 if ev == selected_event else 1.0; alpha = 0.30 if ev == selected_event else 0.10
        ax_radar.plot(angles, vals_radar, color=ev_colors[ev], linewidth=lw, label=ev)
        ax_radar.fill(angles, vals_radar, color=ev_colors[ev], alpha=alpha)
    ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(all_feats, fontsize=9, color='#444')
    ax_radar.set_facecolor('#FAFFF8'); ax_radar.grid(color='#CCCCCC', linestyle='--', alpha=0.5)
    ax_radar.set_title('Factor Influence Magnitude\n(absolute days per 1σ)',
                       fontsize=11, fontweight='bold', color='#1B4332', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.85,
                    handles=[mpatches.Patch(color=ev_colors[e], label=e) for e in ev_list])
    ax_grid = fig.add_subplot(gs[1]); ax_grid.set_xlim(0, 1); ax_grid.set_ylim(0, 1); ax_grid.axis('off')
    ax_grid.set_title('Cross-Event Driver Dominance\n(top driver for each event)',
                      fontsize=11, fontweight='bold', color='#1B4332', pad=14)
    n_ev   = len(ev_list); cell_h = 0.80 / n_ev; y_start = 0.88
    for i, ev in enumerate(ev_list):
        ev_sens = sensitivity[ev]
        if not ev_sens: continue
        dom_feat = max(ev_sens, key=lambda f: abs(ev_sens[f]['days_per_std']))
        dom_val  = ev_sens[dom_feat]['days_per_std']
        dom_dir  = 'delays' if dom_val > 0 else 'advances'
        sign     = '+' if dom_val > 0 else ''
        bar_color = ev_colors[ev]
        y_box = y_start - i * (cell_h + 0.05)
        box = FancyBboxPatch((0.04, y_box - cell_h + 0.01), 0.92, cell_h - 0.01,
                             boxstyle='round,pad=0.02', linewidth=1.5,
                             edgecolor=bar_color, facecolor=bar_color + '18')
        ax_grid.add_patch(box)
        ax_grid.text(0.08, y_box - 0.012, f'{ev}  {ev_labels[ev]}',
                     fontsize=9, color=bar_color, fontweight='bold', va='top')
        ax_grid.text(0.08, y_box - 0.038, dom_feat, fontsize=13, color='#1A1A1A', fontweight='bold', va='top')
        ax_grid.text(0.08, y_box - 0.068, f'↑ 1σ {dom_dir} {ev} by {sign}{dom_val:.1f} days',
                     fontsize=9, color='#555555', va='top')
        ranked_feats = sorted(ev_sens.items(), key=lambda x: abs(x[1]['days_per_std']), reverse=True)
        bar_x = 0.60
        for j, (feat, finfo) in enumerate(ranked_feats[:4]):
            frac = abs(finfo['days_per_std']) / max(abs(v['days_per_std']) for v in ev_sens.values())
            fc   = '#E53935' if finfo['days_per_std'] > 0 else '#1E88E5'
            ax_grid.barh(y_box - 0.025 - j * 0.022, frac * 0.32, left=bar_x, height=0.016,
                         color=fc, alpha=0.75)
            ax_grid.text(bar_x + frac * 0.32 + 0.01, y_box - 0.025 - j * 0.022,
                         feat, fontsize=7, va='center', color='#555')
    fig.suptitle('Radar: Factor Influence  ·  Cross-Event Driver Summary',
                 fontsize=12, fontweight='bold', color='#1B4332')
    fig.tight_layout()
    return fig


def plot_met_with_ndvi(met_df, ndvi_df, raw_params, pheno_df, interp_freq=5):
    ndvi_s = ndvi_df.set_index('Date')['NDVI'].sort_index()
    if ndvi_s.index.duplicated().any():
        ndvi_s = ndvi_s.groupby(ndvi_s.index).mean()
    full_r  = pd.date_range(start=ndvi_s.index.min(), end=ndvi_s.index.max(), freq=f'{interp_freq}D')
    ndvi_5d = ndvi_s.reindex(ndvi_s.index.union(full_r)).interpolate(method='time').reindex(full_r)
    if pheno_df is None or len(pheno_df) == 0: return []
    ALL_COLS    = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#546E7A',
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
            eos_d    = row.get('EOS_Date', pd.NaT)
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
            if n_panels == 1: axes_p = [axes_p]
            fig.patch.set_facecolor('#FAFFF8')
            fig.suptitle(f"NDVI + Met — Season {yr}  [ {sos_str} → {eos_str} ]",
                         fontsize=14, fontweight='bold', y=0.99)
            def _draw_panel(ax, param_list, title, bar_keys=('PRECTOTCORR','PRECTOT','RAIN')):
                ax.fill_between(df_met['Date'], ndvi_seg, alpha=0.18, color='#2E7D32')
                ax.plot(df_met['Date'], ndvi_seg, color='#2E7D32', lw=2.5, label='NDVI')
                ax.set_ylabel('NDVI', color='#2E7D32', fontsize=11, fontweight='bold')
                ax.set_ylim(0, 1.05); ax.tick_params(axis='y', labelcolor='#2E7D32')
                ax.grid(True, linestyle='--', alpha=0.28); ax.set_facecolor('#FAFFF8')
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
                return ax
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
# MAIN APP  (unchanged from v2, with model-card updated to show best model name)
# ═══════════════════════════════════════════════════════════════

def main():
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
    ndvi_file = st.sidebar.file_uploader("NDVI File (CSV)", type=['csv'], key="ndvi_uploader")
    met_file  = st.sidebar.file_uploader("Meteorological File (CSV)", type=['csv'], key="met_uploader")

    _fp_ndvi = f"{ndvi_file.name}:{ndvi_file.size}" if ndvi_file else ""
    _fp_met  = f"{met_file.name}:{met_file.size}"   if met_file  else ""

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📅 Growing Season Window")
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
            st.sidebar.info(f"Cross-year window: **{sm_names[start_m]} → {sm_names[end_m]}**")
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
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📈 Prediction Model")
    st.sidebar.caption(
        "The app now fits ALL models (Ridge, LOESS, Polynomial, GPR) simultaneously "
        "and automatically selects the best one per event by LOO R². "
        "This setting is used as a *preference* — if another model performs better, it wins.")
    model_opts = {
        "Ridge Regression (default)":    "ridge",
        "LOESS Smoothing":               "loess",
        "Polynomial Regression (Deg 2)": "poly2",
        "Polynomial Regression (Deg 3)": "poly3",
        "Gaussian Process":              "gpr",
    }
    model_sel = st.sidebar.radio("Preferred model type", list(model_opts.keys()), index=0,
                                  key="model_type_radio")
    model_key = model_opts[model_sel]

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🗓️ Climate Window")
    feat_window = st.sidebar.slider("Days before event to average climate", 7, 60, 15, 1,
                                     key="feat_window_slider")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔢 Maximum Features in Model")
    max_features_override = st.sidebar.slider("Max climate variables per model", 1, 4, 1, 1,
                                               key="max_feat_slider")
    if max_features_override >= 2:
        st.sidebar.markdown(
            '<div style="background:#FFF8E1;padding:8px 12px;border-radius:8px;'
            'border-left:3px solid #F9A825;font-size:0.80rem;margin-top:4px">'
            '⚠️ Using 2+ features with fewer than 6 seasons can overfit.</div>',
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

    if not (ndvi_file and met_file):
        st.markdown("""
<div class="upload-panel">
<h3>👈 Upload your two data files using the sidebar to begin</h3>
<b>File 1 — NDVI CSV</b><br>
Any CSV with a date column and an NDVI column.<br><br>
<b>File 2 — Meteorological CSV</b><br>
Daily climate data. Download free from
<a href="https://power.larc.nasa.gov/data-access-viewer/" target="_blank">NASA POWER</a>.<br><br>
<b>Model Engine (updated):</b><br>
• All models (Ridge, LOESS, Polynomial, GPR) are fitted simultaneously with LOO cross-validation<br>
• Best model selected automatically per event by LOO R²<br>
• RidgeCV with 30 log-spaced alphas · LOESS via statsmodels + PCA · GPR with ConstantKernel×RBF<br>
• All v2 features retained: phenology extraction, feature selection, sensitivity analysis<br>
</div>
        """, unsafe_allow_html=True)
        return

    # ── PARSE ─────────────────────────────────────────────────
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

    _, _, interp_freq = detect_ndvi_cadence(ndvi_df)
    ndvi_info  = characterize_ndvi_data(ndvi_df)
    met_info   = characterize_met_data(met_df, raw_params)

    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ NDVI loaded — {ndvi_info['n_obs']} observations · {ndvi_info['n_years']} years")
    st.sidebar.success(f"✅ Met loaded — {len(raw_params)} climate parameters")
    if derived:
        st.sidebar.info(f"+ {len(derived)} derived features computed automatically")

    icons = {'SOS': '🌱', 'POS': '🌿', 'EOS': '🍂'}

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Summary", "🔬 Season Extraction & Models",
        "📈 Climate Correlations", "🎯 Driver Sensitivity", "🔮 Predict", "📖 User Guide"])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — DATA SUMMARY
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<p class="section-title">Your Uploaded Data</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="label">📅 Year Range</div>'
                    f'<div class="value">{ndvi_info["year_range"]}</div>'
                    f'<div class="sub">{ndvi_info["n_obs"]} NDVI observations</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">🌿 Mean NDVI</div>'
                    f'<div class="value">{ndvi_info["ndvi_mean"]}</div>'
                    f'<div class="sub">std = {ndvi_info["ndvi_std"]}</div></div>',
                    unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="label">📏 NDVI Range</div>'
                    f'<div class="value">{ndvi_info["data_range"]}</div>'
                    f'<div class="sub">P5 = {ndvi_info["ndvi_p5"]}  ·  P95 = {ndvi_info["ndvi_p95"]}</div></div>',
                    unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="label">🔁 Cadence</div>'
                    f'<div class="value">{ndvi_info["cadence_d"]:.0f} days</div>'
                    f'<div class="sub">Auto-detected from file</div></div>',
                    unsafe_allow_html=True)
        fig_ds = plot_data_summary(ndvi_info, met_info)
        st.pyplot(fig_ds, use_container_width=True)
        st.markdown('<p class="section-title">Climate Parameters Available</p>', unsafe_allow_html=True)
        if met_info:
            met_summary_df = pd.DataFrame([
                {'Parameter': p, 'Mean': round(v['mean'], 3), 'Std Dev': round(v['std'], 3),
                 'Min': round(v['min'], 3), 'Max': round(v['max'], 3)}
                for p, v in met_info.items()])
            st.dataframe(met_summary_df.style.background_gradient(subset=['Mean'], cmap='Greens'),
                         use_container_width=True, hide_index=True)
        if derived:
            st.markdown(
                f'<div class="banner-info">ℹ️ In addition to the {len(raw_params)} parameters in your file, '
                f'<b>{len(derived)} derived variables</b> were computed automatically.</div>',
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
        if n_seasons == 0:
            st.error("No complete seasons found."); return
        elif n_seasons == 1:
            st.warning("Only **1 season** was extracted. Model training requires at least 2 seasons.")
        elif n_seasons == 2:
            st.warning("**2 seasons** extracted — model results are exploratory only.")
        elif n_seasons <= 4:
            st.info(f"**{n_seasons} seasons** extracted — small dataset. Results are indicative.")
        elif n_seasons < 7:
            st.info(f"**{n_seasons} seasons** extracted — usable dataset.")
        else:
            st.success(f"✅ **{n_seasons} growing seasons** extracted — good dataset for modelling.")

        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.pyplot(plot_ndvi_phenology(ndvi_df, pheno_df,
                                          season_window=(start_m, end_m), interp_freq=interp_freq))
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

        st.pyplot(plot_pheno_trends(pheno_df))

        # ── MODEL TRAINING ────────────────────────────────────
        st.markdown('<p class="section-title">Predictive Model Training</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="banner-info">🆕 <b>Updated model engine (v5.3):</b> '
            'Ridge, LOESS, Polynomial-2, Polynomial-3, and GPR are all fitted simultaneously. '
            'The best model per event is selected automatically by Leave-One-Out R². '
            'Model equations and the "All Models" comparison are shown below.</div>',
            unsafe_allow_html=True)

        with st.spinner("Training models (Ridge + LOESS + Poly + GPR)…"):
            train_df  = make_training_features(pheno_df, met_df, all_params, window=feat_window)
            predictor = UniversalPredictor()
            predictor.train(train_df, all_params, model_key=model_key,
                            user_max_features=max_features_override)

        met_audit = audit_met_coverage(met_df, ndvi_df, pheno_df, window=feat_window)
        for w in met_audit['warnings']:
            st.markdown(f'<div class="banner-warn">{w}</div>', unsafe_allow_html=True)

        cov = met_audit['per_event_coverage']
        if cov:
            cov_rows = []
            for ev in ['SOS', 'POS', 'EOS']:
                if ev not in cov: continue
                c = cov[ev]
                missing_str = (", ".join(str(y) for y in c['seasons_missing'])
                               if c['seasons_missing'] else "None")
                reliable = "✅ OK" if c['n_seasons_with_data'] >= 5 else (
                    "⚠️ Low" if c['n_seasons_with_data'] >= 3 else "❌ Very low")
                cov_rows.append({'Event': ev,
                                 'Seasons with climate data': c['n_seasons_with_data'],
                                 'Total seasons': c['n_seasons_total'],
                                 'Coverage': f"{c['coverage_pct']:.0f}%",
                                 'Missing years': missing_str,
                                 'Model reliability': reliable})
            if cov_rows:
                st.markdown("**Climate window coverage per event:**")
                st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

        st.session_state.update({
            'pheno_df': pheno_df, 'met_df': met_df, 'train_df': train_df,
            'predictor': predictor, 'all_params': all_params,
            'raw_params': raw_params, 'ndvi_df': ndvi_df,
            'ndvi_info': ndvi_info, 'met_info': met_info, 'interp_freq': interp_freq,
        })

        # ── Model performance cards ───────────────────────────
        st.markdown("**Model performance (LOO cross-validation):**")
        c1, c2, c3 = st.columns(3)

        def _card(col, ev):
            ev_full = {'SOS':'Start of Season','POS':'Peak of Season','EOS':'End of Season'}
            n_ev    = predictor.n_seasons.get(ev, 0)
            if ev not in predictor._fits:
                col.markdown(
                    f'<div class="metric-card"><div class="label">{icons[ev]} {ev}</div>'
                    f'<div class="value" style="color:#9E9E9E;font-size:1.3rem">Not fitted</div>'
                    f'<div class="sub">Need ≥ 2 seasons.<br>{n_ev} season(s) available</div></div>',
                    unsafe_allow_html=True); return
            fit      = predictor._fits[ev]
            r2       = predictor.r2.get(ev, 0)
            mae      = predictor.mae.get(ev, 0)
            n        = predictor.n_seasons.get(ev, 0)
            best_name = fit.get('best_name', fit.get('mode', '—'))
            if fit['mode'] == 'mean':
                col.markdown(
                    f'<div class="metric-card"><div class="label">{icons[ev]} {ev}</div>'
                    f'<div class="value" style="color:#9E9E9E">Mean only</div>'
                    f'<div class="sub">No feature met correlation threshold.<br>'
                    f'Typical error: ±{mae:.1f} days</div></div>', unsafe_allow_html=True)
            else:
                clr   = '#1B5E20' if r2 > 0.6 else '#E65100' if r2 > 0.3 else '#B71C1C'
                feats = fit.get('features', [])
                # Show all model LOO R² scores
                all_scores = "  |  ".join(
                    f"{mn}: {mres['loo_r2']:.2f}"
                    for mn, mres in fit.get('all_models', {}).items()
                    if not np.isnan(mres['loo_r2'])
                )
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{icons[ev]} {ev} — {ev_full[ev]}</div>'
                    f'<div class="value" style="color:{clr}">{r2*100:.1f}%</div>'
                    f'<div class="sub">Best: <b>{best_name}</b> · {n} seasons<br>'
                    f'Driver(s): <b>{", ".join(feats) or "—"}</b><br>'
                    f'Typical error: ±{mae:.1f} days<br>'
                    f'<span style="font-size:0.72rem;color:#888">All: {all_scores}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True)

        _card(c1, 'SOS'); _card(c2, 'POS'); _card(c3, 'EOS')

        for ev in ['SOS', 'POS', 'EOS']:
            n_ev = predictor.n_seasons.get(ev, 0)
            if 0 < n_ev <= 3:
                st.markdown(
                    f'<div class="banner-error">🔴 <b>{ev} model uses only {n_ev} season(s).</b> '
                    f'With n≤3, correlation values are mathematically unreliable. '
                    f'Upload more years of data.</div>', unsafe_allow_html=True)

        # ── Model equations ───────────────────────────────────
        st.markdown('<p class="section-title">Model Equations & All-Model Comparison</p>',
                    unsafe_allow_html=True)
        t_sos, t_pos, t_eos = st.tabs(
            [f"{icons['SOS']} SOS", f"{icons['POS']} POS", f"{icons['EOS']} EOS"])
        for ui_tab, ev in zip([t_sos, t_pos, t_eos], ['SOS', 'POS', 'EOS']):
            with ui_tab:
                eq   = predictor.equation_str(ev, season_start_month=start_m)
                eq_h = eq.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
                st.markdown(f'<div class="eq-box">{eq_h}</div>', unsafe_allow_html=True)

                # All-model LOO comparison table
                fit = predictor._fits.get(ev)
                if fit and fit.get('all_models'):
                    rows_cmp = []
                    for mname, mres in fit['all_models'].items():
                        r2_v  = mres['loo_r2'];  mae_v = mres['loo_mae']
                        rows_cmp.append({
                            'Model':    mname,
                            'LOO R²':   round(r2_v, 3) if not np.isnan(r2_v) else float('nan'),
                            'MAE (d)':  round(mae_v, 1) if not np.isnan(mae_v) else float('nan'),
                            'Selected': '★ Best' if mname == fit['best_name'] else '',
                        })
                    cmp_df = pd.DataFrame(rows_cmp).sort_values('LOO R²', ascending=False)
                    st.dataframe(
                        cmp_df.style.background_gradient(subset=['LOO R²'], cmap='Greens', vmin=0, vmax=1),
                        use_container_width=True, hide_index=True)

                ct = predictor.corr_table_for_display(ev)
                if not ct.empty:
                    def _sr(val):
                        if val.startswith('✅'): return 'background-color:#C8E6C9;color:#1B5E20;font-weight:600'
                        if val.startswith('➖') and 'Redundant' in val: return 'color:#9E9E9E;font-style:italic'
                        if val.startswith('➖'): return 'color:#555'
                        return 'color:#bbb'
                    fmt = {'Pearson r': '{:+.3f}', 'Spearman ρ': '{:+.3f}', 'Composite': '{:.3f}'}
                    sty = ct.style
                    if 'Pearson r'  in ct.columns: sty = sty.background_gradient(subset=['Pearson r'],  cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Spearman ρ' in ct.columns: sty = sty.background_gradient(subset=['Spearman ρ'], cmap='RdYlGn', vmin=-1, vmax=1)
                    if 'Composite'  in ct.columns: sty = sty.background_gradient(subset=['Composite'],  cmap='Greens', vmin=0, vmax=1)
                    sty = sty.applymap(_sr, subset=['Role']).format(fmt)
                    st.dataframe(sty, use_container_width=True, hide_index=True)

        fig_s = plot_obs_vs_pred(predictor, train_df)
        if fig_s:
            st.markdown('<p class="section-title">Observed vs Predicted</p>', unsafe_allow_html=True)
            st.pyplot(fig_s)

        st.markdown("---")
        dl_cols = [c for c in ['Year','SOS_DOY','POS_DOY','EOS_DOY','LOS_Days',
                                'Peak_NDVI','Amplitude','Base_NDVI','SOS_Date',
                                'POS_Date','EOS_Date'] if c in pheno_df.columns]
        coef_df = predictor.export_coefficients(season_start_month=start_m)
        col_d1, col_d2 = st.columns(2)
        col_d1.download_button("📥 Download Phenology Table (CSV)",
                               pheno_df[dl_cols].to_csv(index=False), "phenology_table.csv", "text/csv")
        col_d2.download_button("📥 Download Model Coefficients (CSV)",
                               coef_df.to_csv(index=False), "model_coefficients.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # TAB 3 — CLIMATE CORRELATIONS  (unchanged)
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<p class="section-title">Which Climate Variables Drive Each Seasonal Event?</p>',
                    unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        pheno_df_ss  = st.session_state.get('pheno_df')
        if predictor_ss is None:
            st.info("Complete the Season Extraction step first."); return
        fig_c = plot_correlation_summary(predictor_ss)
        if fig_c: st.pyplot(fig_c, use_container_width=True)
        st.markdown('<p class="section-title">Full Correlation Table</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col_st, ev in zip([c1, c2, c3], ['SOS', 'POS', 'EOS']):
            with col_st:
                st.markdown(f"**{icons[ev]} {ev}**")
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
                                     if 'Spearman ρ' in disp.columns else {'Pearson r':'{:+.3f}','Composite':'{:.3f}'})
                    st.dataframe(sty, use_container_width=True, hide_index=True, height=320)
                else:
                    st.info("No correlation data.")

        st.markdown('<p class="section-title">NDVI and Climate — Year by Year</p>',
                    unsafe_allow_html=True)
        _met  = st.session_state.get('met_df')
        _ndvi = st.session_state.get('ndvi_df')
        _rp   = st.session_state.get('raw_params', [])
        _if   = st.session_state.get('interp_freq', 5)
        if _met is not None and _ndvi is not None:
            figs_l = plot_met_with_ndvi(_met, _ndvi, _rp, pheno_df_ss, interp_freq=_if)
            for s_lbl, f_m in figs_l:
                st.markdown(f"**Season {s_lbl}**")
                st.pyplot(f_m, use_container_width=True); plt.close(f_m)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — DRIVER SENSITIVITY  (unchanged)
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<p class="section-title">Climate Driver Sensitivity Analysis</p>',
                    unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        train_df_ss  = st.session_state.get('train_df')
        if predictor_ss is None:
            st.info("Complete Season Extraction first.")
        else:
            ridge_events = [ev for ev in ['SOS','POS','EOS']
                            if ev in predictor_ss._fits
                            and predictor_ss._fits[ev].get('all_models', {}).get('Ridge')
                            and predictor_ss._fits[ev]['all_models']['Ridge'].get('coefs')]
            if not ridge_events:
                st.markdown(
                    '<div class="banner-warn">⚠️ Sensitivity analysis requires Ridge coefficients. '
                    'No Ridge models fitted yet.</div>', unsafe_allow_html=True)
            else:
                sensitivity, dominants = compute_sensitivity_analysis(predictor_ss, train_df_ss)
                if not sensitivity:
                    st.warning("No sensitivity data available.")
                else:
                    ev_colors_hex  = {'SOS':'#E8F5E9','POS':'#E3F2FD','EOS':'#FFF3E0'}
                    ev_border_hex  = {'SOS':'#2E7D32','POS':'#1565C0','EOS':'#E65100'}
                    ev_icons       = {'SOS':'🌱','POS':'🌿','EOS':'🍂'}
                    ev_labels_full = {'SOS':'Start of Season','POS':'Peak of Season','EOS':'End of Season'}
                    dom_cols = st.columns(len(ridge_events))
                    for col_d, ev in zip(dom_cols, ridge_events):
                        dom = dominants.get(ev)
                        if dom:
                            d_days = dom['days_per_std']; sign = '+' if d_days > 0 else ''
                            dirstr = 'delays' if d_days > 0 else 'advances'
                            col_d.markdown(
                                f"<div style='background:{ev_colors_hex[ev]};padding:16px;border-radius:10px;"
                                f"border-left:4px solid {ev_border_hex[ev]};margin:6px 0'>"
                                f"<div style='font-size:0.78rem;color:#666;font-weight:600'>{ev_icons[ev]} {ev_labels_full[ev]}</div>"
                                f"<div style='font-size:1.4rem;font-weight:700;color:{ev_border_hex[ev]};margin:4px 0'>{dom['feature']}</div>"
                                f"<div style='font-size:0.85rem;color:#555'>↑ 1σ {dirstr} {ev} by <b>{sign}{d_days:.1f} days</b></div>"
                                f"</div>", unsafe_allow_html=True)
                    fig_hm = plot_sensitivity_heatmap(sensitivity, predictor_ss, train_df_ss)
                    if fig_hm: st.pyplot(fig_hm, use_container_width=True); plt.close(fig_hm)
                    fig_dc = plot_driver_dominance_cards(sensitivity, dominants)
                    if fig_dc: st.pyplot(fig_dc, use_container_width=True); plt.close(fig_dc)
                    available_evs = [ev for ev in ['SOS','POS','EOS'] if ev in sensitivity]
                    radar_ev = st.radio("Highlight event on radar:", available_evs, horizontal=True,
                                        key="radar_ev_sel",
                                        format_func=lambda e: {'SOS':'🌱 SOS','POS':'🌿 POS','EOS':'🍂 EOS'}[e])
                    fig_rd = plot_radar_chart(sensitivity, selected_event=radar_ev)
                    if fig_rd: st.pyplot(fig_rd, use_container_width=True); plt.close(fig_rd)

    # ══════════════════════════════════════════════════════════
    # TAB 5 — PREDICT  (updated for new fit dict structure)
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<p class="section-title">Predict Phenology Dates for Any Year</p>',
                    unsafe_allow_html=True)
        predictor_ss = st.session_state.get('predictor')
        train_df_ss  = st.session_state.get('train_df')
        pheno_ss     = st.session_state.get('pheno_df')
        if predictor_ss is None:
            st.info("Complete Season Extraction first."); return

        st.markdown(
            '<div class="banner-info">Enter expected climate conditions for your target year. '
            'Values are pre-filled with historical averages. '
            'The best-performing model (shown in the metric cards) will be used for each event.</div>',
            unsafe_allow_html=True)

        mo = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
              7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        ev_colors_hex  = {'SOS':'#E8F5E9','POS':'#E3F2FD','EOS':'#FFF3E0'}
        ev_border_hex  = {'SOS':'#2E7D32','POS':'#1565C0','EOS':'#E65100'}
        ev_labels_full = {'SOS':'🌱 Start of Season (SOS)','POS':'🌿 Peak of Season (POS)','EOS':'🍂 End of Season (EOS)'}
        ev_inputs = {ev: {} for ev in ['SOS', 'POS', 'EOS']}

        any_model = False
        for ev in ['SOS', 'POS', 'EOS']:
            fit = predictor_ss._fits.get(ev, {})
            if not fit or fit.get('mode') == 'mean': continue
            feats = fit.get('features', [])
            if not feats: continue
            any_model = True
            r2  = predictor_ss.r2.get(ev, 0)
            mae = predictor_ss.mae.get(ev, 0)
            best_name = fit.get('best_name', '—')

            hist_hint = ""
            if pheno_ss is not None and f'{ev}_Date' in pheno_ss.columns:
                ev_dates = pheno_ss[f'{ev}_Date'].dropna()
                if len(ev_dates) > 0:
                    med_m = int(ev_dates.dt.month.median()); med_d = int(ev_dates.dt.day.median())
                    hist_hint = f" · Historically around {mo[med_m]} {med_d} (±{mae:.0f} d)"

            st.markdown(
                f"<div style='background:{ev_colors_hex[ev]};padding:14px 18px;border-radius:10px;"
                f"border-left:4px solid {ev_border_hex[ev]};margin:10px 0'>"
                f"<b>{ev_labels_full[ev]}</b>"
                f"<span style='font-size:0.82rem;color:#666'>&nbsp;&nbsp;"
                f"Best model: <b>{best_name}</b> · R²={r2:.0%}{hist_hint}</span></div>",
                unsafe_allow_html=True)

            col_list = st.columns(min(len(feats), 4))
            for idx, f in enumerate(feats):
                default = 0.0
                if train_df_ss is not None and f in train_df_ss.columns:
                    ev_sub = train_df_ss[train_df_ss['Event'] == ev]
                    vals   = ev_sub[f].dropna() if len(ev_sub) > 0 else train_df_ss[f].dropna()
                    if len(vals): default = float(vals.mean())
                is_sum = any(k in f.upper() for k in ACCUM_KEYWORDS)
                vmin = vmax = None
                if train_df_ss is not None and f in train_df_ss.columns:
                    col_vals = train_df_ss[f].dropna()
                    if len(col_vals) >= 2:
                        vmin = float(col_vals.min()); vmax = float(col_vals.max())
                with col_list[idx % len(col_list)]:
                    ev_inputs[ev][f] = st.number_input(
                        f"{f}  [{ev}]", value=round(default, 3), format="%.3f",
                        key=f"inp_{ev}_{f}",
                        help=(f"{'Total (sum)' if is_sum else f'{feat_window}-day average'} of {f} "
                              f"before expected {ev}.\nHistorical mean: {default:.3f}"
                              + (f" | data range: [{vmin:.2f}–{vmax:.2f}]" if vmin is not None else "")))

        if not any_model:
            st.markdown(
                '<div class="banner-warn">⚠️ No predictive models fitted — '
                'not enough seasons or no climate variable showed sufficient correlation.</div>',
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
                        order_warns.append(f"Peak adjusted to ~{nd.strftime('%b %d')}")
                if 'POS' in results and 'EOS' in results:
                    if results['EOS']['rel_days'] <= results['POS']['rel_days']:
                        fb = (int(round(pheno_ss['EOS_Target'].mean()))
                              if pheno_ss is not None and 'EOS_Target' in pheno_ss.columns
                              else results['POS']['rel_days'] + 90)
                        corrected = max(fb, results['POS']['rel_days'] + 14)
                        nd = datetime(pred_year, start_m, 1) + timedelta(days=corrected)
                        results['EOS'].update({'rel_days': corrected,
                                               'doy': nd.timetuple().tm_yday, 'date': nd})
                        order_warns.append(f"End adjusted to ~{nd.strftime('%b %d')}")
                if order_warns:
                    st.markdown('<div class="banner-warn"><b>Note:</b> '
                                + ' · '.join(order_warns) + '</div>', unsafe_allow_html=True)

                cols = st.columns(len(results))
                for col, (ev, res) in zip(cols, results.items()):
                    ev_full = {'SOS':'Start of Season','POS':'Peak of Season','EOS':'End of Season'}
                    best_n  = predictor_ss._fits.get(ev, {}).get('best_name', '—')
                    col.markdown(
                        f'<div class="metric-card"><div class="label">{icons[ev]} {ev_full[ev]}</div>'
                        f'<div class="value">{res["date"].strftime("%b %d")}</div>'
                        f'<div class="sub">Day {res["doy"]} of {res["date"].year}<br>'
                        f'Model: {best_n} · R²={res["r2"]:.0%}<br>'
                        f'Typical error: ±{res["mae"]:.0f} days</div></div>',
                        unsafe_allow_html=True)

                if 'SOS' in results and 'EOS' in results:
                    sd = results['SOS']['date']; ed = results['EOS']['date']
                    los = (ed - sd).days if ed >= sd else (ed - sd).days + 365
                    st.markdown("---")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("📏 Season Length", f"{los} days")
                    if 'POS' in results:
                        mc2.metric("Green-up (SOS→POS)",
                                   f"{(results['POS']['date']-results['SOS']['date']).days} days")
                        mc3.metric("Senescence (POS→EOS)",
                                   f"{(results['EOS']['date']-results['POS']['date']).days} days")

                out = pd.DataFrame({
                    'Event':         list(results.keys()),
                    'Best Model':    [predictor_ss._fits.get(ev, {}).get('best_name','—')
                                      for ev in results],
                    'Predicted Date':[r['date'].strftime('%Y-%m-%d') for r in results.values()],
                    'Day of Year':   [r['doy'] for r in results.values()],
                    'R² (accuracy)': [round(r['r2'], 3) for r in results.values()],
                    'Typical error (days)': [round(r['mae'], 1) for r in results.values()],
                })
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Predictions (CSV)", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # TAB 6 — USER GUIDE
    # ══════════════════════════════════════════════════════════
    with tab6:
        st.markdown('<p class="section-title">User Guide</p>', unsafe_allow_html=True)
        st.markdown("""
This tool analyses the timing of seasonal changes in forest vegetation using satellite NDVI data
and daily climate records. It is designed for researchers and ecologists studying **forest phenology**
across any region of India — no configuration for forest type is required.

---

### 🆕 Model Engine Update (v5.3 Engine)

The prediction and model selection logic has been upgraded:

| Feature | Previous | Updated |
|---|---|---|
| Models fitted | One (user's choice) | All 5 simultaneously |
| Model selection | User-selected | Automatic — best LOO R² |
| Ridge alphas | 9 fixed values | 30 log-spaced (1e-3 → 1e4) |
| LOESS | Custom implementation | statsmodels + PCA projection |
| GPR kernel | RBF + WhiteKernel | ConstantKernel × RBF + WhiteKernel |
| LOO CV | 4 separate functions | Single generic `loo_cv_generic()` |
| Equation display | Best model only | All models with LOO scores |
| Small-n safety | ✅ (kept) | ✅ (kept) |
| Feature selection | ✅ (kept) | ✅ (kept) |

---

### 📘 Key Terms

**SOS** — Start of Season: when NDVI first rises above the amplitude threshold  
**POS** — Peak of Season: date of maximum greenness  
**EOS** — End of Season: when NDVI falls back below the threshold  
**LOS** — Length of Season: days between SOS and EOS  
**LOO R²** — Leave-One-Out R²: model accuracy estimated by holding out each season in turn  

---

### 📂 Data Format Requirements

**NDVI file:** Any CSV with date + NDVI columns (any standard date format).

**Meteorological file:** Continuous daily CSV (one row per day). Download free from
[NASA POWER](https://power.larc.nasa.gov/data-access-viewer/) → Daily → Point → your coordinates → CSV.

---

### 📋 How Many Years Are Needed?

| Years | What the tool can do |
|---|---|
| 1 | NDVI chart and season dates only |
| 2 | Exploratory models |
| 3–4 | Indicative models |
| 5–9 | Reliable predictions |
| 10+ | Best results |
        """)


if __name__ == "__main__":
    for k in ['predictor','pheno_df','met_df','train_df','all_params','raw_params',
              'ndvi_df','ndvi_info','met_info','interp_freq','_fp']:
        if k not in st.session_state:
            st.session_state[k] = None
    main()
