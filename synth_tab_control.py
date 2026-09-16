"""
synth_tab.py  —  Class-I Synthetic Time-Series Generation (any variable)
Drop this file next to app.py and:
    from synth_tab import tab_synth, register_synth_callbacks
    ... add `tab_synth` to the dbc.Tabs children list ...
    register_synth_callbacks(app)

WHAT THIS TAB DOES
-------------------
Of the three classes of time-series synthetic-data method —
  (1) mimic the *shape* (classical linear models: ARIMA/SARIMAX …)
  (2) reproduce the same *statistical characteristics* (bootstrap, GARCH,
      copulas, moment-matching …)
  (3) reproduce the same *dynamical potential* (chaos theory, RQA,
      nonlinear state-space reconstruction …)
this tab implements Class (1) for any variable the user picks from a
dropdown: it identifies a seasonal ARIMA (SARIMA) model directly from that
variable's own data (periodogram for the seasonal period, AIC grid-search
for the orders), and uses it to draw new sample paths that share the trend,
the seasonal cycle and the persistence ("shape") of the real series without
being copies of it.

All variables are fit and simulated once, eagerly, at import time — not
lazily on first request. Every variable's shape-validation ensemble is
written into one shared tensor, SYNTHETIC_TENSOR, shaped
    (observations, features, realizations)
where "realizations" indexes independent applications of the generation
method (independent stochastic draws from the same fitted model). Selecting
a variable in the dropdown is then just an index lookup into that tensor —
no refit, no wait — while the (slower) future-projection horizon slider
still simulates on demand from the same cached fitted model.
"""

from pathlib import Path
from datetime import datetime
import itertools
import threading
import time
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from cache_utils import make_signature, load_cache, save_cache

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS  (kept in sync with app.py — do not edit independently)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#080b12"
SURFACE  = "#0e1219"
CARD2    = "#181f30"
BORDER   = "rgba(255,255,255,0.07)"
GRID     = "rgba(255,255,255,0.05)"
ZERO_L   = "rgba(255,255,255,0.13)"
TEXT     = "#dde3ee"
MUTED    = "#5a6680"
MUTED2   = "#8899bb"
ACCENT   = "#4f8ef7"
ACCENT2  = "#7eb3ff"
GREEN    = "#2dd4a0"
RED      = "#f4736e"
AMBER    = "#f5a623"
SYN_C    = "#7eb3ff"
FONT_MONO = "'IBM Plex Mono','Fira Code',monospace"

CSV_PATH = Path("spg.csv")

# ─────────────────────────────────────────────────────────────────────────────
#  VARIABLE CATALOGUE  (kept in sync with app.py — do not edit independently)
# ─────────────────────────────────────────────────────────────────────────────
COLS = [
    "temperature_2_m_above_gnd", "relative_humidity_2_m_above_gnd",
    "mean_sea_level_pressure_MSL", "total_precipitation_sfc",
    "snowfall_amount_sfc", "total_cloud_cover_sfc",
    "high_cloud_cover_high_cld_lay", "medium_cloud_cover_mid_cld_lay",
    "low_cloud_cover_low_cld_lay", "shortwave_radiation_backwards_sfc",
    "wind_speed_10_m_above_gnd", "wind_direction_10_m_above_gnd",
    "wind_speed_80_m_above_gnd", "wind_direction_80_m_above_gnd",
    "wind_speed_900_mb", "wind_direction_900_mb",
    "wind_gust_10_m_above_gnd", "angle_of_incidence", "zenith", "azimuth",
]

LABELS = {
    "temperature_2_m_above_gnd":        "Temperature (2 m)",
    "relative_humidity_2_m_above_gnd":  "Relative Humidity (2 m)",
    "mean_sea_level_pressure_MSL":      "Sea-level Pressure",
    "total_precipitation_sfc":          "Total Precipitation",
    "snowfall_amount_sfc":              "Snowfall",
    "total_cloud_cover_sfc":            "Cloud Cover (total)",
    "high_cloud_cover_high_cld_lay":    "Cloud Cover (high)",
    "medium_cloud_cover_mid_cld_lay":   "Cloud Cover (medium)",
    "low_cloud_cover_low_cld_lay":      "Cloud Cover (low)",
    "shortwave_radiation_backwards_sfc":"Shortwave Radiation",
    "wind_speed_10_m_above_gnd":        "Wind Speed (10 m)",
    "wind_direction_10_m_above_gnd":    "Wind Direction (10 m)",
    "wind_speed_80_m_above_gnd":        "Wind Speed (80 m)",
    "wind_direction_80_m_above_gnd":    "Wind Direction (80 m)",
    "wind_speed_900_mb":                "Wind Speed (900 mb)",
    "wind_direction_900_mb":            "Wind Direction (900 mb)",
    "wind_gust_10_m_above_gnd":         "Wind Gust (10 m)",
    "angle_of_incidence":               "Angle of Incidence",
    "zenith":                           "Zenith Angle",
    "azimuth":                          "Azimuth",
}

# hex colours per variable (subset of app.py's GROUPS colour assignment)
COLORS = {
    "temperature_2_m_above_gnd":        "#f07040",
    "relative_humidity_2_m_above_gnd":  "#34c9a0",
    "mean_sea_level_pressure_MSL":      "#a78bfa",
    "total_precipitation_sfc":          "#60a5fa",
    "snowfall_amount_sfc":              "#60a5fa",
    "total_cloud_cover_sfc":            "#94a3b8",
    "high_cloud_cover_high_cld_lay":    "#94a3b8",
    "medium_cloud_cover_mid_cld_lay":   "#94a3b8",
    "low_cloud_cover_low_cld_lay":      "#94a3b8",
    "shortwave_radiation_backwards_sfc":"#fbbf24",
    "wind_speed_10_m_above_gnd":        "#34d399",
    "wind_direction_10_m_above_gnd":    "#34d399",
    "wind_speed_80_m_above_gnd":        "#34d399",
    "wind_direction_80_m_above_gnd":    "#34d399",
    "wind_speed_900_mb":                "#34d399",
    "wind_direction_900_mb":            "#34d399",
    "wind_gust_10_m_above_gnd":         "#34d399",
    "angle_of_incidence":               "#fbbf24",
    "zenith":                           "#fbbf24",
    "azimuth":                          "#fbbf24",
}

UNITS = {
    "temperature_2_m_above_gnd":        "°C",
    "relative_humidity_2_m_above_gnd":  "%",
    "mean_sea_level_pressure_MSL":      "hPa",
    "total_precipitation_sfc":          "mm",
    "snowfall_amount_sfc":              "cm",
    "total_cloud_cover_sfc":            "%",
    "high_cloud_cover_high_cld_lay":    "%",
    "medium_cloud_cover_mid_cld_lay":   "%",
    "low_cloud_cover_low_cld_lay":      "%",
    "shortwave_radiation_backwards_sfc":"W/m²",
    "wind_speed_10_m_above_gnd":        "m/s",
    "wind_direction_10_m_above_gnd":    "°",
    "wind_speed_80_m_above_gnd":        "m/s",
    "wind_direction_80_m_above_gnd":    "°",
    "wind_speed_900_mb":                "m/s",
    "wind_direction_900_mb":            "°",
    "wind_gust_10_m_above_gnd":         "m/s",
    "angle_of_incidence":               "°",
    "zenith":                           "°",
    "azimuth":                          "°",
}

DEFAULT_COL = "temperature_2_m_above_gnd"

# ─────────────────────────────────────────────────────────────────────────────
#  DATA  (independent load, mirrors app.py's load_data())
# ─────────────────────────────────────────────────────────────────────────────
def _load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    end_date = datetime.today().replace(minute=0, second=0, microsecond=0)
    df["date"] = pd.date_range(end=end_date, periods=len(df), freq="h")
    return df.set_index("date")

_RAW = _load_df()
AVAILABLE = [c for c in COLS if c in _RAW.columns]
DF = _RAW[AVAILABLE].copy()
VAR_OPTIONS = [{"label": LABELS.get(c, c), "value": c} for c in AVAILABLE]
if DEFAULT_COL not in AVAILABLE:
    DEFAULT_COL = AVAILABLE[0]

# ─────────────────────────────────────────────────────────────────────────────
#  SEASONAL-PERIOD DETECTION  (periodogram)
#  Rather than assume a textbook 24-sample day, we let the data say what its
#  own cycle length is: the dominant frequency of the (mean-removed) series,
#  restricted to a physically sensible band, sets the SARIMA seasonal period.
#  The band is capped at 30 samples — wide enough to find every genuine
#  sub-daily cycle in this dataset (all land between ~13 and ~25 samples),
#  but tight enough that noisy/circular variables (wind direction, cloud
#  cover) can't lock onto a spurious long "cycle" and blow up the SARIMAX
#  state dimension (cost scales with the seasonal period).
# ─────────────────────────────────────────────────────────────────────────────
def _periodogram(y, min_period=4, max_period=30):
    x = y - y.mean()
    n = len(x)
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    band = [(f, p) for f, p in zip(freqs, power) if f > 0 and min_period <= 1.0 / f <= max_period]
    periods = np.array([1.0 / f for f, _ in band])
    powers  = np.array([p for _, p in band])
    order   = np.argsort(periods)
    return periods[order], powers[order]


def _squeeze(sim_out):
    arr = np.asarray(sim_out)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    elif arr.ndim == 1:
        arr = arr[:, None]
    return arr


# ─────────────────────────────────────────────────────────────────────────────
#  PER-VARIABLE MODEL IDENTIFICATION
#  1. periodogram -> seasonal period s
#  2. ADF test    -> differencing order d
#  3. bounded AIC grid search over (p,q) x (P,Q) at that (s,d)
#  4. simulate a small fixed ensemble for shape validation (ACF/PACF, dist.)
#  Run once per variable, eagerly, for every column — see the precompute
#  loop below that fills SYNTHETIC_TENSOR.
# ─────────────────────────────────────────────────────────────────────────────
N_VALIDATION = 6   # realizations per variable stored in SYNTHETIC_TENSOR

# Fitting is done on the *standardized* series z = (y - mean)/std.  This is a
# pure numerical-conditioning fix: series like sea-level pressure (values
# ~1035 hPa with small variance) make the Kalman filter's LU decomposition
# blow up ("LU decomposition error") under enforce_stationarity.  Fitting on
# z-scores removes those failures entirely; simulations are rescaled back.
_FIT_KW = dict(trend="c", concentrate_scale=True)


def _try_fit(z, order, seasonal):
    """Fit one candidate, retrying without the stationarity/invertibility
    constraints before giving up. Returns (result, note) — result is None on
    total failure, note says which relaxation (if any) was needed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for enforce, note in ((True, ""), (False, "relaxed")):
            try:
                mod = SARIMAX(z, order=order, seasonal_order=seasonal,
                              enforce_stationarity=enforce,
                              enforce_invertibility=enforce, **_FIT_KW)
                return mod.fit(disp=False, maxiter=40, method="lbfgs"), note
            except Exception:
                continue
    return None, "failed"


def _safe_acf(x, nlags):
    """ACF that never raises: constant / non-finite input -> NaN vector."""
    x = np.asarray(x, dtype=float)
    if (not np.all(np.isfinite(x))) or np.std(x) < 1e-12:
        return np.full(nlags + 1, np.nan)
    try:
        return acf(x, nlags=nlags, fft=True)
    except Exception:
        return np.full(nlags + 1, np.nan)


def _safe_pacf(x, nlags):
    """PACF that never raises (singular-matrix on degenerate sims, etc.)."""
    x = np.asarray(x, dtype=float)
    if (not np.all(np.isfinite(x))) or np.std(x) < 1e-12:
        return np.full(nlags + 1, np.nan)
    try:
        return np.clip(pacf(x, nlags=nlags), -1.5, 1.5)
    except Exception:
        return np.full(nlags + 1, np.nan)


def _fit_model(col: str) -> dict:
    s_ser = DF[col].dropna()
    y = s_ser.values.astype(float)
    idx = s_ser.index
    n = len(y)

    mu_y = float(np.mean(y))
    sd_y = float(np.std(y))
    if sd_y < 1e-12:
        sd_y = 1.0
    z = (y - mu_y) / sd_y

    periods, powers = _periodogram(y)
    seasonal_period = int(round(periods[np.argmax(powers)])) if len(periods) else 24
    seasonal_period = max(seasonal_period, 3)

    adf_p = float(adfuller(y, autolag="AIC")[1])
    d_order = 0 if adf_p < 0.05 else 1

    # ── Stepwise minimization of AIC & BIC at the periodogram period ─────────
    # Hyndman-Khandakar-style local search over (p, q, P, Q) with s fixed to
    # the dominant periodogram cycle (never overridden) and d fixed by ADF.
    # Every fitted candidate records AIC, AICc and BIC.  Selection rule:
    #   1. minimize AICc (finite-sample-corrected AIC) via stepwise descent;
    #   2. if the BIC-minimizing candidate is statistically indistinguishable
    #      from the AICc winner (ΔAICc ≤ 2), prefer it — the parsimony
    #      consensus that minimizes both criteria at once.
    MAX_P, MAX_Q, MAX_SP, MAX_SQ = 3, 2, 1, 1
    if seasonal_period >= 20:      # long-period fits are much slower (state dim ∝ s)
        MAX_P = 2
    SEARCH_BUDGET_S = 240.0        # per-variable wall-clock cap; keeps best-so-far
    _t_search = time.time()
    _tried: dict = {}

    def _prune_results():
        """Keep fitted-results objects only for the current AICc- and
        BIC-winners; free every other candidate's memory immediately."""
        fitted_now = [r for r in _tried.values() if r is not None]
        if not fitted_now:
            return
        keep = {id(min(fitted_now, key=lambda r: r["aicc"])),
                id(min(fitted_now, key=lambda r: r["bic"]))}
        for r in fitted_now:
            if id(r) not in keep:
                r["res"] = None

    def _fit_key(k):
        if k in _tried:
            return _tried[k]
        p, q, P, Q = k
        res, note = _try_fit(z, (p, d_order, q), (P, 0, Q, seasonal_period))
        if res is None:
            rec = None
        else:
            rec = dict(order=(p, d_order, q), seasonal=(P, 0, Q, seasonal_period),
                       aic=float(res.aic), bic=float(res.bic),
                       aicc=float(getattr(res, "aicc", res.aic)),
                       note=note, res=res)
        _tried[k] = rec
        _prune_results()
        return rec

    def _budget_left():
        return (time.time() - _t_search) < SEARCH_BUDGET_S

    seeds = [(1, 0, 0, 0), (1, 0, 1, 1), (2, 1, 1, 1), (0, 1, 0, 1)]
    best, best_k = None, None
    for kk in seeds:
        r = _fit_key(kk)
        if r and (best is None or r["aicc"] < best["aicc"]):
            best, best_k = r, kk
        if not _budget_left():
            break

    if best is not None:
        improved = True
        while improved and _budget_left():   # descend until no neighbour improves AICc
            improved = False
            p, q, P, Q = best_k
            for k in {(p - 1, q, P, Q), (p + 1, q, P, Q),
                      (p, q - 1, P, Q), (p, q + 1, P, Q),
                      (p, q, P - 1, Q), (p, q, P + 1, Q),
                      (p, q, P, Q - 1), (p, q, P, Q + 1)}:
                if not (0 <= k[0] <= MAX_P and 0 <= k[1] <= MAX_Q and
                        0 <= k[2] <= MAX_SP and 0 <= k[3] <= MAX_SQ):
                    continue
                if k == (0, 0, 0, 0):
                    continue
                r = _fit_key(k)
                if r and r["aicc"] < best["aicc"] - 1e-6:
                    best, best_k, improved = r, k, True
                if not _budget_left():
                    break

    fitted = [r for r in _tried.values() if r is not None]
    n_skipped = sum(1 for r in _tried.values() if r is None)

    if not fitted:
        # Guaranteed-fittable fallbacks — a plain AR(1), then white noise.
        for order in ((1, d_order, 0), (0, d_order, 0)):
            res, note = _try_fit(z, order, (0, 0, 0, seasonal_period))
            if res is not None:
                fitted = [dict(order=order, seasonal=(0, 0, 0, seasonal_period),
                               aic=float(res.aic), bic=float(res.bic),
                               aicc=float(getattr(res, "aicc", res.aic)),
                               note=note or "fallback", res=res)]
                best = fitted[0]
                break

    # BIC-parsimony consensus: adopt the BIC winner when its AICc is within 2.
    bic_best = min(fitted, key=lambda r: r["bic"])
    selected = best
    if bic_best is not best and (bic_best["aicc"] - best["aicc"]) <= 2.0:
        selected = bic_best

    res = selected.get("res")
    if res is None:                          # pruned in a tie — refit the winner once
        res, _ = _try_fit(z, selected["order"], selected["seasonal"])
    # keep the fitted-results object only for the selected model (memory)
    order_search = [{kk: r[kk] for kk in ("order", "seasonal", "aic", "bic", "aicc", "note")}
                    for r in fitted]
    aic_winner = min(fitted, key=lambda r: r["aicc"])
    bic_winner = bic_best
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_sims = _squeeze(res.simulate(nsimulations=n, repetitions=N_VALIDATION,
                                       anchor=0, random_state=42))
    z_sims = np.nan_to_num(z_sims, nan=0.0, posinf=0.0, neginf=0.0)
    valid_sims = z_sims * sd_y + mu_y            # back to physical units
    nlags_view = min(48, n // 4)
    acf_real  = _safe_acf(y, nlags_view)
    pacf_real = _safe_pacf(y, nlags_view)
    acf_sim   = np.column_stack([_safe_acf(valid_sims[:, k], nlags_view) for k in range(N_VALIDATION)])
    pacf_sim  = np.column_stack([_safe_pacf(valid_sims[:, k], nlags_view) for k in range(N_VALIDATION)])

    sim_pool = valid_sims.ravel()
    moments_real = dict(mean=float(np.mean(y)), std=float(np.std(y, ddof=1)),
                        skew=float(sp_stats.skew(y)), kurt=float(sp_stats.kurtosis(y)))
    moments_sim  = dict(mean=float(np.mean(sim_pool)), std=float(np.std(sim_pool, ddof=1)),
                        skew=float(sp_stats.skew(sim_pool)), kurt=float(sp_stats.kurtosis(sim_pool)))

    return dict(
        col=col, y=y, idx=idx, n=n, mu_y=mu_y, sd_y=sd_y,
        periods=periods, powers=powers, seasonal_period=seasonal_period,
        adf_p=adf_p, d_order=d_order, n_skipped=n_skipped,
        order_search=order_search,
        best_order=selected["order"], best_seasonal=selected["seasonal"],
        best_aic=selected["aic"], best_bic=selected["bic"], best_aicc=selected["aicc"],
        aic_winner=(aic_winner["order"], aic_winner["seasonal"]),
        bic_winner=(bic_winner["order"], bic_winner["seasonal"]),
        res=res,
        valid_sims=valid_sims, nlags_view=nlags_view,
        acf_real=acf_real, pacf_real=pacf_real, acf_sim=acf_sim, pacf_sim=pacf_sim,
        moments_real=moments_real, moments_sim=moments_sim,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  EAGER PRECOMPUTE  —  fit every variable once and stack its shape-validation
#  ensemble into one shared tensor:
#
#      SYNTHETIC_TENSOR.shape == (observations, features, realizations)
#
#  "features" indexes variables (order given by FEATURE_INDEX); "realizations"
#  indexes independent applications of the generation method — i.e. the
#  N_VALIDATION independent stochastic draws simulated from each variable's
#  fitted SARIMA model. Once this loop finishes, selecting a variable in the
#  dropdown is a pure index lookup: SYNTHETIC_TENSOR[:, FEATURE_INDEX[col], :].
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_INDEX = {c: i for i, c in enumerate(AVAILABLE)}
N_OBS = len(DF)
SYNTHETIC_TENSOR = np.full((N_OBS, len(AVAILABLE), N_VALIDATION), np.nan, dtype=float)

_MODEL_CACHE: dict = {}
_STATUS_LOCK = threading.Lock()
_STATUS = dict(current=None, done=0, total=len(AVAILABLE), finished=False, errors={})

# This is by far the most expensive block in the whole app — a bounded AICc/
# BIC stepwise SARIMA search (up to several minutes per variable) run for
# every one of AVAILABLE. Cached to disk via cache_utils so a later run with
# an unchanged spg.csv and N_VALIDATION loads every fitted model and the
# whole SYNTHETIC_TENSOR straight from .cache/ instead of refitting.
_MODEL_SIG = make_signature(
    CSV_PATH, version="synth-sarima-v1",
    n_validation=N_VALIDATION, available=AVAILABLE,
)


def _precompute_worker():
    """Fit every variable in a daemon thread so the Dash server starts serving
    immediately.  DEFAULT_COL is fitted first so the tab has content within
    seconds; the rest fill in progressively and the UI polls the status."""
    t_start = time.time()
    queue = [DEFAULT_COL] + [c for c in AVAILABLE if c != DEFAULT_COL]
    for i, col in enumerate(queue, start=1):
        with _STATUS_LOCK:
            _STATUS["current"] = col
        t0 = time.time()
        try:
            m = _fit_model(col)
            if len(m["y"]) == N_OBS:
                SYNTHETIC_TENSOR[:, FEATURE_INDEX[col], :] = m["valid_sims"]
                m["valid_sims"] = SYNTHETIC_TENSOR[:, FEATURE_INDEX[col], :]  # tensor view
            # (length mismatch: keep m["valid_sims"] as its own array — never crash)
            _MODEL_CACHE[col] = m
            skipped = f"  ({m['n_skipped']} candidates skipped)" if m["n_skipped"] else ""
            print(f"[synth_tab]   ({i}/{len(queue)}) {col}: "
                  f"SARIMA{m['best_order']}x{m['best_seasonal']}  "
                  f"AIC={m['best_aic']:.1f}  BIC={m['best_bic']:.1f}  "
                  f"[{time.time() - t0:.1f}s]{skipped}")
        except Exception as e:                       # never kill the thread
            with _STATUS_LOCK:
                _STATUS["errors"][col] = f"{type(e).__name__}: {e}"
            print(f"[synth_tab]   ({i}/{len(queue)}) {col}: could not fit — {type(e).__name__}: {e}")
        with _STATUS_LOCK:
            _STATUS["done"] = i
    print(f"[synth_tab] Ready OK — tensor built in {time.time() - t_start:.1f}s total")
    # Save to disk *before* flipping `finished`, so a consumer that reacts to
    # finished=True (or a process that exits right after) can never observe
    # a "done" state whose cache write is still in flight — the SARIMA
    # payload is large enough (each fitted model carries its full Kalman
    # filter state) that the write itself takes real, observable time.
    save_cache("synth_sarima_models", _MODEL_SIG,
               dict(tensor=SYNTHETIC_TENSOR, models=dict(_MODEL_CACHE)))
    with _STATUS_LOCK:
        _STATUS["finished"] = True
        _STATUS["current"] = None


_cached_models = load_cache("synth_sarima_models", _MODEL_SIG)
if _cached_models is not None:
    SYNTHETIC_TENSOR[...] = _cached_models["tensor"]
    _MODEL_CACHE.update(_cached_models["models"])
    with _STATUS_LOCK:
        _STATUS.update(done=len(AVAILABLE), total=len(AVAILABLE), finished=True, current=None)
    print(f"[synth_tab] Loaded cached SARIMA fits + synthetic tensor for {len(_MODEL_CACHE)} "
          "variables from .cache/ ✔  (delete .cache/ to force a full recompute)")
else:
    print(f"[synth_tab] Starting background SARIMA precompute for {len(AVAILABLE)} variables "
          f"(observations={N_OBS}, realizations={N_VALIDATION}) — the app serves immediately; "
          "the Synthetic Data tab fills in as models finish.")
    _WORKER = threading.Thread(target=_precompute_worker, daemon=True, name="synth-precompute")
    _WORKER.start()


def get_model(col: str):
    """Look up the pre-fit model + synthetic ensemble for `col`.
    Returns None while the background precompute has not reached it yet."""
    return _MODEL_CACHE.get(col)


def precompute_status() -> dict:
    with _STATUS_LOCK:
        return dict(_STATUS)

# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATION HELPERS  (respond to the dashboard controls)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_history(col: str, n_sims: int) -> np.ndarray:
    """The first n_sims pre-computed realisations for `col`, read straight out
    of SYNTHETIC_TENSOR — no simulation happens here, just an index lookup."""
    idx = FEATURE_INDEX.get(col, FEATURE_INDEX[DEFAULT_COL])
    n_sims = max(1, min(n_sims, N_VALIDATION))
    return SYNTHETIC_TENSOR[:, idx, :n_sims]


def simulate_future(m: dict, horizon: int, n_sims: int, seed: int) -> np.ndarray:
    """n_sims realisations continuing `horizon` steps past the last observation,
    conditioned on the model's filtered state at that observation."""
    if horizon <= 0:
        return np.empty((0, n_sims))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_sims = _squeeze(m["res"].simulate(nsimulations=horizon, repetitions=n_sims,
                                            anchor="end", random_state=seed))
    z_sims = np.nan_to_num(z_sims, nan=0.0, posinf=0.0, neginf=0.0)
    return z_sims * m["sd_y"] + m["mu_y"]        # model was fit on z-scores


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY FIGURE BUILDERS  (all take the model dict `m` returned by get_model)
# ─────────────────────────────────────────────────────────────────────────────
_AX = dict(showgrid=True, gridcolor=GRID, gridwidth=1,
           zeroline=True, zerolinecolor=ZERO_L, zerolinewidth=1,
           linecolor=BORDER, linewidth=1, showline=True, mirror=False,
           tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
           title_font=dict(size=10, color=MUTED2, family=FONT_MONO))

def _base_layout(**kw):
    base = dict(paper_bgcolor=BG, plot_bgcolor=SURFACE,
               font=dict(family=FONT_MONO, color=TEXT, size=11),
               hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(size=12, color=TEXT)),
               margin=dict(t=36, b=40, l=56, r=16))
    base.update(kw)
    return base

def _unit_suffix(col):
    u = UNITS.get(col, "")
    return f" {u}" if u else ""


def fig_periodogram(m: dict) -> go.Figure:
    s = m["seasonal_period"]
    periods, powers = m["periods"], m["powers"]
    fig = go.Figure(go.Scatter(
        x=periods, y=powers, mode="lines",
        line=dict(color=MUTED2, width=1.4), fill="tozeroy",
        fillcolor="rgba(136,153,187,0.08)",
        hovertemplate="period: %{x:.1f}<br>power: %{y:.3g}<extra></extra>",
    ))
    if len(powers):
        peak_pow = powers[np.argmax(powers)]
        fig.add_trace(go.Scatter(
            x=[s], y=[peak_pow], mode="markers+text",
            marker=dict(size=11, color=AMBER, line=dict(color=BG, width=1.5)),
            text=[f"  s = {s}"], textposition="top right",
            textfont=dict(size=11, color=AMBER, family=FONT_MONO),
            hovertemplate=f"dominant period ≈ {s} samples<extra></extra>",
        ))
    fig.update_xaxes(**_AX, title=dict(text="period (samples)", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="spectral power", standoff=4))
    fig.update_layout(**_base_layout(height=260, showlegend=False,
        title=dict(text="Periodogram · dominant cycle sets the SARIMA seasonal period",
                   font=dict(size=12, color=TEXT), x=0.01)))
    return fig


def fig_order_search(m: dict) -> go.Figure:
    rows = sorted(m["order_search"], key=lambda r: r["aicc"])
    sel = (m["best_order"], m["best_seasonal"])
    aic_w, bic_w = m["aic_winner"], m["bic_winner"]

    def _lbl(r):
        o, s = r["order"], r["seasonal"]
        tag = ""
        if (o, s) == sel:
            tag = " ✔"
        marks = []
        if (o, s) == aic_w:
            marks.append("AIC★")
        if (o, s) == bic_w:
            marks.append("BIC★")
        if marks:
            tag += " " + "/".join(marks)
        return f"({o[0]},{o[1]},{o[2]})x({s[0]},{s[1]},{s[2]},{s[3]}){tag}"

    labels = [_lbl(r) for r in rows]
    aics   = [r["aic"] for r in rows]
    bics   = [r["bic"] for r in rows]
    colors = [GREEN if (r["order"], r["seasonal"]) == sel else ACCENT2 for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=aics, name="AIC",
        marker_color=colors, marker_line=dict(color=BORDER, width=1),
        hovertemplate="SARIMA%{x}<br>AIC: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=bics, name="BIC", mode="markers",
        marker=dict(color=AMBER, size=9, symbol="diamond",
                    line=dict(color=BG, width=1)),
        hovertemplate="SARIMA%{x}<br>BIC: %{y:.1f}<extra></extra>",
    ))
    fig.update_xaxes(**_AX, tickangle=45, title=None)
    fig.update_yaxes(**_AX, title=dict(text="information criterion (lower is better)", standoff=4))
    fig.update_layout(**_base_layout(
        height=320, showlegend=True,
        legend=dict(font=dict(size=10, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=1.14, x=0.55),
        title=dict(text=f"Stepwise AIC/BIC search — {len(rows)} candidates fitted, "
                        "green = selected (AICc winner, or BIC winner when ΔAICc ≤ 2)",
                   font=dict(size=12, color=TEXT), x=0.01)))
    return fig


def _downsample(arr, max_pts=1500):
    step = max(1, len(arr) // max_pts)
    return arr[::step]


def fig_overlay(m: dict, n_sims: int, horizon: int, seed: int):
    col = m["col"]
    label = LABELS.get(col, col)
    unit = _unit_suffix(col)
    color = COLORS.get(col, ACCENT2)

    fig = go.Figure()
    x_hist = _downsample(m["idx"])
    y_hist = _downsample(m["y"])
    fig.add_trace(go.Scatter(
        x=x_hist, y=y_hist, mode="lines", name="Real (observed)",
        line=dict(color=color, width=1.6),
        hovertemplate=f"Real: %{{y:.2f}}{unit}<extra></extra>",
    ))

    if horizon <= 0:
        n_show = max(1, min(int(n_sims), N_VALIDATION))
        sims = m["valid_sims"][:, :n_show]      # tensor view stored on the model
        for k in range(sims.shape[1]):
            y_sim = _downsample(sims[:, k])
            fig.add_trace(go.Scatter(
                x=x_hist, y=y_sim, mode="lines",
                name=f"Synthetic realisation {k+1}",
                line=dict(color=SYN_C, width=1, dash="dot"), opacity=0.55,
                hovertemplate=f"Synthetic #{k+1}: %{{y:.2f}}{unit}<extra></extra>",
            ))
        mode_note = ("Alternate-history mode: each dotted line is one of the pre-computed "
                     "realisations for this variable, read directly out of SYNTHETIC_TENSOR — "
                     "independent draws from the same fitted SARIMA model over the same historical "
                     "span, not a copy of the real data, but the same generative shape.")
    else:
        sims = simulate_future(m, horizon, n_sims, seed)
        future_idx = pd.date_range(start=m["idx"][-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
        for k in range(sims.shape[1]):
            x_fut = [m["idx"][-1]] + list(future_idx)
            y_fut = [m["y"][-1]] + list(sims[:, k])
            fig.add_trace(go.Scatter(
                x=x_fut, y=y_fut, mode="lines",
                name=f"Synthetic continuation {k+1}",
                line=dict(color=SYN_C, width=1.4, dash="dot"), opacity=0.75,
                hovertemplate=f"Synthetic #{k+1}: %{{y:.2f}}{unit}<extra></extra>",
            ))
        mode_note = (f"Future-projection mode: continuing {horizon}h past the last observation, "
                     "anchored on the model's filtered state — a plausible, not certain, future.")

    fig.update_xaxes(**_AX, title=None)
    fig.update_yaxes(**_AX, title=dict(text=f"{label}{unit}", standoff=4))
    fig.update_layout(**_base_layout(
        height=380, hovermode="x unified", showlegend=True,
        legend=dict(font=dict(size=9, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18, x=0),
    ))
    return fig, mode_note


def fig_acf_compare(m: dict) -> go.Figure:
    nlags = m["nlags_view"]
    lags = list(range(nlags + 1))
    acf_sim, pacf_sim = m["acf_sim"], m["pacf_sim"]
    color = COLORS.get(m["col"], ACCENT2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # all-NaN lag rows are fine
        sim_acf_lo  = np.nan_to_num(np.nanmin(acf_sim, axis=1))
        sim_acf_hi  = np.nan_to_num(np.nanmax(acf_sim, axis=1))
        sim_pacf_lo = np.nan_to_num(np.nanmin(pacf_sim, axis=1))
        sim_pacf_hi = np.nan_to_num(np.nanmax(pacf_sim, axis=1))

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.07, subplot_titles=["ACF", "PACF"])
    for col_i, (real, lo, hi) in enumerate(
        [(m["acf_real"], sim_acf_lo, sim_acf_hi), (m["pacf_real"], sim_pacf_lo, sim_pacf_hi)], start=1
    ):
        fig.add_trace(go.Scatter(x=lags + lags[::-1], y=list(hi) + list(lo)[::-1],
                                 fill="toself", fillcolor="rgba(126,179,255,0.14)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=col_i)
        fig.add_trace(go.Scatter(x=lags, y=real, mode="lines+markers", name="Real",
                                 line=dict(color=color, width=2), marker=dict(size=4),
                                 showlegend=(col_i == 1)), row=1, col=col_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_mean = np.nanmean(acf_sim if col_i == 1 else pacf_sim, axis=1)
        fig.add_trace(go.Scatter(x=lags, y=sim_mean, mode="lines", name=f"Synthetic (mean of {N_VALIDATION})",
                                 line=dict(color=SYN_C, width=2, dash="dash"),
                                 showlegend=(col_i == 1)), row=1, col=col_i)

    fig.update_xaxes(**_AX, title=dict(text="lag", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="correlation", standoff=4), range=[-0.3, 1.05])
    fig.update_layout(**_base_layout(
        height=320, hovermode="closest",
        legend=dict(font=dict(size=10, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=1.16, x=0.3),
    ))
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    return fig


def fig_dist_compare(m: dict) -> go.Figure:
    col = m["col"]
    label = LABELS.get(col, col)
    unit = _unit_suffix(col)
    color = COLORS.get(col, ACCENT2)
    sim_pool = m["valid_sims"].ravel()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=m["y"], name="Real", histnorm="probability density",
                               marker_color=color, opacity=0.55, nbinsx=40))
    fig.add_trace(go.Histogram(x=sim_pool, name="Synthetic (pooled)", histnorm="probability density",
                               marker_color=SYN_C, opacity=0.45, nbinsx=40))
    fig.update_xaxes(**_AX, title=dict(text=f"{label}{unit}", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="density", standoff=4))
    fig.update_layout(**_base_layout(
        height=300, barmode="overlay",
        legend=dict(font=dict(size=10, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0.32),
    ))
    return fig


def moment_banner(m: dict):
    col = m["col"]
    unit = UNITS.get(col, "")
    unit_sfx = f" ({unit})" if unit else ""
    def card(v, l, c):
        return html.Div([
            html.Div(v, className="sbn-val", style={"color": c}),
            html.Div(l, className="sbn-lbl"),
        ], className="sbn")
    r, s = m["moments_real"], m["moments_sim"]
    return html.Div([
        card(f"{r['mean']:.2f}",  f"Real mean{unit_sfx}",      COLORS.get(col, ACCENT2)),
        card(f"{s['mean']:.2f}",  f"Synthetic mean{unit_sfx}",  SYN_C),
        card(f"{r['std']:.2f}",   "Real σ",                    COLORS.get(col, ACCENT2)),
        card(f"{s['std']:.2f}",   "Synthetic σ",                SYN_C),
        card(f"{r['skew']:+.2f}", "Real skew",                  COLORS.get(col, ACCENT2)),
        card(f"{s['skew']:+.2f}", "Synthetic skew",             SYN_C),
        card(f"{r['kurt']:+.2f}", "Real excess kurtosis",       COLORS.get(col, ACCENT2)),
        card(f"{s['kurt']:+.2f}", "Synthetic excess kurt.",     SYN_C),
    ], className="summary-banner")


def banner_children(m: dict):
    order_str = (f"({m['best_order'][0]},{m['best_order'][1]},{m['best_order'][2]})×"
                f"({m['best_seasonal'][0]},{m['best_seasonal'][1]},{m['best_seasonal'][2]},{m['best_seasonal'][3]})")
    return _banner(
        (m["seasonal_period"], "Periodogram seasonal period s", AMBER),
        (order_str, "Selected SARIMA order", GREEN,
         {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
        (f"{m['best_aic']:.1f}", f"AIC (best of {len(m['order_search'])})", ACCENT2),
        (f"{m['best_bic']:.1f}", "BIC of selected", AMBER),
        (f"{m['n']:,}", "Observations", MUTED2),
        ("Stationary" if m["d_order"] == 0 else "Non-stationary", f"ADF p={m['adf_p']:.4f}",
         GREEN if m["d_order"] == 0 else RED),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL LAYOUT HELPERS  (mirrors app.py's, reuses the same global CSS classes)
# ─────────────────────────────────────────────────────────────────────────────
def _section_hdr(text):
    return html.Div(text, className="section-hdr")

def _section_desc(text):
    return html.Div(text, className="section-desc")

def _plot_card(children, label=None):
    inner = []
    if label:
        inner.append(html.Div(label, className="plot-label"))
    inner.extend(children if isinstance(children, list) else [children])
    return html.Div(inner, className="plot-card")

def _banner(*items):
    """items: (value, label, color) or (value, label, color, extra_value_style)"""
    cards = []
    for it in items:
        if len(it) == 4:
            v, l, c, extra = it
        else:
            v, l, c = it
            extra = {}
        style = {"color": c}
        style.update(extra)
        cards.append(html.Div([
            html.Div(str(v), className="sbn-val", style=style),
            html.Div(l, className="sbn-lbl"),
        ], className="sbn"))
    return html.Div(cards, className="summary-banner")

def _theory_card(title, tag, color, paragraphs, math_eq=None, h0=None, h1=None):
    children = [html.H3([html.Span(title, style={"color": color}), html.Span(tag, className="tc-tag")])]
    for p in paragraphs:
        children.append(html.P(p) if isinstance(p, str) else html.Div(p, style={"marginBottom": "8px"}))
    if math_eq:
        children.append(html.Code(math_eq, className="math"))
    if h0:
        children.append(html.Div([html.Span("H₀: ", style={"fontWeight": "700"}), h0], className="h0-box"))
    if h1:
        children.append(html.Div([html.Span("H₁: ", style={"fontWeight": "700"}), h1], className="h1-box"))
    return html.Div(children, className="theory-card")

def _callout(kind, title, intro, items, ordered=True):
    icon = "📋" if kind == "tip" else "⚠"
    box_cls = "callout callout-tip" if kind == "tip" else "callout callout-note"
    head = html.Div([html.Span(icon, className="callout-icon"), html.Span(title, className="callout-title")],
                    className="callout-head")
    body_children = []
    if intro:
        body_children.append(html.P(intro, className="callout-intro"))
    lis = []
    for it in items:
        if isinstance(it, (tuple, list)):
            lead, rest = it
            lis.append(html.Li([html.B(lead, className="callout-lead"), rest], className="callout-li"))
        else:
            lis.append(html.Li(it, className="callout-li"))
    list_tag = html.Ol if ordered else html.Ul
    body_children.append(list_tag(lis, className="callout-list"))
    return html.Div([head, html.Div(body_children, className="callout-body")], className=box_cls)


def identification_card(m: dict):
    col = m["col"]
    label = LABELS.get(col, col)
    return _theory_card(
        "Identifying the model for THIS series", "IDENTIFICATION", GREEN,
        [
            [
                f"Rather than assume a textbook 24-sample day, the seasonal period s is read directly "
                f"off {label}'s own data: a periodogram of its mean-removed values finds its dominant "
                f"cycle at s ≈ {m['seasonal_period']} samples (see the periodogram chart above), within "
                "a 4–30 sample search band chosen to cover every genuine sub-daily cycle in this dataset "
                "while staying computationally safe (a spurious long 'cycle' would blow up the model's "
                "seasonal state dimension).",
                html.Br(), html.Br(),
                f"An Augmented Dickey-Fuller test on the raw series returns p = {m['adf_p']:.4f} "
                f"({'< 0.05 → stationary, so d = 0' if m['d_order'] == 0 else '≥ 0.05 → unit root, so d = 1'}).",
                html.Br(), html.Br(),
                f"With s and d fixed, a stepwise Hyndman-Khandakar-style search explores the "
                f"(p, q) × (P, Q) neighbourhood, fitting {len(m['order_search'])} candidate models by "
                "maximum likelihood. Each candidate is scored by AICc (finite-sample-corrected AIC) "
                "and BIC; the search descends until no neighbouring order improves AICc, then adopts "
                "the BIC minimiser whenever it is statistically indistinguishable from the AICc winner "
                "(ΔAICc ≤ 2) — jointly minimising both criteria. The winner here is "
                f"SARIMA{m['best_order']}×{m['best_seasonal']} with AIC = {m['best_aic']:.1f} and "
                f"BIC = {m['best_bic']:.1f}.",
            ],
            html.Div([
                html.Span("Why stepwise instead of exhaustive? ", style={"fontWeight": "700", "color": GREEN}),
                "An exhaustive search over every (p,d,q)×(P,D,Q,s) combination is both slow and prone to "
                "overfitting spurious lag structure. Bounding p ≤ 3, q ≤ 2, P,Q ∈ {0,1} and descending "
                "stepwise from multiple seeds finds the same AICc/BIC minimum as the full grid in a "
                "fraction of the fits — a deliberate elegance trade-off appropriate for Class-1 "
                "methods, whose whole appeal is transparency. This whole pipeline runs once for every "
                "variable at startup, not when you switch the dropdown above — every variable's fitted "
                "model and synthetic ensemble are already sitting in SYNTHETIC_TENSOR, so switching here "
                "is an instant lookup, not a refit.",
            ], className="warn-box", style={"marginTop": "10px"}),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  THEORY PANEL  (mostly static; the "Identification" card is per-variable)
# ─────────────────────────────────────────────────────────────────────────────
SYNTH_THEORY = html.Div([
    html.Div("📖 Synthetic Time-Series Generation — Theory", className="theory-title"),

    html.P([
        "Synthetic time-series methods fall into three classes of increasing sophistication. ",
        html.B("Class 1"), " (this tab) fits a conventional linear model — ARIMA / SARIMA / SARIMAX — and "
        "uses it to draw new sample paths that reproduce the ", html.I("shape"), " of the original series: "
        "its trend, its seasonal cycle, its short-run persistence. ", html.B("Class 2"), " methods "
        "(bootstrap residual resampling, GARCH, copulas) go further and explicitly match the target's "
        "statistical moments and distribution. ", html.B("Class 3"), " methods (Recurrent Quantitative "
        "Analysis, phase-space reconstruction, chaos-theoretic embeddings — see the state-space discussion "
        "in the Causality tab) attempt to reproduce the underlying dynamical system itself, including its "
        "nonlinear determinism. Complexity — and data requirements — increase in that order.",
    ], className="section-desc", style={"marginBottom": "18px"}),

    html.Div([
        html.Div([
            _theory_card(
                "SARIMA — the Class-1 workhorse", "MODEL", ACCENT2,
                [
                    "A Seasonal ARIMA model, written SARIMA(p,d,q)×(P,D,Q,s), decomposes a series into "
                    "three additive linear mechanisms: short-run autoregression (AR), a moving-average "
                    "of past shocks (MA), and a seasonal copy of both operating at lag s (one full cycle).",
                    "Because every term is linear and every shock is drawn i.i.d. from a single "
                    "distribution, the model is fully described by a finite list of coefficients — "
                    "which is exactly why it is transparent, fast to fit, and easy to simulate from. "
                    "It is also why it is a Class-1 method: it reproduces the autocorrelation shape of "
                    "the data, not necessarily every nuance of its distribution or its nonlinear dynamics.",
                    dcc.Markdown(
                        r"$\Phi_P(L^s)\,\phi_p(L)\,(1-L)^d(1-L^s)^D\,y_t = \Theta_Q(L^s)\,\theta_q(L)\,\varepsilon_t,"
                        r"\quad \varepsilon_t \sim \mathcal{N}(0,\sigma^2)$",
                        mathjax=True,
                    ),
                    html.Div([
                        html.Span("Plain-English version: ", style={"fontWeight": "700", "color": ACCENT2}),
                        "the next value ≈ a mix of the last few observations (AR), a mix of recent "
                        "unexpected shocks (MA), the same pattern from one seasonal cycle ago (seasonal "
                        "AR/MA), plus a fresh random shock. Simulating the model forward means repeatedly "
                        "drawing that fresh shock and rolling the recursion forward.",
                    ], className="warn-box", style={"marginTop": "10px"}),
                ],
            ),
        ], className="theory-row"),
        html.Div(id="synth-id-card", className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px",
                                       "overflow": "hidden", "background": CARD2}),

    html.Div([
        html.Div([
            _theory_card(
                "From fitted model to synthetic series", "SIMULATE", AMBER,
                [
                    "Fitting the model gives coefficients and a residual variance; it does not by "
                    "itself produce new data. Two distinct operations turn a fitted SARIMAX into a "
                    "synthetic series, and this tab uses both:",
                    html.Div([
                        html.B("① Alternate-history mode"), " (horizon slider = 0): the model is run "
                        "forward for the full length of the historical record, starting from its own "
                        "unconditional (long-run) distribution, drawing a brand-new sequence of random "
                        "shocks each time. The result is a fresh realisation of the ", html.I("same"),
                        " stochastic process — same autocorrelation shape, same seasonal bump, same "
                        "typical spread — that has never seen the real data's actual values. This is "
                        "the direct evidence for the Class-1 claim, and is what the ACF/PACF and "
                        "distribution-overlay charts below are validating.",
                    ], style={"marginTop": "6px"}),
                    html.Div([
                        html.B("② Future-projection mode"), " (horizon slider > 0): the model is anchored "
                        "on its filtered state at the very last real observation, then rolled forward for "
                        "the requested number of hours with fresh random shocks. Each realisation is a "
                        "plausible — not certain — continuation of the real series, diverging further "
                        "from any single 'true' future the longer the horizon.",
                    ], style={"marginTop": "6px"}),
                ],
            ),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px",
                                       "overflow": "hidden", "background": CARD2, "marginTop": "14px"}),

    _callout(
        "tip", "How to read this tab — a walkthrough for any level",
        "Work through the page top to bottom; each chart answers one question in the model-building "
        "pipeline:",
        [
            ("Variable dropdown → ", "\"Which series am I synthesising?\" — pick any of the twenty "
             "recorded weather/geometry variables; every one was pre-fit at startup, so the whole "
             "page below updates instantly from SYNTHETIC_TENSOR."),
            ("Periodogram → ", "\"What cycle length does the data actually have?\" — the amber dot marks "
             "the seasonal period s fed into the SARIMA model."),
            ("AIC bar chart → ", "\"Which of the candidate model orders fits best without over-parameterising?\" "
             "— the green bar is the model used everywhere below."),
            ("Overlay chart → ", "\"Do synthetic paths look like plausible traces of this variable?\" — toggle "
             "the horizon slider to switch between shape-validation (dotted lines over history) and future "
             "scenario generation (dotted lines extending past the last real point)."),
            ("ACF/PACF comparison → ", "\"Does the synthetic process have the same memory structure as the "
             "real one?\" — the real curve should sit inside or near the synthetic band at every lag."),
            ("Distribution & moment cards → ", "\"How close are the marginal statistics?\" — expect them "
             "close but not identical: exact moment-matching is what Class-2 methods are for, not Class 1."),
        ],
    ),
    _callout(
        "note", "Limits of Class-1 fidelity",
        "SARIMA is linear and Gaussian-shock by construction, which buys interpretability at a real cost:",
        [
            "It cannot reproduce heavy-tailed extremes (storms, calms, cold snaps) with the correct "
            "frequency — its shocks are Gaussian, real weather extremes usually are not.",
            "It assumes the seasonal/trend structure is fixed for the life of the simulation — a genuine "
            "structural break (climate shift, sensor relocation) will not be captured.",
            "It has no notion of physical constraints or cross-variable feedback (e.g. cloud-cover / "
            "radiation / temperature coupling) — each fit only knows that one variable's own past values.",
            "For near-white-noise or highly intermittent variables (precipitation, snowfall) the "
            "'seasonal cycle' the periodogram finds is weak by nature — the model will correctly settle "
            "on a low-order, mostly-noise process rather than manufacture a cycle that isn't really there.",
            ("Where to go next: ", "Class-2 methods (residual bootstrap, GARCH, copulas) explicitly "
             "correct the marginal distribution and volatility clustering that SARIMA glosses over; "
             "Class-3 methods (chaos-theoretic embeddings, RQA — previewed in the Causality tab's "
             "phase-space discussion) attempt to recover the deterministic dynamical system generating "
             "the weather itself, at the cost of far more data and far less transparency."),
        ],
        ordered=False,
    ),
], className="theory-section")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab_synth = dbc.Tab(
    label="🧬  Synthetic Data",
    tab_id="tab-synth",
    children=html.Div([
        _section_hdr("Class-I Synthetic Data Generation (SARIMA)"),
        _section_desc(
            "Every variable was fit and simulated once at startup — a seasonal ARIMA model identified "
            "directly from its own data (periodogram for the seasonal period, AIC grid search for the "
            "orders), then a small ensemble of synthetic realisations drawn from it and stored in a "
            "shared observations × features × realizations tensor. Pick any variable below to view its "
            "slice of that tensor instantly. Read the theory panel at the bottom for a full walkthrough "
            "at any level of expertise."
        ),

        # Polls the background precompute; disables itself once all fits finish.
        dcc.Interval(id="synth-poll", interval=2000, n_intervals=0),

        html.Div([
            html.Div([
                html.Label("Target variable"),
                dcc.Dropdown(id="synth-var", options=VAR_OPTIONS, value=DEFAULT_COL,
                            clearable=False, style={"minWidth": "280px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),

        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div(id="synth-banner"),

            html.Div([
                html.Div(_plot_card([dcc.Graph(id="synth-periodogram", config={"displayModeBar": False})],
                                    label="Step 1 · seasonal-period detection"),
                         style={"flex": "1", "minWidth": "360px"}),
                html.Div(_plot_card([dcc.Graph(id="synth-order-search", config={"displayModeBar": False})],
                                    label="Step 2 · AIC-optimal order selection"),
                         style={"flex": "1", "minWidth": "360px"}),
            ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start", "flexWrap": "wrap"}),
        ]),

        _section_hdr("Generate Synthetic Realisations"),
        _section_desc(
            "Set the horizon to 0 to view pre-computed alternate realisations straight out of "
            "SYNTHETIC_TENSOR (shape validation); set it above 0 to simulate new synthetic "
            "continuations beyond the last observation on demand (scenario generation)."
        ),
        html.Div([
            html.Div([
                html.Label(f"Number of realisations (of {N_VALIDATION} stored)"),
                dcc.Slider(id="synth-n-sims", min=1, max=N_VALIDATION, step=1, value=3,
                          marks={i: str(i) for i in range(1, N_VALIDATION + 1)},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label("Horizon beyond history (hours)"),
                dcc.Slider(id="synth-horizon", min=0, max=336, step=24, value=0,
                          marks={0: "0 (shape check)", 168: "1 wk", 336: "2 wk"},
                          tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label("Random seed (future-projection only)"),
                dcc.Input(id="synth-seed", type="number", value=42, min=0, step=1,
                         style={"width": "100px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),

        dcc.Loading(type="circle", color=ACCENT, children=[
            _plot_card([dcc.Graph(id="synth-overlay", config={"displaylogo": False})],
                       label="Real series vs synthetic realisations"),
        ]),
        html.Div(id="synth-mode-note", className="insight-bar"),

        _section_hdr("Shape Validation: Autocorrelation & Distribution"),
        html.Div(id="synth-validation-desc", className="section-desc"),
        dcc.Loading(type="circle", color=ACCENT, children=[
            _plot_card([dcc.Graph(id="synth-acf-compare", config={"displayModeBar": False})],
                       label=f"Real ACF/PACF vs synthetic band (min–max across {N_VALIDATION} realisations)"),
            _plot_card([dcc.Graph(id="synth-dist-compare", config={"displayModeBar": False})],
                       label="Marginal distribution · real vs pooled synthetic"),
            html.Div(id="synth-moment-banner"),
        ]),

        SYNTH_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  — register with your app via register_synth_callbacks(app)
# ─────────────────────────────────────────────────────────────────────────────
def _placeholder_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False,
                       font=dict(size=12, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**_base_layout(height=260, showlegend=False))
    return fig


def _progress_banner(col: str, st: dict):
    cur = st.get("current")
    cur_label = LABELS.get(cur, cur) if cur else "—"
    return _banner(
        (f"{st['done']}/{st['total']}", "Models fitted so far", ACCENT2),
        (LABELS.get(col, col), "Waiting for", AMBER,
         {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
        (cur_label, "Currently fitting", MUTED2,
         {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
    )


def register_synth_callbacks(app):

    @app.callback(
        Output("synth-banner", "children"),
        Output("synth-periodogram", "figure"),
        Output("synth-order-search", "figure"),
        Output("synth-acf-compare", "figure"),
        Output("synth-dist-compare", "figure"),
        Output("synth-moment-banner", "children"),
        Output("synth-validation-desc", "children"),
        Output("synth-id-card", "children"),
        Output("synth-overlay", "figure"),
        Output("synth-mode-note", "children"),
        Output("synth-poll", "disabled"),
        Input("synth-var", "value"),
        Input("synth-n-sims", "value"),
        Input("synth-horizon", "value"),
        Input("synth-seed", "value"),
        Input("synth-poll", "n_intervals"),
    )
    def update_synth(col, n_sims, horizon, seed, _tick):
        col = col or DEFAULT_COL
        st = precompute_status()
        m = get_model(col)

        # Background fit for this variable not finished yet: show progress and
        # keep the interval polling. All outputs stay valid — no callback errors.
        if m is None:
            err = st["errors"].get(col)
            if err:
                wait_msg = f"Could not fit {LABELS.get(col, col)} — {err}"
            else:
                wait_msg = (f"Fitting SARIMA models in the background "
                            f"({st['done']}/{st['total']} done) — this variable will "
                            "appear automatically when ready.")
            ph = _placeholder_fig(wait_msg)
            return (_progress_banner(col, st), ph, ph, ph, ph,
                    None, wait_msg, None, _placeholder_fig(wait_msg),
                    wait_msg, st["finished"])

        n_sims = int(n_sims or 1)
        horizon = int(horizon or 0)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = 42

        overlay_fig, mode_note = fig_overlay(m, n_sims, horizon, seed)
        validation_desc = (
            f"Fixed ensemble of {N_VALIDATION} independent unconditional realisations for "
            f"{LABELS.get(m['col'], m['col'])} — precomputed once at startup and read from "
            "SYNTHETIC_TENSOR; these two charts are the direct evidence that the model reproduces "
            "the real series' shape rather than a lucky single draw."
        )
        return (
            banner_children(m),
            fig_periodogram(m),
            fig_order_search(m),
            fig_acf_compare(m),
            fig_dist_compare(m),
            moment_banner(m),
            validation_desc,
            identification_card(m),
            overlay_fig,
            mode_note,
            st["finished"],           # stop polling once every variable is fitted
        )
