"""
synth2_tab.py  —  Class-II Synthetic Time-Series Generation (any variable)
Drop this file next to app.py and:
    from synth2_tab import tab_synth2, register_synth2_callbacks
    ... add `tab_synth2` to the dbc.Tabs children list ...
    register_synth2_callbacks(app)

WHAT THIS TAB DOES
-------------------
Of the three classes of time-series synthetic-data method —
  (1) mimic the *shape* (classical linear models: ARIMA/SARIMAX …)
  (2) reproduce the same *statistical characteristics* (surrogates,
      bootstrap, copulas, moment-matching …)
  (3) reproduce the same *dynamical potential* (chaos theory, RQA,
      nonlinear state-space reconstruction …)
this tab implements Class (2) for any variable the user picks from a
dropdown.  Where the Class-1 tab asks "does a synthetic path *look* like
the real one?", this tab asks the stricter distributional question:
"does the synthetic series have the *same statistics* — mean, standard
deviation, skewness, kurtosis, the entire marginal distribution — as the
real one?"  Three complementary Class-2 generators are offered, all of
which reproduce the marginal distribution (and therefore every moment)
either exactly or to interpolation accuracy:

  IAAFT surrogates     exact moments (rank-remap of the real values) +
                       iteratively matched power spectrum / ACF
  Circular block       exact value pool (resampled real blocks); block
  bootstrap            length controls how much dependence survives
  Gaussian-rank AR     AR(p) fitted on normal scores, simulated, then
  (copula)             mapped back through the empirical quantile
                       function — exact marginal, AR-matched memory

Per-variable *identification* quantities (sorted sample, FFT amplitude
spectrum, normal scores + AIC-selected AR order, periodogram cycle used
as the default bootstrap block length) are computed once, eagerly, in a
background thread at import time.  Because Class-2 generation needs no
maximum-likelihood fitting, drawing fresh realisations for any method /
seed is then essentially instantaneous and happens on demand in the
callback — unlike the Class-1 tab there is no fixed pre-simulated
ensemble; every seed produces a brand-new ensemble in milliseconds.
"""

from pathlib import Path
from datetime import datetime
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
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.ar_model import ar_select_order

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

# Class-2 method catalogue --------------------------------------------------
METHODS = {
    "iaaft":     "IAAFT surrogate (exact moments + matched spectrum)",
    "bootstrap": "Circular block bootstrap (resampled real blocks)",
    "copula":    "Gaussian-rank AR / copula (exact marginal + AR memory)",
}
DEFAULT_METHOD = "iaaft"
N_REALS_MAX    = 8      # ensemble size ceiling in the UI
IAAFT_ITER     = 120    # spectrum/rank alternation sweeps (converges ≪ 100)
AR_MAXLAG      = 48     # ceiling for AIC lag selection on normal scores
BOOT_BLOCK_MAX = 96     # block-length slider ceiling (hours)

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
METHOD_OPTIONS = [{"label": v, "value": k} for k, v in METHODS.items()]
if DEFAULT_COL not in AVAILABLE:
    DEFAULT_COL = AVAILABLE[0]

# ─────────────────────────────────────────────────────────────────────────────
#  SEASONAL-PERIOD DETECTION  (periodogram — same convention as synth_tab.py)
#  Here the dominant cycle is not a SARIMA seasonal period but the *default
#  block length* for the circular block bootstrap: resampling whole cycles
#  keeps each block internally coherent, so the only dependence damage is at
#  the (rare) block seams.
# ─────────────────────────────────────────────────────────────────────────────
def _periodogram(y, min_period=4, max_period=BOOT_BLOCK_MAX):
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(np.fft.rfft(y)) ** 2
    mask = freqs > 0
    periods = 1.0 / freqs[mask]
    power = power[mask]
    keep = (periods >= min_period) & (periods <= max_period)
    return periods[keep], power[keep]


# ─────────────────────────────────────────────────────────────────────────────
#  PER-VARIABLE IDENTIFICATION  (runs once per variable, eagerly)
#  No maximum-likelihood model here — Class-2 identification is cheap:
#    1. sorted sample + FFT amplitude spectrum        (IAAFT ingredients)
#    2. periodogram cycle -> default block length     (bootstrap ingredient)
#    3. normal scores + AIC-selected AR(p) fit        (copula ingredients)
#    4. real ACF + real moments                       (validation targets)
# ─────────────────────────────────────────────────────────────────────────────
def _moments(x) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)),
        skew=float(sp_stats.skew(x, bias=False)),
        kurt=float(sp_stats.kurtosis(x, bias=False)),   # excess kurtosis
        p05=float(np.percentile(x, 5)),
        p95=float(np.percentile(x, 95)),
    )


def _safe_acf(x, nlags):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm_acf(x, nlags=nlags, fft=True)
    except Exception:
        return np.full(nlags + 1, np.nan)


def _identify(col: str) -> dict:
    s_ser = DF[col].dropna()
    y = s_ser.values.astype(float)
    idx = s_ser.index
    n = len(y)

    y_sorted = np.sort(y)                              # empirical quantile fn
    amp = np.abs(np.fft.rfft(y - y.mean()))            # target amplitude spectrum
    mu_y = float(np.mean(y))

    # default bootstrap block length = dominant periodogram cycle
    periods, powers = _periodogram(y)
    block_len = int(round(periods[np.argmax(powers)])) if len(periods) else 24
    block_len = int(np.clip(block_len, 4, BOOT_BLOCK_MAX))

    # normal scores: rank -> uniform (Hazen plotting position) -> Φ⁻¹
    ranks = sp_stats.rankdata(y, method="average")
    u = (ranks - 0.5) / n
    z = sp_stats.norm.ppf(u)

    # AR(p) on the normal scores, p chosen by AIC up to AR_MAXLAG
    ar_order, ar_const, ar_phi, ar_sigma = 1, 0.0, np.array([0.9]), 0.44
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = ar_select_order(z, maxlag=min(AR_MAXLAG, n // 20),
                                  ic="aic", old_names=False)
            res = sel.model.fit()
        lags = sel.ar_lags or []
        ar_order = int(max(lags)) if lags else 0
        params = np.asarray(res.params, dtype=float)
        ar_const = float(params[0])
        phi_sel = params[1:]
        # scatter selected-lag coefficients into a dense φ₁..φ_p vector
        ar_phi = np.zeros(ar_order)
        for coef, lag in zip(phi_sel, lags):
            ar_phi[lag - 1] = float(coef)
        ar_sigma = float(np.sqrt(np.mean(np.asarray(res.resid, dtype=float) ** 2)))
    except Exception:
        pass                                            # keep AR(1) fallback

    nlags_view = min(48, n // 4)
    return dict(
        col=col, y=y, idx=idx, n=n, mu_y=mu_y,
        y_sorted=y_sorted, amp=amp,
        block_len=block_len, periods=periods, powers=powers,
        z=z, ar_order=ar_order, ar_const=ar_const, ar_phi=ar_phi,
        ar_sigma=ar_sigma,
        nlags_view=nlags_view,
        acf_real=_safe_acf(y, nlags_view),
        moments_real=_moments(y),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  EAGER PRECOMPUTE  —  identification for every variable in a daemon thread,
#  same pattern as synth_tab.py.  Much faster than the SARIMA fits (seconds,
#  not minutes): nothing here needs maximum likelihood except one AR-by-OLS.
#  Also cached to disk via cache_utils: a later run with the same spg.csv and
#  config loads every variable's identification straight from .cache/ instead
#  of redoing even this lightweight background work.
# ─────────────────────────────────────────────────────────────────────────────
_ID_CACHE: dict = {}
_STATUS_LOCK = threading.Lock()
_STATUS = dict(current=None, done=0, total=len(AVAILABLE), finished=False, errors={})

_ID_SIG = make_signature(
    CSV_PATH, version="synth2-identification-v1",
    ar_maxlag=AR_MAXLAG, boot_block_max=BOOT_BLOCK_MAX, available=AVAILABLE,
)


def _precompute_worker():
    t_start = time.time()
    queue = [DEFAULT_COL] + [c for c in AVAILABLE if c != DEFAULT_COL]
    for i, col in enumerate(queue, start=1):
        with _STATUS_LOCK:
            _STATUS["current"] = col
        t0 = time.time()
        try:
            ident = _identify(col)
            _ID_CACHE[col] = ident
            print(f"[synth2_tab]  ({i}/{len(queue)}) {col}: block s={ident['block_len']}  "
                  f"AR(p)={ident['ar_order']}  [{time.time() - t0:.1f}s]")
        except Exception as e:                          # never kill the thread
            with _STATUS_LOCK:
                _STATUS["errors"][col] = f"{type(e).__name__}: {e}"
            print(f"[synth2_tab]  ({i}/{len(queue)}) {col}: identification failed — "
                  f"{type(e).__name__}: {e}")
        with _STATUS_LOCK:
            _STATUS["done"] = i
    print(f"[synth2_tab] Ready OK — Class-2 identification done in {time.time() - t_start:.1f}s")
    # Save before flipping `finished`, so finished=True always implies the
    # cache write already landed on disk (see synth_tab_control.py for why).
    save_cache("synth2_identification", _ID_SIG, dict(_ID_CACHE))
    with _STATUS_LOCK:
        _STATUS["finished"] = True
        _STATUS["current"] = None


_cached_ids = load_cache("synth2_identification", _ID_SIG)
if _cached_ids is not None:
    _ID_CACHE.update(_cached_ids)
    with _STATUS_LOCK:
        _STATUS.update(done=len(AVAILABLE), total=len(AVAILABLE), finished=True, current=None)
    print(f"[synth2_tab] Loaded cached Class-2 identification for {len(_ID_CACHE)} variables "
          "from .cache/ ✔  (delete .cache/ to force a full recompute)")
else:
    print(f"[synth2_tab] Starting background Class-2 identification for {len(AVAILABLE)} variables "
          "(sorted samples, spectra, normal-score AR fits) — generation itself is on-demand and instant.")
    _WORKER = threading.Thread(target=_precompute_worker, daemon=True, name="synth2-precompute")
    _WORKER.start()


def get_ident(col: str):
    return _ID_CACHE.get(col)


def precompute_status() -> dict:
    with _STATUS_LOCK:
        return dict(_STATUS)


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS-2 GENERATORS  (all return one realisation of length n)
# ─────────────────────────────────────────────────────────────────────────────
def _gen_iaaft(ident: dict, rng: np.random.Generator) -> np.ndarray:
    """Iterative Amplitude-Adjusted Fourier Transform surrogate.
    Alternates (a) imposing the real amplitude spectrum with randomised
    phases, and (b) rank-remapping onto the real sorted values.  Step (b)
    is executed last, so the output is a permutation of the real sample —
    every marginal moment is reproduced *exactly*."""
    y_sorted, amp, n = ident["y_sorted"], ident["amp"], ident["n"]
    x = rng.permutation(y_sorted)                       # start: value shuffle
    for _ in range(IAAFT_ITER):
        # (a) spectrum step: keep phases of x, impose real amplitudes
        phase = np.angle(np.fft.rfft(x - x.mean()))
        x = np.fft.irfft(amp * np.exp(1j * phase), n) + ident["mu_y"]
        # (b) distribution step: rank-remap back onto the real values
        order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        x = y_sorted[order]
    return x


def _gen_bootstrap(ident: dict, rng: np.random.Generator, block_len: int) -> np.ndarray:
    """Circular block bootstrap: concatenate blocks of length `block_len`
    starting at uniform random positions, wrapping around the end.  The
    output values are drawn from the real sample itself, so the marginal
    distribution is exact up to multiplicity."""
    y, n = ident["y"], ident["n"]
    block_len = int(np.clip(block_len or ident["block_len"], 2, n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    take = (starts[:, None] + np.arange(block_len)[None, :]) % n
    return y[take.ravel()][:n]


def _gen_copula(ident: dict, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-rank AR ('copula') generator: simulate the AIC-selected
    AR(p) fitted on the normal scores, convert back to uniforms through Φ,
    then through the empirical quantile function of the real data — the
    marginal is exact to interpolation accuracy, the autocorrelation is
    the AR fit's."""
    n, p = ident["n"], ident["ar_order"]
    c, phi, sigma = ident["ar_const"], ident["ar_phi"], ident["ar_sigma"]
    burn = max(200, 5 * max(p, 1))
    if p == 0:
        zs = c + rng.normal(0.0, sigma, n)
    else:
        zs = np.empty(n + burn)
        zs[:p] = rng.normal(0.0, 1.0, p)
        eps = rng.normal(0.0, sigma, n + burn)
        rev_phi = phi[::-1]                             # dot with x[t-p:t]
        for t in range(p, n + burn):
            zs[t] = c + rev_phi @ zs[t - p:t] + eps[t]
        zs = zs[burn:]
    # standardise -> uniforms -> empirical quantiles (linear interpolation)
    sd = np.std(zs)
    u = sp_stats.norm.cdf((zs - np.mean(zs)) / (sd if sd > 1e-12 else 1.0))
    u = np.clip(u, 0.5 / n, 1 - 0.5 / n)
    return np.quantile(ident["y_sorted"], u, method="linear")


def generate(ident: dict, method: str, n_sims: int, seed: int,
             block_len: int | None = None) -> np.ndarray:
    """(n, n_sims) matrix of independent Class-2 realisations."""
    n_sims = max(1, min(int(n_sims), N_REALS_MAX))
    sims = np.empty((ident["n"], n_sims))
    for k in range(n_sims):
        rng = np.random.default_rng(int(seed) + 1000 * k)
        if method == "bootstrap":
            sims[:, k] = _gen_bootstrap(ident, rng, block_len)
        elif method == "copula":
            sims[:, k] = _gen_copula(ident, rng)
        else:
            sims[:, k] = _gen_iaaft(ident, rng)
    return sims


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY FIGURE BUILDERS
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


def _downsample(arr, max_pts=1500):
    step = max(1, len(arr) // max_pts)
    return arr[::step]


def fig_overlay(ident: dict, sims: np.ndarray, method: str) -> go.Figure:
    col = ident["col"]
    label, unit = LABELS.get(col, col), _unit_suffix(col)
    color = COLORS.get(col, ACCENT2)
    x_hist = _downsample(ident["idx"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_hist, y=_downsample(ident["y"]), mode="lines", name="Real (observed)",
        line=dict(color=color, width=1.6),
        hovertemplate=f"Real: %{{y:.2f}}{unit}<extra></extra>",
    ))
    for k in range(sims.shape[1]):
        fig.add_trace(go.Scatter(
            x=x_hist, y=_downsample(sims[:, k]), mode="lines",
            name=f"Synthetic realisation {k+1}",
            line=dict(color=SYN_C, width=1, dash="dot"), opacity=0.55,
            hovertemplate=f"Synthetic #{k+1}: %{{y:.2f}}{unit}<extra></extra>",
        ))
    fig.update_xaxes(**_AX, title=None)
    fig.update_yaxes(**_AX, title=dict(text=f"{label}{unit}", standoff=4))
    fig.update_layout(**_base_layout(
        height=380, hovermode="x unified", showlegend=True,
        legend=dict(font=dict(size=9, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18, x=0),
    ))
    return fig


def fig_periodogram(ident: dict) -> go.Figure:
    s = ident["block_len"]
    periods, powers = ident["periods"], ident["powers"]
    fig = go.Figure(go.Scatter(
        x=periods, y=powers, mode="lines",
        line=dict(color=MUTED2, width=1.4), fill="tozeroy",
        fillcolor="rgba(136,153,187,0.08)",
        hovertemplate="period: %{x:.1f}<br>power: %{y:.3g}<extra></extra>",
    ))
    if len(powers):
        fig.add_trace(go.Scatter(
            x=[s], y=[powers[np.argmax(powers)]], mode="markers+text",
            marker=dict(size=11, color=AMBER, line=dict(color=BG, width=1.5)),
            text=[f"  block = {s}"], textposition="top right",
            textfont=dict(size=11, color=AMBER, family=FONT_MONO),
            hovertemplate=f"dominant cycle ≈ {s} samples<extra></extra>",
        ))
    fig.update_xaxes(**_AX, title=dict(text="period (samples)", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="spectral power", standoff=4))
    fig.update_layout(**_base_layout(height=260, showlegend=False,
        title=dict(text="Periodogram · dominant cycle sets the default bootstrap block length",
                   font=dict(size=12, color=TEXT), x=0.01)))
    return fig


def fig_ecdf_qq(ident: dict, sims: np.ndarray) -> go.Figure:
    """Left: ECDF overlay (real vs pooled synthetic).  Right: QQ plot of
    pooled synthetic quantiles against real quantiles — the sharpest visual
    test of 'same marginal distribution'."""
    col = ident["col"]
    unit = _unit_suffix(col)
    color = COLORS.get(col, ACCENT2)
    y_sorted = ident["y_sorted"]
    n = ident["n"]
    pool = np.sort(sims.ravel())

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=["ECDF · real vs pooled synthetic",
                                        "QQ plot · synthetic vs real quantiles"])
    # ECDF (downsample the plotted staircase, not the statistics)
    q_grid = np.linspace(0, 1, 400)
    fig.add_trace(go.Scatter(x=np.quantile(y_sorted, q_grid), y=q_grid, mode="lines",
                             name="Real", line=dict(color=color, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.quantile(pool, q_grid), y=q_grid, mode="lines",
                             name="Synthetic (pooled)",
                             line=dict(color=SYN_C, width=2, dash="dash")), row=1, col=1)
    # QQ
    q_real = np.quantile(y_sorted, q_grid)
    q_syn  = np.quantile(pool, q_grid)
    lo, hi = float(min(q_real.min(), q_syn.min())), float(max(q_real.max(), q_syn.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect match",
                             line=dict(color=MUTED, width=1, dash="dot"),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=q_real, y=q_syn, mode="markers", name="quantile pairs",
                             marker=dict(size=4, color=GREEN, opacity=0.8),
                             showlegend=False,
                             hovertemplate=(f"real: %{{x:.2f}}{unit}"
                                            f"<br>synthetic: %{{y:.2f}}{unit}<extra></extra>")),
                  row=1, col=2)
    fig.update_xaxes(**_AX, title=dict(text=f"value{unit}", standoff=4), row=1, col=1)
    fig.update_yaxes(**_AX, title=dict(text="F(x)", standoff=4), range=[-0.02, 1.02], row=1, col=1)
    fig.update_xaxes(**_AX, title=dict(text=f"real quantile{unit}", standoff=4), row=1, col=2)
    fig.update_yaxes(**_AX, title=dict(text=f"synthetic quantile{unit}", standoff=4), row=1, col=2)
    fig.update_layout(**_base_layout(
        height=330, hovermode="closest",
        legend=dict(font=dict(size=10, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=1.16, x=0.02),
    ))
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    return fig


def fig_hist_compare(ident: dict, sims: np.ndarray) -> go.Figure:
    col = ident["col"]
    label, unit = LABELS.get(col, col), _unit_suffix(col)
    color = COLORS.get(col, ACCENT2)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ident["y"], name="Real", histnorm="probability density",
                               marker_color=color, opacity=0.55, nbinsx=40))
    fig.add_trace(go.Histogram(x=sims.ravel(), name="Synthetic (pooled)",
                               histnorm="probability density",
                               marker_color=SYN_C, opacity=0.45, nbinsx=40))
    fig.update_xaxes(**_AX, title=dict(text=f"{label}{unit}", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="density", standoff=4))
    fig.update_layout(**_base_layout(
        height=300, barmode="overlay",
        legend=dict(font=dict(size=10, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0.32),
    ))
    return fig


def fig_acf_compare(ident: dict, sims: np.ndarray) -> go.Figure:
    nlags = ident["nlags_view"]
    lags = list(range(nlags + 1))
    color = COLORS.get(ident["col"], ACCENT2)
    acf_sim = np.column_stack([_safe_acf(sims[:, k], nlags) for k in range(sims.shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo = np.nan_to_num(np.nanmin(acf_sim, axis=1))
        hi = np.nan_to_num(np.nanmax(acf_sim, axis=1))
        mean_sim = np.nan_to_num(np.nanmean(acf_sim, axis=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags + lags[::-1], y=list(hi) + list(lo)[::-1],
                             fill="toself", fillcolor="rgba(126,179,255,0.14)",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=lags, y=ident["acf_real"], mode="lines+markers", name="Real",
                             line=dict(color=color, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=lags, y=mean_sim, mode="lines",
                             name="Synthetic (ensemble mean)",
                             line=dict(color=SYN_C, width=2, dash="dash")))
    fig.update_xaxes(**_AX, title=dict(text="lag", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="autocorrelation", standoff=4), range=[-0.3, 1.05])
    fig.update_layout(**_base_layout(
        height=300, hovermode="closest",
        legend=dict(font=dict(size=10, color=MUTED2, family=FONT_MONO),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=1.16, x=0.3),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL SCOREBOARD  (the point of a Class-2 tab)
# ─────────────────────────────────────────────────────────────────────────────
_MOMENT_ROWS = [
    ("mean", "Mean"), ("std", "Std deviation σ"), ("skew", "Skewness"),
    ("kurt", "Excess kurtosis"), ("p05", "5th percentile"), ("p95", "95th percentile"),
]


def moment_table(ident: dict, sims: np.ndarray):
    """Real vs pooled-synthetic moment table with absolute deviations,
    colour-graded: |Δ| within 2% of the real σ (or 0.05 for the unitless
    shape moments) shows green, within 10% amber, beyond red."""
    r = ident["moments_real"]
    s = _moments(sims)
    sd = max(r["std"], 1e-9)

    def tol(key):
        return 0.05 if key in ("skew", "kurt") else 0.02 * sd

    header = html.Tr([html.Th("Statistic"), html.Th("Real"), html.Th("Synthetic (pooled)"),
                      html.Th("|Δ|"), html.Th("Verdict")])
    rows = [header]
    for key, name in _MOMENT_ROWS:
        d = abs(s[key] - r[key])
        t = tol(key)
        if d <= t:
            verdict, colr = "match", GREEN
        elif d <= 5 * t:
            verdict, colr = "close", AMBER
        else:
            verdict, colr = "off", RED
        rows.append(html.Tr([
            html.Td(name),
            html.Td(f"{r[key]:+.3f}"),
            html.Td(f"{s[key]:+.3f}", style={"color": SYN_C}),
            html.Td(f"{d:.3g}"),
            html.Td(verdict, style={"color": colr, "fontWeight": "700"}),
        ]))
    return html.Table(rows, style={
        "width": "100%", "borderCollapse": "collapse", "fontFamily": FONT_MONO,
        "fontSize": "12px", "color": TEXT,
    }, className="s2-moment-table")


def stat_banner(ident: dict, sims: np.ndarray, method: str, block_len: int):
    """KS two-sample test + normalised Wasserstein distance, real vs pooled."""
    pool = sims.ravel()
    ks = sp_stats.ks_2samp(ident["y"], pool)
    wd = sp_stats.wasserstein_distance(ident["y"], pool)
    wd_rel = wd / max(ident["moments_real"]["std"], 1e-9)
    knob = {"iaaft": f"{IAAFT_ITER} iterations",
            "bootstrap": f"block = {block_len} h",
            "copula": f"AR({ident['ar_order']})"}[method]
    return _banner(
        (METHODS[method].split(" (")[0], "Class-2 method", GREEN,
         {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
        (knob, "Method setting", AMBER,
         {"fontSize": "14px", "lineHeight": "1.3"}),
        (f"{ks.statistic:.4f}", "KS distance D (real vs pooled)",
         GREEN if ks.statistic < 0.02 else (AMBER if ks.statistic < 0.05 else RED)),
        (f"{ks.pvalue:.3f}", "KS p-value (H₀: same distribution)",
         GREEN if ks.pvalue >= 0.05 else AMBER),
        (f"{wd_rel:.4f}", "Wasserstein distance / σ",
         GREEN if wd_rel < 0.02 else (AMBER if wd_rel < 0.05 else RED)),
        (f"{ident['n']:,}", "Observations", MUTED2),
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

def _theory_card(title, tag, color, paragraphs, math_eq=None):
    children = [html.H3([html.Span(title, style={"color": color}), html.Span(tag, className="tc-tag")])]
    for p in paragraphs:
        children.append(html.P(p) if isinstance(p, str) else html.Div(p, style={"marginBottom": "8px"}))
    if math_eq:
        children.append(html.Code(math_eq, className="math"))
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


def identification_card(ident: dict):
    col = ident["col"]
    label = LABELS.get(col, col)
    return _theory_card(
        "Identifying the ingredients for THIS series", "IDENTIFICATION", GREEN,
        [
            [
                f"Class-2 identification is deliberately lightweight — no likelihood surface to "
                f"climb. For {label} three ingredients were extracted once at startup: ",
                html.Br(), html.Br(),
                f"① its sorted sample (the empirical quantile function) together with the FFT "
                f"amplitude spectrum of its {ident['n']:,} observations — everything IAAFT needs; ",
                html.Br(), html.Br(),
                f"② its dominant periodogram cycle, ≈ {ident['block_len']} samples, adopted as the "
                "default block length for the circular block bootstrap so that resampled blocks "
                "contain whole cycles and dependence is only broken at block seams; ",
                html.Br(), html.Br(),
                f"③ its normal scores z = Φ⁻¹((rank−½)/n), on which an autoregression was selected "
                f"by AIC up to lag {AR_MAXLAG}: the winner is AR({ident['ar_order']}) with residual "
                f"σ = {ident['ar_sigma']:.3f} — the memory the Gaussian-rank generator reproduces.",
            ],
            html.Div([
                html.Span("Why is there no fitting wait on this tab? ",
                          style={"fontWeight": "700", "color": GREEN}),
                "None of the three generators needs an iterative maximum-likelihood fit; sorting, one "
                "FFT and one OLS autoregression cover all of them. That is also why the seed control "
                "regenerates a brand-new ensemble instantly rather than reading a fixed pre-simulated "
                "tensor as the Class-1 tab does.",
            ], className="warn-box", style={"marginTop": "10px"}),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  THEORY PANEL
# ─────────────────────────────────────────────────────────────────────────────
SYNTH2_THEORY = html.Div([
    html.Div("📖 Class-II Synthetic Generation — Theory", className="theory-title"),

    html.P([
        "Where a ", html.B("Class-1"), " method fits a generative model and hopes the statistics come out "
        "close, a ", html.B("Class-2"), " method inverts the priority: the statistical characteristics of "
        "the real series — its mean, standard deviation, skewness, kurtosis, and in fact its entire "
        "marginal distribution — are matched ", html.I("by construction"), ", and the temporal ordering is "
        "then arranged to be as faithful as the method allows. All three generators on this tab share one "
        "trick: they manipulate the ", html.I("order"), " of values while drawing the values themselves "
        "from the real sample (or its interpolated quantile function). Reordering a sample cannot change "
        "any of its moments — which is precisely why the moment table below shows exact or near-exact "
        "matches, something the Class-1 SARIMA tab can only approximate.",
    ], className="section-desc", style={"marginBottom": "18px"}),

    html.Div([
        html.Div([
            _theory_card(
                "IAAFT — surrogate with exact moments and matched spectrum", "METHOD ①", ACCENT2,
                [
                    "The Iterative Amplitude-Adjusted Fourier Transform starts from a random shuffle of "
                    "the real values and alternates two projections until they agree: (a) impose the real "
                    "series' FFT amplitude spectrum while keeping the current phases — this fixes the "
                    "autocorrelation, by the Wiener-Khinchin theorem; (b) rank-remap the result back onto "
                    "the real sorted values — this fixes the distribution.",
                    "Because the final operation is always (b), the output is literally a permutation of "
                    "the observed sample: mean, σ, skewness, kurtosis and every quantile are reproduced "
                    "exactly, while the ACF converges to the real one as the iterations proceed. IAAFT is "
                    "the standard surrogate in nonlinearity testing for exactly this property.",
                ],
                math_eq="x⁽ᵏ⁺¹⁾ = sort(y)[rank( F⁻¹( |Y| · e^{i·arg F(x⁽ᵏ⁾)} ) )]",
            ),
            _theory_card(
                "Circular block bootstrap — resampling reality", "METHOD ②", GREEN,
                [
                    "Instead of modelling the process, resample it: cut the observed series into blocks "
                    "of length ℓ (here defaulting to the dominant periodogram cycle) starting at uniform "
                    "random positions — wrapping past the end, hence 'circular' — and concatenate blocks "
                    "until the target length is reached.",
                    "Every synthetic value is a real observed value, so the marginal distribution is exact "
                    "up to which blocks happened to be drawn; within each block every lag ≤ ℓ dependence, "
                    "including nonlinear and non-Gaussian dependence that SARIMA cannot represent, is "
                    "carried over untouched. The price is paid at the seams: correlation across block "
                    "boundaries is destroyed, so the ensemble ACF droops below the real one beyond lag ℓ. "
                    "The block-length slider makes that trade-off tangible.",
                ],
                math_eq="x* = [ y_{s₁}, …, y_{s₁+ℓ−1}, y_{s₂}, … ]  (indices mod n),  sⱼ ~ U{0,…,n−1}",
            ),
            _theory_card(
                "Gaussian-rank AR — a copula view", "METHOD ③", AMBER,
                [
                    "Transform the data to normal scores z = Φ⁻¹((rank−½)/n) — a monotone map, so it "
                    "preserves the ordering but makes the marginal exactly standard normal. Fit an AR(p) "
                    "there (order chosen by AIC), simulate a new Gaussian path, and push it back through "
                    "Φ and then through the empirical quantile function of the real data.",
                    "This is a Gaussian copula construction: the dependence lives in the Gaussian layer "
                    "(captured by the AR fit), the marginal lives in the quantile map (exact to "
                    "interpolation accuracy). Unlike IAAFT and the bootstrap it is a genuine parametric "
                    "generative model, so it could also produce series longer than the sample — the bridge "
                    "toward the SARIMAX-style generators of the Class-1 tab, but with the marginal "
                    "guaranteed instead of hoped for.",
                ],
                math_eq="z_t = Φ⁻¹(F̂(y_t)),   z*_t ~ AR(p),   y*_t = F̂⁻¹(Φ(z*_t))",
            ),
        ], className="theory-row"),
        html.Div(id="s2-id-card", className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px",
                                       "overflow": "hidden", "background": CARD2}),

    _callout(
        "tip", "How to read this tab — a walkthrough for any level",
        "Work top to bottom; each element answers one question in the Class-2 pipeline:",
        [
            ("Variable & method dropdowns → ", "\"Which series, and which statistical-matching "
             "strategy?\" — all three methods guarantee the marginal; they differ in how much temporal "
             "dependence survives and how."),
            ("Periodogram → ", "\"What cycle length does the data have?\" — here it sets the default "
             "bootstrap block length rather than a SARIMA seasonal period."),
            ("Overlay chart → ", "\"Do the synthetic paths occupy the same range and texture as the real "
             "one?\" — note that IAAFT/bootstrap paths need not line up with the real calendar: matching "
             "statistics does not require matching phase."),
            ("KS / Wasserstein banner → ", "\"Formally, is the pooled synthetic distribution the same as "
             "the real one?\" — KS D near 0 and Wasserstein/σ near 0 are the Class-2 pass marks."),
            ("Moment table → ", "\"Are mean, σ, skewness, kurtosis and the tail percentiles reproduced?\" "
             "— for IAAFT and the bootstrap the answer is exact by construction; for the copula, exact to "
             "interpolation accuracy."),
            ("ECDF + QQ plot → ", "\"Does the entire distribution match, not just four numbers?\" — the "
             "QQ points should hug the diagonal from tail to tail."),
            ("ACF comparison → ", "\"How much memory survived the statistical matching?\" — IAAFT should "
             "track the real ACF closely; the bootstrap decays after one block length; the copula follows "
             "its fitted AR profile."),
        ],
    ),
    _callout(
        "note", "Limits of Class-2 fidelity",
        "Matching the statistics is a guarantee about distributions, not about everything:",
        [
            "Phase information is deliberately sacrificed: an IAAFT surrogate reproduces the daily "
            "spectral peak but its 'days' do not align with real calendar days. If calendar alignment "
            "matters, that is a shape (Class-1) requirement, not a statistical one.",
            "The bootstrap cannot invent values it never saw — extremes beyond the observed record have "
            "probability zero, which understates tail risk for long synthetic horizons.",
            "The Gaussian-rank AR captures dependence only through a Gaussian copula with linear AR "
            "memory: joint tail dependence and asymmetric (up-fast / down-slow) dynamics are smoothed "
            "over.",
            "None of the three extrapolates a trend: they redistribute or resample history, so a "
            "structural drift (climate signal, degradation) is frozen at its historical mixture.",
            ("Where to go next: ", "Class-3 methods — phase-space reconstruction, recurrence "
             "quantification analysis (RQA), chaos-theoretic embeddings — try to preserve the underlying "
             "dynamical potential itself, at the cost of far more data and far less transparency."),
        ],
        ordered=False,
    ),
], className="theory-section")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab_synth2 = dbc.Tab(
    label="🎲  Synthetic Data II",
    tab_id="tab-synth2",
    children=html.Div([
        _section_hdr("Class-II Synthetic Data Generation (statistical matching)"),
        _section_desc(
            "This tab generates synthetic series that reproduce the statistical characteristics of the "
            "real one — mean, standard deviation, skewness, kurtosis, percentiles, the whole marginal "
            "distribution — exactly or to interpolation accuracy, by construction. Pick a variable and "
            "one of three interpretable Class-2 methods (IAAFT surrogates, circular block bootstrap, "
            "Gaussian-rank AR); identification ingredients were precomputed at startup, so every seed "
            "produces a brand-new ensemble instantly. The scoreboard below (KS test, Wasserstein "
            "distance, moment table, ECDF/QQ) is the formal evidence of the statistical match."
        ),

        # Polls the background identification; disables itself once finished.
        dcc.Interval(id="s2-poll", interval=1500, n_intervals=0),

        html.Div([
            html.Div([
                html.Label("Target variable"),
                dcc.Dropdown(id="s2-var", options=VAR_OPTIONS, value=DEFAULT_COL,
                             clearable=False, style={"minWidth": "280px"}),
            ], className="ctrl-block"),
            html.Div([
                html.Label("Class-2 method"),
                dcc.Dropdown(id="s2-method", options=METHOD_OPTIONS, value=DEFAULT_METHOD,
                             clearable=False, style={"minWidth": "360px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),

        html.Div([
            html.Div([
                html.Label(f"Number of realisations (1–{N_REALS_MAX})"),
                dcc.Slider(id="s2-n-sims", min=1, max=N_REALS_MAX, step=1, value=3,
                           marks={i: str(i) for i in range(1, N_REALS_MAX + 1)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label("Bootstrap block length (hours · method ② only)"),
                dcc.Slider(id="s2-block", min=4, max=BOOT_BLOCK_MAX, step=1, value=24,
                           marks={4: "4", 24: "24", 48: "48", 96: "96"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label("Random seed"),
                dcc.Input(id="s2-seed", type="number", value=42, min=0, step=1,
                          style={"width": "100px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),

        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div(id="s2-banner"),
            _plot_card([dcc.Graph(id="s2-overlay", config={"displaylogo": False})],
                       label="Real series vs Class-2 synthetic realisations"),
            html.Div(id="s2-mode-note", className="insight-bar"),
        ]),

        _section_hdr("Statistical Validation: the Class-2 scoreboard"),
        html.Div(id="s2-validation-desc", className="section-desc"),
        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div([
                html.Div(_plot_card([dcc.Graph(id="s2-periodogram", config={"displayModeBar": False})],
                                    label="Cycle detection · default bootstrap block length"),
                         style={"flex": "1", "minWidth": "360px"}),
                html.Div(_plot_card([html.Div(id="s2-moment-table", style={"padding": "10px 6px"})],
                                    label="Moment table · real vs pooled synthetic"),
                         style={"flex": "1", "minWidth": "360px"}),
            ], style={"display": "flex", "gap": "16px", "alignItems": "stretch", "flexWrap": "wrap"}),
            _plot_card([dcc.Graph(id="s2-ecdf-qq", config={"displayModeBar": False})],
                       label="Distributional match · ECDF overlay and QQ plot"),
            _plot_card([dcc.Graph(id="s2-hist", config={"displayModeBar": False})],
                       label="Marginal distribution · real vs pooled synthetic"),
            _plot_card([dcc.Graph(id="s2-acf", config={"displayModeBar": False})],
                       label="Dependence check · real ACF vs synthetic ensemble band"),
        ]),

        SYNTH2_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  — register with your app via register_synth2_callbacks(app)
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


_MODE_NOTES = {
    "iaaft": ("IAAFT mode: each dotted path is a permutation of the real sample whose spectrum was "
              "iteratively matched to the real one — identical moments by construction, near-identical "
              "ACF, deliberately different phase (its 'days' do not align with the calendar)."),
    "bootstrap": ("Bootstrap mode: each dotted path is a concatenation of real blocks drawn circularly "
                  "from the observed record — every value is a real observation, dependence is intact "
                  "within blocks and broken only at seams. Shorten the block length to watch the ACF "
                  "collapse toward white noise; lengthen it to approach a shuffled copy of history."),
    "copula": ("Gaussian-rank AR mode: each dotted path is a fresh AR simulation in normal-score space "
               "mapped back through the real empirical quantile function — exact marginal, linear "
               "AR memory, and (uniquely on this tab) a genuine parametric model that could extend "
               "beyond the historical span."),
}


def _progress_banner(col: str, st: dict):
    cur = st.get("current")
    cur_label = LABELS.get(cur, cur) if cur else "—"
    return _banner(
        (f"{st['done']}/{st['total']}", "Variables identified so far", ACCENT2),
        (LABELS.get(col, col), "Waiting for", AMBER,
         {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
        (cur_label, "Currently identifying", MUTED2,
         {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
    )


def register_synth2_callbacks(app):

    @app.callback(
        Output("s2-block", "value"),
        Input("s2-var", "value"),
    )
    def _default_block(col):
        """When the variable changes, snap the bootstrap block-length slider
        to that variable's detected dominant cycle (its documented default).
        Falls back to 24 h while identification is still running."""
        ident = get_ident(col or DEFAULT_COL)
        return ident["block_len"] if ident else 24

    @app.callback(
        Output("s2-banner", "children"),
        Output("s2-overlay", "figure"),
        Output("s2-mode-note", "children"),
        Output("s2-validation-desc", "children"),
        Output("s2-periodogram", "figure"),
        Output("s2-moment-table", "children"),
        Output("s2-ecdf-qq", "figure"),
        Output("s2-hist", "figure"),
        Output("s2-acf", "figure"),
        Output("s2-id-card", "children"),
        Output("s2-poll", "disabled"),
        Input("s2-var", "value"),
        Input("s2-method", "value"),
        Input("s2-n-sims", "value"),
        Input("s2-block", "value"),
        Input("s2-seed", "value"),
        Input("s2-poll", "n_intervals"),
    )
    def update_synth2(col, method, n_sims, block_len, seed, _tick):
        col = col or DEFAULT_COL
        method = method if method in METHODS else DEFAULT_METHOD
        st = precompute_status()
        ident = get_ident(col)

        if ident is None:
            err = st["errors"].get(col)
            if err:
                wait_msg = f"Could not identify {LABELS.get(col, col)} — {err}"
            else:
                wait_msg = (f"Identifying Class-2 ingredients in the background "
                            f"({st['done']}/{st['total']} done) — this variable will "
                            "appear automatically when ready.")
            ph = _placeholder_fig(wait_msg)
            return (_progress_banner(col, st), ph, wait_msg, wait_msg, ph,
                    None, ph, ph, ph, None, st["finished"])

        n_sims = int(n_sims or 1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = 42
        try:
            block_len = int(block_len)
        except (TypeError, ValueError):
            block_len = ident["block_len"]

        sims = generate(ident, method, n_sims, seed, block_len)
        eff_block = block_len if method == "bootstrap" else ident["block_len"]

        validation_desc = (
            f"Ensemble of {sims.shape[1]} independent realisations of "
            f"{LABELS.get(col, col)} generated on demand (seed {seed}) with the "
            f"{METHODS[method].split(' (')[0]} method, pooled for the distributional tests below. "
            "The moment table and QQ plot are the direct evidence for the Class-2 claim: the "
            "statistics match because the construction forces them to, not because a fitted model "
            "happened to land close."
        )
        return (
            stat_banner(ident, sims, method, eff_block),
            fig_overlay(ident, sims, method),
            _MODE_NOTES[method],
            validation_desc,
            fig_periodogram(ident),
            moment_table(ident, sims),
            fig_ecdf_qq(ident, sims),
            fig_hist_compare(ident, sims),
            fig_acf_compare(ident, sims),
            identification_card(ident),
            st["finished"],
        )
