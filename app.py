# python app.py
# then open http://127.0.0.1:8050

import warnings; warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import (
    adfuller, kpss, zivot_andrews,
    acf, pacf,
    grangercausalitytests, coint,
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from scipy import stats as sp_stats
from datetime import datetime, timedelta

from arch.unitroot import PhillipsPerron
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = Path("spg.csv")
NLAGS      = 48
THRES      = 0.05
MAX_LAG_GC = 6
TS_POINTS  = 1200

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

GROUPS = {
    "temperature_2_m_above_gnd":        ("Temperature",    "#f07040"),
    "relative_humidity_2_m_above_gnd":  ("Humidity",       "#34c9a0"),
    "mean_sea_level_pressure_MSL":      ("Pressure",       "#a78bfa"),
    "total_precipitation_sfc":          ("Precipitation",  "#60a5fa"),
    "snowfall_amount_sfc":              ("Precipitation",  "#60a5fa"),
    "total_cloud_cover_sfc":            ("Cloud",          "#94a3b8"),
    "high_cloud_cover_high_cld_lay":    ("Cloud",          "#94a3b8"),
    "medium_cloud_cover_mid_cld_lay":   ("Cloud",          "#94a3b8"),
    "low_cloud_cover_low_cld_lay":      ("Cloud",          "#94a3b8"),
    "shortwave_radiation_backwards_sfc":("Radiation",      "#fbbf24"),
    "wind_speed_10_m_above_gnd":        ("Wind",           "#34d399"),
    "wind_direction_10_m_above_gnd":    ("Wind",           "#34d399"),
    "wind_speed_80_m_above_gnd":        ("Wind",           "#34d399"),
    "wind_direction_80_m_above_gnd":    ("Wind",           "#34d399"),
    "wind_speed_900_mb":                ("Wind",           "#34d399"),
    "wind_direction_900_mb":            ("Wind",           "#34d399"),
    "wind_gust_10_m_above_gnd":         ("Wind",           "#34d399"),
    "angle_of_incidence":               ("Solar Geometry", "#fbbf24"),
    "zenith":                           ("Solar Geometry", "#fbbf24"),
    "azimuth":                          ("Solar Geometry", "#fbbf24"),
}

INSIGHTS = {
    "temperature_2_m_above_gnd":        "Sinusoidal ACF peaking at lag 24 (r≈0.98). Diurnal heating cycle dominates. PACF cuts off after lag 12 → AR(12) residual structure.",
    "relative_humidity_2_m_above_gnd":  "Mirror of temperature — peaks overnight, troughs mid-afternoon. Strong 24h period. PACF cuts off after lag 2 → AR(2).",
    "mean_sea_level_pressure_MSL":      "Weak semi-diurnal (~12h) tidal oscillation on synoptic variability. ACF shows gentle peaks at lags 12 and 24.",
    "total_precipitation_sfc":          "Near-zero ACF at all lags — highly intermittent. Near-white-noise at hourly resolution. Essentially unpredictable beyond lag 1.",
    "snowfall_amount_sfc":              "Highly sparse with near-zero ACF. Rare, episodic events dominated by synoptic patterns over days.",
    "total_cloud_cover_sfc":            "Moderate 24h cycle from sea-breeze and convective patterns. PACF cuts off ~lag 3. Explains cloud cover failure in Granger tests.",
    "high_cloud_cover_high_cld_lay":    "Weaker ACF than total cloud — cirrus driven by upper-tropospheric dynamics independent of surface diurnal cycles.",
    "medium_cloud_cover_mid_cld_lay":   "Slightly stronger diurnal structure than high clouds from afternoon convection. ACF decays quickly.",
    "low_cloud_cover_low_cld_lay":      "Stratus/fog shows persistent overnight ACF from nocturnal radiation cooling, burns off mid-morning.",
    "shortwave_radiation_backwards_sfc":"Textbook diurnal sine wave — peaks at lag 24 (r≈0.995) and lag 48. Primary causal driver of solar generation.",
    "wind_speed_10_m_above_gnd":        "Moderate 24h periodicity from sea-breeze. PACF cuts off at lag 3 → low-order AR process.",
    "wind_direction_10_m_above_gnd":    "Circular variable (0-360°) — ACF/PACF interpreted cautiously. Sea-breeze rotation near lag 24.",
    "wind_speed_80_m_above_gnd":        "Stronger persistence than 10m from reduced surface friction. Smoother diurnal cycle.",
    "wind_direction_80_m_above_gnd":    "Circular variable. Less influenced by surface obstacles — cleaner but noisy diurnal signal.",
    "wind_speed_900_mb":                "Upper-level (~1km) wind shows weaker diurnal variation and stronger synoptic-scale persistence.",
    "wind_direction_900_mb":            "Reflects synoptic steering flow. Circular variable — less diurnally driven.",
    "wind_gust_10_m_above_gnd":         "Closely coupled to 10m wind (similar ACF) but higher variance and faster decorrelation from turbulent gusting.",
    "angle_of_incidence":               "Geometrically determined — near-perfect 24h ACF. Clean AR(2) PACF cut-off identical to shortwave radiation.",
    "zenith":                           "Purely deterministic astronomical quantity. ACF is a cosine wave with period 24h. Zero stochastic component.",
    "azimuth":                          "Sweeps east→west every 24h. Circular nature (0°/360° wrap-around) may cause PACF numerical instability.",
}

COINT_PAIRS = [
    ("shortwave_radiation_backwards_sfc", "temperature_2_m_above_gnd"),
    ("shortwave_radiation_backwards_sfc", "relative_humidity_2_m_above_gnd"),
    ("temperature_2_m_above_gnd",         "relative_humidity_2_m_above_gnd"),
    ("wind_speed_10_m_above_gnd",         "wind_speed_80_m_above_gnd"),
    ("wind_speed_80_m_above_gnd",         "wind_speed_900_mb"),
    ("wind_speed_10_m_above_gnd",         "wind_gust_10_m_above_gnd"),
    ("total_cloud_cover_sfc",             "shortwave_radiation_backwards_sfc"),
    ("total_cloud_cover_sfc",             "temperature_2_m_above_gnd"),
    ("zenith",                            "angle_of_incidence"),
    ("zenith",                            "shortwave_radiation_backwards_sfc"),
    ("mean_sea_level_pressure_MSL",       "wind_speed_10_m_above_gnd"),
    ("total_cloud_cover_sfc",             "high_cloud_cover_high_cld_lay"),
    ("wind_direction_10_m_above_gnd",     "wind_direction_80_m_above_gnd"),
]

GRANGER_PAIRS = [
    ("shortwave_radiation_backwards_sfc", "temperature_2_m_above_gnd"),
    ("shortwave_radiation_backwards_sfc", "relative_humidity_2_m_above_gnd"),
    ("shortwave_radiation_backwards_sfc", "total_cloud_cover_sfc"),
    ("temperature_2_m_above_gnd",         "relative_humidity_2_m_above_gnd"),
    ("wind_speed_10_m_above_gnd",         "wind_gust_10_m_above_gnd"),
    ("wind_speed_80_m_above_gnd",         "wind_gust_10_m_above_gnd"),
    ("total_cloud_cover_sfc",             "shortwave_radiation_backwards_sfc"),
    ("zenith",                            "shortwave_radiation_backwards_sfc"),
    ("zenith",                            "angle_of_incidence"),
    ("mean_sea_level_pressure_MSL",       "wind_speed_10_m_above_gnd"),
    ("temperature_2_m_above_gnd",         "shortwave_radiation_backwards_sfc"),
    ("relative_humidity_2_m_above_gnd",   "total_precipitation_sfc"),
    ("wind_speed_10_m_above_gnd",         "wind_speed_80_m_above_gnd"),
    ("angle_of_incidence",                "shortwave_radiation_backwards_sfc"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#080b12"
SURFACE  = "#0e1219"
CARD     = "#131929"
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
SIG_C    = "#4f8ef7"
NS_C     = "#1c2438"
CI_C     = "#f5a623"
CI_FILL  = "rgba(245,166,35,0.07)"

FONT_MONO = "'IBM Plex Mono','Fira Code',monospace"
FONT_SANS = "'IBM Plex Sans','Helvetica Neue',Arial,sans-serif"

_AX = dict(
    showgrid=True, gridcolor=GRID, gridwidth=1,
    zeroline=True, zerolinecolor=ZERO_L, zerolinewidth=1,
    linecolor=BORDER, linewidth=1, showline=True, mirror=False,
)

def _layout(**kw):
    base = dict(
        paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font=dict(family=FONT_MONO, color=TEXT, size=11),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER,
                        font=dict(size=12, color=TEXT)),
        showlegend=False,
        margin=dict(t=48, b=48, l=60, r=20),
    )
    base.update(kw)
    return base

def pf(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v < 0.0001:
        return "< 0.0001"
    return f"{v:.4f}"

def sf(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.4f}"

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & PRECOMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"WARNING: {CSV_PATH} not found — generating synthetic demo data")

    print("Loading data …")
    df = pd.read_csv(CSV_PATH)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=len(df)-1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df["date"] = date_range
    df = df.set_index("date")
    return df

print("Initialising …")
_df_raw  = load_data()
AVAILABLE = [c for c in COLS if c in _df_raw.columns]
DF        = _df_raw[AVAILABLE].copy()
print(f"  {len(AVAILABLE)} columns, {len(DF):,} rows")

# ── ACF / PACF ──────────────────────────────────────────────────────────────
print("Computing ACF/PACF …")
ACF_STATS: dict = {}
for col in AVAILABLE:
    s  = DF[col].dropna().values
    cb = 1.96 / math.sqrt(len(s))
    av, _ = acf(s, nlags=NLAGS, fft=True, alpha=0.05)
    pv, _ = pacf(s, nlags=NLAGS, alpha=0.05)
    pv    = np.clip(pv, -1.5, 1.5).tolist()
    ACF_STATS[col] = dict(acf=av.tolist(), pacf=pv, conf=round(cb, 5))

# ── Stationarity ─────────────────────────────────────────────────────────────
print("Running stationarity tests …")
STAT_RES: dict = {}
for col in AVAILABLE:
    s = DF[col].dropna().values
    adf = adfuller(s, autolag="AIC")
    try:    kp  = kpss(s, regression="c", nlags="auto")
    except: kp  = (float("nan"), float("nan"), None, None)
    try:    pp  = PhillipsPerron(s); pp_s, pp_p = pp.stat, pp.pvalue
    except: pp_s, pp_p = float("nan"), float("nan")
    try:    za  = zivot_andrews(s, maxlag=6, regression="ct"); za_s, za_p = za[0], za[1]
    except: za_s, za_p = float("nan"), float("nan")
    STAT_RES[col] = dict(
        adf_stat=float(adf[0]), adf_p=float(adf[1]),
        adf_lags=int(adf[2]),   adf_nobs=int(adf[3]),
        kpss_stat=float(kp[0]), kpss_p=float(kp[1]),
        pp_stat=float(pp_s),    pp_p=float(pp_p),
        za_stat=float(za_s),    za_p=float(za_p),
        adf_ok=bool(adf[1] < THRES),
        kpss_ok=bool(kp[1] >= THRES),
        pp_ok=bool(pp_p < THRES),
        za_ok=bool(za_p < THRES),
    )
    print(f"  {LABELS.get(col, col):35s}  ADF p={pf(adf[1])}")

# ── Cointegration ────────────────────────────────────────────────────────────
print("Running cointegration tests …")
COINT_RES: dict = {}
for c1, c2 in COINT_PAIRS:
    if c1 not in DF.columns or c2 not in DF.columns:
        continue
    s1, s2 = DF[c1].dropna().values, DF[c2].dropna().values
    n  = min(len(s1), len(s2))
    eg = coint(s1[:n], s2[:n])
    key = f"{LABELS[c1]} ↔ {LABELS[c2]}"
    COINT_RES[key] = dict(
        c1=c1, c2=c2, label1=LABELS[c1], label2=LABELS[c2],
        stat=float(eg[0]), p=float(eg[1]),
        crit1=float(eg[2][0]), crit5=float(eg[2][1]), crit10=float(eg[2][2]),
        sig=bool(eg[1] < THRES),
    )
    print(f"  {key[:55]:55s}  p={pf(eg[1])}")

# ── Granger ──────────────────────────────────────────────────────────────────
print("Running Granger causality tests …")
GRANGER_RES: dict = {}
_df_diff = DF.diff().dropna()
for cause, effect in GRANGER_PAIRS:
    if cause not in DF.columns or effect not in DF.columns:
        continue
    gc   = grangercausalitytests(_df_diff[[effect, cause]].values,
                                  maxlag=MAX_LAG_GC, verbose=False)
    pvals = {lag: gc[lag][0]["ssr_ftest"][1] for lag in gc}
    min_p = min(pvals.values())
    best  = min(pvals, key=pvals.get)
    key   = f"{LABELS[cause]} → {LABELS[effect]}"
    GRANGER_RES[key] = dict(
        cause=cause, effect=effect,
        label_c=LABELS[cause], label_e=LABELS[effect],
        min_p=float(min_p), best_lag=int(best),
        all_p={str(k): float(v) for k, v in pvals.items()},
        sig=bool(min_p < THRES),
    )
    print(f"  {key[:60]:60s}  p={pf(min_p)}  lag={best}")

print("All tests done ✔")

n_stat = sum(1 for c in AVAILABLE if STAT_RES[c]["adf_ok"] or STAT_RES[c]["pp_ok"])
n_co   = sum(1 for v in COINT_RES.values()   if v["sig"])
n_gc   = sum(1 for v in GRANGER_RES.values() if v["sig"])

# ── Diagnostics: KS, Shapiro-Wilk, Jarque-Bera, Ljung-Box, Hannan-Quinn ────
print("Running diagnostics tests …")
DIAG_RES: dict = {}
LB_LAGS = [6, 12, 24]   # lags for Ljung-Box
HQC_MAX_LAG = 12         # max AR order for HQC search

for col in AVAILABLE:
    s = DF[col].dropna().values
    n = len(s)

    # --- Kolmogorov-Smirnov (against N(μ,σ)) ---
    mu, sigma = float(np.mean(s)), float(np.std(s, ddof=1))
    ks_stat, ks_p = sp_stats.kstest(s, "norm", args=(mu, sigma))

    # --- Shapiro-Wilk (capped at 5000 for speed) ---
    sw_sample = s[:5000] if n > 5000 else s
    sw_stat, sw_p = sp_stats.shapiro(sw_sample)

    # --- Jarque-Bera ---
    jb_stat, jb_p = sp_stats.jarque_bera(s)
    skew  = float(sp_stats.skew(s))
    kurt  = float(sp_stats.kurtosis(s))          # excess kurtosis

    # --- Ljung-Box (at lags 6, 12, 24) ---
    lb_df  = acorr_ljungbox(s, lags=LB_LAGS, return_df=True)
    lb_stats = {lag: float(lb_df.loc[lag, "lb_stat"])  for lag in LB_LAGS}
    lb_pvals = {lag: float(lb_df.loc[lag, "lb_pvalue"]) for lag in LB_LAGS}
    lb_min_p  = min(lb_pvals.values())
    lb_best_l = min(lb_pvals, key=lb_pvals.get)

    # --- Hannan-Quinn Criterion (AR model order selection) ---
    best_hqc_val = np.inf; best_hqc_lag = 1; hqc_by_lag = {}
    for p in range(1, HQC_MAX_LAG + 1):
        try:
            mod = AutoReg(s, lags=p, old_names=False).fit()
            k   = p + 1  # AR coefficients + intercept
            hqc = -2 * mod.llf + 2 * k * np.log(np.log(n))
            hqc_by_lag[p] = float(hqc)
            if hqc < best_hqc_val:
                best_hqc_val = hqc; best_hqc_lag = p
        except Exception:
            hqc_by_lag[p] = float("nan")

    DIAG_RES[col] = dict(
        # KS
        ks_stat=float(ks_stat), ks_p=float(ks_p), ks_ok=bool(ks_p >= THRES),
        mu=mu, sigma=sigma,
        # SW
        sw_stat=float(sw_stat), sw_p=float(sw_p), sw_ok=bool(sw_p >= THRES),
        sw_n=int(len(sw_sample)),
        # JB
        jb_stat=float(jb_stat), jb_p=float(jb_p), jb_ok=bool(jb_p >= THRES),
        skew=skew, kurt=kurt,
        # LB
        lb_stats=lb_stats, lb_pvals=lb_pvals,
        lb_min_p=lb_min_p, lb_best_lag=lb_best_l,
        lb_ok=bool(lb_min_p >= THRES),      # fail-to-reject → no autocorrelation
        # HQC
        hqc_by_lag=hqc_by_lag,
        best_hqc_val=float(best_hqc_val),
        best_hqc_lag=int(best_hqc_lag),
    )
    print(f"  {LABELS.get(col, col):35s}  KS p={pf(ks_p)}  SW p={pf(sw_p)}  JB p={pf(jb_p)}  LB_min_p={pf(lb_min_p)}  HQC_lag={best_hqc_lag}")

print("Diagnostics done ✔")

n_normal_ks = sum(1 for c in AVAILABLE if DIAG_RES[c]["ks_ok"])
n_normal_sw = sum(1 for c in AVAILABLE if DIAG_RES[c]["sw_ok"])
n_normal_jb = sum(1 for c in AVAILABLE if DIAG_RES[c]["jb_ok"])
n_white_lb  = sum(1 for c in AVAILABLE if DIAG_RES[c]["lb_ok"])

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _stem_traces(vals, conf, col, kind):
    lags = list(range(0, NLAGS + 1))
    v    = list(vals[:NLAGS + 1])
    sig  = [abs(x) > conf for x in v]
    out  = []
    tag  = f"{col}__{kind}"

    # CI band fill
    out.append(go.Scatter(
        x=lags + lags[::-1],
        y=[conf]*len(lags) + [-conf]*len(lags),
        fill="toself", fillcolor=CI_FILL,
        line=dict(width=0), mode="lines",
        hoverinfo="skip", showlegend=False, name=f"{tag}__band",
    ))
    # CI dashed lines
    for sign in (1, -1):
        out.append(go.Scatter(
            x=lags, y=[sign * conf] * len(lags),
            mode="lines", line=dict(color=CI_C, width=1.5, dash="dot"),
            hovertemplate=f"95% CI ±{conf:.4f}<extra></extra>",
            showlegend=False, name=f"{tag}__ci{sign}",
        ))
    # zero line
    out.append(go.Scatter(
        x=[0, NLAGS], y=[0, 0],
        mode="lines", line=dict(color=ZERO_L, width=1),
        hoverinfo="skip", showlegend=False, name=f"{tag}__zero",
    ))
    # stems
    for lag in range(1, NLAGS + 1):
        val = float(v[lag]) if lag < len(v) else 0.0
        clr = SIG_C if sig[lag] else NS_C
        out.append(go.Scatter(
            x=[lag, lag], y=[0, val],
            mode="lines+markers",
            line=dict(color=clr, width=3.5),
            marker=dict(color=[clr, clr], size=[0, 7], symbol="circle"),
            hovertemplate=f"lag {lag}: {val:.4f}<extra></extra>",
            showlegend=False, name=f"{tag}__s{lag}",
        ))
    return out


def fig_acf_pacf(col: str, max_lag: int = NLAGS) -> go.Figure:
    d    = ACF_STATS[col]
    conf = d["conf"]
    fig  = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                         subplot_titles=["", ""])

    for tr in _stem_traces(d["acf"],  conf, col, "acf"):
        fig.add_trace(tr, row=1, col=1)
    for tr in _stem_traces(d["pacf"], conf, col, "pacf"):
        fig.add_trace(tr, row=1, col=2)

    grp, gc = GROUPS.get(col, ("", MUTED2))
    na  = sum(1 for v in d["acf"][1:] if abs(v) > conf)
    np_ = sum(1 for v in d["pacf"][1:] if abs(v) > conf)
    fn  = next((i+1 for i, v in enumerate(d["acf"][1:]) if abs(v) <= conf), NLAGS+1)

    ax_kw = dict(**_AX,
                 tickfont=dict(size=10, color=MUTED, family=FONT_MONO),
                 title_font=dict(size=11, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, title=dict(text="lag (hours)", standoff=4),
                     range=[-0.5, max_lag + 0.5])
    fig.update_yaxes(**ax_kw, title=dict(text="correlation", standoff=4),
                     range=[-1.15, 1.15], dtick=0.25, tickformat=".2f")

    fig.update_layout(
        **_layout(height=360, margin=dict(t=72, b=52, l=58, r=16)),
        annotations=[
            dict(x=0.21, y=1.10, xref="paper", yref="paper",
                 text="<b>Autocorrelation Function (ACF)</b>",
                 showarrow=False, font=dict(size=13, color=TEXT), xanchor="center"),
            dict(x=0.79, y=1.10, xref="paper", yref="paper",
                 text="<b>Partial Autocorrelation Function (PACF)</b>",
                 showarrow=False, font=dict(size=13, color=TEXT), xanchor="center"),
            dict(x=0.5, y=1.21, xref="paper", yref="paper",
                 text=(
                     f"<span style='color:{gc}'>[{grp}]</span>  "
                     f"sig ACF: <b>{na}</b>/{NLAGS}  ·  "
                     f"sig PACF: <b>{np_}</b>/{NLAGS}  ·  "
                     f"first non-sig ACF lag: <b>{fn}</b>  ·  "
                     f"CI: <b>±{conf:.4f}</b>"
                 ),
                 showarrow=False, font=dict(size=11, color=MUTED2), xanchor="center"),
        ],
    )
    return fig


def fig_ts_panel(col: str) -> go.Figure:
    clr  = GROUPS.get(col, ("", MUTED2))[1]
    s    = DF[col].values
    step = max(1, len(s) // TS_POINTS)
    x    = [str(t) for t in DF.index[::step]]
    y    = s[::step].tolist()

    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=clr, width=1.4),
        hovertemplate=f"{LABELS.get(col, col)}: %{{y:.3f}}<extra></extra>",
    ))
    ax_kw = dict(**_AX,
                 tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
                 title_font=dict(size=10, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw)
    fig.update_yaxes(**ax_kw, title=dict(text=LABELS.get(col, col), standoff=4))
    fig.update_layout(**_layout(height=240, margin=dict(t=28, b=36, l=64, r=12)))
    return fig


def fig_station_heatmap() -> go.Figure:
    tests   = ["ADF", "KPSS", "PP", "ZA"]
    ok_keys = ["adf_ok", "kpss_ok", "pp_ok", "za_ok"]
    ylabels = [LABELS.get(c, c) for c in AVAILABLE]
    z, texts = [], []
    for ok_key in ok_keys:
        rz, rt = [], []
        for col in AVAILABLE:
            v = STAT_RES[col][ok_key]
            rz.append(1 if v else 0)
            rt.append("Stationary ✔" if v else "Non-stat ✘")
        z.append(rz); texts.append(rt)

    fig = go.Figure(go.Heatmap(
        z=z, x=ylabels, y=tests,
        colorscale=[[0, "rgba(244,115,110,0.5)"], [1, "rgba(45,212,160,0.5)"]],
        zmin=0, zmax=1,
        text=texts, texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT),
        hovertemplate="<b>%{x}</b><br>%{y}: %{text}<extra></extra>",
        showscale=False, xgap=3, ygap=3,
    ))
    fig.update_xaxes(tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
                     showgrid=False, linecolor=BORDER, tickangle=35)
    fig.update_yaxes(tickfont=dict(size=11, color=MUTED2, family=FONT_MONO),
                     showgrid=False, linecolor=BORDER)
    fig.update_layout(**_layout(
        height=240,
        margin=dict(t=40, b=110, l=68, r=16),
        title=dict(text="Test Outcome Heatmap  ·  green = stationary at 5%",
                   font=dict(size=13, color=TEXT), x=0.01),
    ))
    return fig


def fig_station_bar(col: str) -> go.Figure:
    t      = STAT_RES[col]
    tests  = ["ADF", "KPSS", "PP", "ZA"]
    pvals  = [t["adf_p"], t["kpss_p"], t["pp_p"], t["za_p"]]
    ok     = [t["adf_ok"], t["kpss_ok"], t["pp_ok"], t["za_ok"]]
    colors = [GREEN if o else RED for o in ok]
    yvals  = [-math.log10(max(p, 1e-10)) if not math.isnan(p) else 0 for p in pvals]
    hover  = [
        f"<b>{tn}</b><br>Statistic: {sf(s)}<br>p-value: {pf(p)}<br>"
        f"Decision: {'Stationary ✔' if o else 'Non-stationary ✘'}<extra></extra>"
        for tn, s, p, o in zip(tests,
            [t["adf_stat"], t["kpss_stat"], t["pp_stat"], t["za_stat"]],
            pvals, ok)
    ]
    fig = go.Figure(go.Bar(
        x=tests, y=yvals,
        marker_color=colors, marker_line=dict(color=BORDER, width=1),
        hovertemplate=hover,
        text=[f"p={pf(p)}" for p in pvals],
        textposition="outside",
        textfont=dict(size=11, color=MUTED2, family=FONT_MONO),
    ))
    fig.add_hline(y=1.301, line=dict(color=AMBER, width=1.5, dash="dot"),
                  annotation_text="  α = 0.05",
                  annotation_font=dict(color=AMBER, size=10))
    ax_kw = dict(**_AX, tickfont=dict(size=12, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw)
    fig.update_yaxes(**ax_kw,
                     title=dict(text="−log₁₀(p)  ·  higher = more significant", standoff=6))
    fig.update_layout(**_layout(
        height=340,
        margin=dict(t=36, b=48, l=70, r=16),
        title=dict(text=f"p-value Significance  ·  {LABELS.get(col, col)}",
                   font=dict(size=13, color=TEXT), x=0.01),
        bargap=0.35,
    ))
    return fig


def fig_coint_bars() -> go.Figure:
    if not COINT_RES:
        return go.Figure()
    keys    = list(COINT_RES.keys())
    stats_  = [COINT_RES[k]["stat"] for k in keys]
    pvals   = [COINT_RES[k]["p"]    for k in keys]
    crits5  = [COINT_RES[k]["crit5"] for k in keys]
    sigs    = [COINT_RES[k]["sig"]   for k in keys]
    colors  = [GREEN if s else RED for s in sigs]
    labels  = [
        f"{COINT_RES[k]['label1']}<br><span style='opacity:.7'>↔</span> {COINT_RES[k]['label2']}"
        for k in keys
    ]
    hover   = [
        f"<b>{k}</b><br>EG stat: {sf(s)}<br>p={pf(p)}<br>5% crit: {sf(c)}<br>"
        f"{'Cointegrated ✔' if sg else 'Not cointegrated ✘'}<extra></extra>"
        for k, s, p, c, sg in zip(keys, stats_, pvals, crits5, sigs)
    ]
    idx = sorted(range(len(keys)), key=lambda i: stats_[i])
    fig = go.Figure(go.Bar(
        x=[stats_[i] for i in idx], y=[labels[i] for i in idx],
        orientation="h",
        marker_color=[colors[i] for i in idx],
        marker_line=dict(color=BORDER, width=1),
        hovertemplate=[hover[i] for i in idx],
        text=[f"p={pf(pvals[i])}" for i in idx],
        textposition="outside",
        textfont=dict(size=10, color=MUTED2, family=FONT_MONO),
        showlegend=False,
    ))
    avg_c5 = sum(crits5) / len(crits5)
    fig.add_vline(x=avg_c5, line=dict(color=AMBER, width=1.5, dash="dot"),
                  annotation_text="  avg 5% critical",
                  annotation_font=dict(color=AMBER, size=10))
    ax_kw = dict(**_AX, tickfont=dict(size=10, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, title=dict(
        text="EG test statistic  (more negative = stronger cointegration)", standoff=6))
    fig.update_yaxes(**ax_kw)
    fig.update_layout(**_layout(
        height=500, margin=dict(t=46, b=50, l=280, r=90),
        title=dict(text="Engle-Granger Cointegration  ·  Test Statistics",
                   font=dict(size=14, color=TEXT), x=0.01),
        bargap=0.28,
    ))
    return fig


def fig_coint_pvals() -> go.Figure:
    if not COINT_RES:
        return go.Figure()
    keys   = list(COINT_RES.keys())
    pvals  = [COINT_RES[k]["p"]   for k in keys]
    sigs   = [COINT_RES[k]["sig"] for k in keys]
    labels = [f"{COINT_RES[k]['label1']} ↔ {COINT_RES[k]['label2']}" for k in keys]
    colors = [GREEN if s else RED for s in sigs]
    fig = go.Figure(go.Scatter(
        x=list(range(len(keys))), y=pvals,
        mode="markers+text",
        marker=dict(color=colors, size=13, line=dict(color=BORDER, width=1)),
        text=[f"p={pf(p)}" for p in pvals],
        textposition="top center",
        textfont=dict(size=9, color=MUTED2, family=FONT_MONO),
        hovertext=labels,
        hovertemplate="%{hovertext}<br>p=%{y:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0.05, line=dict(color=AMBER, width=1.5, dash="dot"),
                  annotation_text="  α = 0.05",
                  annotation_font=dict(color=AMBER, size=10))
    ax_kw = dict(**_AX, tickfont=dict(size=10, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw,
                     tickvals=list(range(len(keys))),
                     ticktext=[str(i+1) for i in range(len(keys))],
                     title=dict(text="pair index", standoff=5))
    fig.update_yaxes(**ax_kw, title=dict(text="p-value", standoff=5),
                     range=[-0.05, 1.05])
    fig.update_layout(**_layout(
        height=320, margin=dict(t=40, b=50, l=60, r=20),
        title=dict(text="p-values  ·  green = cointegrated at 5%",
                   font=dict(size=13, color=TEXT), x=0.01),
    ))
    return fig


def fig_granger_heatmap() -> go.Figure:
    if not GRANGER_RES:
        return go.Figure()
    causes  = sorted(set(v["label_c"] for v in GRANGER_RES.values()))
    effects = sorted(set(v["label_e"] for v in GRANGER_RES.values()))
    z       = [[None]*len(effects) for _ in range(len(causes))]
    txts    = [["—"]*len(effects)  for _ in range(len(causes))]
    for v in GRANGER_RES.values():
        ri = causes.index(v["label_c"])
        ci = effects.index(v["label_e"])
        z[ri][ci]    = -math.log10(max(v["min_p"], 1e-15))
        txts[ri][ci] = f"p={pf(v['min_p'])}<br>lag={v['best_lag']}"
    fig = go.Figure(go.Heatmap(
        z=z, x=effects, y=causes,
        colorscale=[
            [0.0, "rgba(26,30,45,0.9)"],
            [0.3, "rgba(79,142,247,0.4)"],
            [0.7, "rgba(45,212,160,0.6)"],
            [1.0, "rgba(45,212,160,0.95)"],
        ],
        text=txts, texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT),
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>−log₁₀(p)=%{z:.2f}<br>%{text}<extra></extra>",
        colorbar=dict(
            title=dict(text="−log₁₀(p)", side="right",
                       font=dict(size=11, color=MUTED2, family=FONT_MONO)),
            tickfont=dict(size=10, color=MUTED2, family=FONT_MONO),
            bgcolor=CARD, bordercolor=BORDER, borderwidth=1, thickness=14,
        ),
        xgap=4, ygap=4,
    ))
    ax_kw = dict(**_AX, tickfont=dict(size=9, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, tickangle=35)
    fig.update_yaxes(**ax_kw)
    fig.update_layout(**_layout(
        height=460, margin=dict(t=50, b=140, l=220, r=80),
        title=dict(text="Granger Causality Matrix  ·  −log₁₀(p)",
                   font=dict(size=14, color=TEXT), x=0.01),
    ))
    return fig


def fig_granger_bar() -> go.Figure:
    if not GRANGER_RES:
        return go.Figure()
    keys   = list(GRANGER_RES.keys())
    pvals  = [GRANGER_RES[k]["min_p"]   for k in keys]
    lags   = [GRANGER_RES[k]["best_lag"] for k in keys]
    sigs   = [GRANGER_RES[k]["sig"]      for k in keys]
    colors = [GREEN if s else RED for s in sigs]
    hover  = [
        f"<b>{k}</b><br>min p={pf(p)}<br>best lag={l} h<br>"
        f"{'Granger-causes ✔' if s else 'No causality ✘'}<extra></extra>"
        for k, p, l, s in zip(keys, pvals, lags, sigs)
    ]
    idx = sorted(range(len(keys)), key=lambda i: pvals[i])
    fig = go.Figure(go.Bar(
        x=[-math.log10(max(pvals[i], 1e-15)) for i in idx],
        y=[keys[i] for i in idx],
        orientation="h",
        marker_color=[colors[i] for i in idx],
        marker_line=dict(color=BORDER, width=1),
        hovertemplate=[hover[i] for i in idx],
        text=[f"lag={lags[i]}" for i in idx],
        textposition="outside",
        textfont=dict(size=10, color=MUTED2, family=FONT_MONO),
        showlegend=False,
    ))
    fig.add_vline(x=-math.log10(0.05), line=dict(color=AMBER, width=1.5, dash="dot"),
                  annotation_text="  α=0.05",
                  annotation_font=dict(color=AMBER, size=10))
    ax_kw = dict(**_AX, tickfont=dict(size=10, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, title=dict(
        text="-log₁₀(p)  ·  higher = more significant", standoff=6))
    fig.update_yaxes(**ax_kw)
    fig.update_layout(**_layout(
        height=500, margin=dict(t=46, b=50, l=310, r=80),
        title=dict(text="Granger Causality  ·  Significance ranking (best lag among 1–6 h)",
                   font=dict(size=14, color=TEXT), x=0.01),
        bargap=0.3,
    ))
    return fig



# ─────────────────────────────────────────────────────────────────────────────
#  DIAGNOSTIC FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_diag_overview() -> go.Figure:
    """Heatmap: all 5 diagnostics × all variables — pass/fail overview."""
    tests   = ["KS", "Shapiro-Wilk", "Jarque-Bera", "Ljung-Box (white)", "HQC best lag"]
    ok_keys = ["ks_ok", "sw_ok", "jb_ok", "lb_ok"]
    ylabels = [LABELS.get(c, c) for c in AVAILABLE]
    z, texts = [], []

    for ok_key in ok_keys:
        rz, rt = [], []
        for col in AVAILABLE:
            v = DIAG_RES[col][ok_key]
            rz.append(1 if v else 0)
            rt.append("✔ Pass" if v else "✘ Fail")
        z.append(rz); texts.append(rt)

    # HQC best lag row (numeric, not pass/fail — encode 1..12 → 0..1)
    hqc_vals = [DIAG_RES[c]["best_hqc_lag"] for c in AVAILABLE]
    norm_hqc = [(v - 1) / (HQC_MAX_LAG - 1) for v in hqc_vals]
    z.append(norm_hqc)
    texts.append([f"lag {v}" for v in hqc_vals])

    fig = go.Figure(go.Heatmap(
        z=z, x=ylabels, y=tests,
        colorscale=[
            [0.0, "rgba(244,115,110,0.55)"],
            [0.5, "rgba(245,166,35,0.45)"],
            [1.0, "rgba(45,212,160,0.55)"],
        ],
        zmin=0, zmax=1,
        text=texts, texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT),
        hovertemplate="<b>%{x}</b><br>%{y}: %{text}<extra></extra>",
        showscale=False, xgap=3, ygap=3,
    ))
    fig.update_xaxes(tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
                     showgrid=False, linecolor=BORDER, tickangle=35)
    fig.update_yaxes(tickfont=dict(size=11, color=MUTED2, family=FONT_MONO),
                     showgrid=False, linecolor=BORDER)
    fig.update_layout(**_layout(
        height=270,
        margin=dict(t=44, b=120, l=140, r=16),
        title=dict(text="Diagnostic Overview  ·  KS / SW / JB: green = normal / white-noise at 5%  ·  HQC: optimal AR lag",
                   font=dict(size=12, color=TEXT), x=0.01),
    ))
    return fig

## FIX THIS ISSUE: IT GIVES BACK MORE THAN A SINGLE DOT FOR EACH VARIABLE, PROBABLY BECAUSE OF THE WAY THE DATA IS STRUCTURED. NEED TO CHECK THE DATA STRUCTURE AND ENSURE THAT ONLY ONE DOT PER VARIABLE IS PLOTTED, REPRESENTING THE KS STATISTIC, SW STATISTIC, SKEWNESS, KURTOSIS, AND JB RESULT FOR THAT VARIABLE.
def fig_diag_normality_scatter() -> go.Figure:
    """Bubble chart: KS stat vs SW stat, bubble size = |skew|, color = JB pass/fail."""
    cols   = AVAILABLE
    ks_s   = [DIAG_RES[c]["ks_stat"] for c in cols]
    sw_s   = [1 - DIAG_RES[c]["sw_stat"] for c in cols]   # invert so higher = less normal
    skews  = [abs(DIAG_RES[c]["skew"]) for c in cols]
    jb_ok  = [DIAG_RES[c]["jb_ok"] for c in cols]
    colors = [GREEN if o else RED for o in jb_ok]
    names  = [LABELS.get(c, c) for c in cols]
    hover  = [
        f"<b>{names[i]}</b><br>"
        f"KS stat: {sf(ks_s[i])}<br>"
        f"1−SW stat: {sf(sw_s[i])}<br>"
        f"Skewness: {DIAG_RES[cols[i]]['skew']:.3f}<br>"
        f"Excess kurtosis: {DIAG_RES[cols[i]]['kurt']:.3f}<br>"
        f"JB: {'Normal ✔' if jb_ok[i] else 'Non-normal ✘'}<extra></extra>"
        for i in range(len(cols))
    ]
    fig = go.Figure(go.Scatter(
        x=ks_s, y=sw_s,
        mode="markers+text",
        marker=dict(
            color=colors, size=[10 + 20*min(s, 3)/3 for s in skews],
            line=dict(color=BORDER, width=1), opacity=0.85,
        ),
        text=[LABELS.get(c, c).split(" ")[0] for c in cols],
        textfont=dict(size=8, color=MUTED2),
        textposition="top center",
        hovertemplate=hover,
        showlegend=False,
    ))
    ax_kw = dict(**_AX, tickfont=dict(size=10, color=MUTED, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, title=dict(text="KS statistic  (higher = further from normal)", standoff=5))
    fig.update_yaxes(**ax_kw, title=dict(text="1 − SW statistic  (higher = further from normal)", standoff=5))
    fig.update_layout(**_layout(
        height=360, margin=dict(t=44, b=52, l=72, r=20),
        title=dict(text="Normality Landscape  ·  bubble size = |skewness|  ·  color = JB result",
                   font=dict(size=13, color=TEXT), x=0.01),
    ))
    return fig


def fig_diag_variable(col: str) -> go.Figure:
    """5-panel figure for one variable: histogram+QQ, LB p-values, HQC curve."""
    d   = DIAG_RES[col]
    s   = DF[col].dropna().values
    clr = GROUPS.get(col, ("", ACCENT))[1]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Histogram vs Normal", "Q-Q Plot", "Skewness & Kurtosis",
            "Ljung-Box p-values by lag", "HQC by AR order", "Test Summary",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "table"}],
        ],
    )

    # ── (1,1) Histogram ───────────────────────────────────────────────────
    counts, bin_edges = np.histogram(s, bins=60, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig.add_trace(go.Bar(
        x=bin_centers, y=counts,
        marker_color=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.55)",
        marker_line=dict(width=0),
        name="Histogram", showlegend=False,
        hovertemplate="bin: %{x:.3f}<br>density: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    # Normal overlay
    xn = np.linspace(s.min(), s.max(), 200)
    yn = sp_stats.norm.pdf(xn, d["mu"], d["sigma"])
    fig.add_trace(go.Scatter(
        x=xn, y=yn, mode="lines",
        line=dict(color=AMBER, width=2, dash="dot"),
        name="Normal PDF", showlegend=False,
        hovertemplate="x: %{x:.3f}<br>N(μ,σ): %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # ── (1,2) Q-Q plot ────────────────────────────────────────────────────
    qq = sp_stats.probplot(s, dist="norm")
    theoretical_q, ordered_v = qq[0]
    slope, intercept, _ = qq[1]
    fig.add_trace(go.Scatter(
        x=theoretical_q, y=ordered_v, mode="markers",
        marker=dict(color=clr, size=3, opacity=0.5),
        name="Q-Q", showlegend=False,
        hovertemplate="theoretical: %{x:.3f}<br>sample: %{y:.3f}<extra></extra>",
    ), row=1, col=2)
    xq = np.array([theoretical_q[0], theoretical_q[-1]])
    fig.add_trace(go.Scatter(
        x=xq, y=slope * xq + intercept, mode="lines",
        line=dict(color=AMBER, width=1.5, dash="dot"),
        showlegend=False,
    ), row=1, col=2)

    # ── (1,3) Skewness / Kurtosis bar ────────────────────────────────────
    sk_val = d["skew"]; ku_val = d["kurt"]
    fig.add_trace(go.Bar(
        x=["Skewness", "Excess Kurtosis"],
        y=[sk_val, ku_val],
        marker_color=[GREEN if abs(sk_val) < 0.5 else RED,
                      GREEN if abs(ku_val) < 1.0 else RED],
        marker_line=dict(color=BORDER, width=1),
        showlegend=False,
        text=[f"{sk_val:.3f}", f"{ku_val:.3f}"],
        textposition="outside",
        textfont=dict(size=11, color=MUTED2, family=FONT_MONO),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ), row=1, col=3)
    fig.add_hline(y=0, line=dict(color=ZERO_L, width=1), row=1, col=3)

    # ── (2,1) Ljung-Box p-values ──────────────────────────────────────────
    lb_lags = list(d["lb_pvals"].keys())
    lb_ps   = list(d["lb_pvals"].values())
    lb_clrs = [GREEN if p >= THRES else RED for p in lb_ps]
    fig.add_trace(go.Bar(
        x=[str(l) for l in lb_lags], y=lb_ps,
        marker_color=lb_clrs, marker_line=dict(color=BORDER, width=1),
        showlegend=False,
        text=[pf(p) for p in lb_ps],
        textposition="outside",
        textfont=dict(size=10, color=MUTED2, family=FONT_MONO),
        hovertemplate="lag %{x}: p=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0.05, line=dict(color=AMBER, width=1.5, dash="dot"), row=2, col=1)

    # ── (2,2) HQC by AR order ─────────────────────────────────────────────
    hq_lags = list(d["hqc_by_lag"].keys())
    hq_vals = list(d["hqc_by_lag"].values())
    clrs_hq = [GREEN if l == d["best_hqc_lag"] else MUTED for l in hq_lags]
    fig.add_trace(go.Scatter(
        x=hq_lags, y=hq_vals, mode="lines+markers",
        line=dict(color=MUTED2, width=1.5),
        marker=dict(color=clrs_hq, size=8, line=dict(color=BORDER, width=1)),
        showlegend=False,
        hovertemplate="AR(%{x}): HQC=%{y:.2f}<extra></extra>",
    ), row=2, col=2)

    # ── (2,3) Test summary table ──────────────────────────────────────────
    tests_nm  = ["KS Test",    "Shapiro-Wilk", "Jarque-Bera", "Ljung-Box", "HQC best lag"]
    stats_v   = [sf(d["ks_stat"]), sf(d["sw_stat"]), sf(d["jb_stat"]),
                 pf(d["lb_min_p"]), str(d["best_hqc_lag"])]
    pvals_v   = [pf(d["ks_p"]),   pf(d["sw_p"]),  pf(d["jb_p"]),
                 pf(d["lb_min_p"]), "—"]
    verdict_v = [
        "Normal ✔"    if d["ks_ok"] else "Non-normal ✘",
        "Normal ✔"    if d["sw_ok"] else "Non-normal ✘",
        "Normal ✔"    if d["jb_ok"] else "Non-normal ✘",
        "White noise ✔" if d["lb_ok"] else "Autocorr. ✘",
        f"AR({d['best_hqc_lag']})",
    ]
    v_colors = [
        [GREEN if d["ks_ok"]  else RED,
         GREEN if d["sw_ok"]  else RED,
         GREEN if d["jb_ok"]  else RED,
         GREEN if d["lb_ok"]  else RED,
         ACCENT2],
    ]
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Test</b>", "<b>Statistic</b>", "<b>p-value</b>", "<b>Verdict</b>"],
            fill_color=CARD2, font=dict(color=TEXT, size=11, family=FONT_MONO),
            align="left", line_color=BORDER, height=26,
        ),
        cells=dict(
            values=[tests_nm, stats_v, pvals_v, verdict_v],
            fill_color=[CARD, CARD, CARD, v_colors[0]],
            font=dict(color=[TEXT, MUTED2, MUTED2, TEXT], size=10, family=FONT_MONO),
            align="left", line_color=BORDER, height=24,
        ),
    ), row=2, col=3)

    # ── Axes styling ──────────────────────────────────────────────────────
    ax_kw = dict(**_AX, tickfont=dict(size=9, color=MUTED, family=FONT_MONO))
    for r, c in [(1,1),(1,2),(1,3),(2,1),(2,2)]:
        fig.update_xaxes(**ax_kw, row=r, col=c)
        fig.update_yaxes(**ax_kw, row=r, col=c)

    fig.update_xaxes(title=dict(text="Value", standoff=3),  row=1, col=1)
    fig.update_yaxes(title=dict(text="Density", standoff=3), row=1, col=1)
    fig.update_xaxes(title=dict(text="Theoretical quantiles", standoff=3), row=1, col=2)
    fig.update_yaxes(title=dict(text="Sample quantiles", standoff=3), row=1, col=2)
    fig.update_xaxes(title=dict(text="Lag", standoff=3), row=2, col=1)
    fig.update_yaxes(title=dict(text="p-value", standoff=3), row=2, col=1)
    fig.update_xaxes(title=dict(text="AR order p", standoff=3), row=2, col=2)
    fig.update_yaxes(title=dict(text="HQC", standoff=3), row=2, col=2)

    grp, gc = GROUPS.get(col, ("", MUTED2))
    fig.update_layout(
        **_layout(height=620, margin=dict(t=80, b=32, l=60, r=20)),
        title=dict(
            text=f"<span style='color:{gc}'>[{grp}]</span>  <b>{LABELS.get(col, col)}</b>  ·  Diagnostic Panel",
            font=dict(size=14, color=TEXT), x=0.01,
        ),
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=MUTED2, family=FONT_MONO)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  DIAGNOSTICS THEORY CONTENT
# ─────────────────────────────────────────────────────────────────────────────

def _latex_card(title, tag, color, plain_desc, formula_label, formula_latex,
                h0_txt, h1_txt, plain_notes, interpretation, caveats=None):
    children = [
        html.H3([
            html.Span(title, style={"color": color}),
            html.Span(tag, className="tc-tag"),
        ])
    ]
    children.append(html.P(plain_desc, className="tc-plain"))
    children.append(html.Div(formula_label, className="tc-formula-label"))
    children.append(dcc.Markdown(formula_latex, mathjax=True, className="tc-latex"))
    children.append(html.Div([html.Span("H₀: ", style={"fontWeight":"700"}), h0_txt], className="h0-box"))
    children.append(html.Div([html.Span("H₁: ", style={"fontWeight":"700"}), h1_txt], className="h1-box"))
    if interpretation!="":
        children.append(html.Div([
            html.Span("📊 How to read: ", style={"fontWeight":"700", "color": ACCENT2}),
            interpretation,
        ], className="tc-interp"))
    if caveats:
        children.append(html.Div([
            html.Span("⚠ Caveats: ", style={"fontWeight":"700", "color": AMBER}),
            caveats,
        ], className="warn-box", style={"marginTop":"10px"}))
    return html.Div(children, className="theory-card")


DIAG_THEORY = html.Div([
    html.Div("🔬 Diagnostic Tests — Theory & Interpretation", className="theory-title"),

    # ── Intro paragraph ──
    html.P([
        "These five tests examine two fundamental properties of your time series residuals: ",
        html.B("normality"), " (are the values bell-curve distributed?) and ",
        html.B("autocorrelation"), " (do past values predict future values?). ",
        "They are run on the raw series, not on model residuals. ",
        "A failed normality test does not invalidate a model — many forecasting models are robust to non-normality — ",
        "but it informs which distribution family to use and flags outliers or heavy tails.",
    ], className="section-desc", style={"maxWidth":"100%","marginBottom":"20px"}),

    html.Div([
        # Row 1: KS + SW
        html.Div([
            _latex_card(
                "Kolmogorov-Smirnov Test", "KS", ACCENT2,
                "The KS test compares the empirical distribution function (EDF) of your data to a reference "
                "distribution, which is the Gaussian distribution fitted with the sample mean and standard deviation. "
                "The test statistic D is the maximum vertical gap between the data distribution and the aforementioned gaussian distribution. ",
                # "Think of it as asking: 'If I stacked all my data values in order, how far does that staircase "
                # "deviate from the smooth bell-curve ramp?' The test statistic D is the maximum vertical gap "
                # "between the two curves.",
                "Test statistic:",
                r"$$D_n = \sup_x \left| F_n(x) - F(x) \right|$$"
                "\n\nwhere $F_n(x)$ is the empirical CDF and $F(x)$ is the reference Normal CDF.",
                "The data are drawn from a Normal distribution.",
                "The data are NOT drawn from a Normal distribution.",
                "A large D (and small p-value) means the data deviate significantly from Normal. "
                "$p \geq 0.05$ means we cannot reject normality, the data are consistent with Normal.",
                "Useful for any sample size. Less powerful than Shapiro-Wilk for small samples ($n < 50$). ",
                # "For time series with diurnal patterns (e.g. solar radiation), non-normality is expected — "
                # "the bimodal day/night structure will always fail this test.",
                caveats="The KS test is general-purpose but less sensitive to deviations in the tails "
                        "compared to Anderson-Darling. For the solar dataset, variables with heavy zeros "
                        "(precipitation, snowfall) will fail by design.",
            ),
            _latex_card(
                "Shapiro-Wilk Test", "SW", GREEN,
                "The Shapiro-Wilk test measures how well the data fit a straight line on a Q-Q plot. "
                "A Q-Q plot plots your data's quantiles against what they would be if the data were perfectly Normal. "
                "If the points lie on a diagonal line, the data are Normal. The W statistic is the squared "
                "correlation between the sorted data and the expected Normal order statistics — W = 1 is perfect normality.",
                "Test statistic:",
                r"$$W = \frac{\left(\displaystyle\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}$$,"
                "$$"
                "\n\nwhere $x_{(i)}$ are the order statistics (sorted values), $\\bar{x}$ is the sample mean, "
                "and $a_i$ are coefficients derived from the expected Normal order statistics.",
                "The data come from a Normal distribution (W close to 1).",
                "The data do NOT come from a Normal distribution.",
                "W ranges from 0 to 1. Values close to 1 indicate normality. "
                "p < 0.05 rejects normality. "
                "For large samples (n > 5000), even tiny deviations become significant — "
                "inspect the Q-Q plot visually alongside the p-value.",
                "Considered the most powerful normality test for sample sizes up to ~5000. "
                "For larger datasets the test is run on the first 5000 observations.",
                caveats="Very large samples (n > 2000) will almost always reject normality even for "
                        "near-Normal data due to the test's sensitivity. Always inspect the Q-Q plot.",
            ),
        ], className="theory-row"),
        # Row 2: JB + LB
        html.Div([
            _latex_card(
                "Jarque-Bera Test", "JB", AMBER,
                "The Jarque-Bera test checks normality by measuring two specific features: "
                "skewness (how asymmetric the distribution is — Normal distributions are symmetric) "
                "and excess kurtosis (how heavy the tails are — Normal distributions have kurtosis = 3, "
                "so excess kurtosis = kurtosis − 3 = 0). "
                "If both are close to zero, the data look Normal from a moments perspective.",
                "Test statistic:",
                r"$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$"
                "\n\nwhere $n$ is the sample size, $S = \\hat{\\mu}_3/\\hat{\\sigma}^3$ is the sample skewness, "
                "and $K = \\hat{\\mu}_4/\\hat{\\sigma}^4$ is the sample kurtosis. "
                "Under normality, $JB \\sim \\chi^2(2)$.",
                "Skewness = 0 and excess kurtosis = 0 (data are Normal).",
                "Skewness ≠ 0 or excess kurtosis ≠ 0 (data deviate from Normal).",
                "JB = 0 means perfect normality. Larger JB means stronger evidence against normality. "
                "Rule of thumb: |skew| < 0.5 is approximately symmetric; |excess kurtosis| < 1 is near-Normal tails. "
                "Wind direction variables will show near-zero JB due to their circular but roughly symmetric nature.",
                "Asymptotically chi-squared — unreliable for small samples (n < 30) where Type I error rate is high. "
                "Best used for large samples alongside KS and SW.",
                caveats="The chi-squared approximation overstates significance for small samples. "
                        "For solar data with n >> 1000, this is not a concern.",
            ),
            _latex_card(
                "Ljung-Box Test", "LB", "#60a5fa",
                "The Ljung-Box test checks whether the autocorrelations of a series are jointly zero up to lag m. "
                "In plain terms: after fitting a model to your data, the residuals should be 'white noise' — "
                "meaning no value should be predictable from previous values. "
                "If autocorrelations are non-zero, the model has missed some structure in the data. "
                "The Q statistic aggregates the squared autocorrelations across all lags up to m.",
                "Test statistic:",
                r"$$Q = n(n+2)\sum_{k=1}^{m}\frac{\hat{r}_k^2}{n-k}$$"
                "\n\nwhere $\\hat{r}_k$ is the sample autocorrelation at lag $k$, $n$ is the sample size, "
                "and $m$ is the number of lags tested. "
                "Under the null, $Q \\sim \\chi^2(h)$ where $h = m - p - q$ for ARMA$(p,q)$ residuals.",
                "No autocorrelation up to lag m — the series behaves like white noise.",
                "Significant autocorrelation exists — the series (or model residuals) has exploitable structure.",
                "Reported here at lags 6, 12, and 24. "
                "Lag 24 is critical for meteorological data: a significant Q at lag 24 confirms strong diurnal periodicity. "
                "p < 0.05 means autocorrelation is present — this is expected for all raw series in this dataset. "
                "Green = white noise (no autocorrelation). Red = autocorrelation detected.",
                "Run on raw series here — not model residuals. "
                "All weather variables are expected to show significant autocorrelation (red) at lags 12 and 24 "
                "due to the solar diurnal cycle.",
                caveats="Degrees of freedom must be adjusted when applied to ARMA residuals: use h = m − p − q. "
                        "The test is applied to the raw series here, so treat red results as descriptive, "
                        "not as model failures.",
            ),
        ], className="theory-row"),
        # Row 3: HQC (single card, full width via solo-card class)
        html.Div([
            _latex_card(
                "Hannan-Quinn Criterion", "HQC", "#a78bfa",
                "The Hannan-Quinn Criterion selects the optimal order (number of lags) for an autoregressive (AR) model. "
                "Think of it as a 'complexity penalty' — adding more lags improves the fit, but HQC penalises "
                "each additional parameter to prevent overfitting. "
                "The best AR order is the one that minimises HQC. "
                "This tells you: 'how many past hours of data are needed to best predict the next hour?'",
                "Criterion:",
                r"$$HQC(k) = -2\ln\hat{\sigma}_k^2 + 2k\ln\ln(n)$$"
                "\n\nwhere $n$ is the sample size, $k$ is the number of AR parameters, "
                "and $\\hat{\\sigma}_k^2$ is the residual variance of the AR$(k)$ model. "
                "The double-logarithm penalty $\\ln\\ln(n)$ is weaker than BIC but stronger than AIC, "
                "yielding consistent order selection for large samples.",
                "The true AR order is 0 (no autoregressive structure).",
                "An AR order k ≥ 1 minimises HQC — the series has memory.",
                "The optimal lag displayed is the AR order p that minimises HQC across p = 1 to 12. "
                "Solar radiation and temperature typically show HQC-optimal AR(2) or AR(3), "
                "reflecting the strong diurnal carry-over. "
                "Precipitation and snowfall often yield AR(1) due to sparse event structure.",
                "HQC consistently selects a smaller order than AIC but typically larger than BIC. "
                "It is the recommended criterion when n is large and overfitting is a concern.",
                caveats="HQC is computed on the raw series, not first differences. "
                        "For non-stationary series, the selected lag order should be interpreted with caution — "
                        "apply ADF/PP first to confirm I(0) before interpreting the HQC-optimal AR order.",
            ),
            # Empty spacer card to maintain grid symmetry
            html.Div(style={"width": "50%"}),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),

    html.Div([
        html.B("📋 Recommended workflow: ", style={"color": GREEN}),
        "(1) Check normality with all three tests (KS, SW, JB) — agreement across tests gives confidence. "
        "(2) Inspect the Q-Q plot visually — a few outliers can dominate the KS/SW statistics. "
        "(3) Use Ljung-Box at lag 24 to confirm whether the diurnal cycle has been captured by your model. "
        "(4) Use the HQC-optimal lag as the starting point for AR/ARIMA order selection — "
        "then refine with AIC/BIC on the differenced series.",
    ], className="tip-box"),

], className="theory-section")

VAR_OPTIONS  = [{"label": LABELS.get(c, c), "value": c} for c in AVAILABLE]

PLOTLY_CFG = dict(
    displayModeBar=True, displaylogo=False,
    modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"],
    toImageButtonOptions=dict(format="png", filename="solar_chart", scale=2),
    responsive=True,
)

def section_hdr(text):
    return html.Div(text, className="section-hdr")

def section_desc(text):
    return html.Div(text, className="section-desc")

def plot_card(children, label=None):
    inner = []
    if label:
        inner.append(html.Div(label, className="plot-label"))
    inner.extend(children if isinstance(children, list) else [children])
    return html.Div(inner, className="plot-card")

def banner(*items):
    """items: list of (value, label, color)"""
    cards = [
        html.Div([
            html.Div(str(v), className="sbn-val", style={"color": c}),
            html.Div(l, className="sbn-lbl"),
        ], className="sbn")
        for v, l, c in items
    ]
    return html.Div(cards, className="summary-banner")

def insight_bar(col):
    grp, gc = GROUPS.get(col, ("", MUTED2))
    ins = INSIGHTS.get(col, "")
    return html.Div([
        html.Span(f"[{grp}]", style={"color": gc, "fontWeight": "700"}),
        html.Span(f"  {ins}"),
    ], className="insight-bar")

def legend_row():
    return html.Div([
        html.Div([
            html.Span(className="ldot", style={"background": SIG_C}),
            html.Span("Significant"),
        ], className="legend-item"),
        html.Div([
            html.Span(className="ldot", style={"background": NS_C, "border": "1px solid #334155"}),
            html.Span("Non-significant"),
        ], className="legend-item"),
        html.Div([
            html.Span("- -", style={"color": CI_C, "fontSize": "15px"}),
            html.Span(" 95% CI (±1.96/√n)"),
        ], className="legend-item"),
    ], className="legend-row")


# ─────────────────────────────────────────────────────────────────────────────
#  THEORY PANELS (HTML components)
# ─────────────────────────────────────────────────────────────────────────────
def theory_card(title, tag, color, paragraphs, math_eq=None, h0=None, h1=None):
    children = [
        html.H3([
            html.Span(title, style={"color": color}),
            html.Span(tag, className="tc-tag"),
        ])
    ]
    for p in paragraphs:
        if isinstance(p, str):
            children.append(html.P(p))
        else:
            children.append(html.Div(p, style={"marginBottom": "8px"}))
    if math_eq:
        children.append(html.Code(math_eq, className="math"))
    if h0:
        children.append(html.Div([html.Span("H₀: ", style={"fontWeight":"700"}), h0], className="h0-box"))
    if h1:
        children.append(html.Div([html.Span("H₁: ", style={"fontWeight":"700"}), h1], className="h1-box"))
    return html.Div(children, className="theory-card")


STATION_THEORY = html.Div([
    html.Div("📖 Unit Root & Stationarity — Theory", className="theory-title"),
    html.Div([
        html.Div([
            theory_card("Augmented Dickey-Fuller", "ADF", ACCENT2,
                ["The ADF test checks whether a time series has a 'unit root' — meaning it wanders randomly with no tendency to return to a mean. "
                 "A unit root implies the series is non-stationary: its statistical properties (mean, variance) change over time, "
                 "which invalidates most classical forecasting models.",
                 "The test augments the basic Dickey-Fuller regression with lagged differences of the series to absorb any "
                 "serial correlation in the error term. The lag count is chosen automatically by AIC.",
                 dcc.Markdown(r"Model: $\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p}\delta_i \Delta Y_{t-i} + \varepsilon_t$", mathjax=True)],
                None,
                "γ = 0 → unit root present (non-stationary)",
                "γ < 0 → series is stationary"),
            theory_card("KPSS Test", "KPSS", GREEN,
                ["The KPSS test takes the opposite stance from ADF: its null hypothesis is that the series IS stationary. "
                 "This makes it a powerful complement — if ADF rejects its null (evidence of stationarity) AND KPSS fails to "
                 "reject its null (evidence of stationarity), you have strong convergent evidence.",
                 "The test decomposes the series into a deterministic trend, a random walk, and a stationary error. "
                 "It then tests whether the variance of the random walk component equals zero.",
                 dcc.Markdown(r"Model: $Y_t = \xi_t + r_t + \varepsilon_t$, where $r_t = r_{t-1} + u_t$", mathjax=True),
                 "⚠ Note: KPSS has lower power as a unit-root test than ADF or PP. It was designed to test stationarity, "
                 "not the presence of a unit root. Schmidt & Shin (1992) caution against using it as a primary unit-root test — "
                 "use it alongside ADF and PP for a consistent picture."],
                None,
                "σ²ᵤ = 0 → series is stationary",
                "σ²ᵤ > 0 → unit root / non-stationary"),
        ], className="theory-row"),
        html.Div([
            theory_card("Phillips-Perron", "PP", AMBER,
                ["The PP test is a non-parametric extension of the Dickey-Fuller test. Rather than adding lagged differences "
                 "as regressors (as ADF does), it applies a direct correction to the test statistic using the Newey-West "
                 "long-run variance estimator. This makes it robust to serial correlation and heteroskedasticity in the errors "
                 "without requiring a correct lag specification.",
                 "PP is generally preferred over ADF when the error structure is unknown, but Davidson & MacKinnon (2004) "
                 "found it can underperform ADF in finite samples. Best used together with ADF.",
                 dcc.Markdown(r"Model: $\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \varepsilon_t$ (non-parametric correction applied to statistic)", mathjax=True)],
                None,
                "γ = 0 → unit root present",
                "γ < 0 → series is stationary"),
            theory_card("Zivot-Andrews", "ZA", "#a78bfa",
                ["The standard ADF and PP tests assume no structural breaks in the series. For meteorological data, "
                 "this assumption often fails — e.g. a sensor recalibration, a regime shift in cloud patterns, or a "
                 "seasonal discontinuity. The ZA test allows for one endogenously determined structural break.",
                 "The break point is chosen by minimising the ADF statistic across all possible break dates, making the "
                 "test conservative. If ZA rejects but ADF does not, a structural break is likely masking stationarity.",
                 dcc.Markdown(r"Model: $\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \delta DU_t + \theta DT_t + \sum c_i \Delta Y_{t-i} + \varepsilon_t$", mathjax=True)],
                None,
                "Unit root present (no structural break)",
                "Trend-stationary with one endogenous break"),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),
    html.Div([
        html.B("⚠ Interpretation note: ", style={"color": AMBER}),
        "No single test is definitive. Use ADF (H₀: unit root) and KPSS (H₀: stationary) together for robust conclusions. "
        "If ADF rejects AND KPSS fails to reject → strong stationarity evidence. "
        "For series with structural breaks (zenith, shortwave radiation) Zivot-Andrews is most reliable."
    ], className="warn-box"),
], className="theory-section")


COINT_THEORY = html.Div([
    html.Div("🔗 Cointegration — Theory", className="theory-title"),
    html.Div([
        html.Div([
            theory_card("Engle-Granger Two-Step", "EG", ACCENT2,
                ["Cointegration tests ask whether two non-stationary (I(1)) series share a long-run equilibrium. "
                 "Even though each series wanders individually, a linear combination of them may be stationary — "
                 "meaning deviations from their shared trend are temporary and mean-reverting.",
                 "The Engle-Granger procedure runs in two steps: first, regress Y on X by OLS to estimate the "
                 "long-run relationship. Second, apply an ADF test to the residuals. If the residuals are stationary, "
                 "the series are cointegrated.",
                 dcc.Markdown(r"Step 1: $Y_t = \alpha + \beta X_t + \varepsilon_t$ → Step 2: ADF test on $\hat{\varepsilon}_t$", mathjax=True)],
                None,
                "Residuals have a unit root → no cointegration",
                "Residuals are stationary → series are cointegrated"),
            theory_card("What Cointegration Means", "CI(1,1)", GREEN,
                ["Think of two cointegrated series as elastic bands stretched between two posts — they can diverge "
                 "in the short run but are always pulled back toward each other. This long-run equilibrium is the "
                 "cointegrating relationship.",
                 "For the solar dataset: shortwave radiation and temperature are both non-stationary over a year, "
                 "but they track each other closely — when one rises (summer), so does the other. "
                 "A cointegrating model captures this shared seasonal structure without differencing away the trend.",
                 dcc.Markdown(r"$X_t, Y_t$ are CI(1,1) if $\exists\,\beta$ such that $Y_t - \beta X_t \sim I(0)$", mathjax=True)],
                None,
                "No linear combination is stationary",
                "∃ β such that Yₜ − βXₜ is I(0)"),
        ], className="theory-row"),
        html.Div([
            theory_card("Error Correction Model", "ECM", AMBER,
                ["Once cointegration is confirmed, an Error Correction Model (ECM) captures both short-run dynamics "
                 "and the long-run equilibrium in one framework. The error correction term (Yₜ₋₁ − βXₜ₋₁) measures "
                 "how far the system deviated from equilibrium in the previous period.",
                 "The coefficient γ on this term must be negative for the system to correct — a value of γ = −0.2 "
                 "means 20% of any deviation is corrected each hour. For solar datasets, this speed of adjustment "
                 "reflects how quickly temperature responds to a radiation anomaly.",
                 dcc.Markdown(r"$\Delta Y_t = \alpha + \gamma(Y_{t-1} - \beta X_{t-1}) + \sum_i \delta_i \Delta X_{t-i} + \varepsilon_t$", mathjax=True)],
                None,
                "γ = 0 → no error correction (series drift apart)",
                "γ < 0 → system corrects toward long-run equilibrium"),
            theory_card("Johansen Test (note)", "JOH", "#a78bfa",
                ["The Engle-Granger test is limited to pairs of variables. When testing cointegration across three or "
                 "more series simultaneously, the Johansen procedure is the standard tool. It uses a Vector "
                 "Autoregression (VAR) framework and tests the rank of the long-run coefficient matrix Π.",
                 "The rank equals the number of cointegrating vectors. Rank 0 means no cointegration; rank r means "
                 "r independent long-run relationships exist. For the full 20-variable solar dataset, Johansen "
                 "would reveal the complete cointegration structure in a single test.",
                 dcc.Markdown(r"VAR: $\Delta X_t = \mu + \Pi X_{t-1} + \sum_i \Gamma_i \Delta X_{t-i} + \varepsilon_t$, $\text{rank}(\Pi) = r$", mathjax=True)],
                None,
                "rank(Π) = 0 → no cointegration",
                "rank(Π) = r ≥ 1 → r cointegrating vectors"),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),
], className="theory-section")


GRANGER_THEORY = html.Div([
    html.Div("⚡ Granger Causality — Theory", className="theory-title"),
    html.Div([
        html.Div([
            theory_card("Granger Causality", "GC", ACCENT2,
                ["Granger causality is not causality in the philosophical sense — it is predictive causality. "
                 "We say X Granger-causes Y if knowing past values of X improves our forecast of Y, "
                 "beyond what past values of Y alone could tell us.",
                 "The test fits two VAR models: a restricted model (Y predicted only by its own past) and an "
                 "unrestricted model (Y predicted by both its past and X's past). An F-test compares the residual "
                 "sum of squares. A significant F-statistic means X adds predictive power.",
                 dcc.Markdown(r"Unrestricted: $Y_t = \sum_i a_i Y_{t-i} + \sum_i b_i X_{t-i} + \varepsilon_t$", mathjax=True)],
                None,
                "b₁ = b₂ = … = bₖ = 0 (X adds no predictive power for Y)",
                "At least one bᵢ ≠ 0 (X Granger-causes Y at some lag)"),
            theory_card("Key Caveats", "!", RED,
                ["Granger causality has three well-known failure modes that every analyst must understand:",
                 "① Confounding: a hidden third variable Z may drive both X and Y, creating the illusion that X causes Y. "
                 "Example: zenith angle drives both shortwave radiation and temperature — each appears to Granger-cause the other.",
                 "② Instantaneous effects: GC only captures lagged relationships. If X and Y move together within the "
                 "same hour (common in coupled physical systems), GC will miss the relationship entirely.",
                 "③ Non-linearity: standard GC assumes a linear VAR. Chaotic or threshold systems require nonlinear "
                 "alternatives like Convergent Cross Mapping (CCM) or kernel-based GC."]),
        ], className="theory-row"),
        html.Div([
            theory_card("VAR Block Exogeneity", "VAR-GC", AMBER,
                ["When multiple variables are modelled jointly in a Vector Autoregression, the pairwise GC framework "
                 "generalises to block exogeneity tests. A Wald test evaluates whether the entire coefficient block "
                 "for variable X (across all lags) is jointly zero in the equation for Y.",
                 "This is more powerful than pairwise GC because the VAR accounts for the joint dynamics of all "
                 "included variables simultaneously, reducing the risk of spurious causality from omitted variables.",
                 dcc.Markdown(r"VAR: $X_t = \sum_k A^{(k)} X_{t-k} + \varepsilon_t$ — Wald test on block $A^{(k)}_{ij}$", mathjax=True)],
                None,
                "All elements of Aₖ[i,j] = 0 across all lags (X does not Granger-cause Y in the system)",
                "At least one Aₖ[i,j] ≠ 0 (X Granger-causes Y within the VAR)"),
            theory_card("Convergent Cross Mapping", "CCM", GREEN,
                ["CCM (Sugihara et al., 2012) is a nonlinear causal detection method for dynamical systems. "
                 "It reconstructs the state space of a system using time-delay embeddings (Takens' theorem) and "
                 "tests whether the manifold of X can be used to predict the state of Y.",
                 "Unlike GC, CCM detects causality in weakly coupled chaotic systems where linear methods fail. "
                 "For the solar dataset, a Lyapunov exponent of λ ≈ 0.072 confirms chaotic dynamics, making CCM "
                 "the recommended method to validate the Granger causality findings presented here.",
                 "Recommended full pipeline: CCM validation → VAR on confirmed causal graph → block-bootstrap "
                 "VAR residuals with block size 24h (to preserve diurnal autocorrelation) → validate with KS test "
                 "and Lyapunov exponent comparison (target λ ≈ 0.072, Hurst H ≈ 0.19)."]),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),
    html.Div([
        html.B("Pipeline recommendation: ", style={"color": GREEN}),
        "(1) Confirm Granger pairs with CCM to rule out confounding. "
        "(2) Fit a VAR on the significant causal graph. "
        "(3) Block-bootstrap VAR residuals with block size = 24h to preserve diurnal autocorrelation. "
        "(4) Validate synthetic output with KS test + Lyapunov exponent comparison (target λ≈0.072) and Hurst H≈0.19."
    ], className="tip-box"),
], className="theory-section")


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
EXTERNAL_STYLESHEETS = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap",
]

APP_CSS = f"""
*, *::before, *::after {{ box-sizing: border-box; }}

body, html {{
  background: {BG} !important;
  color: {TEXT} !important;
  font-family: {FONT_SANS} !important;
  min-height: 100vh;
}}

/* Navbar */
.navbar, .navbar-default, nav.navbar {{
  background: #090d1a !important;
  border-bottom: 1px solid {BORDER} !important;
  box-shadow: 0 1px 20px rgba(0,0,0,0.5) !important;
  font-family: {FONT_MONO} !important;
  padding: 0 24px !important;
}}
.navbar-brand {{
  color: {TEXT} !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  letter-spacing: -.01em !important;
}}
.nav-link {{
  color: {MUTED} !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: .02em !important;
  padding: 10px 18px !important;
  border-radius: 6px 6px 0 0 !important;
  transition: all .15s !important;
  font-family: {FONT_MONO} !important;
}}
.nav-link:hover {{ color: {MUTED2} !important; background: rgba(255,255,255,0.04) !important; }}
.nav-link.active {{ color: {ACCENT2} !important; border-bottom: 2px solid {ACCENT} !important; background: rgba(79,142,247,0.08) !important; }}

/* Tab content */
.tab-content, .tab-pane {{
  background: {BG} !important;
}}
.container-fluid {{
  padding: 24px 28px 60px !important;
  background: {BG} !important;
}}

/* Section headers */
.section-hdr {{
  font-family: {FONT_MONO};
  font-size: 15px; font-weight: 700; color: {TEXT};
  margin-bottom: 6px; margin-top: 24px;
  display: flex; align-items: center; gap: 10px;
}}
.section-hdr::before {{
  content: ''; width: 3px; height: 16px;
  border-radius: 2px; background: {ACCENT}; flex-shrink: 0;
}}
.section-desc {{
  font-size: 12.5px; color: {MUTED2}; margin-bottom: 18px;
  max-width: 840px; line-height: 1.75;
}}

/* Summary banner */
.summary-banner {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px; margin-bottom: 22px;
}}
.sbn {{
  background: {CARD}; border: 1px solid {BORDER};
  border-radius: 10px; padding: 14px 18px; text-align: center;
}}
.sbn-val {{
  font-size: 26px; font-weight: 700; line-height: 1.1;
  font-family: {FONT_MONO};
}}
.sbn-lbl {{
  font-size: 10px; font-weight: 600; letter-spacing: .07em;
  color: {MUTED}; text-transform: uppercase; margin-top: 4px;
  font-family: {FONT_MONO};
}}

/* Plot card */
.plot-card {{
  background: {CARD}; border: 1px solid {BORDER};
  border-radius: 10px; overflow: hidden;
  margin-bottom: 16px; box-shadow: 0 2px 20px rgba(0,0,0,0.4);
}}
.plot-label {{
  font-size: 10px; font-weight: 700; letter-spacing: .07em;
  color: {MUTED2}; text-transform: uppercase;
  font-family: {FONT_MONO}; padding: 10px 16px 0;
}}

/* Inputs */
.form-label, label {{
  color: {MUTED2} !important; font-family: {FONT_MONO} !important;
  font-size: 11px !important; font-weight: 600 !important;
  letter-spacing: .06em !important; text-transform: uppercase !important;
}}
.Select-control, .Select-menu-outer,
.form-select, .form-control, select, .VirtualizedSelectOption {{
  background: {CARD} !important; border: 1px solid {BORDER} !important;
  color: {TEXT} !important; font-family: {FONT_MONO} !important;
  font-size: 13px !important; border-radius: 6px !important;
}}
.form-select:focus {{ border-color: {ACCENT} !important; box-shadow: 0 0 0 2px rgba(79,142,247,0.2) !important; }}
.Select--single > .Select-control .Select-value {{ color: {TEXT} !important; }}
.Select-placeholder {{ color: {MUTED} !important; }}

/* Dash dcc.Dropdown */
.Select-menu-outer {{ background: {CARD2} !important; border: 1px solid {BORDER} !important; z-index: 9999 !important; }}
.VirtualizedSelectOption {{ background: {CARD2} !important; color: {TEXT} !important; }}
.VirtualizedSelectFocusedOption {{ background: rgba(79,142,247,0.15) !important; color: {TEXT} !important; }}
.Select-value-label {{ color: {TEXT} !important; font-family: {FONT_MONO} !important; font-size: 13px !important; }}
.Select.is-focused > .Select-control {{ border-color: {ACCENT} !important; box-shadow: 0 0 0 2px rgba(79,142,247,0.2) !important; background: {CARD} !important; }}
.Select-arrow {{ border-color: {MUTED} transparent transparent !important; }}

/* Slider */
.rc-slider-track {{ background: {ACCENT} !important; }}
.rc-slider-handle {{ border-color: {ACCENT} !important; background: {ACCENT} !important; }}
.rc-slider-rail {{ background: rgba(255,255,255,0.08) !important; }}
.rc-slider-dot {{ border-color: {BORDER} !important; background: {CARD} !important; }}
.rc-slider-mark-text {{ color: {MUTED} !important; font-family: {FONT_MONO} !important; font-size: 10px !important; }}

/* Controls row */
.controls-row {{
  display: flex; align-items: flex-end; gap: 20px;
  flex-wrap: wrap; margin-bottom: 16px;
}}
.ctrl-block {{ display: flex; flex-direction: column; gap: 4px; min-width: 200px; }}
.ctrl-block-sm {{ display: flex; flex-direction: column; gap: 4px; min-width: 300px; }}

/* Theory section */
.theory-section {{
  margin-top: 36px; border-top: 1px solid {BORDER};
  padding-top: 28px;
}}
.theory-title {{
  font-family: {FONT_MONO}; font-size: 14px; font-weight: 700;
  color: {TEXT}; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}}
.theory-grid {{
  display: flex;
  flex-direction: column;
  gap: 0;
}}
.theory-row {{
  display: flex;
  flex-direction: row;
  gap: 0;
  border-bottom: 1px solid {BORDER};
}}
.theory-row:last-child {{
  border-bottom: none;
}}
.theory-card {{
  width: 50%;
  background: transparent;
  border: none;
  border-right: 1px solid {BORDER};
  border-radius: 0;
  padding: 28px 32px;
}}
.theory-card:last-child {{
  border-right: none;
}}
.theory-card h3 {{
  font-size: 13px; font-weight: 700; font-family: {FONT_MONO};
  margin-bottom: 12px; padding-bottom: 10px;
  border-bottom: 1px solid {BORDER};
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  color: {TEXT};
}}
.tc-tag {{
  font-size: 10px; padding: 2px 7px; border-radius: 3px;
  background: rgba(79,142,247,0.12); color: {ACCENT};
  font-weight: 700; letter-spacing: .04em;
}}
.theory-card p {{ font-size: 12.5px; color: {MUTED2}; line-height: 1.8; margin-bottom: 8px; }}

/* Wrap the theory grid itself in a bordered container */
.theory-grid-wrap {{
  border: 1px solid {BORDER};
  border-radius: 10px;
  overflow: hidden;
  background: {CARD2};
  margin-bottom: 16px;
}}
.math {{
  display: block; margin: 8px 0; padding: 7px 12px;
  background: rgba(0,0,0,0.35); border-radius: 6px;
  font-family: {FONT_MONO}; font-size: 11px;
  color: {AMBER}; border-left: 2px solid {AMBER};
}}
.h0-box {{ margin-top:8px; padding:5px 10px; background:rgba(244,115,110,0.08); border-radius:6px; border-left:2px solid {RED}; font-size:11.5px; color:{RED}; font-family:{FONT_MONO}; }}
.h1-box {{ margin-top:4px; padding:5px 10px; background:rgba(45,212,160,0.08); border-radius:6px; border-left:2px solid {GREEN}; font-size:11.5px; color:{GREEN}; font-family:{FONT_MONO}; }}
.warn-box {{ margin-top:16px; padding:12px 16px; background:rgba(245,166,35,0.07); border-radius:8px; border-left:3px solid {AMBER}; font-size:12px; color:{MUTED2}; line-height:1.75; }}
.tip-box  {{ margin-top:16px; padding:12px 16px; background:rgba(45,212,160,0.07); border-radius:8px; border-left:3px solid {GREEN}; font-size:12px; color:{MUTED2}; line-height:1.75; }}

/* Insight bar */
.insight-bar {{
  padding: 12px 16px; margin-bottom: 16px;
  background: rgba(79,142,247,0.07);
  border: 1px solid rgba(79,142,247,0.18);
  border-radius: 8px; font-size: 13px; color: {MUTED2}; line-height: 1.7;
  font-family: {FONT_MONO};
}}

/* Legend pill row */
.legend-row {{
  display: flex; align-items: center; gap: 18px;
  flex-wrap: wrap; margin-bottom: 14px;
  padding: 8px 14px;
  background: {CARD2}; border-radius: 8px;
  font-size: 12px; color: {MUTED2}; font-family: {FONT_MONO};
}}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.ldot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.08); border-radius: 3px; }}

/* Graph container */
.js-plotly-plot, .plotly, .plot-container {{ background: transparent !important; }}

/* Diagnostic theory card extras */
.tc-plain {{
  font-size: 12.5px; color: {MUTED2}; line-height: 1.8; margin-bottom: 10px;
}}
.tc-formula-label {{
  font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: {MUTED}; font-family: {FONT_MONO}; margin-bottom: 2px;
}}
.tc-latex {{
  background: rgba(0,0,0,0.32); border-radius: 6px; padding: 8px 12px;
  border-left: 2px solid {AMBER}; margin-bottom: 8px; font-size: 13px;
  color: {TEXT};
}}
.tc-latex p {{ margin-bottom: 0 !important; }}
.tc-interp {{
  margin-top: 8px; padding: 7px 12px; background: rgba(79,142,247,0.07);
  border-radius: 6px; border-left: 2px solid {ACCENT}; font-size: 12px;
  color: {MUTED2}; line-height: 1.75;
}}

/* Override MathJax font color inside dark cards */
.MathJax {{ color: {TEXT} !important; }}
mjx-container {{ color: {TEXT} !important; }}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  DASH APP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    title="Solar Energy · Time Series Dashboard",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.index_string = f"""<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{APP_CSS}</style>
    <script>
      window.MathJax = {{
        tex: {{ inlineMath: [['$','$'],['\\\\(','\\\\)']], displayMath: [['$$','$$'],['\\\\[','\\\\]']] }},
        options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }},
        startup: {{ typeset: true }}
      }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
  </head>
  <body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
  </body>
</html>"""

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_exploration = dbc.Tab(
    label="📈  Exploration",
    tab_id="tab-exploration",
    children=html.Div([
        section_hdr("ACF & PACF Explorer"),
        html.Div(
            "Autocorrelation and Partial Autocorrelation up to 48 lags. "
            "Blue stems exceed the 95% confidence interval. Gold dashed lines mark ±1.96/√n.",
            className="section-desc"
        ),
        html.Div([
            html.Div([
                html.Label("Variable"),
                dcc.Dropdown(
                    id="acf-var", options=VAR_OPTIONS,
                    value=AVAILABLE[0], clearable=False,
                    style={"minWidth": "260px"},
                ),
            ], className="ctrl-block"),
            html.Div([
                html.Label("Max lags"),
                dcc.Slider(
                    id="acf-lags", min=12, max=48, step=1, value=48,
                    marks={12: "12", 24: "24", 36: "36", 48: "48"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], className="ctrl-block-sm"),
        ], className="controls-row"),
        legend_row(),
        plot_card([dcc.Graph(id="acf-pacf-plot", config=PLOTLY_CFG)]),
        html.Div(id="acf-insight"),

        section_hdr("Time Series — Selected Variable"),
        html.Div(
            "Hourly observations (downsampled for rendering). Inspect trends, seasonality, and anomalies.",
            className="section-desc"
        ),
        html.Div([
            html.Div([
                html.Label("Variable"),
                dcc.Dropdown(
                    id="ts-var", options=VAR_OPTIONS,
                    value=AVAILABLE[0], clearable=False,
                    style={"minWidth": "260px"},
                ),
            ], className="ctrl-block"),
        ], className="controls-row"),
        plot_card([dcc.Graph(id="ts-plot", config=PLOTLY_CFG)]),
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)

tab_stationarity = dbc.Tab(
    label="🧪  Stationarity",
    tab_id="tab-stationarity",
    children=html.Div([
        section_hdr("Stationarity Test Suite"),
        html.Div(
            f"Four complementary unit-root / stationarity tests applied to all "
            f"{len(AVAILABLE)} variables. The heatmap shows pass/fail; the bar chart shows −log₁₀(p) "
            "for the selected variable.",
            className="section-desc"
        ),
        banner(
            (len(AVAILABLE), "Variables tested", ACCENT2),
            (n_stat, "Stationary (ADF or PP)", GREEN),
            (len(AVAILABLE) - n_stat, "Non-stationary", RED),
            ("5%", "Significance level", AMBER),
        ),
        plot_card([dcc.Graph(id="stat-heatmap", figure=fig_station_heatmap(), config=PLOTLY_CFG)]),
        html.Div([
            html.Div([
                html.Label("Variable for bar chart"),
                dcc.Dropdown(
                    id="stat-var", options=VAR_OPTIONS,
                    value=AVAILABLE[0], clearable=False,
                    style={"minWidth": "260px"},
                ),
            ], className="ctrl-block"),
        ], className="controls-row", style={"marginTop": "16px"}),
        plot_card([dcc.Graph(id="stat-bar", config=PLOTLY_CFG)]),
        STATION_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)

tab_cointegration = dbc.Tab(
    label="🔗  Cointegration",
    tab_id="tab-cointegration",
    children=html.Div([
        section_hdr("Cointegration Tests — Engle-Granger"),
        html.Div(
            f"Pairwise Engle-Granger cointegration test on {len(COINT_PAIRS)} physically motivated variable pairs. "
            "Cointegrated pairs share a long-run equilibrium: deviations are temporary and mean-reverting.",
            className="section-desc"
        ),
        banner(
            (len(COINT_RES), "Pairs tested", ACCENT2),
            (n_co, "Cointegrated", GREEN),
            (len(COINT_RES) - n_co, "Not cointegrated", RED),
            ("5%", "Significance level", AMBER),
        ),
        plot_card(
            [dcc.Graph(id="coint-bar", figure=fig_coint_bars(), config=PLOTLY_CFG)],
            label="EG test statistic  ·  more negative = stronger cointegration"
        ),
        plot_card(
            [dcc.Graph(id="coint-pvals", figure=fig_coint_pvals(), config=PLOTLY_CFG)],
            label="p-values  ·  green = cointegrated at 5%"
        ),
        COINT_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)

tab_causality = dbc.Tab(
    label="⚡  Causality",
    tab_id="tab-causality",
    children=html.Div([
        section_hdr("Granger Causality Tests"),
        html.Div(
            f"Tests whether lagged values of X improve the forecast of Y beyond Y's own lags "
            f"(max lag = {MAX_LAG_GC} h). Applied to first-differenced series to ensure stationarity.",
            className="section-desc"
        ),
        banner(
            (len(GRANGER_RES), "Pairs tested", ACCENT2),
            (n_gc, "Significant (5%)", GREEN),
            (len(GRANGER_RES) - n_gc, "Not significant", RED),
            (MAX_LAG_GC, "Max lags tested (h)", AMBER),
        ),
        plot_card(
            [dcc.Graph(id="gc-heatmap", figure=fig_granger_heatmap(), config=PLOTLY_CFG)],
            label="Causal matrix  ·  −log₁₀(p)  ·  darker = no causality"
        ),
        plot_card(
            [dcc.Graph(id="gc-bar", figure=fig_granger_bar(), config=PLOTLY_CFG)],
            label="Significance ranking  ·  best lag among 1–6 h"
        ),
        GRANGER_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)

tab_diagnostics = dbc.Tab(
    label="🔬  Diagnostics",
    tab_id="tab-diagnostics",
    children=html.Div([
        section_hdr("Residual Diagnostics — Normality & Autocorrelation"),
        html.Div(
            "Five diagnostic tests applied to all variables: "
            "Kolmogorov-Smirnov, Shapiro-Wilk, and Jarque-Bera for normality; "
            "Ljung-Box for autocorrelation structure; "
            "Hannan-Quinn Criterion for optimal AR model order. "
            "Select a variable for the full diagnostic panel.",
            className="section-desc",
        ),

        # ── Summary banner ──────────────────────────────────────────────
        banner(
            (n_normal_ks, f"Normal (KS, n={len(AVAILABLE)})", ACCENT2),
            (n_normal_sw, f"Normal (SW)", GREEN),
            (n_normal_jb, f"Normal (JB)", AMBER),
            (n_white_lb,  "White noise (LB)", "#a78bfa"),
        ),

        # ── Overview heatmap ────────────────────────────────────────────
        plot_card(
            [dcc.Graph(id="diag-overview", figure=fig_diag_overview(), config=PLOTLY_CFG)],
            label="All-variable overview  ·  pass / fail at α = 5%",
        ),

        # ── Normality scatter ───────────────────────────────────────────
        plot_card(
            [dcc.Graph(id="diag-normality-scatter",
                       figure=fig_diag_normality_scatter(), config=PLOTLY_CFG)],
            label="Normality landscape  ·  each dot = one variable",
        ),

        # ── Variable selector + per-variable panel ──────────────────────
        html.Div([
            html.Div([
                html.Label("Variable — full diagnostic panel"),
                dcc.Dropdown(
                    id="diag-var", options=VAR_OPTIONS,
                    value=AVAILABLE[0], clearable=False,
                    style={"minWidth": "280px"},
                ),
            ], className="ctrl-block"),
        ], className="controls-row", style={"marginTop": "20px"}),
        plot_card([dcc.Graph(id="diag-variable-panel", config=PLOTLY_CFG)]),

        DIAG_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)


app.layout = html.Div([
    dbc.NavbarSimple(
        brand=html.Span([
            html.Span("☀", style={"marginRight": "8px"}),
            "Solar Energy  ·  Time Series Dashboard",
        ], style={"fontFamily": FONT_MONO, "fontWeight": "700", "fontSize": "15px"}),
        color="#090d1a",
        dark=True,
        fluid=True,
        style={"borderBottom": f"1px solid {BORDER}", "fontFamily": FONT_MONO},
    ),
    dbc.Container(
        dbc.Tabs(
            [tab_exploration, tab_stationarity, tab_cointegration, tab_causality, tab_diagnostics],
            id="main-tabs",
            active_tab="tab-exploration",
            style={"borderBottom": f"1px solid {BORDER}", "fontFamily": FONT_MONO},
        ),
        fluid=True,
        style={"padding": "0"},
    ),
], style={"background": BG, "minHeight": "100vh"})


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("acf-pacf-plot", "figure"),
    Input("acf-var", "value"),
    Input("acf-lags", "value"),
)
def update_acf_pacf(col, max_lag):
    if col not in ACF_STATS:
        return go.Figure()
    return fig_acf_pacf(col, max_lag=max_lag or NLAGS)


@callback(
    Output("acf-insight", "children"),
    Input("acf-var", "value"),
)
def update_insight(col):
    if not col:
        return ""
    return insight_bar(col)


@callback(
    Output("ts-plot", "figure"),
    Input("ts-var", "value"),
)
def update_ts(col):
    if not col or col not in DF.columns:
        return go.Figure()
    return fig_ts_panel(col)


@callback(
    Output("stat-bar", "figure"),
    Input("stat-var", "value"),
)
def update_stat_bar(col):
    if not col or col not in STAT_RES:
        return go.Figure()
    return fig_station_bar(col)


@callback(
    Output("diag-variable-panel", "figure"),
    Input("diag-var", "value"),
)
def update_diag_panel(col):
    if not col or col not in DIAG_RES:
        return go.Figure()
    return fig_diag_variable(col)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)