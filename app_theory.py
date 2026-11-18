import warnings; warnings.filterwarnings("ignore")
# PyTorch (GENDA, Synthetic Data III) must initialise its DLLs before pandas /
# statsmodels load theirs, or Windows fails with WinError 1114. If torch is
# missing the tab still works, with GENDA disabled.
try:
    import torch  # noqa: F401
except Exception:
    pass
import math
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import (
    adfuller, kpss, zivot_andrews,
    acf, pacf,
    grangercausalitytests, coint,
)
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch.unitroot.cointegration import phillips_ouliaris
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy import stats as sp_stats
from datetime import datetime

from arch.unitroot import PhillipsPerron
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from map_tab import tab_map, register_map_callbacks
from synth_tab_control import tab_synth, register_synth_callbacks
from synth2_tab import tab_synth2, register_synth2_callbacks
from rqa_tab import tab_rqa, register_rqa_callbacks
from synth3_tab import tab_synth3, register_synth3_callbacks
from cache_utils import make_signature, load_cache, save_cache

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH    = Path("data/spg.csv")
NLAGS       = 48
THRES       = 0.05
MAX_LAG_GC  = 6
TS_POINTS   = 1200
LB_LAGS     = [6, 12, 24]   # lags for Ljung-Box
HQC_MAX_LAG = 12            # max AR order for HQC search
BG_LAGS     = 12            # Breusch-Godfrey: test serial correlation up to this lag

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
    "temperature_2_m_above_gnd":        "Sinusoidal ACF peaking at lag 12. Diurnal heating cycle dominates. PACF cuts off after lag 12 → AR(12) residual structure.",
    "relative_humidity_2_m_above_gnd":  "Mirror of temperature (peaks overnight), troughs mid-afternoon. Strong 24h period. PACF cuts off after lag 13 → AR(13).",
    "mean_sea_level_pressure_MSL":      "Weak semi-diurnal (~12h) tidal oscillation on synoptic variability. PACF shows peaks at lags 31.",
    "total_precipitation_sfc":          "Near-zero ACF at all lags (highly intermittent). Near-white-noise at hourly resolution. Essentially unpredictable beyond lag 1.",
    "snowfall_amount_sfc":              "Highly sparse with near-zero ACF. Rare, episodic events dominated by synoptic patterns over days.",
    "total_cloud_cover_sfc":            "Moderate 24h cycle from sea-breeze and convective patterns. PACF cuts off ~lag 3. Explains cloud cover failure in Granger tests.",
    "high_cloud_cover_high_cld_lay":    "Weaker ACF than total cloud: cirrus driven by upper-tropospheric dynamics independent of surface diurnal cycles.",
    "medium_cloud_cover_mid_cld_lay":   "Slightly stronger diurnal structure than high clouds from afternoon convection. ACF decays quickly.",
    "low_cloud_cover_low_cld_lay":      "Stratus/fog shows persistent overnight ACF from nocturnal radiation cooling, burns off mid-morning.",
    "shortwave_radiation_backwards_sfc":"Textbook diurnal sine wave (peaks at lag 24 with r≈0.995 and lag 48). Primary causal driver of solar generation.",
    "wind_speed_10_m_above_gnd":        "Moderate 24h periodicity from sea-breeze. PACF cuts off at lag 3 → low-order AR process.",
    "wind_direction_10_m_above_gnd":    "Circular variable (0-360°) (ACF/PACF interpreted cautiously). Sea-breeze rotation near lag 24.",
    "wind_speed_80_m_above_gnd":        "Stronger persistence than 10m from reduced surface friction. Smoother diurnal cycle.",
    "wind_direction_80_m_above_gnd":    "Circular variable. Less influenced by surface obstacles (cleaner but noisy diurnal signal).",
    "wind_speed_900_mb":                "Upper-level (~1km) wind shows weaker diurnal variation and stronger synoptic-scale persistence.",
    "wind_direction_900_mb":            "Reflects synoptic steering flow. Circular variable (less diurnally driven).",
    "wind_gust_10_m_above_gnd":         "Closely coupled to 10m wind (similar ACF) but higher variance and faster decorrelation from turbulent gusting.",
    "angle_of_incidence":               "Geometrically determined (near-perfect 24h ACF). Clean AR(2) PACF cut-off identical to shortwave radiation.",
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
        raise FileNotFoundError(f"WARNING: {CSV_PATH} not found: generating synthetic demo data")

    print("Loading data …")
    df = pd.read_csv(CSV_PATH)
    # The series is hourly (diurnal cycles at lag 24, axes labelled "lag (hours)"),
    # so stamp an hourly DatetimeIndex. `periods=` guarantees the index length
    # matches the number of rows regardless of any end/start rounding.
    end_date = datetime.today().replace(minute=0, second=0, microsecond=0)
    df["date"] = pd.date_range(end=end_date, periods=len(df), freq="h")
    df = df.set_index("date")
    return df

print("Initialising …")
_df_raw  = load_data()
AVAILABLE = [c for c in COLS if c in _df_raw.columns]
DF        = _df_raw[AVAILABLE].copy()
print(f"  {len(AVAILABLE)} columns, {len(DF):,} rows")


def _run_precompute() -> dict:
    """Run the full statistical test battery once: ACF/PACF, stationarity,
    (Engle-Granger / Johansen / Phillips-Ouliaris) cointegration, Granger
    causality, and the two diagnostics blocks. Pure function of DF/AVAILABLE
    plus the module-level config above — this is exactly what gets cached to
    disk below, so a later run with unchanged spg.csv and config can skip it
    entirely."""
    # ── ACF / PACF ──────────────────────────────────────────────────────────
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

    # ── Johansen Cointegration ────────────────────────────────────────────────────
    print("Running Johansen cointegration tests …")
    JOHANSEN_RES: dict = {}
    for c1, c2 in COINT_PAIRS:
        if c1 not in DF.columns or c2 not in DF.columns:
            continue
        s1, s2 = DF[c1].dropna().values, DF[c2].dropna().values
        n = min(len(s1), len(s2))
        data2 = np.column_stack([s1[:n], s2[:n]])
        key = f"{LABELS[c1]} ↔ {LABELS[c2]}"
        try:
            jres = coint_johansen(data2, det_order=0, k_ar_diff=1)
            # Trace statistics and 5% critical values
            trace_stat0  = float(jres.lr1[0])   # H0: rank=0
            trace_stat1  = float(jres.lr1[1])   # H0: rank<=1
            trace_crit0  = float(jres.cvt[0, 1])  # 5% crit, rank=0
            trace_crit1  = float(jres.cvt[1, 1])  # 5% crit, rank<=1
            # Max-eigenvalue statistics
            maxeig_stat0 = float(jres.lr2[0])
            maxeig_stat1 = float(jres.lr2[1])
            maxeig_crit0 = float(jres.cvm[0, 1])
            maxeig_crit1 = float(jres.cvm[1, 1])
            # Determine rank (number of cointegrating vectors)
            rank = 0
            if trace_stat0 > trace_crit0:
                rank = 1
            if trace_stat1 > trace_crit1:
                rank = 2
            sig = rank >= 1
            JOHANSEN_RES[key] = dict(
                c1=c1, c2=c2, label1=LABELS[c1], label2=LABELS[c2],
                trace_stat0=trace_stat0, trace_stat1=trace_stat1,
                trace_crit0=trace_crit0, trace_crit1=trace_crit1,
                maxeig_stat0=maxeig_stat0, maxeig_stat1=maxeig_stat1,
                maxeig_crit0=maxeig_crit0, maxeig_crit1=maxeig_crit1,
                rank=rank, sig=sig,
            )
            print(f"  {key[:55]:55s}  rank={rank}  trace0={trace_stat0:.2f} (crit={trace_crit0:.2f})")
        except Exception as e:
            print(f"  {key[:55]:55s}  ERROR: {e}")

    # ── Phillips-Ouliaris Cointegration ───────────────────────────────────────────
    print("Running Phillips-Ouliaris cointegration tests …")
    PO_RES: dict = {}
    for c1, c2 in COINT_PAIRS:
        if c1 not in DF.columns or c2 not in DF.columns:
            continue
        s1, s2 = DF[c1].dropna().values, DF[c2].dropna().values
        n = min(len(s1), len(s2))
        key = f"{LABELS[c1]} ↔ {LABELS[c2]}"
        try:
            po      = phillips_ouliaris(s1[:n], s2[:n], trend="c", test_type="Zt")
            po_stat = float(po.stat)
            po_pval = float(po.pvalue)
            sig     = po_pval < THRES
            PO_RES[key] = dict(
                c1=c1, c2=c2, label1=LABELS[c1], label2=LABELS[c2],
                stat=po_stat, p=po_pval, sig=sig,
            )
            print(f"  {key[:55]:55s}  Zt={po_stat:.4f}  p={pf(po_pval)}")
        except Exception as e:
            print(f"  {key[:55]:55s}  ERROR: {e}")

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

    # ── Diagnostics: KS, Shapiro-Wilk, Jarque-Bera, Ljung-Box, Hannan-Quinn ────
    print("Running diagnostics tests …")
    DIAG_RES: dict = {}

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

    # ── Regression-residual diagnostics: Breusch-Pagan, Breusch-Godfrey, Durbin-Watson ──
    #   These three classical tests operate on the residuals of a regression, so for each
    #   variable we fit a simple model — a constant plus a linear time trend
    #   (y_t = α + β·t + e_t) — and test its residuals e_t for:
    #     · heteroskedasticity   → Breusch-Pagan
    #     · serial correlation   → Breusch-Godfrey (up to lag p) and Durbin-Watson (lag 1)
    print("Running regression-residual diagnostics …")
    REGDIAG_RES: dict = {}

    for col in AVAILABLE:
        s = DF[col].dropna().values
        n = len(s)
        trend = np.arange(n, dtype=float)
        X   = sm.add_constant(trend)        # design matrix: [const, time-trend]
        ols = sm.OLS(s, X).fit()
        resid = ols.resid

        # Durbin-Watson — first-order serial correlation (≈2 none, <2 positive, >2 negative)
        dw = float(durbin_watson(resid))

        # Breusch-Godfrey — serial correlation up to BG_LAGS (LM test)
        try:
            bg_lm, bg_lmp, bg_f, bg_fp = acorr_breusch_godfrey(ols, nlags=BG_LAGS)
        except Exception:
            bg_lm = bg_lmp = bg_f = bg_fp = float("nan")

        # Breusch-Pagan — heteroskedasticity of residuals w.r.t. the regressors (LM test)
        try:
            bp_lm, bp_lmp, bp_f, bp_fp = het_breuschpagan(resid, ols.model.exog)
        except Exception:
            bp_lm = bp_lmp = bp_f = bp_fp = float("nan")

        REGDIAG_RES[col] = dict(
            # Breusch-Pagan (H0: homoskedastic → fail-to-reject = "ok")
            bp_stat=float(bp_lm), bp_p=float(bp_lmp),
            bp_f=float(bp_f),     bp_fp=float(bp_fp),
            bp_ok=bool(bp_lmp >= THRES),
            # Breusch-Godfrey (H0: no serial correlation → fail-to-reject = "ok")
            bg_stat=float(bg_lm), bg_p=float(bg_lmp),
            bg_f=float(bg_f),     bg_fp=float(bg_fp),
            bg_lags=int(BG_LAGS),
            bg_ok=bool(bg_lmp >= THRES),
            # Durbin-Watson (no p-value; ≈2 = no first-order serial correlation)
            dw_stat=dw,
            dw_ok=bool(1.5 <= dw <= 2.5),
        )
        print(f"  {LABELS.get(col, col):35s}  BP p={pf(bp_lmp)}  BG p={pf(bg_lmp)}  DW={dw:.3f}")

    print("Regression-residual diagnostics done ✔")

    return dict(
        acf_stats=ACF_STATS, stat_res=STAT_RES, coint_res=COINT_RES,
        johansen_res=JOHANSEN_RES, po_res=PO_RES, granger_res=GRANGER_RES,
        diag_res=DIAG_RES, regdiag_res=REGDIAG_RES,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED EXECUTION  —  hash the CSV file + every config knob the battery above
#  depends on; on a match, load the previous run's results straight from disk
#  instead of re-running ~20 variables through nine different statistical
#  tests. Delete .cache/ (or touch spg.csv, or change NLAGS/THRES/pairs/...)
#  to force a full recompute.
# ─────────────────────────────────────────────────────────────────────────────
_PRECOMPUTE_SIG = make_signature(
    CSV_PATH, version="app-precompute-v1",
    nlags=NLAGS, thres=THRES, max_lag_gc=MAX_LAG_GC,
    lb_lags=LB_LAGS, hqc_max_lag=HQC_MAX_LAG, bg_lags=BG_LAGS,
    coint_pairs=COINT_PAIRS, granger_pairs=GRANGER_PAIRS,
    available=AVAILABLE,
)
_cached_precompute = load_cache("app_precompute", _PRECOMPUTE_SIG)
if _cached_precompute is not None:
    print("Loaded cached statistical test battery from .cache/ ✔  "
          "(delete .cache/ to force a full recompute)")
    _bundle = _cached_precompute
else:
    _bundle = _run_precompute()
    save_cache("app_precompute", _PRECOMPUTE_SIG, _bundle)
print("All tests ready ✔")

ACF_STATS    = _bundle["acf_stats"]
STAT_RES     = _bundle["stat_res"]
COINT_RES    = _bundle["coint_res"]
JOHANSEN_RES = _bundle["johansen_res"]
PO_RES       = _bundle["po_res"]
GRANGER_RES  = _bundle["granger_res"]
DIAG_RES     = _bundle["diag_res"]
REGDIAG_RES  = _bundle["regdiag_res"]

n_stat      = sum(1 for c in AVAILABLE if STAT_RES[c]["adf_ok"] or STAT_RES[c]["pp_ok"])
n_co        = sum(1 for v in COINT_RES.values()    if v["sig"])
n_joh       = sum(1 for v in JOHANSEN_RES.values() if v["sig"])
n_po        = sum(1 for v in PO_RES.values()       if v["sig"])
n_gc        = sum(1 for v in GRANGER_RES.values()  if v["sig"])
n_normal_ks = sum(1 for c in AVAILABLE if DIAG_RES[c]["ks_ok"])
n_normal_sw = sum(1 for c in AVAILABLE if DIAG_RES[c]["sw_ok"])
n_normal_jb = sum(1 for c in AVAILABLE if DIAG_RES[c]["jb_ok"])
n_white_lb  = sum(1 for c in AVAILABLE if DIAG_RES[c]["lb_ok"])
n_homosk    = sum(1 for c in AVAILABLE if REGDIAG_RES[c]["bp_ok"])
n_nocorr_bg = sum(1 for c in AVAILABLE if REGDIAG_RES[c]["bg_ok"])
n_dw_ok     = sum(1 for c in AVAILABLE if REGDIAG_RES[c]["dw_ok"])

# ─────────────────────────────────────────────────────────────────────────────
#  THEORY-CARD COMPANION ANALYSES
#  Every theory card on the Cointegration and Causality tabs gets a plot of
#  its own. The original battery above already covers EG, Johansen, PO and
#  pairwise Granger; the analyses below cover the remaining cards:
#    • CI(1,1)            → long-run equilibrium error of each cointegrating pair
#    • Key Caveats        → lag-robustness of every Granger result (lags 1–6 h)
#    • VAR Block Exog.    → Wald block tests inside one VAR of all variables,
#                           plus pairwise tests conditioned on the full system
#    • CCM                → cross-map skill ρ vs library size L for each pair
#  Cached to .cache/app_theory_extras.pkl (≈ 30 s to recompute).
# ─────────────────────────────────────────────────────────────────────────────
from scipy.spatial import cKDTree
from statsmodels.tsa.api import VAR

VAR_MAX_LAG   = 12    # HQIC order search, consistent with the HQC card
CCM_MAX_TAU   = 12    # cap on the embedding delay (h)
CCM_E_RANGE   = range(2, 9)
CCM_N_LIB     = 10    # number of library sizes on the convergence curve
CCM_N_SAMPLES = 8     # random libraries per library size
CCM_N_PRED    = 800   # prediction points per cross-map


def _ccm_tau(x: np.ndarray, max_tau: int = CCM_MAX_TAU) -> int:
    """Delay = first lag where the ACF drops below 1/e or reaches a local
    minimum, whichever comes first (capped at max_tau)."""
    x = x - x.mean()
    v = float(np.dot(x, x)) or 1.0
    prev = 1.0
    for t in range(1, max_tau + 1):
        r = float(np.dot(x[:-t], x[t:])) / v
        if r < 1 / np.e:
            return t
        if r > prev:                      # first local minimum
            return max(t - 1, 1)
        prev = r
    return max_tau


def _ccm_embed(x: np.ndarray, E: int, tau: int) -> np.ndarray:
    n = len(x) - (E - 1) * tau
    return np.column_stack([x[(E - 1 - j) * tau:(E - 1 - j) * tau + n] for j in range(E)])


def _ccm_simplex_E(y: np.ndarray, tau: int) -> int:
    """Embedding dimension that maximises one-step simplex self-prediction."""
    best_E, best_rho = CCM_E_RANGE[0], -np.inf
    for E in CCM_E_RANGE:
        M = _ccm_embed(y, E, tau)
        tgt = y[(E - 1) * tau + 1:]
        M = M[:len(tgt)]
        n = len(M)
        lib, pred = np.arange(n // 2), np.arange(n // 2, n)
        dist, nn = cKDTree(M[lib]).query(M[pred], k=E + 1)
        w = np.exp(-dist / np.maximum(dist[:, :1], 1e-12))
        w /= w.sum(1, keepdims=True)
        rho = np.corrcoef((w * tgt[nn]).sum(1), tgt[pred])[0, 1]
        if rho > best_rho:
            best_E, best_rho = E, rho
    return best_E


def _ccm_xmap(M, target, lib_idx, pred_idx, E, excl) -> float:
    """Cross-map skill: predict target at pred_idx from the E+1 nearest
    library neighbours on M (excluding points within `excl` steps in time)."""
    k = min(E + 2 + 2 * excl, len(lib_idx))
    dist, nn = cKDTree(M[lib_idx]).query(M[pred_idx], k=k)
    nn = lib_idx[nn]
    ok = np.abs(nn - pred_idx[:, None]) > excl
    preds = np.empty(len(pred_idx))
    for i in range(len(pred_idx)):
        dd, ii = dist[i][ok[i]][:E + 1], nn[i][ok[i]][:E + 1]
        if len(ii) == 0:
            preds[i] = np.nan
            continue
        w = np.exp(-dd / max(dd[0], 1e-12))
        preds[i] = np.dot(w / w.sum(), target[ii])
    m = ~np.isnan(preds)
    return float(np.corrcoef(preds[m], target[pred_idx][m])[0, 1])


def _ccm_direction(x: np.ndarray, y: np.ndarray, rng) -> dict:
    """Test X → Y: reconstruct M_Y and cross-map X from it."""
    tau = _ccm_tau(y)
    E = _ccm_simplex_E(y, tau)
    M = _ccm_embed(y, E, tau)
    xt = x[(E - 1) * tau:]
    n = len(M)
    pred = rng.choice(n, min(CCM_N_PRED, n), replace=False)
    Ls = np.unique(np.geomspace(E + 12, n, CCM_N_LIB).astype(int))
    med, lo, hi = [], [], []
    for L in Ls:
        rr = [_ccm_xmap(M, xt, np.sort(rng.choice(n, L, replace=False)), pred, E, tau)
              for _ in range(CCM_N_SAMPLES)]
        med.append(float(np.median(rr)))
        lo.append(float(np.percentile(rr, 25)))
        hi.append(float(np.percentile(rr, 75)))
    return dict(tau=int(tau), E=int(E), L=Ls.tolist(), rho=med, lo=lo, hi=hi,
                rho_max=med[-1], delta=med[-1] - med[0])


def _run_theory_extras() -> dict:
    # ── Lag robustness (Key Caveats card) is read straight from GRANGER_RES ──
    # ── VAR block exogeneity ─────────────────────────────────────────────────
    print("Running VAR block-exogeneity tests …")
    d = DF.diff().dropna()
    d = d.loc[:, d.std() > 0]
    z = (d - d.mean()) / d.std()
    order = int(max(1, VAR(z).select_order(VAR_MAX_LAG).hqic))
    res = VAR(z).fit(order)
    blocks: dict = {}
    for c in z.columns:
        blocks.setdefault(GROUPS.get(c, ("Other", MUTED2))[0], []).append(c)
    block_res = {}
    for b, cols in blocks.items():
        for tgt in z.columns:
            causing = [c for c in cols if c != tgt]
            if not causing:
                continue
            r = res.test_causality(tgt, causing, kind="wald")
            block_res[(b, tgt)] = dict(stat=float(r.test_statistic), p=float(r.pvalue),
                                       df=int(r.df), n_causing=len(causing))
    print(f"  VAR({order}) on {z.shape[1]} differenced variables · {len(block_res)} block tests")
    cond_res = {}
    for key, g in GRANGER_RES.items():
        if g["cause"] in z.columns and g["effect"] in z.columns:
            r = res.test_causality(g["effect"], [g["cause"]], kind="wald")
            cond_res[key] = dict(p=float(r.pvalue), stat=float(r.test_statistic))

    # ── Convergent cross mapping ─────────────────────────────────────────────
    print("Running convergent cross mapping …")
    rng = np.random.default_rng(42)
    ccm_res = {}
    for key, g in GRANGER_RES.items():
        x = DF[g["cause"]].values.astype(float)
        y = DF[g["effect"]].values.astype(float)
        if x.std() == 0 or y.std() == 0:
            continue
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        ccm_res[key] = dict(fwd=_ccm_direction(x, y, rng),   # cause → effect
                            rev=_ccm_direction(y, x, rng))   # effect → cause
        print(f"  {key[:55]:55s}  ρ(X→Y)={ccm_res[key]['fwd']['rho_max']:.3f}"
              f"  ρ(Y→X)={ccm_res[key]['rev']['rho_max']:.3f}")
    return dict(var_order=order, var_blocks=list(blocks), var_targets=list(z.columns),
                block_res=block_res, cond_res=cond_res, ccm_res=ccm_res)


_EXTRAS_SIG = make_signature(
    CSV_PATH, version="app-theory-extras-v1",
    max_lag_gc=MAX_LAG_GC, granger_pairs=GRANGER_PAIRS, available=AVAILABLE,
    var_max_lag=VAR_MAX_LAG, ccm=(CCM_MAX_TAU, tuple(CCM_E_RANGE), CCM_N_LIB,
                                  CCM_N_SAMPLES, CCM_N_PRED),
)
_extras = load_cache("app_theory_extras", _EXTRAS_SIG)
if _extras is not None:
    print("Loaded cached theory-card analyses (VAR block exogeneity, CCM) from .cache/ ✔")
else:
    _extras = _run_theory_extras()
    save_cache("app_theory_extras", _EXTRAS_SIG, _extras)

VAR_ORDER = _extras["var_order"]
BLOCK_RES = _extras["block_res"]
COND_RES  = _extras["cond_res"]
CCM_RES   = _extras["ccm_res"]
n_block_sig = sum(1 for v in BLOCK_RES.values() if v["p"] < THRES)
n_cond_sig  = sum(1 for v in COND_RES.values()  if v["p"] < THRES)
n_ccm_conv  = sum(1 for v in CCM_RES.values()
                  if v["fwd"]["delta"] > 0.05 and v["fwd"]["rho_max"] > 0.3)
n_cond_lost = sum(1 for k, v in COND_RES.items()
                  if GRANGER_RES[k]["sig"] and v["p"] >= THRES)
n_robust    = sum(1 for v in GRANGER_RES.values()
                  if all(p < THRES for p in v["all_p"].values()))

COINT_OPTIONS = [{"label": k, "value": k} for k in COINT_RES]
GC_OPTIONS    = [{"label": k, "value": k} for k in CCM_RES]


# ── Theory-card figure builders ──────────────────────────────────────────────
def _ax(**extra):
    return dict(**_AX, tickfont=dict(size=10, color=MUTED2, family=FONT_MONO),
                title_font=dict(size=10, color=MUTED2, family=FONT_MONO), **extra)


def fig_ci_equilibrium(key: str) -> go.Figure:
    """CI(1,1) card: Y vs the long-run fit α+βX, and the equilibrium error
    ε̂ = Y − α − βX whose stationarity is what cointegration asserts."""
    r = COINT_RES[key]
    y = DF[r["c1"]].values.astype(float)
    x = DF[r["c2"]].values.astype(float)
    beta, alpha = np.polyfit(x, y, 1)
    fit = alpha + beta * x
    eps = y - fit
    step = max(1, len(y) // TS_POINTS)
    t = [str(v) for v in DF.index[::step]]
    sd = float(eps.std())
    hl = None
    e0, e1 = eps[:-1] - eps.mean(), eps[1:] - eps.mean()
    phi = float(np.dot(e0, e1) / max(np.dot(e0, e0), 1e-12))
    if 0 < phi < 1:
        hl = math.log(0.5) / math.log(phi)
    verdict = "cointegrated" if r["sig"] else "not cointegrated"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.55, 0.45],
                        subplot_titles=[
                            f"{r['label1']} and its long-run fit  α + β·{r['label2']}  (β = {beta:.3f})",
                            f"Equilibrium error  ε̂ = Y − α − βX  ·  EG p = {pf(r['p'])} → {verdict}"
                            + (f"  ·  half-life ≈ {hl:.1f} h" if hl else "")])
    fig.add_trace(go.Scatter(x=t, y=y[::step], mode="lines", name=r["label1"],
                             line=dict(color=ACCENT2, width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=fit[::step], mode="lines", name=f"α + β·{r['label2']}",
                             line=dict(color=AMBER, width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=eps[::step], mode="lines", name="ε̂",
                             line=dict(color=GREEN if r["sig"] else RED, width=1.1)), row=2, col=1)
    for k in (-2, 2):
        fig.add_hline(y=k * sd, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED2, width=1), row=2, col=1)
    fig.update_xaxes(**_ax())
    fig.update_yaxes(**_ax())
    fig.update_annotations(font=dict(size=12, color=TEXT))
    fig.update_layout(**_layout(height=520, showlegend=True,
                                legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)),
                                margin=dict(t=70, b=40, l=64, r=20)))
    return fig


def fig_granger_lag_robustness() -> go.Figure:
    """Key Caveats card: a credible Granger result should hold across lag
    choices. Heatmap of −log₁₀(p) for every pair at every lag 1..MAX_LAG_GC."""
    keys = sorted(GRANGER_RES, key=lambda k: GRANGER_RES[k]["min_p"])
    lags = [str(l) for l in range(1, MAX_LAG_GC + 1)]
    z, txt = [], []
    for k in keys:
        ps = [GRANGER_RES[k]["all_p"].get(l, np.nan) for l in lags]
        z.append([-math.log10(max(p, 1e-15)) for p in ps])
        txt.append(["✔" if p < THRES else "✘" for p in ps])
    labels = [f"{k}  {'· robust' if all(t == '✔' for t in tr) else ''}" for k, tr in zip(keys, txt)]
    thr = -math.log10(THRES)
    zmax = max(max(r) for r in z) if z else 1
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"lag {l} h" for l in lags], y=labels, text=txt, texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT),
        colorscale=[[0, "rgba(244,115,110,0.55)"], [min(thr / zmax, 1) * 0.999, "rgba(26,30,45,0.9)"],
                    [min(thr / zmax, 1), "rgba(79,142,247,0.45)"], [1, "rgba(45,212,160,0.95)"]],
        zmin=0, zmax=zmax,
        hovertemplate="<b>%{y}</b><br>%{x}<br>−log₁₀(p)=%{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="−log₁₀(p)", side="right",
                                 font=dict(size=11, color=MUTED2, family=FONT_MONO)),
                      tickfont=dict(size=10, color=MUTED2, family=FONT_MONO),
                      bgcolor=CARD, bordercolor=BORDER, borderwidth=1, thickness=14),
        xgap=3, ygap=3))
    fig.update_xaxes(**_ax())
    fig.update_yaxes(**_ax(autorange="reversed"))
    fig.update_layout(**_layout(
        height=520, hovermode="closest", margin=dict(t=50, b=50, l=380, r=80),
        title=dict(text="Lag robustness  ·  ✔ significant at 5% for that lag  ·  "
                        "'robust' = significant at every lag",
                   font=dict(size=14, color=TEXT), x=0.01)))
    return fig


def fig_var_block_heatmap() -> go.Figure:
    """VAR-GC card: does each block of variables (jointly) Granger-cause each
    target inside one VAR of all variables? −log₁₀(p) of the block Wald test."""
    blocks, targets = _extras["var_blocks"], _extras["var_targets"]
    z, txt = [], []
    for b in blocks:
        rz, rt = [], []
        for t in targets:
            v = BLOCK_RES.get((b, t))
            if v is None:
                rz.append(None); rt.append("—")
            else:
                rz.append(-math.log10(max(v["p"], 1e-15)))
                rt.append(f"p={pf(v['p'])}<br>χ²={v['stat']:.1f}, df={v['df']}")
        z.append(rz); txt.append(rt)
    fig = go.Figure(go.Heatmap(
        z=z, x=[LABELS.get(t, t) for t in targets], y=blocks, customdata=txt,
        colorscale=[[0, "rgba(26,30,45,0.9)"], [0.3, "rgba(79,142,247,0.4)"],
                    [0.7, "rgba(45,212,160,0.6)"], [1, "rgba(45,212,160,0.95)"]],
        hovertemplate="block <b>%{y}</b> → <b>%{x}</b><br>−log₁₀(p)=%{z:.2f}<br>%{customdata}<extra></extra>",
        colorbar=dict(title=dict(text="−log₁₀(p)", side="right",
                                 font=dict(size=11, color=MUTED2, family=FONT_MONO)),
                      tickfont=dict(size=10, color=MUTED2, family=FONT_MONO),
                      bgcolor=CARD, bordercolor=BORDER, borderwidth=1, thickness=14),
        xgap=3, ygap=3))
    # mark non-significant cells
    for i, b in enumerate(blocks):
        for j, t in enumerate(targets):
            v = BLOCK_RES.get((b, t))
            if v is not None and v["p"] >= THRES:
                fig.add_annotation(x=j, y=i, text="exog.", showarrow=False,
                                   font=dict(size=8, color=AMBER, family=FONT_MONO))
    fig.update_xaxes(**_ax(tickangle=40))
    fig.update_yaxes(**_ax())
    fig.update_layout(**_layout(
        height=430, hovermode="closest", margin=dict(t=50, b=150, l=130, r=80),
        title=dict(text=f"VAR({VAR_ORDER}) block exogeneity  ·  rows = causing block, "
                        "columns = target  ·  'exog.' = block adds nothing at 5%",
                   font=dict(size=14, color=TEXT), x=0.01)))
    return fig


def fig_var_vs_pairwise() -> go.Figure:
    """VAR-GC card: the same pair tested bivariately vs. conditioned on every
    other variable in the VAR. A link that disappears in the system is a
    candidate omitted-variable artefact."""
    keys = sorted(COND_RES, key=lambda k: GRANGER_RES[k]["min_p"])
    pw = [-math.log10(max(GRANGER_RES[k]["min_p"], 1e-15)) for k in keys]
    cd = [-math.log10(max(COND_RES[k]["p"], 1e-15)) for k in keys]
    fig = go.Figure()
    for k, a, b in zip(keys, pw, cd):
        fig.add_trace(go.Scatter(x=[a, b], y=[k, k], mode="lines",
                                 line=dict(color=MUTED, width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pw, y=keys, mode="markers", name="pairwise (best lag)",
                             marker=dict(color=ACCENT2, size=11, symbol="circle"),
                             hovertemplate="%{y}<br>pairwise −log₁₀(p)=%{x:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=cd, y=keys, mode="markers", name=f"conditional on full VAR({VAR_ORDER})",
                             marker=dict(color=[GREEN if COND_RES[k]["p"] < THRES else RED for k in keys],
                                         size=11, symbol="diamond"),
                             hovertemplate="%{y}<br>VAR-conditional −log₁₀(p)=%{x:.2f}<extra></extra>"))
    fig.add_vline(x=-math.log10(THRES), line=dict(color=AMBER, width=1.5, dash="dot"),
                  annotation_text="  α=0.05", annotation_font=dict(color=AMBER, size=10))
    fig.update_xaxes(**_ax(title=dict(text="−log₁₀(p)  ·  log scale", standoff=6), type="log"))
    fig.update_yaxes(**_ax(autorange="reversed"))
    fig.update_layout(**_layout(
        height=520, hovermode="closest", showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        margin=dict(t=70, b=50, l=330, r=40),
        title=dict(text="Pairwise vs system-conditional Granger causality  ·  "
                        "diamond red = link vanishes once other variables are included",
                   font=dict(size=14, color=TEXT), x=0.01)))
    return fig


def fig_ccm_curves(key: str) -> go.Figure:
    """CCM card: cross-map skill ρ vs library size L, both directions."""
    r = CCM_RES[key]
    g = GRANGER_RES[key]
    fig = go.Figure()
    for d, name, clr, fill in [
        ("fwd", f"{g['label_c']} → {g['label_e']}  (xmap {g['label_c']} from M[{g['label_e']}])",
         GREEN, "rgba(45,212,160,0.15)"),
        ("rev", f"{g['label_e']} → {g['label_c']}  (xmap {g['label_e']} from M[{g['label_c']}])",
         ACCENT2, "rgba(126,179,255,0.15)"),
    ]:
        v = r[d]
        fig.add_trace(go.Scatter(x=v["L"] + v["L"][::-1], y=v["hi"] + v["lo"][::-1],
                                 fill="toself", fillcolor=fill, line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=v["L"], y=v["rho"], mode="lines+markers",
            name=f"{name}  · E={v['E']}, τ={v['tau']}",
            line=dict(color=clr, width=2), marker=dict(size=6),
            hovertemplate="L=%{x}<br>ρ=%{y:.3f}<extra></extra>"))
    fig.update_xaxes(**_ax(type="log", title=dict(text="library size L (points, log scale)", standoff=6)))
    fig.update_yaxes(**_ax(title=dict(text="cross-map skill ρ = corr(X, X̂)", standoff=6)))
    fig.update_layout(**_layout(
        height=430, hovermode="closest", showlegend=True,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        margin=dict(t=50, b=110, l=70, r=20),
        title=dict(text=f"CCM convergence  ·  {key}  ·  shaded = interquartile range over "
                        f"{CCM_N_SAMPLES} random libraries",
                   font=dict(size=14, color=TEXT), x=0.01)))
    return fig


def fig_ccm_summary() -> go.Figure:
    """CCM card: final skill ρ(L_max) in both directions for every pair,
    next to the pairwise Granger verdict."""
    keys = sorted(CCM_RES, key=lambda k: -CCM_RES[k]["fwd"]["rho_max"])
    fwd = [CCM_RES[k]["fwd"]["rho_max"] for k in keys]
    rev = [CCM_RES[k]["rev"]["rho_max"] for k in keys]
    conv = [CCM_RES[k]["fwd"]["delta"] for k in keys]
    gc = ["Granger ✔" if GRANGER_RES[k]["sig"] else "Granger ✘" for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=keys, x=fwd, orientation="h", name="ρ  cause → effect",
                         marker_color=GREEN, text=[f"Δρ={c:+.2f} · {g}" for c, g in zip(conv, gc)],
                         textposition="outside", textfont=dict(size=9, color=MUTED2, family=FONT_MONO),
                         hovertemplate="%{y}<br>ρ(cause→effect)=%{x:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(y=keys, x=rev, orientation="h", name="ρ  effect → cause",
                         marker_color=ACCENT2,
                         hovertemplate="%{y}<br>ρ(effect→cause)=%{x:.3f}<extra></extra>"))
    fig.update_xaxes(**_ax(range=[0, 1.25], title=dict(text="cross-map skill ρ at the full library", standoff=6)))
    fig.update_yaxes(**_ax(autorange="reversed"))
    fig.update_layout(**_layout(
        height=560, hovermode="closest", showlegend=True, barmode="group", bargap=0.25,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=10)),
        margin=dict(t=70, b=50, l=330, r=20),
        title=dict(text="CCM summary  ·  Δρ = gain from the smallest to the full library "
                        "(convergence)", font=dict(size=14, color=TEXT), x=0.01)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  ONE PLOT PER THEORY CARD — Stationarity (ADF, KPSS, PP, ZA) and
#  Diagnostics (KS, SW, JB, LB, HQC, BP, BG, DW). Each figure illustrates the
#  quantity its card defines, for the variable chosen in the tab's dropdown,
#  using the same test settings as the battery above.
# ─────────────────────────────────────────────────────────────────────────────
from functools import lru_cache

PP_BANDWIDTHS = [0, 2, 4, 8, 12, 16, 24, 32, 48]
ZA_STEP       = 10       # evaluate every 10th candidate break date
ZA_TRIM       = 0.15     # same trimming as statsmodels' zivot_andrews
LB_MAX_LAG    = 48
CARD_H        = 360      # height of each per-card figure


def _series(col):
    return DF[col].dropna().values.astype(float)


def _thin(n, max_pts=TS_POINTS):
    return slice(None, None, max(1, n // max_pts))


def _card_layout(fig, title, height=CARD_H, **kw):
    fig.update_xaxes(**_ax())
    fig.update_yaxes(**_ax())
    fig.update_annotations(font=dict(size=11, color=TEXT))
    fig.update_layout(**_layout(height=height, hovermode="closest",
                                margin=dict(t=64, b=44, l=60, r=20),
                                title=dict(text=title, font=dict(size=13, color=TEXT), x=0.01),
                                **kw))
    return fig


def _add_gauge(fig, row, col, stat, crits: dict, reject: str):
    """Number line: the test statistic against its 1/5/10% critical values.
    reject='left' → rejection region below the critical value (ADF, PP, ZA);
    'right' → above it (KPSS)."""
    vals = [stat] + list(crits.values())
    lo, hi = min(vals), max(vals)
    pad = 0.25 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    c5 = crits.get("5%")
    if c5 is not None:
        x0, x1 = (lo, c5) if reject == "left" else (c5, hi)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.5, y1=0.5,
                      fillcolor="rgba(45,212,160,0.12)", line=dict(width=0), row=row, col=col)
    for lvl, v in crits.items():
        fig.add_trace(go.Scatter(x=[v, v], y=[-0.45, 0.45], mode="lines+text",
                                 line=dict(color=AMBER, width=1.5, dash="dot"),
                                 text=[lvl, ""], textposition="top center",
                                 textfont=dict(size=9, color=AMBER),
                                 hovertemplate=f"{lvl} critical value = {v:.3f}<extra></extra>",
                                 showlegend=False), row=row, col=col)
    rej = (stat < c5) if reject == "left" else (stat > c5)
    fig.add_trace(go.Scatter(x=[stat], y=[0], mode="markers+text",
                             marker=dict(size=14, color=GREEN if rej else RED, symbol="diamond"),
                             text=[f"{stat:.3f}"], textposition="bottom center",
                             textfont=dict(size=10, color=TEXT),
                             hovertemplate=f"statistic = {stat:.3f}<extra></extra>",
                             showlegend=False), row=row, col=col)
    fig.update_xaxes(range=[lo, hi], row=row, col=col)
    fig.update_yaxes(visible=False, range=[-1, 1], row=row, col=col)


# ── Stationarity cards ───────────────────────────────────────────────────────
def fig_card_adf(col):
    """ADF card: Δy_t against y_{t-1}; the slope γ is what the test checks
    for being < 0 (mean reversion), via τ = γ̂ / se(γ̂)."""
    s = _series(col)
    r = STAT_RES[col]
    stat, p, crit, res = adfuller(s, maxlag=r["adf_lags"], autolag=None,
                                        regresults=True)
    gamma = float(res.resols.params[0])
    dy, ylag = np.diff(s), s[:-1]
    sl = _thin(len(dy))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.1,
                        subplot_titles=[f"Δy_t vs y_(t−1)  ·  γ̂ = {gamma:.4f}",
                                        f"τ = {stat:.2f}  ·  p = {pf(p)}"])
    fig.add_trace(go.Scattergl(x=ylag[sl], y=dy[sl], mode="markers",
                               marker=dict(size=3, color=ACCENT2, opacity=0.45),
                               hoverinfo="skip"), row=1, col=1)
    xs = np.linspace(ylag.min(), ylag.max(), 2)
    b = np.polyfit(ylag, dy, 1)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines",
                             line=dict(color=AMBER, width=2), hoverinfo="skip"), row=1, col=1)
    _add_gauge(fig, 1, 2, stat, crit, "left")
    fig.update_xaxes(title=dict(text="y_(t−1)", standoff=4), row=1, col=1)
    fig.update_yaxes(title=dict(text="Δy_t", standoff=4), row=1, col=1)
    verdict = "stationary ✔" if r["adf_ok"] else "unit root ✘"
    return _card_layout(fig, f"ADF  ·  negative slope = pull back to the mean  ·  {verdict}")


def fig_card_kpss(col):
    """KPSS card: partial sums S_t of the demeaned series; η is their scaled
    sum of squares, large when the level drifts."""
    s = _series(col)
    stat, p, lags, crit = kpss(s, regression="c", nlags="auto")
    e = s - s.mean()
    S = np.cumsum(e)
    sl = _thin(len(S))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.1,
                        subplot_titles=["Partial sums  S_t = Σ (y_i − ȳ)",
                                        f"η = {stat:.3f}  ·  p = {pf(p)}  (bandwidth {lags})"])
    fig.add_trace(go.Scatter(x=[str(t) for t in DF.index[:len(S)][sl]], y=S[sl], mode="lines",
                             line=dict(color=GREEN, width=1.4), fill="tozeroy",
                             fillcolor="rgba(45,212,160,0.08)", hoverinfo="skip"), row=1, col=1)
    crit = {k: v for k, v in crit.items() if k in ("10%", "5%", "1%")}
    _add_gauge(fig, 1, 2, stat, crit, "right")
    verdict = "stationary ✔" if STAT_RES[col]["kpss_ok"] else "not stationary ✘"
    return _card_layout(fig, f"KPSS  ·  a drifting S_t means a wandering level  ·  {verdict}")


def fig_card_pp(col):
    """PP card: the non-parametric Newey-West correction — Z(t) as the
    bandwidth changes; a robust verdict does not hinge on the bandwidth."""
    s = _series(col)
    zs, crit5 = [], None
    for L in PP_BANDWIDTHS:
        try:
            t = PhillipsPerron(s, lags=L)
            zs.append(float(t.stat)); crit5 = float(t.critical_values["5%"])
        except Exception:
            zs.append(np.nan)
    base = PhillipsPerron(s)
    fig = go.Figure(go.Scatter(x=PP_BANDWIDTHS, y=zs, mode="lines+markers",
                               line=dict(color=AMBER, width=2), marker=dict(size=7),
                               hovertemplate="bandwidth %{x}<br>Z(t) = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[base.lags], y=[base.stat], mode="markers",
                             marker=dict(size=14, symbol="diamond",
                                         color=GREEN if STAT_RES[col]["pp_ok"] else RED),
                             hovertemplate="default bandwidth %{x}<br>Z(t) = %{y:.3f}<extra></extra>"))
    if crit5 is not None:
        fig.add_hline(y=crit5, line=dict(color=AMBER, width=1.5, dash="dot"),
                      annotation_text=f"5% critical value {crit5:.2f}  ·  below = stationary",
                      annotation_font=dict(color=AMBER, size=10))
    fig.update_xaxes(title=dict(text="Newey-West bandwidth ℓ (lags)", standoff=4))
    fig.update_yaxes(title=dict(text="Z(t)", standoff=4))
    verdict = "stationary ✔" if STAT_RES[col]["pp_ok"] else "unit root ✘"
    return _card_layout(fig, f"Phillips-Perron  ·  Z(t) vs bandwidth (diamond = default ℓ={base.lags})  ·  {verdict}")


@lru_cache(maxsize=32)
def _za_scan(col):
    """t-statistic of the unit-root coefficient for every candidate break
    (intercept + trend break, the 'ct' model used in the battery)."""
    s = _series(col)
    stat, p, crit, baselag, bpidx = zivot_andrews(s, maxlag=6, regression="ct")
    n = len(s)
    dy = np.diff(s)
    rows = np.arange(baselag + 1, n)                # usable observations (index in s)
    Y = s[rows]
    t = rows.astype(float)
    lagd = [dy[rows - 1 - j] for j in range(1, baselag + 1)]
    lo, hi = int(ZA_TRIM * n), int((1 - ZA_TRIM) * n)
    bps, tstats = [], []
    for bp in range(lo, hi, ZA_STEP):
        du = (rows > bp).astype(float)
        dt = np.where(rows > bp, t - bp, 0.0)
        X = np.column_stack([np.ones_like(t), t, du, dt, s[rows - 1], *lagd])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        s2 = res @ res / (len(Y) - X.shape[1])
        cov = s2 * np.linalg.pinv(X.T @ X)
        bps.append(bp); tstats.append(float((beta[4] - 1) / np.sqrt(cov[4, 4])))
    return dict(stat=float(stat), p=float(p), crit=dict(crit), bpidx=int(bpidx),
                bps=bps, tstats=tstats)


def fig_card_za(col):
    """ZA card: the break date is chosen where the unit-root t-statistic is
    most negative; the series is shown with that break."""
    z = _za_scan(col)
    s = _series(col)
    idx = DF.index[:len(s)]
    sl = _thin(len(s))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=[f"Series with the estimated break ({idx[z['bpidx']]:%Y-%m-%d %H:%M})",
                                        f"t-statistic over candidate break dates  ·  min = {z['stat']:.2f}, p = {pf(z['p'])}"])
    fig.add_trace(go.Scatter(x=[str(v) for v in idx[sl]], y=s[sl], mode="lines",
                             line=dict(color="#a78bfa", width=1.2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[str(idx[b]) for b in z["bps"]], y=z["tstats"], mode="lines",
                             line=dict(color=ACCENT2, width=1.5),
                             hovertemplate="break %{x}<br>t = %{y:.2f}<extra></extra>"), row=2, col=1)
    for r_ in (1, 2):
        fig.add_vline(x=str(idx[z["bpidx"]]), line=dict(color=RED, width=1.5, dash="dash"), row=r_, col=1)
    fig.add_hline(y=z["crit"]["5%"], line=dict(color=AMBER, width=1.5, dash="dot"),
                  annotation_text=f"5% critical value {z['crit']['5%']:.2f}",
                  annotation_font=dict(color=AMBER, size=10), row=2, col=1)
    verdict = "stationary with a break ✔" if STAT_RES[col]["za_ok"] else "unit root ✘"
    return _card_layout(fig, f"Zivot-Andrews  ·  break = most negative t  ·  {verdict}", height=CARD_H + 80)


# ── Diagnostics cards (normality / autocorrelation on the series) ────────────
def fig_card_ks(col):
    """KS card: empirical CDF vs the fitted normal CDF; D is the largest gap."""
    d = DIAG_RES[col]
    s = np.sort(_series(col))
    n = len(s)
    F = np.arange(1, n + 1) / n
    Phi = sp_stats.norm.cdf(s, d["mu"], d["sigma"])
    gap = np.maximum(np.abs(F - Phi), np.abs(F - 1 / n - Phi))
    i = int(np.argmax(gap))
    sl = _thin(n, 2000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s[sl], y=F[sl], mode="lines", name="empirical F_n(x)",
                             line=dict(color=ACCENT2, width=2, shape="hv")))
    fig.add_trace(go.Scatter(x=s[sl], y=Phi[sl], mode="lines", name="normal Φ(x; μ̂, σ̂)",
                             line=dict(color=AMBER, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[s[i], s[i]], y=[F[i], Phi[i]], mode="lines+markers",
                             name=f"D = {d['ks_stat']:.3f}", line=dict(color=RED, width=3),
                             marker=dict(size=7)))
    fig.update_xaxes(title=dict(text=LABELS.get(col, col), standoff=4))
    fig.update_yaxes(title=dict(text="cumulative probability", standoff=4))
    verdict = "normal ✔" if d["ks_ok"] else "not normal ✘"
    return _card_layout(fig, f"Kolmogorov-Smirnov  ·  D = largest CDF gap, p = {pf(d['ks_p'])}  ·  {verdict}",
                        showlegend=True, legend=dict(x=0.01, y=0.98, font=dict(size=10)))


def fig_card_sw(col):
    """SW card: W is (roughly) the squared correlation of the normal Q-Q plot."""
    d = DIAG_RES[col]
    s = _series(col)[:d["sw_n"]]
    (osm, osr), (slope, icpt, r) = sp_stats.probplot(s, dist="norm")
    sl = _thin(len(osm), 2000)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=osm[sl], y=osr[sl], mode="markers",
                               marker=dict(size=4, color=GREEN, opacity=0.6), name="ordered data"))
    xs = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(x=xs, y=icpt + slope * xs, mode="lines", name="normal line",
                             line=dict(color=AMBER, width=2, dash="dash")))
    fig.update_xaxes(title=dict(text="theoretical normal quantiles", standoff=4))
    fig.update_yaxes(title=dict(text="ordered observations", standoff=4))
    verdict = "normal ✔" if d["sw_ok"] else "not normal ✘"
    return _card_layout(fig, f"Shapiro-Wilk  ·  W = {d['sw_stat']:.4f} (Q-Q r² = {r**2:.4f}), "
                             f"p = {pf(d['sw_p'])}  ·  {verdict}",
                        showlegend=True, legend=dict(x=0.01, y=0.98, font=dict(size=10)))


def fig_card_jb(col):
    """JB card: histogram vs normal, and how much of JB comes from skewness
    versus excess kurtosis: JB = n/6 · (S² + K²/4)."""
    d = DIAG_RES[col]
    s = _series(col)
    n = len(s)
    js, jk = n / 6 * d["skew"] ** 2, n / 24 * d["kurt"] ** 2
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.12,
                        subplot_titles=[f"Histogram vs normal  ·  S = {d['skew']:.2f}, K = {d['kurt']:.2f}",
                                        f"JB = {d['jb_stat']:.0f} split by source"])
    fig.add_trace(go.Histogram(x=s, nbinsx=60, histnorm="probability density",
                               marker_color=AMBER, opacity=0.6, hoverinfo="skip"), row=1, col=1)
    xs = np.linspace(s.min(), s.max(), 200)
    fig.add_trace(go.Scatter(x=xs, y=sp_stats.norm.pdf(xs, d["mu"], d["sigma"]), mode="lines",
                             line=dict(color=ACCENT2, width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Bar(x=["skewness  n·S²/6", "kurtosis  n·K²/24"], y=[js, jk],
                         marker_color=[ACCENT2, "#a78bfa"], text=[f"{js:.0f}", f"{jk:.0f}"],
                         textposition="outside", hovertemplate="%{x}: %{y:.1f}<extra></extra>"),
                  row=1, col=2)
    crit = sp_stats.chi2.ppf(0.95, 2)
    fig.add_hline(y=crit, line=dict(color=RED, width=1.5, dash="dot"), row=1, col=2,
                  annotation_text=f"χ²₂ 5% = {crit:.2f}", annotation_font=dict(color=RED, size=10))
    fig.update_yaxes(type="log", row=1, col=2)
    verdict = "normal ✔" if d["jb_ok"] else "not normal ✘"
    return _card_layout(fig, f"Jarque-Bera  ·  p = {pf(d['jb_p'])}  ·  {verdict}")


def fig_card_lb(col):
    """LB card: Q(h) for every h up to 48 against the χ²_h 95% critical curve."""
    d = DIAG_RES[col]
    s = _series(col)
    lb = acorr_ljungbox(s, lags=LB_MAX_LAG, return_df=True)
    h = lb.index.values
    crit = sp_stats.chi2.ppf(0.95, h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h, y=lb["lb_stat"], mode="lines+markers", name="Q(h)",
                             line=dict(color="#60a5fa", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=h, y=crit, mode="lines", name="χ²_h 95% critical value",
                             line=dict(color=AMBER, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=LB_LAGS, y=[lb.loc[l, "lb_stat"] for l in LB_LAGS], mode="markers",
                             name="lags used in the battery",
                             marker=dict(size=11, symbol="diamond", color=GREEN if d["lb_ok"] else RED)))
    fig.update_xaxes(title=dict(text="lag h (hours)", standoff=4))
    fig.update_yaxes(type="log", title=dict(text="Q(h)  (log scale)", standoff=4))
    verdict = "white noise ✔" if d["lb_ok"] else "autocorrelated ✘"
    return _card_layout(fig, f"Ljung-Box  ·  Q above the curve = autocorrelation  ·  {verdict}",
                        showlegend=True, legend=dict(x=0.01, y=0.98, font=dict(size=10)))


def fig_card_hqc(col):
    """HQC card: the criterion for each AR order; the minimum is the pick."""
    d = DIAG_RES[col]
    p = sorted(d["hqc_by_lag"])
    v = [d["hqc_by_lag"][k] for k in p]
    best = d["best_hqc_lag"]
    fig = go.Figure(go.Scatter(x=p, y=v, mode="lines+markers", line=dict(color="#a78bfa", width=2),
                               marker=dict(size=7), hovertemplate="AR(%{x})  HQC = %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[best], y=[d["best_hqc_val"]], mode="markers+text",
                             marker=dict(size=15, symbol="star", color=GREEN),
                             text=[f"AR({best})"], textposition="top center",
                             textfont=dict(color=GREEN, size=11)))
    fig.update_xaxes(title=dict(text="AR order p", standoff=4), dtick=1)
    fig.update_yaxes(title=dict(text="HQC = −2 ln L + 2k ln ln n", standoff=4))
    return _card_layout(fig, f"Hannan-Quinn  ·  minimum at p = {best}")


# ── Diagnostics cards (regression residuals of y_t = α + βt + ε_t) ──────────
@lru_cache(maxsize=32)
def _trend_resid(col):
    s = _series(col)
    X = sm.add_constant(np.arange(len(s), dtype=float))
    return sm.OLS(s, X).fit().resid


def fig_card_bp(col):
    """BP card: the auxiliary regression of ε̂² on the regressors (here the
    time trend); LM = n·R² of that fit."""
    r = REGDIAG_RES[col]
    e2 = _trend_resid(col) ** 2
    n = len(e2)
    t = np.arange(n)
    b = np.polyfit(t, e2, 1)
    roll = pd.Series(e2).rolling(24 * 7, center=True, min_periods=24).mean().values
    idx = [str(v) for v in DF.index[:n]]
    sl = _thin(n)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=idx[sl], y=e2[sl], mode="markers", name="ε̂²",
                               marker=dict(size=3, color=AMBER, opacity=0.4)))
    fig.add_trace(go.Scatter(x=idx[sl], y=roll[sl], mode="lines", name="7-day mean of ε̂²",
                             line=dict(color=ACCENT2, width=2)))
    fig.add_trace(go.Scatter(x=[idx[0], idx[-1]], y=np.polyval(b, [0, n - 1]), mode="lines",
                             name="auxiliary fit ε̂² = a + b·t", line=dict(color=RED, width=2, dash="dash")))
    fig.update_yaxes(type="log", title=dict(text="squared residual (log scale)", standoff=4))
    verdict = "homoskedastic ✔" if r["bp_ok"] else "heteroskedastic ✘"
    return _card_layout(fig, f"Breusch-Pagan  ·  LM = n·R² = {r['bp_stat']:.1f}, p = {pf(r['bp_p'])}  ·  {verdict}",
                        showlegend=True, legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)))


def fig_card_bg(col):
    """BG card: autocorrelation of the residuals at lags 1..BG_LAGS, the
    lags the auxiliary regression adds."""
    r = REGDIAG_RES[col]
    e = _trend_resid(col)
    a = acf(e, nlags=BG_LAGS, fft=True)[1:]
    conf = 1.96 / math.sqrt(len(e))
    lags = list(range(1, BG_LAGS + 1))
    fig = go.Figure(go.Bar(x=lags, y=a, marker_color=[SIG_C if abs(v) > conf else NS_C for v in a],
                           marker_line=dict(color=BORDER, width=1),
                           hovertemplate="lag %{x}: r = %{y:.3f}<extra></extra>"))
    for y in (conf, -conf):
        fig.add_hline(y=y, line=dict(color=CI_C, width=1.2, dash="dash"))
    fig.update_xaxes(title=dict(text="lag (hours)", standoff=4), dtick=1)
    fig.update_yaxes(title=dict(text="residual autocorrelation", standoff=4))
    verdict = "no serial correlation ✔" if r["bg_ok"] else "serially correlated ✘"
    return _card_layout(fig, f"Breusch-Godfrey (lags 1–{BG_LAGS})  ·  LM = {r['bg_stat']:.0f}, "
                             f"p = {pf(r['bg_p'])}  ·  {verdict}")


def fig_card_dw(col):
    """DW card: ε̂_t against ε̂_(t−1); DW ≈ 2(1 − ρ̂) where ρ̂ is the slope."""
    r = REGDIAG_RES[col]
    e = _trend_resid(col)
    x, y = e[:-1], e[1:]
    rho = float(np.dot(x, y) / np.dot(x, x))
    sl = _thin(len(x), 2500)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x[sl], y=y[sl], mode="markers",
                               marker=dict(size=3, color=GREEN, opacity=0.45), hoverinfo="skip"))
    xs = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=xs, y=rho * xs, mode="lines", line=dict(color=AMBER, width=2),
                             name=f"slope ρ̂ = {rho:.3f}"))
    fig.update_xaxes(title=dict(text="ε̂_(t−1)", standoff=4))
    fig.update_yaxes(title=dict(text="ε̂_t", standoff=4))
    verdict = "≈ 2, no lag-1 correlation ✔" if r["dw_ok"] else "far from 2 ✘"
    return _card_layout(fig, f"Durbin-Watson  ·  DW = {r['dw_stat']:.3f} ≈ 2(1 − ρ̂) = {2*(1-rho):.3f}  ·  {verdict}")


STAT_CARD_FIGS = [("ADF", "card-adf", fig_card_adf), ("KPSS", "card-kpss", fig_card_kpss),
                  ("PP", "card-pp", fig_card_pp), ("ZA", "card-za", fig_card_za)]
DIAG_CARD_FIGS = [("KS", "card-ks", fig_card_ks), ("SW", "card-sw", fig_card_sw),
                  ("JB", "card-jb", fig_card_jb), ("LB", "card-lb", fig_card_lb),
                  ("HQC", "card-hqc", fig_card_hqc), ("BP", "card-bp", fig_card_bp),
                  ("BG", "card-bg", fig_card_bg), ("DW", "card-dw", fig_card_dw)]
CARD_NAMES = {"ADF": "Augmented Dickey-Fuller", "KPSS": "KPSS", "PP": "Phillips-Perron",
              "ZA": "Zivot-Andrews", "KS": "Kolmogorov-Smirnov", "SW": "Shapiro-Wilk",
              "JB": "Jarque-Bera", "LB": "Ljung-Box", "HQC": "Hannan-Quinn",
              "BP": "Breusch-Pagan", "BG": "Breusch-Godfrey", "DW": "Durbin-Watson"}


def card_plot_grid(specs):
    """Two-column grid of plot cards, each labelled with its theory card."""
    return html.Div([
        html.Div(plot_card([dcc.Graph(id=gid, config=PLOTLY_CFG)],
                           label=f"Theory card: {CARD_NAMES[tag]} ({tag})"),
                 style={"flex": "1 1 560px", "minWidth": "0"})
        for tag, gid, _ in specs
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "flex-start"})


def _safe(fn, col):
    try:
        return fn(col)
    except Exception as exc:                      # keep the rest of the grid alive
        f = go.Figure()
        f.add_annotation(text=f"Could not compute for this variable: {exc}", showarrow=False,
                         font=dict(color=MUTED2, size=12))
        f.update_xaxes(visible=False); f.update_yaxes(visible=False)
        return f.update_layout(**_layout(height=CARD_H))

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


def fig_johansen_bars() -> go.Figure:
    """Grouped bar chart: Johansen trace statistic vs 5% critical value for each pair."""
    if not JOHANSEN_RES:
        return go.Figure()
    keys   = list(JOHANSEN_RES.keys())
    sigs   = [JOHANSEN_RES[k]["sig"] for k in keys]
    colors = [GREEN if s else RED for s in sigs]
    labels = [
        f"{JOHANSEN_RES[k]['label1']}<br><span style='opacity:.7'>↔</span> {JOHANSEN_RES[k]['label2']}"
        for k in keys
    ]
    trace_s  = [JOHANSEN_RES[k]["trace_stat0"] for k in keys]
    trace_c  = [JOHANSEN_RES[k]["trace_crit0"] for k in keys]
    maxeig_s = [JOHANSEN_RES[k]["maxeig_stat0"] for k in keys]
    maxeig_c = [JOHANSEN_RES[k]["maxeig_crit0"] for k in keys]
    hover = [
        f"<b>{k}</b><br>"
        f"Trace(r=0): {sf(ts)}  (5% crit: {sf(tc)})<br>"
        f"MaxEig(r=0): {sf(ms)}  (5% crit: {sf(mc)})<br>"
        f"Rank: {JOHANSEN_RES[k]['rank']}<br>"
        f"{'Cointegrated ✔' if sg else 'Not cointegrated ✘'}<extra></extra>"
        for k, ts, tc, ms, mc, sg in zip(keys, trace_s, trace_c, maxeig_s, maxeig_c, sigs)
    ]
    idx = sorted(range(len(keys)), key=lambda i: trace_s[i], reverse=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[labels[i] for i in idx],
        y=[trace_s[i] for i in idx],
        name="Trace statistic",
        marker_color=[colors[i] for i in idx],
        marker_line=dict(color=BORDER, width=1),
        hovertemplate=[hover[i] for i in idx],
        text=[f"r={JOHANSEN_RES[keys[i]]['rank']}" for i in idx],
        textposition="outside",
        textfont=dict(size=9, color=MUTED2, family=FONT_MONO),
    ))
    fig.add_trace(go.Scatter(
        x=[labels[i] for i in idx],
        y=[trace_c[i] for i in idx],
        name="5% critical value",
        mode="markers",
        marker=dict(color=AMBER, size=8, symbol="line-ew", line=dict(width=2, color=AMBER)),
        hovertemplate="5% crit: %{y:.2f}<extra></extra>",
    ))
    ax_kw = dict(**_AX, tickfont=dict(size=9, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, tickangle=35)
    fig.update_yaxes(**ax_kw, title=dict(text="Trace statistic (H₀: rank=0)", standoff=6))
    fig.update_layout(**_layout(
        height=500,
        margin=dict(t=46, b=170, l=70, r=20),
        showlegend=True,
        legend=dict(
            orientation="h", x=0.01, y=1.08,
            font=dict(size=11, color=MUTED2, family=FONT_MONO),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(text="Johansen Cointegration  ·  Trace Statistics (H₀: rank=0)  ·  amber marker = 5% critical value",
                   font=dict(size=13, color=TEXT), x=0.01),
        bargap=0.32,
    ))
    return fig


def fig_johansen_rank_heatmap() -> go.Figure:
    """Heatmap showing Johansen rank (0, 1, 2) for each pair."""
    if not JOHANSEN_RES:
        return go.Figure()
    keys   = list(JOHANSEN_RES.keys())
    ranks  = [JOHANSEN_RES[k]["rank"] for k in keys]
    labels = [f"{JOHANSEN_RES[k]['label1']} ↔ {JOHANSEN_RES[k]['label2']}" for k in keys]
    rank_colors = {0: "rgba(244,115,110,0.55)", 1: "rgba(79,142,247,0.6)", 2: "rgba(45,212,160,0.7)"}
    colors = [rank_colors.get(r, MUTED) for r in ranks]
    rank_labels = {0: "No coint.", 1: "1 vector", 2: "2 vectors"}
    fig = go.Figure(go.Bar(
        x=labels,
        y=ranks,
        marker_color=colors,
        marker_line=dict(color=BORDER, width=1),
        text=[rank_labels.get(r, str(r)) for r in ranks],
        textposition="outside",
        textfont=dict(size=9, color=MUTED2, family=FONT_MONO),
        hovertemplate="%{x}<br>Rank: %{y}<extra></extra>",
        showlegend=False,
    ))
    ax_kw = dict(**_AX, tickfont=dict(size=9, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, tickangle=35)
    fig.update_yaxes(**ax_kw,
                     title=dict(text="Cointegration rank", standoff=5),
                     tickvals=[0, 1, 2],
                     ticktext=["0  (none)", "1  (one vector)", "2  (two vectors)"])
    fig.update_layout(**_layout(
        height=380,
        margin=dict(t=40, b=170, l=80, r=20),
        title=dict(text="Johansen Cointegration  ·  Estimated Rank per Pair",
                   font=dict(size=13, color=TEXT), x=0.01),
        bargap=0.35,
    ))
    return fig


def fig_po_bars() -> go.Figure:
    """Horizontal bar chart of Phillips-Ouliaris Z(t) statistics."""
    if not PO_RES:
        return go.Figure()
    keys   = list(PO_RES.keys())
    stats_ = [PO_RES[k]["stat"]  for k in keys]
    pvals  = [PO_RES[k]["p"]     for k in keys]
    sigs   = [PO_RES[k]["sig"]   for k in keys]
    colors = [GREEN if s else RED for s in sigs]
    labels = [
        f"{PO_RES[k]['label1']}<br><span style='opacity:.7'>↔</span> {PO_RES[k]['label2']}"
        for k in keys
    ]
    hover = [
        f"<b>{k}</b><br>Z(t): {sf(s)}<br>p={pf(p)}<br>"
        f"{'Cointegrated ✔' if sg else 'Not cointegrated ✘'}<extra></extra>"
        for k, s, p, sg in zip(keys, stats_, pvals, sigs)
    ]
    idx = sorted(range(len(keys)), key=lambda i: stats_[i])
    fig = go.Figure(go.Bar(
        x=[stats_[i] for i in idx],
        y=[labels[i] for i in idx],
        orientation="h",
        marker_color=[colors[i] for i in idx],
        marker_line=dict(color=BORDER, width=1),
        hovertemplate=[hover[i] for i in idx],
        text=[f"p={pf(pvals[i])}" for i in idx],
        textposition="outside",
        textfont=dict(size=10, color=MUTED2, family=FONT_MONO),
        showlegend=False,
    ))
    ax_kw = dict(**_AX, tickfont=dict(size=10, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, title=dict(
        text="PO Z(t) statistic  (more negative = stronger cointegration)", standoff=6))
    fig.update_yaxes(**ax_kw)
    fig.update_layout(**_layout(
        height=500, margin=dict(t=46, b=50, l=280, r=90),
        title=dict(text="Phillips-Ouliaris  ·  Z(t) Statistics",
                   font=dict(size=14, color=TEXT), x=0.01),
        bargap=0.28,
    ))
    return fig


def fig_po_pvals() -> go.Figure:
    """Scatter plot of Phillips-Ouliaris p-values."""
    if not PO_RES:
        return go.Figure()
    keys   = list(PO_RES.keys())
    pvals  = [PO_RES[k]["p"]   for k in keys]
    sigs   = [PO_RES[k]["sig"] for k in keys]
    labels = [f"{PO_RES[k]['label1']} ↔ {PO_RES[k]['label2']}" for k in keys]
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
        title=dict(text="Phillips-Ouliaris  ·  p-values  ·  green = cointegrated at 5%",
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
    """Heatmap: all 5 diagnostics × all variables (pass/fail overview)."""
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

    # HQC best lag row (numeric, not pass/fail: encode 1..12 → 0..1)
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
        text=[LABELS.get(c, c) for c in cols],
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
            "Amplitude PDF: P(|X|>x) — GGD / t / Laplace", "Q-Q Plot", "Skewness & Kurtosis",
            "Ljung-Box p-values by lag", "HQC by AR order", "Test Summary",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "table"}],
        ],
    )

    # ── (1,1) Amplitude Probability Density: P(|X − center| > x) ───────────
    # APD (as used in EMI/vibration analysis): for each level x on the axis,
    # the probability that the deviation of the variable exceeds x in
    # absolute value — i.e. the survival function of |X − center|.
    center = d["mu"]
    amp    = np.abs(s - center)
    amp_max = float(amp.max()) if amp.size and amp.max() > 0 else 1.0
    xn     = np.linspace(0.0, amp_max, 300)

    # Empirical APD — direct survival function of the data's own |deviation|
    y_emp = np.array([np.mean(amp > x) for x in xn])
    fig.add_trace(go.Scatter(
        x=xn, y=np.clip(y_emp, 1e-6, None), mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.35)",
        name="Empirical APD", showlegend=True,
        hovertemplate="x: %{x:.3f}<br>P(|X-c|>x): %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # For a symmetric location-scale fit, P(|X-loc|>x) = 2 * SF(loc + x).
    # Each fit is wrapped so a convergence failure on one distribution
    # (rare, but possible on odd data) can't take down the whole panel.
    try:
        ggd_beta, ggd_loc, ggd_scale = sp_stats.gennorm.fit(s)
        y_ggd = 2.0 * sp_stats.gennorm.sf(ggd_loc + xn, ggd_beta, ggd_loc, ggd_scale)
        fig.add_trace(go.Scatter(
            x=xn, y=np.clip(y_ggd, 1e-6, None), mode="lines",
            line=dict(color=AMBER, width=2, dash="dot"),
            name="GGD APD", showlegend=True,
            hovertemplate=f"x: %{{x:.3f}}<br>GGD (β={ggd_beta:.2f}): %{{y:.4f}}<extra></extra>",
        ), row=1, col=1)
    except Exception:
        pass

    try:
        t_df, t_loc, t_scale = sp_stats.t.fit(s)
        y_t = 2.0 * sp_stats.t.sf(t_loc + xn, t_df, t_loc, t_scale)
        fig.add_trace(go.Scatter(
            x=xn, y=np.clip(y_t, 1e-6, None), mode="lines",
            line=dict(color=ACCENT2, width=2, dash="dash"),
            name="Student-t APD", showlegend=True,
            hovertemplate=f"x: %{{x:.3f}}<br>t (ν={t_df:.2f}): %{{y:.4f}}<extra></extra>",
        ), row=1, col=1)
    except Exception:
        pass

    try:
        lap_loc, lap_scale = sp_stats.laplace.fit(s)
        y_lap = 2.0 * sp_stats.laplace.sf(lap_loc + xn, lap_loc, lap_scale)
        fig.add_trace(go.Scatter(
            x=xn, y=np.clip(y_lap, 1e-6, None), mode="lines",
            line=dict(color=GREEN, width=2, dash="longdash"),
            name="Laplace APD", showlegend=True,
            hovertemplate="x: %{x:.3f}<br>Laplace: %{y:.4f}<extra></extra>",
        ), row=1, col=1)
    except Exception:
        pass

    fig.update_yaxes(type="log", title_text="P(|X-c| > x)", row=1, col=1)
    fig.update_xaxes(title_text="x (amplitude level)", row=1, col=1)


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

    fig.update_xaxes(title=dict(text="Theoretical quantiles", standoff=3), row=1, col=2)
    fig.update_yaxes(title=dict(text="Sample quantiles", standoff=3), row=1, col=2)
    fig.update_xaxes(title=dict(text="Lag", standoff=3), row=2, col=1)
    fig.update_yaxes(title=dict(text="p-value", standoff=3), row=2, col=1)
    fig.update_xaxes(title=dict(text="AR order p", standoff=3), row=2, col=2)
    fig.update_yaxes(title=dict(text="HQC", standoff=3), row=2, col=2)

    grp, gc = GROUPS.get(col, ("", MUTED2))
    base_layout = _layout(height=620, margin=dict(t=80, b=32, l=60, r=20))
    base_layout["showlegend"] = True
    fig.update_layout(
        **base_layout,
        title=dict(
            text=f"<span style='color:{gc}'>[{grp}]</span>  <b>{LABELS.get(col, col)}</b>  ·  Diagnostic Panel",
            font=dict(size=14, color=TEXT), x=0.01,
        ),
        legend=dict(
            x=0.265, y=1.0, xanchor="right", yanchor="top",
            bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
            font=dict(size=9, color=MUTED2, family=FONT_MONO),
        ),
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=MUTED2, family=FONT_MONO)
    return fig


def fig_regdiag_overview() -> go.Figure:
    """Heatmap: Breusch-Pagan / Breusch-Godfrey / Durbin-Watson × all variables."""
    tests   = ["Breusch-Pagan (homosk.)", "Breusch-Godfrey (no serial)", "Durbin-Watson (≈2)"]
    ylabels = [LABELS.get(c, c) for c in AVAILABLE]
    z, texts, cdata = [], [], []

    # ── Breusch-Pagan row ──
    rz, rt, rc = [], [], []
    for c in AVAILABLE:
        d  = REGDIAG_RES[c]
        rz.append(1 if d["bp_ok"] else 0)
        rt.append("✔ Homosk." if d["bp_ok"] else "✘ Hetero")
        rc.append(f"Breusch-Pagan<br>LM={sf(d['bp_stat'])}  ·  p={pf(d['bp_p'])}")
    z.append(rz); texts.append(rt); cdata.append(rc)

    # ── Breusch-Godfrey row ──
    rz, rt, rc = [], [], []
    for c in AVAILABLE:
        d  = REGDIAG_RES[c]
        rz.append(1 if d["bg_ok"] else 0)
        rt.append("✔ No serial" if d["bg_ok"] else "✘ Serial")
        rc.append(f"Breusch-Godfrey (p≤{d['bg_lags']})<br>LM={sf(d['bg_stat'])}  ·  p={pf(d['bg_p'])}")
    z.append(rz); texts.append(rt); cdata.append(rc)

    # ── Durbin-Watson row (encode closeness to 2 → 0..1) ──
    rz, rt, rc = [], [], []
    for c in AVAILABLE:
        dw = REGDIAG_RES[c]["dw_stat"]
        rz.append(max(0.0, 1 - min(abs(dw - 2) / 2, 1)))
        rt.append(f"DW={dw:.2f}")
        tag = ("positive autocorr." if dw < 1.5 else
               "negative autocorr." if dw > 2.5 else "no 1st-order autocorr.")
        rc.append(f"Durbin-Watson<br>DW={dw:.3f}  ·  {tag}")
    z.append(rz); texts.append(rt); cdata.append(rc)

    fig = go.Figure(go.Heatmap(
        z=z, x=ylabels, y=tests, customdata=cdata,
        colorscale=[
            [0.0, "rgba(244,115,110,0.55)"],
            [0.5, "rgba(245,166,35,0.45)"],
            [1.0, "rgba(45,212,160,0.55)"],
        ],
        zmin=0, zmax=1,
        text=texts, texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT),
        hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>",
        showscale=False, xgap=3, ygap=3,
    ))
    fig.update_xaxes(tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
                     showgrid=False, linecolor=BORDER, tickangle=35)
    fig.update_yaxes(tickfont=dict(size=11, color=MUTED2, family=FONT_MONO),
                     showgrid=False, linecolor=BORDER)
    fig.update_layout(**_layout(
        height=240,
        margin=dict(t=44, b=120, l=180, r=16),
        title=dict(text="Regression-Residual Overview  ·  green = homoskedastic / no serial correlation / DW≈2",
                   font=dict(size=12, color=TEXT), x=0.01),
    ))
    return fig


def fig_regdiag_variable(col: str) -> go.Figure:
    """Per-variable panel: BP & BG significance bars + a Durbin-Watson gauge."""
    d   = REGDIAG_RES[col]
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.52, 0.48], horizontal_spacing=0.16,
        specs=[[{"type": "xy"}, {"type": "indicator"}]],
        subplot_titles=["Test significance  ·  −log₁₀(p)", "Durbin-Watson statistic"],
    )

    # ── (1,1) BP / BG significance bars ──
    names  = ["Breusch-Pagan<br>(heteroskedasticity)", "Breusch-Godfrey<br>(serial correlation)"]
    ps     = [d["bp_p"], d["bg_p"]]
    oks    = [d["bp_ok"], d["bg_ok"]]
    stats_ = [d["bp_stat"], d["bg_stat"]]
    yvals  = [-math.log10(max(p, 1e-15)) if not math.isnan(p) else 0 for p in ps]
    colors = [GREEN if o else RED for o in oks]
    hover  = [
        f"<b>{nm.replace('<br>', ' ')}</b><br>LM stat: {sf(st)}<br>p-value: {pf(p)}<br>"
        f"{'No violation ✔' if o else 'Violation ✘'}<extra></extra>"
        for nm, st, p, o in zip(names, stats_, ps, oks)
    ]
    fig.add_trace(go.Bar(
        x=names, y=yvals, marker_color=colors, marker_line=dict(color=BORDER, width=1),
        text=[f"p={pf(p)}" for p in ps], textposition="outside",
        textfont=dict(size=11, color=MUTED2, family=FONT_MONO),
        hovertemplate=hover, showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=-math.log10(0.05), line=dict(color=AMBER, width=1.5, dash="dot"),
                  annotation_text="  α = 0.05", annotation_font=dict(color=AMBER, size=10),
                  row=1, col=1)

    # ── (1,2) Durbin-Watson gauge (0–4, ideal ≈ 2) ──
    dw = d["dw_stat"]
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dw,
        number=dict(font=dict(size=26, color=TEXT, family=FONT_MONO), valueformat=".3f"),
        gauge=dict(
            axis=dict(range=[0, 4], tickwidth=1, tickcolor=MUTED, dtick=1,
                      tickfont=dict(size=10, color=MUTED, family=FONT_MONO)),
            bar=dict(color="rgba(0,0,0,0)"),          # hide default bar; use threshold marker
            bgcolor=CARD, borderwidth=1, bordercolor=BORDER,
            steps=[
                dict(range=[0, 1.5], color="rgba(244,115,110,0.45)"),   # positive autocorr
                dict(range=[1.5, 2.5], color="rgba(45,212,160,0.40)"),  # no autocorr
                dict(range=[2.5, 4], color="rgba(245,166,35,0.40)"),    # negative autocorr
            ],
            threshold=dict(line=dict(color=ACCENT2, width=4), thickness=0.85, value=dw),
        ),
    ), row=1, col=2)

    ax_kw = dict(**_AX, tickfont=dict(size=11, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(**ax_kw, row=1, col=1)
    fig.update_yaxes(**ax_kw, row=1, col=1,
                     title=dict(text="−log₁₀(p)  ·  higher = stronger violation", standoff=6))

    grp, gc = GROUPS.get(col, ("", MUTED2))
    fig.update_layout(
        **_layout(height=340, margin=dict(t=74, b=44, l=66, r=24)),
        title=dict(
            text=f"<span style='color:{gc}'>[{grp}]</span>  <b>{LABELS.get(col, col)}</b>"
                 "  ·  Regression-Residual Diagnostics",
            font=dict(size=14, color=TEXT), x=0.01,
        ),
    )
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


VAR_OPTIONS  = [{"label": LABELS.get(c, c), "value": c} for c in AVAILABLE]

PLOTLY_CFG = dict(
    displayModeBar=True, displaylogo=False,
    modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"],
    toImageButtonOptions=dict(format="png", filename="solar_chart", scale=2),
    responsive=True,
)

def section_hdr(text):
    return html.Div(text, className="section-hdr")

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


def callout(kind, title, intro, items, ordered=True):
    """Structured note / recommendation box.

    kind    : 'tip' (green) or 'note' (amber)
    title   : short heading shown next to an icon
    intro   : optional one-line framing sentence (str or list of components)
    items   : list of steps. Each item is either a plain string, or a
              (lead, rest) tuple where `lead` is emphasised and `rest` follows.
    ordered : True → numbered list (workflows), False → bulleted list (notes)
    """
    icon = "📋" if kind == "tip" else "⚠"
    box_cls = "callout callout-tip" if kind == "tip" else "callout callout-note"
    head = html.Div([
        html.Span(icon, className="callout-icon"),
        html.Span(title, className="callout-title"),
    ], className="callout-head")

    body_children = []
    if intro:
        body_children.append(html.P(intro, className="callout-intro"))
    lis = []
    for it in items:
        if isinstance(it, (tuple, list)):
            lead, rest = it
            lis.append(html.Li([html.B(lead, className="callout-lead"), rest],
                               className="callout-li"))
        else:
            lis.append(html.Li(it, className="callout-li"))
    list_tag = html.Ol if ordered else html.Ul
    body_children.append(list_tag(lis, className="callout-list"))
    body = html.Div(body_children, className="callout-body")

    return html.Div([head, body], className=box_cls)


DIAG_THEORY = html.Div([
    html.Div("🔬 Diagnostic Tests: Theory & Interpretation", className="theory-title"),

    # ── Intro paragraph ──
    html.P([
        "These five tests examine two fundamental properties of your time series residuals: ",
        html.B("normality"), " (are the values bell-curve distributed?) and ",
        html.B("autocorrelation"), " (do past values predict future values?). ",
        "They are run on the raw series, not on model residuals. ",
        "A failed normality test does not invalidate a model (many forecasting models are robust to non-normality) ",
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
                "Test statistic:",
                r"$$D_n = \sup_x \left| F_n(x) - F(x) \right|$$"
                "\n\nwhere $F_n(x)$ is the empirical CDF and $F(x)$ is the reference Normal CDF.",
                "The data are drawn from a Normal distribution.",
                "The data are NOT drawn from a Normal distribution.",
                "A large D (and small p-value) means the data deviate significantly from Normal. "
                r"$p \geq 0.05$ means we cannot reject normality, the data are consistent with Normal.",
                "Useful for any sample size. Less powerful than Shapiro-Wilk for small samples ($n < 50$). ",
                caveats="The KS test is general-purpose but less sensitive to deviations in the tails "
                        "compared to Anderson-Darling. For the solar dataset, variables with heavy zeros "
                        "(precipitation, snowfall) will fail by design.",
            ),
            _latex_card(
                "Shapiro-Wilk Test", "SW", GREEN,
                "The Shapiro-Wilk test measures how well the data fit a straight line on a Q-Q plot. "
                "A Q-Q plot plots your data's quantiles against what they would be if the data were perfectly Normal. "
                "If the points lie on a diagonal line, the data are Normal. The W statistic is the squared "
                "correlation between the sorted data and the expected Normal order statistics (W = 1 is perfect normality).",
                "Test statistic:",
                r"$$W = \frac{\left(\displaystyle\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}$$,"
                " where $a_i$ are coefficients derived from the expected Normal order statistics:"
                r"$$(a_1, a_2, \ldots, a_n) = \frac{m^T V^{-1}}{\sqrt{m^T V^{-1} V^{-1} m}}$$,"
                "$m$ is the vector of expected Normal order statistics, $V$ is their covariance matrix."
                "\n\n$x_{(i)}$ are the order statistics (sorted values), and $\\bar{x}$ is the sample mean, ",
                "The data come from a Normal distribution (W close to 1).",
                "The data do NOT come from a Normal distribution.",
                "W ranges from 0 to 1. Values close to 1 indicate normality. "
                "p < 0.05 rejects normality. "
                "For large samples (n > 5000), even tiny deviations become significant "
                "(inspect the Q-Q plot visually alongside the p-value).",
                "Considered the most powerful normality test for sample sizes up to ~5000. "
                "For larger datasets the test is run on the first 5000 observations.",
                caveats="Very large samples (n > 2000) will almost always reject normality even for "
                        "near-Normal data due to the test's sensitivity. Always inspect the Q-Q plot.",
            ),
        ], className="theory-row"),
        # Row 2: JB + LB
        html.Div([
            _latex_card(
                "Jarque-Bera Test","JB",AMBER,
                "The Jarque-Bera test checks normality by measuring two specific features: "
                "skewness (how asymmetric the distribution is, accounting that Normal distributions are symmetric) "
                "and excess kurtosis (how heavy the tails are, accounting that Normal distributions have kurtosis = 3, "
                "so excess kurtosis = kurtosis - 3 = 0). "
                "If both are close to zero, the data look Normal from a moments perspective.",
                "Test statistic:",
                r"$$JB = \frac{n-k}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$"
                "\n\nwhere $n$ is the sample size, $k$ is the number of regressors estimated, $S = \\hat{\\mu}_3/\\hat{\\sigma}^3$ is the sample skewness, "
                "$K = \\hat{\\mu}_4/\\hat{\\sigma}^4$ is the sample kurtosis,"
                " $\\hat{\\mu}_k$ is the k-th sample central moment, and $\\hat{\\sigma}^2$ is the sample variance."
                "Under normality, $JB \\sim \\chi^2(2)$.",
                "Skewness = 0 and excess kurtosis = 0 (data are Normal).",
                "Skewness ≠ 0 or excess kurtosis ≠ 0 (data deviate from Normal).",
                "JB = 0 means perfect normality. Larger JB means stronger evidence against normality. "
                "Rule of thumb: |skew| < 0.5 is approximately symmetric; |excess kurtosis| < 1 is near-Normal tails. "
                "Wind direction variables will show near-zero JB due to their circular but roughly symmetric nature.",
                "Asymptotically chi-squared, unreliable for small samples (n < 30) where Type I error rate is high. "
                "Best used for large samples alongside KS and SW.",
                caveats="The chi-squared approximation overstates significance for small samples. "
                        "For solar data with n >> 1000, this is not a concern."
            ),
            _latex_card(
                "Ljung-Box Test", "LB", "#60a5fa",
                "The Ljung-Box test checks whether the autocorrelations of a series are jointly zero up to lag m. "
                "Namely, after fitting a model to your data, the residuals should be 'white noise' "
                "(meaning no value should be predictable from previous values). "
                "If autocorrelations are non-zero, the model has missed some structure in the data. "
                "The Q statistic aggregates the squared autocorrelations across all lags up to m.",
                "Test statistic:",
                r"$$\mathcal{Q} = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}$$"
                "\n\nwhere $\\hat{\\rho}_k$ is the sample autocorrelation at lag $k$, $n$ is the sample size, "
                "and $m$ is the number of lags tested. "
                "Under the null, $\\mathcal{Q} \\sim \\chi^2(h)$ where $h = m - p - q$ for ARMA$(p,q)$ residuals."
                "For significance level $a$, the critical region for rejection of the hypothesis of randomness is:"
                "$$\\mathcal{Q} > {\\chi}_{1-a}^2(h)$$",
                "No autocorrelation up to lag m (the series behaves like white noise).",
                "Significant autocorrelation exists (the series, or model residuals, has exploitable structure).",
                "Reported here at lags 6, 12, and 24. "
                "Lag 24 is critical for meteorological data: a significant Q at lag 24 confirms strong diurnal periodicity. "
                "p < 0.05 means autocorrelation is present (this is expected for all raw series in this dataset). "
                "Green = white noise (no autocorrelation). Red = autocorrelation detected.",
                "Run on raw series here (not model residuals). "
                "All weather variables are expected to show significant autocorrelation (red) at lags 12 and 24 "
                "due to the solar diurnal cycle.",
                caveats="Degrees of freedom must be adjusted when applied to ARMA residuals: use h = m - p - q. "
                        "The test is applied to the raw series here, so treat red results as descriptive, "
                        "not as model failures.",
            ),
        ], className="theory-row"),
        # Row 3: HQC (single card, full width via solo-card class)
        html.Div([
            _latex_card(
                "Hannan-Quinn Criterion", "HQC", "#a78bfa",
                "The Hannan-Quinn Criterion selects the optimal order (number of lags) for an autoregressive (AR) model. "
                "Think of it as a 'complexity penalty' (adding more lags improves the fit) but HQC penalises "
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
                "An AR order k ≥ 1 minimises HQC (the series has memory).",
                "The optimal lag displayed is the AR order p that minimises HQC across p = 1 to 12. "
                "Solar radiation and temperature typically show HQC-optimal AR(2) or AR(3), "
                "reflecting the strong diurnal carry-over. "
                "Precipitation and snowfall often yield AR(1) due to sparse event structure.",
                "HQC consistently selects a smaller order than AIC but typically larger than BIC. "
                "It is the recommended criterion when n is large and overfitting is a concern.",
                caveats="HQC is computed on the raw series, not first differences. "
                        "For non-stationary series, the selected lag order should be interpreted with caution"
                        " (apply ADF/PP first to confirm I(0) before interpreting the HQC-optimal AR order).",
            ),
            # Empty spacer card to maintain grid symmetry
            html.Div(style={"width": "50%"}),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),

    # ── Regression-residual diagnostics sub-section ──
    html.Div("🧮 Regression-Residual Diagnostics: Heteroskedasticity & Serial Correlation",
             className="theory-title", style={"marginTop": "30px"}),
    html.P([
        "Unlike the five tests above (which run on the raw series), the next three are classical ",
        html.B("regression diagnostics"), dcc.Markdown(r". For each variable we fit a simple model (a constant "
        r"plus a linear time trend, e.g. $y_t = \alpha + \beta t + \varepsilon_t$, and examine its residuals $\varepsilon_t$. ", mathjax=True),
    ], className="section-desc", style={"maxWidth": "100%", "marginBottom": "20px"}),

    html.Div([
        # Row: Breusch-Pagan + Breusch-Godfrey
        html.Div([
            _latex_card(
                "Breusch-Pagan Test", "BP", AMBER,
                dcc.Markdown(
                    r"""The Breusch-Pagan test checks for heteroskedasticity. OLS assumes $\mathrm{Var}(\varepsilon_i)=\sigma^2$.
BP asks whether the variance changes systematically with explanatory variables.
Alternative variance structure: $\mathrm{Var}(\varepsilon_i) = \sigma^2 h(Z_i)$.
The intuition is simple: «Do the residuals become systematically more or less dispersed as $X$ changes?»
The test exploits the fact that heteroskedasticity creates structure in the squared residuals.
Estimate the model and obtain residuals $e_i$. Run the auxiliary regression:

$$e_i^2 = \alpha_0 + \alpha_1 Z_{i1} + \cdots + \alpha_k Z_{ik} + v_i$$""",
                    mathjax=True,
                ),
                "Test statistic:",
                r"""The LM statistic is $LM = nR^2$, where $R^2$ comes from the auxiliary regression. Under $H_0$:

$$LM \sim \chi_k^2$$

Reject for large values of $LM$ or small p-values.""",
                dcc.Markdown(r"$\forall i: \alpha_i = 0$. Namely, no evidence of heteroskedasticity (homoskedastic residuals).", mathjax=True),
                dcc.Markdown(r"$\exists i: \alpha_i \neq 0$. Namely, evidence of heteroskedasticity (non-constant variance).",mathjax=True),
                "p < 0.05 means the residuals have non-constant variance — use HAC standard errors or a GLS/WLS specification.",
                "p < 0.05 signals heteroskedasticity: OLS coefficients remain unbiased but standard errors are wrong, invalidating inference. Use robust (HC) standard errors or re-specify the model.",
                caveats=dcc.Markdown(
                    r"""- Detects only the variance pattern specified in the auxiliary regression.
- Sensitive to non-normality.
- Indicates existence, not severity, of heteroskedasticity.
- OLS coefficients remain unbiased: $E[\hat{\beta}] = \beta$. The main consequence is incorrect standard errors.""",
                    mathjax=True,
                ),
            ),
            _latex_card(
                "Breusch-Godfrey Test", "BG", "#60a5fa",
                dcc.Markdown(
                    r"""The Breusch-Godfrey (LM) test checks for serial correlation in regression residuals
up to a chosen lag $p$. It is more general than Durbin-Watson: it handles higher-order
autocorrelation and stays valid even when the regressors include lagged dependent variables.
It regresses the residuals on the original regressors plus their own $p$ lags.

Consider the linear regression $Y_t = \beta_1 + \beta_2 X_{t,1} + \beta_3 X_{t,2} + u_t$,
where the error term may follow an autoregressive process of order $p$:

$$u_t = \rho_1 u_{t-1} + \rho_2 u_{t-2} + \cdots + \rho_p u_{t-p} + \varepsilon_t$$

The test estimates the original regression using OLS and obtains residuals $\hat{u}_t$.
An auxiliary regression is then run:

$$\hat{u}_t = \alpha_0 + \alpha_1 X_{t,1} + \alpha_2 X_{t,2} + \rho_1 \hat{u}_{t-1} + \cdots + \rho_p \hat{u}_{t-p} + \varepsilon_t$$

The inclusion of the original regressors ensures that any explanatory power attributed to
the lagged residuals reflects serial dependence in the errors rather than variation already
explained by the model.""",
                    mathjax=True,
                ),
                "Test statistic (LM):",
                r"""Let $R^2$ be the coefficient of determination from the auxiliary regression. The LM statistic is:

$$LM = nR^2$$

If the null hypothesis is true, the lagged residuals have no explanatory power and $R^2 \approx 0$.
If serial correlation is present, $R^2$ increases. Breusch and Godfrey showed that under $H_0$:

$$nR^2 \xrightarrow{d} \chi^2_p$$

where $p$ is the number of lagged residuals in the auxiliary regression. Tested here up to $p = 12$.""",
                dcc.Markdown(r"$\rho_1 = \rho_2 = \cdots = \rho_p = 0$: No serial correlation in the residuals up to lag p.",mathjax=True),
                dcc.Markdown(r"$\exists i: \rho_i \neq 0$: The residuals are serially correlated at one or more lags ≤ p.",mathjax=True),
                dcc.Markdown(r"$p < 0.05$ means the residuals are autocorrelated, so the trend-only model has missed "
                        r"dynamic structure. For these meteorological series the diurnal cycle remains in the "
                        r"residuals, so BG is overwhelmingly significant (red) — pointing toward an AR / seasonal "
                        r"specification.", mathjax=True),
                "More powerful and general than Durbin-Watson, but the chosen lag p matters.",
                caveats=dcc.Markdown(r"With strong daily seasonality, a small p can miss the 24-hour cycle — set "
                        r"$p ≥ 24$ to capture it. Asymptotic $\chi^2$ approximation; reliable for large n.", mathjax=True),
            ),
        ], className="theory-row"),
        # Row: Durbin-Watson + spacer
        html.Div([
            _latex_card(
                "Durbin-Watson Test", "DW", GREEN,
                "The Durbin-Watson test is used to detect autocorrelation (specifically first-order serial correlation) "
                "in the residuals of a regression model. If your data is time-series, errors often correlate with their " \
                "own past values, violating the OLS assumption that errors are independent.",
                "Test statistic:",
                "The test statistic is calculated as the sum of squared differences between consecutive residuals " \
                "divided by the total sum of squared residuals: "
                r"$$DW = \frac{\sum_{t=2}^{n}\left(\hat{e}_t - \hat{e}_{t-1}\right)^2}"
                r"{\sum_{t=1}^{n}\hat{e}_t^{\,2}} \;\approx\; 2\left(1 - \hat{\rho}_1\right)$$"
                "\n\nwhere $\\hat{e}_t$ is the residuals at time $t$, and $\\hat{\\rho}_1$ is the lag-1 residual autocorrelation. The approximation "
                "shows the mapping: $\\hat{\\rho}_1 = 0 \\Rightarrow DW \\approx 2$, "
                "$\\hat{\\rho}_1 = 1 \\Rightarrow DW \\approx 0$.",
                dcc.Markdown(r"No first-order serial correlation ($\rho_1 = 0, DW \approx 2$).", mathjax=True),
                dcc.Markdown(r"First-order serial correlation is present ($\rho_1 \neq 0$).", mathjax=True),
                "DW far below 2 (toward 0) signals strong positive autocorrelation; toward 4 signals "
                "negative. For the solar series the trend-regression residuals carry the diurnal cycle, "
                "so DW sits near 0. Exact decision bounds depend on n and k; for large n the "
                "no-correlation band is tight around 2 (we flag 1.5 ≤ DW ≤ 2.5 as clear).",
                "Detects only lag-1 autocorrelation and is invalid when regressors include lagged "
                "dependent variables.",
                caveats=dcc.Markdown(r"It is only valid for first-order autoregressive processes ($AR(1)$), and it is biased if there is a lagged dependent variable (e.g., $y_{t-1}$) as an independent variable.",mathjax=True),
            ),
            # Empty spacer card to maintain grid symmetry
            html.Div(style={"width": "50%"}),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),

    callout(
        "tip",
        "Recommended workflow",
        "A practical order of operations for getting from a raw series to a model you can defend:",
        [
            ("Test normality with all three tests together. ",
             "KS, Shapiro-Wilk and Jarque-Bera each probe a different feature — the whole "
             "distribution, the Q-Q fit, and skew/kurtosis respectively. When they agree, the "
             "verdict is robust; when they disagree, the disagreement itself is diagnostic."),
            ("Always read the Q-Q plot, not just the p-value. ",
             "At several thousand observations even trivial departures from normality become "
             "'significant'. The plot tells you whether that matters in practice — a few outliers "
             "in the tails versus a genuinely wrong shape."),
            ("Use Ljung-Box at lag 24 as your diurnal check. ",
             "A significant statistic there is the signature of an uncaptured daily cycle — exactly "
             "what you expect before modelling, and what should disappear afterwards."),
            ("Let the HQC-optimal lag seed your AR/ARIMA order. ",
             "Treat it as a principled starting point, then refine with AIC/BIC on the differenced "
             "(stationary) series."),
            ("Re-run the residual diagnostics on the fitted model. ",
             "Breusch-Godfrey and Durbin-Watson should now show no serial correlation, and "
             "Breusch-Pagan a stable variance. That confirmation — not the raw-series result — is "
             "what actually validates the model."),
        ],
    ),

], className="theory-section")


STATION_THEORY = html.Div([
    html.Div("📖 Unit Root & Stationarity — Theory", className="theory-title"),
    html.Div([
        html.Div([
            theory_card("Augmented Dickey-Fuller", "ADF", ACCENT2,
                ["The ADF test checks whether a time series has a 'unit root' (the characteristic equation has a unit root,"
                 " which can be translated into the behavior of the series wandering randomly with no tendency to return to a mean). "
                 "A unit root implies the series is non-stationary: its statistical properties (mean, variance) change over time, "
                 "which invalidates most classical forecasting models.",
                 "The test augments the basic Dickey-Fuller regression with lagged differences of the series to absorb any "
                 "serial correlation in the error term. The lag count is chosen automatically by AIC.",
                 dcc.Markdown(r"Model: $\Delta Y_t = \mu  + \delta y_{t-1} + \sum_{i=1}^{p}\beta_i \Delta Y_{t-1} + \varepsilon_t$", mathjax=True),
                 html.Div([
                     dcc.Markdown(r"""The intuition behind the test is that if the series is characterised by a unit root process, 
                             then the lagged level of the series ($y_{t-1}$) will provide no relevant information in predicting the change in $y_t$ 
                             besides the one obtained in the lagged changes ($\Delta y_{t-k}$). In this case, the $\delta=0$ and null hypothesis is not rejected.
                             In contrast, when the process has no unit root, it is stationary and hence exhibits reversion to the mean - so the lagged level will provide
                             relevant information in predicting the change of the series and the null hypothesis of a unit root will be rejected.""", mathjax=True),
                 ], className="warn-box", style={"marginTop": "10px"}),
                ],None,
                dcc.Markdown(r"$\delta = 0$ → unit root present (non-stationary)", mathjax=True),
                dcc.Markdown(r"$\delta \neq 0$ → series is stationary", mathjax=True)),
            theory_card("KPSS Test", "KPSS", GREEN,
                ["The KPSS test takes the opposite stance from ADF: its null hypothesis is that the series IS stationary. "
                 "This makes it a powerful complement (if ADF rejects its null AND KPSS fails to "
                 "reject its null, you have strong convergent evidence.)",
                 "The test decomposes the series into a deterministic trend, a random walk, and a stationary error. "
                 "It then tests whether the variance of the random walk component equals zero. The KPSS test, the absence of "
                 "a unit root is not a proof of stationarity but, by design, of trend-stationarity."
                 " This is an important distinction since it is possible for a time series to be non-stationary,"
                 " have no unit root yet be trend-stationary. In both unit root and trend-stationary processes, "
                 "the mean can be growing or decreasing over time; however, in the presence of a shock,"
                 " trend-stationary processes are mean-reverting (i.e. transitory, the time series will converge again"
                 " towards the growing mean, which was not affected by the shock) while unit-root processes have"
                 " a permanent impact on the mean (i.e. no convergence over time).",
                 dcc.Markdown(r"Model: $Y_t = \xi_t + r_t + \varepsilon_t$, where $r_t = r_{t-1} + u_t$", mathjax=True),
                 html.Div([
                     dcc.Markdown(r"KPSS has lower power as a unit-root test than ADF or PP. It was designed to test stationarity, "
                 "not the presence of a unit root. Schmidt & Shin (1992) caution against using it as a primary unit-root test — "
                 "(use it alongside ADF and PP for a consistent picture). Keep in mind that KPSS is a test of trend-stationarity, not just stationarity." 
                 "and the test statistic is a Lagrange multiplier based on the partial sum of residuals from the regression of $Y_t$ on a deterministic trend. ", mathjax=True),
                 ], className="warn-box", style={"marginTop": "10px"}),
                ],
                None,
                dcc.Markdown(r"$\sigma^2_u = 0$ → series is trend-stationary", mathjax=True),
                dcc.Markdown(r"$\sigma^2_u \ge 0$ → unit root / non-stationary", mathjax=True)),
        ], className="theory-row"),
        html.Div([
            theory_card("Phillips-Perron", "PP", AMBER,
                ["The Phillips-Perron (PP) test is a non-parametric statistical diagnostic used to detect a unit root, "
                "namely the stationarity behavior. While the classic Augmented Dickey-Fuller (ADF) test handles underlying temporal"
                " dependencies and heteroscedasticity parametrically by adding explicit lagged difference terms to the regression "
                "equation, the PP test leaves the baseline regression simple and instead applies a Newey-West robust covariance correction"
                " directly to the test statistics. ",
                 "PP is generally preferred over ADF when the error structure is unknown, but Davidson & MacKinnon (2004) "
                 "found it can underperform ADF in finite samples. Best used together with ADF.",
                 dcc.Markdown(r"Model: $\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \varepsilon_t$ (non-parametric correction applied to statistic)", mathjax=True),
                  html.Div([
                     html.Span("⚠ Caveats: ", style={"fontWeight": "700", "color": AMBER}),
                     "Although the PP test is robust to certain forms of autocorrelation and heteroskedasticity, it" \
                     " can still suffer from low power in finite samples, particularly when the series is highly persistent."
                     " The test is also sensitive to structural breaks, where a level or trend shift may cause a stationary"
                     " process to appear non-stationary. Furthermore, the choice of bandwidth used in estimating the long-run"
                     " variance can affect the results. Like all unit-root tests, the PP test may struggle to distinguish between"
                     " a highly persistent stationary process and a true unit-root process, especially when sample sizes are small.",
                 ], className="warn-box", style={"marginTop": "10px"}),
                ],
                dcc.Markdown(
                    r"Young practitioners should view the Phillips-Perron test as a robust alternative " \
                    r"to the ADF test when residual autocorrelation or heteroskedasticity is suspected." \
                    r" The interpretation is straightforward: if the p-value is below the chosen significance" \
                    r" level, reject $ H_0:\gamma = 0$ and conclude that the series is stationary. Because no " \
                    r"single unit-root test is definitive, it is common practice to use the PP test alongside " \
                    r"the ADF and KPSS tests. When all three tests point to the same conclusion, confidence in " \
                    r"the stationarity assessment is substantially increased. In applied forecasting and " \
                    r"econometrics, stationarity diagnostics should always precede VAR, VECM, ARIMA, and " \
                    r"cointegration analyses.", mathjax=True),
                dcc.Markdown(r"$\gamma = 0$ → unit root present", mathjax=True),
                dcc.Markdown(r"$\gamma < 0$ → series is stationary", mathjax=True)),
            theory_card("Zivot-Andrews", "ZA", "#a78bfa",
                ["The Zivot-Andrews (ZA) test is an advanced unit root test designed to solve"
                " a major flaw in traditional tests like the Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) tests:"
                " structural breaks. Standard unit root tests have notoriously low statistical power when applied to data"
                " that contains a structural break (a sudden, permanent shift in the mean or trend of the data)."
                " If a series is perfectly stationary but contains a sharp structural break, tests like the PP"
                " or ADF will get confused, mistake the break for a unit root, and incorrectly conclude that"
                " the series is non-stationary. The ZA test fixes this by endogenously searching for, identifying, and accounting for"
                " the single most significant structural break in the data while testing for stationarity.",
                 "The break point is chosen by minimising the ADF statistic across all possible break dates, making the "
                 "test conservative. If ZA rejects but ADF does not, a structural break is likely masking stationarity.",
                 html.Div(["We assume the following models incorporating a potential break at time $TB$:",
                            dcc.Markdown(r"Model A (Change in Intercept): $Y_t = \mu + \theta DU_t(\lambda) + \beta t + \alpha Y_{t-1}  + \sum c_i \Delta Y_{t-i} + \varepsilon_t$", mathjax=True),
                            dcc.Markdown(r"Model B (Change in Trend): $Y_t = \mu + \gamma DT_t(\lambda) + \beta t + \alpha Y_{t-1}  + \sum c_i \Delta Y_{t-i} + \varepsilon_t$ ", mathjax=True),
                            dcc.Markdown(r"Model C (Change in Both): $Y_t = \mu + \theta DU_t(\lambda) + \gamma DT_t(\lambda) + \beta t + \alpha Y_{t-1}  + \sum c_i \Delta Y_{t-i} + \varepsilon_t$", mathjax=True),
                            dcc.Markdown(r"""where $\Delta y_{t-1}$ terms control for serial correlation in the error term, identical to an ADF test. The optimal lag length \(k\) is usually chosen using information criteria like AIC or BIC.
                              $\lambda=\frac{TB}{T}$ This represents the ratio of the break date ($TB$) to the total sample size ($T$).
                              $DU_t(\lambda) = 1$ if $t > TB$ else 0 (captures a shift in the mean level after the break),
                              $DT_t(\lambda) = t - TB$ if $t > TB$ else 0 (captures a change in the slope of the trend after the break).
                              $t_\alpha(\lambda) = \frac{\hat{\alpha}-1}{\sigma_{\hat{\alpha}}}$ (t-statistic for the unit root hypothesis).
                              The final Zivot-Andrews test statistic ($t_{ZA} = inf_{\lambda \in \Lambda} t_\alpha(\lambda)$) is defined mathematically as the minimum (most negative) $t$-statistic found across the entire grid search.
                            """, mathjax=True) ], className="warn-box", style={"marginTop": "10px"})
                 ],
                dcc.Markdown(r"""The ZA test evaluates the null hypothesis of a unit root against the alternative of stationarity with a structural break.
                              The test statistic is the minimum t-statistic for the unit root coefficient across all possible break points, and the critical values are derived from simulations that account for the presence of a break.
                                If the ZA test rejects the null while ADF does not, it suggests that the series is actually stationary but contains a structural break that confounds traditional unit root tests.
                             """, mathjax=True),
                dcc.Markdown(r"The series contains a unit root (is non-stationary) and has no structural break ($\alpha=1$).", mathjax=True),
                "The series is trend-stationary with a single structural break occurring at an unknown point in time."),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),
    callout(
        "note",
        "Interpreting the stationarity suite",
        "No single unit-root test is decisive — each trades statistical power against robustness, "
        "so read the four as a panel rather than in isolation:",
        [
            ("Pair the opposite nulls. ",
             "ADF and PP take H₀ = 'unit root'; KPSS takes H₀ = 'stationary'. Because their "
             "hypotheses are reversed, their agreement is genuine corroboration, not coincidence."),
            ("The decisive pattern. ",
             "ADF/PP reject and KPSS fails to reject ⟹ confident stationarity; the mirror image ⟹ "
             "confident non-stationarity. Mixed signals usually flag a near-unit-root or a "
             "structural break."),
            ("Suspect a break? Trust Zivot-Andrews. ",
             "A single shift in level or trend — as in zenith or shortwave radiation — can fool "
             "ADF/PP into reporting non-stationarity. Zivot-Andrews estimates the break date "
             "endogenously and tests around it."),
        ],
        ordered=False,
    ),
], className="theory-section")


COINT_THEORY = html.Div([
    html.Div("🔗 Cointegration — Theory", className="theory-title"),
    html.Div([
        html.Div([
            theory_card("Engle-Granger Two-Step", "EG", ACCENT2,
                ["Cointegration tests ask whether two non-stationary (I(1)) series share a long-run equilibrium. "
                 "Even though each series wanders individually, a linear combination of them may be stationary "
                 "(meaning deviations from their shared trend are temporary and mean-reverting).",
                 "The Engle-Granger procedure runs in two steps: first, regress Y on X by OLS to estimate the "
                 "long-run relationship. Second, apply an ADF test to the residuals. If the residuals are stationary, "
                 "the series are cointegrated.",
                 dcc.Markdown(r"Step 1: $Y_t = \alpha + \beta X_t + \varepsilon_t$ → Step 2: ADF test on $\hat{\varepsilon}_t$", mathjax=True)],
                None,
                "Residuals have a unit root → no cointegration",
                "Residuals are stationary → series are cointegrated"),
            theory_card("What Cointegration Means", "CI(1,1)", GREEN,
                ["Think of two cointegrated series as elastic bands stretched between two posts (they can diverge "
                 "in the short run but are always pulled back toward each other). This long-run equilibrium is the "
                 "cointegrating relationship.",
                 "For the solar dataset: shortwave radiation and temperature are both non-stationary over a year, "
                 "but they track each other closely (when one rises so does the other). "
                 "A cointegrating model captures this shared seasonal structure without differencing away the trend.",
                 dcc.Markdown(r"$X_t, Y_t$ are CI(1,1) if $\exists\,\beta$ such that $Y_t - \beta X_t \sim I(0)$", mathjax=True)],
                None,
                "No linear combination is stationary",
                 dcc.Markdown(r"$\exists\,\beta$ such that $Y_t - \beta X_t \sim I(0)$", mathjax=True)),
        ], className="theory-row"),
        html.Div([
           theory_card("Johansen Test (note)", "JOH", "#a78bfa",
                [dcc.Markdown(r"The Johansen Test is a multivariate cointegration procedure used to determine " \
                r"whether a set of non-stationary $I(1)$ time series share one or more long-run " \
                "equilibrium relationships. It is based on a Vector Error Correction Model (VECM)," \
                r" where the key matrix Π captures long-run dynamics. The rank of $\Pi(r)$ determines " \
                r"the number of cointegrating vectors: $r=0$ implies no cointegration, " \
                "while 0<r<k indicates one or more stationary linear combinations of the variables." \
                " The test uses the Trace and Maximum Eigenvalue statistics to estimate the" \
                r" cointegration rank from the eigenvalues of $\Pi$.", mathjax=True),
                 dcc.Markdown(r"VAR: $\Delta X_t = \mu + \Pi X_{t-1} + \sum_i \Gamma_i \Delta X_{t-i} + \varepsilon_t$, $\text{rank}(\Pi) = r$", mathjax=True),
                 html.Div([
                     html.Span("⚠ Caveats: ", style={"fontWeight": "700", "color": AMBER}),
                     dcc.Markdown("The Johansen Test assumes that the variables are generally integrated of the same order,"
                     " typically $I(1)$, and results can be highly sensitive to lag selection,"
                     " deterministic terms (intercepts and trends), and structural breaks in the data."
                     " Small samples may produce unreliable inference because the test relies on asymptotic"
                     " theory. Furthermore, finding cointegration indicates a long-run equilibrium relationship"
                     " but does not imply causality or economic significance by itself.", mathjax=True) ,
                 ], className="warn-box", style={"marginTop": "10px"})
                 ],
                "A useful intuition is that cointegrated variables may drift individually over time, " \
                "but they remain linked by a stable long-run relationship. The Johansen framework is often" \
                " preferred over the Engle-Granger method because it can identify multiple cointegrating " \
                "relationships simultaneously and treats all variables as potentially endogenous." \
                " In practice, researchers should first verify stationarity properties, carefully select the VAR lag structure, "
                "and interpret the resulting cointegrating vectors within the context of economic or" \
                " domain-specific theory.",
                dcc.Markdown(r"$\text{rank}(\Pi) = 0 \rightarrow$ no cointegration", mathjax=True),
                dcc.Markdown(r"$\text{rank}(\Pi) = r \geq 1 \rightarrow$ r cointegrating vectors", mathjax=True)),
            theory_card("Phillips-Ouliaris Test", "PO", RED,
                [dcc.Markdown("The Phillips-Ouliaris (PO) test is a residual-based cointegration test, like Engle-Granger, "
                 "but replaces the parametric ADF step with a non-parametric Phillips-Perron correction on the OLS residuals. "
                 "This makes PO robust to serial correlation and conditional heteroscedasticity in the error term "
                 "without requiring the analyst to specify a lag length.",
                 "The procedure is: (1) Regress Y on X by OLS to obtain the cointegrating residuals. "
                 r"(2) Apply the PP $Z(t)$ or $Z(\alpha)$ statistic to those residuals. "
                 "A significantly negative statistic rejects the null of no cointegration.",mathjax=True),
                 dcc.Markdown(
                    r"Step 1: $Y_t = \alpha + \beta X_t + \hat{\varepsilon}_t$   "
                    r"Step 2: PP$(\hat{\varepsilon}_t)$: $Z(\tau) = \frac{\hat{\sigma}^2 - \hat{\lambda}^2}{2\hat{\lambda}^2}\cdot\frac{\hat{\sigma}_\tau^2}{\hat{s}_\ell^2} \cdot \tau_\rho - \frac{T(\hat{\lambda}^2 - \hat{\sigma}^2)}{2\hat{\lambda}\hat{s}_\ell}$",
                    mathjax=True),
                 dcc.Markdown(
                    r"where $\hat{\sigma}^2 = T^{-1}\sum_t \hat{\varepsilon}_t^2$, "
                    r"$\hat{s}_\ell^2$ is the Newey-West long-run variance estimator with bandwidth $\ell$, "
                    r"and $\hat{\lambda}^2 = \hat{s}_\ell^2$ under $H_0$.",
                    mathjax=True),
                 html.Div([
                     html.Span("⚠ Caveats: ", style={"fontWeight": "700", "color": AMBER}),
                     "PO p-values here are approximated via the MacKinnon (2010) response surface "
                     "(same as EG), so they share the same asymptotic framework. "
                     "For very short series (n < 100), use bootstrap critical values instead.",
                 ], className="warn-box", style={"marginTop": "10px"}),
                ],
                "PO is preferred over EG when the error term is heteroscedastic or has unknown autocorrelation structure. "
                "Phillips & Ouliaris (1990) showed that, unlike ADF-based tests applied to residuals, their Z(t) statistic "
                "has a distribution that depends only on the number of regressors (not the order of integration), "
                "making critical values tabulation straightforward. "
                "In practice, EG and PO give similar conclusions for large samples; PO is more reliable for shorter series.",
                dcc.Markdown(r"$H_0$: Residuals have a unit root → series are NOT cointegrated", mathjax=True),
                dcc.Markdown(r"$H_1$: Residuals are stationary → series ARE cointegrated", mathjax=True)),
        ], className="theory-row"),            
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),
    callout(
        "note",
        "Choosing among the cointegration tests",
        "All three answer the same question — do these series share a long-run equilibrium? — but "
        "with different strengths. Run them together and weight the agreement:",
        [
            ("Engle-Granger (EG). ",
             "Transparent and easy to interpret, but asymmetric — regressing Y on X can disagree "
             "with X on Y — and it assumes well-behaved residuals."),
            ("Phillips-Ouliaris (PO). ",
             "A residual-based test like EG, but robust to serially correlated and heteroskedastic "
             "errors, with no lag length to choose."),
            ("Johansen. ",
             "The most general: it works inside a VAR, can detect more than one cointegrating "
             "relationship, and is invariant to which variable is ordered first."),
            ("Bottom line. ",
             "When all three concur you have high-confidence evidence of a genuine equilibrium; "
             "when they split, prefer Johansen and inspect the cointegrating residuals directly."),
        ],
        ordered=False,
    ),
], className="theory-section")


GRANGER_THEORY = html.Div([
    html.Div("⚡ Granger Causality: Theory", className="theory-title"),
    html.Div([
        html.Div([
            theory_card("Granger Causality", "GC", ACCENT2,
                [dcc.Markdown(r"The Granger Causality Test examines whether past values of one time series help predict another"
                               r" time series beyond the information already contained in its own past. For two variables $X_t$ "
                               r"and $Y_t$, the test compares a restricted model containing only lagged values of $Y_t$ with"
                               r" an unrestricted model that also includes lagged values of $X_t$. The null hypothesis is that"
                               r" the coefficients on the lagged $X_t$ terms are jointly zero."
                               r"An F-test (or Wald test) is then used to determine ",
                               r"whether including lagged $X_t$ significantly improves the prediction of $Y_t$. If the null is"
                               r" rejected, $X$ is said to Granger-cause $Y$. In cointegrated systems, Granger causality is typically "
                               r"examined within a VECM framework, where both short-run dynamics and long-run equilibrium "
                               r"adjustments are considered.", mathjax=True),
                 dcc.Markdown(r"Unrestricted: $Y_t = \sum_i a_i Y_{t-i} + \sum_i b_i X_{t-i} + \varepsilon_t$", mathjax=True),
                 html.Div([
                     html.Span("⚠ Caveats: ", style={"fontWeight": "700", "color": AMBER}),
                     "Granger causality does not imply true causation; it only indicates predictive content. "
                     "Results can be highly sensitive to lag selection, omitted variables, structural breaks, "
                     "and model misspecification. The test also assumes stationarity (or a properly specified VECM "
                     "if variables are cointegrated). Two variables may exhibit Granger causality simply because both"
                     " are driven by an unobserved third factor. Therefore, statistical significance should always "
                     "be interpreted alongside economic or domain-specific reasoning.",
                 ], className="warn-box", style={"marginTop": "10px"})
                 ],
                dcc.Markdown(r" A useful interpretation is: if knowing yesterday's values of X improves forecasts of Y, then X" \
                r" Granger-causes Y. Possible outcomes include unidirectional causality ($X \rightarrow Y$), " \
                r"reverse causality ($Y \rightarrow X$), bidirectional causality (feedback), or no causality. " \
                r"In cointegration analysis, finding cointegration often implies that at least one direction of Granger" \
                r" causality must exist, making the Granger test a natural follow-up after the Johansen test." \
                r" Always interpret the results as evidence of predictive relationships, not proof of cause-and-effect.", mathjax=True),
                dcc.Markdown(r" $b_1 = b_2 = \dots = b_k = 0$ (X adds no predictive power for Y)", mathjax=True),
                dcc.Markdown(r" $\exists i \in \mathbb{N}_{\leq k}: b_i \neq 0$ (X Granger-causes Y at some lag)", mathjax=True)),
            theory_card("Key Caveats", "!", RED,
                [dcc.Markdown(r"The Granger Causality Test evaluates whether past values of a variable $X_t$ improve the"
                    r" prediction of another variable $Y_t$ beyond the information contained in $Y_t$'s own"
                    r" history. This is achieved by comparing a restricted autoregressive model with an unrestricted"
                    r" model that includes lagged values of $X_t$. The null hypothesis is that all coefficients"
                    r" associated with the lagged $X_t$ terms are jointly zero. If the null is rejected using an "
                    r"F-test or Wald test, then $X_t$ is said to Granger-cause $Y_t$. In practice, the workflow "
                    r"consists of testing for stationarity (or cointegration), selecting an appropriate lag order,"
                    r" estimating a VAR or VECM, and testing the significance of lagged predictors.", mathjax=True),
                    html.Div([
                     html.Span("⚠ Caveats: ", style={"fontWeight": "700", "color": AMBER}),
                     dcc.Markdown(r"Granger causality has three important limitations. " \
                                  r"First, confounding occurs when an unobserved variable $Z_t$ influences both" \
                                  r" $X_t$ and $Y_t$, creating a spurious causal relationship even though no " \
                                  r"direct mechanism exists between them." \
                                  r" Second, instantaneous effects are not captured because the method only evaluates " \
                                  r"lagged dependencies; if $Y_t = \alpha X_t + \varepsilon_t$ within the same time period, " \
                                  r"the test may fail to detect the relationship entirely. " \
                                  r"Third, nonlinearity can invalidate conclusions because the standard VAR framework assumes " \
                                  r"linear dynamics. Systems with thresholds, regime changes, or chaotic behavior may require " \
                                  r"nonlinear alternatives such as Convergent Cross Mapping (CCM), Transfer Entropy, or " \
                                  r"kernel-based Granger causality methods.", mathjax=True)
                 ], className="warn-box", style={"marginTop": "10px"}),
                 html.Code(r"Young practitioners should remember that Granger causality measures predictive causality,"
                 " not true cause-and-effect. A significant result only indicates that past observations of one"
                 " variable contain useful forecasting information about another variable. Therefore, every finding should"
                 " be evaluated against domain knowledge, potential confounders, data frequency, and model assumptions."
                 " As a rule of thumb, Granger-causality results are most credible when they are robust across lag selections,"
                 " supported by theory, and consistent with the known dynamics of the underlying system.", className="math")
                 ]),
        ], className="theory-row"),
        html.Div([
            theory_card("VAR Block Exogeneity", "VAR-GC", AMBER,
                ["The **VAR Block Exogeneity Test examines whether a subset of variables in a Vector Autoregression (VAR)"
                " contributes to the prediction of another variable or group of variables. Consider a VAR(p) model:",
                 dcc.Markdown(
                     r"$X_t = \sum_k A^{(k)} X_{t-k} + \varepsilon_t$", mathjax=True),
                dcc.Markdown(r"where $X_t$ contains multiple endogenous variables. To test whether a block of variables $\tilde{X}_t$"
                " influences another variable $Y_t$, the null hypothesis is that all lagged coefficients associated"
                " with $\tilde{X}_t$ are jointly zero (Wald test on block $A^{(k)}_{ij}$)", mathjax=True),
                dcc.Markdown(r"A Wald or (\chi^2) test is then used to determine whether " \
                             r"excluding the entire block significantly reduces the model's explanatory power." \
                             r" If the null is rejected, the block of variables is said to be not exogenous"
                             r" and contains useful predictive information for the target variable(s).", mathjax=True),
                html.Div([
                    html.Span("⚠ Caveats: ", style={"fontWeight": "700", "color": AMBER}),
                     "The VAR Block Exogeneity Test is fundamentally a test of "
                    "predictive relevance, not structural causation. Results can be highly sensitive" \
                    " to lag selection, omitted variables, model specification, and sample size." \
                    " If important variables are excluded from the VAR, a block may appear exogenous"
                    " or non-exogenous simply because the true driver is missing. In addition, strong" \
                    " contemporaneous relationships are not directly captured because the test focuses on " \
                    "lagged effects. For cointegrated variables, a VECM-based block exogeneity test is " \
                    "generally preferred, as ignoring long-run equilibrium relationships may lead to" \
                    " misleading conclusions."
                 ], className="warn-box", style={"marginTop": "10px"}),
                ],
                "Young practitioners can think of the VAR Block Exogeneity Test as a multivariate extension" \
                " of Granger causality. While a standard Granger test asks whether one variable helps" \
                " predict another, block exogeneity asks whether an entire group of variables jointly" \
                " improves forecasting performance. A failure to reject the" \
                " null suggests that the block adds little predictive information, whereas rejection indicates" \
                " that the block should remain in the system. The test is particularly useful for model" \
                " simplification, variable selection, and understanding information flow within large VAR frameworks." \
                "This is more powerful than pairwise GC because the VAR accounts for the joint dynamics of all "
                "included variables simultaneously, reducing the risk of spurious causality from omitted variables.",
                dcc.Markdown(r"All elements of $A_k[i,j] = 0$ across all lags (X does not Granger-cause Y in the system)", mathjax=True),
                dcc.Markdown(r"At least one $A_k[i,j] \neq 0$ (X Granger-causes Y within the VAR)", mathjax=True)
            ),
            theory_card("Convergent Cross Mapping", "CCM", GREEN,
                [dcc.Markdown(
                    r"Convergent Cross Mapping (CCM) is a nonlinear causality method designed " \
                    r"to detect causal relationships in complex dynamical systems. Unlike Granger causality," \
                    r" which relies on predictability through lagged linear relationships, CCM is based on" \
                    r" state-space reconstruction from dynamical systems theory. Given two variables $X_t$ and $Y_t$," \
                    r" Takens' theorem states that the hidden system dynamics can be reconstructed using time-delayed" \
                    r" embeddings, such as ",mathjax=True),
                 dcc.Markdown(r"$M_X(t)=\left[X_t, X_{t-\tau}, X_{t-2\tau}, \ldots, X_{t-(E-1)\tau}\right]$,", mathjax=True),
                 dcc.Markdown(
                            r"where $E$ is the embedding dimension and $\tau$ is the delay. If $X$ causally influences $Y$," \
                            r" information about $X$ should be encoded in the reconstructed manifold of $Y$. CCM tests this by using" \
                            r" nearest neighbors on $M_Y$ to estimate values of $X_t$. The cross-map skill is typically measured by" \
                            r" the correlation ", mathjax=True),
                dcc.Markdown(r"$\rho = \text{corr}\left(X_t,\hat{X}_t\right)$. ", mathjax=True),
                dcc.Markdown(
                            r"Evidence of causality is obtained when $\rho$ increases and converges as the " \
                            r"library size grows, indicating that the information about $X$ is recoverable " \
                            r"from the dynamics of $Y$. ", mathjax=True),
                html.Div([
                    html.Span("⚠ Caveats: ", style={"fontWeight": "700", "color": AMBER}),
                    dcc.Markdown(r"" \
                        r"CCM assumes that the variables belong to the same underlying dynamical system and that" \
                        r" sufficient observations are available to reconstruct the system's attractor. The method" \
                        r" can perform poorly when the sample size is small, noise levels are high, or the system"
                        r" is weakly coupled. Strong synchronization between variables may also obscure causal" \
                        r" direction because each variable contains nearly identical information. Furthermore," \
                        r" selecting inappropriate embedding parameters $(E,\tau)$ can lead to incorrect conclusions." \
                        r" Although CCM is designed for nonlinear systems, it does not automatically distinguish " \
                        r"direct causation from indirect pathways involving intermediate variables. ", mathjax=True),
                 ], className="warn-box", style={"marginTop": "10px"}), 
                ],
                dcc.Markdown(
                    r"Young practitioners should think of CCM as a method that asks: " \
                    "\"Can I reconstruct the history of $X$ using the geometric structure of $Y$?\"" \
                    r" If the answer improves as more data are used, then $X$ likely leaves a causal footprint "
                    r"in $Y$. CCM is particularly valuable in complex systems where relationships are nonlinear and feedback-driven." \
                    r" Unlike Granger causality, CCM can detect causality even when there is no strong lagged" \
                    r" predictive signal. A useful rule of thumb is that Granger causality asks whether $X$ helps forecast $Y$," \
                    r" whereas CCM asks whether the dynamics of $Y$ contain information about $X$ because both evolve" \
                    r" on the same underlying system attractor.", mathjax=True)
                ),
                ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px", "overflow": "hidden", "background": CARD2}),
    callout(
        "tip",
        "From Granger pairs to a validated simulator",
        "Granger results are a starting hypothesis about predictive structure, not a causal "
        "verdict. A defensible pipeline turns them into a trustworthy model:",
        [
            ("Screen out confounding with CCM. ",
             "Convergent Cross Mapping checks whether an apparent Granger link survives in the "
             "system's reconstructed state space, filtering out spurious edges created by a shared "
             "driver (here, the solar cycle)."),
            ("Fit a VAR on the surviving causal graph. ",
             "Keep only the edges that pass, so the dynamics are estimated jointly rather than one "
             "pair at a time."),
            ("Block-bootstrap the VAR residuals in 24-hour blocks. ",
             "Resampling in daily blocks preserves the diurnal autocorrelation that ordinary "
             "i.i.d. resampling would erase."),
            ("Validate the synthetic series against the original. ",
             "Match both the statistics (a KS test on the distributions) and the dynamical "
             "fingerprints — a Lyapunov exponent near λ ≈ 0.072 and a Hurst exponent near "
             "H ≈ 0.19 — to confirm the simulation reproduces the real chaos, not just the mean."),
        ],
    ),
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

/* dcc.Markdown(mathjax=True) wrappers (rqa_tab._md): block keeps paragraph
   spacing tight, inline flows inside list items / labels / table cells */
.md-block > p:last-child {{ margin-bottom: 0; }}
.md-inline, .md-inline > p {{ display: inline; margin: 0; }}
.theory-intro {{ margin-bottom: 18px; }}
.math.md-block {{ overflow-x: auto; overflow-y: hidden; font-size: 12px; }}
.math.md-block > p {{ margin: 0; }}

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

/* Structured callouts (workflows / notes) — elegant section cards */
.callout {{
  margin-top: 22px;
  border: 1px solid {BORDER};
  border-top: 2px solid var(--accent);
  border-radius: 12px;
  background: linear-gradient(180deg, {CARD2} 0%, {CARD} 100%);
  box-shadow: 0 8px 30px rgba(0,0,0,0.32);
  overflow: hidden;
}}
.callout-tip  {{ --accent: {GREEN}; --accent-soft: rgba(45,212,160,0.14); --accent-bg: rgba(45,212,160,0.07); }}
.callout-note {{ --accent: {AMBER}; --accent-soft: rgba(245,166,35,0.16); --accent-bg: rgba(245,166,35,0.07); }}
.callout-head {{
  display: flex; align-items: center; gap: 11px;
  padding: 13px 22px;
  background: var(--accent-bg);
  border-bottom: 1px solid {BORDER};
}}
.callout-icon {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 27px; height: 27px; border-radius: 8px;
  background: var(--accent-soft); font-size: 14px; line-height: 1; flex-shrink: 0;
}}
.callout-title {{
  font-family: {FONT_MONO}; font-weight: 700; font-size: 13px;
  letter-spacing: .015em; color: var(--accent);
}}
.callout-body {{ padding: 17px 24px 19px; }}
.callout-intro {{
  font-size: 12.5px; color: {MUTED2}; line-height: 1.72; margin: 0 0 15px; max-width: 1000px;
}}
.callout-list {{ margin: 0; display: flex; flex-direction: column; gap: 12px; }}
.callout-list li {{ font-size: 12.5px; color: {MUTED2}; line-height: 1.72; max-width: 1000px; }}
.callout-lead {{ color: {TEXT}; font-weight: 700; }}

/* Numbered badges for ordered workflows */
ol.callout-list {{ counter-reset: step; list-style: none; padding-left: 0; }}
ol.callout-list > li {{ position: relative; padding-left: 42px; min-height: 25px; }}
ol.callout-list > li::before {{
  counter-increment: step; content: counter(step);
  position: absolute; left: 0; top: -1px;
  width: 25px; height: 25px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-family: {FONT_MONO}; font-size: 11px; font-weight: 700;
  background: var(--accent-soft); color: var(--accent);
  border: 1px solid var(--accent);
}}
/* Accent dots for unordered notes */
ul.callout-list {{ list-style: none; padding-left: 0; }}
ul.callout-list > li {{ position: relative; padding-left: 24px; }}
ul.callout-list > li::before {{
  content: ''; position: absolute; left: 4px; top: 7px;
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);
}}

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
/* Labels, plot labels and banner captions are uppercased by CSS; never let
   that reach a formula (it would turn ε into E, δ into Δ, x_t into X_t) */
label mjx-container, .plot-label mjx-container, .sbn-lbl mjx-container,
.callout-lead mjx-container {{ text-transform: none !important; letter-spacing: normal !important; }}
/* Synthetic Data III scoreboard tables */
.s3-table th, .s3-table td {{ padding: 5px 10px; border-bottom: 1px solid {BORDER}; vertical-align: middle; }}
.s3-table th {{ color: {MUTED2}; font-weight: 700; font-size: 11px; text-align: left; }}
.s3-table td:first-child {{ white-space: normal; }}
.s3-table td:not(:first-child) {{ white-space: nowrap; }}
.s3-table tr.s3-group td {{ background: rgba(255,255,255,0.03); padding-top: 9px; font-size: 11px;
  letter-spacing: .03em; border-bottom: 1px solid {BORDER}; }}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  DASH APP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    title="Solar Energy · Time Series Dashboard (theory plots)",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # exposes the underlying Flask app for `gunicorn app:server`

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

        section_hdr("Time Series: Selected Variable"),
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

        # ── One plot per theory card (follows the variable selected above) ──
        section_hdr("Test by Test: one plot per theory card"),
        html.Div(
            "Each plot shows the quantity its theory card defines, for the variable selected "
            "above: the ADF regression slope, the KPSS partial sums, the Phillips-Perron "
            "correction across bandwidths, and the Zivot-Andrews break search. Gauges place "
            "the statistic against its 1/5/10% critical values; the green band is the region "
            "where the series is judged stationary.",
            className="section-desc",
        ),
        card_plot_grid(STAT_CARD_FIGS),
        STATION_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)

tab_cointegration = dbc.Tab(
    label="🔗  Cointegration",
    tab_id="tab-cointegration",
    children=html.Div([
        # ── Engle-Granger ────────────────────────────────────────────────────
        section_hdr("Engle-Granger Cointegration"),
        html.Div(
            f"Pairwise Engle-Granger cointegration test on {len(COINT_PAIRS)} physically motivated variable pairs. "
            "Cointegrated pairs share a long-run equilibrium: deviations are temporary and mean-reverting.",
            className="section-desc"
        ),
        banner(
            (len(COINT_RES), "EG pairs tested", ACCENT2),
            (n_co, "Cointegrated (EG)", GREEN),
            (n_joh, "Cointegrated (Johansen)", "#a78bfa"),
            (n_po, "Cointegrated (PO)", RED),
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

        # ── CI(1,1): what cointegration means (theory card companion) ────────
        section_hdr("Long-run Equilibrium: what Cointegration Means (CI(1,1))"),
        html.Div(
            dcc.Markdown(
                r"Cointegration says that a linear combination $Y_t - \alpha - \beta X_t$ is stationary "
                "even when $X_t$ and $Y_t$ wander. Pick a pair: the top panel shows $Y$ and its "
                r"long-run fit $\alpha + \beta X$; the bottom panel is the equilibrium error, which "
                "should keep returning to zero (within the dotted ±2σ band) if the pair is cointegrated. "
                "The half-life is how long a deviation takes to halve, from an AR(1) fit to the error.",
                mathjax=True),
            className="section-desc",
        ),
        html.Div([
            html.Div([
                html.Label("Cointegration pair (Y ↔ X)"),
                dcc.Dropdown(id="ci-pair", options=COINT_OPTIONS,
                             value=COINT_OPTIONS[0]["value"] if COINT_OPTIONS else None,
                             clearable=False, style={"minWidth": "420px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),
        plot_card([dcc.Graph(id="ci-equilibrium", config=PLOTLY_CFG)],
                  label="Long-run fit and equilibrium error  ·  green error = EG cointegrated at 5%"),

        # ── Johansen ─────────────────────────────────────────────────────────
        section_hdr("Johansen Cointegration"),
        html.Div(
            "Multivariate likelihood-ratio test for cointegrating rank. "
            "Trace statistic tests H₀: rank ≤ r. Amber markers show 5% critical values. "
            "Rank ≥ 1 indicates at least one cointegrating vector.",
            className="section-desc"
        ),
        plot_card(
            [dcc.Graph(id="johansen-bars", figure=fig_johansen_bars(), config=PLOTLY_CFG)],
            label="Trace statistic (H₀: rank=0)  ·  amber marker = 5% critical value"
        ),
        plot_card(
            [dcc.Graph(id="johansen-rank", figure=fig_johansen_rank_heatmap(), config=PLOTLY_CFG)],
            label="Estimated cointegrating rank per pair"
        ),

        # ── Phillips-Ouliaris ─────────────────────────────────────────────────
        section_hdr("Phillips-Ouliaris Cointegration"),
        html.Div(
            "Residual-based test using a non-parametric PP correction on OLS residuals. "
            "Robust to heteroscedastic and autocorrelated errors without manual lag selection.",
            className="section-desc"
        ),
        plot_card(
            [dcc.Graph(id="po-bar", figure=fig_po_bars(), config=PLOTLY_CFG)],
            label="PO Z(t) statistic  ·  more negative = stronger cointegration"
        ),
        plot_card(
            [dcc.Graph(id="po-pvals", figure=fig_po_pvals(), config=PLOTLY_CFG)],
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

        # ── Key Caveats: lag robustness (theory card companion) ──────────────
        section_hdr("Key Caveats: Lag Robustness"),
        html.Div(
            "A Granger result is most credible when it holds across lag choices. Each row is one "
            f"pair tested at every lag from 1 to {MAX_LAG_GC} h; a pair is marked 'robust' only "
            "when it is significant at all of them.",
            className="section-desc",
        ),
        banner(
            (n_robust, "Robust at every lag", GREEN),
            (n_gc - n_robust, "Significant at some lags only", AMBER),
            (len(GRANGER_RES) - n_gc, "Never significant", RED),
        ),
        plot_card([dcc.Graph(id="gc-lag-robust", figure=fig_granger_lag_robustness(),
                             config=PLOTLY_CFG)],
                  label="−log₁₀(p) per pair and lag  ·  ✔ = significant at 5%"),

        # ── VAR block exogeneity (theory card companion) ─────────────────────
        section_hdr("VAR Block Exogeneity"),
        html.Div(
            f"One VAR({VAR_ORDER}) fitted jointly to all {len(_extras['var_targets'])} "
            "first-differenced, standardised variables (lag order chosen by HQIC, up to "
            f"{VAR_MAX_LAG} h). For each block of related variables and each target, a Wald test "
            "asks whether all the block's lagged coefficients in the target's equation are jointly "
            "zero (the target itself is left out of its own block). The second chart re-tests each "
            "Granger pair inside the same VAR, so the other variables are controlled for.",
            className="section-desc",
        ),
        banner(
            (VAR_ORDER, "VAR lag order (HQIC)", ACCENT2),
            (n_block_sig, f"Non-exogenous blocks (of {len(BLOCK_RES)})", GREEN),
            (n_cond_sig, f"Pairs significant in the VAR (of {len(COND_RES)})", AMBER),
            (n_cond_lost, "Pairwise links lost in the VAR", RED),
        ),
        plot_card([dcc.Graph(id="var-block", figure=fig_var_block_heatmap(), config=PLOTLY_CFG)],
                  label="Block → target  ·  −log₁₀(p) of the block Wald test"),
        plot_card([dcc.Graph(id="var-vs-pairwise", figure=fig_var_vs_pairwise(), config=PLOTLY_CFG)],
                  label="Pairwise vs system-conditional p-values"),

        # ── Convergent cross mapping (theory card companion) ─────────────────
        section_hdr("Convergent Cross Mapping (CCM)"),
        html.Div(
            "For each Granger pair X → Y, the shadow manifold of Y is rebuilt from delays "
            "(τ from the ACF, E from simplex self-prediction) and used to estimate X from its "
            "nearest neighbours. Evidence that X drives Y is a skill ρ that rises and levels off "
            "as the library grows. The reverse direction is shown for comparison; both directions "
            "converging strongly usually means a shared driver (here, the solar cycle) or strong "
            "synchronisation rather than one-way causation.",
            className="section-desc",
        ),
        banner(
            (len(CCM_RES), "Pairs cross-mapped", ACCENT2),
            (n_ccm_conv, "Converging X → Y (ρ > 0.3, Δρ > 0.05)", GREEN),
            (CCM_N_SAMPLES, "Random libraries per L", AMBER),
        ),
        html.Div([
            html.Div([
                html.Label("Granger pair (cause → effect)"),
                dcc.Dropdown(id="ccm-pair", options=GC_OPTIONS,
                             value=GC_OPTIONS[0]["value"] if GC_OPTIONS else None,
                             clearable=False, style={"minWidth": "420px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),
        plot_card([dcc.Graph(id="ccm-curve", config=PLOTLY_CFG)],
                  label="Cross-map skill vs library size  ·  both directions"),
        plot_card([dcc.Graph(id="ccm-summary", figure=fig_ccm_summary(), config=PLOTLY_CFG)],
                  label="Final skill for every pair  ·  next to the pairwise Granger verdict"),

        GRANGER_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)

tab_diagnostics = dbc.Tab(
    label="🔬  Diagnostics",
    tab_id="tab-diagnostics",
    children=html.Div([
        section_hdr("Residual Diagnostics: Normality & Autocorrelation"),
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
                html.Label("Variable: full diagnostic panel"),
                dcc.Dropdown(
                    id="diag-var", options=VAR_OPTIONS,
                    value=AVAILABLE[0], clearable=False,
                    style={"minWidth": "280px"},
                ),
            ], className="ctrl-block"),
        ], className="controls-row", style={"marginTop": "20px"}),
        plot_card([dcc.Graph(id="diag-variable-panel", config=PLOTLY_CFG)]),

        # ── Regression-residual diagnostics ─────────────────────────────
        section_hdr("Regression-Residual Diagnostics: Heteroskedasticity & Serial Correlation"),
        html.Div(
            dcc.Markdown("Three classical regression diagnostics applied to the residuals of a constant + " \
            r"linear-trend fit ($y_t = \alpha + \beta t + \varepsilon_t$) for every variable: " \
            "Breusch-Pagan for heteroskedasticity (non-constant variance); " \
            "Breusch-Godfrey (up to lag 12) and Durbin-Watson (lag 1) for serial correlation. " \
            "The per-variable panel below follows the variable selected above.", mathjax=True),
            className="section-desc",
        ),
        banner(
            (n_homosk,    f"Homoskedastic (BP, n={len(AVAILABLE)})", AMBER),
            (n_nocorr_bg, "No serial corr. (BG)",                    "#60a5fa"),
            (n_dw_ok,     "DW ≈ 2 (no lag-1)",                       GREEN),
            ("5%",        "Significance level",                      ACCENT2),
        ),
        plot_card(
            [dcc.Graph(id="regdiag-overview", figure=fig_regdiag_overview(), config=PLOTLY_CFG)],
            label="All-variable overview  ·  green = homoskedastic / no serial correlation / DW≈2",
        ),
        plot_card(
            [dcc.Graph(id="regdiag-variable", config=PLOTLY_CFG)],
            label="Per-variable detail  ·  BP & BG significance + Durbin-Watson gauge",
        ),


        # ── One plot per theory card (follows the variable selected above) ──
        section_hdr("Test by Test: one plot per theory card"),
        html.Div(
            "One plot per diagnostic card, for the variable selected above. KS, SW, JB, LB and "
            "HQC are computed on the series itself; BP, BG and DW on the residuals of the "
            "constant + linear-trend fit, as in the battery.",
            className="section-desc",
        ),
        card_plot_grid(DIAG_CARD_FIGS),
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
            [tab_exploration, tab_stationarity, tab_cointegration, tab_causality, tab_diagnostics, tab_synth, tab_synth2, tab_synth3, tab_rqa, tab_map],
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
register_map_callbacks(app)
register_synth_callbacks(app)
register_synth2_callbacks(app)
register_rqa_callbacks(app)
register_synth3_callbacks(app)

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


@callback(
    Output("regdiag-variable", "figure"),
    Input("diag-var", "value"),
)
def update_regdiag_panel(col):
    if not col or col not in REGDIAG_RES:
        return go.Figure()
    return fig_regdiag_variable(col)


@callback(
    Output("ci-equilibrium", "figure"),
    Input("ci-pair", "value"),
)
def update_ci_equilibrium(key):
    if not key or key not in COINT_RES:
        return go.Figure()
    return fig_ci_equilibrium(key)


@callback(
    Output("ccm-curve", "figure"),
    Input("ccm-pair", "value"),
)
def update_ccm_curve(key):
    if not key or key not in CCM_RES:
        return go.Figure()
    return fig_ccm_curves(key)


@callback(
    [Output(gid, "figure") for _, gid, _ in STAT_CARD_FIGS],
    Input("stat-var", "value"),
)
def update_stat_cards(col):
    if not col or col not in STAT_RES:
        return [go.Figure()] * len(STAT_CARD_FIGS)
    return [_safe(fn, col) for _, _, fn in STAT_CARD_FIGS]


@callback(
    [Output(gid, "figure") for _, gid, _ in DIAG_CARD_FIGS],
    Input("diag-var", "value"),
)
def update_diag_cards(col):
    if not col or col not in DIAG_RES:
        return [go.Figure()] * len(DIAG_CARD_FIGS)
    return [_safe(fn, col) for _, _, fn in DIAG_CARD_FIGS]


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)