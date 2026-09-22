"""
rqa_tab.py  —  Class-III Dynamical-Potential Analysis: Recurrence Quantification
Drop this file next to app.py and:
    from rqa_tab import tab_rqa, register_rqa_callbacks
    ... add `tab_rqa` to the dbc.Tabs children list ...
    register_rqa_callbacks(app)

WHAT THIS TAB DOES
-------------------
Of the three classes of time-series synthetic/analytic method referenced
throughout this dashboard —
  (1) mimic the *shape*               (synth_tab_control.py  — SARIMA)
  (2) reproduce the *statistics*      (synth2_tab.py          — IAAFT / bootstrap / copula)
  (3) reproduce the *dynamical potential* (chaos theory, phase-space reconstruction,
      recurrence quantification analysis)
this tab is the analytic counterpart of Class (3): rather than generating a
synthetic path, it reconstructs each selected variable's phase-space
trajectory (Takens' delay embedding) and quantifies how it recurs — on its
own (auto-recurrence) and jointly with a second, user-selected variable
(cross- and joint-recurrence) — the direct, model-free complement to the
Granger-causality tab's pointer toward CCM, Lyapunov exponents and Hurst
exponents.

Two series are always in view because the question this tab answers is
inherently relational: "when system A revisits a state, does system B tend
to be revisiting one of its own states at the same time?" A high cross-
recurrence rate and determinism between two reconstructed trajectories is
evidence of shared or coupled dynamics that a linear tool (ACF, Granger,
cointegration) can miss entirely, because recurrence only asks about
*distance in reconstructed state space*, never about linear predictability.

Per-variable *identification* (the delay τ from the first minimum of the
average mutual information, and the embedding dimension m from the false-
nearest-neighbours fraction) is computed once, eagerly, in a background
thread at import time — mirroring synth2_tab.py's pattern — and persisted
to disk via cache_utils so a later run with the same spg.csv and config
loads it instantly instead of recomputing. Building the recurrence
matrices themselves is cheap (a handful of milliseconds up to a few
hundred embedded points) and stays fully interactive, driven live by the
tab's controls.
"""

from pathlib import Path
from datetime import datetime
import math
import threading
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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

DEFAULT_COL_A = "shortwave_radiation_backwards_sfc"
DEFAULT_COL_B = "temperature_2_m_above_gnd"

# ─────────────────────────────────────────────────────────────────────────────
#  RQA CONFIG
# ─────────────────────────────────────────────────────────────────────────────
AMI_MAX_LAG        = 30     # lags scanned for the average-mutual-information delay estimate
AMI_BINS           = 16     # histogram bins for the AMI estimator
FNN_MAX_DIM        = 10     # highest embedding dimension tried by false-nearest-neighbours
FNN_RTOL           = 15.0   # Kennel et al. (1992) criterion-1 threshold
FNN_ATOL           = 2.0    # Hegger/Kantz criterion-2 threshold (relative to attractor size)
FNN_THRESHOLD_PCT  = 10.0   # stop increasing m once %FNN falls at/below this

TAU_MIN, TAU_MAX   = 1, 24
M_MIN, M_MAX        = 1, 8
WINDOW_MIN, WINDOW_MAX, WINDOW_DEFAULT = 150, 600, 300
RR_MIN, RR_MAX, RR_DEFAULT = 2, 25, 10       # target recurrence rate, percent
L_MIN, V_MIN        = 2, 2                    # minimum diagonal/vertical line length for DET/LAM
MAX_PLOT_POINTS     = 35_000                  # per-panel recurrence-point cap (keeps WebGL responsive)

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
N_OBS = len(DF)
VAR_OPTIONS = [{"label": LABELS.get(c, c), "value": c} for c in AVAILABLE]

if DEFAULT_COL_A not in AVAILABLE:
    DEFAULT_COL_A = AVAILABLE[0]
if DEFAULT_COL_B not in AVAILABLE or DEFAULT_COL_B == DEFAULT_COL_A:
    DEFAULT_COL_B = next((c for c in AVAILABLE if c != DEFAULT_COL_A), AVAILABLE[0])

WINDOW_DEFAULT = min(WINDOW_DEFAULT, max(WINDOW_MIN, N_OBS - 1))
_WIN_START_MAX = max(0, N_OBS - WINDOW_MIN)
_WIN_START_DEFAULT = max(0, min(N_OBS - WINDOW_DEFAULT, _WIN_START_MAX))


# ─────────────────────────────────────────────────────────────────────────────
#  IDENTIFICATION  —  average mutual information (delay τ) + false nearest
#  neighbours (embedding dimension m), the two ingredients Takens' theorem
#  needs to turn a scalar series into a phase-space trajectory.
# ─────────────────────────────────────────────────────────────────────────────
def _ami(z: np.ndarray, max_lag: int, bins: int) -> np.ndarray:
    """Average mutual information I(lag) between z_t and z_{t+lag}, lag=1..max_lag."""
    n = len(z)
    edges = np.histogram_bin_edges(z, bins=bins)
    codes = np.clip(np.digitize(z, edges) - 1, 0, bins - 1)
    ami = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            ami[lag - 1] = np.nan
            continue
        a, b = codes[:n - lag], codes[lag:]
        joint = np.zeros((bins, bins))
        np.add.at(joint, (a, b), 1)
        total = joint.sum()
        if total <= 0:
            ami[lag - 1] = np.nan
            continue
        joint /= total
        pa = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = joint / (pa * pb)
            terms = np.where(joint > 0, joint * np.log(ratio), 0.0)
        ami[lag - 1] = terms.sum()
    return ami


def _first_minimum(ami: np.ndarray, fallback: int = 6) -> int:
    """First local minimum of the AMI curve (Fraser & Swinney's delay rule);
    falls back to the first lag where AMI drops below AMI(1)/e, then to a
    fixed default if neither criterion fires within the scanned range."""
    valid = np.where(np.isfinite(ami))[0]
    if len(valid) < 3:
        return fallback
    for i in valid[1:-1]:
        if ami[i] <= ami[i - 1] and ami[i] <= ami[i + 1]:
            return int(i + 1)               # ami[0] is lag=1
    thresh = ami[valid[0]] / math.e if ami[valid[0]] > 0 else 0.0
    below = valid[ami[valid] <= thresh]
    if len(below):
        return int(below[0] + 1)
    return fallback


def _fnn_curve(z: np.ndarray, tau: int, max_dim: int, rtol: float, atol: float):
    """% false nearest neighbours at each embedding dimension d=1..max_dim,
    following Kennel, Brown & Abarbanel (1992) with the Hegger-Kantz size
    correction. A neighbour is 'false' if promoting the embedding by one
    dimension pushes it disproportionately far away — evidence the
    apparent proximity at dimension d was a projection artefact, not a
    genuine recurrence of the underlying dynamics."""
    n = len(z)
    sigma = float(np.std(z)) or 1.0
    pct = []
    for d in range(1, max_dim + 1):
        L1 = n - (d + 1) * tau
        if L1 < 50:
            pct.append(np.nan)
            continue
        Ed = np.column_stack([z[i * tau: i * tau + L1] for i in range(d)])
        tree = cKDTree(Ed)
        dist, idx = tree.query(Ed, k=2)
        nn_dist, nn_idx = dist[:, 1], idx[:, 1]
        extra    = z[np.arange(L1) + d * tau]
        extra_nn = z[nn_idx + d * tau]
        delta = np.abs(extra - extra_nn)
        nn_safe = np.where(nn_dist > 1e-10, nn_dist, np.inf)
        crit1 = (delta / nn_safe) > rtol
        crit2 = (np.sqrt(nn_dist ** 2 + delta ** 2) / sigma) > atol
        is_false = crit1 | crit2
        pct.append(float(100.0 * np.mean(is_false)))
    return pct


def _select_dim(fnn_pct, threshold: float = FNN_THRESHOLD_PCT, fallback: int = 3) -> int:
    for d, p in enumerate(fnn_pct, start=1):
        if not np.isnan(p) and p <= threshold:
            return d
    valid = [(d, p) for d, p in enumerate(fnn_pct, start=1) if not np.isnan(p)]
    return min(valid, key=lambda t: t[1])[0] if valid else fallback


def _identify(col: str) -> dict:
    s = DF[col].dropna().values.astype(float)
    n = len(s)
    mu, sd = float(np.mean(s)), float(np.std(s))
    z = (s - mu) / (sd if sd > 1e-12 else 1.0)

    ami_curve = _ami(z, AMI_MAX_LAG, AMI_BINS)
    tau = int(np.clip(_first_minimum(ami_curve), TAU_MIN, TAU_MAX))

    fnn_curve = _fnn_curve(z, tau, FNN_MAX_DIM, FNN_RTOL, FNN_ATOL)
    m = int(np.clip(_select_dim(fnn_curve), M_MIN, M_MAX))

    return dict(col=col, n=n, mu=mu, sd=sd, tau=tau, m=m,
                ami_curve=ami_curve, fnn_curve=fnn_curve)


# ─────────────────────────────────────────────────────────────────────────────
#  EAGER PRECOMPUTE  — identification for every variable, background thread,
#  with an on-disk cache: a later run with the same spg.csv and config skips
#  straight to the cached (τ, m, AMI curve, FNN curve) for every variable.
# ─────────────────────────────────────────────────────────────────────────────
_ID_CACHE: dict = {}
_STATUS_LOCK = threading.Lock()
_STATUS = dict(current=None, done=0, total=len(AVAILABLE), finished=False, errors={})

_ID_SIG = make_signature(
    CSV_PATH, version="rqa-identification-v1",
    ami_max_lag=AMI_MAX_LAG, ami_bins=AMI_BINS,
    fnn_max_dim=FNN_MAX_DIM, fnn_rtol=FNN_RTOL, fnn_atol=FNN_ATOL,
    fnn_threshold=FNN_THRESHOLD_PCT, tau_bounds=(TAU_MIN, TAU_MAX), m_bounds=(M_MIN, M_MAX),
    available=AVAILABLE,
)


def _precompute_worker():
    t_start = time.time()
    queue = [DEFAULT_COL_A, DEFAULT_COL_B] + \
            [c for c in AVAILABLE if c not in (DEFAULT_COL_A, DEFAULT_COL_B)]
    for i, col in enumerate(queue, start=1):
        with _STATUS_LOCK:
            _STATUS["current"] = col
        t0 = time.time()
        try:
            ident = _identify(col)
            _ID_CACHE[col] = ident
            print(f"[rqa_tab]  ({i}/{len(queue)}) {col}: tau={ident['tau']}  "
                  f"m={ident['m']}  [{time.time() - t0:.1f}s]")
        except Exception as e:                          # never kill the thread
            with _STATUS_LOCK:
                _STATUS["errors"][col] = f"{type(e).__name__}: {e}"
            print(f"[rqa_tab]  ({i}/{len(queue)}) {col}: identification failed — "
                  f"{type(e).__name__}: {e}")
        with _STATUS_LOCK:
            _STATUS["done"] = i
    print(f"[rqa_tab] Ready OK — phase-space identification done in {time.time() - t_start:.1f}s")
    # Save before flipping `finished`, so finished=True always implies the
    # cache write already landed on disk (see synth_tab_control.py for why).
    save_cache("rqa_identification", _ID_SIG, dict(_ID_CACHE))
    with _STATUS_LOCK:
        _STATUS["finished"] = True
        _STATUS["current"] = None


_cached_ids = load_cache("rqa_identification", _ID_SIG)
if _cached_ids is not None:
    _ID_CACHE.update(_cached_ids)
    with _STATUS_LOCK:
        _STATUS.update(done=len(AVAILABLE), total=len(AVAILABLE), finished=True, current=None)
    print(f"[rqa_tab] Loaded cached phase-space identification for {len(_ID_CACHE)} "
          "variables from .cache/ ✔  (delete .cache/ to force a full recompute)")
else:
    print(f"[rqa_tab] Starting background phase-space identification (AMI delay + "
          f"FNN embedding dimension) for {len(AVAILABLE)} variables — recurrence "
          "analysis itself does not wait on this and is interactive immediately.")
    _WORKER = threading.Thread(target=_precompute_worker, daemon=True, name="rqa-precompute")
    _WORKER.start()


def get_ident(col: str):
    return _ID_CACHE.get(col)


def precompute_status() -> dict:
    with _STATUS_LOCK:
        return dict(_STATUS)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE-SPACE EMBEDDING & RECURRENCE  (on demand — driven live by the tab's
#  controls; cheap enough at these window sizes to run in every callback)
# ─────────────────────────────────────────────────────────────────────────────
def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Takens delay-coordinate embedding: row i is (x_i, x_{i+τ}, ..., x_{i+(m-1)τ})."""
    n = len(x)
    L = n - (m - 1) * tau
    if L <= 0:
        return np.empty((0, m))
    return np.column_stack([x[i * tau: i * tau + L] for i in range(m)])


def _clean_window(arr: np.ndarray) -> np.ndarray:
    """Fill any gaps in a contiguous slice via linear interpolation (plus
    edge fill), so an occasional missing reading never breaks the embedding
    or misaligns the two series in time."""
    s = pd.Series(arr, dtype=float)
    if s.isna().any():
        s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.values


def _epsilon_for_rr(D: np.ndarray, target_rr: float, exclude_diagonal: bool) -> float:
    """Fixed-recurrence-rate threshold: the distance ε such that a target
    fraction of the (unique, off-diagonal for auto-RP) pairwise distances
    fall at or below it — the standard way to make ε comparable across
    variables of very different physical scale (Marwan et al., 2007)."""
    if exclude_diagonal and D.shape[0] > 1:
        iu = np.triu_indices_from(D, k=1)
        vals = D[iu]
    else:
        vals = D.ravel()
    if vals.size == 0:
        return 0.0
    q = float(np.clip(target_rr / 100.0, 1e-4, 1.0))
    return float(np.quantile(vals, q))


def build_recurrence(y_a: np.ndarray, y_b: np.ndarray, m: int, tau: int, target_rr: float) -> dict:
    """Standardise both windows, embed both with the *same* (m, τ) — required
    for cross- and joint-recurrence to be meaningful — and threshold each
    pairwise-distance matrix at a fixed recurrence rate."""
    za = (y_a - y_a.mean()) / (y_a.std() if y_a.std() > 1e-12 else 1.0)
    zb = (y_b - y_b.mean()) / (y_b.std() if y_b.std() > 1e-12 else 1.0)

    Ea, Eb = embed(za, m, tau), embed(zb, m, tau)
    L = min(len(Ea), len(Eb))
    Ea, Eb = Ea[:L], Eb[:L]

    if L < 10:
        return dict(L=0)

    Da, Db, Dc = cdist(Ea, Ea), cdist(Eb, Eb), cdist(Ea, Eb)
    eps_a = _epsilon_for_rr(Da, target_rr, exclude_diagonal=True)
    eps_b = _epsilon_for_rr(Db, target_rr, exclude_diagonal=True)
    eps_c = _epsilon_for_rr(Dc, target_rr, exclude_diagonal=False)

    R_a, R_b, R_c = Da <= eps_a, Db <= eps_b, Dc <= eps_c
    R_j = R_a & R_b     # joint recurrence: simultaneous auto-recurrence in both systems

    return dict(R_a=R_a, R_b=R_b, R_cross=R_c, R_joint=R_j,
                eps_a=eps_a, eps_b=eps_b, eps_cross=eps_c, L=L)


def _run_lengths(mask: np.ndarray, min_len: int) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=int)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(padded)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    lengths = ends - starts
    return lengths[lengths >= min_len]


def rqa_measures(R: np.ndarray, theiler: int = 0, l_min: int = L_MIN, v_min: int = V_MIN) -> dict:
    """The classical RQA measures (Zbilut & Webber; Marwan et al., 2007):
    RR, DET, LAM, mean/longest diagonal line, divergence, Shannon entropy of
    the diagonal-line-length distribution, trapping time and longest
    vertical line. `theiler` zeroes a band around the main diagonal before
    measuring, so short lines caused by the trajectory's own smoothness
    (not genuine recurrence) never inflate DET/LAM."""
    N = R.shape[0] if R.ndim == 2 else 0
    empty = dict(rr=0.0, det=0.0, lam=0.0, L=0.0, Lmax=0, div=0.0, entr=0.0,
                 tt=0.0, vmax=0, diag_lengths=np.array([], dtype=int), n_points=0)
    if N == 0:
        return empty

    if theiler > 0:
        Rm = R.copy()
        ii = np.arange(N)
        band = np.abs(ii[:, None] - ii[None, :]) < theiler
        Rm[band] = False
        denom = N * N - int(band.sum())
    else:
        Rm, denom = R, N * N
    if denom <= 0:
        return empty

    n_rec = int(Rm.sum())
    rr = 100.0 * n_rec / denom

    diag_chunks = []
    for k in range(-(N - 1), N):
        if theiler > 0 and abs(k) < theiler:
            continue
        diag_chunks.append(_run_lengths(np.diagonal(Rm, offset=k), l_min))
    diag_lengths = np.concatenate(diag_chunks) if diag_chunks else np.array([], dtype=int)

    vert_chunks = [_run_lengths(Rm[:, j], v_min) for j in range(N)]
    vert_lengths = np.concatenate(vert_chunks) if vert_chunks else np.array([], dtype=int)

    det = 100.0 * diag_lengths.sum() / n_rec if n_rec else 0.0
    Lmean = float(diag_lengths.mean()) if diag_lengths.size else 0.0
    Lmax = int(diag_lengths.max()) if diag_lengths.size else 0
    div = (1.0 / Lmax) if Lmax > 0 else 0.0

    if diag_lengths.size:
        _, counts = np.unique(diag_lengths, return_counts=True)
        p = counts / counts.sum()
        entr = float(-(p * np.log(p)).sum())
    else:
        entr = 0.0

    lam = 100.0 * vert_lengths.sum() / n_rec if n_rec else 0.0
    tt = float(vert_lengths.mean()) if vert_lengths.size else 0.0
    vmax = int(vert_lengths.max()) if vert_lengths.size else 0

    return dict(rr=rr, det=det, lam=lam, L=Lmean, Lmax=Lmax, div=div, entr=entr,
                tt=tt, vmax=vmax, diag_lengths=diag_lengths, n_points=n_rec)


def _coupling_verdict(mA: dict, mB: dict, mC: dict):
    avg_auto_det = (mA["det"] + mB["det"]) / 2.0
    if mC["n_points"] == 0:
        return "No cross-recurrence at this threshold", MUTED2
    if avg_auto_det > 0 and mC["det"] >= 0.7 * avg_auto_det:
        return "Strong coupling — shared nonlinear structure", GREEN
    if avg_auto_det > 0 and mC["det"] >= 0.35 * avg_auto_det:
        return "Moderate coupling", AMBER
    return "Weak / no detectable coupling", RED


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
                margin=dict(t=40, b=40, l=56, r=16))
    base.update(kw)
    return base


def _scene():
    return dict(
        bgcolor=SURFACE,
        xaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED2, showbackground=True),
        yaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED2, showbackground=True),
        zaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED2, showbackground=True),
    )


def _placeholder_fig(msg: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=12, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**_base_layout(height=height, showlegend=False))
    return fig


def fig_phase_space(y_a, y_b, m, tau, colA, colB) -> go.Figure:
    za = (y_a - y_a.mean()) / (y_a.std() if y_a.std() > 1e-12 else 1.0)
    zb = (y_b - y_b.mean()) / (y_b.std() if y_b.std() > 1e-12 else 1.0)
    viz_dim = 3 if m >= 3 else 2
    Ea, Eb = embed(za, viz_dim, tau), embed(zb, viz_dim, tau)
    la, lb = LABELS.get(colA, colA), LABELS.get(colB, colB)
    ca, cb = COLORS.get(colA, ACCENT2), COLORS.get(colB, ACCENT)

    if viz_dim == 3:
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                             subplot_titles=(f"Attractor · {la}", f"Attractor · {lb}"))
        fig.add_trace(go.Scatter3d(x=Ea[:, 0], y=Ea[:, 1], z=Ea[:, 2], mode="markers",
                                    marker=dict(size=2, color=ca, opacity=0.65),
                                    name=la, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=Eb[:, 0], y=Eb[:, 1], z=Eb[:, 2], mode="markers",
                                    marker=dict(size=2, color=cb, opacity=0.65),
                                    name=lb, hoverinfo="skip"), row=1, col=2)
        fig.update_layout(**_base_layout(height=420, showlegend=False),
                           scene=_scene(), scene2=_scene())
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Attractor · {la}", f"Attractor · {lb}"))
        fig.add_trace(go.Scattergl(x=Ea[:, 0], y=Ea[:, 1], mode="markers",
                                    marker=dict(size=3, color=ca, opacity=0.55),
                                    hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scattergl(x=Eb[:, 0], y=Eb[:, 1], mode="markers",
                                    marker=dict(size=3, color=cb, opacity=0.55),
                                    hoverinfo="skip"), row=1, col=2)
        fig.update_xaxes(**_AX, title_text="x(t)")
        fig.update_yaxes(**_AX, title_text=f"x(t+τ)")
        fig.update_layout(**_base_layout(height=360, showlegend=False))
    return fig


def _rp_scatter(R: np.ndarray, color: str, max_points: int = MAX_PLOT_POINTS):
    ii, jj = np.where(R)
    n = len(ii)
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        ii, jj = ii[sel], jj[sel]
    return go.Scattergl(x=jj, y=ii, mode="markers",
                         marker=dict(size=2, color=color, opacity=0.75),
                         hoverinfo="skip")


def fig_recurrence_grid(rec: dict, colA: str, colB: str) -> go.Figure:
    la, lb = LABELS.get(colA, colA), LABELS.get(colB, colB)
    ca, cb = COLORS.get(colA, ACCENT2), COLORS.get(colB, ACCENT)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f"Auto-RP · {la}", f"Auto-RP · {lb}",
                         f"Cross-RP · {la} ↔ {lb}", "Joint-RP · simultaneous recurrence"),
        horizontal_spacing=0.09, vertical_spacing=0.14,
    )
    fig.add_trace(_rp_scatter(rec["R_a"], ca), row=1, col=1)
    fig.add_trace(_rp_scatter(rec["R_b"], cb), row=1, col=2)
    fig.add_trace(_rp_scatter(rec["R_cross"], ACCENT2), row=2, col=1)
    fig.add_trace(_rp_scatter(rec["R_joint"], AMBER), row=2, col=2)
    L = rec["L"]
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(row=r, col=c, range=[0, L], **_AX)
            fig.update_yaxes(row=r, col=c, range=[0, L], **_AX)
    fig.update_layout(**_base_layout(height=680, showlegend=False))
    return fig


def fig_rqa_bars(mA, mB, mC, mJ, colA, colB) -> go.Figure:
    metrics = ["rr", "det", "lam"]
    metric_labels = {"rr": "RR (%)", "det": "DET (%)", "lam": "LAM (%)"}
    names = [f"Auto · {LABELS.get(colA, colA)}", f"Auto · {LABELS.get(colB, colB)}",
             "Cross", "Joint"]
    palette = [COLORS.get(colA, ACCENT2), COLORS.get(colB, ACCENT), ACCENT2, AMBER]
    ms = [mA, mB, mC, mJ]
    fig = go.Figure()
    for name, color, m in zip(names, palette, ms):
        fig.add_trace(go.Bar(x=[metric_labels[k] for k in metrics],
                              y=[m[k] for k in metrics],
                              name=name, marker_color=color))
    fig.update_layout(barmode="group", **_base_layout(height=320, showlegend=True))
    fig.update_yaxes(**_AX, title_text="%")
    fig.update_xaxes(**_AX)
    return fig


def fig_diagonal_hist(diag_lengths: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if diag_lengths.size:
        fig.add_trace(go.Histogram(x=diag_lengths, marker_color=ACCENT2,
                                    xbins=dict(size=1)))
    else:
        return _placeholder_fig("No diagonal lines at this threshold / window.", height=260)
    fig.update_layout(**_base_layout(height=260, showlegend=False))
    fig.update_xaxes(**_AX, title_text="Diagonal line length ℓ (cross-recurrence)")
    fig.update_yaxes(**_AX, title_text="Count")
    return fig


def fig_identification(ident: dict, col: str) -> go.Figure:
    la = LABELS.get(col, col)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Average mutual information · {la}", f"False nearest neighbours · {la}"))
    ami = np.asarray(ident["ami_curve"])
    lags = np.arange(1, len(ami) + 1)
    fig.add_trace(go.Scatter(x=lags, y=ami, mode="lines+markers",
                              line=dict(color=ACCENT2, width=1.6),
                              marker=dict(size=4), hoverinfo="skip"), row=1, col=1)
    fig.add_vline(x=ident["tau"], line=dict(color=AMBER, width=1.5, dash="dash"), row=1, col=1)
    fnn = np.asarray(ident["fnn_curve"])
    dims = np.arange(1, len(fnn) + 1)
    fig.add_trace(go.Scatter(x=dims, y=fnn, mode="lines+markers",
                              line=dict(color=GREEN, width=1.6),
                              marker=dict(size=4), hoverinfo="skip"), row=1, col=2)
    fig.add_vline(x=ident["m"], line=dict(color=AMBER, width=1.5, dash="dash"), row=1, col=2)
    fig.update_xaxes(row=1, col=1, **_AX, title_text="Lag τ (hours)")
    fig.update_yaxes(row=1, col=1, **_AX, title_text="AMI (nats)")
    fig.update_xaxes(row=1, col=2, **_AX, title_text="Embedding dimension m")
    fig.update_yaxes(row=1, col=2, **_AX, title_text="False neighbours (%)")
    fig.update_layout(**_base_layout(height=260, showlegend=False))
    return fig


_RQA_ROWS = [
    ("rr",   "Recurrence Rate  RR (%)",            False),
    ("det",  "Determinism  DET (%)",                False),
    ("lam",  "Laminarity  LAM (%)",                 False),
    ("L",    "Mean diagonal length  L",             False),
    ("Lmax", "Longest diagonal line  Lmax",          True),
    ("div",  "Divergence  DIV = 1/Lmax",             False),
    ("entr", "Diagonal entropy  ENTR (nats)",        False),
    ("tt",   "Trapping Time  TT",                    False),
    ("vmax", "Longest vertical line  Vmax",           True),
]


def rqa_metric_table(mA, mB, mC, mJ, colA, colB):
    header = html.Tr([html.Th("Measure"),
                       html.Th(f"Auto · {LABELS.get(colA, colA)}"),
                       html.Th(f"Auto · {LABELS.get(colB, colB)}"),
                       html.Th("Cross", style={"color": ACCENT2}),
                       html.Th("Joint", style={"color": AMBER})])
    rows = [header]
    for key, name, is_int in _RQA_ROWS:
        vals = []
        for m in (mA, mB, mC, mJ):
            v = m[key]
            vals.append(f"{int(v):,}" if is_int else f"{v:.3f}")
        rows.append(html.Tr([html.Td(name), html.Td(vals[0]), html.Td(vals[1]),
                              html.Td(vals[2], style={"color": ACCENT2}),
                              html.Td(vals[3], style={"color": AMBER})]))
    return html.Table(rows, style={
        "width": "100%", "borderCollapse": "collapse", "fontFamily": FONT_MONO,
        "fontSize": "12px", "color": TEXT,
    }, className="s2-moment-table")


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


def identification_card(identA: dict, identB: dict, colA: str, colB: str):
    la, lb = LABELS.get(colA, colA), LABELS.get(colB, colB)
    return html.Div([
        _theory_card(
            "Identifying τ and m for THIS pair", "IDENTIFICATION", GREEN,
            [
                [
                    f"Cross- and joint-recurrence require one shared (m, τ) applied to both "
                    f"series, so the sliders default to {la}'s own identification: a delay "
                    f"τ = {identA['tau']} h from the first minimum of its average mutual "
                    f"information, and an embedding dimension m = {identA['m']} from the "
                    f"false-nearest-neighbours curve dropping at/below "
                    f"{FNN_THRESHOLD_PCT:.0f}%. ",
                    html.Br(), html.Br(),
                    f"{lb}, identified independently, would itself suggest τ = "
                    f"{identB['tau']} h and m = {identB['m']}. When the two disagree, the "
                    "shared-embedding compromise trades a small amount of optimality for "
                    "each individual series in exchange for a cross-recurrence plot that "
                    "is mathematically well-defined; both sliders remain fully adjustable "
                    "to explore either side of that trade-off.",
                ],
                dcc.Graph(figure=fig_identification(identA, colA),
                          config={"displayModeBar": False}),
            ],
        ),
    ], className="theory-row")


# ─────────────────────────────────────────────────────────────────────────────
#  THEORY PANEL
# ─────────────────────────────────────────────────────────────────────────────
RQA_THEORY = html.Div([
    html.Div("📖 Class-III Dynamical-Potential Analysis — Theory", className="theory-title"),

    html.P([
        "The Class-1 and Class-2 tabs each build a ", html.I("model"), " of one series and "
        "validate it against that same series. This tab does something different: it makes "
        "no distributional or shape assumption at all, and it always looks at ", html.B("two"),
        " series together. By Takens' delay-embedding theorem, a single scalar sequence "
        "already contains — under mild conditions — enough information to reconstruct the "
        "geometry of the underlying dynamical system that produced it. Reconstructing that "
        "geometry for two variables and asking where their reconstructed trajectories revisit "
        "the same neighbourhoods (cross-recurrence) or revisit their own neighbourhoods at "
        "the same instants (joint recurrence) is a model-free way to detect coupling that "
        "linear tools — ACF, Granger causality, cointegration — can miss when the "
        "relationship is nonlinear.",
    ], className="section-desc", style={"marginBottom": "18px"}),

    html.Div([
        html.Div([
            _theory_card(
                "Phase-space reconstruction — Takens' theorem", "STEP ①", ACCENT2,
                [
                    "A scalar series x₁,…,x_n is lifted into an m-dimensional trajectory by "
                    "delay embedding: each point is a window of m samples spaced τ apart, "
                    "Xᵢ = (xᵢ, xᵢ₊τ, …, xᵢ₊₍ₘ₋₁₎τ). Takens (1981) showed that for a large "
                    "enough embedding dimension m, this reconstructed trajectory is a smooth, "
                    "one-to-one image of the system's true (unobservable) state space — so "
                    "distances between reconstructed points faithfully reflect distances "
                    "between the real dynamical states that produced them.",
                    "The delay τ is chosen from the first local minimum of the average "
                    "mutual information between xₜ and xₜ₊τ (Fraser & Swinney, 1986): too "
                    "small and consecutive coordinates are redundant; too large and they "
                    "become causally unrelated. The dimension m is chosen by false nearest "
                    "neighbours (Kennel et al., 1992): points that look close together at "
                    "dimension d but fly apart at d+1 are neighbours only because the "
                    "projection was too thin to unfold the attractor; m is the smallest "
                    "dimension at which that stops happening.",
                ],
                math_eq="Xᵢ = (xᵢ, xᵢ₊τ, …, xᵢ₊₍ₘ₋₁₎τ)  ∈ ℝᵐ",
            ),
            _theory_card(
                "Auto-, cross- and joint-recurrence plots", "STEP ②", GREEN,
                [
                    "A recurrence plot marks every pair of times (i, j) whose reconstructed "
                    "states lie within a threshold ε of each other. Applied to one series "
                    "against itself it is an ", html.I("auto-recurrence plot"),
                    " — diagonal structure away from the main diagonal reveals "
                    "deterministic, near-periodic dynamics; scattered, texture-free points "
                    "look like noise.",
                    "Applied between the reconstructed trajectories of two ", html.I("different"),
                    " series (using one shared m, τ so distances are comparable) it is a "
                    "cross-recurrence plot: diagonal structure close to the main diagonal "
                    "means the two systems revisit corresponding states at corresponding "
                    "times — direct evidence of coupling. This tab's threshold ε is set "
                    "adaptively (fixed recurrence rate) so every plot targets the same "
                    "overall point density regardless of each variable's physical units.",
                    "The Hadamard product of the two auto-recurrence matrices — the ",
                    html.I("joint-recurrence plot"), " (Romano et al., 2004) — marks times "
                    "when both systems simultaneously revisit one of their own past states; "
                    "unlike the cross-recurrence plot it does not require the two systems' "
                    "state spaces to be directly comparable, only that they recur at the "
                    "same clock time.",
                ],
                math_eq="R^{cross}_{ij} = Θ(ε − ‖Xᵢᴬ − Xⱼᴮ‖),    R^{joint}_{ij} = R^{A}_{ij} · R^{B}_{ij}",
            ),
            _theory_card(
                "RQA measures — quantifying the texture", "STEP ③", AMBER,
                [
                    "Recurrence Rate (RR) is simply the density of marked points — how "
                    "often either system (or both) is 'close' at all. Determinism (DET) is "
                    "the fraction of those points that align into diagonal lines of length "
                    "≥ 2 rather than sitting in isolation — diagonal structure is the "
                    "signature of deterministic, forecastable trajectory segments, so DET "
                    "is a texture-based, model-free analogue of predictability. Laminarity "
                    "(LAM) is the equivalent count for vertical lines, and flags laminar "
                    "(slowly-changing, 'stuck') states rather than periodic ones.",
                    "The longest diagonal line Lmax bounds how long two trajectory segments "
                    "stay close before diverging — its reciprocal, Divergence (DIV), is a "
                    "coarse, computation-free proxy for the largest Lyapunov exponent "
                    "referenced in the Causality tab's theory: short Lmax (high DIV) means "
                    "fast divergence, i.e. more chaotic-looking dynamics. Entropy (ENTR) is "
                    "the Shannon entropy of the diagonal-line-length distribution — low "
                    "when most lines share one length (a clean periodicity), high when line "
                    "lengths are richly varied. Trapping Time (TT), the mean vertical line "
                    "length, is the laminar-phase counterpart of the same idea.",
                ],
                math_eq="DET = Σℓ·P(ℓ) / RR·N²,   ENTR = −Σ p(ℓ)·log p(ℓ),   DIV = 1 / Lmax",
            ),
        ], className="theory-row"),
        html.Div(id="rqa-id-card", className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px",
                                        "overflow": "hidden", "background": CARD2}),

    _callout(
        "tip", "How to read this tab — a walkthrough for any level",
        "Work top to bottom; each element answers one question in the Class-3 pipeline:",
        [
            ("Series A / Series B → ", "\"Which two variables?\" — Series A also drives the "
             "default embedding parameters, since only one (m, τ) pair can be shared."),
            ("Embedding dimension / delay sliders → ", "\"How is each scalar series lifted "
             "into a trajectory?\" — pre-filled from Series A's false-nearest-neighbours / "
             "mutual-information identification, freely overridable."),
            ("Target recurrence rate → ", "\"How strict is 'close'?\" — sets ε adaptively so "
             "every one of the four recurrence plots below targets the same point density, "
             "making their textures directly comparable."),
            ("Phase-space reconstruction → ", "\"What does each attractor look like?\" — a "
             "tight, structured cloud suggests low-dimensional deterministic dynamics; a "
             "diffuse cloud suggests high-dimensional or noisy dynamics."),
            ("Recurrence plot grid → ", "\"Where do the two systems recur, alone and "
             "together?\" — diagonal texture near the main diagonal in the cross-recurrence "
             "panel is the visual signature of coupling; compare its density and structure "
             "against the two auto-recurrence panels."),
            ("RQA measures (bars + table) → ", "\"How much of that texture is there, "
             "exactly?\" — DET/LAM/ENTR on the cross plot at a level comparable to the two "
             "auto plots is quantitative evidence of shared dynamical structure, not just a "
             "visual impression."),
        ],
    ),
    _callout(
        "note", "Limits of this analysis",
        "Recurrence quantification is powerful precisely because it makes so few "
        "assumptions — which also means it needs care in interpretation:",
        [
            "ε and the target recurrence rate are analyst choices: comparisons across "
            "variables or window lengths are only fair when the recurrence rate is held "
            "fixed, which is why this tab calibrates ε adaptively rather than exposing a "
            "raw distance threshold.",
            "A high cross-recurrence rate shows the two reconstructed trajectories occupy "
            "overlapping state-space regions at corresponding times — it is evidence "
            "consistent with coupling, not a directional causal statement. For a "
            "directionality test on the same nonlinear footing, see Convergent Cross "
            "Mapping (CCM) in the Causality tab's theory panel.",
            "Recurrence-based measures assume the dynamics are (at least locally) "
            "stationary within the analysis window; a long window spanning a genuine "
            "regime change will blend two different textures into one ambiguous plot — "
            "shorten the window or move it to isolate a single regime.",
            "The window length is capped for interactivity: at a few hundred embedded "
            "points the O(N²) recurrence matrix and its diagonal/vertical line scan stay "
            "well under a second, but the same analysis over the full multi-year record "
            "would need a batch (non-interactive) run.",
        ],
        ordered=False,
    ),
], className="theory-section")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
_WIN_MARKS = {0: "start", N_OBS // 2: "mid", _WIN_START_MAX: "end"}

tab_rqa = dbc.Tab(
    label="🔁  Recurrence (RQA)",
    tab_id="tab-rqa",
    children=html.Div([
        _section_hdr("Class-III: Recurrence Quantification Analysis (phase-space coupling)"),
        _section_desc(
            "Reconstructs the phase-space trajectory of two selected variables via Takens delay "
            "embedding and quantifies how they recur — individually (auto-recurrence) and "
            "jointly (cross- and joint-recurrence) — the model-free, nonlinear complement to "
            "the ACF/cointegration/Granger tools elsewhere in this dashboard. Embedding "
            "parameters (τ, m) are pre-identified per variable in the background (cached to "
            "disk after the first run); everything below updates live as you move the sliders."
        ),

        dcc.Interval(id="rqa-poll", interval=1500, n_intervals=0),

        html.Div([
            html.Div([
                html.Label("Series A (drives the default embedding)"),
                dcc.Dropdown(id="rqa-var-a", options=VAR_OPTIONS, value=DEFAULT_COL_A,
                             clearable=False, style={"minWidth": "280px"}),
            ], className="ctrl-block"),
            html.Div([
                html.Label("Series B (compared against A)"),
                dcc.Dropdown(id="rqa-var-b", options=VAR_OPTIONS, value=DEFAULT_COL_B,
                             clearable=False, style={"minWidth": "280px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),

        html.Div([
            html.Div([
                html.Label(f"Embedding dimension m ({M_MIN}–{M_MAX})"),
                dcc.Slider(id="rqa-m", min=M_MIN, max=M_MAX, step=1, value=3,
                           marks={i: str(i) for i in range(M_MIN, M_MAX + 1)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label(f"Time delay τ, hours ({TAU_MIN}–{TAU_MAX})"),
                dcc.Slider(id="rqa-tau", min=TAU_MIN, max=TAU_MAX, step=1, value=6,
                           marks={1: "1", 6: "6", 12: "12", 24: "24"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label(f"Target recurrence rate (% · {RR_MIN}–{RR_MAX})"),
                dcc.Slider(id="rqa-rr", min=RR_MIN, max=RR_MAX, step=1, value=RR_DEFAULT,
                           marks={RR_MIN: str(RR_MIN), 10: "10", RR_MAX: str(RR_MAX)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
        ], className="controls-row"),

        html.Div([
            html.Div([
                html.Label(f"Window length, hours ({WINDOW_MIN}–{WINDOW_MAX})"),
                dcc.Slider(id="rqa-win-len", min=WINDOW_MIN, max=WINDOW_MAX, step=10,
                           value=WINDOW_DEFAULT,
                           marks={WINDOW_MIN: str(WINDOW_MIN), WINDOW_MAX: str(WINDOW_MAX)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label("Window start (hours from series start)"),
                dcc.Slider(id="rqa-win-start", min=0, max=_WIN_START_MAX, step=24,
                           value=_WIN_START_DEFAULT, marks=_WIN_MARKS,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
        ], className="controls-row"),

        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div(id="rqa-banner"),
            _plot_card([dcc.Graph(id="rqa-phase", config={"displaylogo": False})],
                       label="Reconstructed phase-space trajectories (attractors)"),
        ]),

        _section_hdr("Recurrence Plots"),
        html.Div(id="rqa-grid-desc", className="section-desc"),
        dcc.Loading(type="circle", color=ACCENT, children=[
            _plot_card([dcc.Graph(id="rqa-grid", config={"displaylogo": False})],
                       label="Auto- / cross- / joint-recurrence, 2×2"),
        ]),

        _section_hdr("RQA Quantification"),
        html.Div(id="rqa-metrics-desc", className="section-desc"),
        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div([
                html.Div(_plot_card([dcc.Graph(id="rqa-bars", config={"displayModeBar": False})],
                                     label="RR / DET / LAM by matrix"),
                         style={"flex": "1", "minWidth": "360px"}),
                html.Div(_plot_card([html.Div(id="rqa-table", style={"padding": "10px 6px"})],
                                     label="Full RQA measure table"),
                         style={"flex": "1", "minWidth": "360px"}),
            ], style={"display": "flex", "gap": "16px", "alignItems": "stretch", "flexWrap": "wrap"}),
            _plot_card([dcc.Graph(id="rqa-hist", config={"displayModeBar": False})],
                       label="Diagonal line length distribution · cross-recurrence"),
        ]),

        RQA_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  — register with your app via register_rqa_callbacks(app)
# ─────────────────────────────────────────────────────────────────────────────
def register_rqa_callbacks(app):

    @app.callback(
        Output("rqa-m", "value"),
        Output("rqa-tau", "value"),
        Input("rqa-var-a", "value"),
    )
    def _default_embedding(colA):
        """Snap (m, τ) to Series A's identification when the primary variable
        changes; falls back to a generic default while its identification is
        still running in the background."""
        ident = get_ident(colA or DEFAULT_COL_A)
        return (ident["m"], ident["tau"]) if ident else (3, 6)

    @app.callback(
        Output("rqa-banner", "children"),
        Output("rqa-phase", "figure"),
        Output("rqa-grid", "figure"),
        Output("rqa-grid-desc", "children"),
        Output("rqa-bars", "figure"),
        Output("rqa-table", "children"),
        Output("rqa-metrics-desc", "children"),
        Output("rqa-hist", "figure"),
        Output("rqa-id-card", "children"),
        Output("rqa-poll", "disabled"),
        Input("rqa-var-a", "value"),
        Input("rqa-var-b", "value"),
        Input("rqa-m", "value"),
        Input("rqa-tau", "value"),
        Input("rqa-rr", "value"),
        Input("rqa-win-len", "value"),
        Input("rqa-win-start", "value"),
        Input("rqa-poll", "n_intervals"),
    )
    def update_rqa(colA, colB, m, tau, target_rr, win_len, win_start, _tick):
        colA = colA or DEFAULT_COL_A
        colB = colB or DEFAULT_COL_B
        if colB == colA:
            colB = next((c for c in AVAILABLE if c != colA), colA)

        m         = int(m or 3)
        tau       = int(tau or 6)
        target_rr = float(target_rr or RR_DEFAULT)
        win_len   = int(win_len or WINDOW_DEFAULT)
        win_start = int(win_start or 0)
        win_start = max(0, min(win_start, N_OBS - win_len))
        win_end   = win_start + win_len

        raw_a = DF[colA].values[win_start:win_end]
        raw_b = DF[colB].values[win_start:win_end]
        y_a, y_b = _clean_window(raw_a), _clean_window(raw_b)
        idx_window = DF.index[win_start:win_end]

        rec = build_recurrence(y_a, y_b, m, tau, target_rr)
        st = precompute_status()

        if rec.get("L", 0) < 10:
            ph = _placeholder_fig("Window too short for this embedding — increase the "
                                   "window length or lower m / τ.")
            msg = ("Embedded trajectory has too few points (window length minus "
                   "(m−1)·τ). Widen the window, or reduce the embedding dimension / delay.")
            return (_banner(("—", "Window too short", RED)), ph, ph, msg, ph, None, msg,
                    ph, None, st["finished"])

        theiler = max(1, tau)
        mA = rqa_measures(rec["R_a"], theiler=theiler)
        mB = rqa_measures(rec["R_b"], theiler=theiler)
        mC = rqa_measures(rec["R_cross"], theiler=0)
        mJ = rqa_measures(rec["R_joint"], theiler=theiler)
        verdict, verdict_color = _coupling_verdict(mA, mB, mC)

        date_lo = idx_window[0].strftime("%Y-%m-%d %H:%M") if len(idx_window) else "—"
        date_hi = idx_window[-1].strftime("%Y-%m-%d %H:%M") if len(idx_window) else "—"

        banner = _banner(
            (f"{mC['rr']:.2f}%", "Cross Recurrence Rate", ACCENT2),
            (f"{mC['det']:.2f}%", "Cross Determinism", ACCENT2),
            (f"{mC['lam']:.2f}%", "Cross Laminarity", ACCENT2),
            (verdict, "Coupling verdict", verdict_color,
             {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
            (f"m={m}, τ={tau}h", "Shared embedding", AMBER),
            (f"{rec['L']:,}", "Embedded points", MUTED2),
        )

        grid_desc = (
            f"Window {date_lo} → {date_hi} ({win_len} h, {rec['L']:,} embedded points after "
            f"delay-embedding loses (m−1)·τ = {(m - 1) * tau} points). Each panel targets a "
            f"{target_rr:.0f}% recurrence rate via an adaptively chosen ε; diagonal structure "
            f"close to the main diagonal in the cross-recurrence panel (bottom-left) is the "
            f"visual signature of coupling between {LABELS.get(colA, colA)} and "
            f"{LABELS.get(colB, colB)}."
        )
        metrics_desc = (
            f"RQA measures for all four matrices, with a Theiler window of {theiler} "
            f"samples excluding trivially auto-correlated near-diagonal points from the two "
            f"auto-recurrence matrices and the joint-recurrence matrix (the cross-recurrence "
            "matrix has no line-of-identity to exclude). Compare Cross against the two Auto "
            "columns: comparable DET/LAM indicates the coupling is as structured as either "
            "system's own dynamics, not a diffuse, noise-level effect."
        )

        identA, identB = get_ident(colA), get_ident(colB)
        if identA and identB:
            id_card = identification_card(identA, identB, colA, colB)
        else:
            id_card = _theory_card(
                "Identifying τ and m for THIS pair", "IDENTIFICATION", GREEN,
                [f"Background identification in progress "
                 f"({st['done']}/{st['total']} variables done) — this panel fills in "
                 "automatically once both series are ready; the sliders above already work "
                 "with generic defaults in the meantime."],
            )

        return (
            banner,
            fig_phase_space(y_a, y_b, m, tau, colA, colB),
            fig_recurrence_grid(rec, colA, colB),
            grid_desc,
            fig_rqa_bars(mA, mB, mC, mJ, colA, colB),
            rqa_metric_table(mA, mB, mC, mJ, colA, colB),
            metrics_desc,
            fig_diagonal_hist(mC["diag_lengths"]),
            id_card,
            st["finished"],
        )