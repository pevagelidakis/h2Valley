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

from functools import lru_cache
import math
import threading
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from scipy.spatial import cKDTree

from cache_utils import make_signature, load_cache, save_cache
from core.dataset import get_dataset
import rqa_funs
from rqa_funs import (embed, distance_matrix, epsilon_for_rr, recurrence_matrix,
                      dominant_period, sliding_window_rqa, RQA_METRICS)

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

DS = get_dataset()
CSV_PATH = DS.path
UNIT_1, UNIT_N, UNIT = DS.unit   # one sample: e.g. hour / hours / h
_SAMPLES = "samples" if UNIT_N == "samples" else f"samples / {UNIT_N}"

# ─────────────────────────────────────────────────────────────────────────────
#  VARIABLE CATALOGUE  —  from the dataset config (configs/*.yaml via
#  core.dataset); edit the config, not this file, to change it.
# ─────────────────────────────────────────────────────────────────────────────
COLS   = DS.columns()
LABELS = DS.labels
COLORS = DS.colors

DEFAULT_COL_A, DEFAULT_COL_B = DS.default_pair()

# ─────────────────────────────────────────────────────────────────────────────
#  RQA CONFIG
# ─────────────────────────────────────────────────────────────────────────────
AMI_MAX_LAG        = DS.lag(1.25)   # lags scanned for the average-mutual-information delay estimate
AMI_BINS           = 16     # histogram bins for the AMI estimator
FNN_MAX_DIM        = 10     # highest embedding dimension tried by false-nearest-neighbours
FNN_RTOL           = 15.0   # Kennel et al. (1992) criterion-1 threshold
FNN_ATOL           = 2.0    # Hegger/Kantz criterion-2 threshold (relative to attractor size)
FNN_THRESHOLD_PCT  = 10.0   # stop increasing m once %FNN falls at/below this

TAU_MIN, TAU_MAX   = 1, DS.lag(1)   # up to one dominant cycle
M_MIN, M_MAX        = 1, 8
WINDOW_MIN, WINDOW_MAX, WINDOW_DEFAULT = 150, 600, 300
RR_MIN, RR_MAX, RR_DEFAULT = 2, 25, 10       # target recurrence rate, percent
L_MIN, V_MIN        = 2, 2                    # minimum diagonal/vertical line length for DET/LAM
MAX_PLOT_POINTS     = 35_000                  # per-panel recurrence-point cap (keeps WebGL responsive)
METRIC_DEFAULT      = "l2"
METRIC_OPTIONS      = [{"label": "ℓ1 (Manhattan)", "value": "l1"},
                       {"label": "ℓ2 (Euclidean)", "value": "l2"},
                       {"label": "ℓ∞ (Chebyshev)", "value": "linf"}]
SW_PERIOD_MIN_DEFAULT = DS.period             # Lomb-Scargle search band (samples) for the sliding window
SW_STEP_DEFAULT       = 25                    # sliding-window step, % of the window length
SW_MIN_EMBEDDED       = 30                    # min embedded points per window (window is raised to meet it)
PCT_KEYS              = ("rr", "det", "lam")  # measures displayed in %
TAU_RR_FRACTION       = 2                     # RR_τ lags run up to (embedded points) / this

# ─────────────────────────────────────────────────────────────────────────────
#  DATA  (shared loader: core.dataset)
# ─────────────────────────────────────────────────────────────────────────────
def _load_df() -> pd.DataFrame:
    return DS.frame()          # time-indexed, built once per process


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
SW_PERIOD_MAX = max(SW_PERIOD_MIN_DEFAULT + 1, N_OBS // 2)
SW_PERIOD_MAX_DEFAULT = max(SW_PERIOD_MIN_DEFAULT + 1, N_OBS // 4)


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
#  controls). All recurrence / RQA maths comes from rqa_funs.py.
# ─────────────────────────────────────────────────────────────────────────────
def _clean_window(arr: np.ndarray) -> np.ndarray:
    """Fill any gaps in a contiguous slice via linear interpolation (plus
    edge fill), so an occasional missing reading never breaks the embedding
    or misaligns the two series in time."""
    s = pd.Series(arr, dtype=float)
    if s.isna().any():
        s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.values


def _zscore(y: np.ndarray) -> np.ndarray:
    return (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)


def build_recurrence(y_a: np.ndarray, y_b: np.ndarray, m: int, tau: int, target_rr: float,
                     metric: str = METRIC_DEFAULT) -> dict:
    """Standardise both windows, embed both with the *same* (m, τ) — required
    for cross- and joint-recurrence to be meaningful — and threshold each
    pairwise-distance matrix at a fixed recurrence rate (Marwan et al., 2007)."""
    Ea, Eb = embed(_zscore(y_a), m, tau), embed(_zscore(y_b), m, tau)
    L = min(len(Ea), len(Eb))
    Ea, Eb = Ea[:L], Eb[:L]

    if L < 10:
        return dict(L=0)

    Da = distance_matrix(Ea, metric=metric)
    Db = distance_matrix(Eb, metric=metric)
    Dc = distance_matrix(Ea, Eb, metric=metric)
    eps_a = epsilon_for_rr(Da, target_rr / 100.0, exclude_diagonal=True)
    eps_b = epsilon_for_rr(Db, target_rr / 100.0, exclude_diagonal=True)
    eps_c = epsilon_for_rr(Dc, target_rr / 100.0, exclude_diagonal=False)

    R_a = recurrence_matrix(Da, eps_a)
    R_b = recurrence_matrix(Db, eps_b)
    R_c = recurrence_matrix(Dc, eps_c)
    R_j = R_a & R_b     # joint recurrence: simultaneous auto-recurrence in both systems

    return dict(R_a=R_a, R_b=R_b, R_cross=R_c, R_joint=R_j, D_a=Da, D_b=Db, D_cross=Dc,
                eps_a=eps_a, eps_b=eps_b, eps_cross=eps_c, L=L)


def rqa_measures(R: np.ndarray, theiler: int = 0, l_min: int = L_MIN, v_min: int = V_MIN) -> dict:
    """rqa_funs.rqa_measures with RR / DET / LAM expressed in %. `theiler`
    zeroes a band around the main diagonal before measuring, so short lines
    caused by the trajectory's own smoothness never inflate DET/LAM."""
    out = rqa_funs.rqa_measures(R, lmin=l_min, vmin=v_min, theiler=theiler)
    for k in PCT_KEYS:
        out[k] *= 100.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SLIDING-WINDOW RQA  —  window length = Lomb-Scargle period of Series A
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def lomb_scargle_period(col: str, p_min: int, p_max: int):
    y = _clean_window(DF[col].values)
    period, periods, power = dominant_period(y, min_period=p_min, max_period=p_max)
    return period, periods, power


def sliding_window_length(period: float, m: int, tau: int) -> int:
    """Window = Lomb-Scargle period, raised if needed so each window still
    holds SW_MIN_EMBEDDED delay vectors, and capped at the series length."""
    w = max(int(round(period)), (m - 1) * tau + SW_MIN_EMBEDDED)
    return min(w, N_OBS)


@lru_cache(maxsize=16)
def sliding_rqa(colA: str, colB: str, m: int, tau: int, target_rr: float, metric: str,
                window: int, step: int) -> dict:
    """Cached rqa_funs.sliding_window_rqa over the full record of both series:
    ε is calibrated once on the whole trajectory to the target recurrence rate
    and then held fixed, so RR (and every other measure) can drift between
    windows. Keys renamed to match the static grid: A, B, cross, joint."""
    out = sliding_window_rqa(
        _clean_window(DF[colA].values), _clean_window(DF[colB].values),
        window=window, step=step, m=m, tau=tau, metric=metric,
        target_rr=target_rr / 100.0, lmin=L_MIN, vmin=V_MIN, theiler=max(1, tau),
    )
    renamed = {"A": out["auto_x"], "B": out["auto_y"], "cross": out["cross"], "joint": out["joint"]}
    for df in renamed.values():
        for k in PCT_KEYS:
            df[k] = df[k] * 100.0
    return renamed


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


def _placeholder_fig(msg: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=12, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**_base_layout(height=height, showlegend=False))
    return fig


def fig_unthresholded_rp(rec: dict, colA: str, colB: str, metric: str,
                         max_side: int = 400) -> go.Figure:
    """Unthresholded recurrence plots: the raw distance matrices D_ij = ‖Xᵢ − Xⱼ‖
    before the Heaviside step, on one shared colour scale (both series are
    z-scored, so distances are comparable). Matrices larger than `max_side`
    are strided for display only; ε for each matrix is given in its title."""
    la, lb = LABELS.get(colA, colA), LABELS.get(colB, colB)
    panels = [(rec["D_a"], f"Auto · {la}<br>ε = {rec['eps_a']:.3f}"),
              (rec["D_b"], f"Auto · {lb}<br>ε = {rec['eps_b']:.3f}"),
              (rec["D_cross"], f"Cross · {la} ↔ {lb}<br>ε = {rec['eps_cross']:.3f}")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for _, t in panels],
                        horizontal_spacing=0.06)
    L = rec["L"]
    s = max(1, math.ceil(L / max_side))
    idx = np.arange(0, L, s)
    for c, (D, _) in enumerate(panels, start=1):
        fig.add_trace(go.Heatmap(
            z=D[::s, ::s], x=idx, y=idx, coloraxis="coloraxis",
            hovertemplate="i=%{y}, j=%{x}<br>‖Xᵢ − Xⱼ‖ = %{z:.3f}<extra></extra>",
        ), row=1, col=c)
        fig.update_xaxes(row=1, col=c, range=[0, L], **_AX)
        fig.update_yaxes(row=1, col=c, range=[0, L], scaleanchor=f"x{c if c > 1 else ''}", **_AX)
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    fig.update_layout(**_base_layout(height=430, showlegend=False, margin=dict(t=64, b=40, l=56, r=16)),
                      coloraxis=dict(colorscale="Viridis",
                                     colorbar=dict(title=dict(text=f"{metric} distance", side="right"),
                                                   tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
                                                   thickness=12)))
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


def _series_style(colA: str, colB: str):
    """Names / colours of the four matrices, shared by every multi-matrix chart."""
    names = [f"Auto · {LABELS.get(colA, colA)}", f"Auto · {LABELS.get(colB, colB)}", "Cross", "Joint"]
    palette = [COLORS.get(colA, ACCENT2), COLORS.get(colB, ACCENT), ACCENT2, AMBER]
    return names, palette


def fig_line_histograms(mA, mB, mC, mJ, colA, colB) -> go.Figure:
    """Normalised diagonal P(ℓ), ℓ ≥ ℓmin, and vertical P(v), v ≥ vmin, line-length
    distributions of all four matrices on a log-probability axis: an exponential
    tail P(ℓ) ∝ exp(−K₂·ℓ) appears as a straight line whose slope tracks K₂."""
    names, palette = _series_style(colA, colB)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, subplot_titles=(
        f"Diagonal lines P(ℓ), ℓ ≥ {L_MIN}", f"Vertical lines P(v), v ≥ {V_MIN}"))
    any_data = False
    for name, color, m in zip(names, palette, (mA, mB, mC, mJ)):
        for c, (key, min_len) in enumerate((("diag_lengths", L_MIN), ("vert_lengths", V_MIN)), start=1):
            x, p = rqa_funs.line_distribution(m[key], min_len)
            any_data |= bool(x.size)
            fig.add_trace(go.Scatter(
                x=x, y=p, mode="lines+markers", name=name, legendgroup=name, showlegend=(c == 1),
                line=dict(color=color, width=1.4), marker=dict(size=4),
                hovertemplate=("ℓ" if c == 1 else "v") + " = %{x}<br>P = %{y:.4f}<extra>" + name + "</extra>",
            ), row=1, col=c)
    if not any_data:
        return _placeholder_fig("No diagonal or vertical lines at this threshold / window.", height=300)
    fig.update_xaxes(row=1, col=1, **_AX, title_text="Diagonal line length ℓ")
    fig.update_xaxes(row=1, col=2, **_AX, title_text="Vertical line length v")
    fig.update_yaxes(**_AX, type="log", title_text="Probability")
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    fig.update_layout(**_base_layout(height=340, showlegend=True,
                                     legend=dict(orientation="h", y=1.18, x=0),
                                     margin=dict(t=70, b=40, l=56, r=16)))
    return fig


def _recurrence_period(taus: np.ndarray, rr: np.ndarray):
    """Lag of the highest RR_τ peak after RR_τ first stops decreasing — the
    dominant recurrence period (None if RR_τ never turns back up)."""
    pos = taus > 0
    taus, rr = taus[pos], rr[pos]
    if rr.size < 3:
        return None
    rising = np.nonzero(np.diff(rr) > 0)[0]
    if not rising.size:
        return None
    start = rising[0]
    return int(taus[start + int(np.argmax(rr[start:]))])


def fig_tau_rr(rec: dict, colA: str, colB: str, tau_max: int) -> go.Figure:
    """τ-recurrence rate RR_τ for the auto and joint RPs (τ = 1..τmax) and the
    cross RP (τ = −τmax..τmax: a peak at τ > 0 means B revisits A's states τ
    samples later). Dominant recurrence periods of A and B are marked."""
    names, palette = _series_style(colA, colB)
    fig = go.Figure()
    for name, color, key, two_sided in zip(names, palette, ("R_a", "R_b", "R_cross", "R_joint"),
                                           (False, False, True, False)):
        taus, rr = rqa_funs.tau_recurrence_rate(rec[key], tau_max, two_sided=two_sided)
        fig.add_trace(go.Scatter(
            x=taus, y=100.0 * rr, mode="lines", name=name,
            line=dict(color=color, width=1.6, dash="dot" if key == "R_cross" else None),
            hovertemplate="lag %{x} h<br>RR = %{y:.2f}%<extra>" + name + "</extra>",
        ))
        if key in ("R_a", "R_b"):
            T = _recurrence_period(taus, rr)
            if T is not None:
                fig.add_vline(x=T, line=dict(color=color, width=1.2, dash="dash"),
                              annotation_text=f"T ≈ {T} {UNIT}", annotation_position="top",
                              annotation_font=dict(size=10, color=color, family=FONT_MONO))
    fig.add_vline(x=0, line=dict(color=ZERO_L, width=1))
    fig.update_xaxes(**_AX, title_text=f"Lag τ ({_SAMPLES}) — not the embedding delay")
    fig.update_yaxes(**_AX, title_text="RR_τ (%)", rangemode="tozero")
    fig.update_layout(**_base_layout(height=340, showlegend=True,
                                     legend=dict(orientation="h", y=1.14, x=0),
                                     margin=dict(t=60, b=40, l=56, r=16)))
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
    fig.update_xaxes(row=1, col=1, **_AX, title_text=f"Lag τ ({UNIT_N})")
    fig.update_yaxes(row=1, col=1, **_AX, title_text="AMI (nats)")
    fig.update_xaxes(row=1, col=2, **_AX, title_text="Embedding dimension m")
    fig.update_yaxes(row=1, col=2, **_AX, title_text="False neighbours (%)")
    fig.update_layout(**_base_layout(height=260, showlegend=False))
    return fig


def fig_periodogram(lsA, lsB, colA, colB, window: int) -> go.Figure:
    """Lomb-Scargle periodograms of both series over the search band, each
    with its dominant period marked; Series A's sets the window length."""
    fig = go.Figure()
    for (period, periods, power), col, color, dash in (
            (lsA, colA, COLORS.get(colA, ACCENT2), "dash"),
            (lsB, colB, COLORS.get(colB, ACCENT), "dot")):
        la = LABELS.get(col, col)
        fig.add_trace(go.Scatter(x=periods, y=power, mode="lines", name=la,
                                  line=dict(color=color, width=1.4),
                                  hovertemplate="period %{x:.1f} h<br>power %{y:.3f}<extra>" + la + "</extra>"))
        fig.add_vline(x=period, line=dict(color=color, width=1.4, dash=dash))
    fig.add_annotation(x=math.log10(lsA[0]), y=1, xref="x", yref="paper", yanchor="bottom",
                       text=f"window = {window} {UNIT}", showarrow=False,
                       font=dict(size=10, color=AMBER, family=FONT_MONO))
    fig.update_xaxes(**_AX, type="log", title_text=f"Period ({_SAMPLES})")
    fig.update_yaxes(**_AX, title_text="Normalised LS power")
    fig.update_layout(**_base_layout(height=280, showlegend=True,
                                     legend=dict(orientation="h", y=1.12, x=0)))
    return fig


_SW_UNITS = {"rr": " (%)", "det": " (%)", "lam": " (%)"}


def fig_sliding_rqa(sw: dict, colA: str, colB: str, win_start: int, win_end: int) -> go.Figure:
    """The RQA measures per sliding window (x = window centre), one panel
    each, with the same four series / palette as the RR-DET-LAM bar chart;
    the window currently inspected above is shaded."""
    names = {"A": f"Auto · {LABELS.get(colA, colA)}", "B": f"Auto · {LABELS.get(colB, colB)}",
             "cross": "Cross", "joint": "Joint"}
    palette = {"A": COLORS.get(colA, ACCENT2), "B": COLORS.get(colB, ACCENT),
               "cross": ACCENT2, "joint": AMBER}
    n_cols = 2
    n_rows = math.ceil(len(RQA_METRICS) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True,
                        subplot_titles=[lbl + _SW_UNITS.get(k, "") for k, lbl in RQA_METRICS],
                        horizontal_spacing=0.07, vertical_spacing=0.05)
    for kind, df in sw.items():
        x = DF.index[df["center"].values]
        for i, (key, lbl) in enumerate(RQA_METRICS):
            r, c = i // n_cols + 1, i % n_cols + 1
            fig.add_trace(go.Scatter(
                x=x, y=df[key], mode="lines+markers", name=names[kind],
                legendgroup=kind, showlegend=(i == 0),
                line=dict(color=palette[kind], width=1.5), marker=dict(size=4),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + lbl + " %{y:.3f}<extra>" + names[kind] + "</extra>",
            ), row=r, col=c)
    lo, hi = DF.index[win_start], DF.index[min(win_end, N_OBS) - 1]
    for i in range(len(RQA_METRICS)):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=AMBER, opacity=0.10, line_width=0,
                      row=i // n_cols + 1, col=i % n_cols + 1)
    fig.update_xaxes(**_AX)
    fig.update_yaxes(**_AX)
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    fig.update_layout(**_base_layout(height=230 * n_rows, showlegend=True,
                                     legend=dict(orientation="h", y=1.03, x=0),
                                     margin=dict(t=70, b=40, l=56, r=16)))
    return fig


_RQA_ROWS = [
    ("rr",   r"Recurrence Rate $RR$ (%)",                  False),
    ("det",  r"Determinism $DET$ (%)",                     False),
    ("ratio",r"Ratio $DET/RR$",                            False),
    ("lam",  r"Laminarity $LAM$ (%)",                      False),
    ("L",    r"Mean diagonal length $L$",                  False),
    ("Lmax", r"Longest diagonal line $L_{\max}$",           True),
    ("div",  r"Divergence $DIV = 1/L_{\max}$",              False),
    ("entr", r"Diagonal entropy $ENTR$ (nats)",             False),
    ("tt",   r"Trapping Time $TT$",                         False),
    ("vmax", r"Longest vertical line $V_{\max}$",           True),
    ("rte",  r"Recurrence time entropy $RTE$",              False),
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
        rows.append(html.Tr([html.Td(_md(name, inline=True)), html.Td(vals[0]), html.Td(vals[1]),
                              html.Td(vals[2], style={"color": ACCENT2}),
                              html.Td(vals[3], style={"color": AMBER})]))
    return html.Table(rows, style={
        "width": "100%", "borderCollapse": "collapse", "fontFamily": FONT_MONO,
        "fontSize": "12px", "color": TEXT,
    }, className="s2-moment-table")


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL LAYOUT HELPERS  (mirrors app.py's, reuses the same global CSS classes)
# ─────────────────────────────────────────────────────────────────────────────
def _md(text, inline=False, className=""):
    """Markdown with MathJax: $…$ inline and $$…$$ display LaTeX. `inline`
    keeps the rendered paragraph on the same line (list items, labels)."""
    cls = ("md-inline " if inline else "md-block ") + className
    return dcc.Markdown(text, mathjax=True, className=cls.strip())

def _section_hdr(text):
    return html.Div(text, className="section-hdr")

def _section_desc(text):
    return _md(text, className="section-desc")

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
        children.append(_md(p) if isinstance(p, str) else html.Div(p, style={"marginBottom": "8px"}))
    if math_eq:
        children.append(_md(f"$${math_eq}$$", className="math"))
    return html.Div(children, className="theory-card")

def _callout(kind, title, intro, items, ordered=True):
    icon = "📋" if kind == "tip" else "⚠"
    box_cls = "callout callout-tip" if kind == "tip" else "callout callout-note"
    head = html.Div([html.Span(icon, className="callout-icon"), html.Span(title, className="callout-title")],
                    className="callout-head")
    body_children = []
    if intro:
        body_children.append(_md(intro, className="callout-intro"))
    lis = []
    for it in items:
        if isinstance(it, (tuple, list)):
            lead, rest = it
            lis.append(html.Li([html.B(lead, className="callout-lead"), _md(rest, inline=True)],
                               className="callout-li"))
        else:
            lis.append(html.Li(_md(it, inline=True), className="callout-li"))
    list_tag = html.Ol if ordered else html.Ul
    body_children.append(list_tag(lis, className="callout-list"))
    return html.Div([head, html.Div(body_children, className="callout-body")], className=box_cls)


def identification_card(identA: dict, identB: dict, colA: str, colB: str):
    la, lb = LABELS.get(colA, colA), LABELS.get(colB, colB)
    return html.Div([
        _theory_card(
            "Identifying τ and m for THIS pair", "IDENTIFICATION", GREEN,
            [
                rf"Cross- and joint-recurrence require one shared $(m, \tau)$ applied to both "
                rf"series, so the sliders default to {la}'s own identification: a delay "
                rf"$\tau = {identA['tau']}$ h from the first minimum of its average mutual "
                rf"information, and an embedding dimension $m = {identA['m']}$ from the "
                rf"false-nearest-neighbours curve dropping at/below "
                rf"${FNN_THRESHOLD_PCT:.0f}\%$.",
                rf"{lb}, identified independently, would itself suggest $\tau = "
                rf"{identB['tau']}$ h and $m = {identB['m']}$. When the two disagree, the "
                "shared-embedding compromise trades a small amount of optimality for "
                "each individual series in exchange for a cross-recurrence plot that "
                "is mathematically well-defined; both sliders remain fully adjustable "
                "to explore either side of that trade-off.",
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

    _md(
        "The Class-1 and Class-2 tabs each build a *model* of one series and "
        "validate it against that same series. This tab does something different: it makes "
        "no distributional or shape assumption at all, and it always looks at **two** "
        "series together. By Takens' delay-embedding theorem, a single scalar sequence "
        "already contains — under mild conditions — enough information to reconstruct the "
        "geometry of the underlying dynamical system that produced it. Reconstructing that "
        "geometry for two variables and asking where their reconstructed trajectories revisit "
        "the same neighbourhoods (cross-recurrence) or revisit their own neighbourhoods at "
        "the same instants (joint recurrence) is a model-free way to detect coupling that "
        "linear tools — ACF, Granger causality, cointegration — can miss when the "
        "relationship is nonlinear.",
        className="section-desc theory-intro",
    ),

    html.Div([
        html.Div([
            _theory_card(
                "Phase-space reconstruction — Takens' theorem", "STEP ①", ACCENT2,
                [
                    r"A scalar series $x_1, \dots, x_n$ is lifted into an $m$-dimensional "
                    r"trajectory by delay embedding: each point is a window of $m$ samples spaced "
                    r"$\tau$ apart, $\mathbf{X}_i = (x_i, x_{i+\tau}, \dots, x_{i+(m-1)\tau})$. "
                    r"Takens (1981) showed that for a large enough embedding dimension $m$, this "
                    r"reconstructed trajectory is a smooth, one-to-one image of the system's true "
                    r"(unobservable) state space — so distances between reconstructed points "
                    r"faithfully reflect distances between the real dynamical states that "
                    r"produced them.",
                    r"The delay $\tau$ is chosen from the first local minimum of the average "
                    r"mutual information between $x_t$ and $x_{t+\tau}$ (Fraser & Swinney, 1986): "
                    r"too small and consecutive coordinates are redundant; too large and they "
                    r"become causally unrelated. The dimension $m$ is chosen by false nearest "
                    r"neighbours (Kennel et al., 1992): points that look close together at "
                    r"dimension $d$ but fly apart at $d+1$ are neighbours only because the "
                    r"projection was too thin to unfold the attractor; $m$ is the smallest "
                    r"dimension at which that stops happening.",
                ],
                math_eq=r"\mathbf{X}_i = \left(x_i,\; x_{i+\tau},\; \dots,\; x_{i+(m-1)\tau}\right) \in \mathbb{R}^m",
            ),
            _theory_card(
                "Auto-, cross- and joint-recurrence plots", "STEP ②", GREEN,
                [
                    r"A recurrence plot marks every pair of times $(i, j)$ whose reconstructed "
                    r"states lie within a threshold $\varepsilon$ of each other, "
                    r"$R_{ij} = \Theta(\varepsilon - \lVert \mathbf{X}_i - \mathbf{X}_j \rVert)$, "
                    r"where $\Theta$ is the Heaviside step. Applied to one series against itself "
                    r"it is an *auto-recurrence plot* — diagonal structure away from the main "
                    r"diagonal reveals deterministic, near-periodic dynamics; scattered, "
                    r"texture-free points look like noise. Before the step, the raw distance "
                    r"matrix $D_{ij} = \lVert \mathbf{X}_i - \mathbf{X}_j \rVert$ is the "
                    r"*unthresholded* recurrence plot.",
                    r"Applied between the reconstructed trajectories of two *different* series "
                    r"(using one shared $(m, \tau)$ so distances are comparable) it is a "
                    r"*cross-recurrence plot*: diagonal structure close to the main diagonal "
                    r"means the two systems revisit corresponding states at corresponding "
                    r"times — direct evidence of coupling. This tab's threshold $\varepsilon$ is "
                    r"set adaptively (fixed recurrence rate) so every plot targets the same "
                    r"overall point density regardless of each variable's physical units.",
                    r"The Hadamard product of the two auto-recurrence matrices — the "
                    r"*joint-recurrence plot* (Romano et al., 2004) — marks times when both "
                    r"systems simultaneously revisit one of their own past states; unlike the "
                    r"cross-recurrence plot it does not require the two systems' state spaces to "
                    r"be directly comparable, only that they recur at the same clock time.",
                ],
                math_eq=r"R^{\mathrm{cross}}_{ij} = \Theta\!\left(\varepsilon - \lVert \mathbf{X}^{A}_i - \mathbf{X}^{B}_j \rVert\right), \qquad "
                        r"R^{\mathrm{joint}}_{ij} = R^{A}_{ij} \cdot R^{B}_{ij}",
            ),
            _theory_card(
                "RQA measures — quantifying the texture", "STEP ③", AMBER,
                [
                    r"Recurrence Rate ($RR$) is simply the density of marked points — how "
                    r"often either system (or both) is 'close' at all. Determinism ($DET$) is "
                    r"the fraction of those points that align into diagonal lines of length "
                    r"$\ell \geq \ell_{\min} = 2$ rather than sitting in isolation — diagonal "
                    r"structure is the signature of deterministic, forecastable trajectory "
                    r"segments, so $DET$ is a texture-based, model-free analogue of "
                    r"predictability. Laminarity ($LAM$) is the equivalent count for vertical "
                    r"lines of length $v \geq v_{\min}$, and flags laminar (slowly-changing, "
                    r"'stuck') states rather than periodic ones.",
                    r"The longest diagonal line $L_{\max}$ bounds how long two trajectory "
                    r"segments stay close before diverging — its reciprocal, Divergence "
                    r"($DIV = 1/L_{\max}$), is a coarse, computation-free proxy for the largest "
                    r"Lyapunov exponent referenced in the Causality tab's theory: short "
                    r"$L_{\max}$ (high $DIV$) means fast divergence, i.e. more chaotic-looking "
                    r"dynamics. Entropy ($ENTR$) is the Shannon entropy of the "
                    r"diagonal-line-length distribution $p(\ell)$ — low when most lines share "
                    r"one length (a clean periodicity), high when line lengths are richly "
                    r"varied. Trapping Time ($TT$), the mean vertical line length, is the "
                    r"laminar-phase counterpart of the same idea.",
                    r"The full diagonal histogram $P(\ell)$, $\ell \geq \ell_{\min}$, carries more than "
                    r"its mean $\langle L \rangle$: its tail decays as $P(\ell) \propto e^{-K_2 \ell}$, so "
                    r"the slope of $\ln P(\ell)$ estimates the correlation entropy $K_2$, a lower bound on "
                    r"the sum of the positive Lyapunov exponents. The vertical histogram $P(v)$ does the "
                    r"same for laminar phases. The *white* vertical lines — gaps between successive "
                    r"recurrences of one state — are recurrence times $T$; their normalised Shannon "
                    r"entropy is the Recurrence Time Entropy ($RTE$, Little et al., 2007), near 0 for a "
                    r"strictly periodic signal and near 1 for white noise.",
                    r"Finally, the $\tau$-recurrence rate $RR_\tau$ is the density of the diagonal at "
                    r"lag $\tau$ — the probability that the system is back near the state it occupied "
                    r"$\tau$ samples earlier. It is the recurrence analogue of the autocorrelation "
                    r"function: peaks at $\tau = 24, 168, \dots$ reveal daily or weekly cycles, but "
                    r"unlike the ACF it responds to nonlinear as well as linear periodicity.",
                ],
                math_eq=r"RR = \frac{1}{N^2}\sum_{i,j=1}^{N} R_{ij}, \qquad "
                        r"DET = \frac{\sum_{\ell \geq \ell_{\min}} \ell\, P(\ell)}{\sum_{\ell \geq 1} \ell\, P(\ell)}, \qquad "
                        r"LAM = \frac{\sum_{v \geq v_{\min}} v\, P(v)}{\sum_{v \geq 1} v\, P(v)}"
                        r"$$ $$"
                        r"ENTR = -\sum_{\ell \geq \ell_{\min}} p(\ell) \ln p(\ell), \qquad "
                        r"TT = \frac{\sum_{v \geq v_{\min}} v\, P(v)}{\sum_{v \geq v_{\min}} P(v)}, \qquad "
                        r"DIV = \frac{1}{L_{\max}}"
                        r"$$ $$"
                        r"\langle L \rangle = \frac{\sum_{\ell \geq \ell_{\min}} \ell\, P(\ell)}{\sum_{\ell \geq \ell_{\min}} P(\ell)}, \qquad "
                        r"RTE = -\frac{1}{\ln T_{\max}} \sum_{T=1}^{T_{\max}} p(T) \ln p(T), \qquad "
                        r"RR_\tau = \frac{1}{N - \tau} \sum_{i=1}^{N-\tau} R_{i,\,i+\tau}",
            ),
        ], className="theory-row"),
        html.Div(id="rqa-id-card", className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px",
                                        "overflow": "hidden", "background": CARD2}),

    _callout(
        "tip", "How to read this tab — a walkthrough for any level",
        "Work top to bottom; each element answers one question in the Class-3 pipeline:",
        [
            ("Series A / Series B → ", r'"Which two variables?" — Series A also drives the '
             r"default embedding parameters, since only one $(m, \tau)$ pair can be shared."),
            ("Embedding dimension / delay sliders → ", r'"How is each scalar series lifted '
             "into a trajectory?\" — pre-filled from Series A's false-nearest-neighbours / "
             r"mutual-information identification, freely overridable."),
            ("Target recurrence rate → ", "\"How strict is 'close'?\" — sets " r"$\varepsilon$ "
             r"adaptively so every one of the four recurrence plots below targets the same "
             r"point density, making their textures directly comparable."),
            ("Unthresholded recurrence plot → ", r'"How far apart are all pairs of states, '
             r'before any $\varepsilon$ is chosen?" — dark (near) bands parallel to the main '
             r"diagonal show periodic returns, broad blocks show slowly-changing regimes; the "
             r"plot below it is this same matrix after the Heaviside step "
             r"$\Theta(\varepsilon - \lVert \mathbf{X}_i - \mathbf{X}_j \rVert)$."),
            ("Recurrence plot grid → ", r'"Where do the two systems recur, alone and '
             r'together?" — diagonal texture near the main diagonal in the cross-recurrence '
             r"panel is the visual signature of coupling; compare its density and structure "
             r"against the two auto-recurrence panels."),
            ("RQA measures (bars + table) → ", r'"How much of that texture is there, '
             r'exactly?" — $DET$ / $LAM$ / $ENTR$ on the cross plot at a level comparable to '
             r"the two auto plots is quantitative evidence of shared dynamical structure, not "
             r"just a visual impression."),
        ],
    ),
    _callout(
        "note", "Limits of this analysis",
        "Recurrence quantification is powerful precisely because it makes so few "
        "assumptions — which also means it needs care in interpretation:",
        [
            r"$\varepsilon$ and the target recurrence rate are analyst choices: comparisons "
            r"across variables or window lengths are only fair when the recurrence rate is held "
            r"fixed, which is why this tab calibrates $\varepsilon$ adaptively rather than "
            r"exposing a raw distance threshold.",
            "A high cross-recurrence rate shows the two reconstructed trajectories occupy "
            "overlapping state-space regions at corresponding times — it is evidence "
            "consistent with coupling, not a directional causal statement. For a "
            "directionality test on the same nonlinear footing, see Convergent Cross "
            "Mapping (CCM) in the Causality tab's theory panel.",
            "Recurrence-based measures assume the dynamics are (at least locally) "
            "stationary within the analysis window; a long window spanning a genuine "
            "regime change will blend two different textures into one ambiguous plot — "
            "shorten the window or move it to isolate a single regime.",
            r"The window length is capped for interactivity: at a few hundred embedded "
            r"points the $O(N^2)$ recurrence matrix and its diagonal/vertical line scan stay "
            r"well under a second. The full record is covered by the sliding-window section "
            r"instead, which repeats the analysis window by window (cached per setting).",
            "Sliding-window RQA uses one window length — the dominant Lomb-Scargle period "
            "of Series A inside the chosen search band. A series with several strong cycles "
            "(daily and seasonal, say) gives very different windows depending on the band, "
            "so check the periodogram before reading the trends.",
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
            r"parameters $(\tau, m)$ are pre-identified per variable in the background (cached to "
            "disk after the first run); everything below updates live as you move the sliders."
        ),

        dcc.Interval(id="rqa-poll", interval=1500, n_intervals=0),
        dcc.Store(id="rqa-ready"),

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
            html.Div([
                html.Label(_md(r"Distance norm $\lVert\cdot\rVert$ in "
                               r"$\Theta(\varepsilon - \lVert \mathbf{X}_i - \mathbf{X}_j \rVert)$", inline=True)),
                dcc.Dropdown(id="rqa-metric", options=METRIC_OPTIONS, value=METRIC_DEFAULT,
                             clearable=False, style={"minWidth": "200px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),

        html.Div([
            html.Div([
                html.Label(_md(rf"Embedding dimension $m$ ({M_MIN}–{M_MAX})", inline=True)),
                dcc.Slider(id="rqa-m", min=M_MIN, max=M_MAX, step=1, value=3,
                           marks={i: str(i) for i in range(M_MIN, M_MAX + 1)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label(_md(rf"Time delay $\tau$, {UNIT_N} ({TAU_MIN}–{TAU_MAX})", inline=True)),
                dcc.Slider(id="rqa-tau", min=TAU_MIN, max=TAU_MAX, step=1, value=DS.lag(0.25),
                           marks={k: str(k) for k in (1, DS.lag(0.25), DS.lag(0.5), TAU_MAX)},
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
                html.Label(f"Window length, {UNIT_N} ({WINDOW_MIN}–{WINDOW_MAX})"),
                dcc.Slider(id="rqa-win-len", min=WINDOW_MIN, max=WINDOW_MAX, step=10,
                           value=WINDOW_DEFAULT,
                           marks={WINDOW_MIN: str(WINDOW_MIN), WINDOW_MAX: str(WINDOW_MAX)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label(f"Window start ({UNIT_N} from series start)"),
                dcc.Slider(id="rqa-win-start", min=0, max=_WIN_START_MAX, step=DS.period,
                           value=_WIN_START_DEFAULT, marks=_WIN_MARKS,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
        ], className="controls-row"),

        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div(id="rqa-banner"),
        ]),

        _section_hdr("Recurrence Plots"),
        html.Div(id="rqa-grid-desc", className="section-desc"),
        dcc.Loading(type="circle", color=ACCENT, children=[
            _plot_card([dcc.Graph(id="rqa-urp", config={"displaylogo": False})],
                       label=_md(r"Unthresholded RP · distance matrix $D_{ij} = \lVert \mathbf{X}_i - \mathbf{X}_j \rVert$ "
                                 r"before $\Theta(\varepsilon - D_{ij})$", inline=True)),
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
                       label=_md(r"Line-length distributions $P(\ell)$ and $P(v)$ · all four matrices "
                                 r"(log scale: slope of $\ln P(\ell)$ $\sim -K_2$)", inline=True)),
            html.Div(id="rqa-taurr-desc", className="section-desc"),
            _plot_card([dcc.Graph(id="rqa-taurr", config={"displaylogo": False})],
                       label=_md(r"$\tau$-recurrence rate $RR_\tau$ · recurrence analogue of the ACF",
                                 inline=True)),
        ]),

        _section_hdr("Sliding-Window RQA (window = Lomb-Scargle period)"),
        _section_desc(
            "The full record is cut into windows whose length equals Series A's dominant "
            "Lomb-Scargle period — one full cycle per window, so every window sees the same phase "
            "coverage — slid forward by the chosen step. For each window the auto-, cross- and "
            r"joint-recurrence matrices are rebuilt with the shared $(m, \tau)$ and distance norm above, "
            rf"and all {len(RQA_METRICS)} RQA measures are recomputed. $\varepsilon$ is calibrated once on the whole record to "
            "the target recurrence rate and then held fixed, so $RR$ itself can change from window "
            "to window. The shaded band marks the window inspected in the plots above."
        ),
        html.Div([
            html.Div([
                html.Label(f"Lomb-Scargle period search band ({_SAMPLES})"),
                dcc.RangeSlider(id="rqa-sw-band", min=2, max=SW_PERIOD_MAX, step=1,
                                value=[SW_PERIOD_MIN_DEFAULT, SW_PERIOD_MAX_DEFAULT],
                                marks={2: "2", DS.period: str(DS.period), DS.lag(7): str(DS.lag(7)),
                                       SW_PERIOD_MAX_DEFAULT: str(SW_PERIOD_MAX_DEFAULT),
                                       SW_PERIOD_MAX: str(SW_PERIOD_MAX)},
                                tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm", style={"flex": "2"}),
            html.Div([
                html.Label("Window step (% of window length)"),
                dcc.Slider(id="rqa-sw-step", min=5, max=100, step=5, value=SW_STEP_DEFAULT,
                           marks={5: "5", 25: "25", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
        ], className="controls-row"),
        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div(id="rqa-sw-banner"),
            _plot_card([dcc.Graph(id="rqa-sw-periodogram", config={"displayModeBar": False})],
                       label="Lomb-Scargle periodogram · dominant period → window length"),
            html.Div(id="rqa-sw-desc", className="section-desc"),
            _plot_card([dcc.Graph(id="rqa-sw-fig", config={"displaylogo": False})],
                       label=f"{len(RQA_METRICS)} RQA measures per sliding window · auto / cross / joint"),
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

    # Light poller — only the identification card depends on background work.
    # The tick never drives the (expensive, multi-MB) recurrence callback
    # directly; see synth_tab_control._poll for why.
    @app.callback(
        Output("rqa-ready", "data"),
        Output("rqa-poll", "disabled"),
        Input("rqa-poll", "n_intervals"),
        Input("rqa-var-a", "value"),
        Input("rqa-var-b", "value"),
        State("rqa-ready", "data"),
    )
    def _poll(_tick, colA, colB, prev):
        colA = colA or DEFAULT_COL_A
        colB = colB or DEFAULT_COL_B
        if colB == colA:
            colB = next((c for c in AVAILABLE if c != colA), colA)
        st = precompute_status()
        key = "ready" if (get_ident(colA) and get_ident(colB)) else ["wait", st["done"]]
        return (key if key != prev else no_update), key == "ready" or st["finished"]

    @app.callback(
        Output("rqa-banner", "children"),
        Output("rqa-urp", "figure"),
        Output("rqa-grid", "figure"),
        Output("rqa-grid-desc", "children"),
        Output("rqa-bars", "figure"),
        Output("rqa-table", "children"),
        Output("rqa-metrics-desc", "children"),
        Output("rqa-hist", "figure"),
        Output("rqa-taurr", "figure"),
        Output("rqa-taurr-desc", "children"),
        Output("rqa-id-card", "children"),
        Input("rqa-var-a", "value"),
        Input("rqa-var-b", "value"),
        Input("rqa-m", "value"),
        Input("rqa-tau", "value"),
        Input("rqa-rr", "value"),
        Input("rqa-win-len", "value"),
        Input("rqa-win-start", "value"),
        Input("rqa-metric", "value"),
        Input("rqa-ready", "data"),
    )
    def update_rqa(colA, colB, m, tau, target_rr, win_len, win_start, metric, _ready):
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

        metric = metric or METRIC_DEFAULT
        rec = build_recurrence(y_a, y_b, m, tau, target_rr, metric)
        st = precompute_status()

        if rec.get("L", 0) < 10:
            ph = _placeholder_fig("Window too short for this embedding — increase the "
                                   "window length or lower m / τ.")
            msg = _md(r"Embedded trajectory has too few points (window length minus "
                      r"$(m-1)\tau$). Widen the window, or reduce the embedding dimension / delay.")
            return (_banner(("—", "Window too short", RED)), ph, ph, msg, ph, None, msg,
                    ph, ph, None, None)

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
            (f"m={m}, τ={tau}h, {metric}", "Shared embedding · norm", AMBER),
            (f"{rec['L']:,}", "Embedded points", MUTED2),
        )

        grid_desc = _md(
            rf"Window {date_lo} → {date_hi} ({win_len} h, {rec['L']:,} embedded points after "
            rf"delay-embedding loses $(m-1)\tau = {(m - 1) * tau}$ points). The unthresholded "
            rf"plots show the raw distances $D_{{ij}}$ in the {metric} norm; each thresholded "
            rf"panel targets a {target_rr:.0f}% recurrence rate via an adaptively chosen "
            rf"$\varepsilon$ ($\varepsilon_A = {rec['eps_a']:.3f}$, $\varepsilon_B = {rec['eps_b']:.3f}$, "
            rf"$\varepsilon_{{\mathrm{{cross}}}} = {rec['eps_cross']:.3f}$); diagonal structure "
            rf"close to the main diagonal in the cross-recurrence panel (bottom-left) is the "
            rf"visual signature of coupling between {LABELS.get(colA, colA)} and "
            rf"{LABELS.get(colB, colB)}."
        )
        metrics_desc = _md(
            rf"RQA measures for all four matrices, with a Theiler window of $w = {theiler}$ "
            rf"samples (points with $|i - j| < w$ excluded) removing trivially auto-correlated "
            rf"near-diagonal points from the two auto-recurrence matrices and the joint-recurrence "
            r"matrix (the cross-recurrence matrix has no line-of-identity to exclude). Compare "
            r"Cross against the two Auto columns: comparable $DET$ / $LAM$ indicates the coupling "
            r"is as structured as either system's own dynamics, not a diffuse, noise-level effect. "
            r"$RTE$ is the entropy of the recurrence-time (white vertical line) distribution, "
            r"normalised to $[0, 1]$: near 0 for strictly periodic returns, near 1 for noise. "
            r"Below, $P(\ell)$ and $P(v)$ are the normalised line-length histograms behind $DET$, "
            r"$\langle L \rangle$, $ENTR$, $LAM$ and $TT$."
        )

        tau_max = max(1, rec["L"] // TAU_RR_FRACTION)
        T_a = _recurrence_period(*rqa_funs.tau_recurrence_rate(rec["R_a"], tau_max))
        T_b = _recurrence_period(*rqa_funs.tau_recurrence_rate(rec["R_b"], tau_max))
        fmt_T = lambda T: f"$T \\approx {T}$ {UNIT}" if T is not None else "no clear period"
        taurr_desc = _md(
            rf"$RR_\tau = \frac{{1}}{{N-\tau}}\sum_i R_{{i,i+\tau}}$ for lags $\tau = 1..{tau_max}$ "
            rf"(half the {rec['L']:,} embedded points; here $\tau$ is a lag, not the embedding "
            rf"delay). Peaks mark recurrence periods — the RQA analogue of ACF peaks, picking up "
            rf"daily or weekly cycles without assuming linearity: {LABELS.get(colA, colA)} "
            rf"{fmt_T(T_a)}, {LABELS.get(colB, colB)} {fmt_T(T_b)}. The dotted Cross curve "
            rf"runs over $\tau = -{tau_max}..{tau_max}$: a peak at $\tau > 0$ means "
            rf"{LABELS.get(colB, colB)} revisits {LABELS.get(colA, colA)}'s states $\tau$ samples later."
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
            fig_unthresholded_rp(rec, colA, colB, metric),
            fig_recurrence_grid(rec, colA, colB),
            grid_desc,
            fig_rqa_bars(mA, mB, mC, mJ, colA, colB),
            rqa_metric_table(mA, mB, mC, mJ, colA, colB),
            metrics_desc,
            fig_line_histograms(mA, mB, mC, mJ, colA, colB),
            fig_tau_rr(rec, colA, colB, tau_max),
            taurr_desc,
            id_card,
        )

    @app.callback(
        Output("rqa-sw-banner", "children"),
        Output("rqa-sw-periodogram", "figure"),
        Output("rqa-sw-desc", "children"),
        Output("rqa-sw-fig", "figure"),
        Input("rqa-var-a", "value"),
        Input("rqa-var-b", "value"),
        Input("rqa-m", "value"),
        Input("rqa-tau", "value"),
        Input("rqa-rr", "value"),
        Input("rqa-metric", "value"),
        Input("rqa-sw-band", "value"),
        Input("rqa-sw-step", "value"),
        Input("rqa-win-len", "value"),
        Input("rqa-win-start", "value"),
    )
    def update_sliding(colA, colB, m, tau, target_rr, metric, band, step_pct, win_len, win_start):
        colA = colA or DEFAULT_COL_A
        colB = colB or DEFAULT_COL_B
        if colB == colA:
            colB = next((c for c in AVAILABLE if c != colA), colA)
        m         = int(m or 3)
        tau       = int(tau or 6)
        target_rr = float(target_rr or RR_DEFAULT)
        metric    = metric or METRIC_DEFAULT
        p_min, p_max = (band or [SW_PERIOD_MIN_DEFAULT, SW_PERIOD_MAX_DEFAULT])
        p_min, p_max = int(p_min), int(max(p_max, p_min + 1))
        step_pct  = int(step_pct or SW_STEP_DEFAULT)
        win_len   = int(win_len or WINDOW_DEFAULT)
        win_start = max(0, min(int(win_start or 0), N_OBS - win_len))

        lsA = lomb_scargle_period(colA, p_min, p_max)
        lsB = lomb_scargle_period(colB, p_min, p_max)
        window = sliding_window_length(lsA[0], m, tau)
        step = max(1, round(window * step_pct / 100))
        pgram = fig_periodogram(lsA, lsB, colA, colB, window)

        try:
            sw = sliding_rqa(colA, colB, m, tau, target_rr, metric, window, step)
        except ValueError as e:
            ph = _placeholder_fig(str(e))
            return _banner(("—", "Sliding window unavailable", RED)), pgram, str(e), ph

        n_win = len(sw["A"])
        raised = window > round(lsA[0])
        banner = _banner(
            (f"{lsA[0]:.1f} {UNIT}", f"LS period · {LABELS.get(colA, colA)}", COLORS.get(colA, ACCENT2)),
            (f"{lsB[0]:.1f} {UNIT}", f"LS period · {LABELS.get(colB, colB)}", COLORS.get(colB, ACCENT)),
            (f"{window} {UNIT}", "Window length" + (" (raised)" if raised else ""), AMBER),
            (f"{step} {UNIT}", f"Step ({step_pct}%)", MUTED2),
            (f"{n_win:,}", "Windows", MUTED2),
            (f"{sw['cross'].attrs['epsilon']:.3f}", "Fixed ε · cross (z-units)", ACCENT2),
        )
        desc = _md(
            rf"{n_win} windows of $W = {window}$ h (= {sw['A'].attrs['n_emb']} embedded points with "
            rf"$m = {m}$, $\tau = {tau}$ h), stepping {step} h, {metric} norm. $\varepsilon$ fixed per "
            rf"matrix from the whole record at a {target_rr:.0f}% recurrence rate "
            rf"($\varepsilon_A = {sw['A'].attrs['epsilon']:.3f}$, $\varepsilon_B = {sw['B'].attrs['epsilon']:.3f}$, "
            rf"$\varepsilon_{{\mathrm{{cross}}}} = {sw['cross'].attrs['epsilon']:.3f}$); Theiler window "
            rf"$w = {max(1, tau)}$ for auto/joint, none for cross. "
            + (rf"The Lomb-Scargle period ($T = {lsA[0]:.1f}$ h) was too short for this embedding, so "
               rf"the window was raised to keep {SW_MIN_EMBEDDED} embedded points. " if raised else "")
            + "Peaks and drops that move together across panels mark regime changes in the "
              "dynamics; a Cross curve that tracks the two Auto curves indicates coupling that "
              "persists through time rather than in one window only."
        )
        return banner, pgram, desc, fig_sliding_rqa(sw, colA, colB, win_start, win_start + win_len)