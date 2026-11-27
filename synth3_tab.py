"""
synth3_tab.py  —  Class-III Synthetic Time-Series Generation (dynamics-preserving)
Drop this file next to app.py and:
    from synth3_tab import tab_synth3, register_synth3_callbacks
    ... add `tab_synth3` to the dbc.Tabs children list ...
    register_synth3_callbacks(app)

WHAT THIS TAB DOES
-------------------
Of the three classes of time-series synthetic-data method referenced
throughout this dashboard —
  (1) mimic the *shape*                   (synth_tab_control.py — SARIMA)
  (2) reproduce the *statistics*          (synth2_tab.py         — IAAFT / bootstrap / copula)
  (3) reproduce the *dynamical potential* (this tab)
this tab is the generative counterpart of Class (3), and of the RQA tab's
analysis. For any column of spg.csv it reconstructs the phase space with
Takens' delay embedding — τ from the first minimum of the average mutual
information, m from the false-nearest-neighbours fraction, the same
estimators as the RQA tab — and generates synthetic series with two
dynamics-preserving methods:

  Twin Surrogates  (Thiel et al., 2006)        a random walk on the real
                   attractor that jumps between *twins* (states with the same
                   recurrence neighbourhood), so every path is a new
                   trajectory of the same deterministic skeleton
  GENDA            (Troullinou et al., 2024)   a generative neighbourhood-based
                   deep autoencoder, adapted here to a phase-space rollout:
                   new trajectory pieces are decoded from random convex
                   mixtures of the latent codes of neighbouring passages

The validation battery then asks the Class-3 question directly — "is it the
same *dynamical system*?" — by comparing invariants that do not depend on
calendar alignment: re-identified (τ, m), largest Lyapunov exponent,
correlation dimension, sample and permutation entropy, DFA exponent,
time-reversal asymmetry, RQA measures, plus spectral / ordinal-pattern /
recurrence-curve distances. Each tolerance is calibrated by the real
series' own split-half variability.

Per-variable identification (τ, m, twin-availability scan, the reference
invariants) and GENDA training run eagerly in one background thread at
import time — the variable on screen is always processed first — and are
persisted to disk via cache_utils, so a later run loads them instantly.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import time
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from cache_utils import make_signature, load_cache, save_cache
from core.dataset import get_dataset
import synth3_funs as S3
from synth3_funs import (zscore, identify_embedding, twin_scan, twin_model, twin_surrogate,
                         genda_train, genda_generate, dynamical_profile, distances_to_real,
                         TORCH_OK, TORCH_ERR)

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

REAL_C   = "#e2e8f0"                     # the observed series — the reference everywhere
TS_C     = "#38bdf8"                     # twin surrogates
GD_C     = "#c084fc"                     # GENDA
TS_FILL  = "rgba(56,189,248,0.15)"
GD_FILL  = "rgba(192,132,252,0.15)"
METHOD_STYLE = {"ts": ("Twin Surrogates", TS_C, TS_FILL), "gd": ("GENDA", GD_C, GD_FILL)}

DS = get_dataset()
CSV_PATH = DS.path

# ─────────────────────────────────────────────────────────────────────────────
#  VARIABLE CATALOGUE  —  from the dataset config (configs/*.yaml via
#  core.dataset); edit the config, not this file, to change it.
# ─────────────────────────────────────────────────────────────────────────────
COLS   = DS.columns(include_target=True)   # the target too (first)
LABELS = DS.labels
UNITS  = DS.units

DEFAULT_COL   = DS.target or DS.default_variable()
N_REALS_MAX   = 8
N_REALS_DEF   = 4
TS_RR_MIN, TS_RR_MAX = 2, 15
K_MIN, K_MAX  = 2, 6
GENDA_THREADS = 4          # torch intra-op threads for background training

# ─────────────────────────────────────────────────────────────────────────────
#  DATA  (shared loader: core.dataset)
# ─────────────────────────────────────────────────────────────────────────────
def _load_df() -> pd.DataFrame:
    return DS.frame()          # time-indexed, built once per process


_RAW = _load_df()
AVAILABLE = [c for c in COLS if c in _RAW.columns]
DF = _RAW[AVAILABLE].copy()
VAR_OPTIONS = [{"label": LABELS.get(c, c), "value": c} for c in AVAILABLE]
if DEFAULT_COL not in AVAILABLE:
    DEFAULT_COL = AVAILABLE[0]


def _series(col: str) -> np.ndarray:
    s = DF[col]
    if s.isna().any():
        s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.values.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
#  PER-VARIABLE IDENTIFICATION  (stage 1 — seconds per variable)
#    1. τ (AMI first minimum) and m (FNN) — identical estimators to the RQA tab
#    2. twin-availability scan over the tolerance δ at the default ε
#    3. the reference dynamical profile of the real series and of its two
#       halves (the split-half spread calibrates every tolerance)
# ─────────────────────────────────────────────────────────────────────────────
def _identify(col: str) -> dict:
    y = _series(col)
    n = len(y)
    if np.std(y) < 1e-12:
        raise ValueError("constant series — no dynamics to reproduce")
    emb = identify_embedding(y)
    m, tau = emb["m"], emb["tau"]
    scan = twin_scan(zscore(y), m, tau)
    h = n // 2
    y1, y2 = y[:h], y[h:]
    prof = dynamical_profile(y, m, tau)
    p1, p2 = dynamical_profile(y1, m, tau), dynamical_profile(y2, m, tau)
    return dict(col=col, y=y, n=n, sd=float(np.std(y)), m=m, tau=tau, m_fnn=emb["m_fnn"],
                ami_curve=emb["ami_curve"], fnn_curve=emb["fnn_curve"], scan=scan,
                prof=prof, prof_h=(p1, p2),
                dist_h=distances_to_real(p1, y1, p2, y2))


# ─────────────────────────────────────────────────────────────────────────────
#  EAGER PRECOMPUTE  —  one daemon thread, two stages, priority to the
#  variable on screen. Stage 1 (identification) for every variable is cached
#  as one file; each GENDA model (stage 2, ~15 s each) as its own file, so a
#  run interrupted half-way keeps everything already trained.
# ─────────────────────────────────────────────────────────────────────────────
_ID_CACHE: dict = {}
_GD_CACHE: dict = {}
_STATUS_LOCK = threading.Lock()
_STATUS = dict(current=None, stage=None, id_done=0, gd_done=0, total=len(AVAILABLE),
               finished=False, id_errors={}, gd_errors={})
_PRIORITY = {"col": DEFAULT_COL}

_ID_SIG = make_signature(
    CSV_PATH, version="synth3-identification-v5",   # bump on any identification-logic change
    available=AVAILABLE, tau_bounds=S3.TAU_BOUNDS, m_bounds=S3.M_BOUNDS,
    rr=S3.TS_RR_DEFAULT, deltas=S3.TS_DELTA_GRID, twin_frac=S3.TS_TWIN_FRAC, gate=S3.TS_GATE,
)


def _gd_sig(col: str, m: int, tau: int) -> str:
    return make_signature(
        CSV_PATH, version="synth3-genda-v1", col=col, m=m, tau=tau,   # bump on a training change
        k=S3.GENDA_K, latent=S3.GENDA_LATENT, epochs=S3.GENDA_EPOCHS, lr=S3.GENDA_LR,
        batch=S3.GENDA_BATCH, filters=S3.GENDA_FILTERS,
    )


def _gd_cache_name(col: str) -> str:
    return f"synth3_genda_{col}"


def _next_task():
    """(col, stage) to run next: the on-screen variable first, then every
    identification, then every GENDA model."""
    pending_id = [c for c in AVAILABLE if c not in _ID_CACHE and c not in _STATUS["id_errors"]]
    pending_gd = [c for c in AVAILABLE if c in _ID_CACHE and c not in _GD_CACHE
                  and c not in _STATUS["gd_errors"]] if TORCH_OK else []
    want = _PRIORITY["col"]
    if want in pending_id:
        return want, "ident"
    if want in pending_gd:
        return want, "genda"
    if pending_id:
        return pending_id[0], "ident"
    if pending_gd:
        return pending_gd[0], "genda"
    return None, None


def _precompute_worker():
    t_start = time.time()
    id_dirty = False
    while True:
        with _STATUS_LOCK:
            col, stage = _next_task()
            _STATUS.update(current=col, stage=stage)
        if col is None:
            break
        t0 = time.time()
        try:
            if stage == "ident":
                _ID_CACHE[col] = _identify(col)
                id_dirty = True
                ident = _ID_CACHE[col]
                print(f"[synth3_tab]  identified {col}: tau={ident['tau']} m={ident['m']} "
                      f"(FNN {ident['m_fnn']})  delta*={ident['scan']['delta_default']:.2f}  "
                      f"[{time.time() - t0:.1f}s]")
            else:
                ident = _ID_CACHE[col]
                sig = _gd_sig(col, ident["m"], ident["tau"])
                bundle = load_cache(_gd_cache_name(col), sig)  # trained by an earlier run?
                if bundle is None:
                    bundle = genda_train(ident["y"], ident["m"], ident["tau"],
                                         n_threads=GENDA_THREADS)
                    save_cache(_gd_cache_name(col), sig, bundle)
                    print(f"[synth3_tab]  GENDA trained for {col}: loss "
                          f"{bundle['losses'][-1]:.5f}  [{time.time() - t0:.1f}s]")
                _GD_CACHE[col] = bundle
        except Exception as e:                          # never kill the thread
            key = "id_errors" if stage == "ident" else "gd_errors"
            with _STATUS_LOCK:
                _STATUS[key][col] = f"{type(e).__name__}: {e}"
            print(f"[synth3_tab]  {stage} failed for {col} — {type(e).__name__}: {e}")
        with _STATUS_LOCK:
            _STATUS["id_done"] = len(_ID_CACHE)
            _STATUS["gd_done"] = len(_GD_CACHE)
        # persist the identification once every variable has it (one file)
        if id_dirty and all(c in _ID_CACHE or c in _STATUS["id_errors"] for c in AVAILABLE):
            save_cache("synth3_identification", _ID_SIG, dict(_ID_CACHE))
            id_dirty = False
    print(f"[synth3_tab] Ready OK — Class-3 identification + GENDA training done in "
          f"{time.time() - t_start:.1f}s")
    with _STATUS_LOCK:
        _STATUS.update(finished=True, current=None, stage=None)


def _load_cached():
    cached = load_cache("synth3_identification", _ID_SIG)
    if cached is not None:
        _ID_CACHE.update(cached)
    for col, ident in _ID_CACHE.items():
        bundle = load_cache(_gd_cache_name(col), _gd_sig(col, ident["m"], ident["tau"]))
        if bundle is not None:
            _GD_CACHE[col] = bundle
    with _STATUS_LOCK:
        _STATUS.update(id_done=len(_ID_CACHE), gd_done=len(_GD_CACHE))


_load_cached()
_all_done = len(_ID_CACHE) == len(AVAILABLE) and (not TORCH_OK or len(_GD_CACHE) == len(AVAILABLE))
if _all_done:
    with _STATUS_LOCK:
        _STATUS["finished"] = True
    print(f"[synth3_tab] Loaded cached Class-3 identification + {len(_GD_CACHE)} GENDA models "
          "from .cache/ ✔  (delete .cache/ to force a full recompute)")
else:
    if not TORCH_OK:
        print(f"[synth3_tab] PyTorch unavailable — GENDA disabled ({TORCH_ERR})")
    print(f"[synth3_tab] Starting background Class-3 identification ({len(_ID_CACHE)}/"
          f"{len(AVAILABLE)} cached) and GENDA training ({len(_GD_CACHE)}/{len(AVAILABLE)} "
          "cached) — the variable on screen is always processed first.")
    _WORKER = threading.Thread(target=_precompute_worker, daemon=True, name="synth3-precompute")
    _WORKER.start()


def get_ident(col: str):
    return _ID_CACHE.get(col)


def get_genda(col: str):
    return _GD_CACHE.get(col)


def precompute_status() -> dict:
    with _STATUS_LOCK:
        st = dict(_STATUS)
        st["id_errors"] = dict(_STATUS["id_errors"])
        st["gd_errors"] = dict(_STATUS["gd_errors"])
        return st


# ─────────────────────────────────────────────────────────────────────────────
#  ON-DEMAND GENERATION + VALIDATION  (cached; one lock so the two callbacks
#  that need the same ensemble never generate it twice)
# ─────────────────────────────────────────────────────────────────────────────
_GEN_LOCK = threading.Lock()


@lru_cache(maxsize=16)
def _twin_model(col: str, rr: int, delta: float):
    ident = get_ident(col)
    return twin_model(zscore(ident["y"]), ident["m"], ident["tau"], rr, delta)


@lru_cache(maxsize=16)
def _ensemble_cached(col: str, n_sims: int, seed: int, rr: int, delta: float, k: int,
                     with_genda: bool):
    ident = get_ident(col)
    y, n = ident["y"], ident["n"]
    tm = _twin_model(col, rr, delta)
    ts, info = np.empty((n, n_sims)), []
    for s in range(n_sims):
        ts[:, s], inf = twin_surrogate(y, tm, n, np.random.default_rng(seed + 1000 * s))
        info.append(inf)
    gd = genda_generate(get_genda(col), n, n_sims, seed, k=k, y_ref=y) if with_genda else None
    return dict(ts=ts, ts_info=info, tm=tm, gd=gd)


def ensemble(col, n_sims, seed, rr, delta, k):
    with _GEN_LOCK:
        return _ensemble_cached(col, n_sims, seed, rr, delta, k, get_genda(col) is not None)


@lru_cache(maxsize=16)
def _profiles_cached(col, n_sims, seed, rr, delta, k, with_genda):
    ident = get_ident(col)
    ens = _ensemble_cached(col, n_sims, seed, rr, delta, k, with_genda)
    m, tau, y = ident["m"], ident["tau"], ident["y"]
    jobs = [("ts", s, ens["ts"][:, s]) for s in range(n_sims)]
    if ens["gd"] is not None:
        jobs += [("gd", s, ens["gd"][:, s]) for s in range(n_sims)]

    def _run(job):
        meth, _, x = job
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = dynamical_profile(x, m, tau)
            return meth, p, distances_to_real(ident["prof"], y, p, x)

    with ThreadPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(_run, jobs))
    out = {"ts": [], "gd": []}
    for meth, p, d in res:
        out[meth].append(dict(prof=p, dist=d))
    return out


def profiles(col, n_sims, seed, rr, delta, k):
    with_genda = get_genda(col) is not None
    ensemble(col, n_sims, seed, rr, delta, k)          # make sure the ensemble exists
    with _GEN_LOCK:
        return _profiles_cached(col, n_sims, seed, rr, delta, k, with_genda)


# ─────────────────────────────────────────────────────────────────────────────
#  SCOREBOARD  —  one table, grouped by the aspect of the dynamics each row
#  tests. Every tolerance is calibrated by the real series itself: invariants
#  by their split-half spread (first vs second half of the record), distances
#  by the real-½-vs-real-½ baseline. A synthetic ensemble deviating less than
#  the real process deviates from itself is indistinguishable from natural
#  variability. Redundant measures are left out of the table (they remain in
#  the figures): W₁ duplicates KS, permutation entropy is a scalar summary of
#  the ordinal distribution that the JSD row compares in full, ⟨L⟩ and ENTR
#  describe the same diagonal lines as DET, TT the same vertical lines as LAM,
#  and the AMI curve is summarised by the re-identified τ̂.
# ─────────────────────────────────────────────────────────────────────────────
G_LIN, G_MARG, G_PS, G_NL, G_REC = ("Linear dynamics", "Marginal distribution", "Phase space",
                                    "Nonlinear & chaotic dynamics", "Recurrence structure")
GROUP_NOTE = {
    G_LIN:  "memory and spectrum (second-order structure)",
    G_MARG: "values taken by the series",
    G_PS:   "the reconstructed attractor itself",
    G_NL:   "what no linear Gaussian model can reproduce",
    G_REC:  "how and how often states recur",
}
# (group, key, LaTeX label, chart tick, kind, param)
#   kind "int"  integer re-identification        tol = max(0.5, split-half spread)
#   kind "rel"  invariant, param = floor of |real| tol = max(0.1·max(|real|, floor), spread)
#   kind "dist" distance to real, param = threshold  tol = max(threshold, ½-vs-½ baseline)
SCORE_ROWS = [
    (G_LIN,  "acf_rmse",   rf"ACF RMSE, lags $1$–${S3.ACF_LAGS}$",               "ACF",    "dist", 0.05),
    (G_LIN,  "lsd",        r"Log-spectral distance (dB)",                      "PSD",    "dist", 2.0),
    (G_LIN,  "dfa",        r"DFA exponent $\alpha$",                           "DFA α",  "rel",  1e-3),
    (G_MARG, "ks",         r"Kolmogorov–Smirnov $D$",                          "KS",     "dist", 0.05),
    (G_PS,   "tau_hat",    r"Delay $\hat\tau$ re-identified (AMI)",            "τ̂",      "int",  None),
    (G_PS,   "m_hat",      r"Dimension $\hat m$ re-identified (FNN)",          "m̂",      "int",  None),
    (G_PS,   "d2",         r"Correlation dimension $D_2$",                     "D₂",     "rel",  1e-3),
    (G_NL,   "lle",        r"Largest Lyapunov exponent $\lambda_1$",           "λ₁",     "rel",  1e-3),
    (G_NL,   "sampen",     r"Sample entropy $\mathrm{SampEn}(2,\,0.2\sigma)$", "SampEn", "rel",  1e-3),
    (G_NL,   "trev",       r"Time-reversal asymmetry $A_3$",                   "A₃",     "rel",  0.25),
    (G_NL,   "jsd",        r"Ordinal-pattern $\mathrm{JSD}$ (bits)",           "JSD",    "dist", 0.01),
    (G_REC,  "rqa_det",    r"Determinism $DET$",                               "DET",    "rel",  1e-3),
    (G_REC,  "rqa_lam",    r"Laminarity $LAM$",                                "LAM",    "rel",  1e-3),
    (G_REC,  "rrtau_rmse", r"$RR_\tau$-curve RMSE",                            "RR_τ",   "dist", 0.05),
]
_VERDICT = {0: ("match", GREEN), 1: ("close", AMBER), 2: ("off", RED), None: ("n/a", MUTED)}
_CREDIT = {0: 1.0, 1: 0.5, 2: 0.0}


def _mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return (float(a.mean()), float(a.std())) if len(a) else (np.nan, np.nan)


def _grade(dev):
    """dev = |deviation| / tolerance → 0 match, 1 close, 2 off; None when the
    measure cannot be estimated (degenerate series) — excluded from the score."""
    if dev is None or not np.isfinite(dev):
        return None
    return 0 if dev <= 1.0 else (1 if dev <= 2.0 else 2)


def _score(grades):
    g = [x for x in grades if x is not None]
    if not g:
        return None
    return dict(pct=100.0 * np.mean([_CREDIT[x] for x in g]),
                n_match=sum(x == 0 for x in g), n=len(g))


def evaluate(ident: dict, profs: dict) -> dict:
    """Per-row verdicts for both methods, per-group and overall fidelity scores."""
    real, (h1, h2) = ident["prof"], ident["prof_h"]
    rows = []
    for group, key, label, tick, kind, param in SCORE_ROWS:
        if kind == "dist":
            base = float(ident["dist_h"][key])
            r, ref = 0.0, base
            tol = max(param, base) if np.isfinite(base) else param
        else:
            r = float(real[key])
            ref = abs(float(h1[key]) - float(h2[key]))   # split-half spread
            if kind == "int":
                tol = max(0.5, ref) if np.isfinite(ref) else 0.5
            else:
                tol = max(0.10 * max(abs(r), param), ref) if np.isfinite(ref) else 0.10 * max(abs(r), param)
        row = dict(group=group, key=key, label=label, tick=tick, kind=kind,
                   real=r, ref=ref, tol=tol)
        for meth in ("ts", "gd"):
            if profs.get(meth):
                src = "dist" if kind == "dist" else "prof"
                mu, sd = _mean_sd([p[src][key] for p in profs[meth]])
                dev = (mu - r) / tol
                row[meth] = dict(mu=mu, sd=sd, dev=dev, grade=_grade(abs(dev)))
        rows.append(row)
    score, by_group = {}, {}
    for meth in ("ts", "gd"):
        sc = _score([r[meth]["grade"] for r in rows if meth in r])
        if sc:
            score[meth] = sc
        for g in GROUP_NOTE:
            gs = _score([r[meth]["grade"] for r in rows if r["group"] == g and meth in r])
            if gs:
                by_group.setdefault(g, {})[meth] = gs
    return dict(rows=rows, score=score, by_group=by_group)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
_AX = dict(showgrid=True, gridcolor=GRID, gridwidth=1,
           zeroline=True, zerolinecolor=ZERO_L, zerolinewidth=1,
           linecolor=BORDER, linewidth=1, showline=True, mirror=False,
           tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
           title_font=dict(size=10, color=MUTED2, family=FONT_MONO))
_LEGEND = dict(font=dict(size=10, color=MUTED2, family=FONT_MONO), bgcolor="rgba(0,0,0,0)",
               orientation="h", y=1.13, x=0)
# subplot figures: legend above the subplot titles
_LEGEND_SUB = dict(_LEGEND, y=1.25)
_MARGIN_SUB = dict(t=84, b=40, l=58, r=16)


def _base_layout(**kw):
    base = dict(paper_bgcolor=BG, plot_bgcolor=SURFACE,
                font=dict(family=FONT_MONO, color=TEXT, size=11),
                hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(size=12, color=TEXT)),
                margin=dict(t=40, b=40, l=58, r=16))
    base.update(kw)
    return base


def _unit_suffix(col):
    u = UNITS.get(col, "")
    return f" ({u})" if u else ""


def _placeholder_fig(msg: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=12, color=MUTED2, family=FONT_MONO))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**_base_layout(height=height, showlegend=False))
    return fig


def _methods(ens):
    out = [("ts", ens["ts"])]
    if ens["gd"] is not None:
        out.append(("gd", ens["gd"]))
    return out


def fig_series(ident, ens) -> go.Figure:
    """Real series and the first realisation of each method, stacked."""
    col = ident["col"]
    rows = [("Real (observed)", ident["y"], REAL_C)]
    rows += [(f"{METHOD_STYLE[m][0]} · realisation 1", sims[:, 0], METHOD_STYLE[m][1])
             for m, sims in _methods(ens)]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[r[0] for r in rows])
    x = np.arange(ident["n"])
    lo, hi = float(ident["y"].min()), float(ident["y"].max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    for i, (name, yv, c) in enumerate(rows, start=1):
        fig.add_trace(go.Scattergl(x=x, y=yv, mode="lines", name=name, line=dict(color=c, width=1),
                                   hovertemplate="t=%{x}<br>%{y:.3g}<extra>" + name + "</extra>"),
                      row=i, col=1)
        fig.update_yaxes(**_AX, range=[lo - pad, hi + pad], row=i, col=1,
                         title=dict(text=UNITS.get(col, ""), standoff=4))
        fig.update_xaxes(**_AX, row=i, col=1)
    fig.update_xaxes(title=dict(text="sample index", standoff=4), row=len(rows), col=1)
    fig.update_layout(**_base_layout(height=170 * len(rows) + 60, showlegend=False,
                                     hovermode="x unified"))
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    return fig


def fig_portraits(ident, ens) -> go.Figure:
    """Delay portraits (x_t, x_{t+τ}): the 2-D shadow of the reconstructed attractor."""
    tau = ident["tau"]
    rows = [("Real", ident["y"], REAL_C)]
    rows += [(METHOD_STYLE[m][0], sims[:, 0], METHOD_STYLE[m][1]) for m, sims in _methods(ens)]
    fig = make_subplots(rows=1, cols=len(rows), horizontal_spacing=0.06,
                        subplot_titles=[f"{r[0]} · (x_t, x_t+{tau})" for r in rows])
    lo, hi = float(ident["y"].min()), float(ident["y"].max())
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    for i, (name, yv, c) in enumerate(rows, start=1):
        fig.add_trace(go.Scattergl(x=yv[:-tau], y=yv[tau:], mode="markers", name=name,
                                   marker=dict(size=3, color=c, opacity=0.35),
                                   hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(**_AX, range=[lo - pad, hi + pad], row=1, col=i,
                         title=dict(text="x_t", standoff=4))
        fig.update_yaxes(**_AX, range=[lo - pad, hi + pad], row=1, col=i,
                         title=dict(text=f"x_t+{tau}" if i == 1 else "", standoff=4))
    fig.update_layout(**_base_layout(height=360, showlegend=False))
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    return fig


def _band(fig, x, curves, color, fill, name, row=None, col=None, dash="dash"):
    """Ensemble min–max band + ensemble mean line."""
    C = np.asarray(curves, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi, mu = np.nanmin(C, 0), np.nanmax(C, 0), np.nanmean(C, 0)
    x = list(x)
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(hi) + list(lo)[::-1], fill="toself", mode="lines",
                             fillcolor=fill, line=dict(width=0), hoverinfo="skip",
                             showlegend=False, legendgroup=name), **kw)
    fig.add_trace(go.Scatter(x=x, y=mu, mode="lines", name=name, legendgroup=name,
                             line=dict(color=color, width=2, dash=dash),
                             showlegend=row in (None, 1) and col in (None, 1)), **kw)


def fig_embedding(ident, profs) -> go.Figure:
    """AMI and FNN curves of the real series vs each ensemble — synthetic
    data with the same dynamics must re-identify the same (τ, m)."""
    real = ident["prof"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=["Average mutual information I(τ)",
                                        "False nearest neighbours (%)"])
    lags = np.arange(1, len(real["ami_curve"]) + 1)
    dims = np.arange(1, len(real["fnn_curve"]) + 1)
    for meth in ("ts", "gd"):
        if profs.get(meth):
            name, c, f = METHOD_STYLE[meth]
            _band(fig, lags, [p["prof"]["ami_curve"] for p in profs[meth]], c, f, name, 1, 1)
            _band(fig, dims, [p["prof"]["fnn_curve"] for p in profs[meth]], c, f, name, 1, 2)
    fig.add_trace(go.Scatter(x=lags, y=real["ami_curve"], mode="lines+markers", name="Real",
                             line=dict(color=REAL_C, width=2), marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dims, y=real["fnn_curve"], mode="lines+markers", name="Real",
                             showlegend=False, line=dict(color=REAL_C, width=2),
                             marker=dict(size=4)), row=1, col=2)
    fig.add_vline(x=ident["tau"], line=dict(color=AMBER, width=1.3, dash="dot"), row=1, col=1)
    fig.add_vline(x=ident["m_fnn"], line=dict(color=AMBER, width=1.3, dash="dot"), row=1, col=2)
    fig.update_xaxes(**_AX, title=dict(text="lag τ (samples)", standoff=4), row=1, col=1)
    fig.update_yaxes(**_AX, title=dict(text="I(τ) (nats)", standoff=4), row=1, col=1)
    fig.update_xaxes(**_AX, title=dict(text="embedding dimension d", standoff=4), row=1, col=2)
    fig.update_yaxes(**_AX, title=dict(text="FNN (%)", standoff=4), row=1, col=2)
    fig.update_layout(**_base_layout(height=340, legend=_LEGEND_SUB, margin=_MARGIN_SUB))
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    return fig


def fig_invariants(ev) -> go.Figure:
    """One dot per scoreboard row and method: the deviation from the real
    series in units of that row's tolerance, ±1 sd whiskers. Invariants are
    signed, (synthetic − real)/tol; distances are one-sided, distance/tol.
    Inside the green band = within the real series' own natural variability."""
    rows = ev["rows"]
    x = np.arange(len(rows))
    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(45,212,160,0.10)", line_width=0)
    fig.add_hrect(y0=1, y1=2, fillcolor="rgba(245,166,35,0.07)", line_width=0)
    fig.add_hrect(y0=-2, y1=-1, fillcolor="rgba(245,166,35,0.07)", line_width=0)
    # group separators and captions
    starts = [i for i, r in enumerate(rows) if i == 0 or r["group"] != rows[i - 1]["group"]]
    for k, i0 in enumerate(starts):
        i1 = (starts[k + 1] if k + 1 < len(starts) else len(rows)) - 1
        if i0:
            fig.add_vline(x=i0 - 0.5, line=dict(color=BORDER, width=1))
        fig.add_annotation(x=(i0 + i1) / 2, y=4.1, text=rows[i0]["group"].split(" ")[0],
                           showarrow=False, font=dict(size=10, color=MUTED2, family=FONT_MONO))
    for off, meth in ((-0.12, "ts"), (0.12, "gd")):
        if not all(meth in r for r in rows):
            continue
        name, c, _ = METHOD_STYLE[meth]
        dev = [np.clip(r[meth]["dev"], -3.8, 3.8) if np.isfinite(r[meth]["dev"]) else None
               for r in rows]
        err = [r[meth]["sd"] / r["tol"] for r in rows]
        fig.add_trace(go.Scatter(
            x=x + off, y=dev, mode="markers", name=name,
            marker=dict(size=10, color=c, line=dict(color=BG, width=1)),
            error_y=dict(type="data", array=err, color=c, thickness=1.2, width=4),
            customdata=[[r["label"].replace("$", ""), r["real"] if r["kind"] != "dist" else 0.0,
                         r[meth]["mu"], r[meth]["sd"], r["tol"]] for r in rows],
            hovertemplate=("%{customdata[0]}<br>real %{customdata[1]:.4g}<br>synthetic "
                           "%{customdata[2]:.4g} ± %{customdata[3]:.3g}<br>tolerance "
                           "%{customdata[4]:.3g}<extra>" + name + "</extra>")))
    fig.add_hline(y=0, line=dict(color=REAL_C, width=1.2))
    fig.update_xaxes(**_AX, tickmode="array", tickvals=list(x), ticktext=[r["tick"] for r in rows],
                     range=[-0.6, len(rows) - 0.4])
    fig.update_yaxes(**_AX, range=[-4.0, 4.4],
                     title=dict(text="deviation / tolerance", standoff=4))
    fig.update_layout(**_base_layout(height=360, legend=_LEGEND, hovermode="closest",
                                     margin=dict(t=50, b=40, l=64, r=16)))
    return fig


def fig_lyapunov(ident, profs) -> go.Figure:
    """Rosenstein divergence curves ⟨ln d_j(i)⟩ — the slope of the initial
    linear region is λ₁ (fitted over the first τ steps, shaded)."""
    real = ident["prof"]
    steps = np.arange(len(real["lyap_curve"]))
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=max(2, ident["tau"]), fillcolor="rgba(245,166,35,0.06)", line_width=0)
    for meth in ("ts", "gd"):
        if profs.get(meth):
            name, c, f = METHOD_STYLE[meth]
            _band(fig, steps, [p["prof"]["lyap_curve"] for p in profs[meth]], c, f, name)
    fig.add_trace(go.Scatter(x=steps, y=real["lyap_curve"], mode="lines+markers", name="Real",
                             line=dict(color=REAL_C, width=2), marker=dict(size=4)))
    fig.update_xaxes(**_AX, title=dict(text="steps i (samples)", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="⟨ln d(i)⟩", standoff=4))
    fig.update_layout(**_base_layout(height=320, legend=_LEGEND, margin=dict(t=50, b=40, l=58, r=16)))
    return fig


def fig_rrtau(ident, profs) -> go.Figure:
    real = ident["prof"]
    taus = real["rqa_taus"]
    fig = go.Figure()
    for meth in ("ts", "gd"):
        if profs.get(meth):
            name, c, f = METHOD_STYLE[meth]
            _band(fig, taus, [p["prof"]["rqa_rr_tau"] for p in profs[meth]], c, f, name)
    fig.add_trace(go.Scatter(x=taus, y=real["rqa_rr_tau"], mode="lines", name="Real",
                             line=dict(color=REAL_C, width=2)))
    fig.update_xaxes(**_AX, title=dict(text="lag τ (samples)", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="RR_τ", standoff=4))
    fig.update_layout(**_base_layout(height=320, legend=_LEGEND, margin=dict(t=50, b=40, l=58, r=16)))
    return fig


def fig_spectral(ident, profs) -> go.Figure:
    real = ident["prof"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=["Autocorrelation function", "Welch power spectral density"])
    lags = np.arange(len(real["acf"]))
    f = real["psd_f"][1:]
    for meth in ("ts", "gd"):
        if profs.get(meth):
            name, c, fl = METHOD_STYLE[meth]
            _band(fig, lags, [p["prof"]["acf"] for p in profs[meth]], c, fl, name, 1, 1)
            _band(fig, f, [10 * np.log10(p["prof"]["psd"][1:] + 1e-20) for p in profs[meth]],
                  c, fl, name, 1, 2)
    fig.add_trace(go.Scatter(x=lags, y=real["acf"], mode="lines", name="Real",
                             line=dict(color=REAL_C, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f, y=10 * np.log10(real["psd"][1:] + 1e-20), mode="lines",
                             name="Real", showlegend=False, line=dict(color=REAL_C, width=1.6)),
                  row=1, col=2)
    fig.update_xaxes(**_AX, title=dict(text="lag (samples)", standoff=4), row=1, col=1)
    fig.update_yaxes(**_AX, title=dict(text="ρ(lag)", standoff=4), row=1, col=1)
    fig.update_xaxes(**_AX, type="log", title=dict(text="frequency (cycles / sample)", standoff=4),
                     row=1, col=2)
    fig.update_yaxes(**_AX, title=dict(text="PSD (dB)", standoff=4), row=1, col=2)
    fig.update_layout(**_base_layout(height=350, legend=_LEGEND_SUB, margin=_MARGIN_SUB))
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    return fig


def fig_marginal(ident, ens) -> go.Figure:
    col = ident["col"]
    unit = _unit_suffix(col)
    q = np.linspace(0, 1, 300)
    q_real = np.quantile(ident["y"], q)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=["ECDF · real vs pooled synthetic",
                                        "QQ plot · synthetic vs real quantiles"])
    fig.add_trace(go.Scatter(x=q_real, y=q, mode="lines", name="Real",
                             line=dict(color=REAL_C, width=2)), row=1, col=1)
    lo, hi = float(q_real.min()), float(q_real.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                             line=dict(color=MUTED, width=1, dash="dot")), row=1, col=2)
    for meth, sims in _methods(ens):
        name, c, _ = METHOD_STYLE[meth]
        q_syn = np.quantile(sims.ravel(), q)
        fig.add_trace(go.Scatter(x=q_syn, y=q, mode="lines", name=name,
                                 line=dict(color=c, width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=q_real, y=q_syn, mode="markers", name=name, showlegend=False,
                                 marker=dict(size=4, color=c, opacity=0.8)), row=1, col=2)
    fig.update_xaxes(**_AX, title=dict(text=f"value{unit}", standoff=4), row=1, col=1)
    fig.update_yaxes(**_AX, title=dict(text="F(x)", standoff=4), row=1, col=1)
    fig.update_xaxes(**_AX, title=dict(text=f"real quantile{unit}", standoff=4), row=1, col=2)
    fig.update_yaxes(**_AX, title=dict(text=f"synthetic quantile{unit}", standoff=4), row=1, col=2)
    fig.update_layout(**_base_layout(height=350, legend=_LEGEND_SUB, hovermode="closest",
                                     margin=_MARGIN_SUB))
    fig.update_annotations(font=dict(size=11, color=MUTED2, family=FONT_MONO))
    return fig


def fig_ordinal(ident, profs) -> go.Figure:
    """Bandt–Pompe ordinal-pattern distribution, patterns sorted by their
    real probability — forbidden / rare patterns are a fingerprint of
    determinism that a stochastic look-alike fails to reproduce."""
    p_real = np.asarray(ident["prof"]["p_ord"])
    order = np.argsort(-p_real)
    x = np.arange(len(p_real))
    fig = go.Figure(go.Bar(x=x, y=p_real[order], name="Real", marker_color=REAL_C, opacity=0.55))
    for meth in ("ts", "gd"):
        if profs.get(meth):
            name, c, _ = METHOD_STYLE[meth]
            P = np.array([p["prof"]["p_ord"] for p in profs[meth]])[:, order]
            fig.add_trace(go.Scatter(x=x, y=P.mean(0), mode="lines+markers", name=name,
                                     line=dict(color=c, width=1.6), marker=dict(size=4),
                                     error_y=dict(type="data", array=P.std(0), color=c,
                                                  thickness=1, width=2)))
    fig.update_xaxes(**_AX, title=dict(text="ordinal pattern π (ranked by real frequency)",
                                       standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="p(π)", standoff=4))
    fig.update_layout(**_base_layout(height=320, legend=_LEGEND, hovermode="closest",
                                     margin=dict(t=50, b=40, l=58, r=16)))
    return fig


def fig_twin_scan(ident, delta_sel) -> go.Figure:
    sc = ident["scan"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sc["deltas"], y=100 * sc["frac"], mode="lines+markers",
                             name="states with ≥ 1 twin (%)", line=dict(color=TS_C, width=2)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=sc["deltas"], y=sc["mean_twins"], mode="lines+markers",
                             name="mean twins per state", line=dict(color=MUTED2, width=1.4,
                                                                    dash="dot")),
                  secondary_y=True)
    fig.add_vline(x=delta_sel, line=dict(color=AMBER, width=1.4, dash="dash"))
    fig.add_hline(y=100 * S3.TS_TWIN_FRAC, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(**_AX, title=dict(text="tolerance δ (δ = 0 → exact twins)", standoff=4))
    fig.update_yaxes(**_AX, title=dict(text="% of states", standoff=4), secondary_y=False)
    fig.update_yaxes(**{**_AX, "showgrid": False}, title=dict(text="twins / state", standoff=4),
                     secondary_y=True)
    fig.update_layout(**_base_layout(height=260, legend=_LEGEND, margin=dict(t=46, b=40, l=58, r=58)))
    return fig


def fig_genda_loss(bundle) -> go.Figure:
    losses = bundle["losses"]
    fig = go.Figure(go.Scatter(x=np.arange(1, len(losses) + 1), y=losses, mode="lines+markers",
                               line=dict(color=GD_C, width=2), marker=dict(size=4),
                               hovertemplate="epoch %{x}<br>loss %{y:.5f}<extra></extra>"))
    fig.update_xaxes(**_AX, title=dict(text="epoch", standoff=4))
    fig.update_yaxes(**_AX, type="log", title=dict(text="neighbour-reconstruction MSE", standoff=4))
    fig.update_layout(**_base_layout(height=260, showlegend=False))
    return fig


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


def _theory_card(title, tag, color, paragraphs, equations=()):
    children = [html.H3([html.Span(title, style={"color": color}), html.Span(tag, className="tc-tag")])]
    for p in paragraphs:
        children.append(_md(p) if isinstance(p, str) else html.Div(p, style={"marginBottom": "8px"}))
    for eq in equations:
        children.append(_md(f"$${eq}$$", className="math"))
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


_TABLE_STYLE = {"width": "100%", "borderCollapse": "collapse", "fontFamily": FONT_MONO,
                "fontSize": "12px", "color": TEXT}


def _fmt(v, kind="rel"):
    if v is None or not np.isfinite(v):
        return "—"
    if kind == "int":
        return f"{v:.1f}"
    a = abs(v)
    return f"{v:.3f}" if 1e-2 <= a < 1e3 or a == 0 else f"{v:.3g}"


def _verdict_cell(cell):
    word, colr = _VERDICT[cell["grade"]]
    return html.Td(word, style={"color": colr, "fontWeight": "700"})


def _group_summary(ev, group, have):
    parts = []
    for m in have:
        gs = ev["by_group"].get(group, {}).get(m)
        if gs:
            name, c, _ = METHOD_STYLE[m]
            parts.append(html.Span(f"{name} {gs['n_match']}/{gs['n']} match",
                                   style={"color": c, "marginLeft": "14px"}))
    return parts


def scoreboard_table(ev):
    """Single scoreboard: rows grouped by the aspect of the dynamics they test."""
    rows = ev["rows"]
    have = [m for m in ("ts", "gd") if all(m in r for r in rows)]
    head = [html.Th("Measure"), html.Th("Real"), html.Th("Tolerance")]
    for m in have:
        name, c, _ = METHOD_STYLE[m]
        head += [html.Th(f"{name} (mean ± sd)", style={"color": c}), html.Th("Verdict")]
    out = [html.Tr(head)]
    for i, r in enumerate(rows):
        if i == 0 or r["group"] != rows[i - 1]["group"]:
            out.append(html.Tr(html.Td(
                [html.Span(r["group"], style={"color": TEXT, "fontWeight": "700"}),
                 html.Span(f" · {GROUP_NOTE[r['group']]}", style={"color": MUTED})]
                + _group_summary(ev, r["group"], have),
                colSpan=3 + 2 * len(have)), className="s3-group"))
        kind = "int" if r["kind"] == "int" else "rel"
        real_txt = "0 (distance)" if r["kind"] == "dist" else _fmt(r["real"], kind)
        tds = [html.Td(_md(r["label"], inline=True)),
               html.Td(real_txt, style={"color": MUTED2 if r["kind"] == "dist" else TEXT}),
               html.Td(_fmt(r["tol"], kind), style={"color": MUTED2})]
        for m in have:
            cell = r[m]
            tds += [html.Td(f"{_fmt(cell['mu'], kind)} ± {_fmt(cell['sd'], kind)}",
                            style={"color": METHOD_STYLE[m][1]}),
                    _verdict_cell(cell)]
        out.append(html.Tr(tds))
    return html.Table(out, style=_TABLE_STYLE, className="s3-table")


def _score_color(pct):
    return GREEN if pct >= 75 else (AMBER if pct >= 50 else RED)


def score_banner(ev, ens, n):
    items = []
    for m in ("ts", "gd"):
        if m in ev["score"]:
            sc = ev["score"][m]
            items.append((f"{sc['pct']:.0f}%", f"{METHOD_STYLE[m][0]} · dynamical fidelity "
                          f"({sc['n_match']}/{sc['n']} match)", _score_color(sc["pct"])))
    jumps = np.mean([i["jumps"] for i in ens["ts_info"]])
    breaks = np.mean([i["jumps"] + i["restarts"] for i in ens["ts_info"]])
    items.append((f"{jumps:,.0f}", "Twin jumps per realisation", TS_C))
    items.append((f"{n / (breaks + 1):.1f}", "Mean verbatim run (samples) · TS", MUTED2))
    return _banner(*items)


def identification_card(ident: dict, bundle, tm, delta_sel):
    label = LABELS.get(ident["col"], ident["col"])
    clamp_note = ""
    if ident["m_fnn"] < ident["m"]:
        clamp_note = (rf" FNN alone would pick $m = {ident['m_fnn']}$, which is not a phase space "
                      rf"at all (a single coordinate cannot unfold any dynamics), so the generators "
                      rf"use the minimum meaningful $m = {ident['m']}$.")
    ts_par = (
        rf"**Twin Surrogates.** The recurrence threshold $\varepsilon = {tm['eps']:.3f}$ (in $\sigma$ "
        rf"units, $\ell_\infty$ norm) fixes a recurrence rate of ${tm['rr']}\%$. With tolerance "
        rf"$\delta = {delta_sel:.2f}$, ${100 * tm['frac_with_twin']:.0f}\%$ of the "
        rf"{ident['n'] - (ident['m'] - 1) * ident['tau']:,} reconstructed states own at least one "
        rf"twin from a different passage (mean ${tm['mean_twins']:.1f}$ twins per state). The "
        rf"default $\delta^\ast = {ident['scan']['delta_default']:.2f}$ is the smallest tolerance "
        rf"giving at least ${100 * S3.TS_TWIN_FRAC:.0f}\%$ of states a twin."
    )
    children = [
        rf"For **{label}**, the average mutual information first reaches a local minimum at "
        rf"$\tau = {ident['tau']}$ samples, and the false-nearest-neighbours fraction first drops "
        rf"to 10 % or less at $m = {ident['m_fnn']}$.{clamp_note} Every "
        rf"reconstructed state is therefore $\mathbf{{X}}_i = (x_i, x_{{i+{ident['tau']}}}, \dots, "
        rf"x_{{i+{(ident['m'] - 1) * ident['tau']}}}) \in \mathbb{{R}}^{{{ident['m']}}}$.",
        ts_par,
        dcc.Graph(figure=fig_twin_scan(ident, delta_sel), config={"displayModeBar": False}),
    ]
    if bundle is not None:
        children += [
            rf"**GENDA.** One sample is the embedding window plus the next $H = \tau = "
            rf"{bundle['horizon']}$ samples ($L = {bundle['L']}$ values); {bundle['M']:,} samples, "
            rf"$k = {bundle['k']}$ distinct-passage neighbours each, latent dimension "
            rf"${S3.GENDA_LATENT}$, {bundle['epochs']} epochs of Adam ($\eta = 10^{{-3}}$). Every "
            rf"sample is reconstructed from its neighbours only — never from itself — with RMSE "
            rf"${bundle['rec_rmse']:.3g}$ {UNITS.get(ident['col'], '')} "
            rf"(${bundle['rec_rmse'] / ident['sd']:.2f}\,\sigma$): an approximation, not a replica.",
            dcc.Graph(figure=fig_genda_loss(bundle), config={"displayModeBar": False}),
        ]
    return _theory_card("Identifying the ingredients for THIS series", "IDENTIFICATION", GREEN,
                        children)


# ─────────────────────────────────────────────────────────────────────────────
#  THEORY PANEL
# ─────────────────────────────────────────────────────────────────────────────
SYNTH3_THEORY = html.Div([
    html.Div("📖 Class-III Synthetic Generation — Theory", className="theory-title"),

    _md(
        "A Class-2 generator guarantees the *statistics* — the marginal distribution and, at "
        "best, the linear autocorrelation. Two series can share both and still come from very "
        "different dynamical systems: a chaotic oscillator and its phase-randomised surrogate "
        "have identical spectra. A **Class-3** generator must reproduce the *dynamical "
        "potential* — the geometry of the attractor, how fast nearby trajectories separate, "
        "how often states recur and for how long — which is why the evidence on this tab is "
        "built from invariants of the reconstructed phase space rather than from moments. "
        "Neither generator below needs a parametric model of the system: both work directly "
        "on the trajectory that Takens' theorem lets us reconstruct from the single observed "
        "series.",
        className="section-desc theory-intro",
    ),

    html.Div([
        html.Div([
            _theory_card(
                "Phase-space reconstruction — τ from AMI, m from FNN", "STEP ①", ACCENT2,
                [
                    r"Each reconstructed state is a delay vector $\mathbf{X}_i = (x_i, x_{i+\tau}, "
                    r"\dots, x_{i+(m-1)\tau})$. The delay $\tau$ is the first local minimum of the "
                    r"average mutual information $I(\tau)$ between $x_t$ and $x_{t+\tau}$ (Fraser & "
                    r"Swinney, 1986): the first lag at which a new coordinate adds the most new "
                    r"information. Unlike the ACF, $I(\tau)$ measures nonlinear dependence too.",
                    r"The dimension $m$ is the smallest $d$ at which the fraction of false nearest "
                    r"neighbours (Kennel, Brown & Abarbanel, 1992) falls to 10 % or less. A neighbour "
                    r"$n(i)$ of $\mathbf{X}_i$ is *false* if adding the $(d+1)$-th coordinate pushes it "
                    r"disproportionately far away, i.e. it was only close because the attractor was "
                    r"projected onto too few dimensions.",
                ],
                [r"I(\tau) = \sum_{a,b} p_{ab}(\tau)\,\ln\frac{p_{ab}(\tau)}{p_a\,p_b},\qquad "
                 r"\tau^\ast = \min\{\tau : I(\tau-1) > I(\tau) \le I(\tau+1)\}",
                 r"\frac{\left|x_{i+d\tau} - x_{n(i)+d\tau}\right|}{\lVert \mathbf{X}^{(d)}_i - "
                 r"\mathbf{X}^{(d)}_{n(i)} \rVert} > R_{\mathrm{tol}} = 15 \quad\lor\quad "
                 r"\frac{\lVert \mathbf{X}^{(d+1)}_i - \mathbf{X}^{(d+1)}_{n(i)} \rVert}{\sigma_x} "
                 r"> A_{\mathrm{tol}} = 2"],
            ),
            _theory_card(
                "Twin Surrogates (Thiel et al., 2006)", "METHOD ①", TS_C,
                [
                    r"Two states are **twins** when their columns of the recurrence matrix "
                    r"coincide: they have exactly the same neighbourhood, so to resolution "
                    r"$\varepsilon$ they are the same state of the system, and their futures are "
                    r"interchangeable. A surrogate is a random walk on the observed attractor: from "
                    r"state $i_t$, choose $j$ uniformly among $i_t$ and its twins, then move to "
                    r"$j+1$. Each step is a real transition of the system, so the deterministic "
                    r"skeleton (and any synchronisation structure) is preserved, while the walk "
                    r"leaves the observed path at every twin and becomes a new trajectory. The "
                    r"scalar surrogate is the first delay coordinate of each visited state, so its "
                    r"values are real observations re-sequenced along the attractor.",
                    r"For continuous, densely sampled data exact column identity is almost never "
                    r"met (measured here: under 1 % of temperature states own an exact twin), and a "
                    r"walk without twins is just a shifted copy of the original. This tab therefore "
                    r"accepts twins up to a tolerance $\delta$ on the Jaccard overlap of the two "
                    r"neighbourhoods, and requires them to come from different passages "
                    r"($|i-j| > \tau$). $\delta = 0$ is Thiel's exact criterion. A **continuity "
                    r"gate** keeps only twins whose successor continues the current path, so every "
                    r"step of the surrogate is a realistic increment.",
                    r"**Self-avoiding walk.** Thiel's uniform choice among twins is restricted to "
                    r"twins whose successor has not been used yet. Without this, one realisation "
                    r"copied more than half of its material twice (about 2,000 distinct values out "
                    r"of 4,213), and those exact duplicates are spurious zero-distance recurrences "
                    r"that bias FNN and the recurrence measures. When the successor is already used, "
                    r"the walk moves to the nearest unused state within $\varepsilon$ of it. A "
                    r"restart at a *random* state would be an unphysical leap (measured: +21 °C in "
                    r"one temperature sample, while the largest real rise is +6.3 °C), and a few such "
                    r"leaps alone destroy the time-reversal asymmetry.",
                ],
                [r"R_{ij} = \Theta\!\left(\varepsilon - \lVert \mathbf{X}_i - \mathbf{X}_j "
                 r"\rVert_\infty\right),\qquad \text{twins (Thiel): } R_{ik} = R_{jk}\ \ \forall k",
                 r"J_{ij} = \frac{|\mathcal{N}_i \cap \mathcal{N}_j|}{|\mathcal{N}_i \cup "
                 r"\mathcal{N}_j|} \ge 1 - \delta,\quad |i-j| > \tau,\quad "
                 r"|x_{j+1} - x_{i+1}| \le \tfrac12\,\sigma_{\Delta x}",
                 r"j \sim \mathcal{U}\!\left(\{\,c \in \{i_t\} \cup \mathcal{T}(i_t) : c+1 "
                 r"\text{ unused}\,\}\right),\qquad i_{t+1} = j + 1,\qquad x^{\mathrm{TS}}_t = x_{i_t}"],
            ),
        ], className="theory-row"),
        html.Div([
            _theory_card(
                "GENDA (Troullinou et al., 2024)", "METHOD ②", GD_C,
                [
                    r"GENDA is a generative neighbourhood-based deep autoencoder. A shared "
                    r"convolutional encoder maps each of the $k$ nearest neighbours of a sample to a "
                    r"code $z_j$. The latent vector is a *random convex combination* of those codes, "
                    r"with weights drawn uniformly on the simplex, and the decoder must reconstruct "
                    r"$x_i$ from it. Because $x_i$ is never fed to its own encoder, the model learns "
                    r"the local geometry of the data, and with fresh weights it produces new "
                    r"samples that are approximations, not replicas. There is no Gaussian prior on "
                    r"the latent space, so multimodal, non-Gaussian structure survives.",
                    r"**Dynamical adaptation.** A sample is a piece of trajectory: one embedding "
                    r"window plus its next $H = \tau$ samples. Neighbours are the $k$ nearest "
                    r"*distinct passages* of the Takens state ($|i-j| > \tau$). Generation is an "
                    r"autonomous rollout. From the current synthetic state, encode its neighbours, "
                    r"mix, decode a new piece and append its future. The decoded future is "
                    r"(i) shifted by the seam offset $e_t = g_t - \hat x^{\,\mathrm{now}}$, decaying "
                    r"as $\lambda^h$ so the path is continuous, and (ii) forced by the "
                    r"reconstruction residual $\mathbf r_{j'}$ of a random nearby passage. An "
                    r"MSE-trained decoder outputs a conditional mean, and iterating it alone would "
                    r"lose the system's stochastic innovations and collapse onto a smoother "
                    r"attractor. The residual — GENDA's own approximation gap, measured locally — "
                    r"restores exactly that variability.",
                    r"**Amplitude adjustment.** A continuous decoder smears point masses and "
                    r"extremes of the marginal. Each rollout is therefore rank-remapped onto the real "
                    r"sample, $g'_t = y_{(\mathrm{rank}\,g_t)}$, the amplitude-adjustment step of "
                    r"AAFT/IAAFT. The map is strictly monotone, so the ordinal-pattern sequence is "
                    r"*unchanged exactly*, and $\lambda_1$ and $D_2$ are invariant under such a change "
                    r"of observable. Only the marginal is corrected: it becomes an exact permutation of "
                    r"the observed values, quantisation and ties included.",
                ],
                [r"z_j = \tanh(W h_j + b),\qquad \hat z_i = \sum_{j=1}^{k} u_j\, z_j,\quad "
                 r"\mathbf u \sim \mathcal{U}(\Delta^{k-1}),\qquad \hat x_i = \sigma(W' h + b')",
                 r"\mathcal{L} = \frac{1}{M}\sum_{i=1}^{M} \big\lVert x_i - "
                 r"d\big(e(\mathcal{N}(x_i))\big) \big\rVert^2",
                 r"g_{t+h} = \hat x^{\,\mathrm{fut}}_{h} + r^{\,\mathrm{fut}}_{j',h} + "
                 r"\lambda^{h}\, e_t,\qquad h = 1,\dots,H,\quad \lambda = 0.8",
                 r"g'_t = y_{(r_t)},\qquad r_t = \mathrm{rank}(g_t)\ \text{among}\ g_1,\dots,g_n"],
            ),
            _theory_card(
                "Dynamical invariants — the evidence", "VALIDATION ①", AMBER,
                [
                    r"All invariants are measured with the *real* series' $(m, \tau)$, on the "
                    r"standardised series, for the real data and for every realisation. "
                    r"**Re-identified** $\hat\tau, \hat m$: the synthetic series, analysed blind, must "
                    r"suggest the same embedding. **Largest Lyapunov exponent** $\lambda_1$ "
                    r"(Rosenstein et al., 1993): the mean exponential divergence rate of initially "
                    r"nearest neighbours, i.e. the horizon of predictability. **Correlation "
                    r"dimension** $D_2$ (Grassberger & Procaccia, 1983): the fractal dimension of the "
                    r"attractor, from the scaling of the correlation sum.",
                    r"**Sample entropy** (Richman & Moorman, 2000) and **permutation entropy** "
                    r"(Bandt & Pompe, 2002): rate of new-information generation and ordinal "
                    r"complexity. **DFA** $\alpha$ (Peng et al., 1994): long-range correlation "
                    r"scaling. **Time-reversal asymmetry** $A_3$: zero for every linear Gaussian "
                    r"process, so a matching non-zero $A_3$ (e.g. slow rise / fast fall) is direct "
                    r"evidence of preserved *nonlinear* dynamics. **RQA** at a fixed recurrence rate "
                    r"of 5 %: $DET$ (diagonal lines, deterministic structure) and $LAM$ (vertical "
                    r"lines, laminar states), averaged over five windows tiling the whole record. A "
                    r"single window describes one season only: the PV series' $LAM$ alone ranges "
                    r"0.62–0.79 from window to window.",
                ],
                [r"\langle \ln d_j(i) \rangle_j \simeq \ln C + \lambda_1\, i,\qquad "
                 r"C(r) = \frac{2}{N_p}\sum_{i<j,\ |i-j|>w} \Theta\!\left(r - \lVert \mathbf{X}_i - "
                 r"\mathbf{X}_j \rVert\right) \propto r^{D_2}",
                 r"\mathrm{SampEn} = -\ln\frac{A}{B},\qquad H_{\mathrm{PE}} = -\frac{1}{\ln d!}"
                 r"\sum_{\pi} p(\pi)\ln p(\pi),\qquad F(s) \propto s^{\alpha}",
                 r"A_3 = \frac{\left\langle (x_{t+1}-x_t)^3 \right\rangle}{\left\langle "
                 r"(x_{t+1}-x_t)^2 \right\rangle^{3/2}}"],
            ),
        ], className="theory-row"),
        html.Div([
            _theory_card(
                "Distances & calibrated verdicts", "VALIDATION ②", GREEN,
                [
                    r"Curve and distribution **distances** to the real series complement the "
                    r"invariants: Kolmogorov–Smirnov $D$ for the marginal, RMSE between the ACF "
                    r"curves and between the $RR_\tau$ curves, the log-spectral distance between Welch "
                    r"spectra, and the Jensen–Shannon divergence between ordinal-pattern "
                    r"distributions.",
                    r"**One scoreboard, five aspects.** Rows are grouped into linear dynamics (ACF, "
                    r"spectrum, DFA), the marginal, the phase space ($\hat\tau$, $\hat m$, $D_2$), "
                    r"nonlinear and chaotic dynamics ($\lambda_1$, SampEn, $A_3$, ordinal JSD) and "
                    r"recurrence structure ($DET$, $LAM$, $RR_\tau$). Redundant measures stay in the "
                    r"figures but not in the score, so no aspect is counted twice: $W_1$ repeats "
                    r"KS; permutation entropy summarises the ordinal distribution compared in full "
                    r"by JSD; $\langle L\rangle$ and $ENTR$ describe the same diagonal lines as "
                    r"$DET$; $TT$ the same vertical lines as $LAM$; and the AMI curve is summarised "
                    r"by $\hat\tau$.",
                    r"**Calibration.** A fixed threshold would be arbitrary, so each tolerance is "
                    r"set by the real series itself: the invariant is also computed on the first "
                    r"and second half of the record, and their difference measures natural "
                    r"variability (including non-stationarity). A synthetic ensemble whose mean "
                    r"deviates less than the real process deviates *from itself* is "
                    r"indistinguishable from it at this sample size. Verdicts: *match* if the "
                    r"deviation is within the tolerance, *close* within twice it, otherwise *off*. "
                    r"The dynamical-fidelity score credits 1 per match and ½ per close.",
                ],
                [r"D = \sup_x \left|F_{\mathrm{real}}(x) - F_{\mathrm{syn}}(x)\right|,\qquad "
                 r"\mathrm{LSD} = \sqrt{\Big\langle \big(10\log_{10} P_{\mathrm{real}}(f) - "
                 r"10\log_{10} P_{\mathrm{syn}}(f)\big)^2 \Big\rangle_f}",
                 r"\mathrm{JSD}(P \,\Vert\, Q) = \tfrac12 D_{\mathrm{KL}}(P \,\Vert\, M) + "
                 r"\tfrac12 D_{\mathrm{KL}}(Q \,\Vert\, M),\quad M = \tfrac12 (P+Q)",
                 r"\mathrm{tol}_k = \max\!\left(0.1\,|\theta_k^{\mathrm{real}}|,\ "
                 r"\left|\theta_k^{(1)} - \theta_k^{(2)}\right|\right),\qquad "
                 r"\text{match} \iff \left|\bar\theta_k^{\,\mathrm{syn}} - \theta_k^{\mathrm{real}}"
                 r"\right| \le \mathrm{tol}_k"],
            ),
            html.Div(id="s3-id-card", style={"display": "contents"}),
        ], className="theory-row"),
    ], className="theory-grid", style={"border": f"1px solid {BORDER}", "borderRadius": "10px",
                                       "overflow": "hidden", "background": CARD2}),

    _callout(
        "tip", "How to read this tab — a walkthrough for any level",
        "Work top to bottom; each element answers one question about *same dynamics*:",
        [
            ("Banner → ", r"the identified embedding $(m, \tau)$ and one fidelity score per "
             "method: the share of scoreboard rows judged *match* (½ credit for *close*)."),
            ("Stacked series → ", "do the synthetic paths have the same texture as the real one: "
             "the same daily pulses, ramps and quiet spells? They are *not* meant to line up "
             "with the calendar."),
            ("Delay portraits → ", r"the 2-D shadow $(x_t, x_{t+\tau})$ of the attractor. Same "
             "dynamics means the same shape, the same empty regions and the same density."),
            ("Deviation chart → ", "the one-glance verdict: every dot inside the green band "
             "deviates less than the real series deviates from itself."),
            ("Scoreboard → ", "the numbers behind the chart, grouped into linear dynamics, "
             "marginal, phase space, nonlinear dynamics and recurrence. Each group header counts "
             "the matches per method, and the tolerance column is the real series' own yardstick."),
            ("AMI / FNN curves → ", "synthetic data analysed blind must point to the same "
             r"$\tau$ and $m$, a direct check that the embedding was preserved."),
            ("Lyapunov, RR_τ, ACF/PSD, ordinal patterns, ECDF/QQ → ", "the full curves behind "
             "each scalar, so a match is never an accident of one summary number."),
        ],
    ),
    _callout(
        "note", "Limits of Class-3 fidelity",
        "Preserving the dynamics is a strong claim; these are its boundaries:",
        [
            ("Sampling: ", "spg.csv holds daylight hours only (about 9–10 rows per day), so the "
             "series' cycle is about 10 samples, not 24, and each night is a jump. Both generators "
             "reproduce the dynamics *of the series as sampled*, overnight jumps included. Lags "
             "here are in samples, not hours."),
            ("Twin Surrogates ", "re-use real values, so the marginal is exact up to sampling, but "
             r"between twin jumps they copy real segments verbatim (the banner reports the mean "
             r"run). They need twins: raising $\delta$ adds jumps (more novelty) at the price of "
             r"coarser equivalence, and at $\delta = 0$ continuous data often has almost none."),
            ("GENDA ", "is an autonomous model of a stationary system, so slow seasonal drift is "
             "represented only as far as the attractor already contains it. Its marginal is "
             "exact by construction (amplitude adjustment), so the KS row tests the Twin "
             "Surrogates' sampling rather than GENDA's decoder."),
            ("Sample size: ", "invariants such as $D_2$ and $\\lambda_1$ are estimated from about "
             "4,200 points. They are compared like-for-like with identical estimators, but are not "
             "precise absolute values of the true system."),
            ("Neither method ", "extrapolates beyond the observed attractor: regimes never visited "
             "in the record (e.g. an unprecedented heatwave) cannot be synthesised."),
        ],
        ordered=False,
    ),
], className="theory-section")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab_synth3 = dbc.Tab(
    label="🌀  Synthetic Data III",
    tab_id="tab-synth3",
    children=html.Div([
        _section_hdr("Class-III Synthetic Data Generation (dynamics-preserving)"),
        _section_desc(
            "Generates synthetic series that preserve the *dynamics* of any column of spg.csv, "
            "not just its statistics. The phase space is reconstructed with Takens' delay "
            r"embedding: delay $\tau$ from the first minimum of the average mutual information, "
            r"dimension $m$ from false nearest neighbours. Two generators then build new "
            "trajectories on the reconstructed attractor: **Twin Surrogates** (Thiel et al., "
            "2006) and **GENDA** (Troullinou et al., 2024). A battery of dynamical invariants, "
            "each calibrated by the real series' own split-half variability, is the evidence that "
            "the dynamics are the same. Identification and GENDA training run once in the "
            "background (the variable on screen first) and are cached to disk."
        ),

        dcc.Interval(id="s3-poll", interval=2000, n_intervals=0),
        dcc.Store(id="s3-ready"),
        html.Div(id="s3-progress"),

        html.Div([
            html.Div([
                html.Label("Target variable"),
                dcc.Dropdown(id="s3-var", options=VAR_OPTIONS, value=DEFAULT_COL,
                             clearable=False, style={"minWidth": "300px"}),
            ], className="ctrl-block"),
            html.Div([
                html.Label(f"Realisations per method (1–{N_REALS_MAX})"),
                dcc.Slider(id="s3-n-sims", min=1, max=N_REALS_MAX, step=1, value=N_REALS_DEF,
                           marks={i: str(i) for i in range(1, N_REALS_MAX + 1)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label("Random seed"),
                dcc.Input(id="s3-seed", type="number", value=42, min=0, step=1,
                          style={"width": "100px"}),
            ], className="ctrl-block"),
        ], className="controls-row"),

        html.Div([
            html.Div([
                html.Label(_md(r"TS · recurrence rate fixing $\varepsilon$ (%)", inline=True)),
                dcc.Slider(id="s3-rr", min=TS_RR_MIN, max=TS_RR_MAX, step=1, value=S3.TS_RR_DEFAULT,
                           marks={2: "2", 5: "5", 10: "10", 15: "15"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label(_md(r"TS · twin tolerance $\delta$ ($0$ = exact twins)", inline=True)),
                dcc.Slider(id="s3-delta", min=0.0, max=0.4, step=0.05, value=0.2,
                           marks={0: "0", 0.1: "0.1", 0.2: "0.2", 0.3: "0.3", 0.4: "0.4"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
            html.Div([
                html.Label(_md(rf"GENDA · neighbours mixed per step $k$ (trained with "
                               rf"$k = {S3.GENDA_K}$)", inline=True)),
                dcc.Slider(id="s3-k", min=K_MIN, max=K_MAX, step=1, value=S3.GENDA_K,
                           marks={i: str(i) for i in range(K_MIN, K_MAX + 1)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], className="ctrl-block-sm"),
        ], className="controls-row"),

        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div(id="s3-banner"),
            _plot_card([dcc.Graph(id="s3-series", config={"displaylogo": False})],
                       label="Real series vs one realisation of each Class-3 generator"),
            _plot_card([dcc.Graph(id="s3-portraits", config={"displaylogo": False})],
                       label=_md(r"Delay portraits $(x_t,\ x_{t+\tau})$ · the 2-D shadow of the "
                                 r"reconstructed attractor", inline=True)),
        ]),

        _section_hdr("Dynamical Validation: is it the same system?"),
        html.Div(id="s3-validation-desc", className="section-desc"),
        dcc.Loading(type="circle", color=ACCENT, children=[
            html.Div(id="s3-score"),
            _plot_card([dcc.Graph(id="s3-invariants", config={"displayModeBar": False})],
                       label="Deviation of every scoreboard measure / its tolerance · green band = "
                             "natural (split-half) variability of the real series"),
            _plot_card([html.Div(id="s3-table", style={"padding": "10px 6px"})],
                       label="Scoreboard · linear and nonlinear dynamics, real vs ensemble"),
            _plot_card([dcc.Graph(id="s3-embedding", config={"displayModeBar": False})],
                       label=_md(r"Embedding re-identification · $I(\tau)$ and FNN of real vs "
                                 r"synthetic (bands = ensemble min–max)", inline=True)),
            html.Div([
                html.Div(_plot_card([dcc.Graph(id="s3-lyap", config={"displayModeBar": False})],
                                    label=_md(r"Rosenstein divergence $\langle \ln d(i) \rangle$ · "
                                              r"slope = $\lambda_1$ (fit region shaded)",
                                              inline=True)),
                         style={"flex": "1", "minWidth": "420px"}),
                html.Div(_plot_card([dcc.Graph(id="s3-rrtau", config={"displayModeBar": False})],
                                    label=_md(r"$\tau$-recurrence rate $RR_\tau$ · the recurrence "
                                              r"analogue of the ACF", inline=True)),
                         style={"flex": "1", "minWidth": "420px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            _plot_card([dcc.Graph(id="s3-spectral", config={"displayModeBar": False})],
                       label="Linear memory · autocorrelation and power spectrum"),
            _plot_card([dcc.Graph(id="s3-ordinal", config={"displayModeBar": False})],
                       label="Ordinal-pattern (Bandt–Pompe) distribution · a fingerprint of "
                             "determinism"),
            _plot_card([dcc.Graph(id="s3-marginal", config={"displayModeBar": False})],
                       label="Marginal distribution · ECDF overlay and QQ plot"),
        ]),

        SYNTH3_THEORY,
    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  — register with your app via register_synth3_callbacks(app)
# ─────────────────────────────────────────────────────────────────────────────
def _progress_banner(col: str, st: dict):
    if st["finished"]:
        return None
    cur = st.get("current")
    stage = {"ident": "identifying", "genda": "training GENDA"}.get(st.get("stage"), "")
    items = [
        (f"{st['id_done']}/{st['total']}", "Variables identified", ACCENT2),
        (f"{st['gd_done']}/{st['total']}" if TORCH_OK else "off",
         "GENDA models trained" if TORCH_OK else "GENDA (PyTorch unavailable)", GD_C),
        (f"{LABELS.get(cur, cur)}" if cur else "—", f"Now {stage}" if cur else "Idle", MUTED2,
         {"fontSize": "14px", "wordBreak": "break-word", "lineHeight": "1.3"}),
    ]
    return _banner(*items)


def _ready_key(col, st):
    ident = get_ident(col)
    return "|".join([col, "id" if ident is not None else "-",
                     "gd" if get_genda(col) is not None else "-",
                     st["id_errors"].get(col, ""), st["gd_errors"].get(col, "")])


def _clean_inputs(n_sims, seed, rr, delta, k):
    n_sims = int(np.clip(int(n_sims or N_REALS_DEF), 1, N_REALS_MAX))
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 42
    rr = int(np.clip(int(rr or S3.TS_RR_DEFAULT), TS_RR_MIN, TS_RR_MAX))
    delta = round(float(delta if delta is not None else 0.2), 2)
    k = int(np.clip(int(k or S3.GENDA_K), K_MIN, K_MAX))
    return n_sims, seed, rr, delta, k


def _genda_wait_msg(col, st):
    if not TORCH_OK:
        return f"GENDA unavailable — PyTorch could not be loaded ({TORCH_ERR})"
    err = st["gd_errors"].get(col)
    if err:
        return f"GENDA training failed for {LABELS.get(col, col)} — {err}"
    return (f"GENDA is training on {LABELS.get(col, col)} in the background (~15 s); "
            "it will appear automatically.")


def register_synth3_callbacks(app):

    @app.callback(
        Output("s3-ready", "data"),
        Output("s3-progress", "children"),
        Output("s3-poll", "disabled"),
        Input("s3-poll", "n_intervals"),
        Input("s3-var", "value"),
        State("s3-ready", "data"),
    )
    def _poll(_tick, col, prev):
        """Refresh the progress banner; touch `s3-ready` only when the
        selected variable's readiness actually changes, so the heavy
        callbacks never re-run on a mere tick."""
        col = col or DEFAULT_COL
        _PRIORITY["col"] = col
        st = precompute_status()
        key = _ready_key(col, st)
        return (key if key != prev else dash.no_update), _progress_banner(col, st), st["finished"]

    @app.callback(
        Output("s3-delta", "value"),
        Input("s3-var", "value"),
        Input("s3-ready", "data"),
    )
    def _default_delta(col, _ready):
        """Snap the twin tolerance to the variable's δ* once it is identified."""
        ident = get_ident(col or DEFAULT_COL)
        return ident["scan"]["delta_default"] if ident else dash.no_update

    @app.callback(
        Output("s3-banner", "children"),
        Output("s3-series", "figure"),
        Output("s3-portraits", "figure"),
        Output("s3-id-card", "children"),
        Input("s3-var", "value"),
        Input("s3-n-sims", "value"),
        Input("s3-seed", "value"),
        Input("s3-rr", "value"),
        Input("s3-delta", "value"),
        Input("s3-k", "value"),
        Input("s3-ready", "data"),
    )
    def update_generation(col, n_sims, seed, rr, delta, k, _ready):
        col = col or DEFAULT_COL
        ident = get_ident(col)
        st = precompute_status()
        if ident is None:
            err = st["id_errors"].get(col)
            msg = (f"Could not identify {LABELS.get(col, col)} — {err}" if err else
                   f"Identifying the phase space of {LABELS.get(col, col)} in the background "
                   f"({st['id_done']}/{st['total']} done) — it will appear automatically.")
            ph = _placeholder_fig(msg)
            return None, ph, ph, None
        n_sims, seed, rr, delta, k = _clean_inputs(n_sims, seed, rr, delta, k)
        ens = ensemble(col, n_sims, seed, rr, delta, k)
        bundle = get_genda(col)
        tm = ens["tm"]
        if bundle is not None:
            gd_state, gd_c = "ready", GD_C
        elif not TORCH_OK or col in st["gd_errors"]:
            gd_state, gd_c = "unavailable", RED
        else:
            gd_state, gd_c = "training…", AMBER
        banner = _banner(
            (f"m = {ident['m']}", "Embedding dimension (FNN)", ACCENT2),
            (f"τ = {ident['tau']}", "Delay (AMI first minimum)", ACCENT2),
            (f"{100 * tm['frac_with_twin']:.0f}%", f"States with a twin (δ = {delta:.2f})", TS_C),
            (f"{ident['n']:,}", "Samples per realisation", MUTED2),
            (gd_state, "GENDA model", gd_c),
        )
        return (banner, fig_series(ident, ens), fig_portraits(ident, ens),
                identification_card(ident, bundle, tm, delta))

    @app.callback(
        Output("s3-validation-desc", "children"),
        Output("s3-score", "children"),
        Output("s3-invariants", "figure"),
        Output("s3-table", "children"),
        Output("s3-embedding", "figure"),
        Output("s3-lyap", "figure"),
        Output("s3-rrtau", "figure"),
        Output("s3-spectral", "figure"),
        Output("s3-ordinal", "figure"),
        Output("s3-marginal", "figure"),
        Input("s3-var", "value"),
        Input("s3-n-sims", "value"),
        Input("s3-seed", "value"),
        Input("s3-rr", "value"),
        Input("s3-delta", "value"),
        Input("s3-k", "value"),
        Input("s3-ready", "data"),
    )
    def update_validation(col, n_sims, seed, rr, delta, k, _ready):
        col = col or DEFAULT_COL
        ident = get_ident(col)
        if ident is None:
            ph = _placeholder_fig("Waiting for the phase-space identification…")
            return "", None, ph, None, ph, ph, ph, ph, ph, ph
        n_sims, seed, rr, delta, k = _clean_inputs(n_sims, seed, rr, delta, k)
        ens = ensemble(col, n_sims, seed, rr, delta, k)
        profs = profiles(col, n_sims, seed, rr, delta, k)
        ev = evaluate(ident, profs)
        st = precompute_status()
        gd_note = "" if ens["gd"] is not None else f" ({_genda_wait_msg(col, st)})"
        desc = _md(
            f"Ensemble of {n_sims} realisation(s) per method, each {ident['n']:,} samples long "
            f"(seed {seed}). Every invariant is computed blind on each realisation with the real "
            rf"series' embedding $(m, \tau) = ({ident['m']}, {ident['tau']})$ and compared with the "
            "real value. Each tolerance is the larger of 10 % and the real series' own split-half "
            f"spread.{gd_note}"
        )
        return (
            desc,
            score_banner(ev, ens, ident["n"]),
            fig_invariants(ev),
            scoreboard_table(ev),
            fig_embedding(ident, profs),
            fig_lyapunov(ident, profs),
            fig_rrtau(ident, profs),
            fig_spectral(ident, profs),
            fig_ordinal(ident, profs),
            fig_marginal(ident, ens),
        )
