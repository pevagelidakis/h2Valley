"""Class-III synthetic-data toolkit: Twin Surrogates (Thiel et al., 2006), a
phase-space adaptation of GENDA (Troullinou et al., 2024) and the
dynamical-invariant battery used to validate both. Used by synth3_tab.py —
all Synthetic-Data-III maths lives here; the module is pure (no Dash).

Embedding parameters come from the same estimators as the RQA tab, so the
(m, τ) shown in both tabs always agree:
    τ  first minimum of the average mutual information (Fraser & Swinney, 1986)
    m  false-nearest-neighbours fraction ≤ threshold (Kennel et al., 1992)
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from statsmodels.tsa.stattools import acf as sm_acf

from core.dataset import get_dataset
from rqa_funs import embed, distance_matrix, epsilon_for_rr, recurrence_matrix, \
    rqa_measures, tau_recurrence_rate
from rqa_tab import _ami, _first_minimum, _fnn_curve, _select_dim, \
    AMI_MAX_LAG, AMI_BINS, FNN_MAX_DIM, FNN_RTOL, FNN_ATOL

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_DS = get_dataset()        # lags below are multiples of its dominant cycle (24 for spg.csv)
TAU_BOUNDS = (1, _DS.lag(1))
M_BOUNDS   = (2, 8)        # m = 1 is no phase space at all, so synthesis needs m ≥ 2

# Twin surrogates
TS_RR_DEFAULT = 5          # recurrence rate (%) that fixes ε
TS_DELTA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
TS_TWIN_FRAC  = 0.25       # default δ: smallest tolerance giving ≥ 25 % of states a twin —
                           # calibrated on spg.csv (self-avoiding, gated walk): fidelity ≥ 92 %
                           # on every tested variable with 600–1,500 twin jumps per realisation
TS_BLOCK      = 1024       # rows per block of the neighbourhood-overlap (Gram) matrix
TS_GATE       = 0.5        # continuity gate: |x_{j+1} − x_{i+1}| ≤ TS_GATE · sd(Δx)

# GENDA (Troullinou et al., 2024, §IV-B3 time-series configuration)
GENDA_K       = 2          # neighbours per sample — the paper's optimum (Table VI)
GENDA_LATENT  = 32         # latent dimension for time series
GENDA_EPOCHS  = 40
GENDA_LR      = 1e-3
GENDA_BATCH   = 64
GENDA_FILTERS = (16, 32, 64)
GENDA_LAMBDA  = 0.8        # per-sample decay of the seam offset during rollout
GENDA_POOL    = 8          # nearest passages the residual forcing is drawn from

# Validation battery
RQA_WINDOW    = 1000       # samples per RQA window …
RQA_N_WINDOWS = 5          # … and windows tiling the whole record (averaged)
RQA_RR        = 0.05       # fixed recurrence rate → DET/LAM/... comparable across series
RR_TAU_MAX    = _DS.lag(3)  # τ-recurrence-rate curve up to three cycles
D2_SAMPLE     = 1500       # points used for the correlation sum
ACF_LAGS      = _DS.lag(2)
LYAP_HORIZON  = 4          # divergence curve followed for LYAP_HORIZON·τ steps


# ─────────────────────────────────────────────────────────────────────────────
#  IDENTIFICATION  —  τ from AMI, m from FNN
# ─────────────────────────────────────────────────────────────────────────────
def zscore(y):
    y = np.asarray(y, dtype=float)
    sd = y.std()
    return (y - y.mean()) / (sd if sd > 1e-12 else 1.0)


def identify_embedding(y, max_dim: int = FNN_MAX_DIM) -> dict:
    """(τ, m) of a scalar series with the RQA tab's estimators. `m_fnn` is
    the raw FNN choice; `m` is clamped to M_BOUNDS for synthesis."""
    z = zscore(y)
    ami_curve = _ami(z, AMI_MAX_LAG, AMI_BINS)
    tau = int(np.clip(_first_minimum(ami_curve), *TAU_BOUNDS))
    fnn_curve = _fnn_curve(z, tau, max_dim, FNN_RTOL, FNN_ATOL)
    m_fnn = int(_select_dim(fnn_curve))
    return dict(tau=tau, m_fnn=m_fnn, m=int(np.clip(m_fnn, *M_BOUNDS)),
                ami_curve=np.asarray(ami_curve), fnn_curve=np.asarray(fnn_curve, dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
#  TWIN SURROGATES  (Thiel, Romano, Kurths, Rolfs & Kliegl, 2006)
#  Two states are twins when their recurrence-matrix columns coincide,
#  R_{·i} = R_{·j}: same neighbourhood, hence (to resolution ε) the same state
#  of the underlying system, so their futures are interchangeable.
#
#  For continuous, densely sampled data exact column identity is almost never
#  met (measured on spg.csv: < 1 % of temperature states own an exact twin),
#  and a walk without twins is a shifted copy of the original. Twins are
#  therefore defined with a tolerance on the neighbourhood overlap,
#      J_ij = |N_i ∩ N_j| / |N_i ∪ N_j| ≥ 1 − δ,   |i − j| > τ,
#  where δ = 0 recovers Thiel's exact criterion and the Theiler condition
#  removes trivial twins from the same passage (which would only skip a sample).
# ─────────────────────────────────────────────────────────────────────────────
def sup_distance_matrix(X):
    """ℓ∞ distance matrix in float32 (half the memory of scipy's float64)."""
    X = np.asarray(X, dtype=np.float32)
    D = np.abs(X[:, None, 0] - X[None, :, 0])
    for k in range(1, X.shape[1]):
        np.maximum(D, np.abs(X[:, None, k] - X[None, :, k]), out=D)
    return D


def _recurrence(X, rr_pct):
    """Recurrence matrix (ℓ∞) at a fixed recurrence rate, and its ε."""
    D = sup_distance_matrix(X)
    iu = np.triu_indices(len(X), k=1)
    eps = float(np.quantile(D[iu], rr_pct / 100.0))
    del iu
    return D <= eps, eps


def _jaccard_blocks(R, block=TS_BLOCK):
    """Yield (row offset, Jaccard block) of the neighbourhood-overlap matrix
    J_ij = |N_i ∩ N_j| / |N_i ∪ N_j|, a block of rows at a time (R is
    symmetric, so R R gives the intersection counts)."""
    Rf = R.astype(np.float32)
    n = Rf.sum(0)
    for s in range(0, len(Rf), block):
        G = Rf[s:s + block] @ Rf
        yield s, G / (n[s:s + block, None] + n[None, :] - G)


def _theiler_far(s, b, N, w):
    rows = np.arange(s, s + b)[:, None]
    return np.abs(rows - np.arange(N)[None, :]) > w


def _continuity_ok(z, s, b, N, gate):
    """Eligible jumps only: the twin's successor must continue the current
    path, |z_{j+1} − z_{i+1}| ≤ gate · sd(Δz). Without it a twin at a coarse
    ε can turn one step of the surrogate into an unphysical leap; those
    symmetric leaps dilute time-reversal asymmetry and short-lag memory."""
    if gate is None:
        return True
    nxt = np.append(z[1:N + 1], z[N]) if len(z) > N else np.append(z[1:N], z[N - 1])
    lim = gate * np.std(np.diff(z))
    return np.abs(nxt[s:s + b, None] - nxt[None, :N]) <= lim


def twin_scan(z, m, tau, rr_pct=TS_RR_DEFAULT, deltas=TS_DELTA_GRID, gate=TS_GATE) -> dict:
    """Fraction of states owning ≥ 1 twin, and mean twins per state, as a
    function of the tolerance δ — the curve the default δ is read from."""
    X = embed(z, m, tau)
    N = len(X)
    R, eps = _recurrence(X, rr_pct)
    deltas = np.asarray(deltas)
    has = np.zeros((len(deltas), N), dtype=bool)
    cnt = np.zeros((len(deltas), N))
    for s, J in _jaccard_blocks(R):
        far = _theiler_far(s, len(J), N, tau) & _continuity_ok(z, s, len(J), N, gate)
        for d_i, d in enumerate(deltas):
            T = (J >= 1.0 - d - 1e-9) & far
            has[d_i, s:s + len(J)] = T.any(1)
            cnt[d_i, s:s + len(J)] = T.sum(1)
    frac = has.mean(1)
    ok = np.where(frac >= TS_TWIN_FRAC)[0]
    d_default = float(deltas[ok[0]] if len(ok) else deltas[-1])
    return dict(deltas=deltas, frac=frac, mean_twins=cnt.mean(1), rr=rr_pct, eps=eps,
                delta_default=d_default)


def twin_model(z, m, tau, rr_pct, delta, gate=TS_GATE) -> dict:
    """Everything the twin walk needs at one (ε, δ): each state's twin list."""
    X = embed(z, m, tau)
    N = len(X)
    R, eps = _recurrence(X, rr_pct)
    twins = []
    for s, J in _jaccard_blocks(R):
        T = (J >= 1.0 - delta - 1e-9) & _theiler_far(s, len(J), N, tau)             & _continuity_ok(z, s, len(J), N, gate)
        twins.extend(np.flatnonzero(row) for row in T)
    n_tw = np.array([len(t) for t in twins])
    return dict(N=N, eps=eps, rr=rr_pct, delta=delta, twins=twins, X=X, tau=tau,
                frac_with_twin=float((n_tw > 0).mean()), mean_twins=float(n_tw.mean()))


def twin_surrogate(y, tm: dict, n_out: int, rng: np.random.Generator,
                   self_avoiding: bool = True, k_restart: int = 64):
    """One twin surrogate of length n_out (Thiel et al., 2006, §II):
       1. start at a random state i;
       2. pick j uniformly among {i} ∪ twins(i);  3. move to j + 1;  4. repeat.
    The scalar output is the first delay coordinate of each visited state,
    so every value is a real observation, re-sequenced along the attractor.

    self_avoiding: the uniform choice in step 2 is restricted to candidates
    whose successor has not been visited yet (falling back to all candidates
    when none is left). Without it a walk of length N over N states copies
    more than half of its material twice (measured: ~2,000 distinct values
    out of 4,213); those exact duplicates are spurious zero-distance
    recurrences that bias FNN, LAM, TT and RR_τ.

    When the successor is already used, the walk resumes at the nearest
    *unvisited* state to it — but only if that state is recurrence-close
    (ℓ∞ distance ≤ ε); otherwise it accepts a short verbatim repeat. At the
    end of the record it moves to the nearest state of another passage. A
    restart at a random state would be an unphysical leap (measured: +21 °C
    in one temperature sample, against a largest real rise of +6.3 °C), and a
    few such leaps alone distort time-reversal asymmetry and short-lag memory."""
    N, twins, X, eps = tm["N"], tm["twins"], tm["X"], tm["eps"]
    tree = cKDTree(X)
    idx = np.empty(n_out, dtype=int)
    visited = np.zeros(N, dtype=bool)

    def _nearest(target, unvisited_only, max_dist=np.inf, exclude_near=None):
        dist, cand = tree.query(X[target], k=min(k_restart, N), p=np.inf)
        for dd, c in zip(np.atleast_1d(dist), np.atleast_1d(cand)):
            if dd > max_dist:
                break
            if c >= N - 1 or (unvisited_only and visited[c]):
                continue
            if exclude_near is not None and abs(c - exclude_near) <= exclude_near_w:
                continue
            return int(c)
        return None

    exclude_near_w = int(tm.get("tau", 1))
    i = int(rng.integers(N - 1))
    jumps = restarts = 0
    for t in range(n_out):
        idx[t] = i
        visited[i] = True
        tw = twins[i]
        cand = np.concatenate(([i], tw[tw < N - 1])) if len(tw) else np.array([i])
        if self_avoiding and len(cand) > 1:
            free = cand[~visited[cand + 1]]
            if len(free):
                cand = free
        j = int(cand[rng.integers(len(cand))])
        jumps += j != i
        nxt = j + 1
        if nxt >= N - 1:                                # end of record: hop to another passage
            alt = _nearest(nxt, self_avoiding, exclude_near=nxt)
            if alt is None:
                alt = _nearest(nxt, False, exclude_near=nxt)
            nxt = alt if alt is not None else int(rng.integers(N - 1))
            restarts += 1
        elif self_avoiding and visited[nxt]:            # successor used: recurrence-close hop
            alt = _nearest(nxt, True, max_dist=eps)
            if alt is not None:
                nxt = alt
                restarts += 1
        i = nxt
    return np.asarray(y, dtype=float)[idx], dict(jumps=jumps, restarts=restarts)


# ─────────────────────────────────────────────────────────────────────────────
#  GENDA  (Troullinou, Tsagkatakis, Losonczy, Poirazi & Tsakalides, 2024)
#  Generative neighbourhood-based deep autoencoder, adapted to dynamics:
#    sample      x_i = (y_i, …, y_{i+W-1}, y_{i+W}, …, y_{i+W+H-1}) — one
#                embedding window (W = (m-1)τ + 1) plus its next H samples;
#    neighbours  N(x_i) = k nearest *distinct passages* (|i - j| > τ) of the
#                Takens state X_i = (y_i, y_{i+τ}, …, y_{i+(m-1)τ});
#    encoder     z_j = tanh(W h_j + b), one shared ConvNet per neighbour  (eq. 2)
#    latent      ẑ_i = Σ_j u_j z_j,  u ~ U(simplex)                         (eq. 3)
#    decoder     x̂_i = σ(W' h + b'), transpose-ConvNet                      (eq. 4)
#    loss        L = 1/M Σ ‖x_i − d(e(N(x_i)))‖²                            (eq. 6)
#  Generation is an autonomous phase-space rollout in blocks of H = τ: from
#  the current synthetic state, take its k distinct-passage neighbours, mix
#  their codes with fresh random u, decode a new trajectory piece, force it
#  with a local reconstruction residual (see genda_generate) and append its
#  H-sample future. The decoded piece is a mixture of neighbouring passages,
#  so its present value differs from the synthetic present by some offset e;
#  the future is shifted by λ^h·e (h = 1…H) so the path stays continuous at
#  the seam while the offset decays back onto the learned flow.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    from torch import nn
    TORCH_OK = True
    TORCH_ERR = ""
except Exception as _e:                             # pragma: no cover
    # On Windows torch must initialise before pandas/statsmodels load their
    # DLLs (WinError 1114) — app.py therefore imports torch first.
    torch = nn = None
    TORCH_OK = False
    TORCH_ERR = f"{type(_e).__name__}: {_e}"


def _conv_lengths(L):
    lengths = [L]
    for _ in GENDA_FILTERS:
        lengths.append((lengths[-1] + 1) // 2 if lengths[-1] >= 4 else lengths[-1])
    return lengths


def _build_nets(L, d=GENDA_LATENT):
    lengths = _conv_lengths(L)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ch = [], 1
            for out, l_in, l_out in zip(GENDA_FILTERS, lengths[:-1], lengths[1:]):
                layers.append(nn.Conv1d(ch, out, 3, padding=1))
                if l_out != l_in:
                    layers.append(nn.AvgPool1d(2, ceil_mode=True))
                layers.append(nn.Tanh())
                ch = out
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(GENDA_FILTERS[-1] * lengths[-1], d)

        def forward(self, x):                           # (B, L) -> (B, d)
            return torch.tanh(self.fc(self.conv(x.unsqueeze(1)).flatten(1)))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            rev_f = GENDA_FILTERS[::-1]                 # 64, 32, 16
            self.l0 = lengths[-1]
            self.fc = nn.Linear(d, rev_f[0] * self.l0)
            layers, ch = [], rev_f[0]
            for out, l_out in zip(rev_f[1:], lengths[-2:0:-1]):
                layers += [nn.Upsample(size=l_out), nn.ConvTranspose1d(ch, out, 3, padding=1),
                           nn.LeakyReLU(0.2)]
                ch = out
            layers += [nn.Upsample(size=L), nn.ConvTranspose1d(ch, 1, 3, padding=1)]
            self.deconv = nn.Sequential(*layers)

        def forward(self, z):                           # (B, d) -> (B, L)
            h = self.fc(z).view(len(z), -1, self.l0)
            return torch.sigmoid(self.deconv(h)).squeeze(1)

    return Encoder(), Decoder()


def _segments(u, m, tau, horizon):
    """Trajectory pieces (embedding window + H-sample future) and their states."""
    W = (m - 1) * tau + 1
    L = W + horizon
    S = np.lib.stride_tricks.sliding_window_view(u, L).copy()
    return S, S[:, 0:W:tau]


def distinct_neighbours(tree, Q, k, sep, exclude=None, k_cand=None):
    """k nearest neighbours of each query state, greedily constrained to be
    mutually more than `sep` samples apart (and from `exclude[i]`), so each
    belongs to a different passage of the trajectory through the region."""
    n_ref = tree.n
    k_cand = int(min(n_ref, k_cand or (2 * sep + 4 * k + 1)))
    _, cand = tree.query(Q, k=k_cand)
    cand = np.atleast_2d(cand)
    out = np.empty((len(Q), k), dtype=int)
    for r, row in enumerate(cand):
        chosen = []
        ex = None if exclude is None else exclude[r]
        for c in row:
            if ex is not None and abs(c - ex) <= sep:
                continue
            if all(abs(c - o) > sep for o in chosen):
                chosen.append(int(c))
                if len(chosen) == k:
                    break
        for c in row:                                   # rare fallback: relax diversity
            if len(chosen) == k:
                break
            if c not in chosen and (ex is None or c != ex):
                chosen.append(int(c))
        out[r] = chosen[:k]
    return out


def genda_train(y, m, tau, horizon=None, k=GENDA_K, epochs=GENDA_EPOCHS, seed=0,
                n_threads=2) -> dict:
    """Train GENDA on one series; returns a picklable model bundle."""
    if not TORCH_OK:
        raise RuntimeError(f"PyTorch is unavailable — GENDA needs it ({TORCH_ERR})")
    horizon = int(horizon or tau)
    torch.set_num_threads(max(1, int(n_threads)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    y = np.asarray(y, dtype=float)
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo if hi - lo > 1e-12 else 1.0
    u = 0.05 + 0.9 * (y - lo) / span                    # [0.05, 0.95] — inside the sigmoid's range
    S, X = _segments(u, m, tau, horizon)
    M, L = S.shape
    NB = distinct_neighbours(cKDTree(X), X, k, sep=tau, exclude=np.arange(M))

    enc, dec = _build_nets(L)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=GENDA_LR)
    S_t = torch.tensor(S, dtype=torch.float32)
    NB_t = torch.tensor(NB, dtype=torch.long)
    dirich = torch.distributions.Dirichlet(torch.ones(k))
    losses = []
    for _ in range(epochs):
        perm = torch.tensor(rng.permutation(M), dtype=torch.long)
        tot = 0.0
        for b in range(0, M, GENDA_BATCH):
            idx = perm[b:b + GENDA_BATCH]
            z = enc(S_t[NB_t[idx].reshape(-1)]).view(len(idx), k, -1)
            w = dirich.sample((len(idx),)).unsqueeze(-1)
            loss = torch.mean((dec((w * z).sum(1)) - S_t[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        losses.append(tot / M)

    with torch.no_grad():
        Z = enc(S_t).numpy()
        # every sample reconstructed from its neighbours only (it never sees itself)
        rec = dec(torch.tensor(Z[NB].mean(1), dtype=torch.float32)).numpy()
    rec_rmse = float(np.sqrt(np.mean((rec - S) ** 2)) * span / 0.9)

    return dict(m=m, tau=tau, k=k, horizon=horizon, L=L, lo=lo, span=span,
                u=u.astype(np.float32), X=X, Z=Z.astype(np.float32), NB=NB,
                resid=(S - rec).astype(np.float32),
                dec_state={kk: v.detach().cpu().numpy() for kk, v in dec.state_dict().items()},
                losses=np.asarray(losses), rec_rmse=rec_rmse, M=M, epochs=epochs)


def _load_decoder(bundle):
    _, dec = _build_nets(bundle["L"])
    dec.load_state_dict({k: torch.tensor(v) for k, v in bundle["dec_state"].items()})
    dec.eval()
    return dec


def amplitude_adjust(x, y_sorted):
    """Rank-remap x onto the real sample, x' = sort(y)[rank(x)] — the
    amplitude-adjustment step of AAFT/IAAFT (Schreiber & Schmitz, 1996).
    The result is an exact permutation of the observed values (ties and
    quantisation included). The map is strictly monotone, so the ordinal-
    pattern sequence is unchanged exactly, and λ₁ and D₂ are invariant under
    such a change of observable: only the marginal is corrected."""
    ranks = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return np.asarray(y_sorted)[ranks]


def genda_generate(bundle: dict, n_out: int, n_sims: int, seed: int,
                   k: int | None = None, lam: float = GENDA_LAMBDA, n_pool: int = GENDA_POOL,
                   y_ref=None):
    """(n_out, n_sims) autonomous GENDA rollouts in the original units;
    with `y_ref` (the real series, n_out == len(y_ref)) each rollout is
    amplitude-adjusted onto the real sample (see amplitude_adjust).

    Each block: the k distinct-passage neighbours of the current *synthetic*
    state are encoded, mixed with fresh u ~ U(simplex) and decoded; the
    decoded future is forced with the reconstruction residual r_j = x_j − x̂_j
    of a random state among the `n_pool` nearest passages. An MSE-trained
    decoder returns a conditional mean, so iterating it alone loses the
    stochastic innovations and collapses onto a smoother attractor; the
    residual is GENDA's own "approximation, not replica" gap, measured
    locally, and restores exactly that missing variability.
    """
    if not TORCH_OK:
        raise RuntimeError(f"PyTorch is unavailable — GENDA needs it ({TORCH_ERR})")
    m, tau, H = bundle["m"], bundle["tau"], bundle["horizon"]
    k = int(k or bundle["k"])
    X, Z, u, resid = bundle["X"], bundle["Z"], bundle["u"], bundle["resid"]
    tree = cKDTree(X)
    dec = _load_decoder(bundle)
    rng = np.random.default_rng(seed)

    W = (m - 1) * tau + 1                               # samples spanned by one state
    n_blocks = int(np.ceil(n_out / H))
    G = np.empty((n_sims, W + n_blocks * H))
    # initial condition: one real embedding window from a random passage;
    # it is discarded from the output, so every returned value is generated
    for s, s0 in enumerate(rng.integers(0, len(u) - W, n_sims)):
        G[s, :W] = u[s0:s0 + W]
    sel = np.arange(0, W, tau)
    decay = float(lam) ** np.arange(1, H + 1)
    rows = np.arange(n_sims)
    with torch.no_grad():
        for b in range(n_blocks):
            t = b * H                                   # G[:, t:t+W] is the current window
            pool = distinct_neighbours(tree, G[:, t + sel], max(k, n_pool), sep=tau)
            nb = pool[:, :k]
            forcing = resid[pool[rows, rng.integers(0, pool.shape[1], n_sims)], W:]
            w = rng.dirichlet(np.ones(k), size=n_sims)[..., None]
            seg = dec(torch.tensor((w * Z[nb]).sum(1), dtype=torch.float32)).numpy()
            offset = G[:, t + W - 1] - seg[:, W - 1]
            G[:, t + W:t + W + H] = seg[:, W:] + forcing + offset[:, None] * decay[None, :]
    out = (G[:, W:W + n_out] - 0.05) / 0.9 * bundle["span"] + bundle["lo"]
    out = np.clip(out, bundle["lo"], bundle["lo"] + bundle["span"])
    if y_ref is not None:                               # amplitude adjustment onto the real sample
        y_sorted = np.sort(np.asarray(y_ref, dtype=float))
        out = np.stack([amplitude_adjust(o, y_sorted) for o in out])
    return out.T


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMICAL-INVARIANT BATTERY
# ─────────────────────────────────────────────────────────────────────────────
def largest_lyapunov(z, m, tau, theiler=None, horizon=None):
    """Rosenstein, Collins & De Luca (1993): mean log-divergence of initially
    nearest neighbours (outside a Theiler window); λ₁ is the slope of the
    initial linear region, here fitted over the first τ steps (per hour)."""
    X = embed(z, m, tau)
    theiler = int(theiler or (m - 1) * tau + 1)
    horizon = int(horizon or LYAP_HORIZON * tau)
    N = len(X) - horizon
    if N < 50:
        return np.nan, np.full(horizon + 1, np.nan)
    tree = cKDTree(X[:N])
    kq = min(N, 2 * theiler + 3)
    dist, idx = tree.query(X[:N], k=kq)
    i_all = np.arange(N)[:, None]
    ok = (np.abs(idx - i_all) > theiler) & (dist > 0)
    first = np.argmax(ok, axis=1)
    has = ok[np.arange(N), first]
    i_src = np.arange(N)[has]
    j_nn = idx[has, first[has]]
    curve = np.empty(horizon + 1)
    for h in range(horizon + 1):
        d = np.linalg.norm(X[i_src + h] - X[j_nn + h], axis=1)
        d = d[d > 0]
        curve[h] = np.mean(np.log(d)) if len(d) else np.nan
    fit_to = max(2, tau)
    hh = np.arange(fit_to + 1)
    good = np.isfinite(curve[:fit_to + 1])
    lam = float(np.polyfit(hh[good], curve[:fit_to + 1][good], 1)[0]) if good.sum() >= 2 else np.nan
    return lam, curve


def correlation_dimension(z, m, tau, theiler=None, n_sample=D2_SAMPLE, seed=0):
    """Grassberger–Procaccia D₂: slope of log C(r) vs log r over the range
    10⁻³ ≤ C(r) ≤ 10⁻¹, Theiler-corrected, on a random subsample of states."""
    X = embed(z, m, tau)
    theiler = int(theiler or (m - 1) * tau + 1)
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(X), size=min(n_sample, len(X)), replace=False))
    d = pdist(X[sel], metric="chebyshev")
    ii, jj = np.triu_indices(len(sel), k=1)
    d = d[np.abs(sel[ii] - sel[jj]) > theiler]
    d = np.sort(d[d > 0])
    if len(d) < 100:
        return np.nan
    C = np.arange(1, len(d) + 1) / len(d)
    keep = (C >= 1e-3) & (C <= 1e-1)
    if keep.sum() < 10:
        return np.nan
    r_log = np.log(d[keep])
    pick = np.linspace(0, keep.sum() - 1, 60).astype(int)
    return float(np.polyfit(r_log[pick], np.log(C[keep])[pick], 1)[0])


def sample_entropy(z, order=2, r=0.2):
    """SampEn(m=2, r=0.2σ) (Richman & Moorman, 2000) with the ℓ∞ norm."""
    z = zscore(z)
    n = len(z)
    def _count(mm):
        T = np.lib.stride_tricks.sliding_window_view(z, mm)[:n - order]
        tree = cKDTree(T)
        return (tree.count_neighbors(tree, r, p=np.inf) - len(T)) / 2
    B, A = _count(order), _count(order + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan


def ordinal_distribution(z, order, delay):
    """Bandt–Pompe ordinal-pattern probabilities (all order! patterns)."""
    X = embed(z, order, delay)
    ranks = np.argsort(np.argsort(X, axis=1, kind="stable"), axis=1, kind="stable")
    codes = (ranks * order ** np.arange(order)).sum(1)
    perms = np.array([(np.array(p) * order ** np.arange(order)).sum()
                      for p in __import__("itertools").permutations(range(order))])
    counts = np.array([(codes == c).sum() for c in perms], dtype=float)
    return counts / counts.sum()


def permutation_entropy(p):
    p = np.asarray(p)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)) / np.log(len(p)))


def js_divergence(p, q):
    """Jensen–Shannon divergence (base 2, ∈ [0, 1])."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mix = 0.5 * (p + q)
    def _kl(a, b):
        nz = a > 0
        return np.sum(a[nz] * np.log2(a[nz] / b[nz]))
    return float(0.5 * _kl(p, mix) + 0.5 * _kl(q, mix))


def dfa_alpha(z, scales=None):
    """Detrended fluctuation analysis exponent α (Peng et al., 1994)."""
    z = np.asarray(z, float)
    Y = np.cumsum(z - z.mean())
    n = len(Y)
    if scales is None:
        scales = np.unique(np.geomspace(16, n // 4, 14).astype(int))
    F = []
    for s in scales:
        nw = n // s
        seg = Y[:nw * s].reshape(nw, s)
        t = np.arange(s) - (s - 1) / 2
        slope = (seg * t).sum(1) / (t ** 2).sum()
        resid = seg - seg.mean(1, keepdims=True) - slope[:, None] * t
        F.append(np.sqrt(np.mean(resid ** 2)))
    F = np.asarray(F)
    ok = F > 0
    return float(np.polyfit(np.log(scales[ok]), np.log(F[ok]), 1)[0])


def time_asymmetry(z, lag=1):
    """Third-order time-reversal asymmetry ⟨Δ³⟩/⟨Δ²⟩^{3/2}, Δ = x_{t+lag} − x_t.
    Zero for any time-reversible (e.g. linear Gaussian) process."""
    d = np.diff(zscore(z), n=1) if lag == 1 else zscore(z)[lag:] - zscore(z)[:-lag]
    m2 = np.mean(d ** 2)
    return float(np.mean(d ** 3) / m2 ** 1.5) if m2 > 0 else 0.0


def _safe_acf(x, nlags):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm_acf(x, nlags=nlags, fft=True)
    except Exception:
        return np.full(nlags + 1, np.nan)


def _rqa_window(z, m, tau, rr):
    X = embed(zscore(z), m, tau)
    D = distance_matrix(X, metric="l2")
    R = recurrence_matrix(D, epsilon_for_rr(D, rr, exclude_diagonal=True))
    meas = rqa_measures(R, lmin=2, vmin=2, theiler=1)
    taus, rr_tau = tau_recurrence_rate(R, RR_TAU_MAX)
    return meas, taus, rr_tau


def rqa_profile(z, m, tau, window=RQA_WINDOW, rr=RQA_RR, n_windows=RQA_N_WINDOWS):
    """RQA measures at a fixed recurrence rate, plus the τ-recurrence-rate
    curve RR_τ (the recurrence analogue of the ACF), averaged over
    `n_windows` windows tiling the whole record. A single window would
    describe one season only: on spg.csv the PV series' LAM alone ranges
    0.62–0.79 from window to window, so whole-record averaging is what makes
    a real-vs-synthetic comparison like-for-like."""
    n = len(z)
    W = min(window, n)
    starts = np.unique(np.linspace(0, n - W, n_windows).astype(int))
    keys = ("det", "lam", "L", "entr", "tt")
    acc = {k: [] for k in keys}
    curves = []
    for s0 in starts:
        meas, taus, rr_tau = _rqa_window(z[s0:s0 + W], m, tau, rr)
        for k in keys:
            acc[k].append(float(meas[k]))
        curves.append(rr_tau)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out.update(taus=taus, rr_tau=np.mean(curves, axis=0))
    return out


def _safe(fn, *args, default=np.nan):
    """Run one estimator; NaN instead of an exception on degenerate input
    (e.g. a 99 %-zero snowfall series has no scaling region for D₂ or DFA)."""
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return fn(*args)
    except Exception:
        return default


def dynamical_profile(y, m, tau, pe_order=None) -> dict:
    """Every invariant / curve the scoreboard compares, for one series,
    always measured with the *original* series' (m, τ). Each estimator is
    isolated, so one undefined invariant never hides the others."""
    y = np.asarray(y, float)
    z = zscore(y)
    pe_order = int(pe_order or np.clip(m, 3, 5))
    nan_curve = lambda n: np.full(n, np.nan)
    emb = _safe(identify_embedding, y, 8, default=None) or dict(
        tau=np.nan, m_fnn=np.nan, ami_curve=nan_curve(AMI_MAX_LAG), fnn_curve=nan_curve(8))
    lam, lyap_curve = _safe(largest_lyapunov, z, m, tau,
                            default=(np.nan, nan_curve(LYAP_HORIZON * tau + 1)))
    n_pat = math.factorial(pe_order)
    p_ord = _safe(ordinal_distribution, z, pe_order, tau, default=np.full(n_pat, 1.0 / n_pat))
    rqa = _safe(rqa_profile, z, m, tau, default=None) or dict(
        det=np.nan, lam=np.nan, L=np.nan, entr=np.nan, tt=np.nan,
        taus=np.arange(1, RR_TAU_MAX + 1), rr_tau=nan_curve(RR_TAU_MAX))
    f, pxx = welch(z, fs=1.0, nperseg=min(512, len(z)))
    return dict(
        tau_hat=emb["tau"], m_hat=emb["m_fnn"],
        ami_curve=emb["ami_curve"], fnn_curve=emb["fnn_curve"],
        lle=lam, lyap_curve=lyap_curve,
        d2=_safe(correlation_dimension, z, m, tau),
        sampen=_safe(sample_entropy, z),
        pe=_safe(permutation_entropy, p_ord), p_ord=p_ord,
        dfa=_safe(dfa_alpha, z),
        trev=_safe(time_asymmetry, y),
        acf=_safe_acf(y, ACF_LAGS),
        psd_f=f, psd=pxx,
        **{f"rqa_{k}": v for k, v in rqa.items()},
    )


def distances_to_real(real: dict, y_real, prof: dict, y_syn) -> dict:
    """Distribution/structure distances between one synthetic series and the
    real one (0 = identical)."""
    sd = max(float(np.std(y_real)), 1e-12)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return _distances(real, y_real, prof, y_syn, sd)


def _distances(real, y_real, prof, y_syn, sd):
    lp_r = 10 * np.log10(real["psd"][1:] + 1e-20)
    lp_s = 10 * np.log10(prof["psd"][1:] + 1e-20)
    return dict(
        ks=float(sp_stats.ks_2samp(y_real, y_syn).statistic),
        w1=float(sp_stats.wasserstein_distance(y_real, y_syn) / sd),
        acf_rmse=float(np.sqrt(np.nanmean((real["acf"][1:] - prof["acf"][1:]) ** 2))),
        lsd=float(np.sqrt(np.mean((lp_r - lp_s) ** 2))),
        jsd=js_divergence(real["p_ord"], prof["p_ord"]),
        ami_rmse=float(np.sqrt(np.nanmean((real["ami_curve"] - prof["ami_curve"]) ** 2))),
        rrtau_rmse=float(np.sqrt(np.nanmean((real["rqa_rr_tau"] - prof["rqa_rr_tau"]) ** 2))),
    )
