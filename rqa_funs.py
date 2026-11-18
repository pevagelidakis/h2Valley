"""RQA toolkit: delay embedding, distance / recurrence matrices, the classical
RQA measures (Marwan et al., 2007), Lomb-Scargle period detection and
sliding-window RQA. Used by rqa_tab.py — all RQA maths lives here."""

from recurrence_plot import RecurrencePlot
import pandas as pd
import numpy as np
from scipy.signal import lombscargle
from scipy.spatial.distance import cdist


# the RQA measures reported per window: (key, label)
RQA_METRICS = [
  ("rr",    "Recurrence Rate RR"),
  ("det",   "Determinism DET"),
  ("ratio", "Ratio DET/RR"),
  ("L",     "Mean diagonal line L"),
  ("Lmax",  "Longest diagonal line Lmax"),
  ("div",   "Divergence DIV = 1/Lmax"),
  ("entr",  "Diagonal entropy ENTR"),
  ("lam",   "Laminarity LAM"),
  ("tt",    "Trapping Time TT"),
  ("vmax",  "Longest vertical line Vmax"),
  ("rte",   "Recurrence time entropy RTE"),
]

_CDIST_METRIC = {"l1": "cityblock", "l2": "euclidean", "linf": "chebyshev"}
_NORM_ORD = {"l1": 1, "l2": 2, "linf": np.inf}


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING, DISTANCES, RECURRENCE MATRICES
# ─────────────────────────────────────────────────────────────────────────────
def embed(x, m, tau):
  # Takens delay embedding: row i is (x_i, x_{i+tau}, ..., x_{i+(m-1)tau})
  x = np.asarray(x, dtype=float)
  L = len(x) - (m-1) * tau
  if L <= 0:
    return np.empty((0, m))
  return np.column_stack([x[i*tau: i*tau + L] for i in range(m)])


def distance_matrix(A, B=None, metric="l2"):
  # pairwise distances ||A_i - B_j|| (B=None -> auto distances of A)
  if metric not in _CDIST_METRIC:
    raise ValueError("Unknown metric: {}".format(metric))
  return cdist(A, A if B is None else B, metric=_CDIST_METRIC[metric])


def epsilon_for_rr(D, rr, exclude_diagonal=True):
  # threshold epsilon such that a fraction rr (0..1) of the distances in D are <= epsilon
  if exclude_diagonal and D.shape[0] > 1:
    vals = D[np.triu_indices_from(D, k=1)]
  else:
    vals = D.ravel()
  if vals.size == 0:
    return 0.0
  return float(np.quantile(vals, np.clip(rr, 1e-4, 1.0)))


def epsilon_from_sample(A, B=None, rr=0.1, metric="l2", theiler=0, n_pairs=200_000, seed=0):
  # epsilon for a target recurrence rate, estimated from random pairs of a long
  # trajectory (avoids building the full N x N matrix)
  B_ = A if B is None else B
  rng = np.random.default_rng(seed)
  i = rng.integers(0, len(A), n_pairs)
  j = rng.integers(0, len(B_), n_pairs)
  keep = np.abs(i - j) >= theiler
  d = np.linalg.norm(A[i[keep]] - B_[j[keep]], ord=_NORM_ORD[metric], axis=1)
  return float(np.quantile(d, np.clip(rr, 1e-4, 1.0))) if d.size else 0.0


def recurrence_matrix(D, epsilon):
  # Heaviside step R_ij = Theta(epsilon - D_ij)
  return D <= epsilon


# ─────────────────────────────────────────────────────────────────────────────
#  LINE STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
def _run_lengths(line):
  # lengths of consecutive runs of 1s in a 1D binary array
  padded = np.concatenate(([0], (np.asarray(line) == 1).astype(np.int8), [0]))
  edges = np.diff(padded)
  starts = np.where(edges == 1)[0]
  ends = np.where(edges == -1)[0]
  return ends - starts


def _theiler_mask(R, theiler):
  # binary R with the band |i-j| < theiler removed, and the number of cells left
  R = np.asarray(R) == 1
  if theiler <= 0:
    return R, R.size
  N, M = R.shape
  band = np.abs(np.arange(N)[:, None] - np.arange(M)[None, :]) < theiler
  return R & ~band, R.size - int(band.sum())


def diagonal_lines(R, theiler=1):
  # lengths of all diagonal lines, skipping diagonals with |offset| < theiler
  R = np.asarray(R) == 1
  N, M = R.shape
  chunks = [_run_lengths(np.diagonal(R, offset=k))
            for k in range(-(N-1), M) if abs(k) >= theiler]
  return np.concatenate(chunks) if chunks else np.array([], dtype=int)


def vertical_lines(R):
  # lengths of all vertical lines (runs of 1s down each column), vectorised
  R = (np.asarray(R) == 1).astype(np.int8)
  edges = np.diff(np.pad(R, ((1, 1), (0, 0))), axis=0).T  # one row per column of R
  starts = np.nonzero(edges == 1)[1]
  ends = np.nonzero(edges == -1)[1]
  return ends - starts


def white_vertical_lines(R):
  # lengths of white vertical lines (runs of 0s down each column bounded by a
  # recurrence point above AND below) = recurrence times of the trajectory
  B = np.asarray(R) == 1
  N = B.shape[0]
  edges = np.diff(np.pad((~B).astype(np.int8), ((1, 1), (0, 0))), axis=0).T
  starts = np.nonzero(edges == 1)[1]
  ends = np.nonzero(edges == -1)[1]
  keep = (starts > 0) & (ends < N)  # drop runs touching the top / bottom border
  return (ends - starts)[keep]


def line_distribution(lengths, min_len=1):
  # normalised line-length histogram: lengths l >= min_len and P(l), sum P(l) = 1
  lengths = np.asarray(lengths)
  lengths = lengths[lengths >= min_len]
  if lengths.size == 0:
    return np.array([], dtype=int), np.array([])
  counts = np.bincount(lengths)
  l = np.nonzero(counts)[0]
  l = l[l >= min_len]
  return l, counts[l] / lengths.size


def tau_recurrence_rate(R, tau_max, two_sided=False):
  # tau-recurrence rate RR_tau = 1/(N - tau) * sum_i R_{i, i+tau}, tau = 1..tau_max
  # (two_sided: tau = -tau_max..tau_max, for the asymmetric cross-RP)
  R = np.asarray(R) == 1
  tau_max = int(min(tau_max, min(R.shape) - 1))
  taus = np.arange(-tau_max if two_sided else 1, tau_max + 1)
  return taus, np.array([np.diagonal(R, offset=k).mean() for k in taus])


def diagonal_hist(R, theiler=1):
  # histogram of diagonal line lengths: hist[l] = number of lines of length l
  return np.bincount(diagonal_lines(R, theiler), minlength=max(R.shape) + 1)


def vertical_hist(R):
  # histogram of vertical line lengths: hist[v] = number of lines of length v
  return np.bincount(vertical_lines(R), minlength=R.shape[0] + 1)


# ─────────────────────────────────────────────────────────────────────────────
#  RQA MEASURES
# ─────────────────────────────────────────────────────────────────────────────
def rqa_measures(R, lmin=2, vmin=2, theiler=0):
  # all RQA measures of RQA_METRICS (rr, det, lam as fractions 0..1), plus the
  # diagonal line lengths >= lmin, vertical line lengths >= vmin, white vertical
  # line lengths (recurrence times) and the number of recurrence points
  Rm, n_cells = _theiler_mask(R, theiler)
  n_rec = int(Rm.sum())
  empty = np.array([], dtype=int)
  out = dict(rr=0.0, det=0.0, ratio=0.0, L=0.0, Lmax=0, div=0.0, entr=0.0,
             lam=0.0, tt=0.0, vmax=0, rte=0.0, diag_lengths=empty, vert_lengths=empty,
             white_lengths=empty, n_points=n_rec)
  if n_cells == 0 or n_rec == 0:
    return out

  # RTE: normalised Shannon entropy of the recurrence-time distribution, taken on
  # the unmasked R (the Theiler band would otherwise create artificial gaps)
  w = white_vertical_lines(R)
  if w.size and w.max() > 1:
    _, pw = line_distribution(w)
    out["rte"] = float(abs(np.sum(pw * np.log(pw))) / np.log(w.max()))
  out["white_lengths"] = w

  d = diagonal_lines(Rm, theiler)
  v = vertical_lines(Rm)
  d = d[d >= lmin]
  v = v[v >= vmin]

  out["rr"] = n_rec / n_cells
  out["det"] = d.sum() / n_rec
  out["ratio"] = out["det"] / out["rr"]
  if d.size:
    out["L"] = float(d.mean())
    out["Lmax"] = int(d.max())
    out["div"] = 1.0 / out["Lmax"]
    _, counts = np.unique(d, return_counts=True)
    p = counts / counts.sum()
    out["entr"] = float(abs(np.sum(p * np.log(p))))
  out["lam"] = v.sum() / n_rec
  if v.size:
    out["tt"] = float(v.mean())
    out["vmax"] = int(v.max())
  out["diag_lengths"] = d
  out["vert_lengths"] = v
  return out


def recurrence_rate(recurrence_matrix):
  N = len(recurrence_matrix)
  return np.sum(recurrence_matrix == 1) / (N**2)


def determinism(recurrence_matrix, lmin):
  # main diagonal (line of identity) excluded
  hist = diagonal_hist(recurrence_matrix)
  l = np.arange(len(hist))
  total = np.sum(l * hist)
  if total == 0:
    return np.nan
  return np.sum(l[lmin:] * hist[lmin:]) / total


def ShanEn(R, lmin):
  # Shannon entropy of the diagonal line length distribution (l >= lmin)
  hist = diagonal_hist(R)[lmin:]
  if hist.sum() == 0:
    return np.nan
  P = hist / hist.sum()
  P = P[P > 0]
  return abs(np.sum(P * np.log(P)))


def LAM(R, vmin=2):
  # laminarity: fraction of recurrence points forming vertical lines (v >= vmin)
  hist = vertical_hist(R)
  v = np.arange(len(hist))
  total = np.sum(v * hist)
  if total == 0:
    return np.nan
  return np.sum(v[vmin:] * hist[vmin:]) / total


def LMIN(R):
  L = []
  N = R.shape[0]
  for i in range(N):
    l = 0
    for j in range(N-i):
      if R[i+j,j] == 0:
        l += 1
      else:
        break
    L.append(l)
  return min(L)


def TT(R, vmin=2):
  # trapping time: average length of vertical lines (v >= vmin)
  hist = vertical_hist(R)[vmin:]
  if hist.sum() == 0:
    return np.nan
  v = np.arange(vmin, vmin + len(hist))
  return np.sum(v * hist) / hist.sum()


# ─────────────────────────────────────────────────────────────────────────────
#  LOMB-SCARGLE PERIOD
# ─────────────────────────────────────────────────────────────────────────────
def lomb_scargle(x, t=None, min_period=2.0, max_period=None, n_periods=3000):
  # normalised Lomb-Scargle power on a log-spaced grid of periods (in units of t)
  x = np.asarray(x, dtype=float)
  t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, dtype=float)
  ok = np.isfinite(x)
  x, t = x[ok], t[ok]
  if max_period is None:
    max_period = (t[-1] - t[0]) / 2
  periods = np.geomspace(min_period, max_period, n_periods)
  power = lombscargle(t, x - x.mean(), 2 * np.pi / periods, normalize=True)
  return periods, power


def dominant_period(x, t=None, min_period=2.0, max_period=None, n_periods=3000):
  # period of the highest Lomb-Scargle peak, plus the full periodogram
  periods, power = lomb_scargle(x, t, min_period, max_period, n_periods)
  return float(periods[np.argmax(power)]), periods, power


# ─────────────────────────────────────────────────────────────────────────────
#  SLIDING-WINDOW RQA
# ─────────────────────────────────────────────────────────────────────────────
def _zscore(x):
  x = np.asarray(x, dtype=float)
  sd = x.std()
  return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


def sliding_window_rqa(x, y=None, window=None, step=None, m=2, tau=1, metric="l2",
                       target_rr=0.1, epsilon=None, lmin=2, vmin=2, theiler=1,
                       standardize=True):
  # RQA measures in windows of `window` samples sliding by `step` samples.
  # epsilon (absolute) or target_rr fixes ONE threshold for the whole record
  # (calibrated on the full trajectory), so RR and the other measures can vary
  # between windows. If y is given, also the auto-RP of y, the cross-RP x<->y
  # and the joint-RP. If window is None it is the Lomb-Scargle period of x.
  # Returns {kind: DataFrame}, kind in auto_x / auto_y / cross / joint, one row
  # per window with columns start, end, center + the keys of RQA_METRICS.
  x = _zscore(x) if standardize else np.asarray(x, dtype=float)
  if window is None:
    window = int(round(dominant_period(x)[0]))
  window = int(window)
  step = max(1, int(step if step is not None else window // 4))
  n_emb = window - (m-1) * tau  # embedded points per window
  if n_emb < 10:
    raise ValueError("Window of {} samples leaves {} embedded points for m={}, tau={}; "
                     "need at least 10.".format(window, n_emb, m, tau))
  if window > len(x):
    raise ValueError("Window ({}) longer than the series ({}).".format(window, len(x)))

  Ex = embed(x, m, tau)
  E = {"auto_x": Ex}
  if y is not None:
    y = _zscore(y) if standardize else np.asarray(y, dtype=float)
    E["auto_y"] = embed(y, m, tau)

  if epsilon is not None:
    eps = {k: float(epsilon) for k in ("auto_x", "auto_y", "cross")}
  else:
    eps = {"auto_x": epsilon_from_sample(Ex, rr=target_rr, metric=metric, theiler=theiler)}
    if y is not None:
      eps["auto_y"] = epsilon_from_sample(E["auto_y"], rr=target_rr, metric=metric, theiler=theiler)
      eps["cross"] = epsilon_from_sample(Ex, E["auto_y"], rr=target_rr, metric=metric)

  kinds = ["auto_x"] + (["auto_y", "cross", "joint"] if y is not None else [])
  rows = {k: [] for k in kinds}
  for s in range(0, len(x) - window + 1, step):
    pos = dict(start=s, end=s + window, center=s + window // 2)
    Wx = Ex[s:s + n_emb]
    Rx = recurrence_matrix(distance_matrix(Wx, metric=metric), eps["auto_x"])
    R = {"auto_x": Rx}
    if y is not None:
      Wy = E["auto_y"][s:s + n_emb]
      R["auto_y"] = recurrence_matrix(distance_matrix(Wy, metric=metric), eps["auto_y"])
      R["cross"] = recurrence_matrix(distance_matrix(Wx, Wy, metric=metric), eps["cross"])
      R["joint"] = R["auto_x"] & R["auto_y"]
    for k in kinds:
      meas = rqa_measures(R[k], lmin=lmin, vmin=vmin, theiler=0 if k == "cross" else theiler)
      rows[k].append({**pos, **{key: meas[key] for key, _ in RQA_METRICS}})

  out = {k: pd.DataFrame(v) for k, v in rows.items()}
  for k, df in out.items():
    df.attrs.update(window=window, step=step, n_emb=n_emb, epsilon=eps.get(k))
  return out


if __name__ == "__main__":
  import matplotlib.pyplot as plt

  n_points = 500  # full series (4213 points) makes the N x N matrix large
  X = pd.read_csv("data/spg.csv")["temperature_2_m_above_gnd"].values#[:n_points]
  print("Input time series:", X)
  m = 2
  tau = 1
  epsilon = 10
  lmin = 2

  metrics = ["l1", "l2", "linf"]

  fig, axes = plt.subplots(2, len(metrics), figsize=(7 * len(metrics), 13))
  results = {}

  for col, metric in enumerate(metrics):
    # unthresholded RP: D[i,j] = ||x_i - x_j|| (threshold=None)
    rp = RecurrencePlot(dimension = m, time_delay = tau, metric=metric, threshold = None)
    D = rp.transform(X.reshape(1, -1))[0]  # transform expects (n_samples, n_timestamps)

    # thresholded RP: R[i,j] = Theta(epsilon - D[i,j])
    R = D <= epsilon

    results[metric] = {
      "RR": recurrence_rate(R),
      "DET": determinism(R, lmin),
      "ENTR": ShanEn(R, lmin),
      "LAM": LAM(R),
      "TT": TT(R),
      "LMIN": LMIN(R),
    }

    ax = axes[0, col]
    im = ax.imshow(D, cmap='viridis', origin='lower')
    ax.set_title("Unthresholded RP ({}, m={}, tau={})".format(metric, m, tau))
    ax.set_xlabel("Time")
    ax.set_ylabel("Time")
    fig.colorbar(im, ax=ax, label="Distance", shrink=0.8)

    ax = axes[1, col]
    ax.imshow(R, cmap='binary', origin='lower')
    ax.set_title("Recurrence Plot ({}, eps={}, RR={:.2%})".format(metric, epsilon, results[metric]["RR"]))
    ax.set_xlabel("Time")
    ax.set_ylabel("Time")

    del D, R  # free memory before the next metric

  print(pd.DataFrame(results).T.to_string(float_format="%.4f"))

  plt.tight_layout()
  plt.show()
