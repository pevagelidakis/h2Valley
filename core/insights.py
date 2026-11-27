"""
core/insights.py  —  Plain-language interpretation generated from results.

The analysis tab used to show one hand-written sentence per variable
("PACF cuts off after lag 12 → AR(12) ..."). Those sentences describe one
particular file and silently become wrong when the data changes. This module
writes the equivalent sentence from the numbers actually computed for the
active dataset, so it is right for any input.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence


MIN_CYCLE_R = 0.2   # an ACF peak weaker than this is not called a "cycle"
                    # (with n in the thousands, ±1.96/√n flags r ≈ 0.03 as significant)


def _cycle_peak(acf: Sequence[float], conf: float, min_lag: int = 2):
    """Strongest local maximum of the ACF beyond lag 1 that is both
    significant and large enough to matter."""
    best = None
    for k in range(max(min_lag, 1), len(acf) - 1):
        if acf[k] > acf[k - 1] and acf[k] >= acf[k + 1] and acf[k] > max(conf, MIN_CYCLE_R):
            if best is None or acf[k] > acf[best]:
                best = k
    return best


def _pacf_cutoff(pacf: Sequence[float], conf: float) -> int:
    """Order p of the leading run of significant PACF lags (0 if lag 1 is not)."""
    p = 0
    for k in range(1, len(pacf)):
        if abs(pacf[k]) > conf:
            p = k
        else:
            break
    return p


def describe_series(acf: Sequence[float], pacf: Sequence[float], conf: float,
                    stat: Optional[dict] = None, unit: str = "samples",
                    period: Optional[int] = None) -> str:
    """One or two sentences on memory, cycle, AR order and stationarity.

    acf, pacf  values at lags 0..L     conf   ± significance band (1.96/√n)
    stat       the stationarity results dict (adf_ok, kpss_ok, ...)
    unit       abbreviation of one sample, used in "lag k (unit)"
    period     the dataset's dominant cycle, mentioned when the series shares it
    """
    if not len(acf) or any(isinstance(v, float) and math.isnan(v) for v in acf[1:2]):
        return "Constant or undefined series: no autocorrelation structure to describe."
    lags = len(acf) - 1
    n_sig = sum(abs(a) > conf for a in acf[1:])
    parts = []

    if n_sig <= max(1, lags // 10):
        parts.append(f"Near white noise: only {n_sig} of {lags} ACF lags are significant, "
                     f"so past values carry little predictive information.")
    else:
        r1 = acf[1]
        memory = "very strong" if r1 > 0.9 else "strong" if r1 > 0.7 else \
                 "moderate" if r1 > 0.4 else "weak"
        parts.append(f"{memory.capitalize()} persistence (lag-1 r = {r1:.2f}).")
        peak = _cycle_peak(acf, conf)
        if peak is not None:
            note = ""
            if period and abs(peak - period) <= 1:
                note = " — the dataset's dominant cycle"
            elif period and acf[peak] > 0.5 and abs(peak - period) > 0.2 * period:
                note = f" — differs from the configured period of {period}"
            parts.append(f"Cycle of ≈ {peak} {unit} (ACF peak r = {acf[peak]:.2f}){note}.")
        else:
            parts.append(f"No clear repeating cycle within {lags} lags.")
        p = _pacf_cutoff(pacf, conf)
        horizon = peak + 1 if peak is not None else len(pacf) - 1
        later = [k for k in range(p + 2, min(horizon, len(pacf) - 1) + 1) if abs(pacf[k]) > conf]
        if p == 0:
            parts.append("PACF not significant at lag 1.")
        elif p >= len(pacf) - 1:
            parts.append(f"PACF significant up to lag {p}: long memory, consider differencing.")
        elif len(later) >= 3:
            core = (f"short-memory AR({p}) core" if not period or p <= period // 2
                    else f"significant up to lag {p} without a break (long memory)")
            parts.append(f"PACF: {core}, plus significant partial "
                         f"autocorrelation out to lag {later[-1]}"
                         + (" (the cycle), better captured by a seasonal term than a long AR."
                            if peak is not None else "."))
        else:
            parts.append(f"PACF cuts off after lag {p} → AR({p}) structure.")

    if stat:
        adf, kps = stat.get("adf_ok"), stat.get("kpss_ok")
        if adf and kps:
            parts.append("ADF and KPSS agree: stationary.")
        elif adf is False and kps is False:
            parts.append("ADF and KPSS agree: non-stationary (unit root likely).")
        elif adf and kps is False:
            parts.append("ADF rejects a unit root but KPSS rejects stationarity: "
                         "likely trend- or level-shift non-stationarity.")
        elif adf is False and kps:
            parts.append("ADF and KPSS disagree (neither rejects): inconclusive, low power.")
    return " ".join(parts)
