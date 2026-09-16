# Hydrogen-Proxy Time-Series Analytics Dashboard

An interactive **Dash / Plotly** dashboard for the exploratory data analysis (EDA),
statistical diagnostics, and synthetic-data groundwork of **hydrogen-production time
series**. Because real operational hydrogen data is not yet available, the platform is
built and validated on a **photovoltaic (PV) / meteorological dataset used as a
structural proxy** — the working assumption being that PV generation and electrolytic
hydrogen production share the same underlying temporal dynamics (diurnal and seasonal
cycles, ramp rates, intermittency, and missing-data behaviour).

The dashboard runs the full battery of tests a practitioner needs *before* committing to
a generative model: stationarity, cointegration, Granger causality, normality, and
regression-residual diagnostics — each paired with a plain-language theory panel so the
output is interpretable from junior to expert level.

---

## Table of contents

- [What it does](#what-it-does)
- [The PV → hydrogen proxy assumption](#the-pv--hydrogen-proxy-assumption)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Statistical methods reference](#statistical-methods-reference)
- [Installation](#installation)
- [Running locally](#running-locally)
- [Running on a remote / shared VM](#running-on-a-remote--shared-vm)
- [Configuration](#configuration)
- [Interpreting the results](#interpreting-the-results)
- [Synthetic & dynamical-analysis tabs](#synthetic--dynamical-analysis-tabs)
- [Troubleshooting](#troubleshooting)
- [Reproducibility, data licensing & FAIR notes](#reproducibility-data-licensing--fair-notes)
- [Acknowledgements](#acknowledgements)

---

## What it does

The app opens as a single dark-themed page with six tabs. Every analytical tab combines
**precomputed results**, **interactive Plotly figures**, and a **theory section** that
explains each test (hypotheses, formula, how to read it, and caveats).

| Tab | What you get |
|-----|--------------|
| 📈 **Exploration** | Per-variable time-series panels for all 20 meteorological/solar drivers plus the PV power target — levels, distribution, and the raw shape of each series. |
| 🧪 **Stationarity** | Unit-root and stationarity testing: **ADF**, **KPSS**, **Phillips–Perron**, and **Zivot–Andrews** (structural-break aware). Overview heatmap + per-variable detail. |
| 🔗 **Cointegration** | Long-run equilibrium testing across variable pairs: **Engle–Granger**, **Phillips–Ouliaris**, and **Johansen** (trace statistic & estimated rank). |
| ⚡ **Causality** | **Granger causality** matrix across all pairs, with the best lag selected over a 1–6 h window and a significance ranking. |
| 🔬 **Diagnostics** | Normality (**KS**, **Shapiro–Wilk**, **Jarque–Bera**), autocorrelation (**Ljung–Box**), lag selection (**HQC**), and regression-residual diagnostics (**Breusch–Pagan**, **Breusch–Godfrey**, **Durbin–Watson**) with a DW gauge. |
| 🗺 **EU H₂ Map** | Geographic map of EU hydrogen-production plants (capacity bubbles, country breakdown, build-year timeline) sourced from the IEA capacity dataset. |

A static preview of the theory callout cards (Recommended workflow, Interpretation note,
Test comparison, Pipeline recommendation) is included as `callouts_preview.png`.

---

## The PV → hydrogen proxy assumption

This platform treats PV electricity generation as a **structural analogue**, not an
identical substitute, for hydrogen production. The assumption holds well for the
features that matter most to a forecaster, but breaks in places that practitioners must
flag explicitly:

**Transfers reasonably well**
- Strong diurnal and seasonal periodicity (solar-driven).
- Weather-dependent intermittency and ramp behaviour.
- Non-stationarity and the autocorrelation structure of the driving signal.

**Does *not* transfer without adjustment**
- **Electrolyser physics**: minimum stable load, ramp limits, efficiency curves, and
  cold-start dynamics have no PV equivalent.
- **Demand coupling**: hydrogen output is shaped by an off-taker duty cycle (refinery
  baseload, ferry schedule, refuelling station, FSRU, fuel-cell APU), whereas PV is
  supply-only.
- **Storage and curtailment**: buffering decouples instantaneous generation from
  delivered hydrogen.

Treat any conclusion drawn from the proxy as a hypothesis about the hydrogen series, to
be revisited once real data is available.

---

## Repository layout

```
.
├── app.py                                              # Main Dash app: exploration/stationarity/cointegration/
│                                                        #   causality/diagnostics tabs, the statistical test
│                                                        #   battery, figures, theory, CSS, app assembly
├── cache_utils.py                                      # Shared on-disk memoization for every tab's precompute
├── map_tab.py                                          # EU hydrogen-production map tab (figures + callbacks)
├── synth_tab_control.py                                # Class-I synthetic data tab (SARIMA shape-matching)
├── synth2_tab.py                                       # Class-II synthetic data tab (IAAFT / bootstrap / copula)
├── rqa_tab.py                                          # Class-III recurrence quantification analysis (RQA) tab
├── spg.csv                                             # Hourly PV/meteorological series (proxy dataset)
├── Hydrogen_production_capacity_2024__Update_2025__0_0.xlsx   # IEA plant-by-plant capacity (drives the map)
├── Hydrogen_trade_2024.xlsx                            # IEA hydrogen-trade reference data (included; not yet wired in)
├── callouts_preview.png                                # Visual reference for the theory callout cards
├── .cache/                                             # On-disk precompute cache (git-ignored; see Configuration)
└── README.md
```

All file paths are **relative to the launch directory**, so run the app from the repo
root (where `spg.csv` and the `.xlsx` files live).

---

## Data

### `spg.csv` — proxy time series

Hourly observations: **4,213 rows × 21 columns** for a single site (Saronic Gulf,
Greece, ≈ 37.9° N, 23.1° E). Twenty meteorological/solar drivers plus one target.

| Column | Meaning |
|--------|---------|
| `temperature_2_m_above_gnd` | Air temperature at 2 m |
| `relative_humidity_2_m_above_gnd` | Relative humidity at 2 m |
| `mean_sea_level_pressure_MSL` | Mean sea-level pressure |
| `total_precipitation_sfc` | Total precipitation |
| `snowfall_amount_sfc` | Snowfall amount |
| `total_cloud_cover_sfc` | Total cloud cover |
| `high_cloud_cover_high_cld_lay` | High-layer cloud cover |
| `medium_cloud_cover_mid_cld_lay` | Mid-layer cloud cover |
| `low_cloud_cover_low_cld_lay` | Low-layer cloud cover |
| `shortwave_radiation_backwards_sfc` | Shortwave (solar) radiation at surface |
| `wind_speed_10_m_above_gnd` | Wind speed at 10 m |
| `wind_direction_10_m_above_gnd` | Wind direction at 10 m |
| `wind_speed_80_m_above_gnd` | Wind speed at 80 m |
| `wind_direction_80_m_above_gnd` | Wind direction at 80 m |
| `wind_speed_900_mb` | Wind speed at the 900 mb level |
| `wind_direction_900_mb` | Wind direction at the 900 mb level |
| `wind_gust_10_m_above_gnd` | Wind gust at 10 m |
| `angle_of_incidence` | Solar angle of incidence |
| `zenith` | Solar zenith angle |
| `azimuth` | Solar azimuth angle |
| `generated_power_kw` | **PV generated power (kW)** — the target / hydrogen proxy |

The CSV has no timestamp column; the app stamps an **hourly** `DatetimeIndex`
(`freq="h"`) on load, consistent with the diurnal (lag-24) structure the tests detect.

### Hydrogen reference spreadsheets

- **`Hydrogen_production_capacity_2024__Update_2025__0_0.xlsx`** — IEA capacity dataset.
  The map tab reads the *plant-by-plant* sheet, auto-detects the header row, drops empty
  leading rows/columns, and filters to EU/European countries (≈ 487 plants with
  coordinates and capacity in MWel).
- **`Hydrogen_trade_2024.xlsx`** — IEA trade reference data, included for context; not
  currently consumed by the app.

---

## Statistical methods reference

| Family | Test | Null hypothesis (H₀) | Use it for |
|--------|------|----------------------|------------|
| Stationarity | **ADF** | Unit root (non-stationary) | Detecting a stochastic trend |
| Stationarity | **KPSS** | Stationary | Confirming ADF via the opposite null |
| Stationarity | **Phillips–Perron** | Unit root | ADF robust to heteroskedastic/autocorrelated errors |
| Stationarity | **Zivot–Andrews** | Unit root with no break | Stationarity in the presence of a structural break |
| Cointegration | **Engle–Granger** | No cointegration | A simple, interpretable residual-based test |
| Cointegration | **Phillips–Ouliaris** | No cointegration | EG, robust to serial correlation, no lag choice |
| Cointegration | **Johansen** | Rank = 0 | Multiple cointegrating vectors, order-invariant |
| Causality | **Granger** | X does not Granger-cause Y | Predictive (not structural) lead–lag structure |
| Normality | **KS / Shapiro–Wilk / Jarque–Bera** | Data are Normal | Distribution shape before modelling |
| Autocorrelation | **Ljung–Box** | No autocorrelation up to lag h | Diurnal-cycle / residual whiteness checks |
| Lag selection | **HQC** | — | Picking AR/ARIMA order |
| Heteroskedasticity | **Breusch–Pagan** | Constant variance | Stability of regression-residual variance |
| Serial correlation | **Breusch–Godfrey** | No serial correlation ≤ lag p | Higher-order residual autocorrelation |
| Serial correlation | **Durbin–Watson** | No 1st-order serial correlation | Quick lag-1 residual check (≈ 2 = clean) |

> The regression-residual trio (BP / BG / DW) is computed on the residuals of a
> constant + linear-trend fit of each variable, so it reflects what a *trend-only*
> baseline leaves behind — typically the uncaptured diurnal cycle.

---

## Installation

Requires **Python ≥ 3.9**.

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install dash dash-bootstrap-components plotly pandas numpy scipy statsmodels arch openpyxl
```

Or pin them in a `requirements.txt`:

```text
dash
dash-bootstrap-components
plotly
pandas
numpy
scipy
statsmodels
arch
openpyxl
```

---

## Running locally

From the repo root (so the data files resolve):

```bash
python app.py
```

Wait for the startup line:

```
Dash is running on http://0.0.0.0:8050/
```

Then open **http://127.0.0.1:8050** in your browser.

> **First run takes a moment; every run after that is fast.** All tests are precomputed
> before the server binds the port (stationarity, cointegration, Granger, normality, and
> the BP/BG/DW loop over 4,213 rows × 20 variables), and the SARIMA fits behind the
> Class-1 synthetic-data tab run in the background. Every one of these results is cached
> to disk in `.cache/` (see **Configuration**) the first time it's computed; a later
> `python app.py` with an unchanged `spg.csv` loads them straight from disk instead of
> redoing the work, so only the very first run — or a run after editing the analysis
> config or the data — pays the full cost. The page only loads once you see the
> “Dash is running” line.

---

## Running on a remote / shared VM

The app binds `0.0.0.0`, so it accepts external connections — but `127.0.0.1` on **your**
machine is your laptop, not the VM. Use an **SSH tunnel** (no firewall changes needed):

```bash
# on the VM — keep it alive across disconnects
tmux new -s dash
python app.py            # wait for "Dash is running on http://0.0.0.0:8050/"
# detach: Ctrl-b then d

# on your laptop — forward the VM's port 8050 to your localhost
ssh -L 8050:localhost:8050 <user>@<vm-address>
```

Then browse to **http://127.0.0.1:8050** locally.

On a shared VM, **port 8050 may already be taken**. Pick a unique port — change the last
line of `app.py` to e.g. `port=8071`, tunnel `-L 8071:localhost:8071`, and visit
`127.0.0.1:8071`. Confirm the listener with `ss -ltnp | grep <port>` on the VM.

If you use **VS Code Remote-SSH**, just run the app in its integrated terminal — VS Code
auto-forwards the port (see the “Ports” panel); no manual tunnel required.

---

## Configuration

| What | Where | Notes |
|------|-------|-------|
| Server host/port/debug | last line of `app.py`: `app.run(debug=True, host="0.0.0.0", port=8050)` | Use **`debug=False`** on a VM — `debug=True` runs the whole heavy precompute **twice** (Werkzeug reloader). |
| Proxy data path | `CSV_PATH = Path("spg.csv")` in `app.py` | Relative to launch dir. |
| Map data path | `XLSX_PATH` in `map_tab.py` | Relative to launch dir; loader is resilient to header offsets. |
| EU country filter | `EU_COUNTRIES` in `map_tab.py` | Edit to widen/narrow the map scope. |
| Fonts | `EXTERNAL_STYLESHEETS` in `app.py` | Pulls IBM Plex from Google Fonts; see Troubleshooting for offline VMs. |
| Precompute cache | `.cache/` (created on first run), via `cache_utils.py` | Delete this directory (or any one `.pkl` file in it) to force that block to recompute from scratch — e.g. after editing `spg.csv` or a test's config. Each cache entry is signature-tagged with the source file's size/mtime and the relevant config, so a changed `spg.csv` invalidates it automatically; nothing changes if you just restart the app. Not committed to git (see `.gitignore`). |

---

## Interpreting the results

Each tab ends with an elegant **theory section** that you can read top-to-bottom:

- A **theory card** per test — plain description, the statistic in LaTeX, H₀/H₁, how to
  read it, and caveats.
- A **callout box** summarising the practical takeaway:
  - *Recommended workflow* (Diagnostics) — the order of operations from raw series to a
    validated model.
  - *Interpreting the stationarity suite* — how to read ADF/PP against KPSS, and when to
    trust Zivot–Andrews.
  - *Choosing among the cointegration tests* — EG vs PO vs Johansen.
  - *From Granger pairs to a validated simulator* — CCM → VAR → block-bootstrap →
    validation.

The guiding principle throughout: **no single test is decisive** — read each family as a
panel and weight the agreement.

---

## Synthetic & dynamical-analysis tabs

The diagnostics layer above is the foundation for the platform's end goal — generating
**synthetic hydrogen-production series** that are faithful on three axes, plus a
model-free way to *analyse* the third axis directly on the real data. All three are
implemented, one tab each:

1. **Shape fidelity** (`synth_tab_control.py`, tab *Synthetic Data I*) — peaks, valleys,
   ramp behaviour and seasonal structure match the original visually and structurally. A
   seasonal ARIMA (SARIMA) is identified per variable (periodogram → seasonal period,
   ADF → differencing order, AICc/BIC stepwise search over the remaining orders) and
   simulated for both shape validation and on-demand future projection.
2. **Statistical fidelity** (`synth2_tab.py`, tab *Synthetic Data II*) — mean, variance,
   skew, kurtosis and the full marginal distribution match exactly or to interpolation
   accuracy, by construction. Three interpretable generators are offered: IAAFT
   surrogates, circular block bootstrap, and a Gaussian-rank AR (copula) model.
3. **Dynamic fidelity** (`rqa_tab.py`, tab *Recurrence (RQA)*) — rather than generating a
   synthetic path, this tab reconstructs the real phase-space trajectory (Takens delay
   embedding, with τ and m identified per variable via average mutual information and
   false nearest neighbours) and quantifies recurrence — auto-, cross- and joint-
   recurrence between two selected variables, with the full RQA measure set (RR, DET,
   LAM, L, Lmax, divergence, entropy, trapping time). It is the direct, computable
   counterpart to the chaos fingerprints (Lyapunov and Hurst exponents, CCM) referenced
   in the Causality tab's theory panel.

Validation throughout favours the same panel-of-evidence principle as the rest of the
app: KS tests and Wasserstein distance for the statistical match, ACF/PACF overlays for
the shape and dependence match, and RQA measures compared across auto- vs. cross-
recurrence for the dynamical match — interpretable methods over opaque ones wherever
they perform comparably.

---

## Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| `127.0.0.1:8050` shows nothing | The server is on the VM, not your laptop. Use the SSH tunnel (above), or browse `http://<vm-ip>:8050` after opening the firewall. |
| `ss -ltnp \| grep 8050` returns nothing | Nothing is listening yet — the app is still precomputing or it crashed. Run in the foreground and read the last line. |
| Stuck after `… diagnostics done ✔`, no server line | With `debug=True`, the reloader re-runs the entire startup a second time. Wait for the second pass, or set `debug=False`. |
| `Could not resolve hostname …` | The address after `@` isn’t a real/reachable host. Use the VM’s actual IP/hostname, an `~/.ssh/config` alias, or connect to the VPN first. |
| Crash on startup with a `ValueError`/`FileNotFoundError` | A data file is missing from the launch directory, or a package isn’t installed. Confirm `spg.csv`, `map_tab.py`, and the `.xlsx` are present and reinstall dependencies. |
| Titles render in a plain font (no monospace) | Google Fonts is blocked (offline VM). The app still works; to fix the look, vendor IBM Plex into a local `assets/` folder and reference it from `EXTERNAL_STYLESHEETS`. |
| Port already in use on a shared VM | Choose another port in `app.run(...)` and tunnel that port instead. |

---

## Reproducibility, data licensing & FAIR notes

- **Determinism** — the analytical tests are deterministic given the input data; record
  package versions (`pip freeze`) alongside results for full reproducibility.
- **Licensing of inputs** — verify and respect the licence of each source before
  redistributing derived data. In particular, datasets derived from
  **Renewables.ninja** carry a **CC BY-NC 4.0** (non-commercial) restriction, and **IEA**
  datasets have their own terms of use. Confirm the provenance and licence of `spg.csv`
  before publishing any derived or synthetic dataset built on it.
- **FAIR / data management** — when releasing synthetic datasets, anonymise as required,
  document the generating method and parameters, and follow the project’s Data Management
  Plan (retention, metadata, and repository targets).

> Update this section with the exact source and licence of `spg.csv` and the IEA files
> for your release.

---

## Acknowledgements

Developed as part of a research effort on time-series forecasting and synthetic-data
generation for green-hydrogen systems. The PV/meteorological dataset is used as a
structural proxy pending the availability of real operational hydrogen data.

> Add your project name, funding/grant agreement number, and institutional affiliation
> here before publishing the repository.

---

*Contributions and issues welcome. When reporting a bug, include the last ~10 lines of
the `python app.py` output and your Python/package versions.*
