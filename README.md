# h2Valley — Hydrogen-Proxy Time-Series Analytics Dashboard

An interactive web dashboard that takes an hourly weather-and-solar dataset, runs a full
battery of time-series statistics on it, and generates **synthetic data** that reproduces
its shape, its statistics, and its dynamics — the groundwork for simulating
**green-hydrogen production** before real plant data exists.

This README is written **bottom-up**: it starts from first principles and builds toward
the statistics, the software architecture, and deployment. Read from the top if you are
new; jump straight to the part you need if you are not.

| If you are… | Start at |
|---|---|
| Just trying to run it | [Quick start](#quick-start) |
| New to time series / statistics | [Part 1 — Foundations](#part-1--foundations) |
| A data scientist / statistician | [Part 3 — The analysis tabs](#part-3--the-analysis-tabs) |
| A developer | [Part 5 — How the software works](#part-5--how-the-software-works) |
| Deploying it for others | [Part 6 — Running and sharing it](#part-6--running-and-sharing-it) |

---

## Quick start

You need **Docker Desktop** (Windows/macOS) or **Docker Engine + Compose** (Linux).

```bash
docker compose up -d --build      # build the image and start everything
docker compose logs -f app        # watch startup (Ctrl+C stops watching, not the app)
```

Open **http://localhost** once the log shows `All tests ready ✔`.

- **First start** is slow (minutes) because every statistical model is computed once.
- **Every start after that** loads results from a persistent cache and takes seconds.
- A **public link** is printed by the tunnel service:
  `docker compose logs tunnel | grep trycloudflare` (PowerShell: `| Select-String trycloudflare`).

Everything else in this document explains *what* you are looking at and *why* it works.

---

## Table of contents

- [Part 1 — Foundations](#part-1--foundations)
  - [1.1 What a time series is](#11-what-a-time-series-is)
  - [1.2 Why a solar dataset stands in for hydrogen](#12-why-a-solar-dataset-stands-in-for-hydrogen)
  - [1.3 Five ideas you need for everything else](#13-five-ideas-you-need-for-everything-else)
- [Part 2 — The data](#part-2--the-data)
- [Part 3 — The analysis tabs](#part-3--the-analysis-tabs)
- [Part 4 — The synthetic-data tabs](#part-4--the-synthetic-data-tabs)
- [Part 5 — How the software works](#part-5--how-the-software-works)
- [Part 6 — Running and sharing it](#part-6--running-and-sharing-it)
- [Part 7 — Configuration reference](#part-7--configuration-reference)
- [Part 8 — Performance](#part-8--performance)
- [Part 9 — Troubleshooting](#part-9--troubleshooting)
- [Part 10 — Reproducibility, licensing & limitations](#part-10--reproducibility-licensing--limitations)
- [Glossary](#glossary)

---

## Part 1 — Foundations

### 1.1 What a time series is

A **time series** is a sequence of measurements taken at regular intervals — here, one
row per **hour**. What makes time series special is that **order matters**: the
temperature at 14:00 tells you a lot about the temperature at 15:00. Ordinary statistics
assume observations are independent; time-series statistics are built around the fact
that they are *not*.

Three patterns dominate this dataset:

| Pattern | Meaning | Example here |
|---|---|---|
| **Cycle / seasonality** | A shape that repeats at a fixed period | Solar radiation peaks every 24 h |
| **Trend** | A slow drift up or down | Temperature rising from spring into summer |
| **Noise** | Short-term randomness | Gusts, passing clouds |

Most of this dashboard is about separating these three and checking whether a model —
or a synthetic generator — gets each of them right.

### 1.2 Why a solar dataset stands in for hydrogen

Green hydrogen is made by **electrolysers** powered by renewable electricity. Real
operational hydrogen data is not yet available, so the platform is built and validated on
a **photovoltaic (PV) / meteorological dataset** used as a *structural proxy*: PV output
and electrolytic hydrogen production share the same underlying temporal drivers.

**What transfers well**
- Strong diurnal (daily) and seasonal periodicity, because both are solar-driven.
- Weather-driven intermittency and ramp behaviour.
- Non-stationarity and the autocorrelation structure of the driving signal.

**What does *not* transfer without adjustment**
- **Electrolyser physics** — minimum stable load, ramp limits, efficiency curves and
  cold-start dynamics have no PV equivalent.
- **Demand coupling** — hydrogen output follows an off-taker's duty cycle (refinery
  baseload, ferry schedule, refuelling station, FSRU, fuel-cell APU); PV is supply-only.
- **Storage and curtailment** — buffering decouples instantaneous generation from
  delivered hydrogen.

Treat every conclusion drawn from the proxy as a **hypothesis about the hydrogen series**,
to be re-tested once real data arrives. The code is written so that swapping in a new CSV
is the only change needed (see [Part 7](#part-7--configuration-reference)).

### 1.3 Five ideas you need for everything else

**1. Hypothesis tests and p-values.** Every test in the app starts from a default
assumption, the **null hypothesis (H₀)** — e.g. "this series has no trend-like
wandering". The test computes how surprising the data would be *if H₀ were true*. That
surprise is the **p-value**. The app uses **α = 0.05**: if p < 0.05 we *reject* H₀.
A large p-value does **not** prove H₀; it only means the data is not strong evidence
against it.

> Watch which way each test points. For ADF, rejecting H₀ means "stationary" (good).
> For KPSS, the null is *stationary*, so rejecting it means "non-stationary". The app
> colours every result green/red by its *meaning*, not by its p-value, so you don't have
> to track this yourself.

**2. Stationarity.** A series is **stationary** if its statistical behaviour (mean,
variance, correlation structure) does not drift over time. Most forecasting models assume
it. A non-stationary series has a **unit root**: shocks never die out, so it wanders like
a random walk. Differencing (`yₜ − yₜ₋₁`) usually fixes that.

**3. Autocorrelation (ACF) and partial autocorrelation (PACF).** The **ACF at lag k**
is the correlation between the series and itself shifted by k hours:

```
ρ(k) = Corr(yₜ, yₜ₋ₖ)
```

A spike at lag 24 means "today at this hour looks like yesterday at this hour". The
**PACF** is the same correlation *after removing the effect of the lags in between*; it
tells you how many past values a model actually needs. Bars beyond the band
±1.96/√n are significant at 5%.

**4. Models vs. tests.** Tests (stationarity, causality, normality) *describe* the data.
Models (SARIMA, autoencoders) *reproduce* it. The workflow is always: **test first →
choose a model the tests justify → validate the model against the same tests.**

**5. Synthetic data fidelity.** A synthetic series is only useful if it is faithful.
This project defines three levels, each with its own tab:

| Fidelity | Question it answers | Tab |
|---|---|---|
| **Shape** | Do peaks, valleys, ramps and cycles look right? | Synthetic Data (I) |
| **Statistical** | Do the mean, variance, skew and full distribution match? | Synthetic Data II |
| **Dynamic** | Does it evolve the way the real system does (same "attractor")? | Recurrence (RQA), Synthetic Data III |

---

## Part 2 — The data

All data lives in [`data/`](data/).

### `data/spg.csv` — the proxy time series

**4,213 hourly rows × 21 columns** for a single site (Saronic Gulf, Greece,
≈ 37.9° N, 23.1° E): 20 meteorological/solar drivers plus one target.

| Column | Meaning | Group |
|---|---|---|
| `temperature_2_m_above_gnd` | Air temperature at 2 m | Temperature |
| `relative_humidity_2_m_above_gnd` | Relative humidity at 2 m | Humidity |
| `mean_sea_level_pressure_MSL` | Mean sea-level pressure | Pressure |
| `total_precipitation_sfc` | Total precipitation | Precipitation |
| `snowfall_amount_sfc` | Snowfall amount | Precipitation |
| `total_cloud_cover_sfc` | Total cloud cover | Cloud |
| `high_cloud_cover_high_cld_lay` | High-layer cloud cover | Cloud |
| `medium_cloud_cover_mid_cld_lay` | Mid-layer cloud cover | Cloud |
| `low_cloud_cover_low_cld_lay` | Low-layer cloud cover | Cloud |
| `shortwave_radiation_backwards_sfc` | Shortwave (solar) radiation at the surface | Radiation |
| `wind_speed_10_m_above_gnd` | Wind speed at 10 m | Wind |
| `wind_direction_10_m_above_gnd` | Wind direction at 10 m (circular, 0–360°) | Wind |
| `wind_speed_80_m_above_gnd` | Wind speed at 80 m | Wind |
| `wind_direction_80_m_above_gnd` | Wind direction at 80 m (circular) | Wind |
| `wind_speed_900_mb` | Wind speed at the 900 mb level (~1 km) | Wind |
| `wind_direction_900_mb` | Wind direction at 900 mb (circular) | Wind |
| `wind_gust_10_m_above_gnd` | Wind gust at 10 m | Wind |
| `angle_of_incidence` | Solar angle of incidence on the panel | Solar geometry |
| `zenith` | Solar zenith angle | Solar geometry |
| `azimuth` | Solar azimuth angle (circular) | Solar geometry |
| `generated_power_kw` | **PV generated power (kW)** — the target / hydrogen proxy | Target |

Notes for practitioners:

- **No timestamp column.** On load the app stamps an hourly `DatetimeIndex` that **ends
  at the current hour**, so the calendar dates on chart axes move with the run date. Only
  the hourly spacing is meaningful; it is consistent with the lag-24 structure every test
  detects.
- The statistical tabs analyse the **20 drivers**. `generated_power_kw` is used by
  Synthetic Data III, where it is the default variable.
- **Circular variables** (wind directions, azimuth) wrap at 0°/360°. Linear statistics on
  them (ACF, means, Gaussian tests) should be read with caution; the app's insight notes
  flag this.

### Hydrogen reference spreadsheets

| File | Used by | Content |
|---|---|---|
| `Hydrogen production capacity 2024 (Update 2025)_0_0.xlsx` | EU H₂ Map tab | IEA plant-by-plant capacity. The loader auto-detects the header row and filters to European countries (≈ 487 plants with coordinates and capacity in MWel). |
| `Hydrogen trade 2024.xlsx` | — | IEA trade reference data; included for context, not yet wired in. |
| `ElectricDevices.zip` | — | Not referenced by the code. |

---

## Part 3 — The analysis tabs

The app is one page with ten tabs. The first five are classical time-series diagnostics;
each pairs **precomputed results**, **interactive Plotly charts**, and a **theory panel**
(plain description, formula, H₀/H₁, how to read it, caveats).

The guiding principle throughout: **no single test is decisive — read each family as a
panel and weigh the agreement.**

### 📈 Exploration

- **ACF & PACF explorer** for any variable, up to 48 lags (two full days), with the 95%
  confidence band, a count of significant lags, and a short expert note per variable
  (e.g. *"Shortwave radiation: textbook diurnal sine wave, r ≈ 0.995 at lag 24"*).
- **Time-series view** of the selected variable (downsampled to ~1,200 points for
  rendering speed).

**How to read it:** a slowly decaying ACF means strong persistence (possibly
non-stationary); a sinusoidal ACF means a cycle; the lag at which the PACF cuts off
suggests the order of an autoregressive model, AR(p).

### 🧪 Stationarity

Four tests, deliberately with opposite nulls so they cross-check each other:

| Test | H₀ | Strength |
|---|---|---|
| **ADF** (Augmented Dickey–Fuller) | Unit root (non-stationary) | The standard; lag length chosen by AIC |
| **KPSS** | Stationary | Opposite null — confirms ADF |
| **Phillips–Perron** | Unit root | Robust to heteroskedastic/autocorrelated errors without adding lags |
| **Zivot–Andrews** | Unit root with no structural break | Allows one endogenous break in level and trend |

ADF fits

```
Δyₜ = α + βt + γ·yₜ₋₁ + Σᵢ δᵢ·Δyₜ₋ᵢ + εₜ        H₀: γ = 0
```

**How to read it:** ADF/PP reject *and* KPSS fails to reject → clearly stationary.
ADF/PP fail *and* KPSS rejects → clearly non-stationary. Disagreement usually means a
structural break (trust Zivot–Andrews) or near-unit-root behaviour.

On the bundled data, almost every variable is stationary by ADF; the notable exceptions
are the purely geometric **zenith** and **angle of incidence**, whose deterministic
seasonal drift ADF reads as a unit root.

### 🔗 Cointegration

Two individually wandering series are **cointegrated** if some combination of them is
stable: they may drift, but they drift *together* (a long-run equilibrium). The tab
tests 13 physically motivated pairs (radiation ↔ temperature, wind at 10 m ↔ 80 m, …):

| Test | H₀ | Notes |
|---|---|---|
| **Engle–Granger** | No cointegration | Regress one on the other, ADF-test the residual. Simple and interpretable, but direction-dependent. |
| **Phillips–Ouliaris** (Zₜ) | No cointegration | Residual-based like EG, robust to serial correlation, no lag choice. |
| **Johansen** (trace & max-eigenvalue) | Rank = 0 | System-based, order-invariant, finds the number of cointegrating vectors. |

> **Expert note.** Johansen rank = 2 for a pair of two series means the system is full
> rank, i.e. *both series are already stationary*, so "cointegration" is trivially
> satisfied. On this dataset most pairs return rank 2, consistent with the Stationarity
> tab. Read EG/PO significance alongside it rather than as a stand-alone finding.

### ⚡ Causality

**Granger causality**: X "Granger-causes" Y if past values of X improve the prediction of
Y beyond what Y's own past provides. It is **predictive, not physical** causation.

- Run on **first differences** (to remove trends) for 14 pairs, lags 1–6 h; the best
  lag (lowest p) is reported.
- Shown as a causal matrix (−log₁₀ p) and a significance ranking.

On the bundled data, most physically expected links are significant (e.g. radiation →
temperature at lag 4 h), while *pressure → wind speed* and *humidity → precipitation* are
not.

The theory panel also outlines the route from Granger pairs to a validated simulator
(convergent cross mapping → VAR → block bootstrap → validation). `app_theory.py` is an
extended edition of the app that additionally computes **VAR block-exogeneity tests** and
**convergent cross mapping (CCM)**, with one plot per theory card (see
[Part 5](#51-file-map)).

### 🔬 Diagnostics

Two families:

**Distribution & dependence** (on each raw series)

| Test | H₀ | What it checks |
|---|---|---|
| **Kolmogorov–Smirnov** | Data ~ Normal(μ, σ) | Overall distribution shape |
| **Shapiro–Wilk** | Data are Normal | Most powerful normality test (capped at 5,000 points) |
| **Jarque–Bera** | Skew = 0 and excess kurtosis = 0 | Normality via the third and fourth moments |
| **Ljung–Box** (lags 6, 12, 24) | No autocorrelation up to lag h | Whether the series is white noise |
| **Hannan–Quinn criterion** | — | Best AR order p ∈ 1…12 for each variable |

**Regression-residual diagnostics** (on the residuals of a constant + linear-trend fit,
`yₜ = α + βt + eₜ`)

| Test | H₀ | What it checks |
|---|---|---|
| **Breusch–Pagan** | Constant variance | Heteroskedasticity |
| **Breusch–Godfrey** (up to lag 12) | No serial correlation | Higher-order residual autocorrelation |
| **Durbin–Watson** | No lag-1 correlation | ≈ 2 is clean; < 2 positive, > 2 negative autocorrelation |

> A trend-only baseline leaves the daily cycle in its residuals, so failing BG/DW here is
> *expected*: it is the evidence that a seasonal model is required, which is exactly what
> Synthetic Data I builds.

### 🗺 EU H₂ Map

A geographic map of European hydrogen-production plants from the IEA dataset: capacity
bubbles, country breakdown, build-year timeline. Filter by project status, country, and
top-N plants.

---

## Part 4 — The synthetic-data tabs

The diagnostics above justify the models used here. Each tab generates (or analyses)
series at one fidelity level, and validates them against the real data with the same
"panel of evidence" principle.

### 🧬 Synthetic Data (I) — shape fidelity · `synth_tab_control.py`

A **seasonal ARIMA (SARIMA)** model is identified per variable:

1. **Periodogram** → dominant seasonal period.
2. **ADF** → differencing order *d*.
3. **Bounded stepwise search** over (p, q, P, Q), scored by AICc with a BIC-parsimony
   rule (prefer the simpler model when AICc differs by ≤ 2), capped at 240 s per variable.

Each fitted model is simulated **6 times** (independent realisations) into one shared
tensor `SYNTHETIC_TENSOR[observations, variables, realisations]`, so switching variables
in the UI is an instant lookup. Validation overlays ACF/PACF and moments of real vs.
synthetic. You can also simulate **future** paths conditioned on the last observation.

### 🎲 Synthetic Data II — statistical fidelity · `synth2_tab.py`

Three interpretable generators that preserve the marginal distribution exactly or to
interpolation accuracy, by construction:

- **IAAFT surrogates** — keep the exact value distribution *and* the power spectrum.
- **Circular block bootstrap** — resample real blocks (default length = the detected
  dominant cycle) to keep short-range structure.
- **Gaussian-rank AR (copula)** — map to normal scores, fit an AR model, map back.

Validated with KS distance, ECDF/Q–Q plots, histograms, moments and ACF overlays.

### 🌀 Recurrence (RQA) — dynamic fidelity, analysed · `rqa_tab.py`

Instead of generating anything, this tab reconstructs the real system's **phase space**
and measures how it recurs.

- **Takens delay embedding**: the 1-D series becomes points
  `(yₜ, yₜ₊τ, …, yₜ₊₍ₘ₋₁₎τ)`. The delay **τ** comes from the first minimum of the
  **average mutual information**; the dimension **m** from **false nearest neighbours**.
- **Recurrence plots** — auto-, cross- and joint-recurrence between two chosen variables,
  at a target recurrence rate.
- **RQA measures** — RR (recurrence rate), DET (determinism), LAM (laminarity),
  L / Lmax (diagonal line lengths), divergence, entropy, trapping time; plus a sliding-
  window view and a τ–RR scan.

It is the computable counterpart to the chaos fingerprints (Lyapunov, Hurst, CCM)
referenced in the Causality theory panel.

### 🔮 Synthetic Data III — dynamic fidelity, generative · `synth3_tab.py` + `synth3_funs.py`

Generates new trajectories **on the reconstructed attractor** of any `spg.csv` column
(PV target included), using the same τ/m estimators as the RQA tab:

- **Twin Surrogates** (Thiel et al., 2006) — a random walk on the attractor that jumps
  between states sharing the same recurrence neighbourhood. Exact twins are almost absent
  in continuous data, so a Jaccard tolerance **δ** is used.
- **GENDA** (Troullinou et al., 2024) — a generative neighbourhood-based deep
  autoencoder run as a phase-space rollout (PyTorch; trained once per variable in the
  background and cached).

Validation compares the re-identified (τ, m), Lyapunov exponent, correlation dimension,
sample/permutation entropy, DFA, time-reversal asymmetry, RQA measures, and spectral /
ordinal / recurrence distances. **Every tolerance is calibrated by the real series'
own split-half variability**: a synthetic ensemble passes if it deviates from the real
series less than the real series deviates from itself.

---

## Part 5 — How the software works

### 5.1 File map

```
.
├── app.py                 # Entry point: data loading, the statistical test battery,
│                          #   figures, theory panels, CSS, tab assembly, `server` for gunicorn
├── app_theory.py          # Extended edition (+VAR block exogeneity, +CCM, one plot per
│                          #   theory card). Standalone; runs on port 8051
├── cache_utils.py         # On-disk memoisation shared by every heavy computation
├── core/dataset.py        # DatasetSpec: loads the active dataset config, shared by every module
├── configs/spg.yaml       # Dataset config: path, time axis, target, labels/units/groups, pairs
├── map_tab.py             # EU H₂ Map tab
├── synth_tab_control.py   # Synthetic Data I  (SARIMA)
├── synth2_tab.py          # Synthetic Data II (IAAFT / bootstrap / copula)
├── rqa_tab.py             # Recurrence (RQA) tab
├── rqa_funs.py            # RQA maths
├── recurrence_plot.py     # Recurrence-plot construction (scikit-learn style estimator)
├── synth3_tab.py          # Synthetic Data III (Twin Surrogates / GENDA) — UI and orchestration
├── synth3_funs.py         # Synthetic Data III maths: twin surrogates, GENDA, dynamical invariants
├── warm_cache.py          # Fills .cache/ ahead of time, then exits
├── data/                  # spg.csv + IEA spreadsheets
├── requirements.txt       # Pinned Python dependencies (CPU-only PyTorch)
├── Dockerfile             # App image (gunicorn)
├── docker-compose.yml     # app + Caddy reverse proxy + Cloudflare tunnel
├── Caddyfile              # Reverse-proxy config (compression, optional HTTPS)
└── .dockerignore
```

### 5.2 What happens when the app starts

Importing `app.py` does all the work, in this order:

1. **Import PyTorch first.** On Windows, torch must load its DLLs before pandas/statsmodels
   or it fails with `WinError 1114`. If torch is missing, only GENDA is disabled.
2. **Import the tab modules.** Each one checks the cache; if its results are missing it
   starts a **background daemon thread** to compute them, so the server is not blocked:
   - `synth_tab_control` — 20 SARIMA fits (the slowest job by far)
   - `synth2_tab` — Class-II identification
   - `rqa_tab` — phase-space (τ, m) identification
   - `synth3_tab` — Class-III identification, then GENDA training
3. **Load the active dataset** (`configs/spg.yaml` → `data/spg.csv` by default) through
   `core.dataset`, which builds the time index once for every module.
4. **Run the statistical test battery** (ACF/PACF, 4 stationarity tests, 3 cointegration
   tests, Granger, normality, Ljung–Box, HQC, BP/BG/DW) — or load it from cache. This step
   **is** blocking: the web server answers only after it finishes.
5. **Build the layout and register callbacks.** `server = app.server` is exposed for
   gunicorn.

While a background job is still running, its tab shows a progress banner and fills in
automatically when the result for the selected variable is ready.

### 5.3 The cache (`cache_utils.py`)

Every heavy computation is a pure function of its input files plus a few constants, so it
is cached in `.cache/<name>.pkl`:

```python
sig = make_signature(CSV_PATH, version="app-precompute-v1", nlags=NLAGS, ...)
result = load_cache("app_precompute", sig)        # hit → done
if result is None:
    result = run_expensive_battery()
    save_cache("app_precompute", sig, result)     # atomic write (temp file + rename)
```

The **signature** hashes each source file's *(path, size, modification time)*, every
relevant config value, and a manual `version` tag. Change any of them and that cache entry
is recomputed automatically; nothing else is touched.

Consequences worth knowing:
- Delete `.cache/` (or one `.pkl`) to force a recompute.
- Because the signature includes the **absolute path and mtime**, a cache built on one
  machine (or outside Docker) is **not** reused inside a container. In Docker the cache
  lives in its own persistent volume instead (see [6.2](#62-docker-recommended)).
- The SARIMA cache is ~1.9 GB (each fitted model keeps its full Kalman-filter state) and
  is written only after all 20 variables finish. GENDA models are saved one per variable,
  so an interrupted run keeps what it has finished.

### 5.4 How the UI stays responsive (callbacks and polling)

The UI is built with **Dash**: Python functions (*callbacks*) run on the server whenever
an input changes, and return new figures as JSON to the browser.

Tabs whose data is computed in the background use a **two-step polling pattern**:

```
dcc.Interval (every 1.5–2 s) ─▶ light _poll callback ─▶ dcc.Store "<tab>-ready"
                                    (tiny response)          │ changes only when the
                                                             │ selected variable's
                                                             ▼ readiness changes
                                                  heavy figure callback (MBs of JSON)
```

The heavy callback **never** listens to the timer directly. If it did, then on a slow
connection a multi-megabyte response could take longer than one tick, be superseded by
the next request, and never render. Once the selected variable is ready the poller also
**disables the timer**; choosing a variable that is still pending turns it back on.

### 5.5 Production stack

```
browser ──▶ [Cloudflare tunnel] ──▶ Caddy :80/:443 ──▶ gunicorn :7860 ──▶ Dash app (app.py)
             optional public link    compression,        1 worker,
                                     optional HTTPS      4 threads
```

- **gunicorn** runs **one worker** on purpose: the app holds gigabytes of models in memory
  at module level, and every extra worker would duplicate them. Four threads serve
  concurrent requests. `--timeout 1800` leaves room for a slow first start.
- **BLAS/OpenMP threads are capped at 1** (`OMP_NUM_THREADS=1`, etc.). The app runs 4–5
  computation threads at once on small matrices; letting each one start a full-core BLAS
  pool oversubscribed the CPU and made startup **3–4× slower**. PyTorch sets its own
  thread count separately.
- **Caddy** compresses responses (`encode zstd gzip`; Plotly JSON shrinks about 6×) and,
  when given a domain name, obtains HTTPS certificates automatically.

---

## Part 6 — Running and sharing it

### 6.1 Requirements

| Way to run | You need |
|---|---|
| Docker (recommended) | Docker Desktop or Docker Engine with Compose v2; ~8 GB RAM for Docker; ~6 GB free disk (image + cache) |
| Plain Python | Python 3.13 (the version the Docker image uses) and the packages in `requirements.txt` |

### 6.2 Docker (recommended)

```bash
docker compose up -d --build     # build and start: app + caddy + tunnel
docker compose ps                # what is running
docker compose logs -f app       # follow app logs
docker compose down              # stop everything (cache volume is kept)
```

**First start, step by step.** The image build takes ~5 minutes (mostly installing
PyTorch). Then the app starts computing:

| Log line | Meaning |
|---|---|
| `Initialising …` / `Running stationarity tests …` | Blocking test battery — site not up yet |
| `All tests ready ✔` | **Site is up.** Open http://localhost |
| `[synth_tab] (k/20) …` | SARIMA fits continuing in the background |
| `[synth3_tab] GENDA trained for …` | GENDA models training in the background |
| `Loaded cached … from .cache/ ✔` | Later starts: everything comes from cache |

**Where the cache lives.** A named volume, `app_cache`, is mounted at the app's `.cache/`
directory, so computed results survive restarts and image rebuilds. To force a full
recompute:

```bash
docker compose down
docker volume rm h2valley-main_app_cache    # the name is prefixed with the project folder
docker compose up -d
```

### 6.3 Plain Python (development)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # includes CPU-only PyTorch
python app.py                          # http://127.0.0.1:8050
```

`app.py` runs Dash's development server with `debug=True`. The debug reloader runs the
whole startup **twice**; use `debug=False` (or gunicorn) for anything long-lived. For the
extended edition run `python app_theory.py` (port 8051).

To fill the cache ahead of time without serving: `python warm_cache.py` (waits for every
background job, then exits).

### 6.4 Letting other people reach it

Pick the option that matches who needs access:

| Option | Who can reach it | Setup | Speed | Trade-offs |
|---|---|---|---|---|
| **Local network** | Same Wi-Fi / office | Nothing (+ firewall rule) | Fast | Your PC must be on |
| **Cloudflare quick tunnel** (included) | Anyone with the link | Nothing | Limited by your upload + relay | Random URL changes on every restart; no uptime guarantee |
| **Router port forwarding** | Anyone | Router access + free DNS name | Your upload speed | Exposes your home IP; fails behind CGNAT |
| **Cloud VM** | Anyone | A VM + `docker compose up` | Data-centre bandwidth | Cost (or a free tier); best for real use |

> ⚠️ The dashboard has **no login**. Anyone with the address can use it.

**Local network.** Find your IPv4 address (`ipconfig` on Windows, `ip a` on Linux) and
share `http://<that-address>`. On Windows, allow port 80 if other devices can't connect
(admin PowerShell):

```powershell
New-NetFirewallRule -DisplayName "h2Valley HTTP" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow -Profile Any
```

**Cloudflare quick tunnel.** The `tunnel` service in `docker-compose.yml` opens an
outbound connection to Cloudflare, with no router changes. Get the current public URL:

```bash
docker compose logs tunnel | grep trycloudflare            # PowerShell: | Select-String trycloudflare
docker compose stop tunnel                                  # turn public access off
docker compose up -d app caddy                              # run without the tunnel at all
```

It uses `--protocol http2` (TCP), because the default QUIC/UDP was throttled to a few KB/s
on a home network. For a permanent URL, create a named tunnel in a free Cloudflare account
with your own domain.

**Router port forwarding.** Forward TCP **80** and **443** to your PC's local IP (reserve
that IP in the router's DHCP settings), point a DNS name at your public IP (e.g. a free
DuckDNS name), then:

```bash
SITE_ADDRESS=yourname.duckdns.org docker compose up -d app caddy
```

Caddy obtains and renews HTTPS certificates automatically.

**Cloud VM.** Copy the project to any Linux VM with Docker (e.g. a free Oracle Cloud ARM
VM with 4 cores and 24 GB RAM is ample), then run the same `docker compose up -d --build`. Add
`SITE_ADDRESS=<your-domain>` for HTTPS. To skip the long first start, copy the cache
volume across, or let it compute once.

---

## Part 7 — Configuration reference

| What | Where | Default / notes |
|---|---|---|
| Analysed dataset | `H2V_DATASET` env var → a config in `configs/` | `configs/spg.yaml` (→ `data/spg.csv`) |
| Columns, target, labels, units, groups, notes | `series:`, `target:`, `include`/`exclude` in the config | all numeric columns except the target |
| Time axis | `time:` in the config (`column`, `freq`, `end`) | no timestamp column → hourly index ending now |
| Tested pairs | `pairs:` in the config (explicit list or `auto`) | 13 cointegration / 14 Granger pairs |
| Default variables in the UI | `defaults:` in the config | temperature; RQA pair shortwave ↔ temperature |
| Map dataset | `XLSX_PATH` in `map_tab.py` | IEA capacity workbook in `data/` |
| Map country filter | `EU_COUNTRIES` in `map_tab.py` | European countries |
| Dominant cycle (all lags scale with it) | `time.period` in the config (samples; omit for auto-detect) | 24 for `spg.csv` |
| ACF/PACF lags | `NLAGS = DS.lag(2)` in `app.py` | 2 cycles = 48 |
| Significance level | `THRES` in `app.py` | 0.05 |
| Granger max lag | `MAX_LAG_GC` in `app.py` | ¼ cycle = 6 |
| Ljung–Box lags / HQC max lag / BG lags | `LB_LAGS`, `HQC_MAX_LAG`, `BG_LAGS` | ¼, ½, 1 cycle = [6, 12, 24] / 12 / 12 |
| SARIMA realisations | `N_VALIDATION` in `synth_tab_control.py` | 6 |
| SARIMA search budget | `SEARCH_BUDGET_S` in `synth_tab_control.py` | 240 s per variable |
| RQA window / target RR | `WINDOW_DEFAULT`, `RR_DEFAULT` in `rqa_tab.py` | 300 h / 10% |
| GENDA training | `GENDA_*` in `synth3_funs.py`; `GENDA_THREADS` in `synth3_tab.py` | 40 epochs; 4 threads |
| Port (Docker) | `PORT` env in `Dockerfile` | 7860 (internal; Caddy publishes 80/443) |
| Public hostname / HTTPS | `SITE_ADDRESS` env for Caddy | `:80` (plain HTTP on any address) |
| Port (dev server) | last line of `app.py` | 8050 |
| Fonts | `EXTERNAL_STYLESHEETS` in `app.py` | IBM Plex from Google Fonts |

Changing the data or any of these values invalidates only the affected cache entries;
the next start recomputes them.

**Analysing another file.** Copy `configs/spg.yaml`, set `path`, `time.column` /
`time.freq` and `target`, and drop the `notes:` (they describe the proxy data only). Then
start with `H2V_DATASET=configs/<name>.yaml` (PowerShell: `$env:H2V_DATASET="configs/<name>.yaml"`),
or set it under `environment:` for the `app` service in `docker-compose.yml`.

Every lag, window and horizon is a multiple of the dataset's dominant cycle (`time.period`,
in samples): ACF to 2 cycles, Ljung–Box at ¼/½/1 cycle, bootstrap blocks up to 4 cycles,
embedding delay up to 1 cycle, and so on. Axis units ("hours", "days", "samples") follow
the data's frequency. The interpretation next to each ACF/PACF is generated from that
dataset's own results; `notes:` in the config are shown underneath as analyst notes.
Each dataset caches its results in its own folder, `.cache/<id>/`.

---

## Part 8 — Performance

Measured on a 14-core Windows PC under Docker Desktop (WSL2). Your numbers will differ.

| Stage | Blocks the site? | First start | Cached start |
|---|---|---|---|
| Image build | — | ~5 min (once) | layer-cached |
| Statistical test battery | **Yes** | a few minutes | seconds |
| RQA identification | No | ~15 s | instant |
| Synthetic Data II identification | No | ~1 min | instant |
| SARIMA fits (20 variables) | No | ~18 min | load ~1.9 GB from disk |
| GENDA training (per variable) | No | ~15–40 s each | instant |

Memory: ~2.5 GB for the app once everything is loaded.

Where the time goes for remote visitors: the first visit downloads Plotly (~1.5 MB
compressed, then cached by the browser), and the heaviest synthetic-data charts are
several MB of JSON before compression. On a home connection, **your upload speed** is the
ceiling for every visitor; a cloud VM removes it.

---

## Part 9 — Troubleshooting

| Symptom | Cause & fix |
|---|---|
| Container keeps restarting; log shows `ModuleNotFoundError` | A dependency is missing from `requirements.txt`. Add it and run `docker compose up -d --build`. |
| Log shows `WORKER TIMEOUT`, then `Initialising …` again | First start exceeded gunicorn's timeout. Raise `--timeout` in the `Dockerfile` `CMD`. |
| Browser shows **502** | The app is still starting (wait for `All tests ready ✔`), or Caddy points at the wrong port: the `Caddyfile` must say `reverse_proxy app:7860`. |
| Startup far slower than expected, CPU at 100% | BLAS thread oversubscription. Make sure `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) are set. |
| Plots never appear over the public link | Usually a crawling tunnel. Confirm `--protocol http2`, restart it (`docker compose restart tunnel`), and use the **new** URL. |
| Old public link stopped working | Quick-tunnel URLs change on every restart. Fetch the current one from `docker compose logs tunnel`. |
| Other devices on the network can't connect | Windows Firewall / "Public" network profile. Add the firewall rule in [6.4](#64-letting-other-people-reach-it). |
| A tab says "fitting in the background" for a long time | Normal on first start. Check `docker compose logs app` for `[synth_tab] (k/20)` progress. |
| Every restart recomputes everything | The cache volume isn't mounted or was removed. Check `docker compose config` shows `app_cache` mounted at `/home/app/h2valley/.cache`. |
| `FileNotFoundError: data/spg.csv` | Run from the repository root, and check the file exists (and isn't excluded in `.dockerignore`). |
| `WinError 1114` when running `python app.py` on Windows | torch must be imported before pandas; keep the `import torch` at the top of `app.py`. |
| Startup runs twice with `python app.py` | Dash's debug reloader. Use `debug=False`. |
| Titles in a plain font | Google Fonts is blocked (offline machine). Cosmetic only; vendor IBM Plex into `assets/` to fix. |

When reporting a bug, include the last ~30 lines of `docker compose logs app` (or the
`python app.py` output) and your OS / Docker / Python versions.

---

## Part 10 — Reproducibility, licensing & limitations

**Reproducibility**
- Statistical tests are deterministic given the data. Simulations use fixed seeds (e.g.
  validation ensembles use `seed=42`) and expose the seed in the UI.
- `requirements.txt` pins every package; the Docker image pins Python 3.13.
- Remember that chart **dates** are synthetic (anchored to the run date); only hourly
  spacing is real.

**Licensing of inputs.** Verify and respect each source's licence before redistributing
derived data. Datasets derived from **Renewables.ninja** carry a **CC BY-NC 4.0**
(non-commercial) restriction, and **IEA** datasets have their own terms of use. Confirm
the provenance and licence of `spg.csv` before publishing any synthetic dataset built on
it. When releasing synthetic data, document the generating method and parameters and
follow the project's Data Management Plan (FAIR principles).

**Known limitations**
- Single site, ~6 months of hourly data: annual seasonality is not observable.
- PV is a proxy; see [1.2](#12-why-a-solar-dataset-stands-in-for-hydrogen) for what does
  not transfer to electrolysers.
- No authentication; single gunicorn worker (by design, due to in-memory model state).
- The SARIMA cache is large (~1.9 GB) and rebuilt from scratch whenever `spg.csv` changes.

> Fill in before publishing: exact source and licence of `spg.csv` and the IEA files,
> project name, funding/grant agreement number, and institutional affiliation.

---

## Glossary

| Term | Meaning |
|---|---|
| **ACF / PACF** | Correlation of a series with its own past (directly / after removing intermediate lags) |
| **AIC / AICc / BIC / HQC** | Model-selection scores that reward fit and penalise complexity (lower is better) |
| **AMI** | Average mutual information; its first minimum picks the embedding delay τ |
| **Attractor** | The region of phase space a dynamical system keeps revisiting |
| **Callback** | A Dash function that recomputes outputs when inputs change |
| **Cointegration** | A stable long-run relationship between two wandering series |
| **Differencing** | Replacing yₜ by yₜ − yₜ₋₁ to remove a unit root |
| **FNN** | False nearest neighbours; picks the embedding dimension m |
| **GENDA** | Generative neighbourhood-based deep autoencoder (Troullinou et al., 2024) |
| **Granger causality** | X helps predict Y beyond Y's own past (predictive, not physical) |
| **Heteroskedasticity** | Variance that changes over time |
| **IAAFT** | Iterative amplitude-adjusted Fourier transform surrogate |
| **Null hypothesis (H₀)** | The default assumption a test tries to reject |
| **p-value** | How surprising the data would be if H₀ were true; reject when < 0.05 |
| **Phase space / embedding** | Rebuilding a system's state from delayed copies of one series |
| **RQA** | Recurrence quantification analysis: measures of how often and how a system revisits states |
| **SARIMA** | Seasonal autoregressive integrated moving-average model |
| **Stationary** | Statistical properties don't change over time |
| **Twin surrogates** | Synthetic trajectories built by jumping between states with identical neighbourhoods |
| **Unit root** | The property that makes shocks permanent (random-walk behaviour) |
| **Reverse proxy** | A server (Caddy) that sits in front of the app and forwards requests to it |
| **Tunnel** | An outbound connection that exposes a local service at a public URL |

---

## Acknowledgements

Developed as part of a research effort on time-series forecasting and synthetic-data
generation for green-hydrogen systems. The PV/meteorological dataset is used as a
structural proxy pending the availability of real operational hydrogen data.
