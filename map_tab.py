"""
map_tab.py  —  EU Hydrogen Production Map tab
Drop this file next to app.py, then follow the three integration steps
marked with  ★  in app.py (search for "MAP TAB").
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS  (kept in sync with app.py — do not edit independently)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#080b12"
SURFACE  = "#0e1219"
CARD     = "#131929"
CARD2    = "#181f30"
BORDER   = "rgba(255,255,255,0.07)"
GRID     = "rgba(255,255,255,0.05)"
TEXT     = "#dde3ee"
MUTED    = "#5a6680"
MUTED2   = "#8899bb"
ACCENT   = "#4f8ef7"
ACCENT2  = "#7eb3ff"
GREEN    = "#2dd4a0"
RED      = "#f4736e"
AMBER    = "#f5a623"
FONT_MONO = "'IBM Plex Mono','Fira Code',monospace"
FONT_SANS = "'IBM Plex Sans','Helvetica Neue',Arial,sans-serif"

# ─────────────────────────────────────────────────────────────────────────────
#  SPREADSHEET PATH  — update to match your actual file
# ─────────────────────────────────────────────────────────────────────────────
XLSX_PATH = Path("Hydrogen production capacity 2024 (Update 2025)_0_0.xlsx")

# ─────────────────────────────────────────────────────────────────────────────
#  STATUS → visual mapping
# ─────────────────────────────────────────────────────────────────────────────
STATUS_CFG = {
    "Operational":              {"color": GREEN,   "symbol": "circle",         "order": 0},
    "Under construction":       {"color": AMBER,   "symbol": "diamond",        "order": 1},
    "Planned / announced":      {"color": ACCENT,  "symbol": "triangle-up",    "order": 2},
    "FID":                      {"color": "#f472b6","symbol": "star",           "order": 3},
    "Concept / feasibility":    {"color": MUTED2,  "symbol": "circle-open",    "order": 4},
    "Decommissioned":           {"color": RED,     "symbol": "x",              "order": 5},
}
DEFAULT_STATUS_CFG = {"color": "#60a5fa", "symbol": "circle", "order": 99}

EU_COUNTRIES = {
    "Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Czech Republic",
    "Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland",
    "Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland",
    "Portugal","Romania","Slovakia","Slovenia","Spain","Sweden",
    # neighbours often included in IEA EU scope
    "Norway","Switzerland","United Kingdom","Iceland","Albania","Serbia",
    "North Macedonia","Montenegro","Bosnia and Herzegovina","Kosovo",
}

# ─────────────────────────────────────────────────────────────────────────────
#  BUILT-IN SAMPLE DATA  (EU subset, IEA-style columns)
# ─────────────────────────────────────────────────────────────────────────────
_SAMPLE_ROWS = [
    # Germany
    {"Project name":"Refhyne","Country":"Germany","City":"Wesseling","Status":"Operational","Electrolyzer capacity (MW)":10,"Technology":"PEM","Hydrogen type":"Green","Start year":2021,"Annual H2 output (t)":1300,"Developer":"Shell / ITM Power","Latitude":50.83,"Longitude":6.97,"Notes":"World's largest PEM at launch"},
    {"Project name":"Hamburg Green Hydrogen Hub","Country":"Germany","City":"Hamburg","Status":"Operational","Electrolyzer capacity (MW)":5,"Technology":"PEM","Hydrogen type":"Green","Start year":2020,"Annual H2 output (t)":730,"Developer":"Hamburg Port Authority","Latitude":53.55,"Longitude":9.93,"Notes":"Port decarbonisation pilot"},
    {"Project name":"GET H2 Nukleus","Country":"Germany","City":"Lingen","Status":"Under construction","Electrolyzer capacity (MW)":100,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2024,"Annual H2 output (t)":15000,"Developer":"RWE","Latitude":52.52,"Longitude":7.32,"Notes":"First node of German H2 grid"},
    {"Project name":"HH2E Lubmin","Country":"Germany","City":"Lubmin","Status":"Planned / announced","Electrolyzer capacity (MW)":500,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2026,"Annual H2 output (t)":73000,"Developer":"HH2E","Latitude":54.13,"Longitude":13.67,"Notes":"Wind-to-H2 industrial park"},
    {"Project name":"HyPilot Falkenhagen","Country":"Germany","City":"Falkenhagen","Status":"Operational","Electrolyzer capacity (MW)":2,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2013,"Annual H2 output (t)":200,"Developer":"Uniper","Latitude":52.47,"Longitude":12.03,"Notes":"Pioneer power-to-gas site"},
    # Netherlands
    {"Project name":"NortH2 Eemshaven","Country":"Netherlands","City":"Eemshaven","Status":"Planned / announced","Electrolyzer capacity (MW)":4000,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2030,"Annual H2 output (t)":400000,"Developer":"Shell / Groningen Seaports","Latitude":53.44,"Longitude":6.85,"Notes":"Offshore wind powered"},
    {"Project name":"HyCC Amsterdam","Country":"Netherlands","City":"Amsterdam","Status":"Planned / announced","Electrolyzer capacity (MW)":200,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2025,"Annual H2 output (t)":29000,"Developer":"HyCC","Latitude":52.38,"Longitude":4.92,"Notes":"Port of Amsterdam integration"},
    {"Project name":"Crosswind NortH2","Country":"Netherlands","City":"Den Helder","Status":"Planned / announced","Electrolyzer capacity (MW)":3000,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2031,"Annual H2 output (t)":300000,"Developer":"Shell / Eneco","Latitude":52.96,"Longitude":4.76,"Notes":"Hollandse Kust Noord wind farm"},
    # Spain
    {"Project name":"Iberdrola Puertollano","Country":"Spain","City":"Puertollano","Status":"Operational","Electrolyzer capacity (MW)":20,"Technology":"PEM","Hydrogen type":"Green","Start year":2021,"Annual H2 output (t)":2400,"Developer":"Iberdrola","Latitude":38.69,"Longitude":-4.11,"Notes":"Largest green-H2 plant in Spain at launch"},
    {"Project name":"HyDeal Ambition","Country":"Spain","City":"Badajoz","Status":"Planned / announced","Electrolyzer capacity (MW)":67000,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2030,"Annual H2 output (t)":3650000,"Developer":"HyDeal","Latitude":38.88,"Longitude":-6.97,"Notes":"Gigascale solar-to-H2 export project"},
    {"Project name":"Hydra Spain","Country":"Spain","City":"Huelva","Status":"Planned / announced","Electrolyzer capacity (MW)":2000,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2027,"Annual H2 output (t)":250000,"Developer":"Fertiberia / Plug Power","Latitude":37.26,"Longitude":-6.95,"Notes":"Green ammonia precursor"},
    # France
    {"Project name":"Lhyfe Saint-Nazaire","Country":"France","City":"Saint-Nazaire","Status":"Operational","Electrolyzer capacity (MW)":1,"Technology":"PEM","Hydrogen type":"Green","Start year":2022,"Annual H2 output (t)":146,"Developer":"Lhyfe","Latitude":47.27,"Longitude":-2.21,"Notes":"First offshore-connected green H2"},
    {"Project name":"H2V Normandy","Country":"France","City":"Port-Jérôme","Status":"Planned / announced","Electrolyzer capacity (MW)":200,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2025,"Annual H2 output (t)":29000,"Developer":"H2V","Latitude":49.47,"Longitude":0.56,"Notes":"Seine estuary industrial cluster"},
    {"Project name":"Hydrogen de France Marseille","Country":"France","City":"Marseille","Status":"Planned / announced","Electrolyzer capacity (MW)":3000,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2030,"Annual H2 output (t)":438000,"Developer":"Hydrogen de France","Latitude":43.30,"Longitude":5.37,"Notes":"Mediterranean hydrogen hub"},
    # Denmark
    {"Project name":"HySynergy Phase 1","Country":"Denmark","City":"Fredericia","Status":"Operational","Electrolyzer capacity (MW)":20,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2021,"Annual H2 output (t)":1600,"Developer":"Everfuel / Crossbridge","Latitude":55.57,"Longitude":9.75,"Notes":"Wind-powered refinery H2"},
    {"Project name":"HySynergy Phase 2","Country":"Denmark","City":"Fredericia","Status":"Planned / announced","Electrolyzer capacity (MW)":300,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2026,"Annual H2 output (t)":23000,"Developer":"Everfuel","Latitude":55.57,"Longitude":9.73,"Notes":"Scale-up of Phase 1"},
    {"Project name":"MegaScale PtX","Country":"Denmark","City":"Esbjerg","Status":"Planned / announced","Electrolyzer capacity (MW)":1000,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2028,"Annual H2 output (t)":146000,"Developer":"Ørsted","Latitude":55.47,"Longitude":8.45,"Notes":"North Sea offshore wind hub"},
    # Norway
    {"Project name":"Yara Porsgrunn","Country":"Norway","City":"Porsgrunn","Status":"Operational","Electrolyzer capacity (MW)":24,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2023,"Annual H2 output (t)":20000,"Developer":"Yara / Nel","Latitude":59.15,"Longitude":9.65,"Notes":"Green ammonia production"},
    {"Project name":"Mongstad H2","Country":"Norway","City":"Mongstad","Status":"Planned / announced","Electrolyzer capacity (MW)":200,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2026,"Annual H2 output (t)":29000,"Developer":"Equinor","Latitude":60.80,"Longitude":5.04,"Notes":"Refinery decarbonisation"},
    # United Kingdom
    {"Project name":"Ynni Glan","Country":"United Kingdom","City":"Holyhead","Status":"Planned / announced","Electrolyzer capacity (MW)":50,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2025,"Annual H2 output (t)":7300,"Developer":"Statkraft","Latitude":53.31,"Longitude":-4.63,"Notes":"Welsh offshore wind H2"},
    {"Project name":"H2H Saltend","Country":"United Kingdom","City":"Hull","Status":"Planned / announced","Electrolyzer capacity (MW)":100,"Technology":"PEM","Hydrogen type":"Low-carbon","Start year":2026,"Annual H2 output (t)":68000,"Developer":"Equinor","Latitude":53.74,"Longitude":-0.28,"Notes":"CCS-enabled blue hydrogen"},
    {"Project name":"Gigastack Blyth","Country":"United Kingdom","City":"Blyth","Status":"Under construction","Electrolyzer capacity (MW)":100,"Technology":"PEM","Hydrogen type":"Green","Start year":2024,"Annual H2 output (t)":14600,"Developer":"ITM Power / Ørsted","Latitude":55.12,"Longitude":-1.49,"Notes":"Offshore wind integration"},
    # Italy
    {"Project name":"Edison Marghera","Country":"Italy","City":"Venice","Status":"Under construction","Electrolyzer capacity (MW)":10,"Technology":"PEM","Hydrogen type":"Green","Start year":2024,"Annual H2 output (t)":1460,"Developer":"Edison","Latitude":45.47,"Longitude":12.23,"Notes":"Industrial port decarbonisation"},
    {"Project name":"Eni Ravenna H2","Country":"Italy","City":"Ravenna","Status":"Planned / announced","Electrolyzer capacity (MW)":250,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2027,"Annual H2 output (t)":36500,"Developer":"Eni","Latitude":44.42,"Longitude":12.20,"Notes":"Offshore wind powered"},
    # Portugal
    {"Project name":"Sines Green Hydrogen","Country":"Portugal","City":"Sines","Status":"Planned / announced","Electrolyzer capacity (MW)":1000,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2027,"Annual H2 output (t)":146000,"Developer":"EDP / Galp","Latitude":37.95,"Longitude":-8.87,"Notes":"Atlantic export terminal"},
    # Belgium
    {"Project name":"Hyoffwind Zeebrugge","Country":"Belgium","City":"Zeebrugge","Status":"Planned / announced","Electrolyzer capacity (MW)":25,"Technology":"PEM","Hydrogen type":"Green","Start year":2025,"Annual H2 output (t)":3650,"Developer":"Eneco / Fluxys","Latitude":51.34,"Longitude":3.20,"Notes":"Offshore wind to port H2"},
    # Sweden
    {"Project name":"FlagshipONE","Country":"Sweden","City":"Örnsköldsvik","Status":"Under construction","Electrolyzer capacity (MW)":70,"Technology":"PEM","Hydrogen type":"Green","Start year":2025,"Annual H2 output (t)":50000,"Developer":"Ørsted / Liquid Wind","Latitude":63.29,"Longitude":18.72,"Notes":"e-methanol production"},
    {"Project name":"H2 Green Steel Boden","Country":"Sweden","City":"Boden","Status":"Under construction","Electrolyzer capacity (MW)":700,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2025,"Annual H2 output (t)":102000,"Developer":"H2 Green Steel","Latitude":65.83,"Longitude":21.69,"Notes":"Green steel decarbonisation"},
    # Finland
    {"Project name":"P2X Solutions Harjavalta","Country":"Finland","City":"Harjavalta","Status":"Operational","Electrolyzer capacity (MW)":20,"Technology":"PEM","Hydrogen type":"Green","Start year":2022,"Annual H2 output (t)":2900,"Developer":"P2X Solutions","Latitude":61.32,"Longitude":22.14,"Notes":"Nordic industrial H2"},
    # Poland
    {"Project name":"Orlen Włocławek","Country":"Poland","City":"Włocławek","Status":"Under construction","Electrolyzer capacity (MW)":1,"Technology":"PEM","Hydrogen type":"Green","Start year":2023,"Annual H2 output (t)":200,"Developer":"Orlen","Latitude":52.65,"Longitude":19.07,"Notes":"Refinery pilot"},
    # Greece
    {"Project name":"Ptolemaida Green H2","Country":"Greece","City":"Ptolemaida","Status":"Planned / announced","Electrolyzer capacity (MW)":200,"Technology":"Alkaline","Hydrogen type":"Green","Start year":2027,"Annual H2 output (t)":29000,"Developer":"PPC","Latitude":40.51,"Longitude":21.67,"Notes":"Coal-to-green energy transition"},
]

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, *kws) -> str | None:
    for kw in kws:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None


def _read_excel_smart(path: Path) -> pd.DataFrame:
    """Robustly read an IEA-style workbook.

    Real IEA exports have several blank leading rows, a blank leading column,
    and multiple sheets — only one of which carries plant coordinates. We scan
    every sheet, locate the true header row (the one declaring Latitude/Longitude,
    or failing that the first row with enough populated cells), drop fully-empty
    columns, and prefer the sheet that actually has coordinates.
    """
    xls = pd.ExcelFile(path)
    fallback = None
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        if raw.empty:
            continue
        hdr = None
        for i in range(min(20, len(raw))):
            row = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            if any("latitud" in v for v in row) and any("longitud" in v for v in row):
                hdr = i
                break
        if hdr is None:  # no coords on this sheet — use first substantive row
            for i in range(min(20, len(raw))):
                if raw.iloc[i].notna().sum() >= 3:
                    hdr = i
                    break
        if hdr is None:
            continue
        df = pd.read_excel(xls, sheet_name=sheet, header=hdr)
        df = df.dropna(axis=1, how="all")                       # drop blank leading column(s)
        df.columns = [str(c).strip() for c in df.columns]        # strip stray whitespace
        df = df.dropna(axis=0, how="all")
        has_coords = (any("latitud" in c.lower() for c in df.columns)
                      and any("longitud" in c.lower() for c in df.columns))
        if has_coords:
            return df          # this is the sheet we want
        if fallback is None:
            fallback = df
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not locate a usable data table in {path}")


def load_eu_hydrogen_df() -> pd.DataFrame:
    """Load from xlsx if available, else return embedded sample."""
    if XLSX_PATH.exists():
        print(f"[map_tab] Reading {XLSX_PATH} …")
        df = _read_excel_smart(XLSX_PATH)
        # filter EU/Europe
        cc = _find_col(df, "country")
        if cc:
            df = df[df[cc].isin(EU_COUNTRIES)].copy()
        print(f"[map_tab] {len(df)} EU rows loaded from spreadsheet")
    else:
        print("[map_tab] xlsx not found — using built-in sample data")
        df = pd.DataFrame(_SAMPLE_ROWS)

    # ── normalise essential columns ──────────────────────────────────────
    lat_c  = _find_col(df, "latitude", "lat")
    lon_c  = _find_col(df, "longitude", "lon", "lng")
    mw_c   = _find_col(df, "electrolyzer capacity", "capacity (mw)", "power (mw)", "mw")
    st_c   = _find_col(df, "status", "stage", "phase")
    nm_c   = _find_col(df, "project name", "name", "project")
    co_c   = _find_col(df, "country")
    ty_c   = _find_col(df, "technology", "tech")
    hy_c   = _find_col(df, "hydrogen type", "h2 type", "type")
    yr_c   = _find_col(df, "start year", "year", "commission")
    dev_c  = _find_col(df, "developer", "company", "operator")

    if not lat_c or not lon_c:
        raise ValueError("No Latitude/Longitude columns found in spreadsheet")

    df["_lat"]  = pd.to_numeric(df[lat_c],  errors="coerce")
    df["_lon"]  = pd.to_numeric(df[lon_c],  errors="coerce")
    df["_mw"]   = pd.to_numeric(df[mw_c],   errors="coerce").fillna(1) if mw_c else 1.0
    df["_mw"]   = df["_mw"].clip(lower=0)
    df["_status"] = df[st_c].fillna("Unknown").astype(str) if st_c else "Unknown"
    df["_name"]   = df[nm_c].fillna("—").astype(str) if nm_c else "—"
    df["_country"]= df[co_c].fillna("—").astype(str) if co_c else "—"
    df["_tech"]   = df[ty_c].fillna("—").astype(str) if ty_c else "—"
    df["_h2type"] = df[hy_c].fillna("—").astype(str) if hy_c else "—"
    df["_year"]   = df[yr_c].fillna("—").astype(str) if yr_c else "—"
    df["_dev"]    = df[dev_c].fillna("—").astype(str) if dev_c else "—"

    df = df.dropna(subset=["_lat", "_lon"])
    return df


MAP_DF = load_eu_hydrogen_df()

# ── summary stats ────────────────────────────────────────────────────────────
_total_mw   = MAP_DF["_mw"].sum()
_total_cap  = f"{_total_mw/1000:.1f} GW" if _total_mw >= 1000 else f"{_total_mw:.0f} MW"
_n_projects = len(MAP_DF)
_n_countries= MAP_DF["_country"].nunique()
_n_op       = (MAP_DF["_status"].str.lower().str.contains("operat")).sum()

# ── status normalisation (used by dropdowns, figures and callbacks) ───────────
def _normalise_status(raw: str) -> str:
    r = str(raw).lower()
    if "operat" in r or "commission" in r or "online" in r:
        return "Operational"
    if "construct" in r or "building" in r or "under" in r:
        return "Under construction"
    if "plan" in r or "announc" in r or "develop" in r or "concept" in r or "feasib" in r:
        return "Planned / announced"
    if "fid" in r:
        return "FID"
    if "decommission" in r:
        return "Decommissioned"
    return raw


# ── dropdowns ────────────────────────────────────────────────────────────────
# Status options must use the *normalised* labels, because update_map() filters
# on _normalise_status(...). Building them from raw values would let a user pick
# an option that never matches.
_all_statuses = sorted(MAP_DF["_status"].apply(_normalise_status).unique())
_all_countries= sorted(MAP_DF["_country"].unique())
_status_opts  = [{"label": "All statuses", "value": "__all__"}] + \
                [{"label": s, "value": s} for s in _all_statuses]
_country_opts = [{"label": "All countries", "value": "__all__"}] + \
                [{"label": c, "value": c} for c in _all_countries]

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _bubble_size(mw: float, max_mw: float) -> float:
    if max_mw <= 0:
        return 8
    return 7 + 30 * math.sqrt(max(mw, 0) / max_mw)


def _hover_text(row: pd.Series) -> str:
    lines = [f"<b>{row['_name']}</b>"]
    skip  = {"_lat","_lon","_mw","_status","_name","_country","_tech","_h2type","_year","_dev"}
    # curated top fields
    for label, key in [
        ("Country",    "_country"),
        ("Status",     "_status"),
        ("Capacity",   "_mw"),
        ("Technology", "_tech"),
        ("H₂ type",    "_h2type"),
        ("Start year", "_year"),
        ("Developer",  "_dev"),
    ]:
        val = row.get(key, "—")
        if val == "—" or (isinstance(val, float) and math.isnan(val)):
            continue
        if key == "_mw":
            val = f"{val:,.1f} MW" if val < 1000 else f"{val/1000:.2f} GW"
        lines.append(f"<span style='color:{MUTED2}'>{label}:</span> {val}")
    # remaining spreadsheet columns
    for col in row.index:
        if col in skip or col.startswith("_") or col.startswith("Unnamed"):
            continue
        val = row[col]
        if pd.isna(val) or val == "" or val == "—":
            continue
        lines.append(f"<span style='color:{MUTED2}'>{col}:</span> {val}")
    return "<br>".join(lines)


def fig_eu_map(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    max_mw = df["_mw"].max() if not df.empty else 1

    # one trace per status group (for legend)
    status_groups = df["_status"].apply(_normalise_status)
    df = df.copy()
    df["_norm_status"] = status_groups

    for norm_st in sorted(df["_norm_status"].unique(),
                          key=lambda s: STATUS_CFG.get(s, DEFAULT_STATUS_CFG)["order"]):
        sub  = df[df["_norm_status"] == norm_st]
        cfg  = STATUS_CFG.get(norm_st, DEFAULT_STATUS_CFG)
        clr  = cfg["color"]
        sym  = cfg["symbol"]
        sizes = [_bubble_size(mw, max_mw) for mw in sub["_mw"]]
        hover = [_hover_text(row) for _, row in sub.iterrows()]

        fig.add_trace(go.Scattergeo(
            lat=sub["_lat"],
            lon=sub["_lon"],
            mode="markers",
            name=norm_st,
            marker=dict(
                size=sizes,
                color=clr,
                symbol=sym,
                opacity=0.88,
                line=dict(width=0.8, color="rgba(0,0,0,0.55)"),
                sizemode="diameter",
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            showlegend=True,
        ))

    # ── Capacity size legend (invisible traces) ───────────────────────────
    size_refs = [(1, "1 MW"), (50, "50 MW"), (500, "500 MW"), (5000, "5 GW")]
    for i, (mw_ref, label) in enumerate(size_refs):
        sz = _bubble_size(mw_ref, max_mw)
        fig.add_trace(go.Scattergeo(
            lat=[None], lon=[None], mode="markers",
            name=label,
            legendgroup="capacity",
            legendgrouptitle=dict(
                text="Capacity",
                font=dict(size=11, color=MUTED2, family=FONT_MONO)
            ) if i == 0 else None,
            marker=dict(
                size=sz, color="rgba(140,160,200,0.3)",
                line=dict(width=1, color="rgba(140,160,200,0.55)"),
                sizemode="diameter",
            ),
            showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT_MONO, color=TEXT, size=11),
        geo=dict(
            bgcolor="#0d1420",
            showland=True,   landcolor="#1a2236",
            showocean=True,  oceancolor="#0d1420",
            showlakes=True,  lakecolor="#0d1420",
            showcountries=True,  countrycolor="rgba(100,130,180,0.30)",
            showcoastlines=True, coastlinecolor="rgba(100,130,180,0.40)",
            showframe=False,
            projection_type="mercator",
            lonaxis_range=[-14, 32],
            lataxis_range=[34, 72],
        ),
        legend=dict(
            title=dict(text="<b>Status</b>",
                       font=dict(size=11, color=MUTED2, family=FONT_MONO)),
            bgcolor="rgba(13,20,32,0.92)",
            bordercolor=BORDER,
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            font=dict(size=11, color=TEXT, family=FONT_MONO),
            tracegroupgap=6,
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        hoverlabel=dict(
            bgcolor=CARD2,
            bordercolor=BORDER,
            font=dict(size=12, color=TEXT, family=FONT_MONO),
            align="left",
        ),
    )
    return fig


def fig_capacity_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar: top N projects by MW."""
    sub = df.nlargest(top_n, "_mw").copy()
    sub = sub.sort_values("_mw")
    norm_st = sub["_status"].apply(_normalise_status)
    colors  = [STATUS_CFG.get(s, DEFAULT_STATUS_CFG)["color"] for s in norm_st]
    labels  = [
        f"{nm} ({co})"
        for nm, co in zip(sub["_name"], sub["_country"])
    ]
    cap_txt = [
        f"{mw/1000:.1f} GW" if mw >= 1000 else f"{mw:.0f} MW"
        for mw in sub["_mw"]
    ]
    hover = [
        f"<b>{nm}</b><br>Country: {co}<br>Status: {st}<br>Capacity: {ct}<extra></extra>"
        for nm, co, st, ct in zip(sub["_name"], sub["_country"],
                                   norm_st, cap_txt)
    ]
    fig = go.Figure(go.Bar(
        x=sub["_mw"],
        y=labels,
        orientation="h",
        marker_color=colors,
        marker_line=dict(color=BORDER, width=0.5),
        text=cap_txt,
        textposition="outside",
        textfont=dict(size=10, color=MUTED2, family=FONT_MONO),
        hovertemplate=hover,
        showlegend=False,
    ))
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=True, zerolinecolor="rgba(255,255,255,0.13)",
        tickfont=dict(size=9, color=MUTED, family=FONT_MONO),
        title=dict(text="Electrolyzer capacity (MW)", standoff=5,
                   font=dict(size=10, color=MUTED2, family=FONT_MONO)),
    )
    fig.update_yaxes(
        tickfont=dict(size=9, color=MUTED2, family=FONT_MONO),
        showgrid=False, linecolor=BORDER,
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font=dict(family=FONT_MONO, color=TEXT, size=11),
        margin=dict(t=36, b=40, l=260, r=80),
        height=max(320, top_n * 26),
        bargap=0.28,
        title=dict(
            text=f"Top {top_n} EU projects by capacity",
            font=dict(size=13, color=TEXT), x=0.01,
        ),
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER,
                        font=dict(size=12, color=TEXT, family=FONT_MONO)),
    )
    return fig


def fig_country_pie(df: pd.DataFrame) -> go.Figure:
    """Donut: total capacity by country."""
    by_country = df.groupby("_country")["_mw"].sum().sort_values(ascending=False)
    # collapse tail into "Other"
    top_n = 10
    if len(by_country) > top_n:
        top = by_country.iloc[:top_n]
        other = pd.Series({"Other": by_country.iloc[top_n:].sum()})
        by_country = pd.concat([top, other])

    palette = [
        ACCENT, GREEN, AMBER, RED, "#a78bfa", "#f472b6",
        "#34d399", "#fb923c", "#60a5fa", "#fbbf24", MUTED2,
    ]
    fig = go.Figure(go.Pie(
        labels=by_country.index,
        values=by_country.values,
        hole=0.52,
        marker=dict(
            colors=palette[:len(by_country)],
            line=dict(color=BG, width=2),
        ),
        textfont=dict(size=10, color=TEXT, family=FONT_MONO),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} MW<br>%{percent}<extra></extra>",
        showlegend=True,
    ))
    total_mw = df["_mw"].sum()
    total_lbl = f"{total_mw/1000:.1f} GW" if total_mw >= 1000 else f"{total_mw:.0f} MW"
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(family=FONT_MONO, color=TEXT, size=11),
        margin=dict(t=36, b=16, l=16, r=16),
        height=340,
        legend=dict(
            font=dict(size=10, color=MUTED2, family=FONT_MONO),
            bgcolor="rgba(0,0,0,0)", x=1.02, y=0.5,
        ),
        annotations=[dict(
            text=f"<b>{total_lbl}</b><br><span style='font-size:10px;color:{MUTED2}'>total</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=TEXT, family=FONT_MONO),
        )],
        title=dict(text="Capacity by country", font=dict(size=13, color=TEXT), x=0.01),
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER,
                        font=dict(size=12, color=TEXT, family=FONT_MONO)),
    )
    return fig


def fig_status_timeline(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: number of projects by start year and status."""
    yr_col = df["_year"].replace("—", np.nan).dropna()
    sub = df[df["_year"] != "—"].copy()
    sub["_yr_num"] = pd.to_numeric(sub["_year"], errors="coerce")
    sub = sub.dropna(subset=["_yr_num"])
    if sub.empty:
        return go.Figure()
    sub["_yr_num"] = sub["_yr_num"].astype(int)
    sub["_ns"] = sub["_status"].apply(_normalise_status)

    years = sorted(sub["_yr_num"].unique())
    fig = go.Figure()
    for norm_st in sorted(sub["_ns"].unique(),
                          key=lambda s: STATUS_CFG.get(s, DEFAULT_STATUS_CFG)["order"]):
        cfg = STATUS_CFG.get(norm_st, DEFAULT_STATUS_CFG)
        counts = sub[sub["_ns"] == norm_st].groupby("_yr_num").size().reindex(years, fill_value=0)
        fig.add_trace(go.Bar(
            x=[str(y) for y in years],
            y=counts.values,
            name=norm_st,
            marker_color=cfg["color"],
            marker_line=dict(color=BORDER, width=0.5),
            hovertemplate=f"<b>{norm_st}</b><br>Year: %{{x}}<br>Projects: %{{y}}<extra></extra>",
        ))
    fig.update_xaxes(
        showgrid=False, linecolor=BORDER, tickangle=0,
        tickfont=dict(size=10, color=MUTED, family=FONT_MONO),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID,
        tickfont=dict(size=10, color=MUTED, family=FONT_MONO),
        title=dict(text="Number of projects", standoff=5,
                   font=dict(size=10, color=MUTED2, family=FONT_MONO)),
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=SURFACE,
        barmode="stack",
        font=dict(family=FONT_MONO, color=TEXT, size=11),
        margin=dict(t=36, b=40, l=52, r=16),
        height=280,
        bargap=0.22,
        legend=dict(
            font=dict(size=10, color=MUTED2, family=FONT_MONO),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", y=-0.22, x=0,
        ),
        title=dict(text="Projects by start year & status", font=dict(size=13, color=TEXT), x=0.01),
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER,
                        font=dict(size=12, color=TEXT, family=FONT_MONO)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT HELPERS  (reused from app.py style)
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_CFG_MAP = dict(
    displayModeBar=True, displaylogo=False,
    modeBarButtonsToRemove=["select2d", "lasso2d"],
    toImageButtonOptions=dict(format="png", filename="eu_h2_map", scale=2),
    scrollZoom=True, responsive=True,
)

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
    cards = [
        html.Div([
            html.Div(str(v), className="sbn-val", style={"color": c}),
            html.Div(l, className="sbn-lbl"),
        ], className="sbn")
        for v, l, c in items
    ]
    return html.Div(cards, className="summary-banner")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab_map = dbc.Tab(
    label="🗺  EU H₂ Map",
    tab_id="tab-map",
    children=html.Div([
        _section_hdr("EU Hydrogen Production Capacity Map"),
        _section_desc(
            "Interactive map of European hydrogen production projects from the IEA 2024 dataset. "
            "Bubble size is proportional to electrolyzer capacity (MW). "
            "Hover over any project to see full spreadsheet data. "
            "Filter by status or country using the controls below."
        ),

        _banner(
            (_n_projects,  "EU projects",          ACCENT2),
            (_total_cap,   "Total capacity",        GREEN),
            (_n_countries, "Countries",             AMBER),
            (_n_op,        "Operational",           GREEN),
            (_n_projects - _n_op, "Pipeline",       ACCENT),
        ),

        # ── Filter controls ───────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Status filter"),
                dcc.Dropdown(
                    id="map-status-filter",
                    options=_status_opts,
                    value="__all__",
                    clearable=False,
                    style={"minWidth": "220px"},
                ),
            ], className="ctrl-block"),
            html.Div([
                html.Label("Country filter"),
                dcc.Dropdown(
                    id="map-country-filter",
                    options=_country_opts,
                    value="__all__",
                    clearable=False,
                    style={"minWidth": "200px"},
                ),
            ], className="ctrl-block"),
            html.Div([
                html.Label("Top N bar chart"),
                dcc.Slider(
                    id="map-topn-slider",
                    min=5, max=min(30, _n_projects), step=5,
                    value=min(20, _n_projects),
                    marks={i: str(i) for i in range(5, min(31, _n_projects+1), 5)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], className="ctrl-block-sm"),
        ], className="controls-row"),

        # ── Main map ──────────────────────────────────────────────────────
        _plot_card([dcc.Graph(id="map-main", config=PLOTLY_CFG_MAP)],
                   label="Scroll to zoom  ·  drag to pan  ·  hover for details"),

        # ── Analytics row ─────────────────────────────────────────────────
        html.Div([
            html.Div(
                _plot_card([dcc.Graph(id="map-country-pie", config=PLOTLY_CFG_MAP)],
                           label="Capacity by country"),
                style={"flex": "1", "minWidth": "320px"},
            ),
            html.Div(
                _plot_card([dcc.Graph(id="map-timeline", config=PLOTLY_CFG_MAP)],
                           label="Projects by start year"),
                style={"flex": "2", "minWidth": "420px"},
            ),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start",
                  "flexWrap": "wrap"}),

        # ── Capacity bar chart ────────────────────────────────────────────
        _plot_card([dcc.Graph(id="map-capacity-bar", config=PLOTLY_CFG_MAP)],
                   label="Top projects by electrolyzer capacity"),

    ], style={"maxWidth": "1400px", "margin": "0 auto"}),
)


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  — register with your app via register_map_callbacks(app)
# ─────────────────────────────────────────────────────────────────────────────

def register_map_callbacks(app):

    @app.callback(
        Output("map-main",          "figure"),
        Output("map-capacity-bar",  "figure"),
        Output("map-country-pie",   "figure"),
        Output("map-timeline",      "figure"),
        Input("map-status-filter",  "value"),
        Input("map-country-filter", "value"),
        Input("map-topn-slider",    "value"),
    )
    def update_map(status_filter, country_filter, top_n):
        df = MAP_DF.copy()
        if status_filter and status_filter != "__all__":
            df = df[df["_status"].apply(_normalise_status) == status_filter]
        if country_filter and country_filter != "__all__":
            df = df[df["_country"] == country_filter]
        if df.empty:
            empty = go.Figure()
            empty.update_layout(paper_bgcolor=BG, plot_bgcolor=SURFACE,
                                font=dict(color=TEXT))
            return empty, empty, empty, empty
        return (
            fig_eu_map(df),
            fig_capacity_bar(df, top_n=int(top_n or 20)),
            fig_country_pie(df),
            fig_status_timeline(df),
        )