"""
core/dataset.py  —  One description of the input dataset, shared by every module.

Before this module each tab carried its own copy of the CSV path, the column
list, labels, units, colours and default variables ("kept in sync with app.py —
do not edit independently"). Now a YAML config (configs/<dataset>.yaml) holds
all of it and every module asks for it:

    from core.dataset import get_dataset
    DS = get_dataset()               # the active dataset (H2V_DATASET env var)
    DF = DS.frame()[DS.columns()]    # time-indexed driver columns
    DS.labels[col], DS.units[col], DS.colors[col], DS.pairs("granger"), ...

Swapping the analysed file is then a config change, never a code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/spg.yaml"
ENV_VAR = "H2V_DATASET"

OTHER_GROUP = "Other"
OTHER_COLOR = "#8899bb"
PAIR_KINDS = ("cointegration", "granger")


def _resolve(path: str | Path) -> Path:
    """Relative paths are taken from the project root. If the process runs
    from the project root (the app, Docker, warm_cache.py) the path is kept
    exactly as written, so cache signatures, which include the path string,
    don't change depending on how the path was spelled."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


def _pretty(col: str) -> str:
    return col.replace("_", " ").strip().capitalize()


@dataclass
class DatasetSpec:
    id: str
    name: str
    path: Path
    config_path: Path
    time_column: Optional[str] = None
    freq: Optional[str] = None
    end: Optional[str] = None
    target: Optional[str] = None
    include: Optional[list[str]] = None
    exclude: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)        # group -> colour
    series_meta: dict[str, dict] = field(default_factory=dict)  # col -> {label, group, unit, notes}
    pair_cfg: dict[str, Any] = field(default_factory=dict)
    _frame: Optional[pd.DataFrame] = field(default=None, repr=False)

    # ── data ────────────────────────────────────────────────────────────────
    def frame(self) -> pd.DataFrame:
        """The whole file with a DatetimeIndex (a copy, safe to modify)."""
        if self._frame is None:
            self._frame = self._load()
        return self._frame.copy()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset '{self.id}': {self.path} not found "
                                    f"(configured in {self.config_path})")
        print(f"Loading dataset '{self.id}' from {self.path} …")
        df = pd.read_csv(self.path)

        if self.time_column:
            if self.time_column not in df.columns:
                raise KeyError(f"Dataset '{self.id}': time column '{self.time_column}' "
                               f"not in {self.path}")
            idx = pd.to_datetime(df.pop(self.time_column))
            df.index = pd.DatetimeIndex(idx, name="date")
            df = df.sort_index()
            if df.index.has_duplicates:
                print(f"  WARNING: {int(df.index.duplicated().sum())} duplicate timestamps "
                      f"(kept the first of each)")
                df = df[~df.index.duplicated(keep="first")]
            if self.freq is None:
                self.freq = pd.infer_freq(df.index)
                if self.freq is None:
                    print("  WARNING: timestamps are irregular — no frequency could be inferred")
        else:
            # No timestamps in the file: stamp a regular index of `freq` steps,
            # ending at `end` or at the current hour. `periods=` guarantees the
            # index length matches the row count regardless of rounding.
            freq = self.freq or "h"
            end = (pd.Timestamp(self.end) if self.end
                   else datetime.today().replace(minute=0, second=0, microsecond=0))
            df.index = pd.date_range(end=end, periods=len(df), freq=freq, name="date")
            self.freq = freq
        return df

    @property
    def n_obs(self) -> int:
        return len(self._frame) if self._frame is not None else len(self.frame())

    # ── columns ─────────────────────────────────────────────────────────────
    def _numeric(self) -> list[str]:
        df = self._frame if self._frame is not None else self.frame()
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def columns(self, include_target: bool = False) -> list[str]:
        """Driver columns analysed by the tabs, in file order. With
        `include_target`, the target comes first."""
        numeric = self._numeric()
        if self.include is not None:
            drivers = [c for c in self.include if c in numeric]
        else:
            skip = set(self.exclude) | {self.target, self.time_column}
            drivers = [c for c in numeric if c not in skip]
        drivers = [c for c in drivers if c != self.target]
        if include_target and self.target in numeric:
            return [self.target] + drivers
        return drivers

    # ── metadata (every column present in the file gets an entry) ─────────
    def _meta(self, key: str, fallback) -> dict[str, Any]:
        out = {}
        for c in self._numeric():
            m = self.series_meta.get(c) or {}
            out[c] = m.get(key) if m.get(key) is not None else fallback(c)
        return out

    @property
    def labels(self) -> dict[str, str]:
        return self._meta("label", _pretty)

    @property
    def units(self) -> dict[str, str]:
        return self._meta("unit", lambda c: "")

    @property
    def notes(self) -> dict[str, str]:
        """Hand-written interpretations, only for columns that have one."""
        return {c: m["notes"] for c, m in self.series_meta.items()
                if m and m.get("notes") and c in self._numeric()}

    @property
    def group_of(self) -> dict[str, tuple[str, str]]:
        """col -> (group name, group colour)."""
        out = {}
        for c, g in self._meta("group", lambda c: OTHER_GROUP).items():
            out[c] = (g, self.groups.get(g, OTHER_COLOR))
        return out

    @property
    def colors(self) -> dict[str, str]:
        return {c: gc for c, (_, gc) in self.group_of.items()}

    # ── defaults ────────────────────────────────────────────────────────────
    def default_variable(self) -> str:
        cols = self.columns()
        v = self.defaults.get("variable")
        return v if v in cols else cols[0]

    def default_pair(self) -> tuple[str, str]:
        cols = self.columns()
        pair = list(self.defaults.get("pair") or [])
        a = pair[0] if len(pair) > 0 and pair[0] in cols else cols[0]
        b = pair[1] if len(pair) > 1 and pair[1] in cols and pair[1] != a else \
            next((c for c in cols if c != a), cols[0])
        return a, b

    # ── variable pairs for the bivariate tests ────────────────────────────
    def pairs(self, kind: str) -> list[tuple[str, str]]:
        """Explicit pairs from the config (dropping any column not in the
        drivers), or, for `auto`, the most correlated driver pairs."""
        if kind not in PAIR_KINDS:
            raise ValueError(f"unknown pair kind {kind!r}; expected one of {PAIR_KINDS}")
        spec = self.pair_cfg.get(kind, "auto")
        cols = set(self.columns())
        if spec == "auto" or spec is None:
            return self._auto_pairs(kind)
        out = [tuple(p) for p in spec]
        dropped = [p for p in out if not (p[0] in cols and p[1] in cols)]
        if dropped:
            print(f"  WARNING: {len(dropped)} {kind} pair(s) skipped — columns not in "
                  f"dataset '{self.id}': {dropped}")
        return [p for p in out if p[0] in cols and p[1] in cols]

    def _auto_pairs(self, kind: str) -> list[tuple[str, str]]:
        k = int(self.pair_cfg.get("auto_top_k", 12))
        cols = [c for c in self.columns() if self.frame()[c].std() > 0]
        corr = self.frame()[cols].corr().abs()
        ranked = sorted(combinations(cols, 2), key=lambda p: -np.nan_to_num(corr.loc[p]))
        top = ranked[:k]
        if kind == "granger":
            return [p for a, b in top for p in ((a, b), (b, a))]
        return top


def load_spec(config_path: str | Path) -> DatasetSpec:
    cfg_path = _resolve(config_path)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "path" not in cfg:
        raise KeyError(f"{cfg_path}: 'path' (the data file) is required")
    time_cfg = cfg.get("time") or {}
    spec = DatasetSpec(
        id=str(cfg.get("id") or cfg_path.stem),
        name=str(cfg.get("name") or cfg.get("id") or cfg_path.stem),
        path=_resolve(cfg["path"]),
        config_path=cfg_path,
        time_column=time_cfg.get("column"),
        freq=time_cfg.get("freq"),
        end=time_cfg.get("end"),
        target=cfg.get("target"),
        include=cfg.get("include"),
        exclude=list(cfg.get("exclude") or []),
        defaults=dict(cfg.get("defaults") or {}),
        groups=dict(cfg.get("groups") or {}),
        series_meta=dict(cfg.get("series") or {}),
        pair_cfg=dict(cfg.get("pairs") or {}),
    )
    if spec.target is not None and spec.target not in spec._numeric():
        print(f"  WARNING: target '{spec.target}' is not a numeric column of {spec.path}")
    if not spec.columns():
        raise ValueError(f"Dataset '{spec.id}': no numeric driver columns in {spec.path}")
    return spec


@lru_cache(maxsize=None)
def _cached_spec(config_path: str) -> DatasetSpec:
    return load_spec(config_path)


def get_dataset(config_path: str | Path | None = None) -> DatasetSpec:
    """The active dataset: `config_path`, else $H2V_DATASET, else configs/spg.yaml.
    Loaded once per process, so every module shares the same frame and index."""
    return _cached_spec(str(config_path or os.environ.get(ENV_VAR) or DEFAULT_CONFIG))
