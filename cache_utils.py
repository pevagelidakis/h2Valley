"""
cache_utils.py  —  Shared on-disk memoization for expensive startup precompute.

Every heavy block in this app (app.py's stationarity / cointegration /
Granger-causality / diagnostics battery, the per-variable SARIMA fits in
synth_tab_control.py, the Class-2 identification in synth2_tab.py, and the
phase-space / RQA identification in rqa_tab.py) is a pure function of its
source file(s) plus a handful of config constants. None of it needs to be
redone on every process start: each caller derives a signature from exactly
the inputs its computation depends on, and this module stores/retrieves the
result under that signature in `.cache/` next to this file.

    sig = make_signature(CSV_PATH, version="app-precompute-v1", nlags=NLAGS)
    cached = load_cache("app_precompute", sig)
    if cached is not None:
        result = cached                       # cache hit — nothing to redo
    else:
        result = _run_expensive_battery()      # cache miss — first run, or
        save_cache("app_precompute", sig, result)   # source/config changed

The signature folds in each source file's (path, size, mtime) so editing
spg.csv invalidates the cache automatically, and an explicit `version` tag
so a caller can bust its own cache on a logic change without touching disk
by hand. Results are stored per dataset (.cache/<dataset id>/), so several
datasets can be analysed side by side. Delete `.cache/` (or one dataset's
folder) at any time to force a recompute from scratch.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _file_fingerprint(path) -> tuple:
    p = Path(path)
    try:
        stat = p.stat()
        return (str(p), stat.st_size, int(stat.st_mtime))
    except FileNotFoundError:
        return (str(p), None, None)


def make_signature(*source_files, version: str = "v1", **config) -> str:
    """Stable hash of the given source files' (path, size, mtime) plus any
    keyword config (constants, column lists, pair lists, ...) plus an
    explicit version tag. Two calls with equal arguments always agree;
    changing any input changes the signature."""
    fingerprints = tuple(_file_fingerprint(f) for f in source_files)
    config_items = sorted(config.items(), key=lambda kv: kv[0])
    payload = pickle.dumps((version, fingerprints, config_items),
                            protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()[:24]


def _path_for(name: str) -> Path:
    """Each dataset gets its own folder (.cache/<dataset id>/), so switching
    H2V_DATASET never overwrites another dataset's results."""
    from core.dataset import get_dataset
    return CACHE_DIR / get_dataset().id / f"{name}.pkl"


def _legacy_path_for(name: str) -> Path:
    """Where caches lived before they were split per dataset (.cache/<name>.pkl)."""
    return CACHE_DIR / f"{name}.pkl"


def _read(path: Path, signature: str) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            stored_sig, payload = pickle.load(fh)
    except Exception:
        return None
    if stored_sig != signature:
        return None
    return payload


def load_cache(name: str, signature: str) -> Optional[Any]:
    """Return the payload stored under `name` if its saved signature matches
    `signature`; otherwise None (missing, unreadable, or stale — caller
    should recompute). A matching cache in the old flat layout is moved into
    the dataset's folder, so existing results are reused, not recomputed."""
    path = _path_for(name)
    payload = _read(path, signature)
    if payload is not None:
        return payload
    legacy = _legacy_path_for(name)
    payload = _read(legacy, signature)
    if payload is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        legacy.replace(path)
        print(f"[cache_utils] moved '{name}' into {path.parent.name}/")
    return payload


def save_cache(name: str, signature: str, payload: Any) -> None:
    """Persist `payload` under `name`, tagged with `signature`. Writes to a
    temp file and renames atomically, so a crash mid-write can never leave
    behind a corrupt file that a later run mistakes for a valid cache."""
    path = _path_for(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    t0 = time.time()
    with open(tmp, "wb") as fh:
        pickle.dump((signature, payload), fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    print(f"[cache_utils] saved '{name}' "
          f"({path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
