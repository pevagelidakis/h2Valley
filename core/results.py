"""
core/results.py  —  One storage layout for every model's results.

    results/<dataset id>/<model>/
        manifest.json      what is inside, and the signature it was built from
        <table>.parquet    tidy/columnar tables (realisations, validation, forecasts …)
        <name>.json        small structured metadata (fitted parameters, orders …)

The web app only ever reads these files; it never needs the fitted Python
objects that produced them. A model's results are replaced atomically: they
are written to a temporary folder first and swapped in only when complete,
so a crash mid-write never leaves a half-written run that looks valid.

    store = ResultStore(DS.id, "synth_sarima")
    if store.signature() == sig:                 # results exist for these inputs
        sims = store.read_table("realisations", columns=["t", "realisation", col])
    else:
        ...compute...
        store.write(sig, tables={"realisations": df}, json_files={"models": meta})
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.dataset import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"


class ResultStore:
    def __init__(self, dataset_id: str, model: str, root: Path | str | None = None):
        self.dataset_id = dataset_id
        self.model = model
        self.dir = Path(root or RESULTS_DIR) / dataset_id / model

    # ── reading ─────────────────────────────────────────────────────────────
    def manifest(self) -> Optional[dict]:
        try:
            return json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def signature(self) -> Optional[str]:
        m = self.manifest()
        return m.get("signature") if m else None

    def read_table(self, name: str, columns: Optional[list[str]] = None,
                   filters: Optional[list] = None) -> pd.DataFrame:
        """A Parquet table; `columns` / `filters` (pyarrow syntax) read only a slice."""
        return pd.read_parquet(self.dir / f"{name}.parquet", columns=columns, filters=filters)

    def read_json(self, name: str) -> Any:
        return json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))

    # ── writing ─────────────────────────────────────────────────────────────
    def write(self, signature: str, tables: dict[str, pd.DataFrame] | None = None,
              json_files: dict[str, Any] | None = None, meta: dict | None = None) -> None:
        tables, json_files = tables or {}, json_files or {}
        t0 = time.time()
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.dir.with_name(self.dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        files = {}
        for name, df in tables.items():
            path = tmp / f"{name}.parquet"
            # Float columns (simulations, statistics) are near-random bytes that
            # dictionary encoding only inflates; byte-stream-split + zstd stores
            # them ~40 % smaller and reads faster. Other columns (ids, lags,
            # categories) repeat a lot, so they keep dictionary encoding.
            floats = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
            others = [c for c in df.columns if c not in floats]
            df.to_parquet(path, index=False, compression="zstd",
                          use_dictionary=others if others else False,
                          column_encoding={c: "BYTE_STREAM_SPLIT" for c in floats} or None)
            files[f"{name}.parquet"] = dict(rows=len(df), columns=len(df.columns),
                                            bytes=path.stat().st_size)
        for name, obj in json_files.items():
            path = tmp / f"{name}.json"
            path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            files[f"{name}.json"] = dict(bytes=path.stat().st_size)
        manifest = dict(dataset=self.dataset_id, model=self.model, signature=signature,
                        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        files=files, **(meta or {}))
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
        # swap in: old → .old, tmp → dir, drop .old
        old = self.dir.with_name(self.dir.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if self.dir.exists():
            self.dir.replace(old)
        tmp.replace(self.dir)
        shutil.rmtree(old, ignore_errors=True)
        total = sum(f["bytes"] for f in files.values()) / 1e6
        print(f"[results] wrote {self.dataset_id}/{self.model} "
              f"({len(files)} files, {total:.1f} MB, {time.time() - t0:.1f}s)")
