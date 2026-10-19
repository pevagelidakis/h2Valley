"""
warm_cache.py  —  Populate .cache/ before the server starts.

Importing `app` runs the statistical test battery and kicks off the
background SARIMA / Class-II / RQA identification threads (see cache_utils.py).
Those threads are daemons, so this script waits for every one of them to
finish and write its cache file; run at image-build time, the container then
boots straight from .cache/ instead of recomputing on every cold start.
"""
import sys
import time

import app_constrol_updataed  # noqa: F401  — import side effects do the work
import rqa_tab
import synth2_tab
import synth_tab_control

TABS = {
    "synth_tab_control": synth_tab_control,
    "synth2_tab": synth2_tab,
    "rqa_tab": rqa_tab,
}

t0 = time.time()
for name, mod in TABS.items():
    while not mod.precompute_status()["finished"]:
        time.sleep(2)
    errors = mod.precompute_status()["errors"]
    if errors:
        print(f"[warm_cache] WARNING: {name} had errors: {errors}", file=sys.stderr)
    print(f"[warm_cache] {name} ready ({time.time() - t0:.0f}s elapsed)")

print("[warm_cache] all caches written ✔")
