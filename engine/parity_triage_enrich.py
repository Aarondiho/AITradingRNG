# engine/parity_triage_enrich.py
"""
Parity triage enrichment.

Given a parity report and the feature shard, produce:
 - top-k offending rows (by momentum or var)
 - sample of raw rows surrounding detected gaps/non-monotonic epochs
 - a small diagnostics.json saved next to the run folder for operator review

Usage:
  from engine.parity_triage_enrich import enrich_parity
  enrich_parity(run_dir_path, top_k=20)
"""
import json
from pathlib import Path
from statistics import median
from typing import List, Dict, Any

def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

def _top_by_key(rows: List[Dict[str, Any]], key: str, top_k: int = 10):
    scored = []
    for r in rows:
        try:
            v = r.get(key)
            if v is None:
                continue
            scored.append((abs(float(v)), r))
        except Exception:
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    return [r for _, r in scored[:top_k]]

def enrich_parity(run_dir: Path, top_k: int = 20) -> Path:
    manifest = next(run_dir.glob("*.manifest.json"), None)
    if not manifest:
        raise FileNotFoundError("manifest missing in run_dir")
    jm = json.loads(manifest.read_text(encoding="utf-8"))
    shard = run_dir / jm.get("shard_file")
    rows = _read_ndjson(shard)
    diag = {}
    # top offenders by momentum_60, var_60
    diag["top_momentum_60"] = _top_by_key(rows, "momentum_60", top_k)
    diag["top_var_60"] = _top_by_key(rows, "var_60", top_k)
    # sample surrounding non-monotonic epochs if parity.json indicates gaps
    parity_file = run_dir / "parity.json"
    if parity_file.exists():
        p = json.loads(parity_file.read_text(encoding="utf-8"))
        order = p.get("order_issues", {}) or {}
        gaps = []
        if isinstance(order, dict):
            g = None
            # find keys with gap lists
            for it in (p.get("diagnostics") or {}).get("order_issues", []):
                if isinstance(it, dict) and "gaps_seconds_sample" in it:
                    g = it["gaps_seconds_sample"]
                    break
            if g:
                gaps = g
        diag["gaps_sample"] = gaps
    # sample head/tail rows
    diag["head_sample"] = rows[:min(10, len(rows))]
    diag["tail_sample"] = rows[-min(10, len(rows)):] if rows else []
    out = run_dir / "triage_diagnostics.json"
    out.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    return out
