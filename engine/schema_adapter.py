# engine/schema_adapter.py
"""
Schema adapter that normalizes tick and feature dicts to canonical training schema.

Canonical training tick fields:
  - epoch (int seconds), epoch_ms (int ms optional)
  - price (float, preferred), quote (alias)
  - bid (float), ask (float)
  - symbol (str)

FeatureEngine writes "quote" and Parity/others may use "price". This adapter ensures both exist.

Functions:
  - adapt_tick_dict(d) -> dict (non-destructive copy)
  - adapt_ndjson_file(src_path, dst_path) -> writes normalized NDJSON
"""
import json
from pathlib import Path
from typing import Dict, Any

def adapt_tick_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    # epoch normalization
    if "epoch_ms" in out and "epoch" not in out:
        try:
            out["epoch"] = int(int(out.get("epoch_ms")) // 1000)
        except Exception:
            pass
    # price/quote mapping
    if "price" not in out and "quote" in out:
        try:
            out["price"] = float(out["quote"])
        except Exception:
            out["price"] = out["quote"]
    if "quote" not in out and "price" in out:
        try:
            out["quote"] = float(out["price"])
        except Exception:
            out["quote"] = out["price"]
    # ensure bid/ask numeric if present
    for k in ("bid", "ask"):
        if k in out:
            try:
                out[k] = float(out[k]) if out[k] is not None else None
            except Exception:
                out[k] = None
    # symbol default
    if "symbol" not in out:
        out["symbol"] = out.get("src") or "SYM"
    return out

def adapt_ndjson_file(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as inf, dst_path.open("w", encoding="utf-8") as outf:
        for ln in inf:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            obj2 = adapt_tick_dict(obj)
            outf.write(json.dumps(obj2, ensure_ascii=False) + "\n")
