#!/usr/bin/env python3
"""
Aggregator: create_and_validate_combined.py

Scans per-symbol run folders (data/<symbol>/features/<run_id>) and appends
canonical combined-manifest entries to a per-symbol append-only NDJSON at
data/<symbol>/combined/combined_manifest.ndjson.

Operates per-symbol only. If --symbols is provided (comma list), aggregate only those symbols.
If not provided, discover symbols under data-root and process each symbol in turn.
For each symbol prints a short console summary (accepted/skipped).
"""
# --- repo-root bootstrap (paste at file top, before other imports) ---
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent
if _repo_root.name in ("scripts",):
    _repo_root = _repo_root.parent
_repo_root = _repo_root.resolve()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from metrics_emitter import MetricsEmitter
from lock_leader import FileLock

# Helper: minimal atomic append and atomic write helpers
def atomic_append_line(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")

def atomic_write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

def discover_symbols(data_root: Path) -> List[str]:
    out: List[str] = []
    if not data_root.exists():
        return out
    for p in sorted(data_root.iterdir()):
        if p.is_dir():
            # Heuristic: symbol directory contains "features" subdir
            if (p / "features").exists():
                out.append(p.name)
    return out

def discover_runs_for_symbol(data_root: Path, symbol: str) -> List[Path]:
    out: List[Path] = []
    symbol_dir = data_root / symbol
    feat = symbol_dir / "features"
    if not feat.exists():
        return out
    for run_dir in feat.iterdir():
        if run_dir.is_dir():
            out.append(run_dir)
    return sorted(out, key=lambda p: p.stat().st_mtime)

def build_entry_from_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Build a combined-manifest entry for a run directory.
    Expects run_dir contains a run manifest JSON (*.manifest.json) or metadata.json.
    If neither present, returns a minimal best-effort entry.
    """
    try:
        manifest = None
        for p in run_dir.glob("*.manifest.json"):
            manifest = p
            break
        if manifest and manifest.exists():
            jm = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            meta = run_dir / "metadata.json"
            if meta.exists():
                jm = json.loads(meta.read_text(encoding="utf-8"))
            else:
                jm = {
                    "run_id": run_dir.name,
                    "shard_file": next(iter([str(p.name) for p in run_dir.glob("*.ndjson")]), None),
                }

        shard_name = jm.get("shard_file") or jm.get("shard") or None
        shard_rel = None
        if shard_name:
            try:
                shard_rel = str((run_dir / shard_name).resolve().relative_to(Path.cwd().resolve()))
            except Exception:
                shard_rel = str((run_dir / shard_name).resolve())

        symbol_name = jm.get("symbol") or (run_dir.parent.parent.name if run_dir.parent and run_dir.parent.parent else "SYM")

        entry = {
            "run_id": jm.get("run_id") or run_dir.name,
            "symbol": symbol_name,
            "produced_at": jm.get("produced_at") or datetime.utcnow().isoformat() + "Z",
            "files": [],
            "run_manifest": str(manifest) if manifest else None,
            "shard_file": shard_name
        }

        if shard_rel:
            entry["files"].append({"type": "ndjson", "path": shard_rel})
        for fname in ("scalers.json", "regime.json"):
            p = run_dir / fname
            if p.exists():
                try:
                    entry["files"].append({"type": fname.replace(".json", ""), "path": str(p.resolve().relative_to(Path.cwd().resolve()))})
                except Exception:
                    entry["files"].append({"type": fname.replace(".json", ""), "path": str(p.resolve())})

        entry["source_run_dir"] = str(run_dir)
        return entry
    except Exception:
        return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data", help="Root data directory to scan (default: data)")
    p.add_argument("--out-dir", default=None, help="Ignored for per-symbol mode; kept for CLI compatibility")
    p.add_argument("--index-file", default=None, help="Optional per-symbol index file path template (unused by default)")
    p.add_argument("--force", action="store_true", help="Force re-indexing of all runs (ignore index files)")
    p.add_argument("--symbols", default=None, help="Comma-separated list of symbols to process (default: discover all under data-root)")
    return p.parse_args()

def process_symbol(symbol: str, data_root: Path, force: bool):
    """
    Process one symbol: discover runs, append entries to data/<symbol>/combined/combined_manifest.ndjson,
    and maintain a per-symbol processed index at data/<symbol>/combined/combined_index.json.
    Prints symbol summary to console.
    """
    symbol_dir = data_root / symbol
    symbol_combined_dir = symbol_dir / "combined"
    symbol_combined_dir.mkdir(parents=True, exist_ok=True)
    symbol_combined_nd = symbol_combined_dir / "combined_manifest.ndjson"
    symbol_combined_json = symbol_combined_dir / "combined_manifest.json"
    indexed_file = symbol_combined_dir / "combined_index.json"

    processed = set()
    if indexed_file.exists() and not force:
        try:
            processed = set(json.loads(indexed_file.read_text(encoding="utf-8")))
        except Exception:
            processed = set()

    runs = discover_runs_for_symbol(data_root, symbol)
    accepted: List[Dict[str, Any]] = []
    skipped: List[str] = []

    # Leader lock per-symbol to avoid cross-process races
    lock_path = str(symbol_combined_dir / "aggregator.lock")
    with FileLock(lock_path, ttl=60) as lk:
        if not lk.acquired:
            print(f"Aggregator [{symbol}]: another process holds lock; skipping symbol")
            return {"symbol": symbol, "accepted": 0, "skipped": len(runs)}

        metrics = MetricsEmitter(base_dir=str(data_root), default_symbol=symbol)

        for run_dir in runs:
            run_id = run_dir.name
            if not force and run_id in processed:
                skipped.append(run_id)
                continue

            entry = build_entry_from_run(run_dir)
            if not entry:
                skipped.append(run_id)
                continue

            try:
                line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
                atomic_append_line(symbol_combined_nd, line)
                accepted.append(entry)
                try:
                    metrics.counter_inc("aggregator.accepted_runs", 1, component="aggregator")
                except Exception:
                    pass
                processed.add(run_id)
            except Exception:
                skipped.append(run_id)
                continue

        # rebuild per-symbol combined_json index (best-effort)
        try:
            entries = []
            if symbol_combined_nd.exists():
                with symbol_combined_nd.open("r", encoding="utf-8") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            entries.append(json.loads(ln))
                        except Exception:
                            continue
            atomic_write_json(symbol_combined_json, {"produced_at": datetime.utcnow().isoformat() + "Z", "n_entries": len(entries), "entries_sample": entries[-10:]})
        except Exception:
            pass

        # persist per-symbol processed index
        try:
            atomic_write_json(indexed_file, sorted(list(processed)))
        except Exception:
            pass

    print(f"Aggregator [{symbol}]: accepted {len(accepted)} runs, skipped {len(skipped)}")
    return {"symbol": symbol, "accepted": len(accepted), "skipped": len(skipped)}

def main():
    args = parse_args()
    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"data-root not found: {data_root}")
        return

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = discover_symbols(data_root)

    if not symbols:
        print("No symbols discovered under data-root; nothing to do")
        return

    summaries = []
    for symbol in symbols:
        summaries.append(process_symbol(symbol, data_root, force=args.force))

    # Print concise multi-symbol summary
    for s in summaries:
        print(f"Summary [{s['symbol']}]: accepted={s['accepted']} skipped={s['skipped']}")

if __name__ == "__main__":
    main()
