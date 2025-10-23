#!/usr/bin/env python3
"""
Run full Phase1 -> Aggregator -> Training smoke cycle for one symbol.

Usage:
  python scripts/run_full_smoke.py --symbol R_TEST --data-root data --exp-root experiment
"""
# --- repo-root bootstrap (paste at file top, before other imports) ---
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent
if _repo_root.name == "scripts":
    _repo_root = _repo_root.parent
_repo_root = _repo_root.resolve()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

import argparse
import subprocess
import sys
import json
from pathlib import Path
import asyncio
from engine.merge_pipeline import MergePipeline
from engine.feature_engine import FeatureEngine
from engine.parity_validator import ParityValidator

def run_merge(symbol: str, base_dir: str):
    mp = MergePipeline(symbol, base_dir)
    shard = mp.merge_once()
    print(f"[SMOKE] merge_once -> {shard}")
    return shard

async def run_feature_from_stream(symbol: str, base_dir: str):
    fe = FeatureEngine(symbol, base_dir)
    src = Path(base_dir) / symbol / "live_agg" / "1s" / "current.ndjson"
    if not src.exists():
        raise FileNotFoundError(src)
    run_id = await fe.process_stream(src, shard_rows=10000, max_secs=600)
    print(f"[SMOKE] FeatureEngine produced run_id -> {run_id}")
    return run_id

def run_parity(symbol: str, base_dir: str, run_id: str):
    pv = ParityValidator(symbol, base_dir)
    run_dir = Path(base_dir) / symbol / "features" / run_id
    rep = pv.run_validation_for_run(run_dir)
    print(f"[SMOKE] Parity decision -> {rep.get('decision') if rep else 'none'}")
    return rep

def run_aggregator(base_dir: str):
    # write per-symbol combined manifests under data/<symbol>/combined
    cmd = [sys.executable, "engine/create_and_validate_combined.py", "--data-root", base_dir, "--out-dir", base_dir]
    print("[SMOKE] Running aggregator:", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    print("[SMOKE] Aggregator exit code:", res.returncode)
    return res.returncode

def run_training_runner(data_root: str, exp_root: str, symbol: str):
    combined_manifest = f"{data_root}/{symbol}/combined/combined_manifest.ndjson"
    cmd = [
            sys.executable,
            "training/runner.py",
            "--combined-manifest",
            combined_manifest,
            "--data-root",
            data_root,
            "--experiment-root",
            f"{exp_root}/{symbol}"
        ]

    print("[SMOKE] Running training runner:", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    print("[SMOKE] Training runner exit code:", res.returncode)
    return res.returncode

def verify_artifacts(base_dir: str, symbol: str, run_id: str):
    base = Path(base_dir) / symbol
    run_dir = base / "features" / run_id
    print("[SMOKE] Inspecting run dir:", run_dir)
    for p in ["*.ndjson", "*.manifest.json", "*.scalers.json", "metadata.json", "parity.json"]:
        found = list(run_dir.glob(p))
        print(f"  {p} -> {len(found)} matches")
    merged = list((Path(base_dir) / symbol / "merged").glob("*.ndjson"))
    print(f"[SMOKE] Merged shards for {symbol}: {len(merged)}")
    combined = Path(base_dir) / symbol / "combined" / "combined_manifest.ndjson"
    print(f"[SMOKE] Combined manifest exists: {combined.exists()}")
    fp_dir = Path(base_dir) / symbol / "fingerprints"
    exp_sym_dir = Path(exp_root := "experiment") / symbol
    print(f"[SMOKE] Symbol fingerprints dir exists: {fp_dir.exists()} (path: {fp_dir})")
    print(f"[SMOKE] Example experiment dir exists: {exp_sym_dir.exists()} (path: {exp_sym_dir})")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="R_TEST")
    p.add_argument("--data-root", default="data")
    p.add_argument("--exp-root", default="experiment")
    return p.parse_args()

async def main():
    args = parse_args()
    symbol = args.symbol
    base = args.data_root
    exp = args.exp_root

    shard = run_merge(symbol, base)

    run_id = await run_feature_from_stream(symbol, base)
    if not run_id:
        print("[SMOKE] FeatureEngine produced no run_id -> abort")
        return 2

    rep = run_parity(symbol, base, run_id)

    verify_artifacts(base, symbol, run_id)

    run_aggregator(base)

    rc = run_training_runner(base, exp, symbol)
    return rc

if __name__ == "__main__":
    asyncio.run(main())
