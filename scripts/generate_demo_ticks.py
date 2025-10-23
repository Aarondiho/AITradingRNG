#!/usr/bin/env python3
"""
Generate deterministic demo tick NDJSON suitable for Phase1 smoke tests.

Usage:
  python scripts/generate_demo_ticks.py --symbol R_TEST --out-root data --minutes 10 --ticks-per-sec 1 --seed 42

Produces:
  data/<symbol>/live_agg/1s/current.ndjson

Tick format:
  JSON lines with keys: epoch, epoch_ms, quote, price, bid, ask, src
"""
import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone

def generate_ticks(symbol: str, minutes: int, ticks_per_sec: float, seed: int = 0):
    rng = random.Random(seed)
    now = int(time.time())
    # align start to minute boundary - produce minutes contiguous backwards so newest at end
    start_minute = (now // 60) - minutes
    ticks = []
    seq = 0
    price = 100.0 + rng.uniform(-0.5, 0.5)
    for m in range(minutes):
        base_epoch = (start_minute + m) * 60
        # produce ticks across 60 seconds
        total_ticks = max(1, int(round(ticks_per_sec * 60)))
        for i in range(total_ticks):
            # spread uniformly in the minute
            frac = (i + rng.random()) / (total_ticks + 1)
            epoch = int(base_epoch + frac * 60)
            # small random walk
            step = rng.gauss(0, 0.01)
            price = max(0.0001, price + step)
            spread = max(0.0001, abs(rng.gauss(0.01, 0.005)))
            bid = round(price - spread / 2.0, 6)
            ask = round(price + spread / 2.0, 6)
            quote = round((bid + ask) / 2.0, 6)
            # Also include 'price' field for training compatibility
            tick = {
                "symbol": symbol,
                "epoch": epoch,
                "epoch_ms": epoch * 1000,
                "quote": quote,
                "price": quote,
                "bid": bid,
                "ask": ask,
                "src": "demo",
                "schema": "tick.v1",
                "seq": seq
            }
            ticks.append(tick)
            seq += 1
    # ensure sorted by epoch ascending
    ticks.sort(key=lambda t: int(t["epoch"]))
    return ticks

def write_ticks(symbol: str, ticks, out_root: Path):
    target = out_root / symbol / "live_agg" / "1s"
    target.mkdir(parents=True, exist_ok=True)
    out_file = target / "current.ndjson"
    with out_file.open("w", encoding="utf-8") as fh:
        for t in ticks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ticks)} ticks -> {out_file}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="R_TEST")
    p.add_argument("--out-root", default="data")
    p.add_argument("--minutes", type=int, default=10, help="Number of minutes of ticks to synthesize")
    p.add_argument("--ticks-per-sec", type=float, default=1.0, help="Avg ticks per second")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_root = Path(args.out_root)
    ticks = generate_ticks(args.symbol, args.minutes, args.ticks_per_sec, seed=args.seed)
    write_ticks(args.symbol, ticks, out_root)
