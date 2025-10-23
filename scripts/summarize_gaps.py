#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

def extract_epoch(obj):
    for k in ("epoch","ts","time","timestamp"):
        v = obj.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            try:
                dt = datetime.fromisoformat(str(v).replace("Z","+00:00"))
                return int(dt.timestamp())
            except Exception:
                continue
    return None

def load_epochs(shard_path: Path):
    out = []
    with shard_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            e = extract_epoch(obj)
            out.append(e if e is not None else -1)
    return out

def find_gaps(epochs, min_gap=2):
    gaps = []
    for i in range(len(epochs)-1):
        e1 = epochs[i]
        e2 = epochs[i+1]
        if e1 < 0 or e2 < 0:
            continue
        g = e2 - e1
        if g >= min_gap:
            gaps.append((i, e1, e2, g))
    return gaps

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True)
    p.add_argument("--gap", type=int, default=2)
    args = p.parse_args()

    shard = Path(args.shard)
    epochs = load_epochs(shard)
    gaps = find_gaps(epochs, args.gap)
    if not gaps:
        print("No gaps found")
        raise SystemExit(0)

    gap_sizes = [g for _,_,_,g in gaps]
    c = Counter(gap_sizes)
    print("Total rows:", len(epochs))
    print("Total gaps >= %ds: %d" % (args.gap, len(gaps)))
    print("Gap size counts (gap_seconds: count):")
    for gap, cnt in c.most_common():
        print(f"  {gap}: {cnt}")
    print()
    # print first 5 gaps summary
    print("First 5 gaps (index, prev_epoch, next_epoch, gap):")
    for x in gaps[:5]:
        print(" ", x)
    # last 5
    print("Last 5 gaps:")
    for x in gaps[-5:]:
        print(" ", x)
