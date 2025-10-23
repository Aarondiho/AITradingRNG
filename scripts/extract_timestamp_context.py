#!/usr/bin/env python3
"""
Extract timestamp context around detected gaps in a merged shard NDJSON.

Usage:
  python extract_timestamp_context.py --shard data/R_TEST/merged/R_TEST_merged_1761222780_1761223378.ndjson --gap 2 --window 30

Output:
  Prints for each gap found where (next_epoch - prev_epoch) >= gap threshold:
  - context header with gap index, prev_epoch, next_epoch, gap_seconds
  - up to `window` lines: half before and half after the gap (ISO timestamps and raw JSON line)
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

def extract_epoch(obj: dict) -> Optional[int]:
    for k in ("epoch", "ts", "time", "timestamp"):
        v = obj.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            try:
                # sometimes ISO string
                from datetime import datetime
                dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                return int(dt.timestamp())
            except Exception:
                continue
    return None

def load_lines(shard_path: Path) -> List[Tuple[int, str, dict]]:
    out = []
    with shard_path.open("r", encoding="utf-8") as fh:
        for i, ln in enumerate(fh):
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            epoch = extract_epoch(obj)
            out.append((epoch if epoch is not None else -1, ln, obj))
    return out

def find_gaps(rows: List[Tuple[int,str,dict]], gap_threshold: int) -> List[Tuple[int,int,int]]:
    """
    Returns list of tuples (index_prev, epoch_prev, epoch_next) where epoch_next - epoch_prev >= gap_threshold
    index_prev is index of the previous row in rows list.
    """
    gaps = []
    for i in range(len(rows)-1):
        e1 = rows[i][0]
        e2 = rows[i+1][0]
        if e1 < 0 or e2 < 0:
            continue
        if (e2 - e1) >= gap_threshold:
            gaps.append((i, e1, e2))
    return gaps

def print_context(rows: List[Tuple[int,str,dict]], gap_info: Tuple[int,int,int], window: int):
    idx_prev, e1, e2 = gap_info
    idx_next = idx_prev + 1
    half = max(1, window // 2)
    start = max(0, idx_prev - half + 1)
    end = min(len(rows), idx_next + half + 1)
    print("="*80)
    print(f"GAP at rows {idx_prev}->{idx_next} : prev_epoch={e1} next_epoch={e2} gap_seconds={e2-e1}")
    print(f"Context rows [{start} .. {end-1}] (total {end-start})")
    for j in range(start, end):
        epoch, line, obj = rows[j]
        # best-effort ISO representation for epoch
        iso = "N/A"
        if epoch and epoch > 0:
            try:
                from datetime import datetime, timezone
                iso = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
            except Exception:
                iso = str(epoch)
        print(f"[{j}] epoch={epoch} iso={iso} raw={line}")
    print("="*80)
    print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True, help="Path to merged shard NDJSON")
    p.add_argument("--gap", type=int, default=2, help="gap threshold in seconds (inclusive)")
    p.add_argument("--window", type=int, default=30, help="number of context lines to show around each gap")
    p.add_argument("--max-gaps", type=int, default=10, help="max gaps to show")
    args = p.parse_args()

    shard = Path(args.shard)
    if not shard.exists():
        print(f"Shard not found: {shard}")
        raise SystemExit(2)

    rows = load_lines(shard)
    if not rows:
        print("No rows parsed from shard")
        return

    gaps = find_gaps(rows, args.gap)
    if not gaps:
        print(f"No gaps >= {args.gap}s found in shard (parsed {len(rows)} rows)")
        return

    print(f"Found {len(gaps)} gaps >= {args.gap}s (showing up to {args.max_gaps})")
    for i, g in enumerate(gaps[:args.max_gaps]):
        print_context(rows, g, args.window)

if __name__ == "__main__":
    main()
