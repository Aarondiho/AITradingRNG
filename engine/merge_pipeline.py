#!/usr/bin/env python3
# --- repo-root bootstrap (paste at file top, before other imports) ---
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent
# if this file is inside a subfolder (e.g. scripts), climb one level:
if _repo_root.name in ("scripts",):  # adjust if needed
    _repo_root = _repo_root.parent
_repo_root = _repo_root.resolve()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

# engine/merge_pipeline.py
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from .metrics_emitter import MetricsEmitter

logger = logging.getLogger("merge_pipeline")

def _atomic_write_lines(path: Path, lines: List[str]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line if line.endswith("\n") else line + "\n")
    os.replace(tmp, path)

def _atomic_write_obj(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)

class MergePipeline:
    DEFAULT_MAX_ROWS = 10000
    DEFAULT_MAX_SECS = 600

    def __init__(self, symbol: str, base_dir: str, max_rows: int = DEFAULT_MAX_ROWS, max_secs: int = DEFAULT_MAX_SECS):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        self.live_stream_dir = self.base_dir / "live_agg" / "1s"
        self.live_stream_file = self.live_stream_dir / "current.ndjson"
        self.merged_dir = self.base_dir / "merged"
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.merged_dir / "merge_state.json"
        self.max_rows = int(max_rows)
        self.max_secs = int(max_secs)
        # scope metrics under data/<symbol>/metrics by giving the data root and default_symbol
        # self.base_dir is data_root/<symbol>, so parent is data_root
        data_root = self.base_dir.parent
        self.metrics = MetricsEmitter(base_dir=str(data_root), default_symbol=self.symbol)

        logger.info("[MERGE] Initialized for %s", self.symbol)

    def _read_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"last_offset": 0}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("[MERGE] Failed reading state, resetting")
            return {"last_offset": 0}

    def _write_state(self, state: Dict[str, Any]):
        _atomic_write_obj(self.state_path, state)

    def _read_new_lines(self, start_offset: int) -> List[str]:
        if not self.live_stream_file.exists():
            return []
        lines: List[str] = []
        try:
            with self.live_stream_file.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i < start_offset:
                        continue
                    lines.append(line.rstrip("\n"))
        except Exception:
            logger.exception("[MERGE] Failed reading live stream")
        return lines

    def merge_once(self) -> Optional[str]:
        state = self._read_state()
        last_offset = int(state.get("last_offset", 0))
        new_lines = self._read_new_lines(last_offset)
        if not new_lines:
            logger.debug("[MERGE] No new lines to merge for %s", self.symbol)
            return None

        records = []
        for ln in new_lines:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            epoch = obj.get("epoch")
            if epoch is None:
                epoch = obj.get("ts") or obj.get("time") or obj.get("timestamp")
            try:
                epoch = int(epoch) if epoch is not None else None
            except Exception:
                epoch = None
            records.append((epoch, ln))

        if not records:
            new_offset = last_offset + len(new_lines)
            state["last_offset"] = new_offset
            self._write_state(state)
            logger.warning("[MERGE] Read new lines but none parseable for %s; advanced offset", self.symbol)
            return None

        records.sort(key=lambda t: (t[0] if t[0] is not None else 0,))

        created_shard = None
        i = 0
        n = len(records)
        while i < n:
            start_idx = i
            start_epoch = records[start_idx][0] or int(time.time())
            end_idx = start_idx
            while end_idx < n and (end_idx - start_idx) < self.max_rows:
                current_epoch = records[end_idx][0] or start_epoch
                if (current_epoch - start_epoch) > self.max_secs:
                    break
                end_idx += 1
            batch = records[start_idx:end_idx]
            if not batch:
                break

            epochs = [e for e, _ in batch if e is not None]
            tmin = min(epochs) if epochs else int(time.time())
            tmax = max(epochs) if epochs else tmin
            shard_id = f"{self.symbol}_merged_{tmin}_{tmax}"
            out_path = self.merged_dir / f"{shard_id}.ndjson"
            lines = [ln for _, ln in batch]

            try:
                _atomic_write_lines(out_path, lines)
                logger.info("[MERGE] Wrote merged shard %s rows=%d", out_path.name, len(lines))
                try:
                    self.metrics.counter_inc("merge.new_shards", 1, component="merge")
                except Exception:
                    logger.exception("[MERGE] Failed emitting merge metric")

            except Exception:
                logger.exception("[MERGE] Failed writing merged shard %s", out_path)
                i = end_idx
                continue

            manifest = {
                "symbol": self.symbol,
                "shard_file": out_path.name,
                "count": len(lines),
                "epoch_range": [tmin, tmax],
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            try:
                _atomic_write_obj(self.merged_dir / f"{shard_id}.manifest.json", manifest)
            except Exception:
                logger.exception("[MERGE] Failed writing manifest for %s", shard_id)

            created_shard = shard_id
            i = end_idx

        new_offset = last_offset + len(new_lines)
        state["last_offset"] = new_offset
        self._write_state(state)
        logger.info("[MERGE] Advanced offset to %d for %s", new_offset, self.symbol)
        return created_shard
