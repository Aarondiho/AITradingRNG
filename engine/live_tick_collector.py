import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List

logger = logging.getLogger("live_tick_collector")


class LiveTickCollector:
    """
    Buffers live ticks for one symbol and periodically flushes
    1s ndjson shards with atomic writes (Windows-safe).
    """

    def __init__(self, symbol: str, base_dir: str, flush_interval: float = 1.0):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol / "live_agg" / "1s"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self._buffer: List[dict] = []
        self._task: asyncio.Task | None = None
        self.ticks_received = 0

        

    def add_tick(self, tick: dict):
        """
        Synchronous callback from DerivSocket. Minimal work:
        - normalize epoch if needed
        - append to buffer
        """
        if "epoch_ms" in tick and "epoch" not in tick:
            t = dict(tick)
            t["epoch"] = int(t["epoch_ms"] / 1000)
            tick = t
        self._buffer.append(tick)
        self.ticks_received += 1

    async def start(self):
        """
        Create background flush task.
        """
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._flush_loop())
        logger.info("[LTC] Started flush loop for %s", self.symbol)

    async def stop(self):
        """
        Cancel flush loop and perform final flush.
        """
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._flush_once()
        logger.info("[LTC] Stopped collector for %s", self.symbol)

    async def _flush_loop(self):
        try:
            while True:
                await asyncio.sleep(self.flush_interval)
                await self._flush_once()
        except asyncio.CancelledError:
            await self._flush_once()
            raise

    async def _flush_once(self):
        """
        Atomically write one shard if buffer has records.
        """
        if not self._buffer:
            return
        buf = self._buffer
        self._buffer = []

        # Validate and compute range
        records = [r for r in buf if isinstance(r, dict) and "epoch" in r]
        if not records:
            return
        tmin = min(int(r["epoch"]) for r in records)
        tmax = max(int(r["epoch"]) for r in records)
        shard_id = f"{self.symbol}_live_1s_{tmin}_{tmax}"
        final_path = self.base_dir / f"{shard_id}.ndjson"

        # Atomic write: tmp -> fsync -> replace
        fd, tmp_path = tempfile.mkstemp(prefix=shard_id + "_", dir=str(self.base_dir), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)
           
        except Exception:
            logger.exception("[LTC] Failed writing live shard %s", final_path)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
