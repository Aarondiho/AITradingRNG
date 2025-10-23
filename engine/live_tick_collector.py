# engine/live_tick_collector.py
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List
from datetime import datetime, timezone

logger = logging.getLogger("live_tick_collector")


class LiveTickCollector:
    def __init__(self, symbol: str, base_dir: str,
                 flush_interval: float = 1.0,
                 stream_file_name: str = "current.ndjson"):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol / "live_agg" / "1s"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = float(flush_interval)
        self._buffer: List[dict] = []
        self._task = None
        self.ticks_received = 0

        self.target_file = self.base_dir / stream_file_name
        self.manifest_file = self.base_dir / (Path(stream_file_name).stem + ".manifest.json")

    @staticmethod
    def normalize_live_tick(obj: dict) -> dict:
        bid = obj.get("bid")
        ask = obj.get("ask")
        if "quote" not in obj and bid is not None and ask is not None:
            try:
                obj["quote"] = (float(bid) + float(ask)) / 2.0
            except Exception:
                pass

        if "quote" in obj:
            q = obj["quote"]
            obj["open"] = obj.get("open", q)
            obj["high"] = obj.get("high", q)
            obj["low"] = obj.get("low", q)
            obj["close"] = obj.get("close", q)
            obj["count"] = obj.get("count", 1)

        if bid is not None:
            try:
                obj["bid"] = float(bid)
            except Exception:
                obj.pop("bid", None)
        if ask is not None:
            try:
                obj["ask"] = float(ask)
            except Exception:
                obj.pop("ask", None)

        return obj

    def add_tick(self, tick: dict):
        if "epoch_ms" in tick and "epoch" not in tick:
            tick = dict(tick)
            try:
                tick["epoch"] = int(tick["epoch_ms"] // 1000)
            except Exception:
                tick["epoch"] = int(tick["epoch_ms"]) // 1000 if isinstance(tick["epoch_ms"], (int, float)) else None

        if tick.get("schema") == "tick.v1" or "quote" in tick:
            tick = self.normalize_live_tick(tick)

        self._buffer.append(tick)
        self.ticks_received += 1

    async def start(self):
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._flush_loop())
        logger.info("[LTC] Started flush loop for %s", self.symbol)

    async def stop(self):
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
        if not self._buffer:
            return
        buf = self._buffer
        self._buffer = []

        records = [r for r in buf if isinstance(r, dict) and "epoch" in r and r.get("epoch") is not None]
        if not records:
            return

        fd, tmp_path = tempfile.mkstemp(prefix="flush_", dir=str(self.base_dir), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            with open(self.target_file, "a", encoding="utf-8") as target:
                with open(tmp_path, "r", encoding="utf-8") as source:
                    for line in source:
                        target.write(line)
                target.flush()
                os.fsync(target.fileno())

            os.remove(tmp_path)

            manifest = {
                "symbol": self.symbol,
                "stream_file": self.target_file.name,
                "last_update": datetime.now(tz=timezone.utc).isoformat(),
                "appended_rows": len(records),
                "total_ticks": self.ticks_received,
            }
            tmp_manifest = self.manifest_file.with_suffix(".json.tmp")
            tmp_manifest.write_text(json.dumps(manifest, indent=2))
            os.replace(tmp_manifest, self.manifest_file)

        except Exception:
            logger.exception("[LTC] Failed appending live ticks to %s", self.target_file)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
