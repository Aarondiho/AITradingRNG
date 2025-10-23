import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("historical_fetcher")


class HistoricalFetcher:
    """
    Historical data fetcher. Assumes the caller provides a connected websocket
    and handles authorization. This class does not (re)connect or close sockets.
    """

    GRANULARITIES = [60, 120, 180, 300, 600, 900, 1800,
                     3600, 7200, 14400, 28800, 86400]
    MAX_COUNT = 5000  # last N ticks for bootstrap

    def __init__(self, app_id: str, api_key: str = None):
        self.app_id = app_id
        self.api_key = api_key

    async def fetch_symbol(self, symbol: str, base_dir: Path, mode: str, ws):
        """
        Fetch historical candles or ticks for a single symbol using ws.send/ws.recv.
        """
        symbol = symbol.strip().replace(" ", "_")
        base_p = Path(base_dir) / symbol / "historical"
        base_p.mkdir(parents=True, exist_ok=True)

        logger.info("[HF] Fetching %s history for %s", mode, symbol)

        if mode == "candles":
            for gran in self.GRANULARITIES:
                await self._fetch_granularity(ws, symbol, gran, base_p, mode)
        elif mode == "ticks":
            # For ticks bootstrap, use a single request for last MAX_COUNT ticks
            await self._fetch_granularity(ws, symbol, gran=0, base_p=base_p, mode=mode)
        else:
            logger.warning("[HF] Unknown mode %s for %s", mode, symbol)

        logger.info("[HF] Completed %s for %s", mode, symbol)

    async def _fetch_granularity(self, ws, symbol: str, gran: int, base_p: Path, mode: str):
        """
        Request and write one granularity shard + manifest.
        mode: "candles" or "ticks"
        """
        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": self.MAX_COUNT,
            "end": "latest",
            "start": 1,
            "style": mode,
            "granularity": gran if mode == "candles" else None
        }
        if mode == "ticks":
            req.pop("granularity")

        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())

        # Parse response
        if mode == "candles" and resp.get("msg_type") == "candles":
            candles = resp.get("candles", [])
            if not candles:
                logger.warning("[HF] No candles for %s g%d", symbol, gran)
                return
            records = [
                {
                    "symbol": symbol,
                    "epoch": c["epoch"],
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "granularity": gran,
                    "src": "deriv:candles"
                }
                for c in candles
            ]
            shard_suffix = f"g{gran}"
        elif mode == "ticks" and resp.get("msg_type") == "history":
            h = resp.get("history", {})
            prices = h.get("prices", [])
            times = h.get("times", [])
            if not prices or not times:
                logger.warning("[HF] No ticks for %s", symbol)
                return
            records = [
                {"symbol": symbol, "epoch": t, "quote": p, "src": "deriv:ticks"}
                for t, p in zip(times, prices)
            ]
            shard_suffix = "last5000"
        else:
            logger.warning("[HF] Unexpected response for %s mode=%s: %s",
                           symbol, mode, resp.get("msg_type"))
            return

        # Write shard + manifest
        tmin = min(r["epoch"] for r in records)
        tmax = max(r["epoch"] for r in records)
        shard_id = f"{symbol}_{mode}_{shard_suffix}_{tmin}_{tmax}"
        shard_path = base_p / f"{shard_id}.ndjson"
        with shard_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        manifest = {
            "symbol": symbol,
            "mode": mode,
            "granularity": gran if mode == "candles" else None,
            "count": len(records),
            "epoch_range": [tmin, tmax],
            "source": f"deriv:{mode}",
            "fetch_ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        (base_p / f"{shard_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("[HF] Wrote %d %s records for %s (%s)", len(records), mode, symbol, shard_suffix)
