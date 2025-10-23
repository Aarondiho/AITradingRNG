import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import websockets

from api.deriv_socket import DerivSocket
from engine.historical_fetcher import HistoricalFetcher
from engine.live_tick_collector import LiveTickCollector
from engine.merge_pipeline import MergePipeline
from engine.feature_engine import FeatureEngine
from engine.parity_validator import ParityValidator

logger = logging.getLogger("launch_coordinator")


class LaunchCoordinator:
    """
    Orchestrates Phase 1A with bootstrap:
    1) Connect raw websocket and authorize
    2) Fetch historical candles for all symbols
    3) Fetch last 5000 ticks for all symbols
    4) Close raw websocket
    5) Immediately run bootstrap merge -> features -> parity (uses freshly fetched ticks)
    6) Start live collectors
    7) Create DerivSocket, register collectors, await readiness
    8) Periodically run merge -> features -> parity
    """

    DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"

    


    def __init__(self, symbols, base_dir: str, app_id: str, api_key: Optional[str] = None,historical_mode: str = "candles", bootstrap_ticks: bool = True):
        self.symbols = [s.strip().replace(" ", "_") for s in symbols]
        self.base_dir = Path(base_dir).resolve()
        self.app_id = app_id
        self.api_key = api_key

        # Per-symbol modules
        self.historical = {}
        self.live_collectors = {}
        self.mergers = {}
        self.features = {}
        self.validators = {}

        self.historical_mode = historical_mode
        self.bootstrap_ticks = bootstrap_ticks

        for sym in self.symbols:
            self.historical[sym] = HistoricalFetcher(self.app_id, self.api_key)
            self.live_collectors[sym] = LiveTickCollector(sym, str(self.base_dir))
            self.mergers[sym] = MergePipeline(sym, str(self.base_dir))
            self.features[sym] = FeatureEngine(sym, str(self.base_dir))
            self.validators[sym] = ParityValidator(sym, str(self.base_dir))

        self._socket: Optional[DerivSocket] = None
        logger.info("[COORD]  for symbols: %s", self.symbols)

    async def start(self):
        logger.info("[COORD] Starting Phase 1A pipeline (with bootstrap option)")

        raw_ws = await self._open_raw_ws_for_history()

        try:
            # Fetch configured historical mode (candles or ticks)
            await self._fetch_historical_all(ws=raw_ws, mode=self.historical_mode)
            logger.info("[COORD] Historical %s fetched for all symbols", self.historical_mode)

            # Bootstrap: fetch last 5000 ticks if enabled
            if self.bootstrap_ticks:
                await self._fetch_historical_all(ws=raw_ws, mode="ticks")
                logger.info("[COORD] Bootstrap ticks (last 5000) fetched for all symbols")
        finally:
            try:
                await raw_ws.close()
                logger.info("[COORD] Closed raw historical websocket")
            except Exception:
                logger.exception("[COORD] Error closing raw historical websocket")

        # If bootstrap ticks were fetched, run an immediate merge/features/validation
        if self.bootstrap_ticks:
            await self._bootstrap_merge_features_validate()

        await self._start_live_all()
        await self._create_register_and_start_live_socket()
        logger.info("[COORD] Startup complete: live ticks flowing")

    async def stop(self):
        """
        Graceful shutdown: stop live socket then collectors.
        """
        logger.info("[COORD] Stopping Phase 1A pipeline")
        if self._socket:
            try:
                await self._socket.stop()
            except Exception:
                logger.exception("[COORD] Error stopping DerivSocket")
        for sym, collector in self.live_collectors.items():
            try:
                await collector.stop()
            except Exception:
                logger.exception("[COORD] Error stopping collector %s", sym)
        logger.info("[COORD] Pipeline stopped")

    async def _open_raw_ws_for_history(self):
        """
        Connect a raw websocket and authorize once for historical fetch.
        """
        url = f"{self.DERIV_WS_URL}?app_id={self.app_id}"
        logger.info("[COORD] Opening raw WebSocket for historical fetch")
        ws = await websockets.connect(url)
        if self.api_key:
            await ws.send(json.dumps({"authorize": self.api_key}))
            auth = json.loads(await ws.recv())
            logger.info("[COORD] Historical WS authorized: %s", auth.get("msg_type"))
        return ws

    async def _fetch_historical_all(self, ws, mode: str = "candles"):
        logger.info("[COORD] Fetching historical (%s) for all symbols", mode)
        for sym in self.symbols:
            try:
                await self.historical[sym].fetch_symbol(sym, self.base_dir, mode=mode, ws=ws)
                logger.info("[COORD] Historical (%s) fetch succeeded for %s", mode, sym)
            except Exception as e:
                logger.exception("[COORD] Historical (%s) fetch failed for %s: %s", mode, sym, e)
        logger.info("[COORD] Historical (%s) fetch complete", mode)


    async def _bootstrap_merge_features_validate(self):
        """
        Immediately run a single merge -> features -> parity cycle using fetched ticks.
        Ensures downstream artifacts exist before live ticks accumulate.
        """
        logger.info("[COORD] Running bootstrap merge + features + validation")
        for sym in self.symbols:
            try:
                self.mergers[sym].run_merge()
            except Exception:
                logger.exception("[COORD] Bootstrap merge failed for %s", sym)
            try:
                self.features[sym].run_features()
            except Exception:
                logger.exception("[COORD] Bootstrap features failed for %s", sym)
            try:
                self.validators[sym].run_validation()
            except Exception:
                logger.exception("[COORD] Bootstrap parity validation failed for %s", sym)
        logger.info("[COORD] Bootstrap cycle complete")

    async def _start_live_all(self):
        """
        Start background flush loops for all collectors.
        """
        logger.info("[COORD] Starting live collectors")
        for sym, collector in self.live_collectors.items():
            try:
                await collector.start()
            except Exception:
                logger.exception("[COORD] Failed to start collector %s", sym)

    async def _create_register_and_start_live_socket(self):
        """
        Create DerivSocket, register per-symbol callbacks, start and await readiness.
        """
        logger.info("[COORD] Creating and starting DerivSocket for live ticks")
        self._socket = DerivSocket(self.app_id, self.api_key)
        for sym, collector in self.live_collectors.items():
            self._socket.register(sym, collector.add_tick)
        try:
            await self._socket.start()  # waits until ready
            logger.info("[COORD] DerivSocket ready and streaming")
        except Exception:
            logger.exception("[COORD] Failed to start DerivSocket")
            raise

    async def run_merge_features_validate(self):
        """
        Periodically run merge -> features -> parity for each symbol.
        Adds a small stabilization delay to avoid file races.
        """
        logger.info("[COORD] Running merge + features + validation")
        for sym in self.symbols:
            await asyncio.sleep(0.5)  # small stabilization
            try:
                self.mergers[sym].run_merge()
            except Exception:
                logger.exception("[COORD] Merge failed for %s", sym)
            try:
                self.features[sym].run_features()
            except Exception:
                logger.exception("[COORD] Feature engine failed for %s", sym)
            try:
                self.validators[sym].run_validation()
            except Exception:
                logger.exception("[COORD] Parity validation failed for %s", sym)
        logger.info("[COORD] Merge + features + validation complete")
