import asyncio
import logging
from datetime import datetime

from engine.drift_monitor import DriftMonitor

logger = logging.getLogger("phase4_coordinator")


class Phase4Coordinator:
    """
    Orchestrates Phase 4 drift detection:
      - Runs DriftMonitor for each symbol
      - Schedules checks at configured cadence
      - Writes drift reports into reports/drift/<symbol>/
    """

    def __init__(self, symbols, base_dir="data", reports_dir="reports", config=None):
        self.symbols = symbols
        self.base_dir = base_dir
        self.reports_dir = reports_dir
        self.config = config or {}
        self.tasks = []
        self.running = False

    async def _run_drift_loop(self, symbol):
        cadence = self.config.get("cadence_seconds", 1800)
        while self.running:
            try:
                monitor = DriftMonitor(
                    symbol=symbol,
                    base_dir=self.base_dir,
                    reports_dir=self.reports_dir,
                    config=self.config,
                )
                monitor.run_check()
            except Exception:
                logger.exception("[COORD4] Drift check failed for %s", symbol)
            await asyncio.sleep(cadence)

    async def start(self):
        self.running = True
        logger.info("[COORD4] Starting Phase 4 Coordinator at %s", datetime.utcnow().isoformat())

        if self.config.get("enabled", False):
            for sym in self.symbols:
                self.tasks.append(asyncio.create_task(self._run_drift_loop(sym)))

    async def stop(self):
        self.running = False
        for t in self.tasks:
            t.cancel()
        self.tasks = []
        logger.info("[COORD4] Phase 4 Coordinator stopped")

    async def run_once(self):
        """
        Run drift check once for all symbols (manual/debug mode).
        """
        for sym in self.symbols:
            try:
                monitor = DriftMonitor(
                    symbol=sym,
                    base_dir=self.base_dir,
                    reports_dir=self.reports_dir,
                    config=self.config,
                )
                monitor.run_check()
            except Exception:
                logger.exception("[COORD4] Drift check failed for %s", sym)
