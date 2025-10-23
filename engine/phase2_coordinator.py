import asyncio
import logging
from datetime import datetime

from engine.adversarial_validator import AdversarialValidator
from engine.fingerprint_analyzer import FingerprintAnalyzer

logger = logging.getLogger("phase2_coordinator")


class Phase2Coordinator:
    """
    Orchestrates Phase 2 modules:
    - Adversarial validation
    - Fingerprint analysis
    Runs them at configured cadences, independent of Phase 1A.
    """

    def __init__(self, symbols, base_dir, reports_dir, config):
        self.symbols = symbols
        self.base_dir = base_dir
        self.reports_dir = reports_dir
        self.config = config
        self.tasks = []
        self.running = False

    async def _run_adversarial_loop(self, symbol):
        cadence = self.config["adversarial"].get("cadence_seconds", 900)
        while self.running:
            try:
                validator = AdversarialValidator(
                    symbol,
                    self.base_dir,
                    self.reports_dir,
                    self.config["adversarial"],
                )
                validator.run_validation()
            except Exception:
                logger.exception("[COORD2] Adversarial validation failed for %s", symbol)
            await asyncio.sleep(cadence)

    async def _run_fingerprint_loop(self, symbol):
        cadence = self.config["fingerprint"].get("cadence_seconds", 900)
        while self.running:
            try:
                analyzer = FingerprintAnalyzer(
                    symbol,
                    self.base_dir,
                    self.reports_dir,
                    self.config["fingerprint"],
                )
                analyzer.run_analysis()
            except Exception:
                logger.exception("[COORD2] Fingerprint analysis failed for %s", symbol)
            await asyncio.sleep(cadence)

    async def start(self):
        self.running = True
        logger.info("[COORD2] Starting Phase 2 Coordinator at %s", datetime.utcnow().isoformat())

        if self.config.get("adversarial", {}).get("enabled", False):
            for sym in self.symbols:
                self.tasks.append(asyncio.create_task(self._run_adversarial_loop(sym)))

        if self.config.get("fingerprint", {}).get("enabled", False):
            for sym in self.symbols:
                self.tasks.append(asyncio.create_task(self._run_fingerprint_loop(sym)))

    async def stop(self):
        self.running = False
        for t in self.tasks:
            t.cancel()
        self.tasks = []
        logger.info("[COORD2] Phase 2 Coordinator stopped")

    async def run_once(self):
        """
        Run both adversarial and fingerprint analyses once for all symbols.
        Useful for debugging or manual triggering.
        """
        for sym in self.symbols:
            if self.config.get("adversarial", {}).get("enabled", False):
                try:
                    validator = AdversarialValidator(
                        sym,
                        self.base_dir,
                        self.reports_dir,
                        self.config["adversarial"],
                    )
                    validator.run_validation()
                except Exception:
                    logger.exception("[COORD2] Adversarial validation failed for %s", sym)

            if self.config.get("fingerprint", {}).get("enabled", False):
                try:
                    analyzer = FingerprintAnalyzer(
                        sym,
                        self.base_dir,
                        self.reports_dir,
                        self.config["fingerprint"],
                    )
                    analyzer.run_analysis()
                except Exception:
                    logger.exception("[COORD2] Fingerprint analysis failed for %s", sym)
