import logging
import json
from pathlib import Path
from datetime import datetime

from engine.drift_monitor import DriftMonitor
from engine.drift_profiler import DriftProfiler

logger = logging.getLogger("phase12_coordinator")


class Phase12Coordinator:
    """
    Orchestrates Phase 12 long-term drift management:
      - Collect fingerprints (real vs synthetic)
      - Update drift monitor
      - Archive daily drift profiles
      - Trigger gradual adaptation if persistent drift detected
    """

    def __init__(self, symbols, reports_dir="reports/drift", config=None):
        self.symbols = symbols
        self.reports_dir = Path(reports_dir).resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.monitor = DriftMonitor(
            window_size=self.config.get("window_size", 7),
            tolerance=self.config.get("tolerance", 0.05),
        )
        self.profiler = DriftProfiler(out_dir=self.reports_dir)

    def run_daily(self, fingerprints: dict):
        """
        Run daily drift management for all symbols.
        Args:
          fingerprints: dict mapping symbol -> (real_fp, synth_fp)
        """
        for sym in self.symbols:
            try:
                real_fp, synth_fp = fingerprints.get(sym, (None, None))
                if real_fp is None or synth_fp is None:
                    logger.warning("[P12] Missing fingerprints for %s", sym)
                    continue

                # --- Step 1: Update drift monitor ---
                profile = self.monitor.update(sym, real_fp, synth_fp)

                # --- Step 2: Detect persistent drift ---
                drift_flag = self.monitor.detect_persistent_drift(profile)

                if drift_flag:
                    decision = "ADAPT"
                    rationale = "Persistent drift detected beyond tolerance"
                    # Here you could trigger gradual evolver adjustments
                else:
                    decision = "PASS"
                    rationale = "No persistent drift detected"

                # --- Step 3: Archive profile ---
                self.profiler.archive_profile(sym, profile, decision, rationale)

                logger.info("[P12] %s drift decision: %s", sym, decision)

            except Exception:
                logger.exception("[P12] Drift management failed for %s", sym)
