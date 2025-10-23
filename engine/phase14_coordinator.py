import logging
from datetime import datetime
from pathlib import Path
import json

from engine.forecaster import Forecaster
from engine.integration_bridge import IntegrationBridge
from engine.provenance_tracker import ProvenanceTracker
from engine.audit_logger import AuditLogger
from engine.drift_profiler import DriftProfiler

logger = logging.getLogger("phase14_coordinator")


class Phase14Coordinator:
    """
    Full autonomy orchestrator:
      - Collects drift atlas + live fingerprints
      - Runs forecasts
      - Publishes forecasts locally and via integration bridge
      - Logs governance decisions (FORECAST) immutably
      - Links provenance for replayability
    """

    def __init__(self, symbols, reports_root="reports", node_id="unknown-node"):
        self.symbols = symbols
        self.reports_root = Path(reports_root).resolve()
        self.forecaster = Forecaster(horizon_hours=24)
        self.bridge = IntegrationBridge(out_dir=str(self.reports_root / "forecasts"))
        self.profiler = DriftProfiler(out_dir=str(self.reports_root / "drift"))
        self.audit = AuditLogger(log_dir=str(self.reports_root / "audit"))
        self.tracker = ProvenanceTracker(out_dir=str(self.reports_root / "provenance"))
        self.node_id = node_id

    def _load_latest_live_fp(self, symbol: str) -> dict:
        """
        Load the latest live fingerprint comparison for a symbol.
        """
        live_dir = self.reports_root / "live"
        if not live_dir.exists():
            return {}
        files = sorted(live_dir.glob(f"{symbol}*.json"))
        if not files:
            return {}
        try:
            return json.loads(files[-1].read_text())
        except Exception:
            logger.warning("[P14] Failed to parse live fingerprint for %s", symbol)
            return {}

    def run_cycle(self):
        """
        Run one full autonomy cycle for all symbols.
        """
        results = {}
        for sym in self.symbols:
            try:
                # --- Step 1: Load drift atlas + live fingerprints ---
                atlas = self.profiler.build_drift_atlas(sym)
                live_fp = self._load_latest_live_fp(sym)

                if not atlas.get("history"):
                    logger.info("[P14] No drift history for %s, skipping forecast", sym)
                    continue

                # --- Step 2: Generate forecast ---
                forecast = self.forecaster.forecast(sym, atlas, live_fp)

                # --- Step 3: Publish forecast ---
                pub_manifest = self.bridge.publish_forecast(forecast)

                # --- Step 4: Governance logging ---
                audit_entry = self.audit.log_event(
                    symbol=sym,
                    decision="FORECAST",
                    rationale=[forecast["rationale"]],
                    config_version="N/A",  # Forecasts are model outputs, not config pushes
                    extra={"node_id": self.node_id}
                )

                # --- Step 5: Provenance linking ---
                prov_record = self.tracker.link_event(
                    symbol=sym,
                    config_version="N/A",
                    report_path=pub_manifest["local_path"],
                    audit_entry=audit_entry,
                    rationale=forecast["rationale"]
                )

                results[sym] = {
                    "forecast": forecast,
                    "publication": pub_manifest,
                    "audit_entry": audit_entry,
                    "provenance": prov_record
                }

                logger.info("[P14] Forecast cycle complete for %s", sym)

            except Exception:
                logger.exception("[P14] Forecast cycle failed for %s", sym)

        return results
