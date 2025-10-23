import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from engine.multi_scale_aggregator import MultiScaleAggregator
from engine.multi_scale_validator import MultiScaleValidator
from engine.fingerprint_extractor import FingerprintExtractor
from engine.simulator_core import SyntheticSimulator
from engine.simulator_evolver import SimulatorEvolver

logger = logging.getLogger("phase8_coordinator")


class Phase8Coordinator:
    """
    Orchestrates Phase 8 multi-scale simulation:
      - Load real ticks
      - Generate synthetic ticks
      - Aggregate to multiple horizons
      - Extract fingerprints
      - Validate cross-scale alignment
      - Evolve simulator if needed
      - Write multiscale reports
    """

    def __init__(self, symbols, base_dir="data", reports_dir="reports", config=None):
        self.symbols = symbols
        self.base_dir = Path(base_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve() / "multiscale"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.horizons = self.config.get("horizons", [60, 120, 180, 300])
        self.extractor = FingerprintExtractor()
        self.aggregator = MultiScaleAggregator(self.horizons)
        self.validator = MultiScaleValidator(tolerance=self.config.get("tolerance", 0.05))

    def run_once(self):
        for sym in self.symbols:
            try:
                logger.info("[P8] Starting multi-scale validation for %s", sym)

                # --- Step 1: Load real ticks ---
                real_prices = self._load_real_ticks(sym)

                # --- Step 2: Generate synthetic ticks ---
                sim = SyntheticSimulator(sym, self.base_dir)
                sim.fit_from_real(
                    mixture_components=self.config.get("mixture_components", 2),
                    seed=self.config.get("seed", 42),
                    min_records=self.config.get("min_records", 500),
                )
                synth_records = sim.generate_ticks(
                    mode=self.config.get("mode", "hybrid"),
                    count=self.config.get("count", 5000),
                    seed=self.config.get("seed", 42),
                    bootstrap_block=self.config.get("bootstrap_block", 5),
                    mixture_components=self.config.get("mixture_components", 2),
                    noise_scale=self.config.get("noise_scale", 0.1),
                )
                synth_prices = np.array([r["quote"] for r in synth_records])

                # --- Step 3: Aggregate to candles ---
                real_candles = self.aggregator.build_candles(real_prices)
                synth_candles = self.aggregator.build_candles(synth_prices)

                # --- Step 4: Extract fingerprints ---
                real_fp = self.extractor.full_fingerprint(real_prices, horizons=self.horizons)
                synth_fp = self.extractor.full_fingerprint(synth_prices, horizons=self.horizons)

                # --- Step 5: Validate cross-scale alignment ---
                result = self.validator.compare(real_candles, synth_candles, real_fp, synth_fp)

                # --- Step 6: Evolve simulator if needed ---
                if result["decision"] == "FAIL":
                    evolver = SimulatorEvolver(sym, self.base_dir, self.config)
                    evolve_manifest = evolver.evolve({
                        "best_name": "multiscale_check",
                        "best_auc": 1.0  # force evolve rationale
                    })
                    result["evolution"] = evolve_manifest

                # --- Step 7: Write report ---
                self._write_report(sym, real_fp, synth_fp, result)

                logger.info("[P8] Completed multi-scale validation for %s: %s",
                            sym, result["decision"])

            except Exception:
                logger.exception("[P8] Multi-scale validation failed for %s", sym)

    def _load_real_ticks(self, symbol):
        """
        Placeholder for loading real tick data.
        Replace with actual loader from your Deriv tick store.
        """
        return np.cumsum(np.random.randn(10000)) + 100

    def _write_report(self, symbol, real_fp, synth_fp, result):
        report = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "real_fingerprint": real_fp,
            "synthetic_fingerprint": synth_fp,
            "comparison": result.get("comparison", {}),
            "decision": result.get("decision"),
            "rationale": result.get("rationale", []),
            "evolution": result.get("evolution", None),
        }
        out_path = self.reports_dir / f"{symbol}_multiscale_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(report, indent=2))
