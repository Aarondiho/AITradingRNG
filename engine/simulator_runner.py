import logging
from pathlib import Path
from datetime import datetime

from engine.simulator_core import SyntheticSimulator
from engine.simulator_validator import SimulatorValidator

logger = logging.getLogger("simulator_runner")


class SimulatorRunner:
    """
    Orchestrates synthetic tick generation and validation for all symbols.
    - Calls SyntheticSimulator to generate synthetic shards
    - Calls SimulatorValidator to run Phase1B + Phase2 modules on synthetic data
    """

    def __init__(self, symbols, base_dir="data", reports_dir="reports", config=None):
        self.symbols = symbols
        self.base_dir = Path(base_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve()
        self.config = config or {}

    def run_once(self):
        """
        Generate synthetic ticks and validate them once for all symbols.
        """
        for sym in self.symbols:
            try:
                logger.info("[SIMRUN] Starting synthetic generation for %s", sym)
                sim = SyntheticSimulator(sym, self.base_dir)
                sim.fit_from_real(
                    mixture_components=self.config.get("mixture_components", 0),
                    seed=self.config.get("seed", 42),
                    min_records=self.config.get("min_records", 500),
                )

                records = sim.generate_ticks(
                    mode=self.config.get("mode", "parametric"),
                    count=self.config.get("count", 1000),
                    seed=self.config.get("seed", 42),
                    bootstrap_block=self.config.get("bootstrap_block", 5),
                    mixture_components=self.config.get("mixture_components", 0),
                    noise_scale=self.config.get("noise_scale", 0.1),
                )

                shard_path, manifest_path = sim.write_shard(records)

                # Validate synthetic shard
                validator = SimulatorValidator(
                    symbol=sym,
                    base_dir=self.base_dir,
                    reports_dir=self.reports_dir,
                    config=self.config,
                )
                validator.run_validation(shard_path)

            except Exception:
                logger.exception("[SIMRUN] Failed synthetic run for %s", sym)
