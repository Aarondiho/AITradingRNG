import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from engine.stress_injector import StressInjector
from engine.survival_analyzer import SurvivalAnalyzer
from engine.market_ecosystem import MarketEcosystem
from engine.agents.noise_agent import NoiseAgent
from engine.agents.momentum_agent import MomentumAgent
from engine.agents.arbitrage_agent import ArbitrageAgent
from engine.fingerprint_extractor import FingerprintExtractor

logger = logging.getLogger("phase9_coordinator")


class Phase9Coordinator:
    """
    Orchestrates Phase 9 stress-testing & survival analysis:
      - Load baseline ticks
      - Apply shocks via StressInjector
      - Run agents in MarketEcosystem under stress
      - Analyze survival metrics
      - Write stress reports
    """

    def __init__(self, symbols, base_dir="data", reports_dir="reports", config=None):
        self.symbols = symbols
        self.base_dir = Path(base_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve() / "stress"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.injector = StressInjector(seed=self.config.get("seed", 42))
        self.analyzer = SurvivalAnalyzer(tolerance=self.config.get("tolerance", 0.05))
        self.extractor = FingerprintExtractor()

    def run_once(self):
        for sym in self.symbols:
            try:
                logger.info("[P9] Starting stress test for %s", sym)

                # --- Step 1: Load baseline ticks ---
                baseline_prices = self._load_baseline_ticks(sym)

                # --- Step 2: Apply shock ---
                shock_type = self.config.get("shock_type", "volatility_spike")
                shocked_prices = self.injector.apply(
                    baseline_prices,
                    shock_type=shock_type,
                    intensity=self.config.get("intensity", 2.0),
                    duration=self.config.get("duration", 300),
                )

                # --- Step 3: Run agents in ecology ---
                agents = [
                    NoiseAgent("noise1", sym),
                    MomentumAgent("mom1", sym, short_window=3, long_window=10),
                    ArbitrageAgent("arb1", sym, sym, baseline_ratio=1.0, threshold=0.02),
                ]
                eco = MarketEcosystem(symbols=[sym], agents=agents, initial_price=baseline_prices[0])
                eco.run(ticks=len(shocked_prices))

                # --- Step 4: Extract fingerprints ---
                real_fp = self.extractor.full_fingerprint(baseline_prices)
                shocked_fp = self.extractor.full_fingerprint(shocked_prices)

                # --- Step 5: Analyze survival ---
                survival_time = self.analyzer.survival_time(real_fp, shocked_fp)
                recovery_time = self.analyzer.recovery_time(shocked_prices, baseline_prices)
                drawdowns = self.analyzer.agent_drawdowns(agents, {sym: shocked_prices[-1]})
                stability = self.analyzer.ecology_stability(shocked_prices)

                # --- Step 6: Write report ---
                report = {
                    "symbol": sym,
                    "timestamp": datetime.utcnow().isoformat(),
                    "shock": {
                        "type": shock_type,
                        "intensity": self.config.get("intensity", 2.0),
                        "duration": self.config.get("duration", 300),
                    },
                    "metrics": {
                        "survival_time": survival_time,
                        "recovery_time": recovery_time,
                        "agent_drawdowns": drawdowns,
                        "ecology_volatility": stability,
                    },
                    "decision": "PASS" if survival_time > 0 else "FAIL",
                    "rationale": "Simulator maintained alignment under stress"
                                if survival_time > 0 else "Simulator collapsed under stress",
                }

                out_path = self.reports_dir / f"{sym}_stress_{int(datetime.utcnow().timestamp())}.json"
                out_path.write_text(json.dumps(report, indent=2))

                logger.info("[P9] Completed stress test for %s: %s", sym, report["decision"])

            except Exception:
                logger.exception("[P9] Stress test failed for %s", sym)

    def _load_baseline_ticks(self, symbol):
        """
        Placeholder for loading baseline tick data.
        Replace with actual loader from your Deriv tick store.
        """
        return np.cumsum(np.random.randn(5000)) + 100
