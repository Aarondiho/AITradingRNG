import logging
import json
from pathlib import Path
from datetime import datetime

from engine.adversarial_trainer import AdversarialTrainer
from engine.simulator_evolver import SimulatorEvolver
from engine.simulator_core import SyntheticSimulator
from models.adversarial.utils import build_feature_matrix, zscore_fit, zscore_apply

logger = logging.getLogger("phase5_coordinator")


class Phase5Coordinator:
    """
    Orchestrates Phase 5 adversarial hardening:
      - Generate synthetic ticks
      - Train stronger adversaries
      - If adversary succeeds (AUC > threshold), evolve simulator
      - Write battle reports into reports/hardening/<symbol>/
    """

    def __init__(self, symbols, base_dir="data", reports_dir="reports", config=None):
        self.symbols = symbols
        self.base_dir = Path(base_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve() / "hardening"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def run_once(self):
        for sym in self.symbols:
            try:
                logger.info("[HARDEN] Starting adversarial hardening for %s", sym)

                # --- Step 1: Generate synthetic ticks ---
                sim = SyntheticSimulator(sym, self.base_dir)
                sim.fit_from_real(
                    mixture_components=self.config.get("mixture_components", 2),
                    seed=self.config.get("seed", 42),
                    min_records=self.config.get("min_records", 500),
                )
                records = sim.generate_ticks(
                    mode=self.config.get("mode", "hybrid"),
                    count=self.config.get("count", 1000),
                    seed=self.config.get("seed", 42),
                    bootstrap_block=self.config.get("bootstrap_block", 5),
                    mixture_components=self.config.get("mixture_components", 2),
                    noise_scale=self.config.get("noise_scale", 0.1),
                )

                # Build feature matrix for adversary
                X_real = build_feature_matrix(sim.real_quotes.reshape(-1, 1))
                y_real = [0] * len(X_real)
                X_synth = build_feature_matrix(np.array([r["quote"] for r in records]).reshape(-1, 1))
                y_synth = [1] * len(X_synth)

                X = np.vstack([X_real, X_synth])
                y = np.array(y_real + y_synth)

                # Normalize
                mu, sigma = zscore_fit(X)
                X_norm = zscore_apply(X, mu, sigma)

                # --- Step 2: Train adversary ---
                trainer = AdversarialTrainer(seed=self.config.get("seed", 42))
                adv_result = trainer.train_and_evaluate(X_norm, y)

                # --- Step 3: Evolve simulator if needed ---
                evolver = SimulatorEvolver(sym, self.base_dir, self.config)
                evolve_result = evolver.evolve(adv_result)

                # --- Step 4: Write battle report ---
                report = {
                    "symbol": sym,
                    "timestamp": datetime.utcnow().isoformat(),
                    "adversary": adv_result["best_name"],
                    "auc": adv_result["best_auc"],
                    "all_results": adv_result["all_results"],
                    "decision": evolve_result.get("decision"),
                    "rationale": evolve_result.get("rationale", []),
                }
                out_path = self.reports_dir / sym / f"{sym}_battle_{int(datetime.utcnow().timestamp())}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(report, indent=2))

                logger.info("[HARDEN] Wrote battle report for %s: %s (AUC=%.3f)",
                            sym, report["decision"], report["auc"])

            except Exception:
                logger.exception("[HARDEN] Hardening failed for %s", sym)
