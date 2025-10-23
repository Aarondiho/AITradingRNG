import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("simulator_evolver")


class SimulatorEvolver:
    """
    Evolves the simulator configuration when adversary succeeds.
    - Adjusts mixture components, bootstrap block size, or mode
    - Writes new versioned config into data/simulator_versions/
    """

    def __init__(self, symbol: str, base_dir: str, config: dict):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve()
        self.versions_dir = self.base_dir / "simulator_versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def evolve(self, adversary_result: dict):
        """
        Evolve simulator if adversary AUC > threshold.
        Args:
          adversary_result: dict with keys {best_name, best_auc, all_results}
        Returns:
          dict with new config and decision
        """
        auc = adversary_result["best_auc"]
        threshold = self.config.get("auc_threshold", 0.6)

        if auc <= threshold:
            logger.info("[EVOLVE] Simulator for %s survived adversary (AUC=%.3f)", self.symbol, auc)
            return {"decision": "SURVIVED", "auc": auc}

        # Otherwise, evolve
        new_config = dict(self.config)  # copy current
        rationale = []

        # Strategy 1: increase mixture components
        if new_config.get("mode") in ("parametric", "hybrid"):
            comps = new_config.get("mixture_components", 1)
            new_config["mixture_components"] = comps + 1
            rationale.append(f"Increased mixture components {comps}→{comps+1}")

        # Strategy 2: increase bootstrap block size
        elif new_config.get("mode") == "bootstrap":
            block = new_config.get("bootstrap_block", 5)
            new_config["bootstrap_block"] = block + 2
            rationale.append(f"Increased bootstrap block {block}→{block+2}")

        # Strategy 3: switch mode if stuck
        else:
            old_mode = new_config.get("mode", "parametric")
            new_mode = "hybrid" if old_mode != "hybrid" else "parametric"
            new_config["mode"] = new_mode
            rationale.append(f"Switched mode {old_mode}→{new_mode}")

        # Versioning
        version_id = int(datetime.utcnow().timestamp())
        out_path = self.versions_dir / f"{self.symbol}_simulator_v{version_id}.json"
        manifest = {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "old_config": self.config,
            "new_config": new_config,
            "adversary": adversary_result["best_name"],
            "auc": auc,
            "decision": "UPDATED",
            "rationale": rationale,
        }
        out_path.write_text(json.dumps(manifest, indent=2))

        logger.info("[EVOLVE] Simulator for %s updated due to adversary success (AUC=%.3f)", self.symbol, auc)
        return manifest
