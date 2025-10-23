import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("continuous_evolver")


class ContinuousEvolver:
    """
    Continuously adjusts simulator parameters in real time
    based on fingerprint divergences.
    """

    def __init__(self, config=None, reports_dir="reports/live"):
        self.config = config or {}
        self.reports_dir = Path(reports_dir).resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_update(self, symbol: str, real_fp: dict, synth_fp: dict):
        """
        Compare fingerprints and apply small parameter updates if divergence exceeds tolerance.
        Returns:
          decision: "PASS" or "ADJUSTED"
          rationale: explanation of changes
        """
        tolerance = self.config.get("tolerance", 0.05)
        adjustments = {}
        rationale = []

        # --- Tick-level checks ---
        for k, v in real_fp["tick_level"].items():
            sv = synth_fp["tick_level"].get(k, None)
            if sv is None:
                continue
            diff = abs(v - sv)
            if diff > tolerance * max(abs(v), 1e-6):
                if "autocorr" in k:
                    # Too much autocorr → increase noise
                    old = self.config.get("noise_scale", 0.1)
                    new = round(old * 1.05, 4)
                    self.config["noise_scale"] = new
                    adjustments["noise_scale"] = (old, new)
                    rationale.append(f"{k} divergence {diff:.4f}, increased noise_scale")

        # --- Candle-level checks ---
        for horizon, metrics in real_fp["candle_level"].items():
            for k, v in metrics.items():
                sv = synth_fp["candle_level"].get(horizon, {}).get(k, None)
                if sv is None:
                    continue
                diff = abs(v - sv)
                if diff > tolerance * max(abs(v), 1e-6):
                    if k == "direction_bias":
                        # Too bullish/bearish → adjust mixture components
                        old = self.config.get("mixture_components", 2)
                        new = min(old + 1, 5)
                        self.config["mixture_components"] = new
                        adjustments["mixture_components"] = (old, new)
                        rationale.append(f"{horizon} {k} divergence {diff:.4f}, increased mixture_components")
                    elif k == "volatility":
                        # Volatility mismatch → adjust bootstrap block size
                        old = self.config.get("bootstrap_block", 5)
                        new = max(1, old + (1 if sv < v else -1))
                        self.config["bootstrap_block"] = new
                        adjustments["bootstrap_block"] = (old, new)
                        rationale.append(f"{horizon} {k} divergence {diff:.4f}, adjusted bootstrap_block")

        # --- Decision ---
        if adjustments:
            decision = "ADJUSTED"
            self._write_update(symbol, adjustments, rationale)
        else:
            decision = "PASS"
            rationale.append("All divergences within tolerance")

        return decision, rationale

    def _write_update(self, symbol, adjustments, rationale):
        """
        Write an update manifest for transparency and reproducibility.
        """
        manifest = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "adjustments": adjustments,
            "rationale": rationale,
            "new_config": self.config,
        }
        out_path = self.reports_dir / f"{symbol}_evolver_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(manifest, indent=2))
        logger.info("[EVOLVER] Updated config for %s: %s", symbol, adjustments)
