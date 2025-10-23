import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("drift_profiler")


class DriftProfiler:
    """
    Archives and analyzes long-term drift profiles.
    Builds a historical 'drift atlas' for each symbol.
    """

    def __init__(self, out_dir="reports/drift"):
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def archive_profile(self, symbol: str, profile: Dict[str, float], decision: str, rationale: str):
        """
        Save a daily drift profile to JSON.
        """
        report = {
            "symbol": symbol,
            "date": datetime.utcnow().date().isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
            "profile": profile,
            "decision": decision,
            "rationale": rationale,
        }
        out_path = self.out_dir / f"{symbol}_drift_{datetime.utcnow().date().isoformat()}.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[DRIFT] Archived profile for %s: %s", symbol, out_path)
        return out_path

    def build_drift_atlas(self, symbol: str) -> Dict[str, Any]:
        """
        Aggregate all drift reports for a symbol into a historical atlas.
        Returns a dict with time-series of drift metrics.
        """
        atlas = {"symbol": symbol, "history": []}
        for f in sorted(self.out_dir.glob(f"{symbol}_drift_*.json")):
            try:
                data = json.loads(f.read_text())
                atlas["history"].append({
                    "date": data["date"],
                    "bias_drift": data["profile"].get("bias_drift"),
                    "volatility_drift": data["profile"].get("volatility_drift"),
                    "autocorr_drift": data["profile"].get("autocorr_drift"),
                    "decision": data.get("decision"),
                })
            except Exception:
                logger.warning("[DRIFT] Failed to parse %s", f)

        # Optionally compute averages
        if atlas["history"]:
            bias_vals = [h["bias_drift"] for h in atlas["history"] if h["bias_drift"] is not None]
            vol_vals = [h["volatility_drift"] for h in atlas["history"] if h["volatility_drift"] is not None]
            auto_vals = [h["autocorr_drift"] for h in atlas["history"] if h["autocorr_drift"] is not None]
            atlas["averages"] = {
                "bias_drift": sum(bias_vals) / len(bias_vals) if bias_vals else None,
                "volatility_drift": sum(vol_vals) / len(vol_vals) if vol_vals else None,
                "autocorr_drift": sum(auto_vals) / len(auto_vals) if auto_vals else None,
            }
        return atlas
