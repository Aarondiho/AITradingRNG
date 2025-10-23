import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("fingerprint_reporter")


class FingerprintReporter:
    """
    Compares real vs synthetic fingerprints and generates alignment reports.
    """

    def __init__(self, reports_dir="reports/fingerprints", tolerance=0.05):
        self.reports_dir = Path(reports_dir).resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.tolerance = tolerance

    def compare(self, real_fp: dict, synth_fp: dict) -> dict:
        """
        Compare two fingerprint dicts.
        Returns:
          dict with per-metric differences and pass/fail decision.
        """
        comparison = {"tick_level": {}, "candle_level": {}}
        decision = "PASS"
        rationale = []

        # Tick-level comparison
        for k, v in real_fp["tick_level"].items():
            sv = synth_fp["tick_level"].get(k, None)
            if sv is None:
                continue
            diff = abs(v - sv)
            comparison["tick_level"][k] = {"real": v, "synthetic": sv, "diff": diff}
            if diff > self.tolerance * max(abs(v), 1e-6):
                decision = "FAIL"
                rationale.append(f"Tick-level {k} diverged (diff={diff:.4f})")

        # Candle-level comparison
        for horizon, metrics in real_fp["candle_level"].items():
            comparison["candle_level"][horizon] = {}
            for k, v in metrics.items():
                sv = synth_fp["candle_level"].get(horizon, {}).get(k, None)
                if sv is None:
                    continue
                diff = abs(v - sv)
                comparison["candle_level"][horizon][k] = {
                    "real": v, "synthetic": sv, "diff": diff
                }
                if diff > self.tolerance * max(abs(v), 1e-6):
                    decision = "FAIL"
                    rationale.append(f"Candle {horizon} {k} diverged (diff={diff:.4f})")

        return {
            "comparison": comparison,
            "decision": decision,
            "rationale": rationale,
        }

    def write_report(self, symbol: str, real_fp: dict, synth_fp: dict):
        """
        Generate and save a fingerprint comparison report.
        """
        result = self.compare(real_fp, synth_fp)
        report = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "real_fingerprint": real_fp,
            "synthetic_fingerprint": synth_fp,
            "comparison": result["comparison"],
            "decision": result["decision"],
            "rationale": result["rationale"],
        }

        out_path = self.reports_dir / f"{symbol}_fp_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[FPREPORT] Wrote fingerprint report for %s: %s", symbol, result["decision"])
        return report
