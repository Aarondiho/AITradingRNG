import json
import logging
from pathlib import Path
from datetime import datetime
from statistics import mean

logger = logging.getLogger("metrics_aggregator")


class MetricsAggregator:
    """
    Aggregates stress/live/multiscale reports from distributed nodes
    into global summaries for dashboards and monitoring.
    """

    def __init__(self, reports_root="reports", out_dir="reports/global"):
        self.reports_root = Path(reports_root).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def collect_reports(self, report_type="live") -> list:
        """
        Collect all JSON reports of a given type (live, stress, multiscale).
        """
        reports = []
        report_dir = self.reports_root / report_type
        if not report_dir.exists():
            return reports
        for f in report_dir.glob("*.json"):
            try:
                reports.append(json.loads(f.read_text()))
            except Exception:
                logger.warning("[AGGREGATOR] Failed to parse %s", f)
        return reports

    def aggregate(self, report_type="live") -> dict:
        """
        Aggregate metrics across all reports of a given type.
        Returns a global summary dict.
        """
        reports = self.collect_reports(report_type)
        if not reports:
            return {"decision": "NO_DATA", "rationale": "No reports found"}

        symbols = [r.get("symbol") for r in reports]
        bias_diffs, vol_diffs, autocorr_diffs = [], [], []

        for r in reports:
            comp = r.get("comparison", {})
            # Try to extract some common metrics
            for horizon, metrics in comp.items():
                if "direction_bias" in metrics:
                    bias_diffs.append(metrics["direction_bias"]["diff"])
                if "volatility" in metrics:
                    vol_diffs.append(metrics["volatility"]["diff"])
            # Tick-level autocorr if available
            tick_fp = r.get("real_fingerprint", {}).get("tick_level", {})
            synth_fp = r.get("synthetic_fingerprint", {}).get("tick_level", {})
            if "autocorr_lag1" in tick_fp and "autocorr_lag1" in synth_fp:
                diff = abs(tick_fp["autocorr_lag1"] - synth_fp["autocorr_lag1"])
                autocorr_diffs.append(diff)

        summary = {
            "date": datetime.utcnow().date().isoformat(),
            "symbols_tested": list(set(symbols)),
            "reports_count": len(reports),
            "global_metrics": {
                "avg_bias_diff": mean(bias_diffs) if bias_diffs else None,
                "avg_volatility_diff": mean(vol_diffs) if vol_diffs else None,
                "avg_autocorr_diff": mean(autocorr_diffs) if autocorr_diffs else None,
            },
            "decision": "PASS" if (
                (not bias_diffs or mean(bias_diffs) < 0.05) and
                (not vol_diffs or mean(vol_diffs) < 0.05) and
                (not autocorr_diffs or mean(autocorr_diffs) < 0.05)
            ) else "FAIL",
            "rationale": "Aggregated divergences within tolerance"
                         if bias_diffs or vol_diffs or autocorr_diffs else "No metrics available"
        }

        out_path = self.out_dir / f"{report_type}_summary_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        logger.info("[AGGREGATOR] Wrote global summary: %s", out_path)
        return summary
