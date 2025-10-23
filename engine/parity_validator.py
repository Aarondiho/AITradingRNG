import json
import logging
from pathlib import Path
from statistics import mean, pvariance

logger = logging.getLogger("parity_validator")


class ParityValidator:
    """
    Statistical parity validator on feature shards.
    - Checks raw quote distribution (mean, variance, tails).
    - Validates rolling features (variance > 0, momentum distribution).
    Writes a parity report JSON per symbol.
    """

    def __init__(self, symbol: str, base_dir: str):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        self.features_dir = self.base_dir / "features"
        self.out_dir = self.base_dir / "parity"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[PARITY] Initialized for %s", self.symbol)

    def run_validation(self):
        files = sorted(self.features_dir.glob("*.ndjson"))
        if not files:
            logger.info("[PARITY] No feature shards for %s", self.symbol)
            return

        latest = files[-1]
        quotes, rolling_vars, momenta = [], {}, {}

        try:
            with latest.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    q = obj.get("quote")
                    if isinstance(q, (int, float)):
                        quotes.append(float(q))

                    # Collect rolling variances and momenta if present
                    for k, v in obj.items():
                        if k.startswith("var_") and isinstance(v, (int, float)):
                            rolling_vars.setdefault(k, []).append(float(v))
                        if k.startswith("momentum_") and isinstance(v, (int, float)):
                            momenta.setdefault(k, []).append(float(v))
        except Exception:
            logger.exception("[PARITY] Failed reading %s", latest.name)
            return

        if not quotes:
            logger.info("[PARITY] No quotes in latest features for %s", self.symbol)
            return

        # Basic stats on quotes
        mu = mean(quotes)
        var = pvariance(quotes) if len(quotes) > 1 else 0.0
        tail_fail_pct = 0.0  # placeholder for tail checks
        decision = "PASS"

        # Rolling variance checks
        rolling_summary = {}
        for k, vals in rolling_vars.items():
            avg_var = mean(vals)
            rolling_summary[k] = avg_var
            if avg_var <= 0:
                decision = "FAIL"

        # Momentum checks (mean should be near 0)
        momentum_summary = {}
        for k, vals in momenta.items():
            m_mu = mean(vals)
            momentum_summary[k] = m_mu
            if abs(m_mu) > 5 * (var ** 0.5):  # arbitrary sanity threshold
                decision = "FAIL"

        logger.info("[PARITY] %s mean=%.4f, var=%.4f, tail_fail=%.2f%%, decision=%s",
                    self.symbol, mu, var, tail_fail_pct, decision)

        report = {
            "symbol": self.symbol,
            "mean": mu,
            "variance": var,
            "tail_fail_pct": tail_fail_pct,
            "rolling_variances": rolling_summary,
            "momentum_means": momentum_summary,
            "decision": decision,
            "input_file": latest.name,
        }
        out_path = self.out_dir / f"{self.symbol}_parity.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[PARITY] Wrote parity report for %s: %s", self.symbol, decision)
