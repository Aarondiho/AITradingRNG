import json
import logging
from pathlib import Path
from datetime import datetime

from engine.feature_engine import FeatureEngine
from engine.parity_validator import ParityValidator
from engine.adversarial_validator import AdversarialValidator
from engine.fingerprint_analyzer import FingerprintAnalyzer

logger = logging.getLogger("simulator_validator")


class SimulatorValidator:
    """
    Validates synthetic shards by running them through:
      - FeatureEngine
      - ParityValidator
      - AdversarialValidator
      - FingerprintAnalyzer
    Produces a combined validation report.
    """

    def __init__(self, symbol: str, base_dir: str, reports_dir: str, config: dict):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        self.synthetic_dir = self.base_dir / "synthetic"
        self.reports_dir = Path(reports_dir).resolve() / "simulator" / symbol
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def run_validation(self, shard_path: Path):
        # Run FeatureEngine on synthetic shard
        feat_engine = FeatureEngine(self.symbol, str(self.base_dir))
        records = []
        with shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        # Write features directly
        features = []
        for r in records:
            features.append({
                "symbol": self.symbol,
                "epoch": r["epoch"],
                "quote": r["quote"],
                "src": "synthetic_features:v1"
            })
        tmin = min(r["epoch"] for r in features)
        tmax = max(r["epoch"] for r in features)
        feat_path = self.base_dir / "features" / f"{self.symbol}_synthetic_features_{tmin}_{tmax}.ndjson"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        with feat_path.open("w", encoding="utf-8") as f:
            for r in features:
                f.write(json.dumps(r) + "\n")

        # Run ParityValidator
        parity = ParityValidator(self.symbol, str(self.base_dir))
        parity.run_validation()

        # Run AdversarialValidator
        adv = AdversarialValidator(self.symbol, str(self.base_dir), str(self.reports_dir), self.config.get("adversarial", {}))
        adv.run_validation()

        # Run FingerprintAnalyzer
        fprint = FingerprintAnalyzer(self.symbol, str(self.base_dir), str(self.reports_dir), self.config.get("fingerprint", {}))
        fprint.run_analysis()

        # Combined report
        report = {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "synthetic_shard": shard_path.name,
            "features_file": feat_path.name,
            "parity_report": f"{self.symbol}_parity.json",
            "adversarial_reports": [p.name for p in (self.reports_dir.glob("*adv*.json"))],
            "fingerprint_reports": [p.name for p in (self.reports_dir.glob("*fprint*.json"))],
        }
        out_path = self.reports_dir / f"{self.symbol}_simreport_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[SIMVAL] Wrote simulator validation report for %s", self.symbol)
