import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

from models.adversarial.utils import (
    build_feature_matrix,
    zscore_fit,
    zscore_apply,
    synthetic_generate,
    split_train_test,
    metrics_roc_auc,
    metrics_pr_auc,
    metrics_confusion,
)
from models.adversarial.logistic_baseline import LogisticBaseline
from models.adversarial.random_forest_baseline import RandomForestBaseline

logger = logging.getLogger("adversarial_validator")


class AdversarialValidator:
    """
    Runs adversarial validation:
    - Loads latest feature shard
    - Generates synthetic negatives
    - Trains baseline classifier (logistic or RF)
    - Computes metrics and writes JSON report
    """

    def __init__(self, symbol: str, base_dir: str, reports_dir: str, config: dict):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol / "features"
        self.reports_dir = Path(reports_dir).resolve() / "adversarial" / symbol
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def run_validation(self):
        files = sorted(self.base_dir.glob("*.ndjson"))
        if not files:
            logger.info("[ADV] No feature shards for %s", self.symbol)
            return

        latest = files[-1]
        records = []
        try:
            with latest.open("r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
        except Exception:
            logger.exception("[ADV] Failed reading %s", latest.name)
            return

        if len(records) < self.config.get("min_records", 500):
            logger.info("[ADV] Not enough records (%d) for %s", len(records), self.symbol)
            return

        # Build feature matrix
        X_real = build_feature_matrix(records, windows=self.config.get("train_window_features", [5, 30, 60]))
        y_real = np.zeros(len(X_real), dtype=int)

        # Generate synthetic negatives
        synth_mode = self.config.get("negative_baseline", "gaussian")
        X_synth = synthetic_generate(X_real, mode=synth_mode, seed=42)
        y_synth = np.ones(len(X_synth), dtype=int)

        # Combine
        X = np.vstack([X_real, X_synth])
        y = np.concatenate([y_real, y_synth])

        # Normalize
        mu, sigma = zscore_fit(X)
        X_norm = zscore_apply(X, mu, sigma)

        # Split
        X_train, X_test, y_train, y_test = split_train_test(X_norm, y, ratio=0.7, seed=42)

        # Select model
        model_type = self.config.get("baseline", "logistic")
        if model_type == "logistic":
            model = LogisticBaseline()
        elif model_type == "random_forest":
            model = RandomForestBaseline()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        model_params = model.fit(X_train, y_train)

        # Evaluate
        y_score = model.predict_proba(X_test)
        roc_auc = metrics_roc_auc(y_test, y_score)
        pr_auc = metrics_pr_auc(y_test, y_score)
        cm = metrics_confusion(y_test, y_score, threshold=0.5)

        # Decision
        if 0.45 <= roc_auc <= 0.55:
            decision = "PASS"
            rationale = "ROC-AUC within [0.45,0.55], indistinguishable from noise"
        elif roc_auc > 0.6:
            decision = "FAIL"
            rationale = "ROC-AUC > 0.6, RNG distinguishable from synthetic baseline"
        else:
            decision = "BORDERLINE"
            rationale = "ROC-AUC marginally outside neutral range"

        # Report
        report = {
            "symbol": self.symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "feature_version": "v2",
            "model": model_type,
            "model_params": model_params,
            "synthetic_mode": synth_mode,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "decision": decision,
            "rationale": rationale,
            "normalization": {"mu": mu.tolist(), "sigma": sigma.tolist()},
            "input_shard": latest.name,
        }

        out_path = self.reports_dir / f"{self.symbol}_adv_{int(datetime.utcnow().timestamp())}.json"
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[ADV] Wrote adversarial report for %s: %s (ROC-AUC=%.3f)", self.symbol, decision, roc_auc)
