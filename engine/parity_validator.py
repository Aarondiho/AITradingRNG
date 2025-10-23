# engine/parity_validator.py
# (This implementation follows the parity validator previously provided and updated to verify ordering,
# missing ticks, and field presence; it is non-blocking and writes parity.json in the run folder.)
import json
import logging
import hashlib
import os
from pathlib import Path
from statistics import mean, median
from typing import List, Dict, Any, Optional

logger = logging.getLogger("parity_validator")

TAIL_Z_WARN = 3.0
TAIL_Z_FAIL = 10.0
MISSING_FIELD_WARN_RATIO = 0.02
MISSING_FIELD_FAIL_RATIO = 0.5
VAR_ZERO_FAIL_FRAC = 0.5
MOMENTUM_MEDIAN_MULT = 5.0

DEFAULT_SAMPLE_LINES = 200
DEFAULT_MIN_SHARD_ROWS_FOR_STRICT_REJECT = 20
REQUIRED_FIELDS = [
    "f1","f2","f3","f4","f5","f6","f7","f8","f9","f10",
    "bid","ask","mid_price","spread","imbalance"
]

def _mad(xs: List[float]) -> float:
    if not xs:
        return 0.0
    med = median(xs)
    return float(median([abs(x - med) for x in xs]))

def _sha256_of_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def _atomic_write_obj(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)

class ParityValidator:
    def __init__(self, symbol: str, base_dir: str, config: Dict[str, Any] = None):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        cfg = config or {}
        self.sample_lines = int(cfg.get("sample_lines", DEFAULT_SAMPLE_LINES))
        self.min_shard_rows_for_strict_reject = int(cfg.get("min_shard_rows_for_strict_reject", DEFAULT_MIN_SHARD_ROWS_FOR_STRICT_REJECT))
        self.required_fields = list(cfg.get("required_fields", REQUIRED_FIELDS))
        self.features_root = self.base_dir / "features"
        logger.info("[PARITY] Initialized for %s (sample=%d min_reject_rows=%d)", self.symbol, self.sample_lines, self.min_shard_rows_for_strict_reject)

    def _select_run_dirs(self) -> List[Path]:
        runs = [p for p in (self.features_root.glob("*")) if p.is_dir()]
        runs.sort()
        return runs

    def _select_manifest_in_run(self, run_dir: Path) -> Optional[Path]:
        manifests = sorted(run_dir.glob("*.manifest.json"))
        return manifests[0] if manifests else None

    def run_validation_for_run(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        manifest_path = self._select_manifest_in_run(run_dir)
        if not manifest_path:
            logger.info("[PARITY] No manifest in run %s", run_dir)
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("[PARITY] Failed reading manifest %s", manifest_path)
            return None

        shard_file = run_dir / manifest.get("shard_file")
        report: Dict[str, Any] = {
            "symbol": self.symbol,
            "run_id": manifest.get("run_id"),
            "manifest_file": manifest_path.name,
            "manifest_sha256": manifest.get("sha256"),
            "declared_count": manifest.get("count"),
            "computed_count": None,
            "computed_sha256": None,
            "decision": "WARN",
            "pass_for_training": True,
            "diagnostics": {},
            "timestamp": None,
            "input_shard": manifest.get("input_shard"),
            "feature_version": manifest.get("feature_version"),
        }

        if not shard_file or not shard_file.exists():
            report["decision"] = "FAIL"
            report["pass_for_training"] = False
            report["diagnostics"]["reason"] = "feature_shard_missing"
            report["timestamp"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
            try:
                _atomic_write_obj(run_dir / "parity.json", report)
            except Exception:
                logger.exception("[PARITY] Failed writing parity (missing) for %s", run_dir)
            logger.info("[PARITY] Wrote parity (missing) for %s", run_dir.name)
            return report

        try:
            cs = _sha256_of_file(shard_file)
            report["computed_sha256"] = cs
            with shard_file.open("r", encoding="utf-8") as fh:
                total_rows = sum(1 for _ in fh)
            report["computed_count"] = total_rows
        except Exception:
            report["decision"] = "FAIL"
            report["pass_for_training"] = False
            report["diagnostics"]["reason"] = "compute_error"
            report["timestamp"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
            try:
                _atomic_write_obj(run_dir / "parity.json", report)
            except Exception:
                logger.exception("[PARITY] Failed writing parity (compute_error) for %s", run_dir)
            return report

        declared = manifest.get("count")
        if declared is not None and declared != report["computed_count"]:
            report["diagnostics"]["count_mismatch"] = {"declared": declared, "computed": report["computed_count"]}
            try:
                ratio = abs((declared - report["computed_count"]) / float(declared)) if declared else 1.0
            except Exception:
                ratio = 1.0
            if ratio > 0.5:
                report["decision"] = "FAIL"
                report["pass_for_training"] = False
            else:
                report["decision"] = "WARN"

        if manifest.get("sha256") and manifest.get("sha256") != report["computed_sha256"]:
            report["diagnostics"]["sha256_mismatch"] = {"declared": manifest.get("sha256"), "computed": report["computed_sha256"]}
            report["decision"] = "FAIL"
            report["pass_for_training"] = False

        quotes: List[float] = []
        sample_seen = 0
        field_presence = {k: 0 for k in self.required_fields}
        epochs: List[int] = []
        rolling_vars = {}
        momenta = {}

        try:
            with shard_file.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= self.sample_lines:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    sample_seen += 1
                    ep = None
                    try:
                        ep = int(obj.get("epoch")) if obj.get("epoch") is not None else None
                    except Exception:
                        ep = None
                    if ep is not None:
                        epochs.append(ep)
                    q = obj.get("quote")
                    if isinstance(q, (int, float)):
                        quotes.append(float(q))
                    for f in self.required_fields:
                        if obj.get(f) is not None:
                            field_presence[f] = field_presence.get(f, 0) + 1
                    for k, v in obj.items():
                        if isinstance(k, str) and k.startswith("var_") and isinstance(v, (int, float)):
                            rolling_vars.setdefault(k, []).append(float(v))
                        if isinstance(k, str) and k.startswith("momentum_") and isinstance(v, (int, float)):
                            momenta.setdefault(k, []).append(float(v))
        except Exception:
            logger.exception("[PARITY] Error sampling shard %s", shard_file)

        order_issues = {}
        if epochs:
            non_mono = any(epochs[i] > epochs[i+1] for i in range(len(epochs)-1))
            if non_mono:
                order_issues["non_monotonic"] = True
                if report["decision"] != "FAIL":
                    report["decision"] = "WARN"
                report["diagnostics"].setdefault("order_issues", []).append("non_monotonic_epochs_in_sample")
            gaps = []
            for i in range(len(epochs)-1):
                gap = epochs[i+1] - epochs[i]
                if gap > 1:
                    gaps.append(gap)
            if gaps:
                order_issues["gaps"] = gaps[:10]
                if report["decision"] != "FAIL":
                    report["decision"] = "WARN"
                report["diagnostics"].setdefault("order_issues", []).append({"gaps_seconds_sample": gaps[:10]})

        presence_summary = {}
        for k, cnt in field_presence.items():
            denom = sample_seen if sample_seen else 1
            presence_summary[k] = round(cnt / denom, 4)

        if quotes:
            q_med = median(quotes)
            q_mad = _mad(quotes) or 1.0
            outliers_warn = sum(1 for q in quotes if abs((q - q_med) / q_mad) > TAIL_Z_WARN)
            outliers_fail = sum(1 for q in quotes if abs((q - q_med) / q_mad) > TAIL_Z_FAIL)
            tail_warn_pct = 100.0 * outliers_warn / len(quotes)
            tail_fail_pct = 100.0 * outliers_fail / len(quotes)
            report["diagnostics"]["tail_warn_pct"] = round(tail_warn_pct, 4)
            report["diagnostics"]["tail_fail_pct"] = round(tail_fail_pct, 4)

            small_shard = (report.get("computed_count") or 0) < self.min_shard_rows_for_strict_reject

            if tail_fail_pct > 5.0:
                report["decision"] = "FAIL"
                report["pass_for_training"] = False
            elif tail_warn_pct > 5.0:
                if not small_shard:
                    report["decision"] = "WARN"
                else:
                    report["diagnostics"].setdefault("notes", []).append("tail_warn_on_small_shard_downgraded")

        rolling_summary = {}
        for k, vals in rolling_vars.items():
            avg_var = mean(vals) if vals else 0.0
            frac_nonpos = sum(1 for v in vals if v <= 0) / len(vals) if vals else 0.0
            rolling_summary[k] = {"avg_var": avg_var, "frac_nonpos": round(frac_nonpos, 4)}
            if frac_nonpos >= VAR_ZERO_FAIL_FRAC:
                report["decision"] = "FAIL"
                report["pass_for_training"] = False

        momentum_summary = {}
        quote_mad = _mad(quotes) or 1.0
        for k, vals in momenta.items():
            m_mu = mean(vals) if vals else 0.0
            momentum_summary[k] = round(m_mu, 6)
            if abs(m_mu) > MOMENTUM_MEDIAN_MULT * quote_mad and (report.get("computed_count") or 0) >= self.min_shard_rows_for_strict_reject:
                report["decision"] = "FAIL"
                report["pass_for_training"] = False
            elif abs(m_mu) > MOMENTUM_MEDIAN_MULT * quote_mad:
                report["diagnostics"].setdefault("notes", []).append("momentum_large_but_shard_small_downgraded")

        feature_issues = []
        for f, ratio in presence_summary.items():
            if ratio <= 0:
                feature_issues.append({"feature": f, "issue": "MISSING", "presence_ratio": ratio})
                report["decision"] = "FAIL"
                report["pass_for_training"] = False
            elif ratio < MISSING_FIELD_WARN_RATIO:
                feature_issues.append({"feature": f, "issue": "LOW_PRESENCE", "presence_ratio": ratio})
                if report["decision"] != "FAIL":
                    report["decision"] = "WARN"
            elif ratio < MISSING_FIELD_FAIL_RATIO:
                if report["decision"] != "FAIL":
                    report["decision"] = "WARN"

        report.update({
            "feature_presence": presence_summary,
            "feature_issues": feature_issues,
            "order_issues": order_issues,
            "rolling_variances": rolling_summary,
            "momentum_means": momentum_summary,
            "computed_sha256": report.get("computed_sha256"),
            "computed_count": report.get("computed_count"),
            "input_file": shard_file.name,
            "decision": report.get("decision"),
            "pass_for_training": report.get("pass_for_training"),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        })

        try:
            _atomic_write_obj(run_dir / "parity.json", report)
            logger.info("[PARITY] Wrote parity report for %s: decision=%s pass_for_training=%s", run_dir.name, report.get("decision"), report.get("pass_for_training"))
        except Exception:
            logger.exception("[PARITY] Failed writing parity for %s", run_dir)

        return report

    def run_validation_all(self) -> List[Dict[str, Any]]:
        runs = self._select_run_dirs()
        out = []
        for r in runs:
            try:
                rep = self.run_validation_for_run(r)
                if rep:
                    out.append(rep)
            except Exception:
                logger.exception("[PARITY] Validation failed for run %s", r)
        return out
