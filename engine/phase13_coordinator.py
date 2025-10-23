# engine/phase13_coordinator.py

import json
import logging
from pathlib import Path
from datetime import datetime

from engine.audit_logger import AuditLogger
from engine.provenance_tracker import ProvenanceTracker
from engine.config_registry import ConfigRegistry

logger = logging.getLogger("phase13_coordinator")


class Phase13Coordinator:
    """
    Governance & Auditability orchestrator:
      - Collects daily reports (live, drift, evolver)
      - Logs append-only, hashed audit entries
      - Links provenance records (config, report, decision, rationale)
      - Produces daily audit bundles for tamper-evident archiving
    """

    def __init__(
        self,
        symbols,
        reports_root="reports",
        audit_dir="reports/audit",
        provenance_dir="reports/provenance",
        bundle_dir="reports/audit/bundles",
        registry_dir="registry",
        node_id="unknown-node",
    ):
        self.symbols = symbols
        self.reports_root = Path(reports_root).resolve()
        self.audit_dir = Path(audit_dir).resolve()
        self.provenance_dir = Path(provenance_dir).resolve()
        self.bundle_dir = Path(bundle_dir).resolve()
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        self.audit = AuditLogger(log_dir=str(self.audit_dir))
        self.tracker = ProvenanceTracker(out_dir=str(self.provenance_dir))
        self.registry = ConfigRegistry(registry_dir=str(registry_dir))
        self.node_id = node_id

    def _collect_reports_for_symbol(self, symbol: str) -> dict:
        """
        Collect latest reports for a symbol across categories.
        Returns a dict of paths and parsed contents when available.
        """
        categories = {
            "live": self.reports_root / "live",
            "drift": self.reports_root / "drift",
            "evolver": self.reports_root / "live",  # evolver updates live alongside
        }
        collected = {}

        for name, dirpath in categories.items():
            if not dirpath.exists():
                continue
            files = sorted(dirpath.glob(f"{symbol}*.json"))
            if not files:
                continue
            latest = files[-1]
            try:
                data = json.loads(latest.read_text())
                collected[name] = {"path": str(latest), "data": data}
            except Exception:
                logger.warning("[P13] Failed to parse %s report for %s: %s", name, symbol, latest)

        return collected

    def _derive_decision_and_rationale(self, reports: dict) -> tuple:
        """
        Decide governance outcome based on available reports.
        Priority: evolver decisions > drift decisions > live pass/fail.
        """
        # Defaults
        decision = "PASS"
        rationale = ["No significant divergences recorded"]

        # Evolver manifest embedded in live report (ADJUSTED)
        live = reports.get("live", {})
        if live:
            d = live["data"].get("decision")
            r = live["data"].get("rationale", [])
            if d == "ADJUSTED":
                return "ADJUSTED", r if isinstance(r, list) else [str(r)]
            if d == "FAIL":
                return "FAIL", r if isinstance(r, list) else [str(r)]

        # Drift report outcome (ADAPT)
        drift = reports.get("drift", {})
        if drift:
            d = drift["data"].get("decision")
            r = drift["data"].get("rationale", "")
            if d == "ADAPT":
                return "ADAPT", [r] if r else ["Persistent drift detected"]

        # Live PASS fallback
        if live and live["data"].get("decision") == "PASS":
            rationale = live["data"].get("rationale", ["Live alignment within tolerance"])
            rationale = rationale if isinstance(rationale, list) else [str(rationale)]

        return decision, rationale

    def _resolve_config_version(self, symbol: str) -> str:
        """
        Resolve the latest config version tag from registry for the symbol.
        """
        versions = self.registry.list_versions(symbol)
        return versions[-1] if versions else f"{symbol}_v0"

    def run_daily(self):
        """
        Perform daily governance:
          - Collect latest reports per symbol
          - Log tamper-evident audit entries
          - Link provenance records
          - Produce a signed daily bundle (manifest + hash)
        """
        bundle_manifest = {
            "date": datetime.utcnow().date().isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": self.node_id,
            "symbols": [],
            "entries": [],
        }

        for sym in self.symbols:
            try:
                reports = self._collect_reports_for_symbol(sym)
                if not reports:
                    logger.info("[P13] No reports found for %s", sym)
                    continue

                decision, rationale = self._derive_decision_and_rationale(reports)
                config_version = self._resolve_config_version(sym)

                # Audit entry (append-only, hashed, chain-of-custody)
                audit_entry = self.audit.log_event(
                    symbol=sym,
                    decision=decision,
                    rationale=rationale,
                    config_version=config_version,
                    extra={"node_id": self.node_id},
                )

                # Pick a primary report to anchor provenance
                anchor = (
                    reports.get("evolver")
                    or reports.get("drift")
                    or reports.get("live")
                )
                anchor_path = anchor["path"] if anchor else None

                prov_record = self.tracker.link_event(
                    symbol=sym,
                    config_version=config_version,
                    report_path=anchor_path or "",
                    audit_entry=audit_entry,
                    rationale="; ".join(rationale),
                )

                bundle_manifest["symbols"].append(sym)
                bundle_manifest["entries"].append(
                    {
                        "symbol": sym,
                        "config_version": config_version,
                        "decision": decision,
                        "audit_entry_hash": audit_entry["entry_hash"],
                        "report_path": anchor_path,
                        "provenance_manifest": prov_record,
                    }
                )

                logger.info("[P13] Governance recorded for %s: %s", sym, decision)

            except Exception:
                logger.exception("[P13] Governance loop failed for %s", sym)

        # Write bundle manifest and its hash for tamper evidence
        bundle_path = self.bundle_dir / f"audit_bundle_{int(datetime.utcnow().timestamp())}.json"
        bundle_path.write_text(json.dumps(bundle_manifest, indent=2))
        bundle_hash = self._compute_file_hash(bundle_path)
        (self.bundle_dir / (bundle_path.stem + ".sha256")).write_text(bundle_hash)

        logger.info("[P13] Wrote daily audit bundle: %s (sha256: %s)", bundle_path, bundle_hash)
        return {"manifest": str(bundle_path), "sha256": bundle_hash}

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
