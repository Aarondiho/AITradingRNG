import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("provenance_tracker")


class ProvenanceTracker:
    """
    Tracks provenance of configs, reports, and audit logs.
    Provides replay manifests to reconstruct simulator state at any timestamp.
    """

    def __init__(self, out_dir="reports/provenance"):
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.out_dir / "provenance_index.json"
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps({"entries": []}, indent=2))

    def _load_index(self) -> Dict[str, Any]:
        return json.loads(self.index_file.read_text())

    def _save_index(self, index: Dict[str, Any]):
        self.index_file.write_text(json.dumps(index, indent=2))

    def link_event(self, symbol: str, config_version: str, report_path: str,
                   audit_entry: dict, rationale: str) -> dict:
        """
        Link a config, report, and audit log entry into a provenance record.
        """
        record = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "config_version": config_version,
            "report_path": str(report_path),
            "audit_entry_hash": audit_entry.get("entry_hash"),
            "decision": audit_entry.get("decision"),
            "rationale": rationale,
        }

        index = self._load_index()
        index["entries"].append(record)
        self._save_index(index)

        # Also write standalone manifest
        manifest_path = self.out_dir / f"{symbol}_prov_{int(datetime.utcnow().timestamp())}.json"
        manifest_path.write_text(json.dumps(record, indent=2))

        logger.info("[PROVENANCE] Linked event for %s with config %s", symbol, config_version)
        return record

    def replay_manifest(self, symbol: str, timestamp: str) -> dict:
        """
        Retrieve the closest provenance record before a given timestamp.
        """
        index = self._load_index()
        candidates = [e for e in index["entries"] if e["symbol"] == symbol]
        if not candidates:
            return {}
        # Sort by timestamp
        candidates.sort(key=lambda e: e["timestamp"])
        for rec in reversed(candidates):
            if rec["timestamp"] <= timestamp:
                return rec
        return candidates[0]  # fallback to earliest
