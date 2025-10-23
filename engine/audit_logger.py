import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("audit_logger")


class AuditLogger:
    """
    Append-only, tamper-evident audit logger.
    Each entry includes a hash of the previous entry, forming a chain of custody.
    """

    def __init__(self, log_dir="reports/audit", log_file="audit_log.jsonl"):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_file
        self.last_hash = self._load_last_hash()

    def _load_last_hash(self):
        if not self.log_path.exists():
            return "GENESIS"
        try:
            with self.log_path.open("r") as f:
                last_line = None
                for line in f:
                    last_line = line.strip()
                if last_line:
                    entry = json.loads(last_line)
                    return entry.get("entry_hash", "GENESIS")
        except Exception:
            logger.warning("[AUDIT] Failed to load last hash, starting fresh")
        return "GENESIS"

    def _compute_hash(self, entry: dict) -> str:
        """
        Compute SHA-256 hash of entry (excluding entry_hash).
        """
        entry_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
        encoded = json.dumps(entry_copy, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def log_event(self, symbol: str, decision: str, rationale: list,
                  config_version: str, extra: dict = None) -> dict:
        """
        Append a new audit log entry.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "decision": decision,
            "rationale": rationale,
            "config_version": config_version,
            "previous_hash": self.last_hash,
        }
        if extra:
            entry.update(extra)

        entry["entry_hash"] = self._compute_hash(entry)

        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        self.last_hash = entry["entry_hash"]
        logger.info("[AUDIT] Logged event for %s: %s", symbol, decision)
        return entry
