import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("config_registry")


class ConfigRegistry:
    """
    Central registry for simulator configs across distributed nodes.
    Provides versioning, provenance, and synchronization APIs.
    """

    def __init__(self, registry_dir="registry"):
        self.registry_dir = Path(registry_dir).resolve()
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_dir / "index.json"
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps({"versions": []}, indent=2))

    def _load_index(self):
        return json.loads(self.index_file.read_text())

    def _save_index(self, index):
        self.index_file.write_text(json.dumps(index, indent=2))

    def push_config(self, symbol: str, config: dict, node_id: str) -> str:
        """
        Push a new config version into the registry.
        Returns version tag.
        """
        index = self._load_index()
        version_tag = f"{symbol}_v{len(index['versions'])+1}"
        manifest = {
            "symbol": symbol,
            "version": version_tag,
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": node_id,
            "config": config,
        }
        out_path = self.registry_dir / f"{version_tag}.json"
        out_path.write_text(json.dumps(manifest, indent=2))
        index["versions"].append(version_tag)
        self._save_index(index)
        logger.info("[REGISTRY] Pushed config %s from node %s", version_tag, node_id)
        return version_tag

    def pull_latest(self, symbol: str) -> dict:
        """
        Pull the latest config for a given symbol.
        """
        index = self._load_index()
        candidates = [v for v in index["versions"] if v.startswith(symbol)]
        if not candidates:
            return {}
        latest = candidates[-1]
        manifest = json.loads((self.registry_dir / f"{latest}.json").read_text())
        return manifest["config"]

    def list_versions(self, symbol: str = None):
        """
        List all versions, optionally filtered by symbol.
        """
        index = self._load_index()
        if symbol:
            return [v for v in index["versions"] if v.startswith(symbol)]
        return index["versions"]
