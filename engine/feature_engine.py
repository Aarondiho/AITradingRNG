import json
import logging
from pathlib import Path
from collections import deque
from statistics import mean, pvariance

logger = logging.getLogger("feature_engine")


class FeatureEngine:
    """
    Computes enriched features from merged shards:
    - Raw quotes
    - Rolling mean/variance over configurable windows
    - Momentum (current - rolling mean)
    Writes feature shards with provenance manifest.
    """

    WINDOWS = [5, 30, 60]  # window sizes in ticks

    def __init__(self, symbol: str, base_dir: str):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        self.merged_dir = self.base_dir / "merged"
        self.out_dir = self.base_dir / "features"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run_features(self):
        files = list(self.merged_dir.glob("*.ndjson"))
        if not files:
            logger.info("[FEAT] No merged shards for %s", self.symbol)
            return

        # Pick shard with max end-epoch from filename
        def parse_range(fname: Path) -> int:
            parts = fname.stem.split("_")
            try:
                return int(parts[-1])  # last token is tmax
            except Exception:
                return 0

        latest = max(files, key=parse_range)
        logger.info("[FEAT] Selected merged shard %s for %s", latest.name, self.symbol)

        records = []
        try:
            with latest.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if "quote" in obj:
                        records.append(obj)
        except Exception:
            logger.exception("[FEAT] Failed reading %s", latest.name)
            return

        if not records:
            logger.warning("[FEAT] No quote records in %s for %s", latest.name, self.symbol)
            return

        # Compute rolling features
        features = []
        deques = {w: deque(maxlen=w) for w in self.WINDOWS}

        for r in records:
            q = float(r["quote"])
            epoch = int(r["epoch"])
            feat = {"symbol": self.symbol, "epoch": epoch, "quote": q, "src": "features:v2"}

            for w, dq in deques.items():
                dq.append(q)
                if len(dq) >= 2:
                    feat[f"mean_{w}"] = mean(dq)
                    feat[f"var_{w}"] = pvariance(dq)
                    feat[f"momentum_{w}"] = q - feat[f"mean_{w}"]
            features.append(feat)

        # Write feature shard
        tmin = min(f["epoch"] for f in features)
        tmax = max(f["epoch"] for f in features)
        shard_id = f"{self.symbol}_features_{tmin}_{tmax}"
        out_path = self.out_dir / f"{shard_id}.ndjson"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                for feat in features:
                    f.write(json.dumps(feat) + "\n")
            logger.info("[FEAT] Computed %d features for %s", len(features), self.symbol)
            logger.info("[FEAT] Wrote feature shard %s for %s", shard_id, self.symbol)
        except Exception:
            logger.exception("[FEAT] Failed writing feature shard %s", shard_id)
            return

        # Write provenance manifest
        manifest = {
            "symbol": self.symbol,
            "feature_version": "v2",
            "windows": self.WINDOWS,
            "count": len(features),
            "epoch_range": [tmin, tmax],
            "input_shard": latest.name,
        }
        (self.out_dir / f"{shard_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("[FEAT] Wrote manifest for %s", shard_id)
