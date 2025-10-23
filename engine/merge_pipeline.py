import json
import logging
from pathlib import Path
from statistics import mean
import hashlib

logger = logging.getLogger("merge_pipeline")


class MergePipeline:
    """
    Loads live shards and produces merged shards for downstream features.
    Adds data-quality checks:
    - Detect duplicate epochs
    - Detect gaps larger than expected
    - Compute file hash and record count for provenance
    """

    def __init__(self, symbol: str, base_dir: str):
        self.symbol = symbol
        self.base_dir = Path(base_dir).resolve() / symbol
        self.live_dir = self.base_dir / "live_agg" / "1s"
        self.out_dir = self.base_dir / "merged"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[MERGE] Initialized for %s", self.symbol)

    def run_merge(self):
        files = sorted(self.live_dir.glob("*.ndjson"))
        if not files:
            logger.info("[MERGE] No live shards for %s", self.symbol)
            return

        records = []
        for f in files:
            try:
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        records.append(rec)
            except Exception:
                logger.exception("[MERGE] Failed reading %s", f.name)

        n = len(records)
        if n == 0:
            logger.info("[MERGE] No records to merge for %s", self.symbol)
            return

        # Sort by epoch
        records.sort(key=lambda r: int(r.get("epoch", 0)))

        # Data-quality checks
        epochs = [int(r["epoch"]) for r in records if "epoch" in r]
        duplicates = len(epochs) - len(set(epochs))
        gaps = []
        for i in range(1, len(epochs)):
            delta = epochs[i] - epochs[i - 1]
            if delta > 2:  # >2s gap
                gaps.append((epochs[i - 1], epochs[i], delta))

        if duplicates > 0:
            logger.warning("[MERGE] %s duplicate epochs detected for %s", duplicates, self.symbol)
        if gaps:
            logger.warning("[MERGE] %d gaps detected for %s (examples: %s)",
                           len(gaps), self.symbol, gaps[:3])

        # Determine epoch range
        tmin, tmax = min(epochs), max(epochs)

        # Write merged shard
        shard_id = f"{self.symbol}_merged_{tmin}_{tmax}"
        out_path = self.out_dir / f"{shard_id}.ndjson"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            logger.info("[MERGE] Produced %d merged records for %s", n, self.symbol)
            logger.info("[MERGE] Wrote merged shard %s for %s", shard_id, self.symbol)
        except Exception:
            logger.exception("[MERGE] Failed writing merged shard %s", shard_id)
            return

        # Compute file hash for provenance
        sha256 = hashlib.sha256()
        with out_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        # Write manifest
        manifest = {
            "symbol": self.symbol,
            "count": n,
            "epoch_range": [tmin, tmax],
            "duplicates": duplicates,
            "gap_count": len(gaps),
            "hash": file_hash,
            "input_files": [f.name for f in files],
        }
        (self.out_dir / f"{shard_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("[MERGE] Wrote manifest for %s", shard_id)
