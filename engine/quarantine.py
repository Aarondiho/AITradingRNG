# engine/quarantine.py
"""
Quarantine writer for suspect runs or raw inputs.

Usage:
  from engine.quarantine import QuarantineWriter
  q = QuarantineWriter(base_dir="data")
  q.quarantine_run(symbol, run_id, source_dir_or_file, reason="parity_fail")

Writes:
  data/<symbol>/incidents/<run_id>/original/<...> and metadata incident.json
"""
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Union

class QuarantineWriter:
    def __init__(self, base_dir: Union[str, Path] = "data"):
        self.base = Path(base_dir)

    def quarantine_run(self, symbol: str, run_id: str, source_path: Union[str, Path], reason: str = "quarantine"):
        target_dir = self.base / symbol / "incidents" / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "symbol": symbol,
            "run_id": run_id,
            "reason": reason,
            "quarantined_at": datetime.utcnow().isoformat() + "Z",
            "source": str(source_path)
        }
        try:
            src = Path(source_path)
            if src.is_dir():
                # copy dir contents
                dst = target_dir / "original"
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            elif src.is_file():
                dstf = target_dir / ("original_" + src.name)
                shutil.copy2(src, dstf)
        except Exception:
            pass
        # write metadata
        try:
            (target_dir / "incident.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        return target_dir
