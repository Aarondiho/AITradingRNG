# utils/leader_lock.py
"""
Simple filesystem leader lock for single-writer aggregator or other singleton tasks.

Usage:
  from leader_lock import FileLock
  with FileLock("/var/run/myapp/aggregator.lock", ttl=60) as lk:
      if not lk.acquired:
          return  # another leader holds lock
      # perform single-writer work
Notes:
- TTL prevents stale locks from permanently blocking leadership.
- Lock uses atomic symlink or atomic directory creation where available.
- No external dependencies.
"""
import os
import time
from pathlib import Path
from typing import Optional

class FileLock:
    def __init__(self, path: str, ttl: int = 60):
        self.path = Path(path)
        self.ttl = int(ttl)
        self.pid = os.getpid()
        self.acquired = False

    def _now(self) -> float:
        return time.time()

    def _lock_meta_path(self) -> Path:
        return self.path.with_suffix(".meta")

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Attempt to create directory atomically
            os.mkdir(self.path)
            meta = {"pid": self.pid, "acquired_at": self._now()}
            self._write_meta(meta)
            self.acquired = True
            return True
        except FileExistsError:
            # check TTL on meta
            try:
                meta = self._read_meta()
                if not meta:
                    # no meta -> possible stale lock, try to remove if expired
                    try:
                        st = self.path.stat()
                    except Exception:
                        st = None
                    # if no meta and dir exists, treat as stale if mtime old
                    if st and (self._now() - st.st_mtime) > self.ttl:
                        try:
                            os.rmdir(self.path)
                        except Exception:
                            return False
                        return self.acquire()
                    return False
                age = self._now() - float(meta.get("acquired_at", 0))
                if age > self.ttl:
                    # stale lock, remove and retry
                    try:
                        # best-effort: remove dir then meta
                        for p in [self.path / ".dummy", self.path]:
                            pass
                        os.rmdir(self.path)
                    except Exception:
                        pass
                    try:
                        os.remove(self._lock_meta_path())
                    except Exception:
                        pass
                    return self.acquire()
            except Exception:
                return False
            return False
        except Exception:
            return False

    def _write_meta(self, meta: dict):
        p = self._lock_meta_path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(str(meta), encoding="utf-8")
        os.replace(tmp, p)

    def _read_meta(self) -> Optional[dict]:
        p = self._lock_meta_path()
        if not p.exists():
            return None
        try:
            txt = p.read_text(encoding="utf-8")
            # stored as Python dict repr; parse safely
            # allow JSON too
            try:
                import json
                return json.loads(txt)
            except Exception:
                # fallback eval-like parse for simple dicts
                txt = txt.strip()
                if txt.startswith("{") and txt.endswith("}"):
                    items = {}
                    for part in txt[1:-1].split(","):
                        if ":" not in part:
                            continue
                        k, v = part.split(":", 1)
                        items[k.strip().strip("'\"")] = v.strip().strip("'\"")
                    return items
            return None
        except Exception:
            return None

    def release(self):
        if not self.acquired:
            return
        try:
            meta = self._lock_meta_path()
            if meta.exists():
                try:
                    os.remove(meta)
                except Exception:
                    pass
            if self.path.exists():
                try:
                    os.rmdir(self.path)
                except Exception:
                    pass
        finally:
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
