# engine/metrics_emitter.py
"""
Lightweight metrics emitter that writes simple JSON metrics to disk for scraping/inspection.

Design:
- No external deps. Emits metrics as atomic JSON files under:
    <base_dir>/<symbol or global>/metrics/<component>/<metric_name>.json
- Each metric file contains {"value": ..., "ts": ISO}
- If a symbol is provided to the emitter or to individual calls, metrics are written under that symbol's folder.
- Useful for lightweight dashboards or periodic scraping by Prometheus node_exporter textfile style collectors.

Usage:
  from engine.metrics_emitter import MetricsEmitter
  m = MetricsEmitter(base_dir="data", default_symbol="R_TEST")
  m.gauge("parity.pass_rate", 0.95, component="parity")  # writes to data/R_TEST/metrics/parity/parity.pass_rate.json
  m.counter_inc("merge.new_shards", 1, symbol="ANOTHER")  # writes to data/ANOTHER/metrics/default/merge.new_shards.json
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

class MetricsEmitter:
    def __init__(self, base_dir: str = "data", default_symbol: Optional[str] = None):
        """
        base_dir: top-level data directory
        default_symbol: optional symbol name to scope metrics by default
        """
        self.base_dir = Path(base_dir)
        self.default_symbol = default_symbol

    def _metrics_dir(self, component: str, symbol: Optional[str]) -> Path:
        """
        Resolve directory where metric files should be written.
        If symbol provided (or default_symbol set), metrics go under <base_dir>/<symbol>/metrics/<component>.
        Otherwise they go under <base_dir>/metrics/<component>.
        """
        if symbol is None:
            symbol = self.default_symbol
        if symbol:
            dir_path = self.base_dir / symbol / "metrics" / component
        else:
            dir_path = self.base_dir / "metrics" / component
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _write(self, component: str, name: str, payload: dict, symbol: Optional[str]):
        comp_dir = self._metrics_dir(component, symbol)
        # sanitize filename: use metric name as provided (should avoid path separators)
        filename = f"{name}.json"
        path = comp_dir / filename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def gauge(self, name: str, value, component: str = "default", symbol: Optional[str] = None):
        """
        Write a gauge metric (sets value).
        If symbol provided, metric stored under that symbol folder; otherwise default_symbol or global used.
        """
        payload = {"type": "gauge", "value": value, "ts": datetime.utcnow().isoformat() + "Z"}
        self._write(component, name, payload, symbol)

    def counter_inc(self, name: str, inc: int = 1, component: str = "default", symbol: Optional[str] = None):
        """
        Increment a counter. Counter value is stored as a gauge file with the accumulated integer.
        Counter lookup respects symbol scoping.
        """
        comp_dir = self._metrics_dir(component, symbol)
        path = comp_dir / f"{name}.json"
        val = 0
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                val = int(data.get("value", 0))
        except Exception:
            val = 0
        val += int(inc)
        self.gauge(name, val, component=component, symbol=symbol or self.default_symbol)

    def timing(self, name: str, seconds: float, component: str = "default", symbol: Optional[str] = None):
        """
        Record a timing metric (seconds).
        """
        payload = {"type": "timing", "value": float(seconds), "ts": datetime.utcnow().isoformat() + "Z"}
        self._write(component, name, payload, symbol)

    def set_default_symbol(self, symbol: Optional[str]):
        """
        Set or clear the emitter's default symbol.
        """
        self.default_symbol = symbol
