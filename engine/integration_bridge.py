import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("integration_bridge")


class IntegrationBridge:
    """
    Publishes forecasts into external systems:
      - Local JSON files (for dashboards)
      - API payloads (for external consumers)
      - Console/log streaming
    """

    def __init__(self, out_dir="reports/forecasts", api_hook=None):
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.api_hook = api_hook  # Optional callable for pushing to external API

    def publish_forecast(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish forecast to local storage and optionally to external API.
        Returns publication manifest.
        """
        symbol = forecast.get("symbol", "UNKNOWN")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = self.out_dir / f"{symbol}_forecast_{ts}.json"

        # --- Save locally ---
        out_path.write_text(json.dumps(forecast, indent=2))
        logger.info("[BRIDGE] Forecast saved: %s", out_path)

        # --- Push to API if hook provided ---
        api_status = None
        if self.api_hook:
            try:
                api_status = self.api_hook(forecast)
                logger.info("[BRIDGE] Forecast pushed to API for %s", symbol)
            except Exception as e:
                logger.error("[BRIDGE] API push failed for %s: %s", symbol, e)
                api_status = {"status": "failed", "error": str(e)}

        return {
            "symbol": symbol,
            "local_path": str(out_path),
            "api_status": api_status or {"status": "skipped"},
        }

    def batch_publish(self, forecasts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Publish multiple forecasts at once.
        """
        results = {}
        for sym, fc in forecasts.items():
            results[sym] = self.publish_forecast(fc)
        return results
