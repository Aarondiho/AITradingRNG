import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger("forecaster")


class Forecaster:
    """
    Generates forward projections for RNG behavior using drift atlas + live fingerprints.
    Produces bias, volatility, and autocorr forecasts with confidence intervals.
    """

    def __init__(self, horizon_hours: int = 24, confidence_level: float = 0.8):
        self.horizon_hours = horizon_hours
        self.confidence_level = confidence_level

    def forecast(self, symbol: str, drift_atlas: Dict[str, Any], live_fp: dict) -> dict:
        """
        Generate forecast for a symbol.
        Args:
          drift_atlas: historical drift profile (from DriftProfiler.build_drift_atlas)
          live_fp: latest real vs synthetic fingerprint comparison
        Returns:
          forecast dict with projections and confidence intervals
        """
        history = drift_atlas.get("history", [])
        if not history:
            logger.warning("[FORECAST] No drift history for %s", symbol)
            return {"decision": "NO_FORECAST", "rationale": "Insufficient history"}

        # Extract time series
        bias_vals = [h["bias_drift"] for h in history if h["bias_drift"] is not None]
        vol_vals = [h["volatility_drift"] for h in history if h["volatility_drift"] is not None]
        auto_vals = [h["autocorr_drift"] for h in history if h["autocorr_drift"] is not None]

        def project_trend(series):
            if len(series) < 2:
                return 0.0, (0.0, 0.0)
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, 1)  # linear trend
            slope = coeffs[0]
            forecast_val = series[-1] + slope * (self.horizon_hours / 24.0)
            ci = (forecast_val - abs(slope), forecast_val + abs(slope))
            return forecast_val, ci

        bias_forecast, bias_ci = project_trend(bias_vals)
        vol_forecast, vol_ci = project_trend(vol_vals)
        auto_forecast, auto_ci = project_trend(auto_vals)

        forecast = {
            "symbol": symbol,
            "date": datetime.utcnow().date().isoformat(),
            "forecast_horizon": f"{self.horizon_hours}h",
            "predictions": {
                "bias_trend": "upward" if bias_forecast > bias_vals[-1] else "downward",
                "bias_confidence": round(self.confidence_level, 2),
                "bias_forecast": bias_forecast,
                "bias_ci": bias_ci,
                "volatility_trend": "upward" if vol_forecast > vol_vals[-1] else "downward",
                "volatility_confidence": round(self.confidence_level, 2),
                "volatility_forecast": vol_forecast,
                "volatility_ci": vol_ci,
                "autocorr_trend": "upward" if auto_forecast > auto_vals[-1] else "downward",
                "autocorr_confidence": round(self.confidence_level, 2),
                "autocorr_forecast": auto_forecast,
                "autocorr_ci": auto_ci,
            },
            "decision": "FORECAST",
            "rationale": "Forecast generated from drift atlas trends and live fingerprints"
        }

        logger.info("[FORECAST] %s forecast: %s", symbol, forecast["predictions"])
        return forecast
