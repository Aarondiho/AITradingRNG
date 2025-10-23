import numpy as np
import logging
from collections import deque
from typing import Dict, List

logger = logging.getLogger("drift_monitor")


class DriftMonitor:
    """
    Detects long-term drift in fingerprint metrics using rolling windows.
    Tracks persistent divergence beyond tolerance thresholds.
    """

    def __init__(self, window_size: int = 7, tolerance: float = 0.05):
        """
        Args:
          window_size: number of periods (e.g. days) in rolling window
          tolerance: acceptable relative difference before drift is flagged
        """
        self.window_size = window_size
        self.tolerance = tolerance
        self.history: Dict[str, Dict[str, deque]] = {}

    def update(self, symbol: str, real_fp: dict, synth_fp: dict) -> Dict[str, float]:
        """
        Update rolling drift metrics for a symbol.
        Returns current drift profile.
        """
        if symbol not in self.history:
            self.history[symbol] = {
                "bias": deque(maxlen=self.window_size),
                "volatility": deque(maxlen=self.window_size),
                "autocorr": deque(maxlen=self.window_size),
            }

        # --- Compute diffs ---
        bias_diff = self._safe_diff(
            real_fp.get("candle_level", {}).get("horizon_60", {}).get("direction_bias"),
            synth_fp.get("candle_level", {}).get("horizon_60", {}).get("direction_bias"),
        )
        vol_diff = self._safe_diff(
            real_fp.get("candle_level", {}).get("horizon_60", {}).get("volatility"),
            synth_fp.get("candle_level", {}).get("horizon_60", {}).get("volatility"),
        )
        autocorr_diff = self._safe_diff(
            real_fp.get("tick_level", {}).get("autocorr_lag1"),
            synth_fp.get("tick_level", {}).get("autocorr_lag1"),
        )

        # --- Update history ---
        self.history[symbol]["bias"].append(bias_diff)
        self.history[symbol]["volatility"].append(vol_diff)
        self.history[symbol]["autocorr"].append(autocorr_diff)

        # --- Compute rolling averages ---
        profile = {
            "bias_drift": float(np.mean(self.history[symbol]["bias"])),
            "volatility_drift": float(np.mean(self.history[symbol]["volatility"])),
            "autocorr_drift": float(np.mean(self.history[symbol]["autocorr"])),
        }

        logger.info("[DRIFT] %s profile: %s", symbol, profile)
        return profile

    def detect_persistent_drift(self, profile: Dict[str, float]) -> bool:
        """
        Returns True if persistent drift is detected beyond tolerance.
        """
        for k, v in profile.items():
            if abs(v) > self.tolerance:
                return True
        return False

    @staticmethod
    def _safe_diff(a, b):
        if a is None or b is None:
            return 0.0
        return abs(a - b)
