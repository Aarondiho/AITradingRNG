import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf


class FingerprintExtractor:
    """
    Extracts statistical fingerprints from tick or candle data.
    Supports:
      - Tick-level metrics (autocorr, skew, kurtosis)
      - Candle-level metrics (directional bias, volatility, body/wick ratios)
    """

    def __init__(self, lags=(1, 5, 10)):
        self.lags = lags

    def tick_level(self, prices: np.ndarray) -> dict:
        """
        Compute tick-level fingerprints.
        Args:
          prices: np.ndarray of tick prices
        Returns:
          dict of metrics
        """
        returns = np.diff(prices)
        acorr = acf(returns, nlags=max(self.lags), fft=True)

        return {
            f"autocorr_lag{lag}": float(acorr[lag]) for lag in self.lags
        } | {
            "skew": float(skew(returns)),
            "kurtosis": float(kurtosis(returns, fisher=True) + 3),  # normal=3
        }

    def candle_level(self, prices: np.ndarray, horizon: int = 60) -> dict:
        """
        Compute candle-level fingerprints.
        Args:
          prices: np.ndarray of tick prices
          horizon: number of ticks per candle
        Returns:
          dict of metrics
        """
        n = len(prices) // horizon
        if n == 0:
            return {}

        candles = []
        for i in range(n):
            chunk = prices[i * horizon:(i + 1) * horizon]
            o, h, l, c = chunk[0], np.max(chunk), np.min(chunk), chunk[-1]
            candles.append({"open": o, "high": h, "low": l, "close": c})

        df = pd.DataFrame(candles)
        bodies = (df["close"] - df["open"]).values
        wicks = (df["high"] - df["low"]).values

        return {
            "direction_bias": float(np.mean(df["close"] > df["open"])),
            "avg_body": float(np.mean(np.abs(bodies))),
            "avg_wick": float(np.mean(wicks)),
            "volatility": float(np.std(bodies)),
        }

    def full_fingerprint(self, prices: np.ndarray, horizons=(60, 120, 180)) -> dict:
        """
        Compute full fingerprint across tick-level and multiple candle horizons.
        """
        result = {"tick_level": self.tick_level(prices), "candle_level": {}}
        for h in horizons:
            result["candle_level"][f"horizon_{h}"] = self.candle_level(prices, horizon=h)
        return result
