import pandas as pd
import numpy as np

class MultiScaleAggregator:
    """
    Aggregates tick prices into candles across multiple horizons.
    """

    def __init__(self, horizons=(60, 120, 180, 300)):
        self.horizons = horizons

    def build_candles(self, prices: np.ndarray) -> dict:
        """
        Build OHLC candles for each horizon.
        Args:
          prices: np.ndarray of tick prices
        Returns:
          dict {horizon: DataFrame with open, high, low, close}
        """
        results = {}
        for h in self.horizons:
            n = len(prices) // h
            if n == 0:
                continue
            chunks = [prices[i*h:(i+1)*h] for i in range(n)]
            ohlc = []
            for chunk in chunks:
                ohlc.append({
                    "open": chunk[0],
                    "high": np.max(chunk),
                    "low": np.min(chunk),
                    "close": chunk[-1],
                })
            results[h] = pd.DataFrame(ohlc)
        return results
