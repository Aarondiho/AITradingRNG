import numpy as np
import random


class StressInjector:
    """
    Injects controlled shocks into tick price streams for stress-testing.
    Supported shock types:
      - volatility_spike
      - drift_shift
      - liquidity_drought
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def apply(self, prices: np.ndarray, shock_type: str,
              intensity: float = 2.0, duration: int = 300) -> np.ndarray:
        """
        Apply a shock to a tick price series.

        Args:
          prices: np.ndarray of tick prices
          shock_type: one of {"volatility_spike", "drift_shift", "liquidity_drought"}
          intensity: magnitude of shock (e.g. multiplier for volatility, drift per tick)
          duration: number of ticks to apply shock

        Returns:
          np.ndarray of shocked prices
        """
        shocked = prices.copy()
        n = len(prices)
        if n < duration:
            duration = n

        start = random.randint(0, n - duration)
        end = start + duration

        if shock_type == "volatility_spike":
            # Multiply returns by intensity factor
            returns = np.diff(shocked[start:end+1])
            shocked[start+1:end+1] = shocked[start] + np.cumsum(returns * intensity)

        elif shock_type == "drift_shift":
            # Add constant drift per tick
            drift = np.linspace(0, intensity * duration, duration)
            shocked[start:end] += drift

        elif shock_type == "liquidity_drought":
            # Flatten variance by reducing tick-to-tick changes
            segment = shocked[start:end]
            mean_price = np.mean(segment)
            shocked[start:end] = mean_price + (segment - mean_price) * (1.0 / intensity)

        else:
            raise ValueError(f"Unsupported shock_type: {shock_type}")

        return shocked
