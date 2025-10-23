from engine.agent_base import AgentBase


class ArbitrageAgent(AgentBase):
    """
    Arbitrage agent:
    - Monitors two correlated symbols (e.g. R_25 vs R_50).
    - Computes price ratio relative to baseline.
    - If ratio deviates beyond threshold, trades to exploit mispricing.
    """

    def __init__(self, name: str, symbol_a: str, symbol_b: str,
                 baseline_ratio: float = 1.0, threshold: float = 0.02,
                 cash: float = 1000.0, position: float = 0.0):
        # Note: symbol here is the "primary" one for accounting
        super().__init__(name, symbol_a, cash, position)
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.baseline_ratio = baseline_ratio
        self.threshold = threshold
        self.last_prices = {symbol_a: None, symbol_b: None}

    def observe(self, market_state: dict):
        # Expecting market_state = {"symbol": str, "price": float}
        sym = market_state.get("symbol")
        price = market_state.get("price")
        if sym in self.last_prices:
            self.last_prices[sym] = price

    def decide(self) -> dict:
        pa = self.last_prices[self.symbol_a]
        pb = self.last_prices[self.symbol_b]
        if pa is None or pb is None:
            return {"action": "hold", "size": 0.0}

        ratio = pa / pb if pb > 0 else 1.0
        deviation = (ratio - self.baseline_ratio) / self.baseline_ratio

        if deviation > self.threshold:
            # Symbol A overpriced relative to B → sell A
            return {"action": "sell", "size": 1.0}
        elif deviation < -self.threshold:
            # Symbol A underpriced relative to B → buy A
            return {"action": "buy", "size": 1.0}
        else:
            return {"action": "hold", "size": 0.0}
