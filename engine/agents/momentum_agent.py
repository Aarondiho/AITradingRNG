import collections
from engine.agent_base import AgentBase


class MomentumAgent(AgentBase):
    """
    Momentum agent: trades based on moving average crossover.
    - If short-term MA > long-term MA → buy
    - If short-term MA < long-term MA → sell
    - Otherwise → hold
    """

    def __init__(self, name: str, symbol: str,
                 short_window: int = 5, long_window: int = 20,
                 cash: float = 1000.0, position: float = 0.0):
        super().__init__(name, symbol, cash, position)
        self.short_window = short_window
        self.long_window = long_window
        self.prices = collections.deque(maxlen=long_window)

    def observe(self, market_state: dict):
        price = market_state.get("price")
        if price is not None:
            self.prices.append(price)

    def decide(self) -> dict:
        if len(self.prices) < self.long_window:
            return {"action": "hold", "size": 0.0}

        short_ma = sum(list(self.prices)[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices) / len(self.prices)

        if short_ma > long_ma * 1.001:  # small buffer to avoid noise
            return {"action": "buy", "size": 1.0}
        elif short_ma < long_ma * 0.999:
            return {"action": "sell", "size": 1.0}
        else:
            return {"action": "hold", "size": 0.0}
