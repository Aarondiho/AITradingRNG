import random
from engine.agent_base import AgentBase


class NoiseAgent(AgentBase):
    """
    Noise agent: submits random buy/sell/hold actions.
    Provides background liquidity and randomness in the ecology.
    """

    def __init__(self, name: str, symbol: str, cash: float = 1000.0, position: float = 0.0, seed: int = 42):
        super().__init__(name, symbol, cash, position)
        random.seed(seed)

    def observe(self, market_state: dict):
        # Noise agent ignores market state (purely random)
        pass

    def decide(self) -> dict:
        action = random.choice(["buy", "sell", "hold"])
        size = random.uniform(0.1, 1.0) if action in ("buy", "sell") else 0.0
        return {"action": action, "size": size}
