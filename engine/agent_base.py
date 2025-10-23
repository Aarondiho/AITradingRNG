import abc
import logging

logger = logging.getLogger("agent_base")


class AgentBase(abc.ABC):
    """
    Abstract base class for all agents in the synthetic market ecology.
    Agents observe market state, decide on an action, and act (submit orders).
    """

    def __init__(self, name: str, symbol: str, cash: float = 1000.0, position: float = 0.0):
        self.name = name
        self.symbol = symbol
        self.cash = cash
        self.position = position
        self.trade_log = []

    @abc.abstractmethod
    def observe(self, market_state: dict):
        """Observe current market state (quotes, order book, etc.)."""
        pass

    @abc.abstractmethod
    def decide(self) -> dict:
        """
        Decide on an action.
        Returns:
          dict with keys {action: "buy"/"sell"/"hold", size: float}
        """
        pass

    def act(self, decision: dict, price: float):
        """
        Execute decision at given price.
        Updates cash, position, and logs trade.
        """
        action = decision.get("action", "hold")
        size = decision.get("size", 0.0)

        if action == "buy" and size > 0:
            cost = price * size
            if self.cash >= cost:
                self.cash -= cost
                self.position += size
                self.trade_log.append({"action": "buy", "size": size, "price": price})
                logger.debug("[%s] Bought %.2f @ %.4f", self.name, size, price)

        elif action == "sell" and size > 0:
            if self.position >= size:
                self.cash += price * size
                self.position -= size
                self.trade_log.append({"action": "sell", "size": size, "price": price})
                logger.debug("[%s] Sold %.2f @ %.4f", self.name, size, price)

        else:
            # hold or invalid
            self.trade_log.append({"action": "hold", "size": 0, "price": price})
            logger.debug("[%s] Held position", self.name)

    def net_worth(self, price: float) -> float:
        """Compute current net worth (cash + position*price)."""
        return self.cash + self.position * price
