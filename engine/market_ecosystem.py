import logging
import random
from collections import defaultdict

logger = logging.getLogger("market_ecosystem")


class MarketEcosystem:
    """
    Shared synthetic market environment.
    - Maintains prices per symbol
    - Broadcasts prices to agents
    - Collects their decisions
    - Matches trades in a simple clearing mechanism
    - Tracks agent PnL and market metrics
    """

    def __init__(self, symbols, agents, initial_price=100.0):
        self.symbols = symbols
        self.agents = agents
        self.prices = {sym: initial_price for sym in symbols}
        self.trade_history = defaultdict(list)
        self.tick_count = 0

    def step(self):
        """
        Run one tick of the market:
          - Update prices (random walk baseline)
          - Broadcast state to agents
          - Collect decisions
          - Execute trades
        """
        self.tick_count += 1

        # --- Price update (random walk baseline) ---
        for sym in self.symbols:
            drift = random.uniform(-0.5, 0.5)
            self.prices[sym] = max(0.01, self.prices[sym] + drift)

        # --- Broadcast state ---
        for agent in self.agents:
            if hasattr(agent, "symbol_b"):  # ArbitrageAgent needs both
                for sym, price in self.prices.items():
                    agent.observe({"symbol": sym, "price": price})
            else:
                price = self.prices[agent.symbol]
                agent.observe({"symbol": agent.symbol, "price": price})

        # --- Collect decisions & execute trades ---
        for agent in self.agents:
            decision = agent.decide()
            price = self.prices[agent.symbol]
            agent.act(decision, price)
            self.trade_history[agent.name].append({
                "tick": self.tick_count,
                "decision": decision,
                "price": price,
                "net_worth": agent.net_worth(price)
            })

    def run(self, ticks=100):
        for _ in range(ticks):
            self.step()

    def report(self):
        """
        Summarize agent performance and market stats.
        """
        summary = {}
        for agent in self.agents:
            price = self.prices[agent.symbol]
            summary[agent.name] = {
                "final_cash": agent.cash,
                "final_position": agent.position,
                "final_net_worth": agent.net_worth(price),
                "trades": len(agent.trade_log),
            }
        return {
            "ticks": self.tick_count,
            "final_prices": self.prices,
            "agents": summary,
        }
