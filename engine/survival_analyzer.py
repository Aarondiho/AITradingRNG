import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger("survival_analyzer")


class SurvivalAnalyzer:
    """
    Analyzes simulator + agent resilience under stress.
    Computes survival time, recovery time, agent drawdowns, and ecology stability.
    """

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    def survival_time(self, real_fp: dict, shocked_fp: dict) -> int:
        """
        Estimate survival time: number of ticks until fingerprints diverge beyond tolerance.
        """
        ticks_survived = 0
        for k, v in real_fp["tick_level"].items():
            sv = shocked_fp["tick_level"].get(k, None)
            if sv is None:
                continue
            diff = abs(v - sv)
            if diff <= self.tolerance * max(abs(v), 1e-6):
                ticks_survived += 1
        return ticks_survived

    def recovery_time(self, shocked_series: np.ndarray, baseline_series: np.ndarray) -> int:
        """
        Estimate recovery time: ticks until shocked series re-aligns with baseline mean/variance.
        """
        baseline_mean, baseline_std = np.mean(baseline_series), np.std(baseline_series)
        for i in range(1, len(shocked_series)):
            window = shocked_series[-i:]
            if abs(np.mean(window) - baseline_mean) < self.tolerance * abs(baseline_mean) and \
               abs(np.std(window) - baseline_std) < self.tolerance * baseline_std:
                return i
        return len(shocked_series)

    def agent_drawdowns(self, agents: List, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Compute max drawdown for each agent based on trade log and net worth.
        """
        results = {}
        for agent in agents:
            nw_series = []
            for trade in agent.trade_log:
                price = trade.get("price", current_prices.get(agent.symbol, 0))
                nw_series.append(agent.net_worth(price))
            if not nw_series:
                results[agent.name] = 0.0
                continue
            peak = nw_series[0]
            max_dd = 0.0
            for nw in nw_series:
                peak = max(peak, nw)
                dd = (peak - nw) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            results[agent.name] = round(max_dd * 100, 2)  # percentage
        return results

    def ecology_stability(self, prices: np.ndarray) -> float:
        """
        Measure ecology stability as volatility of returns.
        """
        returns = np.diff(prices)
        return float(np.std(returns))
