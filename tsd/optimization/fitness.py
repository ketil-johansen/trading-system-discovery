"""Win-rate fitness function with profitability gate.

Evaluates a BacktestMetrics result against hard gates (minimum trades,
net profitability, minimum win rate) and returns the win rate as the
fitness score if all gates pass, otherwise 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsd.strategy.evaluator import BacktestMetrics


@dataclass(frozen=True)
class FitnessConfig:
    """Configuration for the fitness function gates.

    Attributes:
        min_trades: Minimum number of trades required.
        min_win_rate: Minimum win rate to pass (as fraction, e.g. 0.80).
        require_net_profitable: Whether net profit must be positive.
    """

    min_trades: int = 30
    min_win_rate: float = 0.80
    require_net_profitable: bool = True


def compute_fitness(
    metrics: BacktestMetrics,
    config: FitnessConfig | None = None,
) -> float:
    """Compute fitness score from backtest metrics.

    Returns the win rate if all hard gates pass, otherwise 0.0.

    Gates checked in order:
    1. Minimum number of trades.
    2. Net profitability (if required).
    3. Minimum win rate.

    Args:
        metrics: Aggregate backtest metrics.
        config: Fitness configuration. Uses defaults if None.

    Returns:
        Win rate as fitness score, or 0.0 if any gate fails.
    """
    if config is None:
        config = FitnessConfig()

    if metrics.num_trades < config.min_trades:
        return 0.0

    if config.require_net_profitable and metrics.net_profit <= 0:
        return 0.0

    if metrics.win_rate < config.min_win_rate:
        return 0.0

    return metrics.win_rate
