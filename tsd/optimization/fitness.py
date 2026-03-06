"""Composite fitness function with win rate, trade volume, and regularity.

Evaluates a BacktestMetrics result against hard gates (minimum trades,
net profitability, minimum win rate), then scores as:

    fitness = win_rate × trade_volume_score × regularity_score

where trade_volume_score rewards more trades (saturating at target_trades)
and regularity_score penalizes uneven yearly trade distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsd.config import env_float, env_int
from tsd.strategy.evaluator import BacktestMetrics, TradeRecord


@dataclass(frozen=True)
class FitnessConfig:
    """Configuration for the composite fitness function.

    Attributes:
        min_trades: Minimum number of trades required.
        min_win_rate: Minimum win rate to pass (as fraction, e.g. 0.80).
        require_net_profitable: Whether net profit must be positive.
        target_trades: Trade count at which volume score saturates to 1.0.
    """

    min_trades: int = 10
    min_win_rate: float = 0.80
    require_net_profitable: bool = True
    target_trades: int = 50


def load_fitness_config() -> FitnessConfig:
    """Load fitness configuration from environment variables."""
    return FitnessConfig(
        min_trades=env_int("TSD_FITNESS_MIN_TRADES", 10),
        min_win_rate=env_float("TSD_FITNESS_MIN_WIN_RATE", 0.80),
        target_trades=env_int("TSD_FITNESS_TARGET_TRADES", 50),
    )


def compute_fitness(
    metrics: BacktestMetrics,
    config: FitnessConfig | None = None,
    trades: tuple[TradeRecord, ...] = (),
) -> float:
    """Compute composite fitness from backtest metrics and trades.

    Fitness = win_rate × trade_volume_score × regularity_score

    Gates checked first (binary pass/fail):
    1. Minimum number of trades.
    2. Net profitability (if required).
    3. Minimum win rate.

    Components (all 0.0–1.0):
    - win_rate: directly from metrics.
    - trade_volume_score: min(num_trades / target_trades, 1.0).
    - regularity_score: max(0, 1 - CV of yearly trade counts).
      Falls back to 1.0 if trades are not provided.

    Args:
        metrics: Aggregate backtest metrics.
        config: Fitness configuration. Uses defaults if None.
        trades: All trade records for regularity calculation.

    Returns:
        Composite fitness score, or 0.0 if any gate fails.
    """
    if config is None:
        config = FitnessConfig()

    if metrics.num_trades < config.min_trades:
        return 0.0

    if config.require_net_profitable and metrics.net_profit <= 0:
        return 0.0

    if metrics.win_rate < config.min_win_rate:
        return 0.0

    volume_score = min(metrics.num_trades / config.target_trades, 1.0)
    regularity = _compute_regularity_score(trades) if trades else 1.0

    return metrics.win_rate * volume_score * regularity


def _compute_regularity_score(trades: tuple[TradeRecord, ...]) -> float:
    """Compute regularity score from yearly trade distribution.

    Returns max(0, 1 - CV) where CV is coefficient of variation
    of yearly trade counts. Score of 1.0 means perfectly even
    distribution; 0.0 means extremely uneven.

    Args:
        trades: All trade records.

    Returns:
        Regularity score between 0.0 and 1.0.
    """
    yearly_counts: dict[str, int] = {}
    for t in trades:
        year = t.entry_date[:4]
        yearly_counts[year] = yearly_counts.get(year, 0) + 1

    counts = list(yearly_counts.values())
    if len(counts) < 2:  # noqa: PLR2004
        return 0.0

    mean = sum(counts) / len(counts)
    if mean <= 0:
        return 0.0

    std = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5
    cv = std / mean
    score: float = max(0.0, 1.0 - cv)
    return score
