"""Composite fitness function with win rate, frequency band-pass, and regularity.

Evaluates a BacktestMetrics result against hard gates (minimum trades,
net profitability, minimum win rate), then scores as:

    fitness = win_rate × frequency_score × regularity_score

where frequency_score rewards a per-stock-per-year trade rate within a
target band and regularity_score penalizes uneven yearly trade distribution.
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
        min_rate: Minimum trades per stock per year for full score.
        max_rate: Maximum trades per stock per year for full score.
            Above this, frequency_score declines linearly to 0 at 2×max_rate.
    """

    min_trades: int = 10
    min_win_rate: float = 0.80
    require_net_profitable: bool = True
    min_rate: float = 0.5
    max_rate: float = 2.0


def load_fitness_config() -> FitnessConfig:
    """Load fitness configuration from environment variables."""
    return FitnessConfig(
        min_trades=env_int("TSD_FITNESS_MIN_TRADES", 10),
        min_win_rate=env_float("TSD_FITNESS_MIN_WIN_RATE", 0.80),
        min_rate=env_float("TSD_FITNESS_MIN_RATE", 0.5),
        max_rate=env_float("TSD_FITNESS_MAX_RATE", 2.0),
    )


def compute_fitness(
    metrics: BacktestMetrics,
    config: FitnessConfig | None = None,
    trades: tuple[TradeRecord, ...] = (),
    num_stocks: int = 1,
) -> float:
    """Compute composite fitness from backtest metrics and trades.

    Fitness = win_rate × frequency_score × regularity_score

    Gates checked first (binary pass/fail):
    1. Minimum number of trades.
    2. Net profitability (if required).
    3. Minimum win rate.

    Components (all 0.0–1.0):
    - win_rate: directly from metrics.
    - frequency_score: band-pass on trades per stock per year.
      Ramps up below min_rate, flat 1.0 between min_rate and max_rate,
      ramps down above max_rate (reaches 0.0 at 2×max_rate).
    - regularity_score: max(0, 1 - CV of yearly trade counts).
      Falls back to 1.0 if trades are not provided.

    Args:
        metrics: Aggregate backtest metrics.
        config: Fitness configuration. Uses defaults if None.
        trades: All trade records for regularity and frequency calculation.
        num_stocks: Number of stocks in the universe.

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

    frequency = _compute_frequency_score(trades, num_stocks, config)
    regularity = _compute_regularity_score(trades) if trades else 1.0

    return metrics.win_rate * frequency * regularity


def _compute_frequency_score(
    trades: tuple[TradeRecord, ...],
    num_stocks: int,
    config: FitnessConfig,
) -> float:
    """Compute frequency score as a band-pass on trades per stock per year.

    The score is 1.0 when the annualized per-stock rate is between
    min_rate and max_rate. Below min_rate it ramps up linearly from 0.
    Above max_rate it ramps down linearly, reaching 0 at 2×max_rate.

    Args:
        trades: All trade records (used to determine year span).
        num_stocks: Number of stocks in the universe.
        config: Fitness configuration with min_rate and max_rate.

    Returns:
        Frequency score between 0.0 and 1.0.
    """
    if not trades or num_stocks < 1:
        return 0.0

    years = {t.entry_date[:4] for t in trades}
    num_years = len(years)
    if num_years < 1:
        return 0.0

    rate = len(trades) / (num_stocks * num_years)

    if rate < config.min_rate:
        return rate / config.min_rate
    if rate <= config.max_rate:
        return 1.0
    # Ramp down: 1.0 at max_rate, 0.0 at 2×max_rate
    score: float = max(0.0, 2.0 - rate / config.max_rate)
    return score


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
