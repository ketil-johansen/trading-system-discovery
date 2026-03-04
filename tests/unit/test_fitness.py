"""Unit tests for the fitness function."""

from __future__ import annotations

import pytest

from tsd.optimization.fitness import FitnessConfig, compute_fitness
from tsd.strategy.evaluator import BacktestMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(
    num_trades: int = 50,
    win_rate: float = 0.85,
    net_profit: float = 5000.0,
) -> BacktestMetrics:
    """Build a BacktestMetrics with controllable gate-relevant fields."""
    num_wins = int(num_trades * win_rate)
    num_losses = num_trades - num_wins
    return BacktestMetrics(
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        net_profit=net_profit,
        gross_profit=max(net_profit, 0.0),
        gross_loss=min(net_profit, 0.0),
        profit_factor=2.0,
        avg_win_pct=0.03,
        avg_loss_pct=-0.02,
        win_loss_ratio=1.5,
        max_drawdown_pct=0.05,
        max_drawdown_duration=3,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        calmar_ratio=2.0,
        avg_holding_days=5.0,
        longest_win_streak=5,
        longest_loss_streak=2,
        expectancy_per_trade=0.01,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeFitness:
    """Tests for compute_fitness."""

    def test_all_gates_pass(self) -> None:
        """Returns win_rate when all gates pass."""
        metrics = _make_metrics(num_trades=50, win_rate=0.85, net_profit=5000.0)
        assert compute_fitness(metrics) == pytest.approx(0.85)

    def test_fails_min_trades(self) -> None:
        """Returns 0.0 when trades below minimum."""
        metrics = _make_metrics(num_trades=10, win_rate=0.90, net_profit=5000.0)
        assert compute_fitness(metrics) == 0.0

    def test_fails_net_profit(self) -> None:
        """Returns 0.0 when net profit is not positive."""
        metrics = _make_metrics(num_trades=50, win_rate=0.85, net_profit=-100.0)
        assert compute_fitness(metrics) == 0.0

    def test_fails_win_rate(self) -> None:
        """Returns 0.0 when win rate below minimum."""
        metrics = _make_metrics(num_trades=50, win_rate=0.60, net_profit=5000.0)
        assert compute_fitness(metrics) == 0.0

    def test_exactly_at_thresholds(self) -> None:
        """Edge case: exactly at min_trades and min_win_rate passes."""
        cfg = FitnessConfig(min_trades=30, min_win_rate=0.80)
        metrics = _make_metrics(num_trades=30, win_rate=0.80, net_profit=1.0)
        assert compute_fitness(metrics, cfg) == pytest.approx(0.80)

    def test_exactly_below_thresholds(self) -> None:
        """Edge case: one below min_trades fails."""
        cfg = FitnessConfig(min_trades=30, min_win_rate=0.80)
        metrics = _make_metrics(num_trades=29, win_rate=0.80, net_profit=1.0)
        assert compute_fitness(metrics, cfg) == 0.0

    def test_custom_config(self) -> None:
        """Non-default thresholds work correctly."""
        cfg = FitnessConfig(min_trades=10, min_win_rate=0.50, require_net_profitable=False)
        metrics = _make_metrics(num_trades=15, win_rate=0.55, net_profit=-100.0)
        assert compute_fitness(metrics, cfg) == pytest.approx(0.55)

    def test_zero_trades(self) -> None:
        """Zero trades returns 0.0."""
        metrics = _make_metrics(num_trades=0, win_rate=0.0, net_profit=0.0)
        assert compute_fitness(metrics) == 0.0
