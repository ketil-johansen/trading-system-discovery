"""Unit tests for the shared metrics aggregation module."""

from __future__ import annotations

import pytest

from tsd.optimization.metrics import aggregate_metrics, empty_metrics
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult


def _make_backtest_result(
    num_trades: int = 50,
    num_wins: int = 42,
    net_profit: float = 500.0,
) -> BacktestResult:
    """Create a BacktestResult with specified metrics."""
    num_losses = num_trades - num_wins
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0
    return BacktestResult(
        trades=(),
        metrics=BacktestMetrics(
            num_trades=num_trades,
            num_wins=num_wins,
            num_losses=num_losses,
            win_rate=win_rate,
            net_profit=net_profit,
            gross_profit=max(net_profit, 0.0),
            gross_loss=min(net_profit, 0.0),
            profit_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            win_loss_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            avg_holding_days=0.0,
            longest_win_streak=0,
            longest_loss_streak=0,
            expectancy_per_trade=0.0,
        ),
    )


@pytest.mark.unit
class TestAggregateMetrics:
    """Tests for aggregate_metrics."""

    def test_aggregate_single_result(self) -> None:
        """Single result passes through with correct values."""
        result = _make_backtest_result(num_trades=40, num_wins=32, net_profit=200.0)
        agg = aggregate_metrics([result])
        assert agg.num_trades == 40
        assert agg.num_wins == 32
        assert agg.num_losses == 8
        assert agg.win_rate == pytest.approx(0.8)
        assert agg.net_profit == pytest.approx(200.0)

    def test_aggregate_multiple_sums(self) -> None:
        """Multiple results are summed and win_rate recomputed."""
        r1 = _make_backtest_result(num_trades=20, num_wins=16, net_profit=100.0)
        r2 = _make_backtest_result(num_trades=30, num_wins=24, net_profit=200.0)
        agg = aggregate_metrics([r1, r2])
        assert agg.num_trades == 50
        assert agg.num_wins == 40
        assert agg.num_losses == 10
        assert agg.win_rate == pytest.approx(0.8)
        assert agg.net_profit == pytest.approx(300.0)

    def test_aggregate_empty_returns_zeros(self) -> None:
        """Empty results list returns all-zero metrics."""
        agg = aggregate_metrics([])
        assert agg.num_trades == 0
        assert agg.num_wins == 0
        assert agg.win_rate == 0.0
        assert agg.net_profit == 0.0


@pytest.mark.unit
class TestEmptyMetrics:
    """Tests for empty_metrics."""

    def test_all_zeros(self) -> None:
        """All fields are zero."""
        m = empty_metrics()
        assert m.num_trades == 0
        assert m.num_wins == 0
        assert m.num_losses == 0
        assert m.win_rate == 0.0
        assert m.net_profit == 0.0
        assert m.max_drawdown_pct == 0.0
