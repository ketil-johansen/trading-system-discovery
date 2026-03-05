"""Shared metric aggregation utilities for optimization engines."""

from __future__ import annotations

from tsd.strategy.evaluator import BacktestMetrics, BacktestResult


def aggregate_metrics(results: list[BacktestResult]) -> BacktestMetrics:
    """Aggregate backtest metrics across multiple stocks.

    Sums num_trades, wins, losses, and net_profit. Recomputes win_rate.
    Other fields are set to 0.0 as only gate-relevant fields matter
    for fitness evaluation.

    Args:
        results: List of backtest results from individual stocks.

    Returns:
        Aggregated BacktestMetrics.
    """
    if not results:
        return empty_metrics()

    total_trades = sum(r.metrics.num_trades for r in results)
    total_wins = sum(r.metrics.num_wins for r in results)
    total_losses = sum(r.metrics.num_losses for r in results)
    total_net_profit = sum(r.metrics.net_profit for r in results)

    win_rate = total_wins / total_trades if total_trades > 0 else 0.0

    return BacktestMetrics(
        num_trades=total_trades,
        num_wins=total_wins,
        num_losses=total_losses,
        win_rate=win_rate,
        net_profit=total_net_profit,
        gross_profit=0.0,
        gross_loss=0.0,
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
    )


def empty_metrics() -> BacktestMetrics:
    """Return BacktestMetrics with all fields set to zero.

    Returns:
        BacktestMetrics with zero values.
    """
    return BacktestMetrics(
        num_trades=0,
        num_wins=0,
        num_losses=0,
        win_rate=0.0,
        net_profit=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
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
    )
