"""Backtest engine and strategy performance metrics.

Simulates trades bar-by-bar from a StrategyGenome + OHLCV DataFrame,
applying correct execution timing per exit category, and computes
aggregate performance metrics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from tsd.indicators.base import compute_indicator
from tsd.strategy.execution import check_limit_exit, shift_to_next_open
from tsd.strategy.exits import (
    compute_breakeven_level,
    compute_chandelier_levels,
    compute_stop_loss_level,
    compute_take_profit_level,
    compute_trailing_stop_levels,
    generate_indicator_exit_signals,
    generate_time_exit_signal,
)
from tsd.strategy.genome import OutputMeta, StrategyGenome
from tsd.strategy.signals import generate_entry_signals

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluatorConfig:
    """Configuration for the backtest evaluator.

    Attributes:
        notional_per_trade: Dollar amount per trade for P&L calculation.
        round_trip_cost_pct: Round-trip transaction cost as percentage.
        atr_period: ATR period for exit level calculations.
        max_trades_per_stock: Maximum trades per stock before early exit.
        max_holding_days: Hard cap on holding period in trading days.
            Positions are force-exited at close after this many bars.
            0 means no cap.
    """

    notional_per_trade: float = 10_000.0
    round_trip_cost_pct: float = 0.30
    atr_period: int = 14
    max_trades_per_stock: int = 200
    max_holding_days: int = 63


@dataclass(frozen=True)
class TradeRecord:
    """Record of a single completed trade.

    Attributes:
        entry_bar: iloc index of the entry bar.
        entry_date: Entry date in ISO format.
        entry_price: Entry execution price.
        exit_bar: iloc index of the exit bar.
        exit_date: Exit date in ISO format.
        exit_price: Exit execution price.
        exit_type: Type of exit that closed the trade.
        gross_return_pct: Gross return as a fraction (not percentage).
        cost_pct: Transaction cost as a fraction.
        net_return_pct: Net return after costs as a fraction.
        net_profit: Net profit in dollars.
        is_win: Whether the trade was profitable after costs.
        holding_days: Number of bars held.
    """

    entry_bar: int
    entry_date: str
    entry_price: float
    exit_bar: int
    exit_date: str
    exit_price: float
    exit_type: str
    gross_return_pct: float
    cost_pct: float
    net_return_pct: float
    net_profit: float
    is_win: bool
    holding_days: int


@dataclass(frozen=True)
class BacktestMetrics:
    """Aggregate performance metrics for a backtest.

    Attributes:
        num_trades: Total number of completed trades.
        num_wins: Number of winning trades.
        num_losses: Number of losing trades.
        win_rate: Fraction of winning trades.
        net_profit: Total net profit in dollars.
        gross_profit: Sum of profits from winning trades.
        gross_loss: Sum of losses from losing trades (negative).
        profit_factor: Gross profit / |gross loss|.
        avg_win_pct: Average net return of winning trades.
        avg_loss_pct: Average net return of losing trades.
        win_loss_ratio: Average win / |average loss|.
        max_drawdown_pct: Maximum drawdown as fraction of peak equity.
        max_drawdown_duration: Longest drawdown in number of trades.
        sharpe_ratio: Annualized Sharpe ratio of trade returns.
        sortino_ratio: Annualized Sortino ratio of trade returns.
        calmar_ratio: Annualized return / max drawdown.
        avg_holding_days: Average holding period in bars.
        longest_win_streak: Longest consecutive winning trades.
        longest_loss_streak: Longest consecutive losing trades.
        expectancy_per_trade: Expected return per trade.
    """

    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    net_profit: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    win_loss_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_holding_days: float
    longest_win_streak: int
    longest_loss_streak: int
    expectancy_per_trade: float


@dataclass(frozen=True)
class BacktestResult:
    """Complete backtest result with trades and metrics.

    Attributes:
        trades: Tuple of all completed trade records.
        metrics: Aggregate performance metrics.
    """

    trades: tuple[TradeRecord, ...]
    metrics: BacktestMetrics


# ---------------------------------------------------------------------------
# Internal simulation context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SimContext:
    """Bundles precomputed data for the simulation loop.

    Attributes:
        df: OHLCV DataFrame (retained for date access in trade records).
        open_arr: Open prices as numpy array.
        high_arr: High prices as numpy array.
        low_arr: Low prices as numpy array.
        close_arr: Close prices as numpy array.
        entry_arr: Entry signals (shifted) as boolean numpy array.
        ind_exit_arr: Indicator exit signals (shifted) as boolean numpy array.
        raw_entry_arr: Unshifted entry signals as boolean numpy array.
        atr_arr: ATR values as numpy array.
        indicator_outputs: Output metadata for indicator comparisons.
        config: Evaluator configuration.
    """

    df: pd.DataFrame
    open_arr: npt.NDArray[np.float64]
    high_arr: npt.NDArray[np.float64]
    low_arr: npt.NDArray[np.float64]
    close_arr: npt.NDArray[np.float64]
    entry_arr: npt.NDArray[np.bool_]
    ind_exit_arr: npt.NDArray[np.bool_]
    raw_entry_arr: npt.NDArray[np.bool_]
    atr_arr: npt.NDArray[np.float64]
    indicator_outputs: dict[str, tuple[OutputMeta, ...]]
    config: EvaluatorConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    genome: StrategyGenome,
    df: pd.DataFrame,
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    config: EvaluatorConfig | None = None,
) -> BacktestResult:
    """Run a backtest for a strategy genome on OHLCV data.

    Args:
        genome: Strategy genome encoding.
        df: OHLCV DataFrame with DatetimeIndex and columns
            Open, High, Low, Close, Volume.
        indicator_outputs: Mapping of indicator name to output metadata.
        config: Evaluator configuration. Uses defaults if None.

    Returns:
        BacktestResult with trade records and aggregate metrics.
    """
    if config is None:
        config = EvaluatorConfig()

    # Compute ATR for exit levels
    atr_result = compute_indicator("atr", df, {"period": config.atr_period})
    atr_series = atr_result.values["atr"]

    # Generate raw entry signals and shift to next open
    raw_entry_signals = generate_entry_signals(genome, df, indicator_outputs)
    shifted_entries = shift_to_next_open(raw_entry_signals)

    # Generate indicator exit signals and shift to next open
    raw_indicator_exits = generate_indicator_exit_signals(genome.indicator_exits, df, indicator_outputs)
    shifted_indicator_exits = shift_to_next_open(raw_indicator_exits)

    ctx = _SimContext(
        df=df,
        open_arr=df["Open"].to_numpy(dtype=np.float64),
        high_arr=df["High"].to_numpy(dtype=np.float64),
        low_arr=df["Low"].to_numpy(dtype=np.float64),
        close_arr=df["Close"].to_numpy(dtype=np.float64),
        entry_arr=shifted_entries.to_numpy(dtype=np.bool_),
        ind_exit_arr=shifted_indicator_exits.to_numpy(dtype=np.bool_),
        raw_entry_arr=raw_entry_signals.to_numpy(dtype=np.bool_),
        atr_arr=atr_series.to_numpy(dtype=np.float64),
        indicator_outputs=indicator_outputs,
        config=config,
    )

    trades = _simulate_trades(genome, ctx)
    metrics = _compute_metrics(trades, config)
    return BacktestResult(trades=tuple(trades), metrics=metrics)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


@dataclass
class _ExitLevels:
    """Precomputed exit levels for a single trade."""

    fixed_stop: float | None = None
    fixed_target: float | None = None
    trailing_levels: npt.NDArray[np.float64] | None = None
    chandelier_levels: npt.NDArray[np.float64] | None = None
    breakeven_levels: npt.NDArray[np.float64] | None = None
    time_exit_signals: npt.NDArray[np.bool_] | None = None


def _compute_exit_levels(
    genome: StrategyGenome,
    ctx: _SimContext,
    bar: int,
    entry_price: float,
    atr_at_entry: float,
) -> _ExitLevels:
    """Compute all exit levels for a new trade."""
    limits = genome.limit_exits
    levels = _ExitLevels()
    if limits.stop_loss.enabled:
        levels.fixed_stop = compute_stop_loss_level(entry_price, limits.stop_loss, atr_at_entry)
    if limits.take_profit.enabled:
        levels.fixed_target = compute_take_profit_level(entry_price, limits.take_profit, atr_at_entry)
    if limits.trailing_stop.enabled:
        levels.trailing_levels = compute_trailing_stop_levels(
            entry_price,
            limits.trailing_stop,
            ctx.high_arr[bar:],
            ctx.atr_arr[bar:],
        )
    if limits.chandelier.enabled:
        levels.chandelier_levels = compute_chandelier_levels(
            limits.chandelier,
            ctx.high_arr[bar:],
            ctx.atr_arr[bar:],
            0,
        )
    if limits.breakeven.enabled:
        levels.breakeven_levels = compute_breakeven_level(
            entry_price,
            limits.breakeven,
            ctx.high_arr[bar:],
            atr_at_entry,
        )
    levels.time_exit_signals = generate_time_exit_signal(genome.time_exits, bar, ctx.df)
    return levels


def _simulate_trades(
    genome: StrategyGenome,
    ctx: _SimContext,
) -> list[TradeRecord]:
    """Walk through bars and simulate trades one at a time.

    Args:
        genome: Strategy genome for exit configuration.
        ctx: Precomputed simulation context.

    Returns:
        List of completed trade records.
    """
    df = ctx.df
    n_bars = len(df)
    max_trades = ctx.config.max_trades_per_stock
    max_hold = ctx.config.max_holding_days
    trades: list[TradeRecord] = []

    in_position = False
    entry_bar = 0
    entry_price = 0.0
    levels = _ExitLevels()

    for bar in range(n_bars):
        if not in_position:
            if ctx.entry_arr[bar] and not np.isnan(ctx.atr_arr[bar]):
                in_position = True
                entry_bar = bar
                entry_price = ctx.open_arr[bar]
                levels = _compute_exit_levels(genome, ctx, bar, entry_price, ctx.atr_arr[bar])
        else:
            # Hard cap on holding period
            if max_hold > 0 and (bar - entry_bar) >= max_hold:
                exit_price = ctx.close_arr[bar]
                trade = _build_trade_record(entry_bar, entry_price, bar, exit_price, "max_holding", df, ctx.config)
                trades.append(trade)
                in_position = False
                if len(trades) >= max_trades:
                    break
                continue

            exit_result = _check_all_exits(
                bar=bar,
                entry_bar=entry_bar,
                entry_price=entry_price,
                ctx=ctx,
                fixed_stop=levels.fixed_stop,
                fixed_target=levels.fixed_target,
                trailing_levels=levels.trailing_levels,
                chandelier_levels=levels.chandelier_levels,
                breakeven_levels=levels.breakeven_levels,
                time_exit_signals=levels.time_exit_signals,
                genome=genome,
            )
            if exit_result is not None:
                exit_type, exit_price = exit_result
                trade = _build_trade_record(entry_bar, entry_price, bar, exit_price, exit_type, df, ctx.config)
                trades.append(trade)
                in_position = False
                if len(trades) >= max_trades:
                    break

    # Close position at end of data if still open
    if in_position and len(trades) < max_trades:
        exit_price = ctx.close_arr[-1]
        trade = _build_trade_record(
            entry_bar,
            entry_price,
            n_bars - 1,
            exit_price,
            "end_of_data",
            df,
            ctx.config,
        )
        trades.append(trade)

    return trades


def _check_all_exits(  # noqa: PLR0913
    bar: int,
    entry_bar: int,
    entry_price: float,
    ctx: _SimContext,
    fixed_stop: float | None,
    fixed_target: float | None,
    trailing_levels: npt.NDArray[np.float64] | None,
    chandelier_levels: npt.NDArray[np.float64] | None,
    breakeven_levels: npt.NDArray[np.float64] | None,
    time_exit_signals: npt.NDArray[np.bool_] | None,
    genome: StrategyGenome,
) -> tuple[str, float] | None:
    """Check all exit types in priority order for the current bar.

    Args:
        bar: Current bar iloc index.
        entry_bar: Entry bar iloc index.
        entry_price: Entry price.
        ctx: Simulation context.
        fixed_stop: Fixed stop-loss level or None.
        fixed_target: Fixed take-profit level or None.
        trailing_levels: Trailing stop levels array or None.
        chandelier_levels: Chandelier levels array or None.
        breakeven_levels: Breakeven levels array or None.
        time_exit_signals: Time exit signals array or None.
        genome: Strategy genome.

    Returns:
        Tuple of (exit_type, exit_price) or None if no exit triggered.
    """
    open_price = ctx.open_arr[bar]

    # Priority 1: Time exits at open
    if time_exit_signals is not None and time_exit_signals[bar]:
        return ("time", float(open_price))

    # Priority 2: Limit exits intraday
    limit_result = _check_limit_exits(
        bar=bar,
        entry_bar=entry_bar,
        ctx=ctx,
        fixed_stop=fixed_stop,
        fixed_target=fixed_target,
        trailing_levels=trailing_levels,
        chandelier_levels=chandelier_levels,
        breakeven_levels=breakeven_levels,
    )
    if limit_result is not None:
        return limit_result

    # Priority 3: Indicator exits at open (shifted)
    indicator_exit = _check_indicator_exits(bar, ctx, genome)
    if indicator_exit:
        return ("indicator", float(open_price))

    return None


def _check_limit_exits(  # noqa: PLR0913
    bar: int,
    entry_bar: int,
    ctx: _SimContext,
    fixed_stop: float | None,
    fixed_target: float | None,
    trailing_levels: npt.NDArray[np.float64] | None,
    chandelier_levels: npt.NDArray[np.float64] | None,
    breakeven_levels: npt.NDArray[np.float64] | None,
) -> tuple[str, float] | None:
    """Check all limit-based exits and return the binding one.

    Merges all stop-type levels into an effective stop (max of all active
    levels), then delegates to check_limit_exit.

    Args:
        bar: Current bar iloc index.
        entry_bar: Entry bar iloc index.
        ctx: Simulation context with numpy arrays.
        fixed_stop: Fixed stop-loss level or None.
        fixed_target: Fixed take-profit level or None.
        trailing_levels: Trailing stop levels array or None.
        chandelier_levels: Chandelier levels array or None.
        breakeven_levels: Breakeven levels array or None.

    Returns:
        Tuple of (exit_type, exit_price) or None.
    """
    # Relative index for per-trade arrays (sliced from entry_bar)
    rel = bar - entry_bar

    # Collect all stop levels and track which is binding
    stop_levels: list[tuple[str, float]] = []
    if fixed_stop is not None:
        stop_levels.append(("stop_loss", fixed_stop))
    if trailing_levels is not None and rel < len(trailing_levels):
        val = trailing_levels[rel]
        if not np.isnan(val):
            stop_levels.append(("trailing_stop", float(val)))
    if chandelier_levels is not None and rel < len(chandelier_levels):
        val = chandelier_levels[rel]
        if not np.isnan(val):
            stop_levels.append(("chandelier", float(val)))
    if breakeven_levels is not None and rel < len(breakeven_levels):
        val = breakeven_levels[rel]
        if not np.isnan(val):
            stop_levels.append(("breakeven", float(val)))

    # Effective stop = max of all active stop levels (tightest for longs)
    effective_stop: float | None = None
    binding_stop_type = "stop_loss"
    if stop_levels:
        binding = max(stop_levels, key=lambda x: x[1])
        binding_stop_type = binding[0]
        effective_stop = binding[1]

    high = float(ctx.high_arr[bar])
    low = float(ctx.low_arr[bar])
    open_price = float(ctx.open_arr[bar])

    exit_type, exit_price = check_limit_exit(high, low, open_price, effective_stop, fixed_target)

    if exit_type is None:
        return None

    # Map generic "stop_loss" from check_limit_exit to the binding type
    if exit_type == "stop_loss":
        return (binding_stop_type, exit_price)  # type: ignore[return-value]
    return ("take_profit", exit_price)  # type: ignore[return-value]


def _check_indicator_exits(
    bar: int,
    ctx: _SimContext,
    genome: StrategyGenome,
) -> bool:
    """Check indicator exits including opposite_entry logic.

    Args:
        bar: Current bar iloc index.
        ctx: Simulation context.
        genome: Strategy genome.

    Returns:
        True if an indicator exit is triggered.
    """
    if ctx.ind_exit_arr[bar]:
        return True

    # Check opposite_entry: any exit gene with opposite_entry=True triggers
    # when entry conditions no longer hold at close of previous bar
    has_opposite = any(g.opposite_entry for g in genome.indicator_exits if g.enabled)
    if has_opposite and bar >= 1 and not ctx.raw_entry_arr[bar - 1]:
        return True

    return False


# ---------------------------------------------------------------------------
# Trade record builder
# ---------------------------------------------------------------------------


def _build_trade_record(
    entry_bar: int,
    entry_price: float,
    exit_bar: int,
    exit_price: float,
    exit_type: str,
    df: pd.DataFrame,
    config: EvaluatorConfig,
) -> TradeRecord:
    """Build a TradeRecord from trade parameters.

    Args:
        entry_bar: Entry bar iloc index.
        entry_price: Entry execution price.
        exit_bar: Exit bar iloc index.
        exit_price: Exit execution price.
        exit_type: Exit type string.
        df: OHLCV DataFrame.
        config: Evaluator configuration.

    Returns:
        Completed TradeRecord.
    """
    cost_pct = config.round_trip_cost_pct / 100.0
    gross_return_pct = (exit_price - entry_price) / entry_price
    net_return_pct = gross_return_pct - cost_pct
    net_profit = config.notional_per_trade * net_return_pct

    return TradeRecord(
        entry_bar=entry_bar,
        entry_date=str(df.index[entry_bar].date()),
        entry_price=entry_price,
        exit_bar=exit_bar,
        exit_date=str(df.index[exit_bar].date()),
        exit_price=exit_price,
        exit_type=exit_type,
        gross_return_pct=gross_return_pct,
        cost_pct=cost_pct,
        net_return_pct=net_return_pct,
        net_profit=net_profit,
        is_win=net_return_pct > 0,
        holding_days=exit_bar - entry_bar,
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    trades: list[TradeRecord],
    config: EvaluatorConfig,
) -> BacktestMetrics:
    """Compute aggregate metrics from a list of trade records.

    Args:
        trades: List of completed trade records.
        config: Evaluator configuration (for notional size).

    Returns:
        BacktestMetrics with all aggregate statistics.
    """
    if not trades:
        return _zero_metrics()

    num_trades = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = num_wins / num_trades

    net_profit = sum(t.net_profit for t in trades)
    gross_profit = sum(t.net_profit for t in wins)
    gross_loss = sum(t.net_profit for t in losses)

    profit_factor = _safe_ratio(gross_profit, abs(gross_loss))

    avg_win_pct = _mean_of([t.net_return_pct for t in wins])
    avg_loss_pct = _mean_of([t.net_return_pct for t in losses])
    win_loss_ratio = _safe_ratio(avg_win_pct, abs(avg_loss_pct))

    max_dd_pct, max_dd_duration = _compute_drawdown(trades, config)

    returns = [t.net_return_pct for t in trades]
    sharpe = _compute_sharpe(returns, trades)
    sortino = _compute_sortino(returns, trades)
    calmar = _compute_calmar(returns, trades, max_dd_pct)

    avg_holding = sum(t.holding_days for t in trades) / num_trades
    win_streak, loss_streak = _compute_streaks(trades)
    expectancy = win_rate * avg_win_pct + (1 - win_rate) * avg_loss_pct

    return BacktestMetrics(
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        net_profit=net_profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        win_loss_ratio=win_loss_ratio,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        avg_holding_days=avg_holding,
        longest_win_streak=win_streak,
        longest_loss_streak=loss_streak,
        expectancy_per_trade=expectancy,
    )


def _zero_metrics() -> BacktestMetrics:
    """Return metrics with all values at zero."""
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Compute ratio, returning inf if denominator is zero and numerator > 0."""
    if denominator == 0.0:
        return float("inf") if numerator > 0 else 0.0
    return numerator / denominator


def _mean_of(values: list[float]) -> float:
    """Compute mean of a list, returning 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_drawdown(
    trades: list[TradeRecord],
    config: EvaluatorConfig,
) -> tuple[float, int]:
    """Compute max drawdown percentage and duration from trade sequence.

    Args:
        trades: List of trade records.
        config: Evaluator configuration for notional size.

    Returns:
        Tuple of (max_drawdown_pct, max_drawdown_duration).
        Drawdown pct is as a fraction of peak equity (notional-based).
        Duration is in number of trades from peak to recovery.
    """
    if not trades:
        return (0.0, 0)

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    peak_idx = 0
    max_dd_duration = 0

    for i, trade in enumerate(trades):
        cumulative += trade.net_profit
        if cumulative > peak:
            peak = cumulative
            peak_idx = i
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - peak_idx

    peak_equity = config.notional_per_trade + peak
    max_dd_pct = max_dd / peak_equity if peak_equity > 0 else 0.0

    return (max_dd_pct, max_dd_duration)


def _compute_sharpe(
    returns: list[float],
    trades: list[TradeRecord],
) -> float:
    """Compute annualized Sharpe ratio from trade returns.

    Args:
        returns: List of net return percentages per trade.
        trades: Trade records for holding days.

    Returns:
        Annualized Sharpe ratio, 0.0 if fewer than 2 trades.
    """
    if len(returns) < 2:  # noqa: PLR2004
        return 0.0
    avg = sum(returns) / len(returns)
    std = _std(returns)
    if std == 0.0:
        return 0.0
    avg_holding = sum(t.holding_days for t in trades) / len(trades)
    annualization = math.sqrt(252.0 / max(avg_holding, 1.0))
    return (avg / std) * annualization


def _compute_sortino(
    returns: list[float],
    trades: list[TradeRecord],
) -> float:
    """Compute annualized Sortino ratio from trade returns.

    Args:
        returns: List of net return percentages per trade.
        trades: Trade records for holding days.

    Returns:
        Annualized Sortino ratio, 0.0 if fewer than 2 trades.
    """
    if len(returns) < 2:  # noqa: PLR2004
        return 0.0
    avg = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0 if avg <= 0 else float("inf")
    downside_std = _std(downside)
    if downside_std == 0.0:
        return 0.0
    avg_holding = sum(t.holding_days for t in trades) / len(trades)
    annualization = math.sqrt(252.0 / max(avg_holding, 1.0))
    return (avg / downside_std) * annualization


def _compute_calmar(
    returns: list[float],
    trades: list[TradeRecord],
    max_dd_pct: float,
) -> float:
    """Compute Calmar ratio (annualized return / max drawdown).

    Args:
        returns: List of net return percentages per trade.
        trades: Trade records for holding days.
        max_dd_pct: Maximum drawdown as fraction.

    Returns:
        Calmar ratio, 0.0 if max drawdown is zero.
    """
    if max_dd_pct == 0.0 or not trades:
        return 0.0
    total_return = sum(returns)
    avg_holding = sum(t.holding_days for t in trades) / len(trades)
    total_holding = avg_holding * len(trades)
    annualized_return = total_return * (252.0 / max(total_holding, 1.0))
    return annualized_return / max_dd_pct


def _compute_streaks(trades: list[TradeRecord]) -> tuple[int, int]:
    """Compute longest win and loss streaks.

    Args:
        trades: List of trade records.

    Returns:
        Tuple of (longest_win_streak, longest_loss_streak).
    """
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0

    for trade in trades:
        if trade.is_win:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)

    return (max_win, max_loss)


def _std(values: list[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:  # noqa: PLR2004
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
