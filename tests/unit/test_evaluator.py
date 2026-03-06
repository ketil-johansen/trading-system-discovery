"""Unit tests for the backtest evaluator."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tsd.indicators.base import IndicatorResult
from tsd.strategy.evaluator import (
    BacktestResult,
    EvaluatorConfig,
    TradeRecord,
    _compute_metrics,
    _compute_streaks,
    run_backtest,
)
from tsd.strategy.genome import (
    BreakevenConfig,
    ChandelierConfig,
    IndicatorExitGene,
    LimitExitGene,
    StopLossConfig,
    StrategyGenome,
    TakeProfitConfig,
    TimeExitGene,
    TrailingStopConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 30, start_price: float = 100.0) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame with a steady uptrend.

    Prices rise by 1.0 per bar. Open = prev Close, High = Close + 0.5,
    Low = Open - 0.5.
    """
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = start_price + np.arange(n, dtype=float)
    open_ = np.empty(n)
    open_[0] = start_price - 0.5
    open_[1:] = close[:-1]
    high = close + 0.5
    low = open_ - 0.5
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _disabled_limit_exits() -> LimitExitGene:
    """Return limit exits with everything disabled."""
    return LimitExitGene(
        stop_loss=StopLossConfig(enabled=False, mode="percent", percent=5.0, atr_multiple=2.0),
        take_profit=TakeProfitConfig(enabled=False, mode="percent", percent=5.0, atr_multiple=2.0),
        trailing_stop=TrailingStopConfig(
            enabled=False, mode="percent", percent=3.0, atr_multiple=2.0, activation_percent=1.0
        ),
        chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
        breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
    )


def _disabled_time_exits() -> TimeExitGene:
    """Return time exits with everything disabled."""
    return TimeExitGene(
        max_days_enabled=False,
        max_days=10,
        weekday_exit_enabled=False,
        weekday=4,
        eow_enabled=False,
        eom_enabled=False,
        stagnation_enabled=False,
        stagnation_days=5,
        stagnation_threshold=1.0,
    )


def _make_genome(
    entry_signals: bool = True,
    limit_exits: LimitExitGene | None = None,
    indicator_exits: tuple[IndicatorExitGene, ...] = (),
    time_exits: TimeExitGene | None = None,
) -> StrategyGenome:
    """Build a minimal genome for testing.

    The actual entry signal generation is mocked, so entry_indicators
    content doesn't matter for most tests.
    """
    return StrategyGenome(
        entry_indicators=(),
        combination_logic="AND",
        limit_exits=limit_exits or _disabled_limit_exits(),
        indicator_exits=indicator_exits,
        time_exits=time_exits or _disabled_time_exits(),
        filters=(),
    )


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """30-bar uptrend DataFrame."""
    return _make_df(30)


@pytest.fixture
def config() -> EvaluatorConfig:
    """Default evaluator config."""
    return EvaluatorConfig()


def _mock_atr(df: pd.DataFrame, atr_period: int = 14) -> IndicatorResult:
    """Return constant ATR of 2.0, with NaN for first atr_period-1 bars."""
    atr = pd.Series(2.0, index=df.index)
    atr.iloc[: atr_period - 1] = np.nan
    return IndicatorResult(name="atr", values={"atr": atr}, params={"period": atr_period})


def _run_with_mocks(
    genome: StrategyGenome,
    df: pd.DataFrame,
    entry_mask: pd.Series,
    indicator_exit_mask: pd.Series | None = None,
    config: EvaluatorConfig | None = None,
) -> BacktestResult:
    """Run backtest with mocked signal generation and ATR.

    Args:
        genome: Strategy genome.
        df: OHLCV DataFrame.
        entry_mask: Boolean Series for raw entry signals.
        indicator_exit_mask: Boolean Series for indicator exit signals.
        config: Optional evaluator config.
    """
    if indicator_exit_mask is None:
        indicator_exit_mask = pd.Series(False, index=df.index, dtype=bool)

    cfg = config or EvaluatorConfig()

    with (
        patch("tsd.strategy.evaluator.compute_indicator") as mock_ci,
        patch("tsd.strategy.evaluator.generate_entry_signals", return_value=entry_mask),
        patch("tsd.strategy.evaluator.generate_indicator_exit_signals", return_value=indicator_exit_mask),
    ):
        mock_ci.return_value = _mock_atr(df, cfg.atr_period)
        return run_backtest(genome, df, {}, cfg)


# ---------------------------------------------------------------------------
# TestRunBacktest
# ---------------------------------------------------------------------------


class TestRunBacktest:
    """Tests for the run_backtest public API."""

    def test_no_entries_returns_zero_trades(self, simple_df: pd.DataFrame) -> None:
        """All entry slots disabled produces no trades."""
        genome = _make_genome()
        entries = pd.Series(False, index=simple_df.index, dtype=bool)
        result = _run_with_mocks(genome, simple_df, entries)
        assert result.metrics.num_trades == 0

    def test_single_trade_stop_loss(self, simple_df: pd.DataFrame) -> None:
        """Entry fires, fixed SL hit, verify exit_type and price."""
        limits = _disabled_limit_exits()
        # Enable SL at 3% below entry
        limits = LimitExitGene(
            stop_loss=StopLossConfig(enabled=True, mode="percent", percent=3.0, atr_multiple=2.0),
            take_profit=limits.take_profit,
            trailing_stop=limits.trailing_stop,
            chandelier=limits.chandelier,
            breakeven=limits.breakeven,
        )
        genome = _make_genome(limit_exits=limits)

        # Build df where price drops sharply after entry
        n = 30
        dates = pd.bdate_range("2020-01-02", periods=n)
        close = np.array([100.0] * 15 + [90.0] * 15)
        open_ = close.copy()
        high = close + 1.0
        low = close - 5.0  # Low enough to trigger 3% SL
        volume = np.full(n, 1e6)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

        # Signal on bar 14 (close), execute entry at bar 15 (open)
        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True

        result = _run_with_mocks(genome, df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.exit_type == "stop_loss"
        # Entry at Open of bar 15 = 90.0, SL at 90*(1-0.03) = 87.3
        # Low of bar 16 = 85.0 which is < 87.3, so SL hit
        assert trade.exit_price == pytest.approx(87.3, rel=0.01)

    def test_single_trade_take_profit(self, simple_df: pd.DataFrame) -> None:
        """Entry fires, TP hit in uptrend."""
        limits = LimitExitGene(
            stop_loss=StopLossConfig(enabled=False, mode="percent", percent=5.0, atr_multiple=2.0),
            take_profit=TakeProfitConfig(enabled=True, mode="percent", percent=2.0, atr_multiple=2.0),
            trailing_stop=TrailingStopConfig(
                enabled=False, mode="percent", percent=3.0, atr_multiple=2.0, activation_percent=1.0
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
        )
        genome = _make_genome(limit_exits=limits)

        # Entry signal at bar 14 (after ATR warmup), execute at bar 15
        entries = pd.Series(False, index=simple_df.index, dtype=bool)
        entries.iloc[14] = True

        result = _run_with_mocks(genome, simple_df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.exit_type == "take_profit"

    def test_entry_at_next_open(self, simple_df: pd.DataFrame) -> None:
        """Entry price equals Open of bar after signal."""
        genome = _make_genome()
        entries = pd.Series(False, index=simple_df.index, dtype=bool)
        entries.iloc[14] = True  # Signal at close of bar 14

        result = _run_with_mocks(genome, simple_df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        # Shifted entry: execute at bar 15's Open
        expected_entry_price = float(simple_df["Open"].iloc[15])
        assert trade.entry_price == pytest.approx(expected_entry_price)
        assert trade.entry_bar == 15

    def test_cost_deduction(self, simple_df: pd.DataFrame) -> None:
        """Verify cost_pct matches config."""
        genome = _make_genome()
        entries = pd.Series(False, index=simple_df.index, dtype=bool)
        entries.iloc[14] = True

        cfg = EvaluatorConfig(round_trip_cost_pct=0.20)
        result = _run_with_mocks(genome, simple_df, entries, config=cfg)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.cost_pct == pytest.approx(0.002)

    def test_end_of_data_closes_position(self, simple_df: pd.DataFrame) -> None:
        """Position open at last bar gets closed with end_of_data."""
        genome = _make_genome()
        # Entry near end so no exit triggers before data ends
        entries = pd.Series(False, index=simple_df.index, dtype=bool)
        entries.iloc[27] = True  # Signal at bar 27, enter at bar 28

        result = _run_with_mocks(genome, simple_df, entries)
        assert result.metrics.num_trades == 1
        trade = result.trades[0]
        assert trade.exit_type == "end_of_data"
        assert trade.exit_price == pytest.approx(float(simple_df["Close"].iloc[-1]))

    def test_one_position_at_a_time(self, simple_df: pd.DataFrame) -> None:
        """Second entry signal ignored while in position."""
        genome = _make_genome()
        entries = pd.Series(False, index=simple_df.index, dtype=bool)
        entries.iloc[14] = True
        entries.iloc[18] = True  # Should be ignored

        result = _run_with_mocks(genome, simple_df, entries)
        # Only one trade (end_of_data close)
        assert result.metrics.num_trades == 1

    def test_skip_entry_when_atr_nan(self, simple_df: pd.DataFrame) -> None:
        """Entry signal before ATR warmup is skipped."""
        genome = _make_genome()
        entries = pd.Series(False, index=simple_df.index, dtype=bool)
        entries.iloc[5] = True  # Bar 6 has NaN ATR (warmup = 14 bars)

        result = _run_with_mocks(genome, simple_df, entries)
        assert result.metrics.num_trades == 0


# ---------------------------------------------------------------------------
# TestLimitExitResolution
# ---------------------------------------------------------------------------


class TestLimitExitResolution:
    """Tests for limit exit resolution behavior."""

    def test_both_hit_stop_wins(self) -> None:
        """When both SL and TP hit same bar, stop wins (conservative)."""
        n = 20
        dates = pd.bdate_range("2020-01-02", periods=n)
        # Wide bar that triggers both levels
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 110.0)  # High enough for TP
        low = np.full(n, 90.0)  # Low enough for SL
        volume = np.full(n, 1e6)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

        limits = LimitExitGene(
            stop_loss=StopLossConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=2.0),
            take_profit=TakeProfitConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=2.0),
            trailing_stop=TrailingStopConfig(
                enabled=False, mode="percent", percent=3.0, atr_multiple=2.0, activation_percent=1.0
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
        )
        genome = _make_genome(limit_exits=limits)

        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True  # After ATR warmup

        result = _run_with_mocks(genome, df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        # Conservative: stop wins when both hit
        assert trade.exit_type == "stop_loss"

    def test_gap_up_target_wins(self) -> None:
        """When open gaps above target, TP wins at open price."""
        n = 20
        dates = pd.bdate_range("2020-01-02", periods=n)
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        volume = np.full(n, 1e6)
        # Entry at bar 15, then bar 16 gaps up above TP
        open_[16] = 120.0  # Gap above TP
        high[16] = 120.0
        low[16] = 90.0  # Also below SL
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

        limits = LimitExitGene(
            stop_loss=StopLossConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=2.0),
            take_profit=TakeProfitConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=2.0),
            trailing_stop=TrailingStopConfig(
                enabled=False, mode="percent", percent=3.0, atr_multiple=2.0, activation_percent=1.0
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
        )
        genome = _make_genome(limit_exits=limits)

        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True

        result = _run_with_mocks(genome, df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.exit_type == "take_profit"
        # Open exceeds target, so exit at open price
        assert trade.exit_price == pytest.approx(120.0)

    def test_trailing_stop_tightens(self) -> None:
        """Trailing stop becomes tighter than fixed SL over time."""
        n = 30
        dates = pd.bdate_range("2020-01-02", periods=n)
        # Price rises then drops
        close = np.array(
            [100.0] * 14
            + [100.0, 102.0, 105.0, 108.0, 110.0, 112.0]
            + [112.0, 112.0, 112.0, 105.0, 100.0, 95.0, 90.0, 85.0, 80.0, 75.0]
        )
        open_ = close.copy()
        open_[0] = 99.5
        for i in range(1, n):
            open_[i] = close[i - 1]
        high = close + 1.0
        low = close - 1.0
        volume = np.full(n, 1e6)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

        limits = LimitExitGene(
            stop_loss=StopLossConfig(enabled=True, mode="percent", percent=20.0, atr_multiple=2.0),
            take_profit=TakeProfitConfig(enabled=False, mode="percent", percent=50.0, atr_multiple=2.0),
            trailing_stop=TrailingStopConfig(
                enabled=True, mode="percent", percent=3.0, atr_multiple=2.0, activation_percent=1.0
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
        )
        genome = _make_genome(limit_exits=limits)

        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True

        result = _run_with_mocks(genome, df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        # Trailing stop should trigger before the wide fixed SL
        assert trade.exit_type == "trailing_stop"


# ---------------------------------------------------------------------------
# TestExitPriority
# ---------------------------------------------------------------------------


class TestExitPriority:
    """Tests for exit priority ordering."""

    def test_time_exit_at_open_price(self) -> None:
        """Time exit uses Open price of the exit bar."""
        time_exits = TimeExitGene(
            max_days_enabled=True,
            max_days=3,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=False,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        genome = _make_genome(time_exits=time_exits)
        df = _make_df(30)

        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True  # Signal at bar 14, enter bar 15

        result = _run_with_mocks(genome, df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.exit_type == "time"
        # max_days=3 from entry_bar=15 → exit at bar 18
        assert trade.exit_bar == 18
        assert trade.exit_price == pytest.approx(float(df["Open"].iloc[18]))

    def test_indicator_exit_at_open_price(self) -> None:
        """Indicator exit uses Open price (shifted to next bar)."""
        genome = _make_genome()
        df = _make_df(30)

        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True

        # Indicator exit signal at bar 18 (close) → shifted to bar 19
        ind_exits = pd.Series(False, index=df.index, dtype=bool)
        ind_exits.iloc[18] = True

        result = _run_with_mocks(genome, df, entries, indicator_exit_mask=ind_exits)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.exit_type == "indicator"
        assert trade.exit_bar == 19
        assert trade.exit_price == pytest.approx(float(df["Open"].iloc[19]))

    def test_limit_exit_at_limit_price(self) -> None:
        """Limit exit uses the computed level price, not Open."""
        n = 25
        dates = pd.bdate_range("2020-01-02", periods=n)
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        # Bar 16: drop below SL
        low[16] = 90.0
        volume = np.full(n, 1e6)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

        limits = LimitExitGene(
            stop_loss=StopLossConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=2.0),
            take_profit=TakeProfitConfig(enabled=False, mode="percent", percent=50.0, atr_multiple=2.0),
            trailing_stop=TrailingStopConfig(
                enabled=False, mode="percent", percent=3.0, atr_multiple=2.0, activation_percent=1.0
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
        )
        genome = _make_genome(limit_exits=limits)

        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True

        result = _run_with_mocks(genome, df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.exit_type == "stop_loss"
        # SL at 100*(1-0.05) = 95.0
        assert trade.exit_price == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# TestComputeMetrics
# ---------------------------------------------------------------------------


def _make_trade(
    net_return_pct: float,
    holding_days: int = 5,
    entry_bar: int = 0,
    notional: float = 10_000.0,
) -> TradeRecord:
    """Build a TradeRecord with given return for metric computation tests."""
    cost_pct = 0.002
    gross_return_pct = net_return_pct + cost_pct
    entry_price = 100.0
    exit_price = entry_price * (1 + gross_return_pct)
    return TradeRecord(
        entry_bar=entry_bar,
        entry_date="2020-01-02",
        entry_price=entry_price,
        exit_bar=entry_bar + holding_days,
        exit_date="2020-01-09",
        exit_price=exit_price,
        exit_type="stop_loss",
        gross_return_pct=gross_return_pct,
        cost_pct=cost_pct,
        net_return_pct=net_return_pct,
        net_profit=notional * net_return_pct,
        is_win=net_return_pct > 0,
        holding_days=holding_days,
    )


class TestComputeMetrics:
    """Tests for _compute_metrics."""

    def test_zero_trades(self) -> None:
        """Empty trade list returns all-zero metrics."""
        metrics = _compute_metrics([], EvaluatorConfig())
        assert metrics.num_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.net_profit == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_all_wins(self) -> None:
        """All winning trades give win_rate=1.0 and profit_factor=inf."""
        trades = [_make_trade(0.05), _make_trade(0.03), _make_trade(0.02)]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        assert metrics.win_rate == pytest.approx(1.0)
        assert metrics.num_wins == 3
        assert metrics.num_losses == 0
        assert metrics.profit_factor == float("inf")

    def test_all_losses(self) -> None:
        """All losing trades give win_rate=0.0."""
        trades = [_make_trade(-0.05), _make_trade(-0.03), _make_trade(-0.02)]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        assert metrics.win_rate == pytest.approx(0.0)
        assert metrics.num_losses == 3

    def test_mixed_trades(self) -> None:
        """Verify key metrics with known win/loss outcomes."""
        trades = [
            _make_trade(0.05),  # win: $500
            _make_trade(-0.03),  # loss: -$300
            _make_trade(0.04),  # win: $400
            _make_trade(-0.02),  # loss: -$200
            _make_trade(0.06),  # win: $600
        ]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        assert metrics.num_trades == 5
        assert metrics.num_wins == 3
        assert metrics.num_losses == 2
        assert metrics.win_rate == pytest.approx(0.6)
        assert metrics.net_profit == pytest.approx(1000.0)
        assert metrics.gross_profit == pytest.approx(1500.0)
        assert metrics.gross_loss == pytest.approx(-500.0)
        assert metrics.profit_factor == pytest.approx(3.0)

    def test_streaks(self) -> None:
        """Specific win/loss sequence produces correct streaks."""
        trades = [
            _make_trade(0.01),  # W
            _make_trade(0.02),  # W
            _make_trade(0.01),  # W
            _make_trade(-0.01),  # L
            _make_trade(-0.02),  # L
            _make_trade(0.03),  # W
        ]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        assert metrics.longest_win_streak == 3
        assert metrics.longest_loss_streak == 2

    def test_max_drawdown(self) -> None:
        """Known cumulative P&L produces correct drawdown."""
        # Cumulative: +500, +200, +600, +100, +700
        trades = [
            _make_trade(0.05),  # cum: 500
            _make_trade(-0.03),  # cum: 200
            _make_trade(0.04),  # cum: 600
            _make_trade(-0.05),  # cum: 100
            _make_trade(0.06),  # cum: 700
        ]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        # Max drawdown: peak 600 → 100 = 500 drop
        # Peak equity = 10000 + 600 = 10600
        # DD pct = 500 / 10600
        assert metrics.max_drawdown_pct == pytest.approx(500.0 / 10600.0, rel=0.01)
        assert metrics.max_drawdown_duration == 1  # 1 trade from peak to trough

    def test_expectancy_formula(self) -> None:
        """Expectancy = win_rate * avg_win + (1-win_rate) * avg_loss."""
        trades = [
            _make_trade(0.04),
            _make_trade(-0.02),
            _make_trade(0.06),
            _make_trade(-0.03),
        ]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        expected = metrics.win_rate * metrics.avg_win_pct + (1 - metrics.win_rate) * metrics.avg_loss_pct
        assert metrics.expectancy_per_trade == pytest.approx(expected)

    def test_avg_holding_days(self) -> None:
        """Average holding days computed correctly."""
        trades = [
            _make_trade(0.01, holding_days=3),
            _make_trade(0.02, holding_days=7),
            _make_trade(-0.01, holding_days=5),
        ]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        assert metrics.avg_holding_days == pytest.approx(5.0)

    def test_sharpe_with_one_trade(self) -> None:
        """Sharpe is 0.0 with fewer than 2 trades."""
        trades = [_make_trade(0.05)]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        assert metrics.sharpe_ratio == 0.0

    def test_win_loss_ratio(self) -> None:
        """Win/loss ratio = avg_win / |avg_loss|."""
        trades = [
            _make_trade(0.06),  # win
            _make_trade(-0.02),  # loss
        ]
        metrics = _compute_metrics(trades, EvaluatorConfig())
        assert metrics.win_loss_ratio == pytest.approx(0.06 / 0.02)


# ---------------------------------------------------------------------------
# TestStreaks (standalone)
# ---------------------------------------------------------------------------


class TestStreaks:
    """Tests for _compute_streaks helper."""

    def test_empty(self) -> None:
        """No trades gives zero streaks."""
        assert _compute_streaks([]) == (0, 0)

    def test_all_wins(self) -> None:
        """All wins, no losses."""
        trades = [_make_trade(0.01) for _ in range(5)]
        assert _compute_streaks(trades) == (5, 0)

    def test_alternating(self) -> None:
        """Alternating W/L gives streak of 1."""
        trades = [_make_trade(0.01), _make_trade(-0.01)] * 3
        assert _compute_streaks(trades) == (1, 1)


# ---------------------------------------------------------------------------
# TestOppositeEntry
# ---------------------------------------------------------------------------


class TestOppositeEntry:
    """Tests for opposite_entry exit logic."""

    def test_opposite_entry_triggers_exit(self) -> None:
        """Exit when entry conditions no longer hold."""
        ind_exit = IndicatorExitGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=70.0,
            opposite_entry=True,
        )
        genome = _make_genome(indicator_exits=(ind_exit,))
        df = _make_df(30)

        # Entry signal True at bars 14-18, then False at 19+
        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True
        entries.iloc[15] = True
        entries.iloc[16] = True
        entries.iloc[17] = True
        entries.iloc[18] = True
        # Bar 19: entry condition False → opposite_entry checks bar 18's raw signal
        # But raw_entry_signals[18] = True, so no exit at bar 19
        # raw_entry_signals[19] = False → exit at bar 20

        result = _run_with_mocks(genome, df, entries)
        assert result.metrics.num_trades >= 1
        trade = result.trades[0]
        assert trade.exit_type == "indicator"
        # Exit at bar 20 (first bar after entry where raw_entry[bar-1]=False)
        assert trade.exit_bar == 20

    def test_max_holding_days_force_exit(self) -> None:
        """Position force-exited after max_holding_days bars."""
        genome = _make_genome()
        df = _make_df(80)
        entries = pd.Series(False, index=df.index, dtype=bool)
        entries.iloc[14] = True  # Signal at bar 14, position opens bar 15

        config = EvaluatorConfig(max_holding_days=10)
        result = _run_with_mocks(genome, df, entries, config=config)
        assert result.metrics.num_trades == 1
        trade = result.trades[0]
        assert trade.exit_type == "max_holding"
        assert trade.exit_bar == 25  # entry_bar=15 + 10 = bar 25
