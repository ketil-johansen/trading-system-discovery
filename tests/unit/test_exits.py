"""Unit tests for exit computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsd.strategy.exits import (
    compute_breakeven_level,
    compute_chandelier_levels,
    compute_stop_loss_level,
    compute_take_profit_level,
    compute_trailing_stop_levels,
    generate_indicator_exit_signals,
    generate_time_exit_signal,
)
from tsd.strategy.genome import (
    BreakevenConfig,
    ChandelierConfig,
    IndicatorExitGene,
    StopLossConfig,
    TakeProfitConfig,
    TimeExitGene,
    TrailingStopConfig,
    load_strategy_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with 200 rows."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def highs() -> np.ndarray:
    """Monotonically increasing highs for trailing stop tests."""
    return np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0])


@pytest.fixture
def flat_atr() -> np.ndarray:
    """Constant ATR for predictable tests."""
    return np.array([2.0] * 8)


# ---------------------------------------------------------------------------
# Stop loss
# ---------------------------------------------------------------------------


class TestStopLoss:
    """Tests for compute_stop_loss_level."""

    def test_percent_mode(self) -> None:
        cfg = StopLossConfig(enabled=True, mode="percent", percent=3.0, atr_multiple=2.0)
        level = compute_stop_loss_level(100.0, cfg, atr_at_entry=2.0)
        assert level == pytest.approx(97.0)

    def test_atr_mode(self) -> None:
        cfg = StopLossConfig(enabled=True, mode="atr", percent=3.0, atr_multiple=2.0)
        level = compute_stop_loss_level(100.0, cfg, atr_at_entry=2.0)
        assert level == pytest.approx(96.0)

    def test_stop_below_entry(self) -> None:
        cfg = StopLossConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=2.0)
        level = compute_stop_loss_level(100.0, cfg, atr_at_entry=2.0)
        assert level < 100.0


# ---------------------------------------------------------------------------
# Take profit
# ---------------------------------------------------------------------------


class TestTakeProfit:
    """Tests for compute_take_profit_level."""

    def test_percent_mode(self) -> None:
        cfg = TakeProfitConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=3.0)
        level = compute_take_profit_level(100.0, cfg, atr_at_entry=2.0)
        assert level == pytest.approx(105.0)

    def test_atr_mode(self) -> None:
        cfg = TakeProfitConfig(enabled=True, mode="atr", percent=5.0, atr_multiple=3.0)
        level = compute_take_profit_level(100.0, cfg, atr_at_entry=2.0)
        assert level == pytest.approx(106.0)

    def test_target_above_entry(self) -> None:
        cfg = TakeProfitConfig(enabled=True, mode="percent", percent=5.0, atr_multiple=3.0)
        level = compute_take_profit_level(100.0, cfg, atr_at_entry=2.0)
        assert level > 100.0


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------


class TestTrailingStop:
    """Tests for compute_trailing_stop_levels."""

    def test_nan_before_activation(self, highs: np.ndarray, flat_atr: np.ndarray) -> None:
        cfg = TrailingStopConfig(
            enabled=True,
            mode="percent",
            percent=2.0,
            atr_multiple=2.0,
            activation_percent=5.0,
        )
        levels = compute_trailing_stop_levels(100.0, cfg, highs, flat_atr)
        # Activation at 105.0, first hit at index 5
        assert np.isnan(levels[0])
        assert np.isnan(levels[4])

    def test_activates_after_threshold(self, highs: np.ndarray, flat_atr: np.ndarray) -> None:
        cfg = TrailingStopConfig(
            enabled=True,
            mode="percent",
            percent=2.0,
            atr_multiple=2.0,
            activation_percent=3.0,
        )
        levels = compute_trailing_stop_levels(100.0, cfg, highs, flat_atr)
        # Activation at 103.0, first activated at index 3
        assert not np.isnan(levels[3])

    def test_monotonically_non_decreasing(self, highs: np.ndarray, flat_atr: np.ndarray) -> None:
        cfg = TrailingStopConfig(
            enabled=True,
            mode="atr",
            percent=2.0,
            atr_multiple=1.0,
            activation_percent=1.0,
        )
        levels = compute_trailing_stop_levels(100.0, cfg, highs, flat_atr)
        active = levels[~np.isnan(levels)]
        if len(active) > 1:
            diffs = np.diff(active)
            assert (diffs >= 0).all(), "Trailing stop must be monotonically non-decreasing"

    def test_atr_mode(self, highs: np.ndarray, flat_atr: np.ndarray) -> None:
        cfg = TrailingStopConfig(
            enabled=True,
            mode="atr",
            percent=2.0,
            atr_multiple=1.5,
            activation_percent=1.0,
        )
        levels = compute_trailing_stop_levels(100.0, cfg, highs, flat_atr)
        # After activation, stop = max_high - 1.5 * 2.0 = max_high - 3.0
        active = levels[~np.isnan(levels)]
        assert len(active) > 0


# ---------------------------------------------------------------------------
# Chandelier
# ---------------------------------------------------------------------------


class TestChandelier:
    """Tests for compute_chandelier_levels."""

    def test_nan_before_entry(self, highs: np.ndarray, flat_atr: np.ndarray) -> None:
        cfg = ChandelierConfig(enabled=True, atr_multiple=2.0)
        levels = compute_chandelier_levels(cfg, highs, flat_atr, entry_bar=3)
        assert np.isnan(levels[0])
        assert np.isnan(levels[2])
        assert not np.isnan(levels[3])

    def test_uses_highest_high_since_entry(self, flat_atr: np.ndarray) -> None:
        highs = np.array([100.0, 101.0, 105.0, 103.0, 102.0])
        atr = np.array([2.0] * 5)
        cfg = ChandelierConfig(enabled=True, atr_multiple=1.0)
        levels = compute_chandelier_levels(cfg, highs, atr, entry_bar=0)
        # At index 2, highest high is 105, so level = 105 - 1*2 = 103
        assert levels[2] == pytest.approx(103.0)
        # At index 4, highest high still 105, level = 105 - 2 = 103
        assert levels[4] == pytest.approx(103.0)

    def test_values_after_entry(self, highs: np.ndarray, flat_atr: np.ndarray) -> None:
        cfg = ChandelierConfig(enabled=True, atr_multiple=1.5)
        levels = compute_chandelier_levels(cfg, highs, flat_atr, entry_bar=0)
        # At index 0: highest=100, level=100-3=97
        assert levels[0] == pytest.approx(97.0)


# ---------------------------------------------------------------------------
# Breakeven
# ---------------------------------------------------------------------------


class TestBreakeven:
    """Tests for compute_breakeven_level."""

    def test_nan_before_trigger(self) -> None:
        cfg = BreakevenConfig(enabled=True, mode="percent", trigger_percent=5.0, trigger_atr_multiple=2.0)
        highs = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        levels = compute_breakeven_level(100.0, cfg, highs, atr_at_entry=2.0)
        # Trigger at 105.0, never reached
        assert np.all(np.isnan(levels))

    def test_equals_entry_after_trigger(self) -> None:
        cfg = BreakevenConfig(enabled=True, mode="percent", trigger_percent=3.0, trigger_atr_multiple=2.0)
        highs = np.array([100.0, 101.0, 103.5, 104.0, 102.0])
        levels = compute_breakeven_level(100.0, cfg, highs, atr_at_entry=2.0)
        # Trigger at 103.0, first triggered at index 2
        assert levels[2] == pytest.approx(100.0)
        assert levels[3] == pytest.approx(100.0)
        assert levels[4] == pytest.approx(100.0)

    def test_atr_mode_trigger(self) -> None:
        cfg = BreakevenConfig(enabled=True, mode="atr", trigger_percent=3.0, trigger_atr_multiple=1.5)
        highs = np.array([100.0, 103.0, 104.0, 105.0])
        levels = compute_breakeven_level(100.0, cfg, highs, atr_at_entry=2.0)
        # Trigger at 100 + 1.5*2 = 103.0, triggered at index 1
        assert levels[1] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Indicator exits
# ---------------------------------------------------------------------------


class TestIndicatorExits:
    """Tests for generate_indicator_exit_signals."""

    def test_returns_bool_series(self, ohlcv_df: pd.DataFrame) -> None:
        gene = IndicatorExitGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=70.0,
            params={"period": 14},
        )
        meta = load_strategy_config(Path("config"))
        signals = generate_indicator_exit_signals((gene,), ohlcv_df, meta.indicator_outputs)
        assert signals.dtype == bool

    def test_disabled_gene_ignored(self, ohlcv_df: pd.DataFrame) -> None:
        gene = IndicatorExitGene(
            enabled=False,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=70.0,
            params={"period": 14},
        )
        meta = load_strategy_config(Path("config"))
        signals = generate_indicator_exit_signals((gene,), ohlcv_df, meta.indicator_outputs)
        assert not signals.any()

    def test_or_combination(self, ohlcv_df: pd.DataFrame) -> None:
        gene1 = IndicatorExitGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=70.0,
            params={"period": 14},
        )
        gene2 = IndicatorExitGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="LT",
            threshold=30.0,
            params={"period": 14},
        )
        meta = load_strategy_config(Path("config"))
        signals = generate_indicator_exit_signals((gene1, gene2), ohlcv_df, meta.indicator_outputs)
        # OR: should have signals wherever RSI > 70 OR RSI < 30
        assert signals.dtype == bool


# ---------------------------------------------------------------------------
# Time exits
# ---------------------------------------------------------------------------


class TestTimeExits:
    """Tests for generate_time_exit_signal."""

    @pytest.fixture
    def time_df(self) -> pd.DataFrame:
        """Create a small DataFrame with known dates."""
        dates = pd.bdate_range("2020-01-06", periods=20)  # Starts Monday
        n = len(dates)
        return pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.0 + i * 0.1 for i in range(n)],
                "Volume": [500_000.0] * n,
            },
            index=dates,
        )

    def test_max_days(self, time_df: pd.DataFrame) -> None:
        cfg = TimeExitGene(
            max_days_enabled=True,
            max_days=5,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=False,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        signals = generate_time_exit_signal(cfg, entry_bar=2, df=time_df)
        assert signals[7]  # entry_bar(2) + max_days(5) = 7
        assert signals.sum() == 1

    def test_weekday_exit(self, time_df: pd.DataFrame) -> None:
        # Starts Monday (0), Friday is weekday 4
        cfg = TimeExitGene(
            max_days_enabled=False,
            max_days=10,
            weekday_exit_enabled=True,
            weekday=4,  # Friday
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=False,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        signals = generate_time_exit_signal(cfg, entry_bar=0, df=time_df)
        # Fridays in the date range should be flagged
        friday_signals = [signals[i] for i in range(1, len(time_df)) if time_df.index[i].weekday() == 4]
        assert all(friday_signals)

    def test_eow_friday(self, time_df: pd.DataFrame) -> None:
        cfg = TimeExitGene(
            max_days_enabled=False,
            max_days=10,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=True,
            eom_enabled=False,
            stagnation_enabled=False,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        signals = generate_time_exit_signal(cfg, entry_bar=0, df=time_df)
        for i in range(1, len(time_df)):
            if time_df.index[i].weekday() == 4:
                assert signals[i]

    def test_eom(self) -> None:
        # Create data spanning month boundary
        dates = pd.bdate_range("2020-01-27", periods=10)
        n = len(dates)
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.0] * n,
                "Volume": [500_000.0] * n,
            },
            index=dates,
        )
        cfg = TimeExitGene(
            max_days_enabled=False,
            max_days=10,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=True,
            stagnation_enabled=False,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        signals = generate_time_exit_signal(cfg, entry_bar=0, df=df)
        # Last trading day of January should be flagged
        eom_found = False
        for i in range(1, len(df)):
            if i + 1 < len(df) and df.index[i].month != df.index[i + 1].month:
                assert signals[i]
                eom_found = True
        assert eom_found

    def test_stagnation_triggers(self, time_df: pd.DataFrame) -> None:
        # Close moves 0.1 per bar, so after 5 bars: 0.5% move from 100.0
        # Threshold 1.0% should trigger stagnation
        cfg = TimeExitGene(
            max_days_enabled=False,
            max_days=10,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=True,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        signals = generate_time_exit_signal(cfg, entry_bar=0, df=time_df)
        # Stagnation check at bar 5, exit at bar 6
        assert signals[6]

    def test_stagnation_no_trigger(self) -> None:
        # Price moves significantly
        dates = pd.bdate_range("2020-01-06", periods=10)
        n = len(dates)
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.0 + i * 2.0 for i in range(n)],
                "Volume": [500_000.0] * n,
            },
            index=dates,
        )
        cfg = TimeExitGene(
            max_days_enabled=False,
            max_days=10,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=True,
            stagnation_days=3,
            stagnation_threshold=1.0,
        )
        signals = generate_time_exit_signal(cfg, entry_bar=0, df=df)
        # Price moved 6% after 3 bars, well above 1% threshold
        assert not signals.any()
