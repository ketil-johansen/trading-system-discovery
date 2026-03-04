"""Unit tests for signal generation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsd.strategy.genome import (
    FilterGene,
    IndicatorGene,
    OutputMeta,
    StrategyGenome,
    load_strategy_config,
    random_genome,
)
from tsd.strategy.signals import apply_condition, generate_entry_signals

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
def meta() -> dict[str, tuple[OutputMeta, ...]]:
    """Load indicator output metadata from config."""
    sm = load_strategy_config(Path("config"))
    return sm.indicator_outputs


# ---------------------------------------------------------------------------
# apply_condition
# ---------------------------------------------------------------------------


class TestApplyCondition:
    """Tests for apply_condition."""

    def test_gt_scalar(self) -> None:
        series = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = apply_condition(series, "GT", 25.0)
        assert list(result) == [False, False, True, True]

    def test_lt_scalar(self) -> None:
        series = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = apply_condition(series, "LT", 25.0)
        assert list(result) == [True, True, False, False]

    def test_cross_above_scalar(self) -> None:
        series = pd.Series([10.0, 20.0, 30.0, 25.0])
        result = apply_condition(series, "CROSS_ABOVE", 25.0)
        # Cross above at index 2 (prev=20 <= 25, current=30 > 25)
        assert list(result) == [False, False, True, False]

    def test_cross_below_scalar(self) -> None:
        series = pd.Series([30.0, 20.0, 10.0, 25.0])
        result = apply_condition(series, "CROSS_BELOW", 25.0)
        # Cross below at index 1 (prev=30 >= 25, current=20 < 25)
        assert list(result) == [False, True, False, False]

    def test_gt_series_threshold(self) -> None:
        series = pd.Series([10.0, 30.0, 20.0, 40.0])
        threshold = pd.Series([15.0, 25.0, 25.0, 35.0])
        result = apply_condition(series, "GT", threshold)
        assert list(result) == [False, True, False, True]

    def test_cross_above_series_threshold(self) -> None:
        series = pd.Series([10.0, 20.0, 30.0, 25.0])
        threshold = pd.Series([15.0, 25.0, 20.0, 30.0])
        result = apply_condition(series, "CROSS_ABOVE", threshold)
        # index 2: current=30 > 20, prev=20 <= 25 -> True
        assert result.iloc[2]

    def test_unknown_comparison_raises(self) -> None:
        series = pd.Series([10.0, 20.0])
        with pytest.raises(ValueError, match="Unknown comparison"):
            apply_condition(series, "INVALID", 15.0)

    def test_nan_handling(self) -> None:
        series = pd.Series([float("nan"), 20.0, 30.0])
        result = apply_condition(series, "GT", 15.0)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_returns_bool_dtype(self) -> None:
        series = pd.Series([10.0, 20.0, 30.0])
        result = apply_condition(series, "GT", 15.0)
        assert result.dtype == bool


# ---------------------------------------------------------------------------
# generate_entry_signals
# ---------------------------------------------------------------------------


class TestGenerateEntrySignals:
    """Tests for generate_entry_signals."""

    def test_returns_bool_series(self, ohlcv_df: pd.DataFrame, meta: dict[str, tuple[OutputMeta, ...]]) -> None:
        sm = load_strategy_config(Path("config"))
        g = random_genome(sm, rng=random.Random(42))
        signals = generate_entry_signals(g, ohlcv_df, meta)
        assert isinstance(signals, pd.Series)
        assert signals.dtype == bool

    def test_aligned_with_index(self, ohlcv_df: pd.DataFrame, meta: dict[str, tuple[OutputMeta, ...]]) -> None:
        sm = load_strategy_config(Path("config"))
        g = random_genome(sm, rng=random.Random(42))
        signals = generate_entry_signals(g, ohlcv_df, meta)
        assert list(signals.index) == list(ohlcv_df.index)

    def test_no_enabled_slots_returns_all_false(
        self, ohlcv_df: pd.DataFrame, meta: dict[str, tuple[OutputMeta, ...]]
    ) -> None:
        sm = load_strategy_config(Path("config"))
        g = random_genome(sm, rng=random.Random(42))
        # Disable all entry slots
        disabled = tuple(
            IndicatorGene(
                enabled=False,
                indicator_name=gene.indicator_name,
                output_key=gene.output_key,
                comparison=gene.comparison,
                threshold=gene.threshold,
                params=gene.params,
            )
            for gene in g.entry_indicators
        )
        no_entry = StrategyGenome(
            entry_indicators=disabled,
            combination_logic=g.combination_logic,
            limit_exits=g.limit_exits,
            indicator_exits=g.indicator_exits,
            time_exits=g.time_exits,
            filters=g.filters,
        )
        signals = generate_entry_signals(no_entry, ohlcv_df, meta)
        assert not signals.any()

    def test_and_requires_all(self, ohlcv_df: pd.DataFrame, meta: dict[str, tuple[OutputMeta, ...]]) -> None:
        # Create 2 entry slots with RSI: GT 30 and LT 70
        # AND means both must be true (RSI between 30 and 70)
        entry1 = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=30.0,
            params={"period": 14},
        )
        entry2 = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="LT",
            threshold=70.0,
            params={"period": 14},
        )
        sm = load_strategy_config(Path("config"))
        g = random_genome(sm, rng=random.Random(42))
        genome = StrategyGenome(
            entry_indicators=(entry1, entry2),
            combination_logic="AND",
            limit_exits=g.limit_exits,
            indicator_exits=g.indicator_exits,
            time_exits=g.time_exits,
            filters=(),
        )
        signals = generate_entry_signals(genome, ohlcv_df, meta)
        # Verify AND: compute each individually
        from tsd.indicators.base import compute_indicator  # noqa: PLC0415

        rsi = compute_indicator("rsi", ohlcv_df, {"period": 14}).values["rsi"]
        expected = ((rsi > 30.0) & (rsi < 70.0)).fillna(False)
        pd.testing.assert_series_equal(signals, expected.astype(bool), check_names=False)

    def test_or_requires_any(self, ohlcv_df: pd.DataFrame, meta: dict[str, tuple[OutputMeta, ...]]) -> None:
        entry1 = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="LT",
            threshold=30.0,
            params={"period": 14},
        )
        entry2 = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=70.0,
            params={"period": 14},
        )
        sm = load_strategy_config(Path("config"))
        g = random_genome(sm, rng=random.Random(42))
        genome = StrategyGenome(
            entry_indicators=(entry1, entry2),
            combination_logic="OR",
            limit_exits=g.limit_exits,
            indicator_exits=g.indicator_exits,
            time_exits=g.time_exits,
            filters=(),
        )
        signals = generate_entry_signals(genome, ohlcv_df, meta)
        from tsd.indicators.base import compute_indicator  # noqa: PLC0415

        rsi = compute_indicator("rsi", ohlcv_df, {"period": 14}).values["rsi"]
        expected = ((rsi < 30.0) | (rsi > 70.0)).fillna(False)
        pd.testing.assert_series_equal(signals, expected.astype(bool), check_names=False)

    def test_filter_gates_signals(self, ohlcv_df: pd.DataFrame, meta: dict[str, tuple[OutputMeta, ...]]) -> None:
        # Entry on RSI > 50, filtered by price_vs_ma
        entry = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=50.0,
            params={"period": 14},
        )
        flt = FilterGene(enabled=True, filter_name="price_vs_ma", params={"ma_period": 50})
        sm = load_strategy_config(Path("config"))
        g = random_genome(sm, rng=random.Random(42))
        genome = StrategyGenome(
            entry_indicators=(entry,),
            combination_logic="AND",
            limit_exits=g.limit_exits,
            indicator_exits=g.indicator_exits,
            time_exits=g.time_exits,
            filters=(flt,),
        )
        signals = generate_entry_signals(genome, ohlcv_df, meta)
        # Filtered signals should be a subset of unfiltered
        genome_no_filter = StrategyGenome(
            entry_indicators=(entry,),
            combination_logic="AND",
            limit_exits=g.limit_exits,
            indicator_exits=g.indicator_exits,
            time_exits=g.time_exits,
            filters=(),
        )
        unfiltered = generate_entry_signals(genome_no_filter, ohlcv_df, meta)
        # Where filtered is True, unfiltered must also be True
        assert (signals & ~unfiltered).sum() == 0
