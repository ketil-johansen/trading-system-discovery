"""Unit tests for the main pipeline entry point."""

from __future__ import annotations

from tsd.config import CORE_INDICATORS
from tsd.main import _filter_strategy_meta
from tsd.strategy.genome import OutputMeta, StrategyMeta


def _make_meta() -> StrategyMeta:
    """Create a StrategyMeta with a mix of core and non-core indicators."""
    return StrategyMeta(
        num_entry_slots=4,
        num_indicator_exit_slots=1,
        num_filter_slots=2,
        indicator_names=("bollinger", "ema", "hma", "macd", "rsi", "sma", "supertrend"),
        indicator_outputs={
            "sma": (OutputMeta(key="sma", output_type="price_level"),),
            "ema": (OutputMeta(key="ema", output_type="price_level"),),
            "hma": (OutputMeta(key="hma", output_type="price_level"),),
            "rsi": (OutputMeta(key="rsi", output_type="oscillator", threshold_min=0, threshold_max=100),),
            "macd": (OutputMeta(key="macd", output_type="price_level"),),
            "bollinger": (OutputMeta(key="mavg", output_type="price_level"),),
            "supertrend": (OutputMeta(key="supertrend", output_type="price_level"),),
            "price_vs_ma": (OutputMeta(key="filter", output_type="binary", threshold_min=0, threshold_max=1),),
            "volatility_regime": (OutputMeta(key="regime", output_type="direction", threshold_min=0, threshold_max=1),),
        },
        indicator_params={
            "sma": (),
            "ema": (),
            "hma": (),
            "rsi": (),
            "macd": (),
            "bollinger": (),
            "supertrend": (),
        },
        max_indicator_params=3,
        max_indicator_outputs=1,
        filter_names=("price_vs_ma", "volatility_regime"),
        filter_params={"price_vs_ma": (), "volatility_regime": ()},
        max_filter_params=2,
        comparisons=("GT", "LT", "CROSS_ABOVE", "CROSS_BELOW"),
        exit_config={},
        time_exit_config={},
    )


class TestFilterStrategyMeta:
    """Tests for _filter_strategy_meta()."""

    def test_full_mode_unchanged(self) -> None:
        """Full mode returns meta unchanged."""
        meta = _make_meta()
        result = _filter_strategy_meta(meta, "full")
        assert result is meta

    def test_core_filters_indicators(self) -> None:
        """Core mode removes non-core indicators."""
        meta = _make_meta()
        result = _filter_strategy_meta(meta, "core")
        # hma and supertrend are not in CORE_INDICATORS
        assert "hma" not in result.indicator_names
        assert "supertrend" not in result.indicator_names
        # sma, ema, rsi, macd, bollinger are core
        assert "sma" in result.indicator_names
        assert "ema" in result.indicator_names
        assert "rsi" in result.indicator_names
        assert "macd" in result.indicator_names
        assert "bollinger" in result.indicator_names

    def test_core_filters_non_core_filters(self) -> None:
        """Core mode keeps price_vs_ma but removes volatility_regime."""
        meta = _make_meta()
        result = _filter_strategy_meta(meta, "core")
        assert "price_vs_ma" in result.filter_names
        assert "volatility_regime" not in result.filter_names

    def test_core_preserves_structure(self) -> None:
        """Core mode preserves non-indicator fields."""
        meta = _make_meta()
        result = _filter_strategy_meta(meta, "core")
        assert result.num_entry_slots == 4
        assert result.num_indicator_exit_slots == 1
        assert result.num_filter_slots == 2
        assert result.comparisons == meta.comparisons

    def test_core_indicators_constant(self) -> None:
        """CORE_INDICATORS has expected members."""
        assert "sma" in CORE_INDICATORS
        assert "ema" in CORE_INDICATORS
        assert "rsi" in CORE_INDICATORS
        assert "macd" in CORE_INDICATORS
        assert "atr" in CORE_INDICATORS
        assert "bollinger" in CORE_INDICATORS
        assert "price_vs_ma" in CORE_INDICATORS
