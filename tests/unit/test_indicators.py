"""Unit tests for the indicator library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsd.indicators.base import (
    IndicatorMeta,
    IndicatorResult,
    compute_indicator,
    get_indicator_names,
    load_indicator_config,
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
def short_df() -> pd.DataFrame:
    """Generate a very short OHLCV DataFrame (3 rows)."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [500_000.0, 600_000.0, 700_000.0],
        },
        index=dates,
    )


@pytest.fixture
def config_dir() -> Path:
    """Return the path to the config directory."""
    return Path("config")


# ---------------------------------------------------------------------------
# TestIndicatorResult
# ---------------------------------------------------------------------------


class TestIndicatorResult:
    """Tests for the IndicatorResult frozen dataclass."""

    def test_frozen(self, ohlcv_df: pd.DataFrame) -> None:
        result = IndicatorResult(name="test", values={"a": ohlcv_df["Close"]}, params={"period": 14})
        with pytest.raises(AttributeError):
            result.name = "changed"  # type: ignore[misc]

    def test_values_accessible(self, ohlcv_df: pd.DataFrame) -> None:
        result = IndicatorResult(name="test", values={"a": ohlcv_df["Close"]}, params={"period": 14})
        assert "a" in result.values
        assert len(result.values["a"]) == len(ohlcv_df)


# ---------------------------------------------------------------------------
# TestLoadIndicatorConfig
# ---------------------------------------------------------------------------


class TestLoadIndicatorConfig:
    """Tests for loading indicator metadata from YAML."""

    def test_loads_16_indicators(self, config_dir: Path) -> None:
        indicators = load_indicator_config(config_dir)
        assert len(indicators) == 16

    def test_returns_tuple_of_indicator_meta(self, config_dir: Path) -> None:
        indicators = load_indicator_config(config_dir)
        assert isinstance(indicators, tuple)
        assert all(isinstance(i, IndicatorMeta) for i in indicators)

    def test_indicator_has_params(self, config_dir: Path) -> None:
        indicators = load_indicator_config(config_dir)
        by_name = {i.name: i for i in indicators}
        rsi_meta = by_name["rsi"]
        assert len(rsi_meta.params) == 1
        assert rsi_meta.params[0].name == "period"
        assert rsi_meta.params[0].param_type == "int"
        assert rsi_meta.params[0].min_value == 5
        assert rsi_meta.params[0].max_value == 30
        assert rsi_meta.params[0].default == 14

    def test_param_meta_is_frozen(self, config_dir: Path) -> None:
        indicators = load_indicator_config(config_dir)
        param = indicators[0].params[0] if indicators[0].params else None
        if param is not None:
            with pytest.raises(AttributeError):
                param.name = "changed"  # type: ignore[misc]

    def test_categories_present(self, config_dir: Path) -> None:
        indicators = load_indicator_config(config_dir)
        categories = {i.category for i in indicators}
        assert "trend" in categories
        assert "momentum" in categories
        assert "volatility" in categories
        assert "volume" in categories
        assert "filter" in categories

    def test_obv_has_no_params(self, config_dir: Path) -> None:
        indicators = load_indicator_config(config_dir)
        by_name = {i.name: i for i in indicators}
        assert len(by_name["obv"].params) == 0

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_indicator_config(tmp_path)

    def test_invalid_yaml_structure(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "indicators.yaml"
        yaml_path.write_text("- just a list")
        with pytest.raises(ValueError, match="expected a mapping"):
            load_indicator_config(tmp_path)


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the indicator registry and dispatcher."""

    def test_16_names_registered(self) -> None:
        names = get_indicator_names()
        assert len(names) == 16

    def test_names_sorted(self) -> None:
        names = get_indicator_names()
        assert names == sorted(names)

    def test_dispatch_sma(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("sma", ohlcv_df, {})
        assert isinstance(result, IndicatorResult)
        assert result.name == "sma"

    def test_unknown_raises_key_error(self, ohlcv_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="Unknown indicator"):
            compute_indicator("nonexistent", ohlcv_df, {})

    def test_expected_indicator_names(self) -> None:
        names = get_indicator_names()
        expected = [
            "atr",
            "bollinger",
            "cmf",
            "ema",
            "hma",
            "ichimoku",
            "keltner",
            "macd",
            "obv",
            "price_vs_ma",
            "rsi",
            "sma",
            "stochastic",
            "supertrend",
            "volatility_regime",
            "williams_r",
        ]
        assert names == expected


# ---------------------------------------------------------------------------
# TestTrendIndicators
# ---------------------------------------------------------------------------


class TestTrendIndicators:
    """Tests for trend indicator functions."""

    def test_sma_output_shape(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("sma", ohlcv_df, {"period": 20})
        assert "sma" in result.values
        assert len(result.values["sma"]) == len(ohlcv_df)

    def test_ema_output_shape(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("ema", ohlcv_df, {"period": 20})
        assert "ema" in result.values
        assert len(result.values["ema"]) == len(ohlcv_df)

    def test_hma_differs_from_sma(self, ohlcv_df: pd.DataFrame) -> None:
        sma_result = compute_indicator("sma", ohlcv_df, {"period": 20})
        hma_result = compute_indicator("hma", ohlcv_df, {"period": 20})
        # Drop NaN rows for comparison
        sma_vals = sma_result.values["sma"].dropna()
        hma_vals = hma_result.values["hma"].dropna()
        common = sma_vals.index.intersection(hma_vals.index)
        assert not np.allclose(sma_vals[common].values, hma_vals[common].values)

    def test_ichimoku_4_outputs(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("ichimoku", ohlcv_df, {})
        assert set(result.values.keys()) == {"conversion", "base", "span_a", "span_b"}

    def test_supertrend_direction_values(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("supertrend", ohlcv_df, {})
        assert "supertrend" in result.values
        assert "direction" in result.values
        direction = result.values["direction"]
        unique_vals = set(direction.unique())
        assert unique_vals <= {1.0, -1.0}

    def test_sma_params_stored(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("sma", ohlcv_df, {"period": 50})
        assert result.params == {"period": 50}


# ---------------------------------------------------------------------------
# TestMomentumIndicators
# ---------------------------------------------------------------------------


class TestMomentumIndicators:
    """Tests for momentum indicator functions."""

    def test_rsi_range(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("rsi", ohlcv_df, {"period": 14})
        rsi_vals = result.values["rsi"].dropna()
        assert (rsi_vals >= 0).all()
        assert (rsi_vals <= 100).all()

    def test_stochastic_range(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("stochastic", ohlcv_df, {})
        k_vals = result.values["k"].dropna()
        d_vals = result.values["d"].dropna()
        assert (k_vals >= 0).all() and (k_vals <= 100).all()
        assert (d_vals >= 0).all() and (d_vals <= 100).all()

    def test_macd_3_outputs(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("macd", ohlcv_df, {})
        assert set(result.values.keys()) == {"macd", "signal", "histogram"}

    def test_williams_r_range(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("williams_r", ohlcv_df, {"period": 14})
        wr_vals = result.values["williams_r"].dropna()
        assert (wr_vals >= -100).all()
        assert (wr_vals <= 0).all()


# ---------------------------------------------------------------------------
# TestVolatilityIndicators
# ---------------------------------------------------------------------------


class TestVolatilityIndicators:
    """Tests for volatility indicator functions."""

    def test_atr_non_negative(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("atr", ohlcv_df, {"period": 14})
        atr_vals = result.values["atr"].dropna()
        assert (atr_vals >= 0).all()

    def test_bollinger_upper_gte_lower(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("bollinger", ohlcv_df, {})
        upper = result.values["upper"].dropna()
        lower = result.values["lower"].dropna()
        common = upper.index.intersection(lower.index)
        assert (upper[common] >= lower[common]).all()

    def test_bollinger_5_outputs(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("bollinger", ohlcv_df, {})
        assert set(result.values.keys()) == {"mavg", "upper", "lower", "pband", "wband"}

    def test_keltner_upper_gte_lower(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("keltner", ohlcv_df, {})
        upper = result.values["upper"].dropna()
        lower = result.values["lower"].dropna()
        common = upper.index.intersection(lower.index)
        assert (upper[common] >= lower[common]).all()

    def test_keltner_5_outputs(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("keltner", ohlcv_df, {})
        assert set(result.values.keys()) == {"mband", "upper", "lower", "pband", "wband"}


# ---------------------------------------------------------------------------
# TestVolumeIndicators
# ---------------------------------------------------------------------------


class TestVolumeIndicators:
    """Tests for volume indicator functions."""

    def test_obv_length(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("obv", ohlcv_df, {})
        assert len(result.values["obv"]) == len(ohlcv_df)

    def test_obv_no_params(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("obv", ohlcv_df, {})
        assert result.params == {}

    def test_cmf_range(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("cmf", ohlcv_df, {"period": 20})
        cmf_vals = result.values["cmf"].dropna()
        assert (cmf_vals >= -1).all()
        assert (cmf_vals <= 1).all()


# ---------------------------------------------------------------------------
# TestFilterIndicators
# ---------------------------------------------------------------------------


class TestFilterIndicators:
    """Tests for filter indicator functions."""

    def test_price_vs_ma_binary(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("price_vs_ma", ohlcv_df, {"ma_period": 20})
        filter_vals = result.values["filter"].dropna()
        unique_vals = set(filter_vals.unique())
        assert unique_vals <= {0.0, 1.0}

    def test_volatility_regime_values(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("volatility_regime", ohlcv_df, {})
        regime_vals = result.values["regime"].dropna()
        unique_vals = set(regime_vals.unique())
        assert unique_vals <= {0.0, 0.5, 1.0}

    def test_volatility_regime_has_ratio(self, ohlcv_df: pd.DataFrame) -> None:
        result = compute_indicator("volatility_regime", ohlcv_df, {})
        assert "ratio" in result.values


# ---------------------------------------------------------------------------
# TestAllIndicatorsCommonProperties
# ---------------------------------------------------------------------------


class TestAllIndicatorsCommonProperties:
    """Tests that apply to all indicators."""

    def test_all_return_indicator_result(self, ohlcv_df: pd.DataFrame) -> None:
        for name in get_indicator_names():
            result = compute_indicator(name, ohlcv_df, {})
            assert isinstance(result, IndicatorResult), f"{name} did not return IndicatorResult"

    def test_all_preserve_index(self, ohlcv_df: pd.DataFrame) -> None:
        for name in get_indicator_names():
            result = compute_indicator(name, ohlcv_df, {})
            for key, series in result.values.items():
                assert len(series) == len(ohlcv_df), f"{name}.{key} length {len(series)} != {len(ohlcv_df)}"

    def test_all_handle_short_data(self, short_df: pd.DataFrame) -> None:
        for name in get_indicator_names():
            result = compute_indicator(name, short_df, {})
            assert isinstance(result, IndicatorResult), f"{name} crashed on short data"
            for key, series in result.values.items():
                assert len(series) == len(short_df), f"{name}.{key} length mismatch on short data"

    def test_name_matches_key(self, ohlcv_df: pd.DataFrame) -> None:
        for name in get_indicator_names():
            result = compute_indicator(name, ohlcv_df, {})
            assert result.name == name, f"Expected name '{name}', got '{result.name}'"
