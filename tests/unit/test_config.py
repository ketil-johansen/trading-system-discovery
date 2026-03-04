"""Tests for tsd.config module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tsd.config import (
    Config,
    MarketConfig,
    env_bool,
    env_float,
    env_int,
    env_str,
    get_market,
    load_config,
    load_markets,
)


@pytest.mark.unit
class TestEnvHelpers:
    """Tests for env_* helper functions."""

    def test_env_str_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TSD_TEST_STR", raising=False)
        assert env_str("TSD_TEST_STR", "fallback") == "fallback"

    def test_env_str_reads_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TSD_TEST_STR", "custom")
        assert env_str("TSD_TEST_STR", "fallback") == "custom"

    def test_env_int_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TSD_TEST_INT", raising=False)
        assert env_int("TSD_TEST_INT", 42) == 42

    def test_env_int_reads_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TSD_TEST_INT", "99")
        assert env_int("TSD_TEST_INT", 42) == 99

    def test_env_float_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TSD_TEST_FLOAT", raising=False)
        assert env_float("TSD_TEST_FLOAT", 1.5) == 1.5

    def test_env_float_reads_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TSD_TEST_FLOAT", "2.7")
        assert env_float("TSD_TEST_FLOAT", 1.5) == 2.7

    def test_env_bool_returns_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TSD_TEST_BOOL", raising=False)
        assert env_bool("TSD_TEST_BOOL", True) is True

    def test_env_bool_reads_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("1", "true", "True", "YES", "yes"):
            monkeypatch.setenv("TSD_TEST_BOOL", val)
            assert env_bool("TSD_TEST_BOOL", False) is True

    def test_env_bool_reads_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("0", "false", "no", ""):
            monkeypatch.setenv("TSD_TEST_BOOL", val)
            assert env_bool("TSD_TEST_BOOL", True) is False


@pytest.mark.unit
class TestLoadMarkets:
    """Tests for YAML market loading."""

    def test_load_markets_from_yaml(self, tmp_path: Path) -> None:
        data = {
            "markets": [
                {
                    "key": "test_market",
                    "name": "Test Market",
                    "index_ticker": "^TEST",
                    "stock_suffix": ".T",
                    "expected_constituents": 10,
                },
            ]
        }
        (tmp_path / "markets.yaml").write_text(yaml.dump(data))
        markets = load_markets(tmp_path)
        assert len(markets) == 1
        assert isinstance(markets[0], MarketConfig)
        assert markets[0].key == "test_market"
        assert markets[0].stock_suffix == ".T"
        assert markets[0].expected_constituents == 10

    def test_load_markets_invalid_yaml_raises(self, tmp_path: Path) -> None:
        (tmp_path / "markets.yaml").write_text("not_markets: []")
        with pytest.raises(ValueError, match="expected top-level 'markets' key"):
            load_markets(tmp_path)

    def test_load_markets_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_markets(tmp_path)

    def test_load_markets_produces_correct_count(self) -> None:
        markets = load_markets(Path("config"))
        assert len(markets) == 6


@pytest.mark.unit
class TestLoadConfig:
    """Tests for load_config round-trip."""

    def test_load_config_returns_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {
            "markets": [
                {
                    "key": "demo",
                    "name": "Demo",
                    "index_ticker": "^DEMO",
                    "stock_suffix": "",
                    "expected_constituents": 5,
                },
            ]
        }
        (tmp_path / "markets.yaml").write_text(yaml.dump(data))
        monkeypatch.setenv("TSD_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("TSD_LOG_LEVEL", "DEBUG")
        config = load_config()
        assert isinstance(config, Config)
        assert config.log_level == "DEBUG"
        assert len(config.markets) == 1
        assert config.markets[0].key == "demo"


@pytest.mark.unit
class TestGetMarket:
    """Tests for get_market lookup helper."""

    def test_get_market_returns_matching(self) -> None:
        config = Config(
            log_level="INFO",
            data_dir=Path("data"),
            results_dir=Path("results"),
            config_dir=Path("config"),
            population_size=100,
            max_generations=50,
            n_trials=100,
            download_delay=1.5,
            quality_gap_threshold_days=5,
            quality_outlier_threshold=0.50,
            quality_min_coverage=0.80,
            quality_min_rows=100,
            markets=(
                MarketConfig(key="abc", name="ABC", index_ticker="^ABC", stock_suffix="", expected_constituents=10),
            ),
        )
        m = get_market(config, "abc")
        assert m.key == "abc"

    def test_get_market_raises_on_invalid_key(self) -> None:
        config = Config(
            log_level="INFO",
            data_dir=Path("data"),
            results_dir=Path("results"),
            config_dir=Path("config"),
            population_size=100,
            max_generations=50,
            n_trials=100,
            download_delay=1.5,
            quality_gap_threshold_days=5,
            quality_outlier_threshold=0.50,
            quality_min_coverage=0.80,
            quality_min_rows=100,
            markets=(),
        )
        with pytest.raises(ValueError, match="Unknown market key"):
            get_market(config, "nonexistent")
