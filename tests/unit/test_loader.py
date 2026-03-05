"""Unit tests for the market data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsd.data.loader import load_market_data


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": range(n),
            "High": range(n),
            "Low": range(n),
            "Close": range(n),
            "Volume": [1000] * n,
        },
        index=dates,
    )


class TestLoadMarketData:
    """Tests for load_market_data()."""

    def test_loads_parquet_files(self, tmp_path: Path) -> None:
        """Loads all parquet files from market directory."""
        market_dir = tmp_path / "raw" / "test_market"
        market_dir.mkdir(parents=True)
        _make_ohlcv(50).to_parquet(market_dir / "AAPL.parquet")
        _make_ohlcv(50).to_parquet(market_dir / "MSFT.parquet")

        result = load_market_data("test_market", tmp_path)
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        assert len(result["AAPL"]) == 50

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing market directory."""
        with pytest.raises(FileNotFoundError, match="No data directory"):
            load_market_data("nonexistent", tmp_path)

    def test_no_parquet_files_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when directory exists but has no parquet files."""
        market_dir = tmp_path / "raw" / "empty_market"
        market_dir.mkdir(parents=True)
        with pytest.raises(ValueError, match="No Parquet files"):
            load_market_data("empty_market", tmp_path)

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        """Skips parquet files with empty DataFrames."""
        market_dir = tmp_path / "raw" / "test_market"
        market_dir.mkdir(parents=True)
        _make_ohlcv(50).to_parquet(market_dir / "GOOD.parquet")
        pd.DataFrame().to_parquet(market_dir / "EMPTY.parquet")

        result = load_market_data("test_market", tmp_path)
        assert len(result) == 1
        assert "GOOD" in result
