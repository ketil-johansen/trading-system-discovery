"""Integration tests for the data download pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsd.config import MarketConfig
from tsd.data.constituents import load_constituents
from tsd.data.downloader import OHLCV_COLUMNS, download_market, is_up_to_date


@pytest.mark.integration
class TestDownloadPipeline:
    """End-to-end tests that download real data from yfinance."""

    def test_download_single_stock(self, tmp_path: Path) -> None:
        """Download one real Nordic stock and verify output."""
        market = MarketConfig(
            key="omxs30",
            name="OMXS30",
            index_ticker="^OMXS30",
            stock_suffix=".ST",
            expected_constituents=30,
        )
        constituents = pd.DataFrame(
            {
                "ticker": ["VOLV-B"],
                "name": ["AB Volvo B"],
                "yahoo_ticker": ["VOLV-B.ST"],
            }
        )
        results = download_market(
            market=market,
            constituents=constituents,
            data_dir=tmp_path,
            start="2024-01-01",
            end="2024-03-01",
            delay=0.0,
        )

        assert len(results) == 1
        r = results[0]
        assert r.success is True
        assert r.skipped is False
        assert r.rows > 0

        # Verify Parquet file
        parquet_path = tmp_path / "raw" / "omxs30" / "VOLV-B.parquet"
        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)
        assert list(df.columns) == OHLCV_COLUMNS
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    def test_rerun_skips_fresh_data(self, tmp_path: Path) -> None:
        """Verify that a second run skips already-downloaded data."""
        market = MarketConfig(
            key="omxs30",
            name="OMXS30",
            index_ticker="^OMXS30",
            stock_suffix=".ST",
            expected_constituents=30,
        )
        constituents = pd.DataFrame(
            {
                "ticker": ["VOLV-B"],
                "name": ["AB Volvo B"],
                "yahoo_ticker": ["VOLV-B.ST"],
            }
        )
        # First run — downloads
        download_market(
            market=market,
            constituents=constituents,
            data_dir=tmp_path,
            start="2024-01-01",
            end="2024-12-31",
            delay=0.0,
        )

        # Second run — should skip
        results = download_market(
            market=market,
            constituents=constituents,
            data_dir=tmp_path,
            start="2024-01-01",
            end="2024-12-31",
            delay=0.0,
        )
        assert len(results) == 1
        assert results[0].skipped is True

    def test_load_real_constituents(self) -> None:
        """Verify that committed constituent CSVs load correctly."""
        data_dir = Path("data")
        for market_key in ["omxs30", "omxc25", "omxh25", "obx", "nasdaq_100", "sp500"]:
            df = load_constituents(market_key, data_dir)
            assert len(df) > 0
            assert "ticker" in df.columns
            assert "yahoo_ticker" in df.columns
