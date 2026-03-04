"""Tests for tsd.data.downloader module."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsd.data.downloader import (
    OHLCV_COLUMNS,
    DownloadResult,
    is_up_to_date,
    save_stock_data,
)


def _make_ohlcv(days: int = 10, end_date: date | None = None) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    if end_date is None:
        end_date = date.today()
    dates = pd.bdate_range(end=end_date, periods=days)
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(days).cumsum()
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 1, days),
            "High": close + rng.uniform(0, 2, days),
            "Low": close - rng.uniform(0, 2, days),
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, days),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


@pytest.mark.unit
class TestIsUpToDate:
    """Tests for is_up_to_date freshness check."""

    def test_returns_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        assert is_up_to_date(tmp_path / "missing.parquet") is False

    def test_returns_true_for_recent_data(self, tmp_path: Path) -> None:
        df = _make_ohlcv(days=10, end_date=date.today())
        path = tmp_path / "recent.parquet"
        df.to_parquet(path, engine="pyarrow")
        assert is_up_to_date(path) is True

    def test_returns_false_for_old_data(self, tmp_path: Path) -> None:
        old_date = date.today() - timedelta(days=30)
        df = _make_ohlcv(days=10, end_date=old_date)
        path = tmp_path / "old.parquet"
        df.to_parquet(path, engine="pyarrow")
        assert is_up_to_date(path) is False

    def test_returns_false_for_empty_parquet(self, tmp_path: Path) -> None:
        df = pd.DataFrame(columns=OHLCV_COLUMNS)
        df.index.name = "Date"
        path = tmp_path / "empty.parquet"
        df.to_parquet(path, engine="pyarrow")
        assert is_up_to_date(path) is False


@pytest.mark.unit
class TestSaveStockData:
    """Tests for save_stock_data."""

    def test_creates_directory_and_file(self, tmp_path: Path) -> None:
        df = _make_ohlcv(days=5)
        path = save_stock_data(df, "test_mkt", "TEST", tmp_path)
        assert path.exists()
        assert path == tmp_path / "raw" / "test_mkt" / "TEST.parquet"

    def test_written_file_has_correct_columns(self, tmp_path: Path) -> None:
        df = _make_ohlcv(days=5)
        path = save_stock_data(df, "test_mkt", "TEST", tmp_path)
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == OHLCV_COLUMNS
        assert loaded.index.name == "Date"
        assert len(loaded) == 5


@pytest.mark.unit
class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_frozen(self) -> None:
        r = DownloadResult(
            market_key="test",
            ticker="T",
            success=True,
            rows=10,
            start_date="2020-01-01",
            end_date="2020-01-10",
            skipped=False,
            error=None,
        )
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]
