"""Tests for tsd.data.quality module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsd.data.quality import (
    MarketQualityReport,
    StockQualityResult,
    save_report,
    validate_market,
    validate_stock,
)


def _make_clean_ohlcv(days: int = 200) -> pd.DataFrame:
    """Create a clean synthetic OHLCV DataFrame that passes all checks."""
    dates = pd.bdate_range(end="2024-06-01", periods=days)
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(days).cumsum() * 0.5
    open_ = close + rng.uniform(-0.5, 0.5, days)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, days)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, days)
    volume = rng.integers(100_000, 1_000_000, days)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save DataFrame to Parquet and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    return path


@pytest.mark.unit
class TestValidateStock:
    """Tests for validate_stock."""

    def test_clean_data_passes(self, tmp_path: Path) -> None:
        df = _make_clean_ohlcv()
        path = _save_parquet(df, tmp_path / "CLEAN.parquet")
        result = validate_stock(path)
        assert result.status == "PASS"
        assert result.ticker == "CLEAN"
        assert result.row_count == 200
        assert result.ohlc_violations == 0
        assert result.gap_count == 0
        assert result.outlier_count == 0
        assert len(result.errors) == 0

    def test_missing_columns_fails(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {"Open": [1.0], "Close": [2.0]},
            index=pd.DatetimeIndex(["2024-01-01"], name="Date"),
        )
        path = _save_parquet(df, tmp_path / "MISSING.parquet")
        result = validate_stock(path)
        assert result.status == "FAIL"
        assert "Missing columns" in result.errors[0]

    def test_nulls_reported(self, tmp_path: Path) -> None:
        df = _make_clean_ohlcv()
        df.iloc[5, df.columns.get_loc("Close")] = np.nan
        df.iloc[10, df.columns.get_loc("Volume")] = np.nan
        path = _save_parquet(df, tmp_path / "NULLS.parquet")
        result = validate_stock(path)
        assert result.status == "WARN"
        assert result.null_counts["Close"] == 1
        assert result.null_counts["Volume"] == 1

    def test_ohlc_violations_reported(self, tmp_path: Path) -> None:
        df = _make_clean_ohlcv()
        # Make High < Close (violation)
        df.iloc[3, df.columns.get_loc("High")] = df.iloc[3]["Close"] - 5.0
        # Make Low > Open (violation)
        df.iloc[7, df.columns.get_loc("Low")] = df.iloc[7]["Open"] + 5.0
        path = _save_parquet(df, tmp_path / "OHLC.parquet")
        result = validate_stock(path)
        assert result.status == "WARN"
        assert result.ohlc_violations >= 2

    def test_gaps_detected(self, tmp_path: Path) -> None:
        df = _make_clean_ohlcv()
        # Remove 10 consecutive business days to create a large gap
        dates_to_drop = df.index[50:60]
        df = df.drop(dates_to_drop)
        path = _save_parquet(df, tmp_path / "GAPS.parquet")
        result = validate_stock(path, gap_threshold_days=5)
        assert result.gap_count >= 1
        assert result.max_gap_days > 5

    def test_outlier_returns_flagged(self, tmp_path: Path) -> None:
        df = _make_clean_ohlcv()
        # Create a 60% single-day spike
        idx = 100
        df.iloc[idx, df.columns.get_loc("Close")] = df.iloc[idx - 1]["Close"] * 1.60
        # Adjust High to maintain OHLC ordering
        df.iloc[idx, df.columns.get_loc("High")] = df.iloc[idx]["Close"] + 1.0
        path = _save_parquet(df, tmp_path / "OUTLIER.parquet")
        result = validate_stock(path, outlier_threshold=0.50)
        assert result.outlier_count >= 1

    def test_too_few_rows_fails(self, tmp_path: Path) -> None:
        df = _make_clean_ohlcv(days=10)
        path = _save_parquet(df, tmp_path / "SHORT.parquet")
        result = validate_stock(path, min_rows=100)
        assert result.status == "FAIL"
        assert result.row_count == 10
        assert any("Too few rows" in e for e in result.errors)

    def test_unreadable_file_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "BAD.parquet"
        path.write_text("not a parquet file")
        result = validate_stock(path)
        assert result.status == "FAIL"
        assert any("Cannot read file" in e for e in result.errors)


@pytest.mark.unit
class TestValidateMarket:
    """Tests for validate_market."""

    def test_aggregates_results(self, tmp_path: Path) -> None:
        market_dir = tmp_path / "raw" / "test_mkt"
        market_dir.mkdir(parents=True)

        # Create 2 clean + 1 short (FAIL)
        for name in ["GOOD1", "GOOD2"]:
            _save_parquet(_make_clean_ohlcv(), market_dir / f"{name}.parquet")

        short_df = _make_clean_ohlcv(days=10)
        _save_parquet(short_df, market_dir / "SHORT.parquet")

        report = validate_market("test_mkt", tmp_path, min_rows=100)
        assert report.total_stocks == 3
        assert report.pass_count == 2
        assert report.fail_count == 1
        assert report.market_key == "test_mkt"
        assert len(report.stock_results) == 3


@pytest.mark.unit
class TestSaveReport:
    """Tests for save_report."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        report = MarketQualityReport(
            market_key="test",
            timestamp="2024-06-01T12:00:00",
            total_stocks=1,
            pass_count=1,
            warn_count=0,
            fail_count=0,
            stock_results=(
                StockQualityResult(
                    ticker="TEST",
                    status="PASS",
                    row_count=200,
                    date_range="2023-01-01 to 2024-06-01",
                    null_counts={"Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 0},
                    ohlc_violations=0,
                    gap_count=0,
                    max_gap_days=0,
                    outlier_count=0,
                    coverage_pct=0.98,
                    errors=[],
                ),
            ),
        )
        path = save_report(report, tmp_path)
        assert path.exists()
        assert path.name == "test_quality.json"

        data = json.loads(path.read_text())
        assert data["market_key"] == "test"
        assert data["total_stocks"] == 1
        assert len(data["stock_results"]) == 1
        assert data["stock_results"][0]["ticker"] == "TEST"

    def test_round_trips_correctly(self, tmp_path: Path) -> None:
        report = MarketQualityReport(
            market_key="rt",
            timestamp="2024-06-01T12:00:00",
            total_stocks=2,
            pass_count=1,
            warn_count=1,
            fail_count=0,
            stock_results=(
                StockQualityResult(
                    ticker="A",
                    status="PASS",
                    row_count=200,
                    date_range="2023-01-01 to 2024-06-01",
                    null_counts={"Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 0},
                    ohlc_violations=0,
                    gap_count=0,
                    max_gap_days=0,
                    outlier_count=0,
                    coverage_pct=0.99,
                    errors=[],
                ),
                StockQualityResult(
                    ticker="B",
                    status="WARN",
                    row_count=150,
                    date_range="2023-01-01 to 2024-06-01",
                    null_counts={"Open": 0, "High": 0, "Low": 0, "Close": 2, "Volume": 0},
                    ohlc_violations=3,
                    gap_count=1,
                    max_gap_days=7,
                    outlier_count=0,
                    coverage_pct=0.75,
                    errors=["Found 2 null values"],
                ),
            ),
        )
        path = save_report(report, tmp_path)
        data = json.loads(path.read_text())

        # Verify all fields survived serialization
        assert data["total_stocks"] == 2
        assert data["pass_count"] == 1
        assert data["warn_count"] == 1
        stock_b = data["stock_results"][1]
        assert stock_b["ohlc_violations"] == 3
        assert stock_b["null_counts"]["Close"] == 2


@pytest.mark.unit
class TestStockQualityResult:
    """Tests for StockQualityResult dataclass."""

    def test_frozen(self) -> None:
        r = StockQualityResult(
            ticker="T",
            status="PASS",
            row_count=100,
            date_range="2024-01-01 to 2024-06-01",
            null_counts={},
            ohlc_violations=0,
            gap_count=0,
            max_gap_days=0,
            outlier_count=0,
            coverage_pct=1.0,
            errors=[],
        )
        with pytest.raises(AttributeError):
            r.status = "FAIL"  # type: ignore[misc]
