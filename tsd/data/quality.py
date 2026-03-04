"""Data validation and gap detection for OHLCV time series."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_MIN_ROWS_FOR_STATS = 2


@dataclass(frozen=True)
class StockQualityResult:
    """Per-stock validation result."""

    ticker: str
    status: str
    row_count: int
    date_range: str
    null_counts: dict[str, int]
    ohlc_violations: int
    gap_count: int
    max_gap_days: int
    outlier_count: int
    coverage_pct: float
    errors: list[str]


@dataclass(frozen=True)
class MarketQualityReport:
    """Per-market quality summary."""

    market_key: str
    timestamp: str
    total_stocks: int
    pass_count: int
    warn_count: int
    fail_count: int
    stock_results: tuple[StockQualityResult, ...]


def validate_stock(
    path: Path,
    gap_threshold_days: int = 5,
    outlier_threshold: float = 0.50,
    min_coverage: float = 0.80,
    min_rows: int = 100,
) -> StockQualityResult:
    """Validate a single OHLCV Parquet file.

    Checks column presence, index integrity, null values, price ordering,
    gaps, coverage, and outlier returns.

    Args:
        path: Path to a Parquet file with OHLCV data and DatetimeIndex.
        gap_threshold_days: Calendar days beyond which a gap is flagged.
        outlier_threshold: Absolute daily return threshold for outlier flagging.
        min_coverage: Minimum business-day coverage fraction before WARN.
        min_rows: Minimum row count before FAIL.

    Returns:
        StockQualityResult with validation findings.
    """
    ticker = path.stem
    errors: list[str] = []

    # Try to read the file
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return StockQualityResult(
            ticker=ticker,
            status="FAIL",
            row_count=0,
            date_range="",
            null_counts={},
            ohlc_violations=0,
            gap_count=0,
            max_gap_days=0,
            outlier_count=0,
            coverage_pct=0.0,
            errors=[f"Cannot read file: {exc}"],
        )

    # Column validation
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        return StockQualityResult(
            ticker=ticker,
            status="FAIL",
            row_count=len(df),
            date_range="",
            null_counts={},
            ohlc_violations=0,
            gap_count=0,
            max_gap_days=0,
            outlier_count=0,
            coverage_pct=0.0,
            errors=[f"Missing columns: {missing_cols}"],
        )

    # Index validation
    errors.extend(_validate_index(df))

    # Too few rows → FAIL
    if len(df) < min_rows:
        return StockQualityResult(
            ticker=ticker,
            status="FAIL",
            row_count=len(df),
            date_range=_date_range_str(df),
            null_counts=_null_counts(df),
            ohlc_violations=0,
            gap_count=0,
            max_gap_days=0,
            outlier_count=0,
            coverage_pct=0.0,
            errors=[f"Too few rows: {len(df)} < {min_rows}", *errors],
        )

    # Null detection
    nulls = _null_counts(df)
    total_nulls = sum(nulls.values())
    if total_nulls > 0:
        errors.append(f"Found {total_nulls} null values")

    # OHLC ordering: High >= max(Open, Close), Low <= min(Open, Close)
    high_violations = df["High"] < np.maximum(df["Open"], df["Close"])
    low_violations = df["Low"] > np.minimum(df["Open"], df["Close"])
    ohlc_violations = int(high_violations.sum() + low_violations.sum())
    if ohlc_violations > 0:
        errors.append(f"OHLC ordering violations: {ohlc_violations}")

    # Gap detection
    gap_count, max_gap_days = _detect_gaps(df, gap_threshold_days)
    if gap_count > 0:
        errors.append(f"Gaps > {gap_threshold_days} days: {gap_count} (max {max_gap_days} days)")

    # Coverage
    coverage_pct = _compute_coverage(df)
    if coverage_pct < min_coverage:
        errors.append(f"Low coverage: {coverage_pct:.1%} < {min_coverage:.0%}")

    # Outlier flagging
    outlier_count = _count_outliers(df, outlier_threshold)
    if outlier_count > 0:
        errors.append(f"Outlier returns (>{outlier_threshold:.0%}): {outlier_count}")

    # Determine status
    if errors:
        status = "WARN"
    else:
        status = "PASS"

    return StockQualityResult(
        ticker=ticker,
        status=status,
        row_count=len(df),
        date_range=_date_range_str(df),
        null_counts=nulls,
        ohlc_violations=ohlc_violations,
        gap_count=gap_count,
        max_gap_days=max_gap_days,
        outlier_count=outlier_count,
        coverage_pct=round(coverage_pct, 4),
        errors=errors,
    )


def validate_market(
    market_key: str,
    data_dir: Path,
    gap_threshold_days: int = 5,
    outlier_threshold: float = 0.50,
    min_coverage: float = 0.80,
    min_rows: int = 100,
) -> MarketQualityReport:
    """Validate all stocks in a market directory.

    Globs ``data/raw/{market_key}/*.parquet`` and validates each file.

    Args:
        market_key: Market identifier (e.g. "omxs30").
        data_dir: Root data directory.
        gap_threshold_days: Calendar days beyond which a gap is flagged.
        outlier_threshold: Absolute daily return threshold for outlier flagging.
        min_coverage: Minimum business-day coverage fraction before WARN.
        min_rows: Minimum row count before FAIL.

    Returns:
        MarketQualityReport with aggregated results.
    """
    market_dir = data_dir / "raw" / market_key
    parquet_files = sorted(market_dir.glob("*.parquet"))
    LOGGER.info("Validating %d files in %s", len(parquet_files), market_dir)

    results: list[StockQualityResult] = []
    for path in parquet_files:
        result = validate_stock(
            path,
            gap_threshold_days=gap_threshold_days,
            outlier_threshold=outlier_threshold,
            min_coverage=min_coverage,
            min_rows=min_rows,
        )
        results.append(result)
        LOGGER.debug("%s: %s", result.ticker, result.status)

    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    return MarketQualityReport(
        market_key=market_key,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        total_stocks=len(results),
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        stock_results=tuple(results),
    )


def save_report(report: MarketQualityReport, data_dir: Path) -> Path:
    """Write a market quality report as JSON.

    Args:
        report: Market quality report to serialize.
        data_dir: Root data directory (report goes to data/reports/).

    Returns:
        Path to the written JSON file.
    """
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{report.market_key}_quality.json"
    data = asdict(report)
    # Convert tuple to list for JSON serialization
    data["stock_results"] = list(data["stock_results"])
    path.write_text(json.dumps(data, indent=2))
    LOGGER.info("Report saved to %s", path)
    return path


def _validate_index(df: pd.DataFrame) -> list[str]:
    """Check DatetimeIndex integrity and return any error messages."""
    errors: list[str] = []
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex")
    if df.index.name != "Date":
        errors.append(f"Index name is '{df.index.name}', expected 'Date'")
    if not df.index.is_monotonic_increasing:
        errors.append("Index is not sorted ascending")
    if df.index.duplicated().any():
        dupe_count = int(df.index.duplicated().sum())
        errors.append(f"Index has {dupe_count} duplicate dates")
    return errors


def _date_range_str(df: pd.DataFrame) -> str:
    """Format the date range of a DataFrame as a string."""
    if df.empty:
        return ""
    start = pd.Timestamp(df.index.min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(df.index.max()).strftime("%Y-%m-%d")
    return f"{start} to {end}"


def _null_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count null values per OHLCV column."""
    counts: dict[str, int] = {}
    for col in EXPECTED_COLUMNS:
        if col in df.columns:
            n = int(df[col].isna().sum())
            counts[col] = n
    return counts


def _detect_gaps(df: pd.DataFrame, threshold_days: int) -> tuple[int, int]:
    """Detect gaps in the date index exceeding threshold calendar days.

    Args:
        df: DataFrame with DatetimeIndex.
        threshold_days: Calendar days beyond which a gap is flagged.

    Returns:
        Tuple of (gap_count, max_gap_days). Both 0 if no gaps found.
    """
    if len(df) < _MIN_ROWS_FOR_STATS:
        return 0, 0
    deltas = pd.Series(df.index).diff().dropna()
    gap_days = deltas.dt.days  # type: ignore[arg-type]
    large_gaps = gap_days[gap_days > threshold_days]
    if large_gaps.empty:
        return 0, 0
    return len(large_gaps), int(large_gaps.max())


def _compute_coverage(df: pd.DataFrame) -> float:
    """Compute business-day coverage fraction.

    Compares actual row count to the number of business days in the
    date range.

    Returns:
        Coverage fraction (0.0 to 1.0+). Returns 0.0 for empty DataFrames.
    """
    if len(df) < _MIN_ROWS_FOR_STATS:
        return 0.0
    start = pd.Timestamp(df.index.min())
    end = pd.Timestamp(df.index.max())
    expected = len(pd.bdate_range(start, end))
    if expected == 0:
        return 0.0
    return len(df) / expected


def _count_outliers(df: pd.DataFrame, threshold: float) -> int:
    """Count days where absolute return exceeds threshold.

    Args:
        df: DataFrame with a "Close" column.
        threshold: Absolute return threshold (e.g. 0.50 for 50%).

    Returns:
        Number of outlier days.
    """
    if len(df) < _MIN_ROWS_FOR_STATS or "Close" not in df.columns:
        return 0
    returns = df["Close"].pct_change(fill_method=None).dropna().abs()
    return int((returns > threshold).sum())
