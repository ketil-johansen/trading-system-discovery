"""Download daily OHLCV data from yfinance and store as Parquet."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from tsd.config import MarketConfig

LOGGER = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class DownloadResult:
    """Result of downloading a single stock."""

    market_key: str
    ticker: str
    success: bool
    rows: int
    start_date: str | None
    end_date: str | None
    skipped: bool
    error: str | None


def is_up_to_date(parquet_path: Path, max_age_days: int = 3) -> bool:
    """Check if existing Parquet data is fresh enough.

    Reads the last date in the file and compares to today. Returns True
    if the data is within max_age_days of today (accounts for weekends).

    Args:
        parquet_path: Path to a Parquet file with a DatetimeIndex.
        max_age_days: Maximum number of calendar days since last data point.

    Returns:
        True if data is fresh enough, False if stale or file missing.
    """
    if not parquet_path.exists():
        return False
    try:
        df = pd.read_parquet(parquet_path)
        if df.empty:
            return False
        last_date = pd.Timestamp(df.index.max()).date()
        age = (date.today() - last_date).days
        return age <= max_age_days
    except Exception:
        LOGGER.warning("Could not read %s for freshness check", parquet_path, exc_info=True)
        return False


def download_stock(yahoo_ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single stock from yfinance.

    Args:
        yahoo_ticker: Yahoo Finance ticker (e.g. "VOLV-B.ST").
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume.

    Raises:
        ValueError: If download returns empty data.
    """
    df = yf.download(yahoo_ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        msg = f"No data returned for {yahoo_ticker}"
        raise ValueError(msg)
    # yfinance returns MultiIndex columns for single ticker: (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    # Select only OHLCV columns
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        msg = f"Missing columns {missing} in data for {yahoo_ticker}"
        raise ValueError(msg)
    df = df[OHLCV_COLUMNS]
    df.index.name = "Date"
    return df


def save_stock_data(df: pd.DataFrame, market_key: str, ticker: str, data_dir: Path) -> Path:
    """Save OHLCV DataFrame as a Parquet file.

    Args:
        df: DataFrame with OHLCV data and DatetimeIndex.
        market_key: Market identifier (e.g. "omxs30").
        ticker: Stock ticker used as filename (e.g. "VOLV-B").
        data_dir: Root data directory.

    Returns:
        Path to the written Parquet file.
    """
    out_dir = data_dir / "raw" / market_key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.parquet"
    df.to_parquet(path, engine="pyarrow")
    return path


def download_market(
    market: MarketConfig,
    constituents: pd.DataFrame,
    data_dir: Path,
    start: str,
    end: str,
    delay: float = 1.5,
    max_retries: int = 3,
) -> list[DownloadResult]:
    """Download OHLCV data for all constituents in a market.

    Skips stocks whose existing data is already up-to-date. Retries
    transient failures with exponential backoff.

    Args:
        market: Market configuration.
        constituents: DataFrame with yahoo_ticker and ticker columns.
        data_dir: Root data directory.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        delay: Seconds to wait between downloads.
        max_retries: Maximum retry attempts per stock.

    Returns:
        List of DownloadResult for each constituent.
    """
    results: list[DownloadResult] = []
    total = len(constituents)

    for idx, row in constituents.iterrows():
        ticker = str(row["ticker"])
        yahoo_ticker = str(row["yahoo_ticker"])
        position = int(idx) + 1 if isinstance(idx, int) else results.__len__() + 1  # type: ignore[arg-type]

        parquet_path = data_dir / "raw" / market.key / f"{ticker}.parquet"

        # Skip if fresh
        if is_up_to_date(parquet_path):
            LOGGER.info("Skipping %s (%d/%d) — up to date", yahoo_ticker, position, total)
            results.append(
                DownloadResult(
                    market_key=market.key,
                    ticker=ticker,
                    success=True,
                    rows=0,
                    start_date=None,
                    end_date=None,
                    skipped=True,
                    error=None,
                )
            )
            continue

        # Download with retry
        last_error: str | None = None
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                LOGGER.info("Downloading %s (%d/%d), attempt %d", yahoo_ticker, position, total, attempt)
                df = download_stock(yahoo_ticker, start, end)
                save_stock_data(df, market.key, ticker, data_dir)
                results.append(
                    DownloadResult(
                        market_key=market.key,
                        ticker=ticker,
                        success=True,
                        rows=len(df),
                        start_date=str(df.index.min().date()),
                        end_date=str(df.index.max().date()),
                        skipped=False,
                        error=None,
                    )
                )
                success = True
                break
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries:
                    backoff = 2**attempt
                    LOGGER.warning(
                        "Attempt %d failed for %s: %s. Retrying in %ds",
                        attempt,
                        yahoo_ticker,
                        last_error,
                        backoff,
                    )
                    time.sleep(backoff)

        if not success:
            LOGGER.error("Failed to download %s after %d attempts: %s", yahoo_ticker, max_retries, last_error)
            results.append(
                DownloadResult(
                    market_key=market.key,
                    ticker=ticker,
                    success=False,
                    rows=0,
                    start_date=None,
                    end_date=None,
                    skipped=False,
                    error=last_error,
                )
            )

        # Throttle between requests
        if position < total:
            time.sleep(delay)

    return results
