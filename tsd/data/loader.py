"""Load cached OHLCV data from Parquet files into memory."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_market_data(market_key: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all cached OHLCV Parquet files for a market.

    Args:
        market_key: Market identifier (e.g. "omxs30").
        data_dir: Root data directory containing raw/{market_key}/.

    Returns:
        Dict mapping ticker to OHLCV DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If the market data directory does not exist.
        ValueError: If no valid Parquet files are found.
    """
    raw_dir = data_dir / "raw" / market_key
    if not raw_dir.exists():
        msg = f"No data directory for market '{market_key}' at {raw_dir}"
        raise FileNotFoundError(msg)

    stocks: dict[str, pd.DataFrame] = {}
    parquet_files = sorted(raw_dir.glob("*.parquet"))

    if not parquet_files:
        msg = f"No Parquet files found in {raw_dir}"
        raise ValueError(msg)

    for path in parquet_files:
        ticker = path.stem
        try:
            df = pd.read_parquet(path)
            if df.empty:
                LOGGER.warning("Empty data for %s, skipping", ticker)
                continue
            stocks[ticker] = df
        except Exception:
            LOGGER.warning("Failed to load %s, skipping", path, exc_info=True)

    LOGGER.info(
        "Loaded %d stocks for market '%s' (%d files)",
        len(stocks),
        market_key,
        len(parquet_files),
    )
    return stocks
