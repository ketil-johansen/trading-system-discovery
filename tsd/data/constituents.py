"""Load and refresh index constituent lists."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"ticker", "name", "yahoo_ticker"}


def load_constituents(market_key: str, data_dir: Path) -> pd.DataFrame:
    """Read a constituent CSV for the given market.

    Args:
        market_key: Market identifier (e.g. "omxs30").
        data_dir: Root data directory containing constituents/ subdirectory.

    Returns:
        DataFrame with columns: ticker, name, yahoo_ticker.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = data_dir / "constituents" / f"{market_key}.csv"
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns in {path}: {missing}"
        raise ValueError(msg)
    return df


def save_constituents(df: pd.DataFrame, market_key: str, data_dir: Path) -> Path:
    """Write a constituent DataFrame to CSV.

    Args:
        df: DataFrame with columns: ticker, name, yahoo_ticker.
        market_key: Market identifier (e.g. "omxs30").
        data_dir: Root data directory containing constituents/ subdirectory.

    Returns:
        Path to the written CSV file.
    """
    out_dir = data_dir / "constituents"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{market_key}.csv"
    df.to_csv(path, index=False)
    LOGGER.info("Saved %d constituents to %s", len(df), path)
    return path


def scrape_sp500() -> pd.DataFrame:
    """Scrape current S&P 500 constituents from Wikipedia.

    Returns:
        DataFrame with columns: ticker, name, yahoo_ticker.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    result = pd.DataFrame(
        {
            "ticker": df["Symbol"].str.strip().str.replace(".", "-", regex=False),
            "name": df["Security"].str.strip(),
        }
    )
    result["yahoo_ticker"] = result["ticker"]
    return result.sort_values("ticker").reset_index(drop=True)


def scrape_nasdaq100() -> pd.DataFrame:
    """Scrape current Nasdaq 100 constituents from Wikipedia.

    Returns:
        DataFrame with columns: ticker, name, yahoo_ticker.
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    # Find the table with a "Ticker" column
    df = None
    for table in tables:
        cols_lower = [c.lower() for c in table.columns]
        if "ticker" in cols_lower:
            df = table
            break
    if df is None:
        msg = "Could not find constituents table on Nasdaq-100 Wikipedia page"
        raise ValueError(msg)
    # Normalize column names to find ticker and company
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    ticker_col = "ticker"
    name_col = "company" if "company" in df.columns else "security" if "security" in df.columns else df.columns[1]
    result = pd.DataFrame(
        {
            "ticker": df[ticker_col].astype(str).str.strip(),
            "name": df[name_col].astype(str).str.strip(),
        }
    )
    result["yahoo_ticker"] = result["ticker"]
    return result.sort_values("ticker").reset_index(drop=True)


_SCRAPERS: dict[str, object] = {
    "sp500": scrape_sp500,
    "nasdaq_100": scrape_nasdaq100,
}


def refresh_constituents(market_key: str, data_dir: Path) -> pd.DataFrame:
    """Refresh constituent list for a market.

    If a scraper exists (sp500, nasdaq_100), scrape and save.
    Otherwise, load the existing CSV.

    Args:
        market_key: Market identifier.
        data_dir: Root data directory.

    Returns:
        DataFrame with columns: ticker, name, yahoo_ticker.
    """
    scraper = _SCRAPERS.get(market_key)
    if scraper is not None:
        LOGGER.info("Scraping constituents for %s", market_key)
        try:
            df = scraper()  # type: ignore[operator]
            save_constituents(df, market_key, data_dir)
            return df
        except Exception:
            LOGGER.warning("Scraping failed for %s, falling back to existing CSV", market_key, exc_info=True)
            return load_constituents(market_key, data_dir)
    else:
        LOGGER.info("No scraper for %s, loading existing CSV", market_key)
        return load_constituents(market_key, data_dir)
