"""CLI script to download OHLCV market data for all constituents."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from tsd.config import load_config
from tsd.data.constituents import load_constituents
from tsd.data.downloader import download_market

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Download OHLCV data for one or all markets."""
    parser = argparse.ArgumentParser(description="Download OHLCV market data")
    parser.add_argument("--market", type=str, default=None, help="Market key to download (default: all)")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (default: 2015-01-01)")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date (default: today)")
    args = parser.parse_args()

    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.market:
        valid_keys = [m.key for m in config.markets]
        if args.market not in valid_keys:
            LOGGER.error("Unknown market '%s'. Valid keys: %s", args.market, valid_keys)
            return 1
        markets = [m for m in config.markets if m.key == args.market]
    else:
        markets = list(config.markets)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for market in markets:
        LOGGER.info("Processing market: %s (%s)", market.name, market.key)
        constituents = load_constituents(market.key, config.data_dir)
        results = download_market(
            market=market,
            constituents=constituents,
            data_dir=config.data_dir,
            start=args.start,
            end=args.end,
            delay=config.download_delay,
        )

        downloaded = sum(1 for r in results if r.success and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)
        failed = sum(1 for r in results if not r.success)
        LOGGER.info("  %s: %d downloaded, %d skipped, %d failed", market.key, downloaded, skipped, failed)

        total_downloaded += downloaded
        total_skipped += skipped
        total_failed += failed

    LOGGER.info(
        "Done. Total: %d downloaded, %d skipped, %d failed",
        total_downloaded,
        total_skipped,
        total_failed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
