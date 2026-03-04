"""CLI script to refresh index constituent lists."""

from __future__ import annotations

import argparse
import logging
import sys

from tsd.config import load_config
from tsd.data.constituents import refresh_constituents

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Refresh constituent lists for one or all markets."""
    parser = argparse.ArgumentParser(description="Refresh index constituent lists")
    parser.add_argument("--market", type=str, default=None, help="Market key to refresh (default: all)")
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

    for market in markets:
        LOGGER.info("Refreshing constituents for %s", market.key)
        df = refresh_constituents(market.key, config.data_dir)
        LOGGER.info("  %s: %d constituents", market.key, len(df))

    LOGGER.info("Done. Refreshed %d market(s).", len(markets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
