"""CLI script to run data quality checks on downloaded market data."""

from __future__ import annotations

import argparse
import logging
import sys

from tsd.config import load_config
from tsd.data.quality import save_report, validate_market

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run data quality validation for one or all markets."""
    parser = argparse.ArgumentParser(description="Check data quality for downloaded market data")
    parser.add_argument("--market", type=str, default=None, help="Market key to validate (default: all)")
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
        market_keys = [args.market]
    else:
        market_keys = [m.key for m in config.markets]

    for market_key in market_keys:
        LOGGER.info("Validating market: %s", market_key)
        report = validate_market(
            market_key=market_key,
            data_dir=config.data_dir,
            gap_threshold_days=config.quality_gap_threshold_days,
            outlier_threshold=config.quality_outlier_threshold,
            min_coverage=config.quality_min_coverage,
            min_rows=config.quality_min_rows,
        )
        path = save_report(report, config.data_dir)
        LOGGER.info(
            "  %s: %d stocks — %d PASS, %d WARN, %d FAIL — report: %s",
            market_key,
            report.total_stocks,
            report.pass_count,
            report.warn_count,
            report.fail_count,
            path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
