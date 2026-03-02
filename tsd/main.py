"""CLI entry point for Trading System Discovery."""

from __future__ import annotations

import logging
import sys

from tsd.config import load_config

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run the main optimization pipeline."""
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    LOGGER.info("Trading System Discovery started")
    return 0


if __name__ == "__main__":
    sys.exit(main())
