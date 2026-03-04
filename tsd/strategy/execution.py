"""Execution timing rules for trade entries and exits.

Provides building-block functions for the backtest evaluator to apply
correct execution timing: entries at next open, limit exits intraday,
indicator/time exits at next open.
"""

from __future__ import annotations

import pandas as pd

# Execution timing constants
ENTRY_TIMING = "next_open"
EXIT_TIMING_LIMIT = "intraday"
EXIT_TIMING_INDICATOR = "next_open"
EXIT_TIMING_TIME = "at_open"


def shift_to_next_open(signals: pd.Series) -> pd.Series:
    """Shift a signal Series forward by 1 bar.

    A signal generated on close of bar T is executed at open of bar T+1.
    The first bar is always False and the last bar's signal is dropped.

    Args:
        signals: Boolean signal Series.

    Returns:
        Shifted boolean Series with same index.
    """
    return signals.shift(1, fill_value=False).astype(bool)


def check_limit_exit(
    high: float,
    low: float,
    open_price: float,
    stop_level: float | None,
    target_level: float | None,
) -> tuple[str | None, float | None]:
    """Check whether stop and/or target were hit intraday.

    Conservative rule: if both stop and target are hit on the same bar,
    stop wins unless the open already exceeds (or equals) the target.

    Args:
        high: Bar high price.
        low: Bar low price.
        open_price: Bar open price.
        stop_level: Stop-loss price level, or None if not active.
        target_level: Take-profit price level, or None if not active.

    Returns:
        Tuple of (exit_type, exit_price) where exit_type is "stop_loss",
        "take_profit", or None, and exit_price is the execution price.
    """
    stop_hit = stop_level is not None and low <= stop_level
    target_hit = target_level is not None and high >= target_level

    if stop_hit and target_hit:
        # Both hit same bar — open exceeds target means TP wins
        if open_price >= target_level:  # type: ignore[operator]
            return ("take_profit", open_price)
        return ("stop_loss", stop_level)

    if target_hit:
        return ("take_profit", target_level)

    if stop_hit:
        return ("stop_loss", stop_level)

    return (None, None)
