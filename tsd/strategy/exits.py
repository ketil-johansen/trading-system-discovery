"""Exit types: stop loss, take profit, trailing stop, time-based.

Computes exit levels and signals for all 3 categories. These produce
the raw exit information; the evaluator (005) walks through bars and
applies them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tsd.strategy.genome import (
    BreakevenConfig,
    ChandelierConfig,
    IndicatorExitGene,
    OutputMeta,
    StopLossConfig,
    TakeProfitConfig,
    TimeExitGene,
    TrailingStopConfig,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category 1 — Limit-based exits (return price levels)
# ---------------------------------------------------------------------------


def compute_stop_loss_level(
    entry_price: float,
    config: StopLossConfig,
    atr_at_entry: float,
) -> float:
    """Compute fixed stop-loss price level.

    Args:
        entry_price: Trade entry price.
        config: Stop-loss configuration.
        atr_at_entry: ATR value at the entry bar.

    Returns:
        Stop-loss price level (below entry for long positions).
    """
    if config.mode == "atr":
        return entry_price - config.atr_multiple * atr_at_entry
    # percent mode
    return entry_price * (1.0 - config.percent / 100.0)


def compute_take_profit_level(
    entry_price: float,
    config: TakeProfitConfig,
    atr_at_entry: float,
) -> float:
    """Compute fixed take-profit price level.

    Args:
        entry_price: Trade entry price.
        config: Take-profit configuration.
        atr_at_entry: ATR value at the entry bar.

    Returns:
        Take-profit price level (above entry for long positions).
    """
    if config.mode == "atr":
        return entry_price + config.atr_multiple * atr_at_entry
    # percent mode
    return entry_price * (1.0 + config.percent / 100.0)


def compute_trailing_stop_levels(
    entry_price: float,
    config: TrailingStopConfig,
    highs: pd.Series,
    atr: pd.Series,
) -> pd.Series:
    """Compute trailing stop levels per bar.

    The trailing stop ratchets up (never down) and activates only after
    price has moved above the activation threshold from entry.

    Args:
        entry_price: Trade entry price.
        config: Trailing stop configuration.
        highs: High prices per bar.
        atr: ATR values per bar.

    Returns:
        Series of trailing stop levels. NaN before activation.
    """
    activation_level = entry_price * (1.0 + config.activation_percent / 100.0)
    levels = pd.Series(np.nan, index=highs.index)
    activated = False
    max_high = entry_price

    for i in range(len(highs)):
        current_high = highs.iloc[i]
        max_high = max(max_high, current_high)

        if not activated and max_high >= activation_level:
            activated = True

        if activated:
            if config.mode == "atr":
                trail = config.atr_multiple * atr.iloc[i]
            else:
                trail = max_high * config.percent / 100.0
            stop = max_high - trail
            if i > 0 and not np.isnan(levels.iloc[i - 1]):
                stop = max(stop, levels.iloc[i - 1])
            levels.iloc[i] = stop

    return levels


def compute_chandelier_levels(
    config: ChandelierConfig,
    highs: pd.Series,
    atr: pd.Series,
    entry_bar: int,
) -> pd.Series:
    """Compute chandelier exit levels per bar.

    Chandelier exit = highest high since entry - N × ATR.

    Args:
        config: Chandelier exit configuration.
        highs: High prices per bar.
        atr: ATR values per bar.
        entry_bar: iloc index of the entry bar.

    Returns:
        Series of chandelier levels. NaN before entry_bar.
    """
    levels = pd.Series(np.nan, index=highs.index)
    highest_high = highs.iloc[entry_bar]

    for i in range(entry_bar, len(highs)):
        current_high = highs.iloc[i]
        highest_high = max(highest_high, current_high)
        levels.iloc[i] = highest_high - config.atr_multiple * atr.iloc[i]

    return levels


def compute_breakeven_level(
    entry_price: float,
    config: BreakevenConfig,
    highs: pd.Series,
    atr_at_entry: float,
) -> pd.Series:
    """Compute breakeven stop level per bar.

    Stop moves to entry price once the trigger profit is reached.

    Args:
        entry_price: Trade entry price.
        config: Breakeven configuration.
        highs: High prices per bar.
        atr_at_entry: ATR value at the entry bar.

    Returns:
        Series of breakeven levels. NaN until trigger profit reached,
        then entry_price.
    """
    if config.mode == "atr":
        trigger_level = entry_price + config.trigger_atr_multiple * atr_at_entry
    else:
        trigger_level = entry_price * (1.0 + config.trigger_percent / 100.0)

    levels = pd.Series(np.nan, index=highs.index)
    triggered = False

    for i in range(len(highs)):
        if not triggered and highs.iloc[i] >= trigger_level:
            triggered = True
        if triggered:
            levels.iloc[i] = entry_price

    return levels


# ---------------------------------------------------------------------------
# Category 2 — Indicator-based exits (return bool Series)
# ---------------------------------------------------------------------------


def generate_indicator_exit_signals(
    exit_genes: tuple[IndicatorExitGene, ...],
    df: pd.DataFrame,
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
) -> pd.Series:
    """Generate combined indicator exit signals.

    Enabled exit genes are OR-ed together. The opposite_entry flag is
    noted but handled by the evaluator, not here.

    Args:
        exit_genes: Tuple of indicator exit gene slots.
        df: OHLCV DataFrame.
        indicator_outputs: Output metadata for comparison mode selection.

    Returns:
        Boolean Series indicating indicator exit signals.
    """
    from tsd.strategy.signals import apply_condition  # noqa: PLC0415

    signals: list[pd.Series] = []
    for gene in exit_genes:
        if not gene.enabled:
            continue

        from tsd.indicators.base import compute_indicator  # noqa: PLC0415

        result = compute_indicator(gene.indicator_name, df, gene.params)
        output_series = result.values[gene.output_key]

        # Find output metadata
        outputs = indicator_outputs.get(gene.indicator_name, ())
        output_meta = None
        for om in outputs:
            if hasattr(om, "key") and om.key == gene.output_key:
                output_meta = om
                break

        if output_meta is not None and hasattr(output_meta, "output_type") and output_meta.output_type == "price_level":
            signal = apply_condition(df["Close"], gene.comparison, output_series)
        else:
            signal = apply_condition(output_series, gene.comparison, gene.threshold)

        signals.append(signal)

    if not signals:
        return pd.Series(False, index=df.index, dtype=bool)

    # OR all exit signals
    combined = signals[0]
    for s in signals[1:]:
        combined = combined | s
    return combined.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Category 3 — Time/calendar-based exits (return bool Series)
# ---------------------------------------------------------------------------


def generate_time_exit_signal(
    config: TimeExitGene,
    entry_bar: int,
    df: pd.DataFrame,
) -> pd.Series:
    """Generate time-based exit signals.

    Checks max holding days, specific weekday, end-of-week (Friday),
    end-of-month, and stagnation (exit if price moved < threshold%
    after N days).

    Args:
        config: Time exit configuration.
        entry_bar: iloc index of the entry bar.
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        Boolean Series marking bars where time exit triggers.
    """
    signals = pd.Series(False, index=df.index, dtype=bool)

    if config.max_days_enabled:
        exit_bar = entry_bar + config.max_days
        if exit_bar < len(df):
            signals.iloc[exit_bar] = True

    if config.weekday_exit_enabled:
        _apply_weekday_exit(signals, config.weekday, entry_bar)

    if config.eow_enabled:
        _apply_weekday_exit(signals, 4, entry_bar)  # noqa: PLR2004

    if config.eom_enabled:
        _apply_eom_exit(signals, df, entry_bar)

    if config.stagnation_enabled:
        _apply_stagnation_exit(
            signals,
            df,
            entry_bar,
            config.stagnation_days,
            config.stagnation_threshold,
        )

    return signals


def _apply_weekday_exit(signals: pd.Series, weekday: int, entry_bar: int) -> None:
    """Mark bars matching a specific weekday after entry."""
    for i in range(entry_bar + 1, len(signals)):
        if hasattr(signals.index[i], "weekday") and signals.index[i].weekday() == weekday:
            signals.iloc[i] = True


def _apply_eom_exit(signals: pd.Series, df: pd.DataFrame, entry_bar: int) -> None:
    """Mark the last trading day of each month after entry."""
    for i in range(entry_bar + 1, len(df)):
        if i + 1 >= len(df):
            signals.iloc[i] = True
        elif df.index[i].month != df.index[i + 1].month:
            signals.iloc[i] = True


def _apply_stagnation_exit(
    signals: pd.Series,
    df: pd.DataFrame,
    entry_bar: int,
    stagnation_days: int,
    stagnation_threshold: float,
) -> None:
    """Check stagnation on day N and exit at N+1 if price moved < threshold%."""
    check_bar = entry_bar + stagnation_days
    if check_bar >= len(df):
        return
    entry_price = df["Close"].iloc[entry_bar]
    check_price = df["Close"].iloc[check_bar]
    pct_move = abs(check_price - entry_price) / entry_price * 100.0
    if pct_move < stagnation_threshold:
        exit_bar = check_bar + 1
        if exit_bar < len(df):
            signals.iloc[exit_bar] = True
