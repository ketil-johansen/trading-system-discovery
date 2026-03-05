"""Exit types: stop loss, take profit, trailing stop, time-based.

Computes exit levels and signals for all 3 categories. These produce
the raw exit information; the evaluator (005) walks through bars and
applies them.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
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
    highs: npt.NDArray[np.float64],
    atr: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute trailing stop levels per bar.

    The trailing stop ratchets up (never down) and activates only after
    price has moved above the activation threshold from entry.

    Args:
        entry_price: Trade entry price.
        config: Trailing stop configuration.
        highs: High prices per bar (numpy array).
        atr: ATR values per bar (numpy array).

    Returns:
        Array of trailing stop levels. NaN before activation.
    """
    n = len(highs)
    activation_level = entry_price * (1.0 + config.activation_percent / 100.0)
    levels = np.full(n, np.nan)
    activated = False
    max_high = entry_price

    for i in range(n):
        current_high = highs[i]
        max_high = max(max_high, current_high)

        if not activated and max_high >= activation_level:
            activated = True

        if activated:
            if config.mode == "atr":
                trail = config.atr_multiple * atr[i]
            else:
                trail = max_high * config.percent / 100.0
            stop = max_high - trail
            if i > 0 and not np.isnan(levels[i - 1]):
                stop = max(stop, levels[i - 1])
            levels[i] = stop

    return levels


def compute_chandelier_levels(
    config: ChandelierConfig,
    highs: npt.NDArray[np.float64],
    atr: npt.NDArray[np.float64],
    entry_bar: int,
) -> npt.NDArray[np.float64]:
    """Compute chandelier exit levels per bar.

    Chandelier exit = highest high since entry - N × ATR.

    Args:
        config: Chandelier exit configuration.
        highs: High prices per bar (numpy array).
        atr: ATR values per bar (numpy array).
        entry_bar: iloc index of the entry bar.

    Returns:
        Array of chandelier levels. NaN before entry_bar.
    """
    n = len(highs)
    levels = np.full(n, np.nan)
    highest_high = highs[entry_bar]

    for i in range(entry_bar, n):
        current_high = highs[i]
        highest_high = max(highest_high, current_high)
        levels[i] = highest_high - config.atr_multiple * atr[i]

    return levels


def compute_breakeven_level(
    entry_price: float,
    config: BreakevenConfig,
    highs: npt.NDArray[np.float64],
    atr_at_entry: float,
) -> npt.NDArray[np.float64]:
    """Compute breakeven stop level per bar.

    Stop moves to entry price once the trigger profit is reached.

    Args:
        entry_price: Trade entry price.
        config: Breakeven configuration.
        highs: High prices per bar (numpy array).
        atr_at_entry: ATR value at the entry bar.

    Returns:
        Array of breakeven levels. NaN until trigger profit reached,
        then entry_price.
    """
    if config.mode == "atr":
        trigger_level = entry_price + config.trigger_atr_multiple * atr_at_entry
    else:
        trigger_level = entry_price * (1.0 + config.trigger_percent / 100.0)

    n = len(highs)
    levels = np.full(n, np.nan)
    triggered = False

    for i in range(n):
        if not triggered and highs[i] >= trigger_level:
            triggered = True
        if triggered:
            levels[i] = entry_price

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
) -> npt.NDArray[np.bool_]:
    """Generate time-based exit signals.

    Checks max holding days, specific weekday, end-of-week (Friday),
    end-of-month, and stagnation (exit if price moved < threshold%
    after N days).

    Args:
        config: Time exit configuration.
        entry_bar: iloc index of the entry bar.
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        Boolean array marking bars where time exit triggers.
    """
    n = len(df)
    signals = np.zeros(n, dtype=np.bool_)

    if config.max_days_enabled:
        exit_bar = entry_bar + config.max_days
        if exit_bar < n:
            signals[exit_bar] = True

    if config.weekday_exit_enabled:
        _apply_weekday_exit(signals, df, config.weekday, entry_bar)

    if config.eow_enabled:
        _apply_weekday_exit(signals, df, 4, entry_bar)  # noqa: PLR2004

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


def _apply_weekday_exit(signals: npt.NDArray[np.bool_], df: pd.DataFrame, weekday: int, entry_bar: int) -> None:
    """Mark bars matching a specific weekday after entry."""
    for i in range(entry_bar + 1, len(signals)):
        if hasattr(df.index[i], "weekday") and df.index[i].weekday() == weekday:
            signals[i] = True


def _apply_eom_exit(signals: npt.NDArray[np.bool_], df: pd.DataFrame, entry_bar: int) -> None:
    """Mark the last trading day of each month after entry."""
    n = len(df)
    for i in range(entry_bar + 1, n):
        if i + 1 >= n:
            signals[i] = True
        elif df.index[i].month != df.index[i + 1].month:
            signals[i] = True


def _apply_stagnation_exit(
    signals: npt.NDArray[np.bool_],
    df: pd.DataFrame,
    entry_bar: int,
    stagnation_days: int,
    stagnation_threshold: float,
) -> None:
    """Check stagnation on day N and exit at N+1 if price moved < threshold%."""
    check_bar = entry_bar + stagnation_days
    if check_bar >= len(df):
        return
    close_arr = df["Close"].to_numpy()
    entry_price = close_arr[entry_bar]
    check_price = close_arr[check_bar]
    pct_move = abs(check_price - entry_price) / entry_price * 100.0
    if pct_move < stagnation_threshold:
        exit_bar = check_bar + 1
        if exit_bar < len(df):
            signals[exit_bar] = True
