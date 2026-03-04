"""Signal generation from strategy genome.

Converts a StrategyGenome and OHLCV DataFrame into boolean entry signals
by computing indicators, applying comparisons, and combining with AND/OR.
"""

from __future__ import annotations

import logging

import pandas as pd

from tsd.indicators.base import compute_indicator
from tsd.strategy.genome import (
    FilterGene,
    IndicatorGene,
    OutputMeta,
    StrategyGenome,
)

LOGGER = logging.getLogger(__name__)


def apply_condition(
    series: pd.Series,
    comparison: str,
    threshold: float | pd.Series,
) -> pd.Series:
    """Apply a comparison condition to a Series.

    Args:
        series: Input numeric Series.
        comparison: One of "GT", "LT", "CROSS_ABOVE", "CROSS_BELOW".
        threshold: Comparison value (scalar or Series).

    Returns:
        Boolean Series indicating where the condition is met.

    Raises:
        ValueError: If comparison is not recognized.
    """
    if comparison == "GT":
        return (series > threshold).fillna(False).astype(bool)
    if comparison == "LT":
        return (series < threshold).fillna(False).astype(bool)
    if comparison == "CROSS_ABOVE":
        prev = series.shift(1)
        prev_thresh: float | pd.Series = threshold.shift(1) if isinstance(threshold, pd.Series) else threshold
        crossed = (series > threshold) & (prev <= prev_thresh)
        return crossed.fillna(False).astype(bool)
    if comparison == "CROSS_BELOW":
        prev = series.shift(1)
        prev_thresh_b: float | pd.Series = threshold.shift(1) if isinstance(threshold, pd.Series) else threshold
        crossed = (series < threshold) & (prev >= prev_thresh_b)
        return crossed.fillna(False).astype(bool)

    msg = f"Unknown comparison '{comparison}'. Expected GT, LT, CROSS_ABOVE, or CROSS_BELOW."
    raise ValueError(msg)


def generate_entry_signals(
    genome: StrategyGenome,
    df: pd.DataFrame,
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
) -> pd.Series:
    """Generate combined entry signals from a genome.

    Evaluates each enabled entry indicator slot, combines them using
    AND/OR logic, then gates through enabled filters.

    Args:
        genome: Strategy genome with entry indicators and filters.
        df: OHLCV DataFrame with DatetimeIndex.
        indicator_outputs: Mapping of indicator name to output metadata
            tuples. Keeps the function pure — no YAML loading inside.

    Returns:
        Boolean Series aligned with df.index indicating entry signals.
    """
    slot_signals: list[pd.Series] = []
    for gene in genome.entry_indicators:
        if not gene.enabled:
            continue
        signal = _evaluate_entry_slot(gene, df, indicator_outputs)
        slot_signals.append(signal)

    if not slot_signals:
        return pd.Series(False, index=df.index, dtype=bool)

    combined = _combine_signals(slot_signals, genome.combination_logic)

    # Apply filters
    for fgene in genome.filters:
        if not fgene.enabled:
            continue
        combined = _apply_filter(fgene, df, combined)

    return combined


def _evaluate_entry_slot(
    gene: IndicatorGene,
    df: pd.DataFrame,
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
) -> pd.Series:
    """Evaluate a single entry indicator slot.

    Computes the indicator, looks up the output type, and applies the
    appropriate comparison mode.

    Args:
        gene: Entry indicator gene.
        df: OHLCV DataFrame.
        indicator_outputs: Output metadata for all indicators.

    Returns:
        Boolean Series for this slot.
    """
    result = compute_indicator(gene.indicator_name, df, gene.params)
    output_series = result.values[gene.output_key]

    # Find output metadata for comparison mode
    output_meta = _find_output_meta(gene.indicator_name, gene.output_key, indicator_outputs)

    if output_meta is not None and output_meta.output_type == "price_level":
        # Compare Close against indicator line
        return apply_condition(df["Close"], gene.comparison, output_series)

    # oscillator, binary, direction: compare indicator against threshold
    return apply_condition(output_series, gene.comparison, gene.threshold)


def _find_output_meta(
    indicator_name: str,
    output_key: str,
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
) -> OutputMeta | None:
    """Find OutputMeta for a specific indicator output key."""
    outputs = indicator_outputs.get(indicator_name, ())
    for om in outputs:
        if om.key == output_key:
            return om
    return None


def _combine_signals(signals: list[pd.Series], logic: str) -> pd.Series:
    """Combine multiple boolean signal Series using AND or OR logic."""
    if logic == "AND":
        combined = signals[0]
        for s in signals[1:]:
            combined = combined & s
        return combined.fillna(False).astype(bool)

    # OR
    combined = signals[0]
    for s in signals[1:]:
        combined = combined | s
    return combined.fillna(False).astype(bool)


def _apply_filter(gene: FilterGene, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """Gate entry signals through a filter indicator.

    Filter output >= 0.5 means pass (signal allowed).

    Args:
        gene: Filter gene with filter name and params.
        df: OHLCV DataFrame.
        signals: Current entry signals.

    Returns:
        Filtered boolean signal Series.
    """
    result = compute_indicator(gene.filter_name, df, gene.params)
    # Use the first output key from the filter
    filter_key = next(iter(result.values))
    filter_series = result.values[filter_key]
    # Filter passes where output >= 0.5
    filter_mask = (filter_series >= 0.5).fillna(False)  # noqa: PLR2004
    return (signals & filter_mask).astype(bool)
