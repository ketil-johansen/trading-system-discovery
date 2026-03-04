"""Trend indicators: SMA, EMA, HMA, Ichimoku, Supertrend."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, IchimokuIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

from tsd.indicators.base import IndicatorFn, IndicatorResult, nan_series


def sma(df: pd.DataFrame, period: int = 20) -> IndicatorResult:
    """Simple Moving Average.

    Args:
        df: DataFrame with 'Close' column.
        period: Window size.
    """
    indicator = SMAIndicator(close=df["Close"], window=period)
    return IndicatorResult(
        name="sma",
        values={"sma": indicator.sma_indicator()},
        params={"period": period},
    )


def ema(df: pd.DataFrame, period: int = 20) -> IndicatorResult:
    """Exponential Moving Average.

    Args:
        df: DataFrame with 'Close' column.
        period: Window size.
    """
    indicator = EMAIndicator(close=df["Close"], window=period)
    return IndicatorResult(
        name="ema",
        values={"ema": indicator.ema_indicator()},
        params={"period": period},
    )


def hma(df: pd.DataFrame, period: int = 20) -> IndicatorResult:
    """Hull Moving Average.

    HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    Args:
        df: DataFrame with 'Close' column.
        period: Window size.
    """
    close = df["Close"]
    half_period = max(int(period / 2), 1)
    sqrt_period = max(int(math.sqrt(period)), 1)

    wma_half = close.rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True,
    )
    wma_full = close.rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True,
    )
    diff = 2 * wma_half - wma_full
    hma_values = diff.rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True,
    )
    return IndicatorResult(
        name="hma",
        values={"hma": hma_values},
        params={"period": period},
    )


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> IndicatorResult:
    """Ichimoku Cloud.

    Args:
        df: DataFrame with 'High' and 'Low' columns.
        tenkan: Tenkan-sen (conversion line) period.
        kijun: Kijun-sen (base line) period.
        senkou_b: Senkou Span B period.
    """
    indicator = IchimokuIndicator(high=df["High"], low=df["Low"], window1=tenkan, window2=kijun, window3=senkou_b)
    return IndicatorResult(
        name="ichimoku",
        values={
            "conversion": indicator.ichimoku_conversion_line(),
            "base": indicator.ichimoku_base_line(),
            "span_a": indicator.ichimoku_a(),
            "span_b": indicator.ichimoku_b(),
        },
        params={"tenkan": tenkan, "kijun": kijun, "senkou_b": senkou_b},
    )


def _supertrend_step(
    i: int,
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
    direction: np.ndarray,  # type: ignore[type-arg]
    st: np.ndarray,  # type: ignore[type-arg]
) -> None:
    """Compute one step of the supertrend algorithm in-place."""
    # Adjust upper band
    if not np.isnan(upper_band.iloc[i - 1]):
        should_tighten = not (upper_band.iloc[i] > upper_band.iloc[i - 1] and direction[i - 1] == -1.0)
        if should_tighten and close.iloc[i - 1] <= upper_band.iloc[i - 1]:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i - 1])

    # Adjust lower band
    if not np.isnan(lower_band.iloc[i - 1]):
        should_tighten = not (lower_band.iloc[i] < lower_band.iloc[i - 1] and direction[i - 1] == 1.0)
        if should_tighten and close.iloc[i - 1] >= lower_band.iloc[i - 1]:
            lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i - 1])

    # Determine direction
    if direction[i - 1] == 1.0:
        direction[i] = -1.0 if close.iloc[i] < lower_band.iloc[i] else 1.0
    elif close.iloc[i] > upper_band.iloc[i]:
        direction[i] = 1.0
    else:
        direction[i] = -1.0

    # Set supertrend value
    st[i] = lower_band.iloc[i] if direction[i] == 1.0 else upper_band.iloc[i]


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> IndicatorResult:
    """Supertrend indicator.

    Uses ATR bands with direction flip logic.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns.
        period: ATR period.
        multiplier: ATR multiplier for band width.
    """
    if len(df) < period:
        nan = nan_series(df.index)
        return IndicatorResult(
            name="supertrend",
            values={"supertrend": nan, "direction": nan.copy()},
            params={"period": period, "multiplier": multiplier},
        )

    atr_indicator = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    atr_values = atr_indicator.average_true_range()

    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + multiplier * atr_values
    lower_band = hl2 - multiplier * atr_values

    n = len(df)
    direction = np.ones(n)
    st = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]):
            continue
        _supertrend_step(i, df["Close"], upper_band, lower_band, direction, st)

    return IndicatorResult(
        name="supertrend",
        values={
            "supertrend": pd.Series(st, index=df.index),
            "direction": pd.Series(direction, index=df.index),
        },
        params={"period": period, "multiplier": multiplier},
    )


INDICATORS: dict[str, IndicatorFn] = {
    "sma": sma,
    "ema": ema,
    "hma": hma,
    "ichimoku": ichimoku,
    "supertrend": supertrend,
}
