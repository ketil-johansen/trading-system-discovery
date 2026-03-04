"""Regime filters: Price vs MA, Volatility Regime."""

from __future__ import annotations

import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

from tsd.indicators.base import IndicatorFn, IndicatorResult, nan_series


def price_vs_ma(df: pd.DataFrame, ma_period: int = 200) -> IndicatorResult:
    """Price vs Moving Average filter.

    Returns 1.0 when price is above MA, 0.0 when below.

    Args:
        df: DataFrame with 'Close' column.
        ma_period: Moving average period.
    """
    indicator = SMAIndicator(close=df["Close"], window=ma_period)
    sma_values = indicator.sma_indicator()
    filter_values = (df["Close"] > sma_values).astype(float)
    return IndicatorResult(
        name="price_vs_ma",
        values={"filter": filter_values},
        params={"ma_period": ma_period},
    )


def volatility_regime(
    df: pd.DataFrame,
    atr_period: int = 14,
    lookback: int = 60,
    low_threshold: float = 0.8,
    high_threshold: float = 1.2,
) -> IndicatorResult:
    """Volatility regime filter based on ATR ratio.

    Classifies market into low (0.0), normal (0.5), or high (1.0) volatility.
    The ratio is ATR / rolling_mean(ATR, lookback).

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns.
        atr_period: ATR calculation period.
        lookback: Rolling mean lookback for ATR normalization.
        low_threshold: Ratio below this is low volatility.
        high_threshold: Ratio above this is high volatility.
    """
    if len(df) < atr_period:
        nan = nan_series(df.index)
        return IndicatorResult(
            name="volatility_regime",
            values={"regime": nan, "ratio": nan.copy()},
            params={
                "atr_period": atr_period,
                "lookback": lookback,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
            },
        )

    atr_indicator = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=atr_period)
    atr_values = atr_indicator.average_true_range()
    atr_mean = atr_values.rolling(window=lookback).mean()
    ratio = atr_values / atr_mean

    regime = pd.Series(0.5, index=df.index)
    regime[ratio < low_threshold] = 0.0
    regime[ratio > high_threshold] = 1.0
    # NaN where ratio is NaN
    regime[ratio.isna()] = float("nan")

    return IndicatorResult(
        name="volatility_regime",
        values={"regime": regime, "ratio": ratio},
        params={
            "atr_period": atr_period,
            "lookback": lookback,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
    )


INDICATORS: dict[str, IndicatorFn] = {
    "price_vs_ma": price_vs_ma,
    "volatility_regime": volatility_regime,
}
