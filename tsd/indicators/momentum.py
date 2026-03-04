"""Momentum indicators: RSI, Stochastic, MACD, Williams %R."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD as MACDIndicator

from tsd.indicators.base import IndicatorFn, IndicatorResult


def rsi(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """Relative Strength Index.

    Args:
        df: DataFrame with 'Close' column.
        period: RSI lookback period.
    """
    indicator = RSIIndicator(close=df["Close"], window=period)
    return IndicatorResult(
        name="rsi",
        values={"rsi": indicator.rsi()},
        params={"period": period},
    )


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> IndicatorResult:
    """Stochastic Oscillator.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns.
        k_period: %K lookback period.
        d_period: %D smoothing period.
        smooth_k: %K smoothing period.
    """
    indicator = StochasticOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=k_period,
        smooth_window=d_period,
    )
    k_values = indicator.stoch()
    # Apply additional %K smoothing if smooth_k > 1
    if smooth_k > 1:
        k_values = k_values.rolling(window=smooth_k).mean()
    d_values = indicator.stoch_signal()
    return IndicatorResult(
        name="stochastic",
        values={"k": k_values, "d": d_values},
        params={"k_period": k_period, "d_period": d_period, "smooth_k": smooth_k},
    )


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorResult:
    """Moving Average Convergence Divergence.

    Args:
        df: DataFrame with 'Close' column.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.
    """
    indicator = MACDIndicator(close=df["Close"], window_fast=fast, window_slow=slow, window_sign=signal)
    return IndicatorResult(
        name="macd",
        values={
            "macd": indicator.macd(),
            "signal": indicator.macd_signal(),
            "histogram": indicator.macd_diff(),
        },
        params={"fast": fast, "slow": slow, "signal": signal},
    )


def williams_r(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """Williams %R.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns.
        period: Lookback period.
    """
    indicator = WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=period)
    return IndicatorResult(
        name="williams_r",
        values={"williams_r": indicator.williams_r()},
        params={"period": period},
    )


INDICATORS: dict[str, IndicatorFn] = {
    "rsi": rsi,
    "stochastic": stochastic,
    "macd": macd,
    "williams_r": williams_r,
}
