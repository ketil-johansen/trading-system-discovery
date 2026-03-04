"""Volatility indicators: ATR, Bollinger Bands, Keltner Channels."""

from __future__ import annotations

import pandas as pd
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel

from tsd.indicators.base import IndicatorFn, IndicatorResult


def atr(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """Average True Range.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns.
        period: ATR lookback period.
    """
    indicator = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=period
    )
    return IndicatorResult(
        name="atr",
        values={"atr": indicator.average_true_range()},
        params={"period": period},
    )


def bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> IndicatorResult:
    """Bollinger Bands.

    Args:
        df: DataFrame with 'Close' column.
        period: Moving average period.
        std_dev: Standard deviation multiplier.
    """
    indicator = BollingerBands(close=df["Close"], window=period, window_dev=std_dev)
    return IndicatorResult(
        name="bollinger",
        values={
            "mavg": indicator.bollinger_mavg(),
            "upper": indicator.bollinger_hband(),
            "lower": indicator.bollinger_lband(),
            "pband": indicator.bollinger_pband(),
            "wband": indicator.bollinger_wband(),
        },
        params={"period": period, "std_dev": std_dev},
    )


def keltner(
    df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5
) -> IndicatorResult:
    """Keltner Channels.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns.
        ema_period: EMA period for the middle band.
        atr_period: ATR period for band width.
        multiplier: ATR multiplier.
    """
    indicator = KeltnerChannel(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=ema_period,
        window_atr=atr_period,
        multiplier=multiplier,
    )
    return IndicatorResult(
        name="keltner",
        values={
            "mband": indicator.keltner_channel_mband(),
            "upper": indicator.keltner_channel_hband(),
            "lower": indicator.keltner_channel_lband(),
            "pband": indicator.keltner_channel_pband(),
            "wband": indicator.keltner_channel_wband(),
        },
        params={"ema_period": ema_period, "atr_period": atr_period, "multiplier": multiplier},
    )


INDICATORS: dict[str, IndicatorFn] = {
    "atr": atr,
    "bollinger": bollinger,
    "keltner": keltner,
}
