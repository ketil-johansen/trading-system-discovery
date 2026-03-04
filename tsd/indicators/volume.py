"""Volume indicators: OBV, Chaikin Money Flow."""

from __future__ import annotations

import pandas as pd
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator

from tsd.indicators.base import IndicatorFn, IndicatorResult


def obv(df: pd.DataFrame) -> IndicatorResult:
    """On Balance Volume.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns.
    """
    indicator = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
    return IndicatorResult(
        name="obv",
        values={"obv": indicator.on_balance_volume()},
        params={},
    )


def cmf(df: pd.DataFrame, period: int = 20) -> IndicatorResult:
    """Chaikin Money Flow.

    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns.
        period: Lookback period.
    """
    indicator = ChaikinMoneyFlowIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=period
    )
    return IndicatorResult(
        name="cmf",
        values={"cmf": indicator.chaikin_money_flow()},
        params={"period": period},
    )


INDICATORS: dict[str, IndicatorFn] = {
    "obv": obv,
    "cmf": cmf,
}
