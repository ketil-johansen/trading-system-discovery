"""Indicator interface, registry, and parameter metadata loading."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndicatorResult:
    """Result of computing a single indicator.

    Attributes:
        name: Indicator name (e.g. "rsi").
        values: Mapping of output key to computed Series (e.g. {"rsi": Series}).
        params: Parameters used for computation (e.g. {"period": 14}).
    """

    name: str
    values: dict[str, pd.Series]
    params: dict[str, int | float]


@dataclass(frozen=True)
class ParamMeta:
    """Metadata for a single indicator parameter.

    Attributes:
        name: Parameter name (e.g. "period").
        param_type: "int" or "float".
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        default: Default value.
    """

    name: str
    param_type: str
    min_value: int | float
    max_value: int | float
    default: int | float


@dataclass(frozen=True)
class IndicatorMeta:
    """Metadata for a single indicator.

    Attributes:
        name: Indicator name (e.g. "rsi").
        category: Category (e.g. "trend", "momentum").
        params: Tuple of parameter metadata.
    """

    name: str
    category: str
    params: tuple[ParamMeta, ...]


def load_indicator_config(config_dir: Path) -> tuple[IndicatorMeta, ...]:
    """Load indicator metadata from indicators.yaml.

    Args:
        config_dir: Path to the config/ directory containing indicators.yaml.

    Returns:
        Tuple of IndicatorMeta frozen dataclasses.

    Raises:
        FileNotFoundError: If indicators.yaml does not exist.
        ValueError: If the YAML structure is invalid.
    """
    path = config_dir / "indicators.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Invalid indicators.yaml: expected a mapping in {path}"
        raise ValueError(msg)
    indicators: list[IndicatorMeta] = []
    for ind_name, ind_data in data.items():
        if not isinstance(ind_data, dict) or "category" not in ind_data:
            msg = f"Invalid entry '{ind_name}' in {path}: missing 'category'"
            raise ValueError(msg)
        params_raw = ind_data.get("params", {})
        params: list[ParamMeta] = []
        for param_name, param_data in params_raw.items():
            params.append(
                ParamMeta(
                    name=param_name,
                    param_type=param_data["type"],
                    min_value=param_data["min"],
                    max_value=param_data["max"],
                    default=param_data["default"],
                )
            )
        indicators.append(
            IndicatorMeta(
                name=ind_name,
                category=ind_data["category"],
                params=tuple(params),
            )
        )
    return tuple(indicators)


def nan_series(index: pd.Index) -> pd.Series:
    """Return a NaN-filled Series matching the given index.

    Args:
        index: Index to use for the resulting Series.
    """
    return pd.Series(np.nan, index=index)


# Type alias for indicator functions
IndicatorFn = Callable[..., IndicatorResult]

# Cached registry
_REGISTRY: dict[str, IndicatorFn] | None = None


def _get_registry() -> dict[str, IndicatorFn]:
    """Build the indicator registry via lazy imports from category modules.

    Returns:
        Dict mapping indicator names to their compute functions.
    """
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is not None:
        return _REGISTRY

    from tsd.indicators.filters import INDICATORS as filters_indicators  # noqa: PLC0415
    from tsd.indicators.momentum import INDICATORS as momentum_indicators  # noqa: PLC0415
    from tsd.indicators.trend import INDICATORS as trend_indicators  # noqa: PLC0415
    from tsd.indicators.volatility import INDICATORS as volatility_indicators  # noqa: PLC0415
    from tsd.indicators.volume import INDICATORS as volume_indicators  # noqa: PLC0415

    registry: dict[str, IndicatorFn] = {}
    registry.update(trend_indicators)
    registry.update(momentum_indicators)
    registry.update(volatility_indicators)
    registry.update(volume_indicators)
    registry.update(filters_indicators)

    _REGISTRY = registry
    return _REGISTRY


def compute_indicator(name: str, df: pd.DataFrame, params: dict[str, Any]) -> IndicatorResult:
    """Compute an indicator by name.

    Args:
        name: Indicator name (e.g. "rsi", "sma").
        df: DataFrame with OHLCV columns and DatetimeIndex.
        params: Parameter overrides. Empty dict uses function defaults.

    Returns:
        IndicatorResult with computed values.

    Raises:
        KeyError: If the indicator name is not registered.
    """
    registry = _get_registry()
    if name not in registry:
        msg = f"Unknown indicator '{name}'. Available: {sorted(registry.keys())}"
        raise KeyError(msg)
    fn = registry[name]
    return fn(df, **params)


def get_indicator_names() -> list[str]:
    """Return sorted list of all registered indicator names."""
    return sorted(_get_registry().keys())
