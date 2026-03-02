"""Configuration loading from environment variables and YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def env_str(key: str, default: str) -> str:
    """Read a string environment variable with a default."""
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    """Read an integer environment variable with a default."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    return int(raw)


def env_float(key: str, default: float) -> float:
    """Read a float environment variable with a default."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    return float(raw)


def env_bool(key: str, default: bool) -> bool:
    """Read a boolean environment variable with a default.

    Truthy values: "1", "true", "yes" (case-insensitive).
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class MarketConfig:
    """Definition of a single market (index + constituents)."""

    key: str
    name: str
    index_ticker: str
    stock_suffix: str
    expected_constituents: int


@dataclass(frozen=True)
class Config:
    """Top-level application configuration."""

    log_level: str
    data_dir: Path
    results_dir: Path
    config_dir: Path
    population_size: int
    max_generations: int
    n_trials: int
    download_delay: float
    markets: tuple[MarketConfig, ...]


def load_markets(config_dir: Path) -> tuple[MarketConfig, ...]:
    """Load market definitions from markets.yaml.

    Args:
        config_dir: Path to the config/ directory containing markets.yaml.

    Returns:
        Tuple of MarketConfig frozen dataclasses.

    Raises:
        FileNotFoundError: If markets.yaml does not exist.
        ValueError: If the YAML structure is invalid.
    """
    path = config_dir / "markets.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "markets" not in data:
        msg = f"Invalid markets.yaml: expected top-level 'markets' key in {path}"
        raise ValueError(msg)
    markets: list[MarketConfig] = []
    for entry in data["markets"]:
        markets.append(
            MarketConfig(
                key=entry["key"],
                name=entry["name"],
                index_ticker=entry["index_ticker"],
                stock_suffix=entry.get("stock_suffix", ""),
                expected_constituents=entry["expected_constituents"],
            )
        )
    return tuple(markets)


def load_config() -> Config:
    """Build application config from environment variables and YAML files."""
    config_dir = Path(env_str("TSD_CONFIG_DIR", "config"))
    return Config(
        log_level=env_str("TSD_LOG_LEVEL", "INFO"),
        data_dir=Path(env_str("TSD_DATA_DIR", "data")),
        results_dir=Path(env_str("TSD_RESULTS_DIR", "results")),
        config_dir=config_dir,
        population_size=env_int("TSD_POPULATION_SIZE", 100),
        max_generations=env_int("TSD_MAX_GENERATIONS", 50),
        n_trials=env_int("TSD_N_TRIALS", 100),
        download_delay=env_float("TSD_DOWNLOAD_DELAY", 1.5),
        markets=load_markets(config_dir),
    )


def get_market(config: Config, key: str) -> MarketConfig:
    """Look up a market by key.

    Raises:
        ValueError: If no market with the given key exists.
    """
    for m in config.markets:
        if m.key == key:
            return m
    valid = [m.key for m in config.markets]
    msg = f"Unknown market key '{key}'. Valid keys: {valid}"
    raise ValueError(msg)
