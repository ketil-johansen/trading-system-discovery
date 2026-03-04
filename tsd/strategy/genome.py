"""Strategy DNA encoding as a genome structure.

Encodes a complete trading strategy as a fixed-length chromosome with
parameter genes and binary switches, ready for GA optimization.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tsd.indicators.base import ParamMeta, load_indicator_config

LOGGER = logging.getLogger(__name__)

# Threshold for converting continuous [0, 1] gene values to booleans.
_BOOL_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputMeta:
    """Metadata for a single indicator output.

    Attributes:
        key: Output key name (e.g. "rsi", "k").
        output_type: One of "price_level", "oscillator", "binary", "direction".
        threshold_min: Minimum threshold value (None for price_level).
        threshold_max: Maximum threshold value (None for price_level).
    """

    key: str
    output_type: str
    threshold_min: float | None = None
    threshold_max: float | None = None


# ---------------------------------------------------------------------------
# Gene dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorGene:
    """One entry indicator slot in the genome.

    Attributes:
        enabled: Whether this slot is active.
        indicator_name: Indicator name from registry.
        output_key: Which output of the indicator to use.
        comparison: Comparison operator (GT, LT, CROSS_ABOVE, CROSS_BELOW).
        threshold: Comparison threshold value.
        params: Indicator parameter overrides.
    """

    enabled: bool
    indicator_name: str
    output_key: str
    comparison: str
    threshold: float
    params: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class StopLossConfig:
    """Stop-loss exit configuration.

    Attributes:
        enabled: Whether stop-loss is active.
        mode: "percent" or "atr".
        percent: Stop distance as percentage of entry price.
        atr_multiple: Stop distance as ATR multiple.
    """

    enabled: bool
    mode: str
    percent: float
    atr_multiple: float


@dataclass(frozen=True)
class TakeProfitConfig:
    """Take-profit exit configuration.

    Attributes:
        enabled: Whether take-profit is active.
        mode: "percent" or "atr".
        percent: Target distance as percentage of entry price.
        atr_multiple: Target distance as ATR multiple.
    """

    enabled: bool
    mode: str
    percent: float
    atr_multiple: float


@dataclass(frozen=True)
class TrailingStopConfig:
    """Trailing stop exit configuration.

    Attributes:
        enabled: Whether trailing stop is active.
        mode: "percent" or "atr".
        percent: Trail distance as percentage.
        atr_multiple: Trail distance as ATR multiple.
        activation_percent: Profit threshold to activate trailing stop.
    """

    enabled: bool
    mode: str
    percent: float
    atr_multiple: float
    activation_percent: float


@dataclass(frozen=True)
class ChandelierConfig:
    """Chandelier exit configuration.

    Attributes:
        enabled: Whether chandelier exit is active.
        atr_multiple: ATR multiple for chandelier distance.
    """

    enabled: bool
    atr_multiple: float


@dataclass(frozen=True)
class BreakevenConfig:
    """Breakeven stop configuration.

    Attributes:
        enabled: Whether breakeven stop is active.
        mode: "percent" or "atr".
        trigger_percent: Profit percentage to trigger breakeven.
        trigger_atr_multiple: Profit as ATR multiple to trigger breakeven.
    """

    enabled: bool
    mode: str
    trigger_percent: float
    trigger_atr_multiple: float


@dataclass(frozen=True)
class LimitExitGene:
    """All Category 1 (limit-based) exit configurations.

    Attributes:
        stop_loss: Stop-loss configuration.
        take_profit: Take-profit configuration.
        trailing_stop: Trailing stop configuration.
        chandelier: Chandelier exit configuration.
        breakeven: Breakeven stop configuration.
    """

    stop_loss: StopLossConfig
    take_profit: TakeProfitConfig
    trailing_stop: TrailingStopConfig
    chandelier: ChandelierConfig
    breakeven: BreakevenConfig


@dataclass(frozen=True)
class IndicatorExitGene:
    """Category 2 indicator-based exit slot.

    Attributes:
        enabled: Whether this exit slot is active.
        indicator_name: Indicator name from registry.
        output_key: Which output of the indicator to use.
        comparison: Comparison operator.
        threshold: Comparison threshold value.
        params: Indicator parameter overrides.
        opposite_entry: Exit when entry condition reverses.
    """

    enabled: bool
    indicator_name: str
    output_key: str
    comparison: str
    threshold: float
    params: dict[str, int | float] = field(default_factory=dict)
    opposite_entry: bool = False


@dataclass(frozen=True)
class TimeExitGene:
    """Category 3 time/calendar-based exit configuration.

    Attributes:
        max_days_enabled: Whether max holding days exit is active.
        max_days: Maximum number of days to hold a position.
        weekday_exit_enabled: Whether weekday exit is active.
        weekday: Day of week to exit (0=Monday, 4=Friday).
        eow_enabled: Whether end-of-week (Friday) exit is active.
        eom_enabled: Whether end-of-month exit is active.
        stagnation_enabled: Whether stagnation exit is active.
        stagnation_days: Days to check for stagnation.
        stagnation_threshold: Minimum percent move to avoid stagnation exit.
    """

    max_days_enabled: bool
    max_days: int
    weekday_exit_enabled: bool
    weekday: int
    eow_enabled: bool
    eom_enabled: bool
    stagnation_enabled: bool
    stagnation_days: int
    stagnation_threshold: float


@dataclass(frozen=True)
class FilterGene:
    """Regime filter slot.

    Attributes:
        enabled: Whether this filter is active.
        filter_name: Filter name from registry.
        params: Filter parameter overrides.
    """

    enabled: bool
    filter_name: str
    params: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyGenome:
    """Complete trading strategy encoded as a genome.

    Attributes:
        entry_indicators: Entry indicator slots.
        combination_logic: How to combine entry signals ("AND" or "OR").
        limit_exits: Category 1 limit-based exits.
        indicator_exits: Category 2 indicator-based exit slots.
        time_exits: Category 3 time/calendar-based exits.
        filters: Regime filter slots.
    """

    entry_indicators: tuple[IndicatorGene, ...]
    combination_logic: str
    limit_exits: LimitExitGene
    indicator_exits: tuple[IndicatorExitGene, ...]
    time_exits: TimeExitGene
    filters: tuple[FilterGene, ...]


# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyMeta:
    """Metadata describing the strategy parameter space.

    Attributes:
        num_entry_slots: Number of entry indicator slots.
        num_indicator_exit_slots: Number of indicator exit slots.
        num_filter_slots: Number of filter slots.
        indicator_names: Sorted list of non-filter indicator names.
        indicator_outputs: Mapping of indicator name to output metadata.
        indicator_params: Mapping of indicator name to parameter metadata.
        max_indicator_params: Maximum parameter count across all indicators.
        max_indicator_outputs: Maximum output count across all indicators.
        filter_names: Sorted list of filter indicator names.
        filter_params: Mapping of filter name to parameter metadata.
        max_filter_params: Maximum parameter count across all filters.
        comparisons: Available comparison operators.
        exit_config: Raw exit parameter ranges from strategy.yaml.
        time_exit_config: Raw time exit parameter ranges from strategy.yaml.
    """

    num_entry_slots: int
    num_indicator_exit_slots: int
    num_filter_slots: int
    indicator_names: tuple[str, ...]
    indicator_outputs: dict[str, tuple[OutputMeta, ...]]
    indicator_params: dict[str, tuple[ParamMeta, ...]]
    max_indicator_params: int
    max_indicator_outputs: int
    filter_names: tuple[str, ...]
    filter_params: dict[str, tuple[ParamMeta, ...]]
    max_filter_params: int
    comparisons: tuple[str, ...]
    exit_config: dict[str, Any]
    time_exit_config: dict[str, Any]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_strategy_config(config_dir: Path) -> StrategyMeta:
    """Load strategy metadata from config/strategy.yaml and indicators.yaml.

    Args:
        config_dir: Path to the config/ directory.

    Returns:
        StrategyMeta frozen dataclass with full parameter space description.

    Raises:
        FileNotFoundError: If either YAML file is missing.
        ValueError: If the YAML structure is invalid.
    """
    strategy_path = config_dir / "strategy.yaml"
    with open(strategy_path) as f:
        strategy_data = yaml.safe_load(f)
    if not isinstance(strategy_data, dict) or "genome" not in strategy_data:
        msg = f"Invalid strategy.yaml: expected mapping with 'genome' key in {strategy_path}"
        raise ValueError(msg)

    genome_cfg = strategy_data["genome"]
    indicator_metas = load_indicator_config(config_dir)

    # Separate indicators from filters
    ind_names: list[str] = []
    filter_names: list[str] = []
    ind_outputs: dict[str, tuple[OutputMeta, ...]] = {}
    ind_params: dict[str, tuple[ParamMeta, ...]] = {}
    filter_params: dict[str, tuple[ParamMeta, ...]] = {}

    # Load output metadata from indicators.yaml
    indicators_path = config_dir / "indicators.yaml"
    with open(indicators_path) as f:
        indicators_raw = yaml.safe_load(f)

    for meta in indicator_metas:
        raw_outputs = indicators_raw.get(meta.name, {}).get("outputs", {})
        outputs = _parse_outputs(raw_outputs)

        if meta.category == "filter":
            filter_names.append(meta.name)
            filter_params[meta.name] = meta.params
            ind_outputs[meta.name] = outputs
        else:
            ind_names.append(meta.name)
            ind_params[meta.name] = meta.params
            ind_outputs[meta.name] = outputs

    ind_names.sort()
    filter_names.sort()

    max_ind_params = max((len(p) for p in ind_params.values()), default=0)
    max_filter_params = max((len(p) for p in filter_params.values()), default=0)
    max_ind_outputs = max((len(o) for o in ind_outputs.values() if o), default=1)

    return StrategyMeta(
        num_entry_slots=genome_cfg["num_entry_slots"],
        num_indicator_exit_slots=genome_cfg["num_indicator_exit_slots"],
        num_filter_slots=genome_cfg["num_filter_slots"],
        indicator_names=tuple(ind_names),
        indicator_outputs=ind_outputs,
        indicator_params=ind_params,
        max_indicator_params=max_ind_params,
        max_indicator_outputs=max_ind_outputs,
        filter_names=tuple(filter_names),
        filter_params=filter_params,
        max_filter_params=max_filter_params,
        comparisons=tuple(genome_cfg["comparisons"]),
        exit_config=strategy_data.get("exits", {}),
        time_exit_config=strategy_data.get("time_exits", {}),
    )


def _parse_outputs(raw_outputs: dict[str, Any]) -> tuple[OutputMeta, ...]:
    """Parse raw output metadata from YAML into OutputMeta tuples.

    Args:
        raw_outputs: Mapping of output key to output metadata dict.

    Returns:
        Tuple of OutputMeta frozen dataclasses.
    """
    outputs: list[OutputMeta] = []
    for key, data in raw_outputs.items():
        outputs.append(
            OutputMeta(
                key=key,
                output_type=data["output_type"],
                threshold_min=data.get("threshold_min"),
                threshold_max=data.get("threshold_max"),
            )
        )
    return tuple(outputs)


# ---------------------------------------------------------------------------
# Random genome generation
# ---------------------------------------------------------------------------


def random_genome(meta: StrategyMeta, rng: random.Random | None = None) -> StrategyGenome:
    """Generate a random valid genome within configured parameter ranges.

    Ensures at least one entry slot and at least one exit type are enabled.

    Args:
        meta: Strategy metadata describing the parameter space.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        A valid StrategyGenome.
    """
    if rng is None:
        rng = random.Random()  # noqa: S311

    entries = [_random_indicator_gene(meta, rng) for _ in range(meta.num_entry_slots)]
    # Ensure at least one entry slot is enabled
    if not any(e.enabled for e in entries):
        idx = rng.randrange(len(entries))
        entries[idx] = _replace_enabled(entries[idx], enabled=True)

    combination = rng.choice(["AND", "OR"])

    limit_exits = _random_limit_exits(meta, rng)
    ind_exits = tuple(_random_indicator_exit_gene(meta, rng) for _ in range(meta.num_indicator_exit_slots))
    time_exits = _random_time_exits(meta, rng)
    filters = tuple(_random_filter_gene(meta, rng) for _ in range(meta.num_filter_slots))

    # Ensure at least one exit type is enabled
    genome = StrategyGenome(
        entry_indicators=tuple(entries),
        combination_logic=combination,
        limit_exits=limit_exits,
        indicator_exits=ind_exits,
        time_exits=time_exits,
        filters=filters,
    )
    if not _has_any_exit(genome):
        genome = _enable_one_exit(genome, meta, rng)

    return genome


def _random_indicator_gene(meta: StrategyMeta, rng: random.Random) -> IndicatorGene:
    """Generate a random IndicatorGene for an entry slot."""
    enabled = rng.random() < _BOOL_THRESHOLD
    ind_name = rng.choice(list(meta.indicator_names))
    outputs = meta.indicator_outputs.get(ind_name, ())
    output = (
        rng.choice(list(outputs))
        if outputs
        else OutputMeta(
            key=ind_name,
            output_type="price_level",
        )
    )
    threshold = _random_threshold(output, rng)
    params = _random_params(meta.indicator_params.get(ind_name, ()), rng)
    comparison = rng.choice(list(meta.comparisons))

    return IndicatorGene(
        enabled=enabled,
        indicator_name=ind_name,
        output_key=output.key,
        comparison=comparison,
        threshold=threshold,
        params=params,
    )


def _random_indicator_exit_gene(meta: StrategyMeta, rng: random.Random) -> IndicatorExitGene:
    """Generate a random IndicatorExitGene for an exit slot."""
    enabled = rng.random() < _BOOL_THRESHOLD
    ind_name = rng.choice(list(meta.indicator_names))
    outputs = meta.indicator_outputs.get(ind_name, ())
    output = (
        rng.choice(list(outputs))
        if outputs
        else OutputMeta(
            key=ind_name,
            output_type="price_level",
        )
    )
    threshold = _random_threshold(output, rng)
    params = _random_params(meta.indicator_params.get(ind_name, ()), rng)
    comparison = rng.choice(list(meta.comparisons))
    opposite_entry = rng.random() < _BOOL_THRESHOLD

    return IndicatorExitGene(
        enabled=enabled,
        indicator_name=ind_name,
        output_key=output.key,
        comparison=comparison,
        threshold=threshold,
        params=params,
        opposite_entry=opposite_entry,
    )


def _random_filter_gene(meta: StrategyMeta, rng: random.Random) -> FilterGene:
    """Generate a random FilterGene."""
    enabled = rng.random() < _BOOL_THRESHOLD
    filter_name = rng.choice(list(meta.filter_names))
    params = _random_params(meta.filter_params.get(filter_name, ()), rng)
    return FilterGene(enabled=enabled, filter_name=filter_name, params=params)


def _random_limit_exits(meta: StrategyMeta, rng: random.Random) -> LimitExitGene:
    """Generate random limit exit configuration."""
    ec = meta.exit_config
    sl_cfg = ec.get("stop_loss", {})
    tp_cfg = ec.get("take_profit", {})
    ts_cfg = ec.get("trailing_stop", {})
    ch_cfg = ec.get("chandelier", {})
    be_cfg = ec.get("breakeven", {})

    return LimitExitGene(
        stop_loss=StopLossConfig(
            enabled=rng.random() < _BOOL_THRESHOLD,
            mode=rng.choice(["percent", "atr"]),
            percent=_rand_float(rng, sl_cfg.get("percent", {})),
            atr_multiple=_rand_float(rng, sl_cfg.get("atr_multiple", {})),
        ),
        take_profit=TakeProfitConfig(
            enabled=rng.random() < _BOOL_THRESHOLD,
            mode=rng.choice(["percent", "atr"]),
            percent=_rand_float(rng, tp_cfg.get("percent", {})),
            atr_multiple=_rand_float(rng, tp_cfg.get("atr_multiple", {})),
        ),
        trailing_stop=TrailingStopConfig(
            enabled=rng.random() < _BOOL_THRESHOLD,
            mode=rng.choice(["percent", "atr"]),
            percent=_rand_float(rng, ts_cfg.get("percent", {})),
            atr_multiple=_rand_float(rng, ts_cfg.get("atr_multiple", {})),
            activation_percent=_rand_float(rng, ts_cfg.get("activation_percent", {})),
        ),
        chandelier=ChandelierConfig(
            enabled=rng.random() < _BOOL_THRESHOLD,
            atr_multiple=_rand_float(rng, ch_cfg.get("atr_multiple", {})),
        ),
        breakeven=BreakevenConfig(
            enabled=rng.random() < _BOOL_THRESHOLD,
            mode=rng.choice(["percent", "atr"]),
            trigger_percent=_rand_float(rng, be_cfg.get("trigger_percent", {})),
            trigger_atr_multiple=_rand_float(rng, be_cfg.get("trigger_atr_multiple", {})),
        ),
    )


def _random_time_exits(meta: StrategyMeta, rng: random.Random) -> TimeExitGene:
    """Generate random time exit configuration."""
    tc = meta.time_exit_config
    md = tc.get("max_days", {})
    wd = tc.get("weekday", {})
    sd = tc.get("stagnation_days", {})
    st = tc.get("stagnation_threshold", {})

    return TimeExitGene(
        max_days_enabled=rng.random() < _BOOL_THRESHOLD,
        max_days=_rand_int(rng, md),
        weekday_exit_enabled=rng.random() < _BOOL_THRESHOLD,
        weekday=_rand_int(rng, wd),
        eow_enabled=rng.random() < _BOOL_THRESHOLD,
        eom_enabled=rng.random() < _BOOL_THRESHOLD,
        stagnation_enabled=rng.random() < _BOOL_THRESHOLD,
        stagnation_days=_rand_int(rng, sd),
        stagnation_threshold=_rand_float(rng, st),
    )


def _random_threshold(output: OutputMeta, rng: random.Random) -> float:
    """Generate a random threshold value appropriate for the output type."""
    if output.threshold_min is not None and output.threshold_max is not None:
        return rng.uniform(output.threshold_min, output.threshold_max)
    # price_level: threshold not used for comparison, set to 0.0
    return 0.0


def _random_params(param_metas: tuple[ParamMeta, ...], rng: random.Random) -> dict[str, int | float]:
    """Generate random parameter values within configured ranges."""
    params: dict[str, int | float] = {}
    for pm in param_metas:
        if pm.param_type == "int":
            params[pm.name] = rng.randint(int(pm.min_value), int(pm.max_value))
        else:
            params[pm.name] = round(rng.uniform(float(pm.min_value), float(pm.max_value)), 4)
    return params


def _rand_float(rng: random.Random, cfg: dict[str, Any]) -> float:
    """Generate random float from a config dict with min/max/default."""
    min_v = float(cfg.get("min", 1.0))
    max_v = float(cfg.get("max", 10.0))
    return round(rng.uniform(min_v, max_v), 4)


def _rand_int(rng: random.Random, cfg: dict[str, Any]) -> int:
    """Generate random int from a config dict with min/max/default."""
    min_v = int(cfg.get("min", 1))
    max_v = int(cfg.get("max", 10))
    return rng.randint(min_v, max_v)


def _replace_enabled(gene: IndicatorGene, enabled: bool) -> IndicatorGene:
    """Return a copy of an IndicatorGene with enabled flag changed."""
    return IndicatorGene(
        enabled=enabled,
        indicator_name=gene.indicator_name,
        output_key=gene.output_key,
        comparison=gene.comparison,
        threshold=gene.threshold,
        params=gene.params,
    )


def _has_any_exit(genome: StrategyGenome) -> bool:
    """Check if any exit type is enabled in the genome."""
    le = genome.limit_exits
    if le.stop_loss.enabled or le.take_profit.enabled:
        return True
    if le.trailing_stop.enabled or le.chandelier.enabled or le.breakeven.enabled:
        return True
    if any(ie.enabled for ie in genome.indicator_exits):
        return True
    te = genome.time_exits
    if te.max_days_enabled or te.weekday_exit_enabled:
        return True
    return te.eow_enabled or te.eom_enabled or te.stagnation_enabled


def _enable_one_exit(genome: StrategyGenome, meta: StrategyMeta, rng: random.Random) -> StrategyGenome:
    """Enable at least one exit type in the genome."""
    le = genome.limit_exits
    new_sl = StopLossConfig(
        enabled=True,
        mode=le.stop_loss.mode,
        percent=le.stop_loss.percent,
        atr_multiple=le.stop_loss.atr_multiple,
    )
    new_le = LimitExitGene(
        stop_loss=new_sl,
        take_profit=le.take_profit,
        trailing_stop=le.trailing_stop,
        chandelier=le.chandelier,
        breakeven=le.breakeven,
    )
    return StrategyGenome(
        entry_indicators=genome.entry_indicators,
        combination_logic=genome.combination_logic,
        limit_exits=new_le,
        indicator_exits=genome.indicator_exits,
        time_exits=genome.time_exits,
        filters=genome.filters,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_genome(genome: StrategyGenome, meta: StrategyMeta) -> bool:
    """Check structural validity of a genome.

    Validates indicator names exist, output keys match, params are within
    ranges, at least one entry slot enabled, and at least one exit enabled.

    Args:
        genome: The genome to validate.
        meta: Strategy metadata for validation context.

    Returns:
        True if valid, False otherwise.
    """
    checks = [
        _validate_entries(genome, meta),
        genome.combination_logic in ("AND", "OR"),
        _validate_limit_exits(genome, meta),
        _validate_indicator_exits(genome, meta),
        _validate_time_exits(genome, meta),
        _validate_filters(genome, meta),
        _has_any_exit(genome),
    ]
    return all(checks)


def _validate_entries(genome: StrategyGenome, meta: StrategyMeta) -> bool:
    """Validate entry indicator slots."""
    if not any(e.enabled for e in genome.entry_indicators):
        return False
    for gene in genome.entry_indicators:
        if not gene.enabled:
            continue
        if gene.indicator_name not in meta.indicator_names:
            return False
        outputs = meta.indicator_outputs.get(gene.indicator_name, ())
        if not any(o.key == gene.output_key for o in outputs):
            return False
        if gene.comparison not in meta.comparisons:
            return False
        if not _validate_indicator_params(gene.indicator_name, gene.params, meta.indicator_params):
            return False
    return True


def _validate_indicator_exits(genome: StrategyGenome, meta: StrategyMeta) -> bool:
    """Validate indicator exit slots."""
    for gene in genome.indicator_exits:
        if not gene.enabled:
            continue
        if gene.indicator_name not in meta.indicator_names:
            return False
        outputs = meta.indicator_outputs.get(gene.indicator_name, ())
        if not any(o.key == gene.output_key for o in outputs):
            return False
        if gene.comparison not in meta.comparisons:
            return False
        if not _validate_indicator_params(gene.indicator_name, gene.params, meta.indicator_params):
            return False
    return True


def _validate_limit_exits(genome: StrategyGenome, meta: StrategyMeta) -> bool:
    """Validate limit exit parameters are within configured ranges."""
    le = genome.limit_exits
    ec = meta.exit_config

    valid_modes = ("percent", "atr")
    modes_ok = all(
        m in valid_modes for m in (le.stop_loss.mode, le.take_profit.mode, le.trailing_stop.mode, le.breakeven.mode)
    )
    if not modes_ok:
        return False

    return _validate_limit_exit_ranges(le, ec)


def _validate_limit_exit_ranges(le: LimitExitGene, ec: dict[str, Any]) -> bool:
    """Validate that all limit exit parameter values fall within configured ranges."""
    sl_cfg = ec.get("stop_loss", {})
    tp_cfg = ec.get("take_profit", {})
    ts_cfg = ec.get("trailing_stop", {})
    ch_cfg = ec.get("chandelier", {})
    be_cfg = ec.get("breakeven", {})

    checks = [
        _in_range(le.stop_loss.percent, sl_cfg.get("percent", {})),
        _in_range(le.stop_loss.atr_multiple, sl_cfg.get("atr_multiple", {})),
        _in_range(le.take_profit.percent, tp_cfg.get("percent", {})),
        _in_range(le.take_profit.atr_multiple, tp_cfg.get("atr_multiple", {})),
        _in_range(le.trailing_stop.percent, ts_cfg.get("percent", {})),
        _in_range(le.trailing_stop.atr_multiple, ts_cfg.get("atr_multiple", {})),
        _in_range(le.trailing_stop.activation_percent, ts_cfg.get("activation_percent", {})),
        _in_range(le.chandelier.atr_multiple, ch_cfg.get("atr_multiple", {})),
        _in_range(le.breakeven.trigger_percent, be_cfg.get("trigger_percent", {})),
        _in_range(le.breakeven.trigger_atr_multiple, be_cfg.get("trigger_atr_multiple", {})),
    ]
    return all(checks)


def _validate_time_exits(genome: StrategyGenome, meta: StrategyMeta) -> bool:
    """Validate time exit parameters."""
    te = genome.time_exits
    tc = meta.time_exit_config

    if not _in_range(te.max_days, tc.get("max_days", {})):
        return False
    if not _in_range(te.weekday, tc.get("weekday", {})):
        return False
    if not _in_range(te.stagnation_days, tc.get("stagnation_days", {})):
        return False
    return _in_range(te.stagnation_threshold, tc.get("stagnation_threshold", {}))


def _validate_filters(genome: StrategyGenome, meta: StrategyMeta) -> bool:
    """Validate filter gene slots."""
    for gene in genome.filters:
        if not gene.enabled:
            continue
        if gene.filter_name not in meta.filter_names:
            return False
        if not _validate_indicator_params(gene.filter_name, gene.params, meta.filter_params):
            return False
    return True


def _validate_indicator_params(
    name: str,
    params: dict[str, int | float],
    all_params: dict[str, tuple[ParamMeta, ...]],
) -> bool:
    """Check indicator parameters are within configured ranges."""
    param_metas = all_params.get(name, ())
    valid_names = {pm.name for pm in param_metas}
    for pname in params:
        if pname not in valid_names:
            return False
    for pm in param_metas:
        if pm.name in params:
            val = params[pm.name]
            if val < pm.min_value or val > pm.max_value:
                return False
    return True


def _in_range(value: int | float, cfg: dict[str, Any]) -> bool:
    """Check if value is within a config dict's min/max range."""
    if not cfg:
        return True
    min_v = cfg.get("min", float("-inf"))
    max_v = cfg.get("max", float("inf"))
    return bool(min_v <= value <= max_v)


# ---------------------------------------------------------------------------
# Flat chromosome encoding/decoding
# ---------------------------------------------------------------------------

# Encoding order:
# 1. Entry slots (N × slot_width)
# 2. Combination logic (1 gene)
# 3. Limit exits (19 genes)
# 4. Indicator exit slots (M × slot_width)
# 5. Time exits (9 genes)
# 6. Filter slots (K × filter_slot_width)


def genome_length(meta: StrategyMeta) -> int:
    """Calculate total chromosome length for the configured genome structure.

    Args:
        meta: Strategy metadata.

    Returns:
        Total number of genes in the flat chromosome.
    """
    entry_width = _entry_slot_width(meta)
    exit_width = _indicator_exit_slot_width(meta)
    filter_width = _filter_slot_width(meta)

    total = meta.num_entry_slots * entry_width  # entry slots
    total += 1  # combination logic
    total += 19  # limit exits
    total += meta.num_indicator_exit_slots * exit_width  # indicator exit slots
    total += 9  # time exits
    total += meta.num_filter_slots * filter_width  # filter slots
    return total


def _entry_slot_width(meta: StrategyMeta) -> int:
    """Width of one entry slot: enabled + ind_idx + out_idx + cmp_idx + threshold + params."""
    return 5 + meta.max_indicator_params


def _indicator_exit_slot_width(meta: StrategyMeta) -> int:
    """Width of one indicator exit slot: same as entry + opposite_entry."""
    return 6 + meta.max_indicator_params


def _filter_slot_width(meta: StrategyMeta) -> int:
    """Width of one filter slot: enabled + filter_idx + params."""
    return 2 + meta.max_filter_params


def genome_to_flat(genome: StrategyGenome, meta: StrategyMeta) -> list[float]:
    """Serialize a StrategyGenome to a flat list of floats.

    Args:
        genome: The genome to serialize.
        meta: Strategy metadata for encoding context.

    Returns:
        List of float values representing the chromosome.
    """
    values: list[float] = []

    # 1. Entry slots
    for entry_gene in genome.entry_indicators:
        values.extend(_encode_entry_slot(entry_gene, meta))

    # 2. Combination logic
    values.append(0.0 if genome.combination_logic == "AND" else 1.0)

    # 3. Limit exits
    values.extend(_encode_limit_exits(genome.limit_exits))

    # 4. Indicator exit slots
    for exit_gene in genome.indicator_exits:
        values.extend(_encode_indicator_exit_slot(exit_gene, meta))

    # 5. Time exits
    values.extend(_encode_time_exits(genome.time_exits))

    # 6. Filter slots
    for filter_gene in genome.filters:
        values.extend(_encode_filter_slot(filter_gene, meta))

    return values


def flat_to_genome(values: list[float], meta: StrategyMeta) -> StrategyGenome:
    """Deserialize a flat list of floats into a StrategyGenome.

    Clamps and rounds values as needed to produce valid gene values.

    Args:
        values: Flat chromosome values.
        meta: Strategy metadata for decoding context.

    Returns:
        Reconstructed StrategyGenome.
    """
    pos = 0
    entry_width = _entry_slot_width(meta)
    exit_width = _indicator_exit_slot_width(meta)
    filter_width = _filter_slot_width(meta)

    # 1. Entry slots
    entries: list[IndicatorGene] = []
    for _ in range(meta.num_entry_slots):
        entry_gene = _decode_entry_slot(values[pos : pos + entry_width], meta)
        entries.append(entry_gene)
        pos += entry_width

    # 2. Combination logic
    combination = "AND" if values[pos] < _BOOL_THRESHOLD else "OR"
    pos += 1

    # 3. Limit exits
    limit_exits = _decode_limit_exits(values[pos : pos + 19], meta)
    pos += 19

    # 4. Indicator exit slots
    ind_exits: list[IndicatorExitGene] = []
    for _ in range(meta.num_indicator_exit_slots):
        exit_gene = _decode_indicator_exit_slot(values[pos : pos + exit_width], meta)
        ind_exits.append(exit_gene)
        pos += exit_width

    # 5. Time exits
    time_exits = _decode_time_exits(values[pos : pos + 9], meta)
    pos += 9

    # 6. Filter slots
    filters: list[FilterGene] = []
    for _ in range(meta.num_filter_slots):
        flt_gene = _decode_filter_slot(values[pos : pos + filter_width], meta)
        filters.append(flt_gene)
        pos += filter_width

    return StrategyGenome(
        entry_indicators=tuple(entries),
        combination_logic=combination,
        limit_exits=limit_exits,
        indicator_exits=tuple(ind_exits),
        time_exits=time_exits,
        filters=tuple(filters),
    )


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode_entry_slot(gene: IndicatorGene, meta: StrategyMeta) -> list[float]:
    """Encode one entry indicator slot to flat values."""
    values: list[float] = [
        1.0 if gene.enabled else 0.0,
        float(meta.indicator_names.index(gene.indicator_name)),
        float(_output_index(gene.indicator_name, gene.output_key, meta)),
        float(meta.comparisons.index(gene.comparison)),
        gene.threshold,
    ]
    values.extend(_encode_params(gene.indicator_name, gene.params, meta.indicator_params, meta.max_indicator_params))
    return values


def _encode_indicator_exit_slot(gene: IndicatorExitGene, meta: StrategyMeta) -> list[float]:
    """Encode one indicator exit slot to flat values."""
    values: list[float] = [
        1.0 if gene.enabled else 0.0,
        float(meta.indicator_names.index(gene.indicator_name)),
        float(_output_index(gene.indicator_name, gene.output_key, meta)),
        float(meta.comparisons.index(gene.comparison)),
        gene.threshold,
        1.0 if gene.opposite_entry else 0.0,
    ]
    values.extend(_encode_params(gene.indicator_name, gene.params, meta.indicator_params, meta.max_indicator_params))
    return values


def _encode_filter_slot(gene: FilterGene, meta: StrategyMeta) -> list[float]:
    """Encode one filter slot to flat values."""
    values: list[float] = [
        1.0 if gene.enabled else 0.0,
        float(meta.filter_names.index(gene.filter_name)),
    ]
    values.extend(_encode_params(gene.filter_name, gene.params, meta.filter_params, meta.max_filter_params))
    return values


def _encode_limit_exits(le: LimitExitGene) -> list[float]:
    """Encode limit exits to 19 flat values.

    Order: SL(4) + TP(4) + TS(5) + CH(2) + BE(4) = 19.
    """
    values: list[float] = [
        # Stop loss (4)
        1.0 if le.stop_loss.enabled else 0.0,
        0.0 if le.stop_loss.mode == "percent" else 1.0,
        le.stop_loss.percent,
        le.stop_loss.atr_multiple,
        # Take profit (4)
        1.0 if le.take_profit.enabled else 0.0,
        0.0 if le.take_profit.mode == "percent" else 1.0,
        le.take_profit.percent,
        le.take_profit.atr_multiple,
        # Trailing stop (5)
        1.0 if le.trailing_stop.enabled else 0.0,
        0.0 if le.trailing_stop.mode == "percent" else 1.0,
        le.trailing_stop.percent,
        le.trailing_stop.atr_multiple,
        le.trailing_stop.activation_percent,
        # Chandelier (2)
        1.0 if le.chandelier.enabled else 0.0,
        le.chandelier.atr_multiple,
        # Breakeven (4)
        1.0 if le.breakeven.enabled else 0.0,
        0.0 if le.breakeven.mode == "percent" else 1.0,
        le.breakeven.trigger_percent,
        le.breakeven.trigger_atr_multiple,
    ]
    return values


def _encode_time_exits(te: TimeExitGene) -> list[float]:
    """Encode time exits to 9 flat values."""
    return [
        1.0 if te.max_days_enabled else 0.0,
        float(te.max_days),
        1.0 if te.weekday_exit_enabled else 0.0,
        float(te.weekday),
        1.0 if te.eow_enabled else 0.0,
        1.0 if te.eom_enabled else 0.0,
        1.0 if te.stagnation_enabled else 0.0,
        float(te.stagnation_days),
        te.stagnation_threshold,
    ]


def _encode_params(
    name: str,
    params: dict[str, int | float],
    all_params: dict[str, tuple[ParamMeta, ...]],
    max_params: int,
) -> list[float]:
    """Encode indicator parameters to flat values, zero-padded to max_params."""
    param_metas = all_params.get(name, ())
    sorted_metas = sorted(param_metas, key=lambda pm: pm.name)
    values: list[float] = []
    for pm in sorted_metas:
        values.append(float(params.get(pm.name, pm.default)))
    # Zero-pad
    while len(values) < max_params:
        values.append(0.0)
    return values


def _output_index(ind_name: str, output_key: str, meta: StrategyMeta) -> int:
    """Get the index of an output key within an indicator's outputs."""
    outputs = meta.indicator_outputs.get(ind_name, ())
    for i, out in enumerate(outputs):
        if out.key == output_key:
            return i
    return 0


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


def _decode_entry_slot(values: list[float], meta: StrategyMeta) -> IndicatorGene:
    """Decode one entry indicator slot from flat values."""
    enabled = values[0] >= _BOOL_THRESHOLD
    ind_idx = _clamp_int(round(values[1]), 0, len(meta.indicator_names) - 1)
    ind_name = meta.indicator_names[ind_idx]
    outputs = meta.indicator_outputs.get(ind_name, ())
    out_idx = _clamp_int(round(values[2]), 0, max(len(outputs) - 1, 0))
    output_key = outputs[out_idx].key if outputs else ind_name
    cmp_idx = _clamp_int(round(values[3]), 0, len(meta.comparisons) - 1)
    comparison = meta.comparisons[cmp_idx]
    threshold = values[4]
    params = _decode_params(values[5:], ind_name, meta.indicator_params)

    return IndicatorGene(
        enabled=enabled,
        indicator_name=ind_name,
        output_key=output_key,
        comparison=comparison,
        threshold=threshold,
        params=params,
    )


def _decode_indicator_exit_slot(values: list[float], meta: StrategyMeta) -> IndicatorExitGene:
    """Decode one indicator exit slot from flat values."""
    enabled = values[0] >= _BOOL_THRESHOLD
    ind_idx = _clamp_int(round(values[1]), 0, len(meta.indicator_names) - 1)
    ind_name = meta.indicator_names[ind_idx]
    outputs = meta.indicator_outputs.get(ind_name, ())
    out_idx = _clamp_int(round(values[2]), 0, max(len(outputs) - 1, 0))
    output_key = outputs[out_idx].key if outputs else ind_name
    cmp_idx = _clamp_int(round(values[3]), 0, len(meta.comparisons) - 1)
    comparison = meta.comparisons[cmp_idx]
    threshold = values[4]
    opposite_entry = values[5] >= _BOOL_THRESHOLD
    params = _decode_params(values[6:], ind_name, meta.indicator_params)

    return IndicatorExitGene(
        enabled=enabled,
        indicator_name=ind_name,
        output_key=output_key,
        comparison=comparison,
        threshold=threshold,
        params=params,
        opposite_entry=opposite_entry,
    )


def _decode_filter_slot(values: list[float], meta: StrategyMeta) -> FilterGene:
    """Decode one filter slot from flat values."""
    enabled = values[0] >= _BOOL_THRESHOLD
    filter_idx = _clamp_int(round(values[1]), 0, len(meta.filter_names) - 1)
    filter_name = meta.filter_names[filter_idx]
    params = _decode_params(values[2:], filter_name, meta.filter_params)
    return FilterGene(enabled=enabled, filter_name=filter_name, params=params)


def _decode_limit_exits(values: list[float], meta: StrategyMeta) -> LimitExitGene:
    """Decode limit exits from 19 flat values."""
    ec = meta.exit_config

    sl_cfg = ec.get("stop_loss", {})
    tp_cfg = ec.get("take_profit", {})
    ts_cfg = ec.get("trailing_stop", {})
    ch_cfg = ec.get("chandelier", {})
    be_cfg = ec.get("breakeven", {})

    return LimitExitGene(
        stop_loss=StopLossConfig(
            enabled=values[0] >= _BOOL_THRESHOLD,
            mode="percent" if values[1] < _BOOL_THRESHOLD else "atr",
            percent=_clamp_float(values[2], sl_cfg.get("percent", {})),
            atr_multiple=_clamp_float(values[3], sl_cfg.get("atr_multiple", {})),
        ),
        take_profit=TakeProfitConfig(
            enabled=values[4] >= _BOOL_THRESHOLD,
            mode="percent" if values[5] < _BOOL_THRESHOLD else "atr",
            percent=_clamp_float(values[6], tp_cfg.get("percent", {})),
            atr_multiple=_clamp_float(values[7], tp_cfg.get("atr_multiple", {})),
        ),
        trailing_stop=TrailingStopConfig(
            enabled=values[8] >= _BOOL_THRESHOLD,
            mode="percent" if values[9] < _BOOL_THRESHOLD else "atr",
            percent=_clamp_float(values[10], ts_cfg.get("percent", {})),
            atr_multiple=_clamp_float(values[11], ts_cfg.get("atr_multiple", {})),
            activation_percent=_clamp_float(values[12], ts_cfg.get("activation_percent", {})),
        ),
        chandelier=ChandelierConfig(
            enabled=values[13] >= _BOOL_THRESHOLD,
            atr_multiple=_clamp_float(values[14], ch_cfg.get("atr_multiple", {})),
        ),
        breakeven=BreakevenConfig(
            enabled=values[15] >= _BOOL_THRESHOLD,
            mode="percent" if values[16] < _BOOL_THRESHOLD else "atr",
            trigger_percent=_clamp_float(values[17], be_cfg.get("trigger_percent", {})),
            trigger_atr_multiple=_clamp_float(values[18], be_cfg.get("trigger_atr_multiple", {})),
        ),
    )


def _decode_time_exits(values: list[float], meta: StrategyMeta) -> TimeExitGene:
    """Decode time exits from 9 flat values."""
    tc = meta.time_exit_config
    md_cfg = tc.get("max_days", {})
    wd_cfg = tc.get("weekday", {})
    sd_cfg = tc.get("stagnation_days", {})
    return TimeExitGene(
        max_days_enabled=values[0] >= _BOOL_THRESHOLD,
        max_days=_clamp_int(round(values[1]), md_cfg.get("min", 1), md_cfg.get("max", 30)),
        weekday_exit_enabled=values[2] >= _BOOL_THRESHOLD,
        weekday=_clamp_int(round(values[3]), wd_cfg.get("min", 0), wd_cfg.get("max", 4)),
        eow_enabled=values[4] >= _BOOL_THRESHOLD,
        eom_enabled=values[5] >= _BOOL_THRESHOLD,
        stagnation_enabled=values[6] >= _BOOL_THRESHOLD,
        stagnation_days=_clamp_int(round(values[7]), sd_cfg.get("min", 2), sd_cfg.get("max", 15)),
        stagnation_threshold=_clamp_float(values[8], tc.get("stagnation_threshold", {})),
    )


def _decode_params(
    values: list[float],
    name: str,
    all_params: dict[str, tuple[ParamMeta, ...]],
) -> dict[str, int | float]:
    """Decode indicator parameters from flat values."""
    param_metas = all_params.get(name, ())
    sorted_metas = sorted(param_metas, key=lambda pm: pm.name)
    params: dict[str, int | float] = {}
    for i, pm in enumerate(sorted_metas):
        if i >= len(values):
            params[pm.name] = pm.default
            continue
        raw = values[i]
        clamped = max(float(pm.min_value), min(float(pm.max_value), raw))
        if pm.param_type == "int":
            params[pm.name] = _clamp_int(round(clamped), int(pm.min_value), int(pm.max_value))
        else:
            params[pm.name] = round(clamped, 4)
    return params


def _clamp_int(value: int, min_v: int, max_v: int) -> int:
    """Clamp an integer to a range."""
    return max(min_v, min(max_v, value))


def _clamp_float(value: float, cfg: dict[str, Any]) -> float:
    """Clamp a float to a config dict's min/max range."""
    if not cfg:
        return value
    min_v = float(cfg.get("min", value))
    max_v = float(cfg.get("max", value))
    return round(max(min_v, min(max_v, value)), 4)
