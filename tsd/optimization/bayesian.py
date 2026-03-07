"""Bayesian optimization engine using Optuna.

Fine-tunes strategy parameters for a fixed genome structure discovered
by the GA (Stage A). Uses TPE sampler with MedianPruner and persists
studies to SQLite for resume support.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from tsd.config import env_int, env_str
from tsd.indicators.base import ParamMeta
from tsd.optimization.fitness import FitnessConfig, compute_fitness
from tsd.optimization.metrics import aggregate_metrics
from tsd.strategy.evaluator import (
    BacktestResult,
    EvaluatorConfig,
    run_backtest,
)
from tsd.strategy.genome import (
    BreakevenConfig,
    ChandelierConfig,
    FilterGene,
    IndicatorExitGene,
    IndicatorGene,
    LimitExitGene,
    OutputMeta,
    StopLossConfig,
    StrategyGenome,
    StrategyMeta,
    TakeProfitConfig,
    TimeExitGene,
    TrailingStopConfig,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianConfig:
    """Configuration for the Bayesian optimizer.

    Attributes:
        n_trials: Number of Optuna trials to run.
        random_seed: Random seed for the TPE sampler.
        checkpoint_dir: Directory for the SQLite study database.
        study_name: Optuna study name.
        log_interval: Log best fitness every N completed trials.
    """

    n_trials: int = 500
    random_seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("results/checkpoints"))
    study_name: str = "tsd_bayesian"
    log_interval: int = 10


@dataclass(frozen=True)
class BayesianResult:
    """Result of a Bayesian optimization run.

    Attributes:
        best_genome: Best strategy genome found.
        best_fitness: Fitness of the best genome.
        trials_run: Total number of completed trials.
        trials_pruned: Number of pruned trials.
        best_params: Best parameter values found.
    """

    best_genome: StrategyGenome
    best_fitness: float
    trials_run: int
    trials_pruned: int
    best_params: dict[str, float]


def load_bayesian_config() -> BayesianConfig:
    """Load Bayesian optimizer configuration from environment variables.

    Returns:
        BayesianConfig with values from TSD_BAYESIAN_* environment variables.
    """
    return BayesianConfig(
        n_trials=env_int("TSD_BAYESIAN_N_TRIALS", 500),
        random_seed=env_int("TSD_BAYESIAN_RANDOM_SEED", 42),
        checkpoint_dir=Path(env_str("TSD_BAYESIAN_CHECKPOINT_DIR", "results/checkpoints")),
        study_name=env_str("TSD_BAYESIAN_STUDY_NAME", "tsd_bayesian"),
        log_interval=env_int("TSD_BAYESIAN_LOG_INTERVAL", 10),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_bayesian(  # noqa: PLR0913
    genome: StrategyGenome,
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    bayesian_config: BayesianConfig | None = None,
    eval_config: EvaluatorConfig | None = None,
    fitness_config: FitnessConfig | None = None,
    resume: bool = False,
) -> BayesianResult:
    """Run Bayesian optimization to fine-tune strategy parameters.

    Takes a fixed genome structure (indicator choices, enabled flags,
    comparisons) and optimizes numeric parameters within their
    configured ranges.

    Args:
        genome: Strategy genome with fixed structure to fine-tune.
        meta: Strategy metadata describing the parameter space.
        stocks_data: Mapping of stock ticker to OHLCV DataFrame.
        indicator_outputs: Mapping of indicator name to output metadata.
        bayesian_config: Bayesian optimizer configuration. Uses defaults if None.
        eval_config: Evaluator configuration. Uses defaults if None.
        fitness_config: Fitness configuration. Uses defaults if None.
        resume: Whether to resume from an existing study.

    Returns:
        BayesianResult with the best genome, fitness, and trial info.
    """
    bayesian_config = bayesian_config or BayesianConfig()
    eval_config = eval_config or EvaluatorConfig()
    fitness_config = fitness_config or FitnessConfig()

    study = _create_or_load_study(bayesian_config, resume)
    objective = _make_objective(genome, meta, stocks_data, indicator_outputs, eval_config, fitness_config)
    callback = _make_log_callback(bayesian_config.log_interval)

    study.optimize(objective, n_trials=bayesian_config.n_trials, callbacks=[callback])

    best_trial = study.best_trial
    best_genome = _suggest_genome(best_trial, genome, meta)
    best_value = best_trial.value if best_trial.value is not None else 0.0
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    LOGGER.info(
        "Bayesian optimization complete: %d trials, %d pruned, best=%.4f",
        len(completed),
        len(pruned),
        best_value,
    )

    return BayesianResult(
        best_genome=best_genome,
        best_fitness=best_value,
        trials_run=len(completed),
        trials_pruned=len(pruned),
        best_params={k: float(v) for k, v in best_trial.params.items()},
    )


# ---------------------------------------------------------------------------
# Study management
# ---------------------------------------------------------------------------


def _create_or_load_study(config: BayesianConfig, resume: bool) -> optuna.Study:
    """Create a new study or load an existing one.

    Args:
        config: Bayesian optimizer configuration.
        resume: If True, load existing study. If False, delete existing DB first.

    Returns:
        Optuna Study instance.
    """
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = config.checkpoint_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"

    if not resume and db_path.exists():
        db_path.unlink()
        LOGGER.info("Deleted existing study DB for fresh start")

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    pruner = optuna.pruners.MedianPruner()

    return optuna.create_study(
        study_name=config.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=resume,
    )


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------


def _make_objective(  # noqa: PLR0913
    genome: StrategyGenome,
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    eval_config: EvaluatorConfig,
    fitness_config: FitnessConfig,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective function as a closure.

    Args:
        genome: Fixed genome structure to fine-tune.
        meta: Strategy metadata.
        stocks_data: Mapping of ticker to OHLCV DataFrame.
        indicator_outputs: Indicator output metadata.
        eval_config: Evaluator configuration.
        fitness_config: Fitness configuration.

    Returns:
        Callable that takes an Optuna Trial and returns fitness.
    """

    def objective(trial: optuna.Trial) -> float:
        """Evaluate a single Optuna trial."""
        suggested = _suggest_genome(trial, genome, meta)

        results: list[BacktestResult] = []
        for _ticker, df in stocks_data.items():
            try:
                result = run_backtest(suggested, df, indicator_outputs, eval_config)
                results.append(result)
            except Exception:  # noqa: BLE001
                LOGGER.warning("Backtest failed for a stock during trial %d", trial.number)

        if not results:
            return 0.0

        aggregated = aggregate_metrics(results)
        all_trades = tuple(t for r in results for t in r.trades)
        return compute_fitness(
            aggregated,
            fitness_config,
            trades=all_trades,
            num_stocks=len(stocks_data),
        )

    return objective


# ---------------------------------------------------------------------------
# Parameter suggestion helpers
# ---------------------------------------------------------------------------


def _suggest_genome(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    genome: StrategyGenome,
    meta: StrategyMeta,
) -> StrategyGenome:
    """Create a new genome with Optuna-suggested numeric parameters.

    Preserves the genome structure (enabled flags, indicator names,
    comparisons) and suggests new values for numeric parameters.

    Args:
        trial: Optuna trial or frozen trial for parameter suggestion.
        genome: Original genome structure.
        meta: Strategy metadata.

    Returns:
        New StrategyGenome with suggested parameters.
    """
    entries = _suggest_entry_indicators(trial, genome, meta)
    limit_exits = _suggest_limit_exits(trial, genome, meta)
    ind_exits = _suggest_indicator_exits(trial, genome, meta)
    time_exits = _suggest_time_exits(trial, genome, meta)
    filters = _suggest_filters(trial, genome, meta)

    return StrategyGenome(
        entry_indicators=entries,
        combination_logic=genome.combination_logic,
        limit_exits=limit_exits,
        indicator_exits=ind_exits,
        time_exits=time_exits,
        filters=filters,
    )


def _suggest_entry_indicators(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    genome: StrategyGenome,
    meta: StrategyMeta,
) -> tuple[IndicatorGene, ...]:
    """Suggest parameters for entry indicator slots.

    Args:
        trial: Optuna trial.
        genome: Original genome.
        meta: Strategy metadata.

    Returns:
        Tuple of IndicatorGene with suggested parameters.
    """
    entries: list[IndicatorGene] = []
    for i, gene in enumerate(genome.entry_indicators):
        if not gene.enabled:
            entries.append(gene)
            continue
        prefix = f"entry_{i}"
        params = _suggest_indicator_params(trial, prefix, gene.indicator_name, meta.indicator_params)
        threshold = _suggest_threshold(trial, prefix, gene, meta)
        entries.append(
            IndicatorGene(
                enabled=gene.enabled,
                indicator_name=gene.indicator_name,
                output_key=gene.output_key,
                comparison=gene.comparison,
                threshold=threshold,
                params=params,
            )
        )
    return tuple(entries)


def _suggest_indicator_exits(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    genome: StrategyGenome,
    meta: StrategyMeta,
) -> tuple[IndicatorExitGene, ...]:
    """Suggest parameters for indicator exit slots.

    Args:
        trial: Optuna trial.
        genome: Original genome.
        meta: Strategy metadata.

    Returns:
        Tuple of IndicatorExitGene with suggested parameters.
    """
    exits: list[IndicatorExitGene] = []
    for i, gene in enumerate(genome.indicator_exits):
        if not gene.enabled:
            exits.append(gene)
            continue
        prefix = f"exit_{i}"
        params = _suggest_indicator_params(trial, prefix, gene.indicator_name, meta.indicator_params)
        threshold = _suggest_threshold(trial, prefix, gene, meta)
        exits.append(
            IndicatorExitGene(
                enabled=gene.enabled,
                indicator_name=gene.indicator_name,
                output_key=gene.output_key,
                comparison=gene.comparison,
                threshold=threshold,
                params=params,
                opposite_entry=gene.opposite_entry,
            )
        )
    return tuple(exits)


def _suggest_limit_exits(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    genome: StrategyGenome,
    meta: StrategyMeta,
) -> LimitExitGene:
    """Suggest parameters for limit-based exits.

    Args:
        trial: Optuna trial.
        genome: Original genome.
        meta: Strategy metadata.

    Returns:
        LimitExitGene with suggested parameters.
    """
    le = genome.limit_exits
    ec = meta.exit_config

    stop_loss = _suggest_stop_loss(trial, le.stop_loss, ec.get("stop_loss", {}))
    take_profit = _suggest_take_profit(trial, le.take_profit, ec.get("take_profit", {}))
    trailing_stop = _suggest_trailing_stop(trial, le.trailing_stop, ec.get("trailing_stop", {}))
    chandelier = _suggest_chandelier(trial, le.chandelier, ec.get("chandelier", {}))
    breakeven = _suggest_breakeven(trial, le.breakeven, ec.get("breakeven", {}))

    return LimitExitGene(
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        chandelier=chandelier,
        breakeven=breakeven,
    )


def _suggest_stop_loss(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    sl: StopLossConfig,
    cfg: dict[str, Any],
) -> StopLossConfig:
    """Suggest stop-loss parameters."""
    if not sl.enabled:
        return sl
    pct_cfg = cfg.get("percent", {})
    atr_cfg = cfg.get("atr_multiple", {})
    return StopLossConfig(
        enabled=sl.enabled,
        mode=sl.mode,
        percent=_suggest_float(trial, "sl_percent", pct_cfg, sl.percent),
        atr_multiple=_suggest_float(trial, "sl_atr_multiple", atr_cfg, sl.atr_multiple),
    )


def _suggest_take_profit(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    tp: TakeProfitConfig,
    cfg: dict[str, Any],
) -> TakeProfitConfig:
    """Suggest take-profit parameters."""
    if not tp.enabled:
        return tp
    pct_cfg = cfg.get("percent", {})
    atr_cfg = cfg.get("atr_multiple", {})
    return TakeProfitConfig(
        enabled=tp.enabled,
        mode=tp.mode,
        percent=_suggest_float(trial, "tp_percent", pct_cfg, tp.percent),
        atr_multiple=_suggest_float(trial, "tp_atr_multiple", atr_cfg, tp.atr_multiple),
    )


def _suggest_trailing_stop(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    ts: TrailingStopConfig,
    cfg: dict[str, Any],
) -> TrailingStopConfig:
    """Suggest trailing stop parameters."""
    if not ts.enabled:
        return ts
    pct_cfg = cfg.get("percent", {})
    atr_cfg = cfg.get("atr_multiple", {})
    act_cfg = cfg.get("activation_percent", {})
    return TrailingStopConfig(
        enabled=ts.enabled,
        mode=ts.mode,
        percent=_suggest_float(trial, "ts_percent", pct_cfg, ts.percent),
        atr_multiple=_suggest_float(trial, "ts_atr_multiple", atr_cfg, ts.atr_multiple),
        activation_percent=_suggest_float(trial, "ts_activation_percent", act_cfg, ts.activation_percent),
    )


def _suggest_chandelier(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    ch: ChandelierConfig,
    cfg: dict[str, Any],
) -> ChandelierConfig:
    """Suggest chandelier exit parameters."""
    if not ch.enabled:
        return ch
    atr_cfg = cfg.get("atr_multiple", {})
    return ChandelierConfig(
        enabled=ch.enabled,
        atr_multiple=_suggest_float(trial, "ch_atr_multiple", atr_cfg, ch.atr_multiple),
    )


def _suggest_breakeven(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    be: BreakevenConfig,
    cfg: dict[str, Any],
) -> BreakevenConfig:
    """Suggest breakeven parameters."""
    if not be.enabled:
        return be
    pct_cfg = cfg.get("trigger_percent", {})
    atr_cfg = cfg.get("trigger_atr_multiple", {})
    return BreakevenConfig(
        enabled=be.enabled,
        mode=be.mode,
        trigger_percent=_suggest_float(trial, "be_trigger_percent", pct_cfg, be.trigger_percent),
        trigger_atr_multiple=_suggest_float(trial, "be_trigger_atr_multiple", atr_cfg, be.trigger_atr_multiple),
    )


def _suggest_time_exits(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    genome: StrategyGenome,
    meta: StrategyMeta,
) -> TimeExitGene:
    """Suggest parameters for time-based exits.

    Args:
        trial: Optuna trial.
        genome: Original genome.
        meta: Strategy metadata.

    Returns:
        TimeExitGene with suggested parameters.
    """
    te = genome.time_exits
    tc = meta.time_exit_config

    max_days = te.max_days
    if te.max_days_enabled:
        md_cfg = tc.get("max_days", {})
        max_days = _suggest_int(trial, "time_max_days", md_cfg, te.max_days)

    weekday = te.weekday
    if te.weekday_exit_enabled:
        wd_cfg = tc.get("weekday", {})
        weekday = _suggest_int(trial, "time_weekday", wd_cfg, te.weekday)

    stagnation_days = te.stagnation_days
    stagnation_threshold = te.stagnation_threshold
    if te.stagnation_enabled:
        sd_cfg = tc.get("stagnation_days", {})
        st_cfg = tc.get("stagnation_threshold", {})
        stagnation_days = _suggest_int(trial, "time_stagnation_days", sd_cfg, te.stagnation_days)
        stagnation_threshold = _suggest_float(trial, "time_stagnation_threshold", st_cfg, te.stagnation_threshold)

    return TimeExitGene(
        max_days_enabled=te.max_days_enabled,
        max_days=max_days,
        weekday_exit_enabled=te.weekday_exit_enabled,
        weekday=weekday,
        eow_enabled=te.eow_enabled,
        eom_enabled=te.eom_enabled,
        stagnation_enabled=te.stagnation_enabled,
        stagnation_days=stagnation_days,
        stagnation_threshold=stagnation_threshold,
    )


def _suggest_filters(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    genome: StrategyGenome,
    meta: StrategyMeta,
) -> tuple[FilterGene, ...]:
    """Suggest parameters for filter slots.

    Args:
        trial: Optuna trial.
        genome: Original genome.
        meta: Strategy metadata.

    Returns:
        Tuple of FilterGene with suggested parameters.
    """
    filters: list[FilterGene] = []
    for i, gene in enumerate(genome.filters):
        if not gene.enabled:
            filters.append(gene)
            continue
        prefix = f"filter_{i}"
        params = _suggest_filter_params(trial, prefix, gene.filter_name, meta.filter_params)
        filters.append(
            FilterGene(
                enabled=gene.enabled,
                filter_name=gene.filter_name,
                params=params,
            )
        )
    return tuple(filters)


# ---------------------------------------------------------------------------
# Low-level suggestion helpers
# ---------------------------------------------------------------------------


def _suggest_indicator_params(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    prefix: str,
    indicator_name: str,
    all_params: dict[str, tuple[ParamMeta, ...]],
) -> dict[str, int | float]:
    """Suggest indicator-specific parameters.

    Args:
        trial: Optuna trial.
        prefix: Parameter name prefix (e.g. "entry_0").
        indicator_name: Name of the indicator.
        all_params: Mapping of indicator name to parameter metadata.

    Returns:
        Dictionary of suggested parameter values.
    """
    param_metas = all_params.get(indicator_name, ())
    params: dict[str, int | float] = {}
    for pm in param_metas:
        name = f"{prefix}_{pm.name}"
        if pm.param_type == "int":
            params[pm.name] = trial.suggest_int(name, int(pm.min_value), int(pm.max_value))
        else:
            params[pm.name] = trial.suggest_float(name, float(pm.min_value), float(pm.max_value))
    return params


def _suggest_filter_params(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    prefix: str,
    filter_name: str,
    all_params: dict[str, tuple[ParamMeta, ...]],
) -> dict[str, int | float]:
    """Suggest filter-specific parameters.

    Args:
        trial: Optuna trial.
        prefix: Parameter name prefix (e.g. "filter_0").
        filter_name: Name of the filter.
        all_params: Mapping of filter name to parameter metadata.

    Returns:
        Dictionary of suggested parameter values.
    """
    param_metas = all_params.get(filter_name, ())
    params: dict[str, int | float] = {}
    for pm in param_metas:
        name = f"{prefix}_{pm.name}"
        if pm.param_type == "int":
            params[pm.name] = trial.suggest_int(name, int(pm.min_value), int(pm.max_value))
        else:
            params[pm.name] = trial.suggest_float(name, float(pm.min_value), float(pm.max_value))
    return params


def _suggest_threshold(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    prefix: str,
    gene: IndicatorGene | IndicatorExitGene,
    meta: StrategyMeta,
) -> float:
    """Suggest a threshold value for an indicator gene.

    Args:
        trial: Optuna trial.
        prefix: Parameter name prefix.
        gene: The indicator gene.
        meta: Strategy metadata.

    Returns:
        Suggested threshold value.
    """
    outputs = meta.indicator_outputs.get(gene.indicator_name, ())
    output = next((o for o in outputs if o.key == gene.output_key), None)
    if output is None or output.threshold_min is None or output.threshold_max is None:
        return gene.threshold
    return trial.suggest_float(f"{prefix}_threshold", output.threshold_min, output.threshold_max)


def _suggest_float(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    name: str,
    cfg: dict[str, Any],
    default: float,
) -> float:
    """Suggest a float value within a config range.

    Args:
        trial: Optuna trial.
        name: Parameter name.
        cfg: Config dict with min/max keys.
        default: Default value if config is empty.

    Returns:
        Suggested float value.
    """
    if not cfg:
        return default
    min_v = float(cfg.get("min", default))
    max_v = float(cfg.get("max", default))
    if min_v >= max_v:
        return default
    return trial.suggest_float(name, min_v, max_v)


def _suggest_int(
    trial: optuna.Trial | optuna.trial.FrozenTrial,
    name: str,
    cfg: dict[str, Any],
    default: int,
) -> int:
    """Suggest an integer value within a config range.

    Args:
        trial: Optuna trial.
        name: Parameter name.
        cfg: Config dict with min/max keys.
        default: Default value if config is empty.

    Returns:
        Suggested integer value.
    """
    if not cfg:
        return default
    min_v = int(cfg.get("min", default))
    max_v = int(cfg.get("max", default))
    if min_v >= max_v:
        return default
    return trial.suggest_int(name, min_v, max_v)


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------


def _make_log_callback(log_interval: int) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Create an Optuna callback that logs progress periodically.

    Args:
        log_interval: Log every N completed trials.

    Returns:
        Callback function for study.optimize().
    """

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Log progress at regular intervals."""
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed % log_interval == 0:
            LOGGER.info(
                "Trial %d | best=%.4f | completed=%d",
                trial.number,
                study.best_value,
                completed,
            )

    return callback
