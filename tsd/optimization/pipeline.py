"""Staged GA to Optuna optimization pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from tsd.config import env_bool, env_str
from tsd.optimization.bayesian import BayesianConfig, BayesianResult, run_bayesian
from tsd.optimization.fitness import FitnessConfig
from tsd.optimization.ga import GAConfig, GAResult, run_ga
from tsd.strategy.evaluator import EvaluatorConfig
from tsd.strategy.genome import OutputMeta, StrategyGenome, StrategyMeta

LOGGER = logging.getLogger(__name__)

VALID_MODES = ("ga_only", "bayesian_only", "both")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the staged optimization pipeline.

    Attributes:
        mode: Pipeline mode — "ga_only", "bayesian_only", or "both".
        resume: Whether to resume from checkpoints.
    """

    mode: str = "both"
    resume: bool = False


@dataclass(frozen=True)
class PipelineResult:
    """Result of a staged optimization pipeline run.

    Attributes:
        mode: Which pipeline mode was used.
        best_genome: Final best strategy genome.
        best_fitness: Final best fitness value.
        ga_result: GA stage result, None if bayesian_only mode.
        bayesian_result: Bayesian stage result, None if ga_only mode.
        top_genomes: Top N genomes with fitness values from the GA hall of fame.
    """

    mode: str
    best_genome: StrategyGenome
    best_fitness: float
    ga_result: GAResult | None
    bayesian_result: BayesianResult | None
    top_genomes: tuple[tuple[StrategyGenome, float], ...] = ()


def load_pipeline_config() -> PipelineConfig:
    """Load pipeline configuration from environment variables.

    Returns:
        PipelineConfig with values from TSD_PIPELINE_* environment variables.
    """
    return PipelineConfig(
        mode=env_str("TSD_PIPELINE_MODE", "both"),
        resume=env_bool("TSD_PIPELINE_RESUME", False),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(  # noqa: PLR0913
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    pipeline_config: PipelineConfig | None = None,
    ga_config: GAConfig | None = None,
    bayesian_config: BayesianConfig | None = None,
    eval_config: EvaluatorConfig | None = None,
    fitness_config: FitnessConfig | None = None,
    seed_genome: StrategyGenome | None = None,
) -> PipelineResult:
    """Run the staged optimization pipeline.

    Stage A (GA) discovers strategy structure; Stage B (Bayesian) fine-tunes
    parameters. Supports three modes: ga_only, bayesian_only, both.

    Args:
        meta: Strategy metadata describing the parameter space.
        stocks_data: Mapping of stock ticker to OHLCV DataFrame.
        indicator_outputs: Mapping of indicator name to output metadata.
        pipeline_config: Pipeline configuration. Uses defaults if None.
        ga_config: GA configuration. Uses defaults if None.
        bayesian_config: Bayesian optimizer configuration. Uses defaults if None.
        eval_config: Evaluator configuration. Uses defaults if None.
        fitness_config: Fitness configuration. Uses defaults if None.
        seed_genome: Seed genome for bayesian_only mode.

    Returns:
        PipelineResult with the best genome, fitness, and stage results.

    Raises:
        ValueError: If mode is invalid or bayesian_only without seed_genome.
    """
    pipeline_config = pipeline_config or PipelineConfig()
    mode = pipeline_config.mode
    resume = pipeline_config.resume

    _validate_mode(mode)
    LOGGER.info("Pipeline starting in '%s' mode", mode)

    if mode == "ga_only":
        return _run_ga_only(meta, stocks_data, indicator_outputs, ga_config, eval_config, fitness_config, resume)

    if mode == "bayesian_only":
        if seed_genome is None:
            msg = "bayesian_only mode requires a seed_genome"
            raise ValueError(msg)
        return _run_bayesian_only(
            seed_genome, meta, stocks_data, indicator_outputs, bayesian_config, eval_config, fitness_config, resume
        )

    # mode == "both"
    return _run_both(
        meta, stocks_data, indicator_outputs, ga_config, bayesian_config, eval_config, fitness_config, resume
    )


# ---------------------------------------------------------------------------
# Mode dispatch helpers
# ---------------------------------------------------------------------------


def _run_ga_only(
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    ga_config: GAConfig | None,
    eval_config: EvaluatorConfig | None,
    fitness_config: FitnessConfig | None,
    resume: bool,
) -> PipelineResult:
    """Run GA-only mode."""
    ga_result = _run_ga_stage(meta, stocks_data, indicator_outputs, ga_config, eval_config, fitness_config, resume)
    LOGGER.info("Pipeline complete (ga_only): best_fitness=%.4f", ga_result.best_fitness)
    return PipelineResult(
        mode="ga_only",
        best_genome=ga_result.best_genome,
        best_fitness=ga_result.best_fitness,
        ga_result=ga_result,
        bayesian_result=None,
        top_genomes=ga_result.top_genomes,
    )


def _run_bayesian_only(  # noqa: PLR0913
    seed_genome: StrategyGenome,
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    bayesian_config: BayesianConfig | None,
    eval_config: EvaluatorConfig | None,
    fitness_config: FitnessConfig | None,
    resume: bool,
) -> PipelineResult:
    """Run Bayesian-only mode."""
    bayesian_result = _run_bayesian_stage(
        seed_genome, meta, stocks_data, indicator_outputs, bayesian_config, eval_config, fitness_config, resume
    )
    LOGGER.info("Pipeline complete (bayesian_only): best_fitness=%.4f", bayesian_result.best_fitness)
    return PipelineResult(
        mode="bayesian_only",
        best_genome=bayesian_result.best_genome,
        best_fitness=bayesian_result.best_fitness,
        ga_result=None,
        bayesian_result=bayesian_result,
    )


def _run_both(  # noqa: PLR0913
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    ga_config: GAConfig | None,
    bayesian_config: BayesianConfig | None,
    eval_config: EvaluatorConfig | None,
    fitness_config: FitnessConfig | None,
    resume: bool,
) -> PipelineResult:
    """Run both stages: GA then Bayesian."""
    ga_result = _run_ga_stage(meta, stocks_data, indicator_outputs, ga_config, eval_config, fitness_config, resume)
    LOGGER.info("Stage A complete, feeding best genome to Stage B")

    bayesian_result = _run_bayesian_stage(
        ga_result.best_genome,
        meta,
        stocks_data,
        indicator_outputs,
        bayesian_config,
        eval_config,
        fitness_config,
        resume,
    )
    LOGGER.info("Pipeline complete (both): best_fitness=%.4f", bayesian_result.best_fitness)
    return PipelineResult(
        mode="both",
        best_genome=bayesian_result.best_genome,
        best_fitness=bayesian_result.best_fitness,
        ga_result=ga_result,
        bayesian_result=bayesian_result,
        top_genomes=ga_result.top_genomes,
    )


# ---------------------------------------------------------------------------
# Stage wrappers
# ---------------------------------------------------------------------------


def _run_ga_stage(
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    ga_config: GAConfig | None,
    eval_config: EvaluatorConfig | None,
    fitness_config: FitnessConfig | None,
    resume: bool,
) -> GAResult:
    """Run the GA stage.

    Args:
        meta: Strategy metadata.
        stocks_data: Mapping of ticker to OHLCV DataFrame.
        indicator_outputs: Indicator output metadata.
        ga_config: GA configuration.
        eval_config: Evaluator configuration.
        fitness_config: Fitness configuration.
        resume: Whether to resume from checkpoint.

    Returns:
        GAResult from the GA engine.
    """
    LOGGER.info("Starting Stage A (GA)")
    result = run_ga(
        meta=meta,
        stocks_data=stocks_data,
        indicator_outputs=indicator_outputs,
        ga_config=ga_config,
        eval_config=eval_config,
        fitness_config=fitness_config,
        resume=resume,
    )
    LOGGER.info("Stage A (GA) finished: best_fitness=%.4f, generations=%d", result.best_fitness, result.generations_run)
    return result


def _run_bayesian_stage(  # noqa: PLR0913
    genome: StrategyGenome,
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    bayesian_config: BayesianConfig | None,
    eval_config: EvaluatorConfig | None,
    fitness_config: FitnessConfig | None,
    resume: bool,
) -> BayesianResult:
    """Run the Bayesian optimization stage.

    Args:
        genome: Seed genome from GA or user.
        meta: Strategy metadata.
        stocks_data: Mapping of ticker to OHLCV DataFrame.
        indicator_outputs: Indicator output metadata.
        bayesian_config: Bayesian optimizer configuration.
        eval_config: Evaluator configuration.
        fitness_config: Fitness configuration.
        resume: Whether to resume from existing study.

    Returns:
        BayesianResult from the Bayesian optimizer.
    """
    LOGGER.info("Starting Stage B (Bayesian)")
    result = run_bayesian(
        genome=genome,
        meta=meta,
        stocks_data=stocks_data,
        indicator_outputs=indicator_outputs,
        bayesian_config=bayesian_config,
        eval_config=eval_config,
        fitness_config=fitness_config,
        resume=resume,
    )
    LOGGER.info(
        "Stage B (Bayesian) finished: best_fitness=%.4f, trials=%d",
        result.best_fitness,
        result.trials_run,
    )
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_mode(mode: str) -> None:
    """Validate the pipeline mode string.

    Args:
        mode: Pipeline mode to validate.

    Raises:
        ValueError: If mode is not one of VALID_MODES.
    """
    if mode not in VALID_MODES:
        msg = f"Invalid pipeline mode '{mode}'. Must be one of: {VALID_MODES}"
        raise ValueError(msg)
