"""GA (DEAP), Bayesian (Optuna) optimization engines, and walk-forward validation."""

from tsd.optimization.bayesian import (
    BayesianConfig,
    BayesianResult,
    load_bayesian_config,
    run_bayesian,
)
from tsd.optimization.fitness import FitnessConfig, compute_fitness
from tsd.optimization.ga import (
    GAConfig,
    GAResult,
    GenerationStats,
    load_ga_config,
    run_ga,
)
from tsd.optimization.metrics import aggregate_metrics, empty_metrics
from tsd.optimization.pipeline import (
    PipelineConfig,
    PipelineResult,
    load_pipeline_config,
    run_pipeline,
)
from tsd.optimization.walkforward import (
    HoldoutResult,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    WindowResult,
    generate_windows,
    load_walkforward_config,
    run_walkforward,
)

__all__ = [
    "BayesianConfig",
    "BayesianResult",
    "FitnessConfig",
    "GAConfig",
    "GAResult",
    "GenerationStats",
    "HoldoutResult",
    "PipelineConfig",
    "PipelineResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    "WindowResult",
    "aggregate_metrics",
    "compute_fitness",
    "empty_metrics",
    "generate_windows",
    "load_bayesian_config",
    "load_ga_config",
    "load_pipeline_config",
    "load_walkforward_config",
    "run_bayesian",
    "run_ga",
    "run_pipeline",
    "run_walkforward",
]
