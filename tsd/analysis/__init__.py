"""Results analysis and robustness checks."""

from tsd.analysis.reports import (
    FitnessEvolution,
    OptimizationSummary,
    PerformanceReport,
    RobustnessSummary,
    StrategySummary,
    TradeAnalysis,
    WalkForwardSummary,
    WindowSummary,
    generate_report,
    save_report,
)
from tsd.analysis.robustness import (
    BootstrapCIResult,
    PermutationTestResult,
    RobustnessConfig,
    RobustnessResult,
    assess_robustness,
    load_robustness_config,
    run_bootstrap_ci,
    run_permutation_test,
)

__all__ = [
    "BootstrapCIResult",
    "FitnessEvolution",
    "OptimizationSummary",
    "PerformanceReport",
    "PermutationTestResult",
    "RobustnessConfig",
    "RobustnessResult",
    "RobustnessSummary",
    "StrategySummary",
    "TradeAnalysis",
    "WalkForwardSummary",
    "WindowSummary",
    "assess_robustness",
    "generate_report",
    "load_robustness_config",
    "run_bootstrap_ci",
    "run_permutation_test",
    "save_report",
]
