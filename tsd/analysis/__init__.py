"""Results analysis and robustness checks."""

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
    "PermutationTestResult",
    "RobustnessConfig",
    "RobustnessResult",
    "assess_robustness",
    "load_robustness_config",
    "run_bootstrap_ci",
    "run_permutation_test",
]
