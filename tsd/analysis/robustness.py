"""Monte Carlo permutation tests and bootstrap confidence intervals.

Provides statistical significance testing for trading strategy returns
to distinguish genuine edge from random chance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tsd.config import env_float, env_int
from tsd.strategy.evaluator import BacktestResult

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessConfig:
    """Configuration for statistical robustness testing.

    Attributes:
        mc_n_permutations: Number of Monte Carlo permutations.
        mc_alpha: Significance level for permutation tests.
        bs_n_resamples: Number of bootstrap resamples.
        bs_confidence_level: Confidence level for bootstrap CIs.
        bs_win_rate_threshold: Minimum bootstrap lower bound for win rate.
        random_seed: Random seed for reproducibility.
        min_trades: Minimum trades required (skip if fewer).
    """

    mc_n_permutations: int = 10_000
    mc_alpha: float = 0.05
    bs_n_resamples: int = 10_000
    bs_confidence_level: float = 0.95
    bs_win_rate_threshold: float = 0.55
    random_seed: int = 42
    min_trades: int = 10


def load_robustness_config() -> RobustnessConfig:
    """Load robustness configuration from environment variables.

    Returns:
        RobustnessConfig with values from TSD_ROBUSTNESS_* environment variables.
    """
    return RobustnessConfig(
        mc_n_permutations=env_int("TSD_ROBUSTNESS_MC_N_PERMUTATIONS", 10_000),
        mc_alpha=env_float("TSD_ROBUSTNESS_MC_ALPHA", 0.05),
        bs_n_resamples=env_int("TSD_ROBUSTNESS_BS_N_RESAMPLES", 10_000),
        bs_confidence_level=env_float("TSD_ROBUSTNESS_BS_CONFIDENCE_LEVEL", 0.95),
        bs_win_rate_threshold=env_float("TSD_ROBUSTNESS_BS_WIN_RATE_THRESHOLD", 0.55),
        random_seed=env_int("TSD_ROBUSTNESS_RANDOM_SEED", 42),
        min_trades=env_int("TSD_ROBUSTNESS_MIN_TRADES", 10),
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationTestResult:
    """Result from a single Monte Carlo permutation test.

    Attributes:
        statistic_name: Name of the statistic tested.
        actual_value: Observed statistic on real returns.
        p_value: Proportion of permutations >= actual (plus correction).
        n_permutations: Number of permutations performed.
        significant: Whether p_value < alpha.
    """

    statistic_name: str
    actual_value: float
    p_value: float
    n_permutations: int
    significant: bool


@dataclass(frozen=True)
class BootstrapCIResult:
    """Result from a bootstrap confidence interval computation.

    Attributes:
        statistic_name: Name of the statistic.
        actual_value: Observed statistic on real returns.
        lower_bound: Lower bound of the confidence interval.
        upper_bound: Upper bound of the confidence interval.
        confidence_level: Confidence level used.
        n_resamples: Number of bootstrap resamples.
    """

    statistic_name: str
    actual_value: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    n_resamples: int


@dataclass(frozen=True)
class RobustnessResult:
    """Combined result from all robustness tests.

    Attributes:
        permutation_tests: Results from all permutation tests.
        bootstrap_cis: Results from all bootstrap CIs.
        passed: Whether all robustness criteria are met.
        num_trades: Number of trades analysed.
        skipped: Whether testing was skipped due to too few trades.
    """

    permutation_tests: tuple[PermutationTestResult, ...]
    bootstrap_cis: tuple[BootstrapCIResult, ...]
    passed: bool
    num_trades: int
    skipped: bool


# ---------------------------------------------------------------------------
# Statistic functions
# ---------------------------------------------------------------------------


def _compute_win_rate(returns: np.ndarray) -> float:  # type: ignore[type-arg]
    """Fraction of returns greater than zero."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def _compute_net_profit(returns: np.ndarray) -> float:  # type: ignore[type-arg]
    """Sum of all returns."""
    return float(np.sum(returns))


def _compute_sharpe(returns: np.ndarray) -> float:  # type: ignore[type-arg]
    """Mean divided by standard deviation (0.0 if std is zero)."""
    if len(returns) < 2:  # noqa: PLR2004
        return 0.0
    std = float(np.std(returns))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns)) / std


def _compute_expectancy(returns: np.ndarray) -> float:  # type: ignore[type-arg]
    """Mean return per trade."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def run_permutation_test(
    returns: np.ndarray,  # type: ignore[type-arg]
    statistic_fn: Callable[[np.ndarray], float],  # type: ignore[type-arg]
    statistic_name: str,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Run a Monte Carlo sign-flip permutation test.

    Tests H0: returns are symmetric around zero (no edge).
    For each permutation, randomly negate each return, then compute
    the test statistic. The p-value is the fraction of permuted
    statistics >= the actual statistic (with continuity correction).

    Args:
        returns: Array of trade returns.
        statistic_fn: Function computing the test statistic from returns.
        statistic_name: Human-readable name for the statistic.
        n_permutations: Number of random permutations.
        alpha: Significance level.
        rng: Numpy random generator for reproducibility.

    Returns:
        PermutationTestResult with p-value and significance flag.
    """
    if rng is None:
        rng = np.random.default_rng()

    actual = statistic_fn(returns)
    n = len(returns)

    # Generate sign matrix: (n_permutations, n) of +1/-1
    signs = rng.choice(np.array([-1, 1]), size=(n_permutations, n))
    permuted_returns = signs * returns[np.newaxis, :]

    # Compute statistic for each permutation
    permuted_stats = np.array([statistic_fn(row) for row in permuted_returns])

    # p-value with continuity correction
    count_ge = int(np.sum(permuted_stats >= actual))
    p_value = (count_ge + 1) / (n_permutations + 1)

    return PermutationTestResult(
        statistic_name=statistic_name,
        actual_value=actual,
        p_value=p_value,
        n_permutations=n_permutations,
        significant=p_value < alpha,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def run_bootstrap_ci(
    returns: np.ndarray,  # type: ignore[type-arg]
    statistic_fn: Callable[[np.ndarray], float],  # type: ignore[type-arg]
    statistic_name: str,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapCIResult:
    """Compute a percentile-based bootstrap confidence interval.

    Resamples returns with replacement, computes the statistic for each
    resample, and takes percentiles for the CI bounds.

    Args:
        returns: Array of trade returns.
        statistic_fn: Function computing the statistic from returns.
        statistic_name: Human-readable name for the statistic.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        rng: Numpy random generator for reproducibility.

    Returns:
        BootstrapCIResult with lower and upper bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    actual = statistic_fn(returns)
    n = len(returns)

    # Generate resample indices: (n_resamples, n)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled = returns[indices]

    # Compute statistic for each resample
    bootstrap_stats = np.array([statistic_fn(row) for row in resampled])

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_stats, alpha / 2 * 100))
    upper = float(np.percentile(bootstrap_stats, (1 - alpha / 2) * 100))

    return BootstrapCIResult(
        statistic_name=statistic_name,
        actual_value=actual,
        lower_bound=lower,
        upper_bound=upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_returns(result: BacktestResult) -> np.ndarray:  # type: ignore[type-arg]
    """Extract net_return_pct from BacktestResult trades.

    Args:
        result: Backtest result with trade records.

    Returns:
        Numpy array of net return percentages.
    """
    return np.array([t.net_return_pct for t in result.trades])


def _run_all_permutation_tests(
    returns: np.ndarray,  # type: ignore[type-arg]
    config: RobustnessConfig,
    rng: np.random.Generator,
) -> tuple[PermutationTestResult, ...]:
    """Run permutation tests on win_rate, net_profit, and sharpe.

    Args:
        returns: Array of trade returns.
        config: Robustness configuration.
        rng: Numpy random generator.

    Returns:
        Tuple of three PermutationTestResult objects.
    """
    tests = [
        ("win_rate", _compute_win_rate),
        ("net_profit", _compute_net_profit),
        ("sharpe", _compute_sharpe),
    ]
    results: list[PermutationTestResult] = []
    for name, fn in tests:
        result = run_permutation_test(
            returns=returns,
            statistic_fn=fn,
            statistic_name=name,
            n_permutations=config.mc_n_permutations,
            alpha=config.mc_alpha,
            rng=rng,
        )
        results.append(result)
    return tuple(results)


def _run_all_bootstrap_cis(
    returns: np.ndarray,  # type: ignore[type-arg]
    config: RobustnessConfig,
    rng: np.random.Generator,
) -> tuple[BootstrapCIResult, ...]:
    """Run bootstrap CIs on win_rate, net_profit, sharpe, and expectancy.

    Args:
        returns: Array of trade returns.
        config: Robustness configuration.
        rng: Numpy random generator.

    Returns:
        Tuple of four BootstrapCIResult objects.
    """
    statistics = [
        ("win_rate", _compute_win_rate),
        ("net_profit", _compute_net_profit),
        ("sharpe", _compute_sharpe),
        ("expectancy", _compute_expectancy),
    ]
    results: list[BootstrapCIResult] = []
    for name, fn in statistics:
        result = run_bootstrap_ci(
            returns=returns,
            statistic_fn=fn,
            statistic_name=name,
            n_resamples=config.bs_n_resamples,
            confidence_level=config.bs_confidence_level,
            rng=rng,
        )
        results.append(result)
    return tuple(results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_robustness(
    result: BacktestResult,
    config: RobustnessConfig | None = None,
) -> RobustnessResult:
    """Run full robustness assessment on a backtest result.

    Performs Monte Carlo permutation tests and bootstrap confidence
    intervals on the trade returns. Skips gracefully if too few trades.

    Pass criteria:
    - All permutation tests must be significant (p < alpha).
    - Bootstrap win_rate lower bound must exceed threshold.

    Args:
        result: Backtest result with trade records.
        config: Robustness configuration. Uses defaults if None.

    Returns:
        RobustnessResult with all test results and pass/fail.
    """
    config = config or RobustnessConfig()
    returns = _extract_returns(result)
    num_trades = len(returns)

    if num_trades < config.min_trades:
        LOGGER.warning(
            "Too few trades (%d < %d) for robustness testing, skipping",
            num_trades,
            config.min_trades,
        )
        return RobustnessResult(
            permutation_tests=(),
            bootstrap_cis=(),
            passed=False,
            num_trades=num_trades,
            skipped=True,
        )

    rng = np.random.default_rng(config.random_seed)

    perm_results = _run_all_permutation_tests(returns, config, rng)
    bs_results = _run_all_bootstrap_cis(returns, config, rng)

    all_perm_significant = all(pr.significant for pr in perm_results)

    # Find win_rate bootstrap CI
    wr_ci = next(ci for ci in bs_results if ci.statistic_name == "win_rate")
    win_rate_above_threshold = wr_ci.lower_bound > config.bs_win_rate_threshold

    passed = all_perm_significant and win_rate_above_threshold

    LOGGER.info(
        "Robustness: passed=%s, perm_all_sig=%s, wr_lower=%.4f>%.4f=%s",
        passed,
        all_perm_significant,
        wr_ci.lower_bound,
        config.bs_win_rate_threshold,
        win_rate_above_threshold,
    )

    return RobustnessResult(
        permutation_tests=perm_results,
        bootstrap_cis=bs_results,
        passed=passed,
        num_trades=num_trades,
        skipped=False,
    )
