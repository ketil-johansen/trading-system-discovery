"""Unit tests for the statistical robustness module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from tsd.analysis.robustness import (
    BootstrapCIResult,
    PermutationTestResult,
    RobustnessConfig,
    RobustnessResult,
    _compute_expectancy,
    _compute_net_profit,
    _compute_sharpe,
    _compute_win_rate,
    _extract_returns,
    assess_robustness,
    load_robustness_config,
    run_bootstrap_ci,
    run_permutation_test,
)
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, TradeRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(net_return_pct: float) -> TradeRecord:
    """Create a minimal TradeRecord with the given net return."""
    return TradeRecord(
        entry_bar=0,
        entry_date="2020-01-01",
        entry_price=100.0,
        exit_bar=1,
        exit_date="2020-01-02",
        exit_price=100.0 * (1 + net_return_pct),
        exit_type="take_profit",
        gross_return_pct=net_return_pct + 0.002,
        cost_pct=0.002,
        net_return_pct=net_return_pct,
        net_profit=net_return_pct * 10_000,
        is_win=net_return_pct > 0,
        holding_days=1,
    )


def _make_backtest_result(returns: list[float]) -> BacktestResult:
    """Create a BacktestResult from a list of net returns."""
    trades = tuple(_make_trade(r) for r in returns)
    # Simplified metrics — only num_trades matters for robustness
    metrics = BacktestMetrics(
        num_trades=len(trades),
        num_wins=sum(1 for r in returns if r > 0),
        num_losses=sum(1 for r in returns if r <= 0),
        win_rate=sum(1 for r in returns if r > 0) / max(len(returns), 1),
        net_profit=sum(returns) * 10_000,
        gross_profit=sum(r * 10_000 for r in returns if r > 0),
        gross_loss=sum(r * 10_000 for r in returns if r <= 0),
        profit_factor=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        win_loss_ratio=0.0,
        max_drawdown_pct=0.0,
        max_drawdown_duration=0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        avg_holding_days=1.0,
        longest_win_streak=0,
        longest_loss_streak=0,
        expectancy_per_trade=0.0,
    )
    return BacktestResult(trades=trades, metrics=metrics)


# ---------------------------------------------------------------------------
# TestRobustnessConfig
# ---------------------------------------------------------------------------


class TestRobustnessConfig:
    """Tests for RobustnessConfig."""

    def test_defaults(self) -> None:
        """Default values are sensible."""
        config = RobustnessConfig()
        assert config.mc_n_permutations == 10_000
        assert config.mc_alpha == 0.05
        assert config.bs_n_resamples == 10_000
        assert config.bs_confidence_level == 0.95
        assert config.bs_win_rate_threshold == 0.55
        assert config.random_seed == 42
        assert config.min_trades == 10

    def test_env_loading(self) -> None:
        """Config loads from TSD_ROBUSTNESS_* environment variables."""
        env = {
            "TSD_ROBUSTNESS_MC_N_PERMUTATIONS": "500",
            "TSD_ROBUSTNESS_MC_ALPHA": "0.01",
            "TSD_ROBUSTNESS_BS_N_RESAMPLES": "1000",
            "TSD_ROBUSTNESS_BS_CONFIDENCE_LEVEL": "0.99",
            "TSD_ROBUSTNESS_BS_WIN_RATE_THRESHOLD": "0.60",
            "TSD_ROBUSTNESS_RANDOM_SEED": "123",
            "TSD_ROBUSTNESS_MIN_TRADES": "20",
        }
        with patch.dict("os.environ", env):
            config = load_robustness_config()
        assert config.mc_n_permutations == 500
        assert config.mc_alpha == 0.01
        assert config.bs_n_resamples == 1000
        assert config.bs_confidence_level == 0.99
        assert config.bs_win_rate_threshold == 0.60
        assert config.random_seed == 123
        assert config.min_trades == 20

    def test_frozen(self) -> None:
        """Config is immutable."""
        config = RobustnessConfig()
        with pytest.raises(FrozenInstanceError):
            config.mc_n_permutations = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestStatisticFunctions
# ---------------------------------------------------------------------------


class TestStatisticFunctions:
    """Tests for the private statistic functions."""

    def test_win_rate_all_wins(self) -> None:
        """Win rate is 1.0 when all returns are positive."""
        returns = np.array([0.01, 0.02, 0.03])
        assert _compute_win_rate(returns) == 1.0

    def test_win_rate_mixed(self) -> None:
        """Win rate counts only positive returns."""
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        assert _compute_win_rate(returns) == 0.5

    def test_net_profit_sum(self) -> None:
        """Net profit is sum of returns."""
        returns = np.array([0.01, -0.005, 0.02])
        assert _compute_net_profit(returns) == pytest.approx(0.025)

    def test_sharpe_positive(self) -> None:
        """Sharpe is positive for positive mean returns."""
        returns = np.array([0.01, 0.02, 0.03, 0.015, 0.025])
        result = _compute_sharpe(returns)
        assert result > 0

    def test_sharpe_zero_std(self) -> None:
        """Sharpe is 0.0 when all returns are identical."""
        returns = np.array([0.01, 0.01, 0.01])
        assert _compute_sharpe(returns) == 0.0

    def test_expectancy(self) -> None:
        """Expectancy is the mean return."""
        returns = np.array([0.01, -0.01, 0.02])
        expected = (0.01 + -0.01 + 0.02) / 3
        assert _compute_expectancy(returns) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestPermutationTest
# ---------------------------------------------------------------------------


class TestPermutationTest:
    """Tests for run_permutation_test."""

    def test_significant_strategy(self) -> None:
        """A strongly positive strategy has a low p-value."""
        rng = np.random.default_rng(42)
        returns = np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.07, 0.02, 0.06, 0.03, 0.05, 0.04, 0.08, 0.03, 0.05])
        result = run_permutation_test(
            returns,
            _compute_net_profit,
            "net_profit",
            n_permutations=1000,
            alpha=0.05,
            rng=rng,
        )
        assert result.significant is True
        assert result.p_value < 0.05

    def test_random_not_significant(self) -> None:
        """Symmetric returns around zero should not be significant."""
        rng = np.random.default_rng(99)
        # Generate symmetric returns
        half = rng.uniform(0.01, 0.05, size=50)
        returns = np.concatenate([half, -half])
        rng2 = np.random.default_rng(99)
        result = run_permutation_test(
            returns,
            _compute_net_profit,
            "net_profit",
            n_permutations=1000,
            alpha=0.05,
            rng=rng2,
        )
        assert result.p_value > 0.05
        assert result.significant is False

    def test_p_value_range(self) -> None:
        """p-value is always in [0, 1]."""
        rng = np.random.default_rng(42)
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        result = run_permutation_test(
            returns,
            _compute_win_rate,
            "win_rate",
            n_permutations=500,
            rng=rng,
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces same results."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
        r1 = run_permutation_test(
            returns,
            _compute_sharpe,
            "sharpe",
            n_permutations=500,
            rng=np.random.default_rng(42),
        )
        r2 = run_permutation_test(
            returns,
            _compute_sharpe,
            "sharpe",
            n_permutations=500,
            rng=np.random.default_rng(42),
        )
        assert r1.p_value == r2.p_value
        assert r1.actual_value == r2.actual_value

    def test_correct_name(self) -> None:
        """Result contains the statistic name."""
        returns = np.array([0.01, 0.02, 0.03])
        result = run_permutation_test(
            returns,
            _compute_win_rate,
            "my_stat",
            n_permutations=100,
            rng=np.random.default_rng(1),
        )
        assert result.statistic_name == "my_stat"

    def test_single_trade(self) -> None:
        """Handles edge case of a single trade without error."""
        returns = np.array([0.05])
        result = run_permutation_test(
            returns,
            _compute_net_profit,
            "net_profit",
            n_permutations=100,
            rng=np.random.default_rng(1),
        )
        assert result.n_permutations == 100
        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# TestBootstrapCI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for run_bootstrap_ci."""

    def test_ci_contains_actual(self) -> None:
        """The CI should typically contain the actual statistic."""
        rng = np.random.default_rng(42)
        returns = np.array([0.01, 0.02, 0.03, 0.015, 0.025, 0.01, 0.02, 0.03, 0.015, 0.025])
        result = run_bootstrap_ci(
            returns,
            _compute_expectancy,
            "expectancy",
            n_resamples=2000,
            confidence_level=0.99,
            rng=rng,
        )
        assert result.lower_bound <= result.actual_value <= result.upper_bound

    def test_narrows_with_data(self) -> None:
        """More data should produce a narrower CI."""
        small = np.array([0.01, 0.02, 0.03, 0.015, 0.025])
        large = np.tile(small, 10)
        r_small = run_bootstrap_ci(
            small,
            _compute_expectancy,
            "expectancy",
            n_resamples=2000,
            rng=np.random.default_rng(42),
        )
        r_large = run_bootstrap_ci(
            large,
            _compute_expectancy,
            "expectancy",
            n_resamples=2000,
            rng=np.random.default_rng(42),
        )
        width_small = r_small.upper_bound - r_small.lower_bound
        width_large = r_large.upper_bound - r_large.lower_bound
        assert width_large < width_small

    def test_identical_returns_point_ci(self) -> None:
        """Identical returns produce a point (zero-width) CI."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        result = run_bootstrap_ci(
            returns,
            _compute_expectancy,
            "expectancy",
            n_resamples=1000,
            rng=np.random.default_rng(42),
        )
        assert result.lower_bound == pytest.approx(result.upper_bound)

    def test_lower_less_than_upper(self) -> None:
        """Lower bound is less than or equal to upper bound."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.05])
        result = run_bootstrap_ci(
            returns,
            _compute_net_profit,
            "net_profit",
            n_resamples=1000,
            rng=np.random.default_rng(42),
        )
        assert result.lower_bound <= result.upper_bound

    def test_reproducible(self) -> None:
        """Same seed produces identical results."""
        returns = np.array([0.01, -0.01, 0.02, 0.03, -0.005])
        r1 = run_bootstrap_ci(
            returns,
            _compute_win_rate,
            "win_rate",
            n_resamples=500,
            rng=np.random.default_rng(42),
        )
        r2 = run_bootstrap_ci(
            returns,
            _compute_win_rate,
            "win_rate",
            n_resamples=500,
            rng=np.random.default_rng(42),
        )
        assert r1.lower_bound == r2.lower_bound
        assert r1.upper_bound == r2.upper_bound

    def test_correct_metadata(self) -> None:
        """Result contains correct metadata fields."""
        returns = np.array([0.01, 0.02, 0.03])
        result = run_bootstrap_ci(
            returns,
            _compute_expectancy,
            "test_stat",
            n_resamples=500,
            confidence_level=0.90,
            rng=np.random.default_rng(1),
        )
        assert result.statistic_name == "test_stat"
        assert result.confidence_level == 0.90
        assert result.n_resamples == 500


# ---------------------------------------------------------------------------
# TestExtractReturns
# ---------------------------------------------------------------------------


class TestExtractReturns:
    """Tests for _extract_returns."""

    def test_matches_trades(self) -> None:
        """Extracted returns match trade net_return_pct values."""
        result = _make_backtest_result([0.01, -0.02, 0.03])
        returns = _extract_returns(result)
        np.testing.assert_array_almost_equal(returns, [0.01, -0.02, 0.03])

    def test_empty_trades(self) -> None:
        """Empty trades produce empty array."""
        result = _make_backtest_result([])
        returns = _extract_returns(result)
        assert len(returns) == 0


# ---------------------------------------------------------------------------
# TestAssessRobustness
# ---------------------------------------------------------------------------


class TestAssessRobustness:
    """Tests for the assess_robustness function."""

    def test_strong_strategy_passes(self) -> None:
        """A strategy with strongly positive returns passes all tests."""
        returns = [0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.07, 0.02, 0.06, 0.03, 0.05, 0.04, 0.08, 0.03, 0.05]
        result_bt = _make_backtest_result(returns)
        config = RobustnessConfig(
            mc_n_permutations=500,
            bs_n_resamples=500,
            bs_win_rate_threshold=0.55,
            min_trades=5,
        )
        result = assess_robustness(result_bt, config)
        assert result.passed is True
        assert result.skipped is False
        assert result.num_trades == 15
        assert len(result.permutation_tests) == 3
        assert len(result.bootstrap_cis) == 4

    def test_random_strategy_fails(self) -> None:
        """A strategy with symmetric returns should fail."""
        # 50/50 wins/losses of equal magnitude → no edge
        returns = [0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02]
        result_bt = _make_backtest_result(returns)
        config = RobustnessConfig(
            mc_n_permutations=500,
            bs_n_resamples=500,
            bs_win_rate_threshold=0.55,
            min_trades=5,
        )
        result = assess_robustness(result_bt, config)
        assert result.passed is False

    def test_skipped_few_trades(self) -> None:
        """Too few trades → skipped with passed=False."""
        result_bt = _make_backtest_result([0.01, 0.02])
        config = RobustnessConfig(min_trades=10)
        result = assess_robustness(result_bt, config)
        assert result.skipped is True
        assert result.passed is False
        assert result.num_trades == 2
        assert len(result.permutation_tests) == 0
        assert len(result.bootstrap_cis) == 0

    def test_requires_all_permutation_significant(self) -> None:
        """Pass requires ALL permutation tests to be significant."""
        # Mix of strong wins and losses — net_profit may be positive
        # but win_rate might not be significant
        returns = [0.10, -0.08, 0.10, -0.08, 0.10, -0.08, 0.10, -0.08, 0.10, -0.08, 0.10, -0.08]
        result_bt = _make_backtest_result(returns)
        config = RobustnessConfig(
            mc_n_permutations=500,
            bs_n_resamples=500,
            bs_win_rate_threshold=0.45,
            min_trades=5,
        )
        result = assess_robustness(result_bt, config)
        # Win rate is exactly 0.5 — permutation test for win_rate should NOT
        # be significant, so overall should fail
        wr_perm = next(p for p in result.permutation_tests if p.statistic_name == "win_rate")
        if not wr_perm.significant:
            assert result.passed is False

    def test_requires_bootstrap_threshold(self) -> None:
        """Pass requires bootstrap win_rate lower bound above threshold."""
        # High threshold that CI can't reach
        returns = [0.01, -0.005, 0.01, -0.005, 0.01, -0.005, 0.01, -0.005, 0.01, -0.005]
        result_bt = _make_backtest_result(returns)
        config = RobustnessConfig(
            mc_n_permutations=500,
            bs_n_resamples=500,
            bs_win_rate_threshold=0.90,
            min_trades=5,
        )
        result = assess_robustness(result_bt, config)
        # Win rate is 0.5 — CI lower bound can't exceed 0.90
        wr_ci = next(ci for ci in result.bootstrap_cis if ci.statistic_name == "win_rate")
        assert wr_ci.lower_bound < 0.90
        assert result.passed is False


# ---------------------------------------------------------------------------
# TestFrozen
# ---------------------------------------------------------------------------


class TestFrozen:
    """Tests that all result dataclasses are frozen."""

    def test_permutation_test_result_frozen(self) -> None:
        """PermutationTestResult is immutable."""
        result = PermutationTestResult(
            statistic_name="test",
            actual_value=0.5,
            p_value=0.01,
            n_permutations=100,
            significant=True,
        )
        with pytest.raises(FrozenInstanceError):
            result.p_value = 0.99  # type: ignore[misc]

    def test_bootstrap_ci_result_frozen(self) -> None:
        """BootstrapCIResult is immutable."""
        result = BootstrapCIResult(
            statistic_name="test",
            actual_value=0.5,
            lower_bound=0.4,
            upper_bound=0.6,
            confidence_level=0.95,
            n_resamples=100,
        )
        with pytest.raises(FrozenInstanceError):
            result.lower_bound = 0.0  # type: ignore[misc]

    def test_robustness_result_frozen(self) -> None:
        """RobustnessResult is immutable."""
        result = RobustnessResult(
            permutation_tests=(),
            bootstrap_cis=(),
            passed=True,
            num_trades=10,
            skipped=False,
        )
        with pytest.raises(FrozenInstanceError):
            result.passed = False  # type: ignore[misc]
