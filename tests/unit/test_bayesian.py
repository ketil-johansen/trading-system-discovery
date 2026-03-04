"""Unit tests for the Bayesian optimizer module."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest

from tsd.optimization.bayesian import (
    BayesianConfig,
    BayesianResult,
    _aggregate_metrics,
    _make_objective,
    _suggest_genome,
    load_bayesian_config,
    run_bayesian,
)
from tsd.optimization.fitness import FitnessConfig
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, EvaluatorConfig
from tsd.strategy.genome import (
    StrategyGenome,
    StrategyMeta,
    load_strategy_config,
    random_genome,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir() -> Path:
    """Return the path to the config directory."""
    return Path("config")


@pytest.fixture
def meta(config_dir: Path) -> StrategyMeta:
    """Load strategy metadata from config files."""
    return load_strategy_config(config_dir)


@pytest.fixture
def genome(meta: StrategyMeta) -> StrategyGenome:
    """Generate a deterministic genome."""
    return random_genome(meta, rng=random.Random(42))


def _make_backtest_result(
    num_trades: int = 50,
    num_wins: int = 42,
    net_profit: float = 500.0,
) -> BacktestResult:
    """Create a BacktestResult with specified metrics."""
    num_losses = num_trades - num_wins
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0
    return BacktestResult(
        trades=(),
        metrics=BacktestMetrics(
            num_trades=num_trades,
            num_wins=num_wins,
            num_losses=num_losses,
            win_rate=win_rate,
            net_profit=net_profit,
            gross_profit=max(net_profit, 0.0),
            gross_loss=min(net_profit, 0.0),
            profit_factor=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            win_loss_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            avg_holding_days=0.0,
            longest_win_streak=0,
            longest_loss_streak=0,
            expectancy_per_trade=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# TestBayesianConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBayesianConfig:
    """Tests for BayesianConfig dataclass."""

    def test_defaults(self) -> None:
        """Verify default values."""
        cfg = BayesianConfig()
        assert cfg.n_trials == 500
        assert cfg.random_seed == 42
        assert cfg.checkpoint_dir == Path("results/checkpoints")
        assert cfg.study_name == "tsd_bayesian"
        assert cfg.log_interval == 10

    def test_load_from_env(self) -> None:
        """Load config from environment variables."""
        env = {
            "TSD_BAYESIAN_N_TRIALS": "100",
            "TSD_BAYESIAN_RANDOM_SEED": "99",
            "TSD_BAYESIAN_CHECKPOINT_DIR": "/tmp/test_checkpoints",
            "TSD_BAYESIAN_STUDY_NAME": "my_study",
            "TSD_BAYESIAN_LOG_INTERVAL": "5",
        }
        with patch.dict("os.environ", env):
            cfg = load_bayesian_config()
        assert cfg.n_trials == 100
        assert cfg.random_seed == 99
        assert cfg.checkpoint_dir == Path("/tmp/test_checkpoints")
        assert cfg.study_name == "my_study"
        assert cfg.log_interval == 5

    def test_frozen(self) -> None:
        """Cannot mutate frozen config."""
        cfg = BayesianConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.n_trials = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestSuggestGenome
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuggestGenome:
    """Tests for _suggest_genome parameter suggestion."""

    def test_preserves_structure(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        """Indicator names, comparisons, enabled flags stay unchanged."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        suggested = _suggest_genome(trial, genome, meta)

        # Enabled flags preserved
        for orig, new in zip(genome.entry_indicators, suggested.entry_indicators, strict=True):
            assert orig.enabled == new.enabled
            assert orig.indicator_name == new.indicator_name
            assert orig.output_key == new.output_key
            assert orig.comparison == new.comparison

        # Combination logic preserved
        assert genome.combination_logic == suggested.combination_logic

        # Limit exit enabled flags preserved
        assert genome.limit_exits.stop_loss.enabled == suggested.limit_exits.stop_loss.enabled
        assert genome.limit_exits.take_profit.enabled == suggested.limit_exits.take_profit.enabled

        # Indicator exit structure preserved
        for orig, new in zip(genome.indicator_exits, suggested.indicator_exits, strict=True):
            assert orig.enabled == new.enabled
            assert orig.indicator_name == new.indicator_name
            assert orig.opposite_entry == new.opposite_entry

        # Filter structure preserved
        for orig, new in zip(genome.filters, suggested.filters, strict=True):
            assert orig.enabled == new.enabled
            assert orig.filter_name == new.filter_name

    def test_respects_param_bounds(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        """Suggested values fall within meta param ranges."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        suggested = _suggest_genome(trial, genome, meta)

        for i, gene in enumerate(suggested.entry_indicators):
            if not gene.enabled:
                continue
            param_metas = meta.indicator_params.get(gene.indicator_name, ())
            for pm in param_metas:
                if pm.name in gene.params:
                    val = gene.params[pm.name]
                    assert pm.min_value <= val <= pm.max_value, (
                        f"entry_{i}.{pm.name}={val} out of [{pm.min_value}, {pm.max_value}]"
                    )

    def test_disabled_slots_unchanged(self, meta: StrategyMeta) -> None:
        """Disabled slots keep original params exactly."""
        # Create a genome with all slots disabled except one entry
        rng = random.Random(123)
        genome = random_genome(meta, rng=rng)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        suggested = _suggest_genome(trial, genome, meta)

        for orig, new in zip(genome.entry_indicators, suggested.entry_indicators, strict=True):
            if not orig.enabled:
                assert orig == new

        for orig, new in zip(genome.indicator_exits, suggested.indicator_exits, strict=True):
            if not orig.enabled:
                assert orig == new

        for orig, new in zip(genome.filters, suggested.filters, strict=True):
            if not orig.enabled:
                assert orig == new


# ---------------------------------------------------------------------------
# TestAggregateMetrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAggregateMetrics:
    """Tests for _aggregate_metrics."""

    def test_single_stock(self) -> None:
        """Single stock result passes through."""
        result = _make_backtest_result(num_trades=40, num_wins=35, net_profit=300.0)
        agg = _aggregate_metrics([result])
        assert agg.num_trades == 40
        assert agg.num_wins == 35
        assert agg.num_losses == 5
        assert agg.win_rate == pytest.approx(35 / 40)
        assert agg.net_profit == pytest.approx(300.0)

    def test_multi_stock_aggregation(self) -> None:
        """Multiple stocks aggregate correctly."""
        r1 = _make_backtest_result(num_trades=30, num_wins=25, net_profit=200.0)
        r2 = _make_backtest_result(num_trades=20, num_wins=18, net_profit=150.0)
        agg = _aggregate_metrics([r1, r2])
        assert agg.num_trades == 50
        assert agg.num_wins == 43
        assert agg.num_losses == 7
        assert agg.win_rate == pytest.approx(43 / 50)
        assert agg.net_profit == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# TestObjective
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestObjective:
    """Tests for the objective function."""

    def test_returns_fitness_value(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        """Objective returns a float fitness value."""
        mock_result = _make_backtest_result(num_trades=50, num_wins=45, net_profit=500.0)

        with patch("tsd.optimization.bayesian.run_backtest", return_value=mock_result):
            objective = _make_objective(
                genome,
                meta,
                {"AAPL": None},  # type: ignore[dict-item]
                {},
                EvaluatorConfig(),
                FitnessConfig(),
            )
            study = optuna.create_study(direction="maximize")
            trial = study.ask()
            fitness = objective(trial)

        assert isinstance(fitness, float)
        assert fitness == pytest.approx(45 / 50)

    def test_zero_fitness_when_gate_fails(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        """Insufficient trades produce 0.0 fitness."""
        mock_result = _make_backtest_result(num_trades=5, num_wins=4, net_profit=50.0)

        with patch("tsd.optimization.bayesian.run_backtest", return_value=mock_result):
            objective = _make_objective(
                genome,
                meta,
                {"AAPL": None},  # type: ignore[dict-item]
                {},
                EvaluatorConfig(),
                FitnessConfig(min_trades=30),
            )
            study = optuna.create_study(direction="maximize")
            trial = study.ask()
            fitness = objective(trial)

        assert fitness == 0.0


# ---------------------------------------------------------------------------
# TestRunBayesian
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunBayesian:
    """Tests for the run_bayesian function."""

    def test_small_run_returns_result(self, genome: StrategyGenome, meta: StrategyMeta, tmp_path: Path) -> None:
        """A small run with mocked eval returns a valid BayesianResult."""
        mock_result = _make_backtest_result(num_trades=50, num_wins=45, net_profit=500.0)

        config = BayesianConfig(n_trials=5, checkpoint_dir=tmp_path)

        with patch("tsd.optimization.bayesian.run_backtest", return_value=mock_result):
            result = run_bayesian(
                genome,
                meta,
                {"AAPL": None},  # type: ignore[dict-item]
                {},
                bayesian_config=config,
            )

        assert isinstance(result, BayesianResult)
        assert result.trials_run == 5
        assert result.best_fitness > 0.0
        assert isinstance(result.best_genome, StrategyGenome)

    def test_resume_adds_trials(self, genome: StrategyGenome, meta: StrategyMeta, tmp_path: Path) -> None:
        """Resuming a study adds more trials."""
        mock_result = _make_backtest_result(num_trades=50, num_wins=45, net_profit=500.0)
        config = BayesianConfig(n_trials=3, checkpoint_dir=tmp_path)

        with patch("tsd.optimization.bayesian.run_backtest", return_value=mock_result):
            result1 = run_bayesian(
                genome,
                meta,
                {"AAPL": None},  # type: ignore[dict-item]
                {},
                bayesian_config=config,
                resume=False,
            )
            result2 = run_bayesian(
                genome,
                meta,
                {"AAPL": None},  # type: ignore[dict-item]
                {},
                bayesian_config=config,
                resume=True,
            )

        assert result1.trials_run == 3
        assert result2.trials_run >= 6

    def test_result_has_best_params(self, genome: StrategyGenome, meta: StrategyMeta, tmp_path: Path) -> None:
        """Result contains non-empty best_params dict."""
        mock_result = _make_backtest_result(num_trades=50, num_wins=45, net_profit=500.0)
        config = BayesianConfig(n_trials=3, checkpoint_dir=tmp_path)

        with patch("tsd.optimization.bayesian.run_backtest", return_value=mock_result):
            result = run_bayesian(
                genome,
                meta,
                {"AAPL": None},  # type: ignore[dict-item]
                {},
                bayesian_config=config,
            )

        assert isinstance(result.best_params, dict)
        assert len(result.best_params) > 0
