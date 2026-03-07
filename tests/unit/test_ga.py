"""Unit tests for the GA engine module."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest
from deap import base, creator, tools  # type: ignore[import-untyped]

from tsd.optimization.fitness import FitnessConfig
from tsd.optimization.ga import (
    GAConfig,
    GAResult,
    GenerationStats,
    _compute_segment_boundaries,
    _evaluate_individual,
    _gene_mutation,
    _load_checkpoint,
    _save_checkpoint,
    _slot_crossover,
    load_ga_config,
    run_ga,
)
from tsd.optimization.metrics import aggregate_metrics
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, EvaluatorConfig, TradeRecord
from tsd.strategy.genome import (
    StrategyMeta,
    genome_length,
    genome_to_flat,
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
def genome_flat(meta: StrategyMeta) -> list[float]:
    """Generate a deterministic flat genome."""
    genome = random_genome(meta, rng=random.Random(42))
    return genome_to_flat(genome, meta)


def _make_trades(num_trades: int) -> tuple[TradeRecord, ...]:
    """Generate synthetic trades spread across multiple years."""
    years = range(2018, 2024)
    trades = []
    for i in range(num_trades):
        year = list(years)[i % len(years)]
        month = (i % 12) + 1
        is_win = i < num_trades * 0.9
        trades.append(
            TradeRecord(
                entry_bar=i * 10,
                entry_date=f"{year}-{month:02d}-15",
                entry_price=100.0,
                exit_bar=i * 10 + 5,
                exit_date=f"{year}-{month:02d}-20",
                exit_price=102.0 if is_win else 98.0,
                exit_type="take_profit" if is_win else "stop_loss",
                gross_return_pct=0.02 if is_win else -0.02,
                cost_pct=0.003,
                net_return_pct=0.017 if is_win else -0.023,
                net_profit=170.0 if is_win else -230.0,
                is_win=is_win,
                holding_days=5,
            )
        )
    return tuple(trades)


def _make_backtest_result(
    num_trades: int = 50,
    num_wins: int = 42,
    net_profit: float = 500.0,
) -> BacktestResult:
    """Create a BacktestResult with specified metrics."""
    num_losses = num_trades - num_wins
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0
    return BacktestResult(
        trades=_make_trades(num_trades),
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
# TestGAConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGAConfig:
    """Tests for GAConfig dataclass."""

    def test_defaults(self) -> None:
        """Verify default values."""
        cfg = GAConfig()
        assert cfg.population_size == 300
        assert cfg.max_generations == 100
        assert cfg.crossover_prob == 0.5
        assert cfg.mutation_prob == 0.2
        assert cfg.mutation_sigma == 0.1
        assert cfg.elitism_pct == 0.05
        assert cfg.tournament_size == 3
        assert cfg.random_seed == 42
        assert cfg.early_stop_generations == 20
        assert cfg.n_workers == 1

    def test_load_from_env(self) -> None:
        """Load config from environment variables."""
        env = {
            "TSD_GA_POPULATION_SIZE": "50",
            "TSD_GA_MAX_GENERATIONS": "10",
            "TSD_GA_CROSSOVER_PROB": "0.7",
            "TSD_GA_MUTATION_PROB": "0.3",
            "TSD_GA_MUTATION_SIGMA": "0.2",
            "TSD_GA_ELITISM_PCT": "0.10",
            "TSD_GA_TOURNAMENT_SIZE": "5",
            "TSD_GA_RANDOM_SEED": "99",
            "TSD_GA_EARLY_STOP_GENERATIONS": "5",
            "TSD_GA_N_WORKERS": "4",
        }
        with patch.dict("os.environ", env):
            cfg = load_ga_config()
        assert cfg.population_size == 50
        assert cfg.max_generations == 10
        assert cfg.crossover_prob == 0.7
        assert cfg.mutation_prob == 0.3
        assert cfg.mutation_sigma == 0.2
        assert cfg.elitism_pct == 0.10
        assert cfg.tournament_size == 5
        assert cfg.random_seed == 99
        assert cfg.early_stop_generations == 5
        assert cfg.n_workers == 4

    def test_frozen(self) -> None:
        """Cannot mutate GAConfig."""
        cfg = GAConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.population_size = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestSegmentBoundaries
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSegmentBoundaries:
    """Tests for segment boundary computation."""

    def test_boundaries_cover_full_length(self, meta: StrategyMeta) -> None:
        """Last boundary end equals genome_length."""
        boundaries = _compute_segment_boundaries(meta)
        assert boundaries[-1][1] == genome_length(meta)

    def test_boundaries_match_known_structure(self, meta: StrategyMeta) -> None:
        """Number of segments matches expected structure."""
        boundaries = _compute_segment_boundaries(meta)
        expected_count = (
            meta.num_entry_slots
            + 1  # combination logic
            + 1  # limit exits
            + meta.num_indicator_exit_slots
            + 1  # time exits
            + meta.num_filter_slots
        )
        assert len(boundaries) == expected_count

    def test_boundaries_are_contiguous(self, meta: StrategyMeta) -> None:
        """Boundaries cover the genome without gaps or overlaps."""
        boundaries = _compute_segment_boundaries(meta)
        assert boundaries[0][0] == 0
        for i in range(1, len(boundaries)):
            assert boundaries[i][0] == boundaries[i - 1][1]


# ---------------------------------------------------------------------------
# TestSlotCrossover
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlotCrossover:
    """Tests for slot-level crossover."""

    def test_offspring_same_length(self, meta: StrategyMeta) -> None:
        """Crossover preserves chromosome length."""
        rng = random.Random(1)
        g1 = genome_to_flat(random_genome(meta, rng), meta)
        g2 = genome_to_flat(random_genome(meta, rng), meta)
        o1, o2 = _slot_crossover(g1[:], g2[:], meta, 0.5)
        assert len(o1) == len(g1)
        assert len(o2) == len(g2)

    def test_crossover_with_prob_one_swaps_all(self, meta: StrategyMeta) -> None:
        """With prob=1.0, all segments are swapped."""
        rng = random.Random(1)
        g1 = genome_to_flat(random_genome(meta, rng), meta)
        g2 = genome_to_flat(random_genome(meta, rng), meta)
        orig1, orig2 = g1[:], g2[:]
        random.seed(99)
        _slot_crossover(g1, g2, meta, 1.0)
        # After swapping all segments, g1 should equal original g2
        assert g1 == orig2
        assert g2 == orig1

    def test_crossover_with_prob_zero_no_change(self, meta: StrategyMeta) -> None:
        """With prob=0.0, parents are unchanged."""
        rng = random.Random(1)
        g1 = genome_to_flat(random_genome(meta, rng), meta)
        g2 = genome_to_flat(random_genome(meta, rng), meta)
        orig1, orig2 = g1[:], g2[:]
        _slot_crossover(g1, g2, meta, 0.0)
        assert g1 == orig1
        assert g2 == orig2


# ---------------------------------------------------------------------------
# TestGeneMutation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGeneMutation:
    """Tests for gene mutation."""

    def test_mutation_changes_values(self, meta: StrategyMeta, genome_flat: list[float]) -> None:
        """Mutation with high probability should change at least some genes."""
        original = genome_flat[:]
        random.seed(42)
        (_mutated,) = _gene_mutation(genome_flat, meta, prob=1.0, sigma=0.5)
        assert _mutated != original

    def test_mutation_prob_zero_no_change(self, meta: StrategyMeta, genome_flat: list[float]) -> None:
        """Mutation with prob=0.0 leaves individual unchanged."""
        original = genome_flat[:]
        (_mutated,) = _gene_mutation(genome_flat, meta, prob=0.0, sigma=0.5)
        assert _mutated == original


# ---------------------------------------------------------------------------
# TestAggregateMetrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAggregateMetrics:
    """Tests for multi-stock metric aggregation."""

    def test_single_stock(self) -> None:
        """Single stock passthrough."""
        result = _make_backtest_result(num_trades=40, num_wins=32, net_profit=200.0)
        agg = aggregate_metrics([result])
        assert agg.num_trades == 40
        assert agg.num_wins == 32
        assert agg.num_losses == 8
        assert agg.win_rate == pytest.approx(0.8)
        assert agg.net_profit == pytest.approx(200.0)

    def test_multi_stock_aggregation(self) -> None:
        """Sums trades and recomputes win_rate."""
        r1 = _make_backtest_result(num_trades=20, num_wins=16, net_profit=100.0)
        r2 = _make_backtest_result(num_trades=30, num_wins=24, net_profit=200.0)
        agg = aggregate_metrics([r1, r2])
        assert agg.num_trades == 50
        assert agg.num_wins == 40
        assert agg.num_losses == 10
        assert agg.win_rate == pytest.approx(0.8)
        assert agg.net_profit == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# TestEvaluateIndividual
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateIndividual:
    """Tests for individual evaluation."""

    def test_returns_fitness_tuple(self, meta: StrategyMeta, genome_flat: list[float]) -> None:
        """Mocked backtest returns correct fitness shape."""
        mock_result = _make_backtest_result(num_trades=50, num_wins=45, net_profit=500.0)
        with patch("tsd.optimization.ga.run_backtest", return_value=mock_result):
            indicator_outputs = meta.indicator_outputs
            stocks_data = {"AAPL": None}  # type: ignore[dict-item]
            fitness = _evaluate_individual(
                genome_flat,
                meta,
                stocks_data,
                indicator_outputs,
                EvaluatorConfig(),
                FitnessConfig(max_rate=20.0),
            )
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert fitness[0] > 0.0

    def test_zero_fitness_when_gate_fails(self, meta: StrategyMeta, genome_flat: list[float]) -> None:
        """Insufficient trades should produce zero fitness."""
        mock_result = _make_backtest_result(num_trades=5, num_wins=4, net_profit=50.0)
        with patch("tsd.optimization.ga.run_backtest", return_value=mock_result):
            fitness = _evaluate_individual(
                genome_flat,
                meta,
                {"AAPL": None},  # type: ignore[dict-item]
                meta.indicator_outputs,
                EvaluatorConfig(),
                FitnessConfig(min_trades=30),
            )
        assert fitness == (0.0,)


# ---------------------------------------------------------------------------
# TestCheckpointing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """All checkpoint fields are preserved."""
        # Ensure DEAP types exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        ind = creator.Individual([1.0, 2.0, 3.0])
        ind.fitness.values = (0.85,)
        population = [ind]
        hof = tools.HallOfFame(1)
        hof.update(population)
        logbook = [GenerationStats(0, 0.85, 0.1, 0.5, 0.2, 0.9)]
        rng_state = random.getstate()

        path = tmp_path / "cp.pkl"
        _save_checkpoint(path, population, 5, hof, logbook, rng_state)
        loaded = _load_checkpoint(path)

        assert loaded["generation"] == 5
        assert len(loaded["population"]) == 1
        assert list(loaded["population"][0]) == [1.0, 2.0, 3.0]
        assert loaded["population"][0].fitness.values == (0.85,)
        assert len(loaded["halloffame"]) == 1
        assert loaded["logbook"][0].generation == 0
        assert loaded["rng_state"] == rng_state


# ---------------------------------------------------------------------------
# TestRunGA
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunGA:
    """Tests for the main GA run function."""

    def _mock_eval(self, fitness_val: float = 0.85) -> BacktestResult:
        """Create a mock backtest result."""
        return _make_backtest_result(
            num_trades=50,
            num_wins=int(50 * fitness_val),
            net_profit=500.0,
        )

    def test_small_run_returns_result(self, meta: StrategyMeta) -> None:
        """Small GA run with mocked evaluation produces a GAResult."""
        mock_result = self._mock_eval()
        ga_cfg = GAConfig(population_size=10, max_generations=3, random_seed=42)
        with patch("tsd.optimization.ga.run_backtest", return_value=mock_result):
            result = run_ga(
                meta=meta,
                stocks_data={"AAPL": None},  # type: ignore[dict-item]
                indicator_outputs=meta.indicator_outputs,
                ga_config=ga_cfg,
            )
        assert isinstance(result, GAResult)
        assert result.best_fitness >= 0.0
        assert result.generations_run >= 1
        assert len(result.logbook) >= 1
        assert result.best_genome is not None

    def test_early_stopping(self, meta: StrategyMeta) -> None:
        """Constant fitness triggers early stopping."""
        mock_result = _make_backtest_result(num_trades=50, num_wins=42, net_profit=500.0)
        ga_cfg = GAConfig(
            population_size=10,
            max_generations=50,
            early_stop_generations=3,
            random_seed=42,
        )
        with patch("tsd.optimization.ga.run_backtest", return_value=mock_result):
            result = run_ga(
                meta=meta,
                stocks_data={"AAPL": None},  # type: ignore[dict-item]
                indicator_outputs=meta.indicator_outputs,
                ga_config=ga_cfg,
            )
        # Should stop well before 50 generations
        assert result.generations_run < 50

    def test_resume_from_checkpoint(self, meta: StrategyMeta, tmp_path: Path) -> None:
        """Save checkpoint, then resume from it."""
        mock_result = self._mock_eval()
        ga_cfg = GAConfig(
            population_size=10,
            max_generations=3,
            random_seed=42,
            checkpoint_dir=tmp_path,
        )

        # First run
        with patch("tsd.optimization.ga.run_backtest", return_value=mock_result):
            result1 = run_ga(
                meta=meta,
                stocks_data={"AAPL": None},  # type: ignore[dict-item]
                indicator_outputs=meta.indicator_outputs,
                ga_config=ga_cfg,
            )

        # Checkpoint should exist
        assert (tmp_path / "ga_checkpoint.pkl").exists()

        # Resume with more generations
        ga_cfg2 = GAConfig(
            population_size=10,
            max_generations=6,
            random_seed=42,
            checkpoint_dir=tmp_path,
        )
        with patch("tsd.optimization.ga.run_backtest", return_value=mock_result):
            result2 = run_ga(
                meta=meta,
                stocks_data={"AAPL": None},  # type: ignore[dict-item]
                indicator_outputs=meta.indicator_outputs,
                ga_config=ga_cfg2,
                resume=True,
            )

        assert isinstance(result2, GAResult)
        assert result2.generations_run >= result1.generations_run
