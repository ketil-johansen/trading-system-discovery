"""Unit tests for the performance reporting module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tsd.analysis.reports import (
    _build_fitness_evolution,
    _build_optimization_summary,
    _build_robustness_summary,
    _build_strategy_summary,
    _build_trade_analysis,
    _build_walkforward_summary,
    _describe_exit,
    _describe_filter,
    _describe_indicator,
    generate_report,
    save_report,
)
from tsd.analysis.robustness import (
    BootstrapCIResult,
    PermutationTestResult,
    RobustnessResult,
)
from tsd.optimization.bayesian import BayesianResult
from tsd.optimization.ga import GAResult, GenerationStats
from tsd.optimization.pipeline import PipelineResult
from tsd.optimization.walkforward import (
    HoldoutResult,
    WalkForwardResult,
    WalkForwardWindow,
    WindowResult,
)
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, TradeRecord
from tsd.strategy.genome import (
    BreakevenConfig,
    ChandelierConfig,
    FilterGene,
    IndicatorExitGene,
    IndicatorGene,
    LimitExitGene,
    StopLossConfig,
    StrategyGenome,
    TakeProfitConfig,
    TimeExitGene,
    TrailingStopConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disabled_limit_exits() -> LimitExitGene:
    """Return limit exits with everything disabled."""
    return LimitExitGene(
        stop_loss=StopLossConfig(enabled=False, mode="percent", percent=5.0, atr_multiple=2.0),
        take_profit=TakeProfitConfig(enabled=False, mode="percent", percent=10.0, atr_multiple=3.0),
        trailing_stop=TrailingStopConfig(
            enabled=False,
            mode="percent",
            percent=3.0,
            atr_multiple=2.0,
            activation_percent=5.0,
        ),
        chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
        breakeven=BreakevenConfig(
            enabled=False,
            mode="percent",
            trigger_percent=2.0,
            trigger_atr_multiple=1.0,
        ),
    )


def _disabled_time_exits() -> TimeExitGene:
    """Return time exits with everything disabled."""
    return TimeExitGene(
        max_days_enabled=False,
        max_days=10,
        weekday_exit_enabled=False,
        weekday=4,
        eow_enabled=False,
        eom_enabled=False,
        stagnation_enabled=False,
        stagnation_days=5,
        stagnation_threshold=0.5,
    )


def _minimal_genome() -> StrategyGenome:
    """Build a minimal genome with no indicators or filters."""
    return StrategyGenome(
        entry_indicators=(),
        combination_logic="AND",
        limit_exits=_disabled_limit_exits(),
        indicator_exits=(),
        time_exits=_disabled_time_exits(),
        filters=(),
    )


def _complex_genome() -> StrategyGenome:
    """Build a genome with entry indicators, exits, and filters."""
    return StrategyGenome(
        entry_indicators=(
            IndicatorGene(
                enabled=True,
                indicator_name="sma",
                output_key="sma",
                comparison="above",
                threshold=50.0,
                params={"period": 20},
            ),
            IndicatorGene(
                enabled=False,
                indicator_name="rsi",
                output_key="rsi",
                comparison="below",
                threshold=70.0,
                params={"period": 14},
            ),
        ),
        combination_logic="OR",
        limit_exits=LimitExitGene(
            stop_loss=StopLossConfig(enabled=True, mode="atr", percent=3.0, atr_multiple=2.5),
            take_profit=TakeProfitConfig(enabled=True, mode="percent", percent=8.0, atr_multiple=3.0),
            trailing_stop=TrailingStopConfig(
                enabled=False,
                mode="percent",
                percent=2.0,
                atr_multiple=1.5,
                activation_percent=4.0,
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(
                enabled=False,
                mode="percent",
                trigger_percent=2.0,
                trigger_atr_multiple=1.0,
            ),
        ),
        indicator_exits=(
            IndicatorExitGene(
                enabled=True,
                indicator_name="rsi",
                output_key="rsi",
                comparison="above",
                threshold=80.0,
                params={"period": 14},
                opposite_entry=False,
            ),
        ),
        time_exits=TimeExitGene(
            max_days_enabled=True,
            max_days=20,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=True,
            stagnation_days=7,
            stagnation_threshold=0.3,
        ),
        filters=(
            FilterGene(
                enabled=True,
                filter_name="sma_regime",
                params={"period": 200},
            ),
        ),
    )


def _make_trade(net_return_pct: float, idx: int = 0, exit_type: str = "take_profit") -> TradeRecord:
    """Create a TradeRecord with the given net return."""
    return TradeRecord(
        entry_bar=idx,
        entry_date=f"2020-01-{(idx + 1):02d}",
        entry_price=100.0,
        exit_bar=idx + 5,
        exit_date=f"2020-01-{(idx + 6):02d}",
        exit_price=100.0 * (1 + net_return_pct),
        exit_type=exit_type,
        gross_return_pct=net_return_pct + 0.002,
        cost_pct=0.002,
        net_return_pct=net_return_pct,
        net_profit=net_return_pct * 10_000,
        is_win=net_return_pct > 0,
        holding_days=5,
    )


def _make_metrics() -> BacktestMetrics:
    """Create simple backtest metrics."""
    return BacktestMetrics(
        num_trades=10,
        num_wins=8,
        num_losses=2,
        win_rate=0.80,
        net_profit=500.0,
        gross_profit=700.0,
        gross_loss=-200.0,
        profit_factor=3.5,
        avg_win_pct=1.2,
        avg_loss_pct=-0.8,
        win_loss_ratio=1.5,
        max_drawdown_pct=5.0,
        max_drawdown_duration=3,
        sharpe_ratio=1.8,
        sortino_ratio=2.5,
        calmar_ratio=3.0,
        avg_holding_days=5.0,
        longest_win_streak=5,
        longest_loss_streak=2,
        expectancy_per_trade=50.0,
    )


def _make_ga_result(genome: StrategyGenome) -> GAResult:
    """Create a GAResult."""
    return GAResult(
        best_genome=genome,
        best_fitness=0.85,
        generations_run=50,
        logbook=(
            GenerationStats(
                generation=0,
                best_fitness=0.70,
                worst_fitness=0.10,
                avg_fitness=0.40,
                std_fitness=0.15,
                diversity=0.90,
            ),
            GenerationStats(
                generation=49,
                best_fitness=0.85,
                worst_fitness=0.30,
                avg_fitness=0.60,
                std_fitness=0.10,
                diversity=0.50,
            ),
        ),
    )


def _make_pipeline_result(
    genome: StrategyGenome | None = None,
    mode: str = "ga_only",
) -> PipelineResult:
    """Create a PipelineResult."""
    g = genome or _minimal_genome()
    ga_result = _make_ga_result(g) if mode in ("ga_only", "both") else None
    bay_result = None
    if mode in ("bayesian_only", "both"):
        bay_result = BayesianResult(
            best_genome=g,
            best_fitness=0.87,
            trials_run=100,
            trials_pruned=20,
            best_params={"period": 14.0, "threshold": 0.5},
        )
    return PipelineResult(
        mode=mode,
        best_genome=g,
        best_fitness=0.85 if ga_result else 0.87,
        ga_result=ga_result,
        bayesian_result=bay_result,
    )


def _make_walkforward_result(genome: StrategyGenome | None = None) -> WalkForwardResult:
    """Create a WalkForwardResult with one window."""
    g = genome or _minimal_genome()
    metrics = _make_metrics()
    window = WalkForwardWindow(
        window_index=0,
        is_start=pd.Timestamp("2018-01-01"),
        is_end=pd.Timestamp("2021-12-31"),
        oos_start=pd.Timestamp("2022-01-01"),
        oos_end=pd.Timestamp("2022-06-30"),
    )
    pipeline_result = _make_pipeline_result(g)
    window_result = WindowResult(
        window=window,
        best_genome=g,
        is_fitness=0.85,
        oos_metrics=metrics,
        pipeline_result=pipeline_result,
    )
    holdout = HoldoutResult(
        holdout_start=pd.Timestamp("2023-01-01"),
        holdout_end=pd.Timestamp("2023-12-31"),
        genome=g,
        metrics=metrics,
        is_profitable=True,
        win_rate_within_tolerance=True,
    )
    return WalkForwardResult(
        window_results=(window_result,),
        holdout_result=holdout,
        best_genome=g,
        passed=True,
        win_rate_pass=True,
        profitability_pass=True,
        holdout_pass=True,
        low_frequency=False,
        windows_with_trades=1,
        windows_passing_win_rate=1,
        windows_profitable=1,
        avg_oos_win_rate=0.80,
    )


def _make_robustness_result(skipped: bool = False) -> RobustnessResult:
    """Create a RobustnessResult."""
    if skipped:
        return RobustnessResult(
            permutation_tests=(),
            bootstrap_cis=(),
            passed=False,
            num_trades=5,
            skipped=True,
        )
    return RobustnessResult(
        permutation_tests=(
            PermutationTestResult(
                statistic_name="win_rate",
                actual_value=0.80,
                p_value=0.01,
                n_permutations=1000,
                significant=True,
            ),
        ),
        bootstrap_cis=(
            BootstrapCIResult(
                statistic_name="win_rate",
                actual_value=0.80,
                lower_bound=0.72,
                upper_bound=0.88,
                confidence_level=0.95,
                n_resamples=1000,
            ),
        ),
        passed=True,
        num_trades=50,
        skipped=False,
    )


def _make_backtest_result() -> BacktestResult:
    """Create a BacktestResult with sample trades."""
    trades = (
        _make_trade(0.05, 0, "take_profit"),
        _make_trade(-0.02, 5, "stop_loss"),
        _make_trade(0.03, 10, "take_profit"),
    )
    return BacktestResult(trades=trades, metrics=_make_metrics())


# ---------------------------------------------------------------------------
# TestDescribeIndicator
# ---------------------------------------------------------------------------


class TestDescribeIndicator:
    """Tests for _describe_indicator()."""

    def test_enabled_indicator(self) -> None:
        """Enabled indicator produces human-readable description."""
        gene = IndicatorGene(
            enabled=True,
            indicator_name="sma",
            output_key="sma",
            comparison="above",
            threshold=50.0,
            params={"period": 20},
        )
        result = _describe_indicator(gene)
        assert result == "sma(20) above 50.0"

    def test_disabled_indicator(self) -> None:
        """Disabled indicator still produces description (filtering is caller's job)."""
        gene = IndicatorGene(
            enabled=False,
            indicator_name="rsi",
            output_key="rsi",
            comparison="below",
            threshold=70.0,
            params={"period": 14},
        )
        result = _describe_indicator(gene)
        assert result == "rsi(14) below 70.0"


# ---------------------------------------------------------------------------
# TestDescribeExit
# ---------------------------------------------------------------------------


class TestDescribeExit:
    """Tests for _describe_exit()."""

    def test_mixed_exits(self) -> None:
        """Multiple enabled exit types produce descriptions."""
        genome = _complex_genome()
        exits = _describe_exit(genome)
        assert "stop_loss: atr 2.5x" in exits
        assert "take_profit: 8.0%" in exits
        assert any("rsi" in e for e in exits)
        assert any("max_days" in e for e in exits)
        assert any("stagnation" in e for e in exits)

    def test_all_disabled(self) -> None:
        """No enabled exits produces empty tuple."""
        genome = _minimal_genome()
        exits = _describe_exit(genome)
        assert exits == ()


# ---------------------------------------------------------------------------
# TestDescribeFilter
# ---------------------------------------------------------------------------


class TestDescribeFilter:
    """Tests for _describe_filter()."""

    def test_enabled_filter(self) -> None:
        """Enabled filter produces human-readable description."""
        gene = FilterGene(enabled=True, filter_name="sma_regime", params={"period": 200})
        result = _describe_filter(gene)
        assert result == "sma_regime(200)"

    def test_disabled_filter(self) -> None:
        """Disabled filter still produces description."""
        gene = FilterGene(enabled=False, filter_name="vol_filter", params={"period": 50})
        result = _describe_filter(gene)
        assert result == "vol_filter(50)"


# ---------------------------------------------------------------------------
# TestBuildStrategySummary
# ---------------------------------------------------------------------------


class TestBuildStrategySummary:
    """Tests for _build_strategy_summary()."""

    def test_minimal_genome(self) -> None:
        """Minimal genome produces summary with zero counts."""
        genome = _minimal_genome()
        summary = _build_strategy_summary(genome)
        assert summary.num_entry_indicators == 0
        assert summary.num_exit_types == 0
        assert summary.num_filters == 0
        assert summary.combination_logic == "AND"

    def test_complex_genome(self) -> None:
        """Complex genome produces populated summary."""
        genome = _complex_genome()
        summary = _build_strategy_summary(genome)
        assert summary.num_entry_indicators == 1  # only enabled
        assert summary.num_exit_types > 0
        assert summary.num_filters == 1
        assert summary.combination_logic == "OR"
        assert len(summary.entry_indicators) == 1
        assert "sma(20)" in summary.entry_indicators[0]


# ---------------------------------------------------------------------------
# TestBuildOptimizationSummary
# ---------------------------------------------------------------------------


class TestBuildOptimizationSummary:
    """Tests for _build_optimization_summary()."""

    def test_ga_only(self) -> None:
        """GA-only pipeline produces summary with GA fields."""
        result = _make_pipeline_result(mode="ga_only")
        summary = _build_optimization_summary(result)
        assert summary.mode == "ga_only"
        assert summary.ga_generations == 50
        assert summary.ga_final_diversity == 0.50
        assert summary.bayesian_trials is None
        assert summary.best_params is None

    def test_bayesian_only(self) -> None:
        """Bayesian-only pipeline produces summary with Bayesian fields."""
        result = _make_pipeline_result(mode="bayesian_only")
        summary = _build_optimization_summary(result)
        assert summary.mode == "bayesian_only"
        assert summary.ga_generations is None
        assert summary.bayesian_trials == 100
        assert summary.bayesian_pruned == 20
        assert summary.best_params is not None

    def test_both(self) -> None:
        """Both-mode pipeline produces summary with all fields."""
        result = _make_pipeline_result(mode="both")
        summary = _build_optimization_summary(result)
        assert summary.mode == "both"
        assert summary.ga_generations == 50
        assert summary.bayesian_trials == 100


# ---------------------------------------------------------------------------
# TestBuildFitnessEvolution
# ---------------------------------------------------------------------------


class TestBuildFitnessEvolution:
    """Tests for _build_fitness_evolution()."""

    def test_with_logbook(self) -> None:
        """GA result with logbook produces fitness evolution."""
        result = _make_pipeline_result(mode="ga_only")
        evo = _build_fitness_evolution(result)
        assert evo is not None
        assert len(evo.generations) == 2
        assert evo.best[0] == 0.70
        assert evo.best[1] == 0.85

    def test_empty_logbook(self) -> None:
        """Bayesian-only result returns None."""
        result = _make_pipeline_result(mode="bayesian_only")
        evo = _build_fitness_evolution(result)
        assert evo is None


# ---------------------------------------------------------------------------
# TestBuildWalkForwardSummary
# ---------------------------------------------------------------------------


class TestBuildWalkForwardSummary:
    """Tests for _build_walkforward_summary()."""

    def test_passing(self) -> None:
        """Passing WF result produces correct summary."""
        wf = _make_walkforward_result()
        summary = _build_walkforward_summary(wf)
        assert summary.passed is True
        assert summary.windows_with_trades == 1
        assert len(summary.window_summaries) == 1

    def test_with_holdout(self) -> None:
        """Summary includes holdout fields."""
        wf = _make_walkforward_result()
        summary = _build_walkforward_summary(wf)
        assert summary.holdout_win_rate == 0.80
        assert summary.holdout_net_profit == 500.0
        assert summary.holdout_profitable is True


# ---------------------------------------------------------------------------
# TestBuildRobustnessSummary
# ---------------------------------------------------------------------------


class TestBuildRobustnessSummary:
    """Tests for _build_robustness_summary()."""

    def test_normal(self) -> None:
        """Normal robustness result produces correct summary."""
        result = _make_robustness_result()
        summary = _build_robustness_summary(result)
        assert summary.passed is True
        assert summary.skipped is False
        assert summary.num_trades == 50
        assert len(summary.permutation_tests) == 1
        assert summary.permutation_tests[0]["statistic"] == "win_rate"
        assert len(summary.bootstrap_cis) == 1

    def test_skipped(self) -> None:
        """Skipped robustness result shows skipped flag."""
        result = _make_robustness_result(skipped=True)
        summary = _build_robustness_summary(result)
        assert summary.skipped is True
        assert summary.passed is False
        assert len(summary.permutation_tests) == 0
        assert len(summary.bootstrap_cis) == 0


# ---------------------------------------------------------------------------
# TestBuildTradeAnalysis
# ---------------------------------------------------------------------------


class TestBuildTradeAnalysis:
    """Tests for _build_trade_analysis()."""

    def test_normal_trades(self) -> None:
        """Trade analysis from multiple trades."""
        trades = (
            _make_trade(0.05, 0, "take_profit"),
            _make_trade(-0.02, 5, "stop_loss"),
            _make_trade(0.03, 10, "take_profit"),
        )
        analysis = _build_trade_analysis(trades)
        assert analysis.total_trades == 3
        assert len(analysis.cumulative_pnl) == 3
        assert analysis.exit_type_counts["take_profit"] == 2
        assert analysis.exit_type_counts["stop_loss"] == 1
        assert analysis.best_trade_pct == 0.05
        assert analysis.worst_trade_pct == -0.02
        assert analysis.avg_return_winners > 0
        assert analysis.avg_return_losers < 0
        assert analysis.median_holding_days == 5.0

    def test_single_trade(self) -> None:
        """Trade analysis with a single trade."""
        trades = (_make_trade(0.05, 0, "take_profit"),)
        analysis = _build_trade_analysis(trades)
        assert analysis.total_trades == 1
        assert len(analysis.cumulative_pnl) == 1
        assert analysis.best_trade_pct == 0.05
        assert analysis.worst_trade_pct == 0.05
        assert analysis.avg_return_losers == 0.0

    def test_empty(self) -> None:
        """Empty trades produce zero analysis."""
        analysis = _build_trade_analysis(())
        assert analysis.total_trades == 0
        assert analysis.cumulative_pnl == ()
        assert analysis.exit_type_counts == {}
        assert analysis.best_trade_pct == 0.0


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_full_report(self) -> None:
        """Full report with all components."""
        genome = _complex_genome()
        report = generate_report(
            run_id="test_run",
            genome=genome,
            backtest_result=_make_backtest_result(),
            pipeline_result=_make_pipeline_result(genome, mode="both"),
            walkforward_result=_make_walkforward_result(genome),
            robustness_result=_make_robustness_result(),
        )
        assert report.run_id == "test_run"
        assert report.strategy.num_entry_indicators == 1
        assert report.metrics["win_rate"] == 0.80
        assert report.optimization is not None
        assert report.optimization.mode == "both"
        assert report.fitness_evolution is not None
        assert report.walkforward is not None
        assert report.walkforward.passed is True
        assert report.robustness is not None
        assert report.robustness.passed is True
        assert report.trade_analysis is not None
        assert report.trade_analysis.total_trades == 3

    def test_minimal_report(self) -> None:
        """Report with only genome (no optional components)."""
        genome = _minimal_genome()
        report = generate_report(
            run_id="minimal_run",
            genome=genome,
        )
        assert report.run_id == "minimal_run"
        assert report.strategy.num_entry_indicators == 0
        assert report.metrics == {}
        assert report.optimization is None
        assert report.fitness_evolution is None
        assert report.walkforward is None
        assert report.robustness is None
        assert report.trade_analysis is None


# ---------------------------------------------------------------------------
# TestSaveReport
# ---------------------------------------------------------------------------


class TestSaveReport:
    """Tests for save_report()."""

    def test_saves_valid_json(self, tmp_path: Path) -> None:
        """Saved report is valid JSON."""
        genome = _complex_genome()
        report = generate_report(
            run_id="save_test",
            genome=genome,
            backtest_result=_make_backtest_result(),
        )
        path = save_report(report, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "save_test"
        assert "strategy" in data
        assert "metrics" in data

    def test_round_trip_fields(self, tmp_path: Path) -> None:
        """Key fields survive JSON round-trip."""
        genome = _complex_genome()
        report = generate_report(
            run_id="roundtrip_test",
            genome=genome,
            backtest_result=_make_backtest_result(),
            pipeline_result=_make_pipeline_result(genome, mode="ga_only"),
        )
        path = save_report(report, tmp_path)
        data = json.loads(path.read_text())
        assert data["optimization"]["mode"] == "ga_only"
        assert data["optimization"]["ga_generations"] == 50
        assert data["strategy"]["combination_logic"] == "OR"
        assert data["trade_analysis"]["total_trades"] == 3
