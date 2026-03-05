"""Performance reporting for strategy optimization runs.

Generates comprehensive JSON reports that aggregate strategy descriptions,
optimization summaries, walk-forward results, robustness checks, and
trade analysis into a single human-readable document.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tsd.analysis.robustness import RobustnessResult
from tsd.optimization.pipeline import PipelineResult
from tsd.optimization.walkforward import WalkForwardResult, WindowResult
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, TradeRecord
from tsd.strategy.genome import (
    FilterGene,
    IndicatorGene,
    StrategyGenome,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySummary:
    """Human-readable description of the strategy."""

    entry_indicators: tuple[str, ...]
    combination_logic: str
    active_exits: tuple[str, ...]
    active_filters: tuple[str, ...]
    num_entry_indicators: int
    num_exit_types: int
    num_filters: int


@dataclass(frozen=True)
class OptimizationSummary:
    """Summary of the optimization process."""

    mode: str
    best_fitness: float
    ga_generations: int | None
    ga_final_diversity: float | None
    bayesian_trials: int | None
    bayesian_pruned: int | None
    best_params: dict[str, float] | None


@dataclass(frozen=True)
class FitnessEvolution:
    """Per-generation fitness stats for plotting."""

    generations: tuple[int, ...]
    best: tuple[float, ...]
    avg: tuple[float, ...]
    worst: tuple[float, ...]
    diversity: tuple[float, ...]


@dataclass(frozen=True)
class WindowSummary:
    """Summary of a single walk-forward window."""

    window_index: int
    is_period: str
    oos_period: str
    is_fitness: float
    oos_win_rate: float
    oos_net_profit: float
    oos_num_trades: int
    oos_profit_factor: float


@dataclass(frozen=True)
class WalkForwardSummary:
    """Summary of walk-forward validation."""

    passed: bool
    win_rate_pass: bool
    profitability_pass: bool
    holdout_pass: bool
    low_frequency: bool
    windows_with_trades: int
    windows_passing_win_rate: int
    windows_profitable: int
    avg_oos_win_rate: float
    window_summaries: tuple[WindowSummary, ...]
    holdout_win_rate: float | None
    holdout_net_profit: float | None
    holdout_profitable: bool | None


@dataclass(frozen=True)
class RobustnessSummary:
    """Summary of statistical robustness checks."""

    passed: bool
    skipped: bool
    num_trades: int
    permutation_tests: tuple[dict[str, str | float | bool], ...]
    bootstrap_cis: tuple[dict[str, str | float], ...]


@dataclass(frozen=True)
class TradeAnalysis:
    """Derived trade statistics for reporting."""

    total_trades: int
    cumulative_pnl: tuple[float, ...]
    exit_type_counts: dict[str, int]
    monthly_returns: dict[str, float]
    avg_return_winners: float
    avg_return_losers: float
    best_trade_pct: float
    worst_trade_pct: float
    median_holding_days: float


@dataclass(frozen=True)
class PerformanceReport:
    """Comprehensive performance report for a single run."""

    run_id: str
    timestamp: str
    strategy: StrategySummary
    metrics: dict[str, float | int]
    optimization: OptimizationSummary | None
    fitness_evolution: FitnessEvolution | None
    walkforward: WalkForwardSummary | None
    robustness: RobustnessSummary | None
    trade_analysis: TradeAnalysis | None


# ---------------------------------------------------------------------------
# Strategy description helpers
# ---------------------------------------------------------------------------


def _describe_indicator(gene: IndicatorGene) -> str:
    """Describe an indicator gene in human-readable form.

    Args:
        gene: Indicator gene to describe.

    Returns:
        String like "sma(20) above 50.0".
    """
    param_str = ", ".join(f"{v}" for v in gene.params.values())
    return f"{gene.indicator_name}({param_str}) {gene.comparison} {gene.threshold}"


def _describe_limit_exits(genome: StrategyGenome) -> list[str]:
    """Describe active limit-based exit types.

    Args:
        genome: Strategy genome.

    Returns:
        List of exit description strings.
    """
    exits: list[str] = []
    limits = genome.limit_exits

    if limits.stop_loss.enabled:
        sl = limits.stop_loss
        exits.append(f"stop_loss: atr {sl.atr_multiple}x" if sl.mode == "atr" else f"stop_loss: {sl.percent}%")

    if limits.take_profit.enabled:
        tp = limits.take_profit
        exits.append(f"take_profit: atr {tp.atr_multiple}x" if tp.mode == "atr" else f"take_profit: {tp.percent}%")

    if limits.trailing_stop.enabled:
        ts = limits.trailing_stop
        exits.append(f"trailing_stop: atr {ts.atr_multiple}x" if ts.mode == "atr" else f"trailing_stop: {ts.percent}%")

    if limits.chandelier.enabled:
        exits.append(f"chandelier: atr {limits.chandelier.atr_multiple}x")

    if limits.breakeven.enabled:
        be = limits.breakeven
        exits.append(
            f"breakeven: atr {be.trigger_atr_multiple}x" if be.mode == "atr" else f"breakeven: {be.trigger_percent}%"
        )

    return exits


def _describe_time_exits(genome: StrategyGenome) -> list[str]:
    """Describe active time-based exit types.

    Args:
        genome: Strategy genome.

    Returns:
        List of exit description strings.
    """
    exits: list[str] = []
    te = genome.time_exits
    if te.max_days_enabled:
        exits.append(f"max_days: {te.max_days}")
    if te.weekday_exit_enabled:
        exits.append(f"weekday: {te.weekday}")
    if te.eow_enabled:
        exits.append("end_of_week")
    if te.eom_enabled:
        exits.append("end_of_month")
    if te.stagnation_enabled:
        exits.append(f"stagnation: {te.stagnation_days}d/{te.stagnation_threshold}%")
    return exits


def _describe_exit(genome: StrategyGenome) -> tuple[str, ...]:
    """Describe all active exit types in human-readable form.

    Args:
        genome: Strategy genome.

    Returns:
        Tuple of exit description strings.
    """
    exits = _describe_limit_exits(genome)

    for ie in genome.indicator_exits:
        if ie.enabled:
            param_str = ", ".join(f"{v}" for v in ie.params.values())
            exits.append(f"indicator: {ie.indicator_name}({param_str}) {ie.comparison} {ie.threshold}")

    exits.extend(_describe_time_exits(genome))
    return tuple(exits)


def _describe_filter(gene: FilterGene) -> str:
    """Describe a filter gene in human-readable form.

    Args:
        gene: Filter gene to describe.

    Returns:
        String like "sma_regime(200)".
    """
    param_str = ", ".join(f"{v}" for v in gene.params.values())
    return f"{gene.filter_name}({param_str})"


def _build_strategy_summary(genome: StrategyGenome) -> StrategySummary:
    """Build a human-readable strategy summary from a genome.

    Args:
        genome: Strategy genome.

    Returns:
        StrategySummary dataclass.
    """
    enabled_indicators = [g for g in genome.entry_indicators if g.enabled]
    enabled_filters = [f for f in genome.filters if f.enabled]
    active_exits = _describe_exit(genome)

    return StrategySummary(
        entry_indicators=tuple(_describe_indicator(g) for g in enabled_indicators),
        combination_logic=genome.combination_logic,
        active_exits=active_exits,
        active_filters=tuple(_describe_filter(f) for f in enabled_filters),
        num_entry_indicators=len(enabled_indicators),
        num_exit_types=len(active_exits),
        num_filters=len(enabled_filters),
    )


# ---------------------------------------------------------------------------
# Optimization summary helpers
# ---------------------------------------------------------------------------


def _build_optimization_summary(result: PipelineResult) -> OptimizationSummary:
    """Build optimization summary from a pipeline result.

    Args:
        result: Pipeline result.

    Returns:
        OptimizationSummary dataclass.
    """
    ga_generations: int | None = None
    ga_final_diversity: float | None = None
    bayesian_trials: int | None = None
    bayesian_pruned: int | None = None
    best_params: dict[str, float] | None = None

    if result.ga_result:
        ga_generations = result.ga_result.generations_run
        if result.ga_result.logbook:
            ga_final_diversity = result.ga_result.logbook[-1].diversity

    if result.bayesian_result:
        bayesian_trials = result.bayesian_result.trials_run
        bayesian_pruned = result.bayesian_result.trials_pruned
        best_params = result.bayesian_result.best_params

    return OptimizationSummary(
        mode=result.mode,
        best_fitness=result.best_fitness,
        ga_generations=ga_generations,
        ga_final_diversity=ga_final_diversity,
        bayesian_trials=bayesian_trials,
        bayesian_pruned=bayesian_pruned,
        best_params=best_params,
    )


def _build_fitness_evolution(result: PipelineResult) -> FitnessEvolution | None:
    """Build fitness evolution from GA logbook.

    Args:
        result: Pipeline result.

    Returns:
        FitnessEvolution or None if no GA result or empty logbook.
    """
    if not result.ga_result or not result.ga_result.logbook:
        return None

    logbook = result.ga_result.logbook
    return FitnessEvolution(
        generations=tuple(s.generation for s in logbook),
        best=tuple(s.best_fitness for s in logbook),
        avg=tuple(s.avg_fitness for s in logbook),
        worst=tuple(s.worst_fitness for s in logbook),
        diversity=tuple(s.diversity for s in logbook),
    )


# ---------------------------------------------------------------------------
# Walk-forward summary helpers
# ---------------------------------------------------------------------------


def _build_window_summary(wr: WindowResult) -> WindowSummary:
    """Build a window summary from a WindowResult.

    Args:
        wr: Window result.

    Returns:
        WindowSummary dataclass.
    """
    w = wr.window
    return WindowSummary(
        window_index=w.window_index,
        is_period=f"{w.is_start.date()} to {w.is_end.date()}",
        oos_period=f"{w.oos_start.date()} to {w.oos_end.date()}",
        is_fitness=wr.is_fitness,
        oos_win_rate=wr.oos_metrics.win_rate,
        oos_net_profit=wr.oos_metrics.net_profit,
        oos_num_trades=wr.oos_metrics.num_trades,
        oos_profit_factor=wr.oos_metrics.profit_factor,
    )


def _build_walkforward_summary(result: WalkForwardResult) -> WalkForwardSummary:
    """Build walk-forward summary from a WalkForwardResult.

    Args:
        result: Walk-forward validation result.

    Returns:
        WalkForwardSummary dataclass.
    """
    holdout_win_rate: float | None = None
    holdout_net_profit: float | None = None
    holdout_profitable: bool | None = None

    if result.holdout_result:
        holdout_win_rate = result.holdout_result.metrics.win_rate
        holdout_net_profit = result.holdout_result.metrics.net_profit
        holdout_profitable = result.holdout_result.is_profitable

    return WalkForwardSummary(
        passed=result.passed,
        win_rate_pass=result.win_rate_pass,
        profitability_pass=result.profitability_pass,
        holdout_pass=result.holdout_pass,
        low_frequency=result.low_frequency,
        windows_with_trades=result.windows_with_trades,
        windows_passing_win_rate=result.windows_passing_win_rate,
        windows_profitable=result.windows_profitable,
        avg_oos_win_rate=result.avg_oos_win_rate,
        window_summaries=tuple(_build_window_summary(wr) for wr in result.window_results),
        holdout_win_rate=holdout_win_rate,
        holdout_net_profit=holdout_net_profit,
        holdout_profitable=holdout_profitable,
    )


# ---------------------------------------------------------------------------
# Robustness summary helpers
# ---------------------------------------------------------------------------


def _build_robustness_summary(result: RobustnessResult) -> RobustnessSummary:
    """Build robustness summary from a RobustnessResult.

    Args:
        result: Robustness test result.

    Returns:
        RobustnessSummary dataclass.
    """
    perm_dicts: list[dict[str, str | float | bool]] = []
    for pt in result.permutation_tests:
        perm_dicts.append(
            {
                "statistic": pt.statistic_name,
                "actual_value": pt.actual_value,
                "p_value": pt.p_value,
                "significant": pt.significant,
            }
        )

    bs_dicts: list[dict[str, str | float]] = []
    for ci in result.bootstrap_cis:
        bs_dicts.append(
            {
                "statistic": ci.statistic_name,
                "actual_value": ci.actual_value,
                "lower_bound": ci.lower_bound,
                "upper_bound": ci.upper_bound,
            }
        )

    return RobustnessSummary(
        passed=result.passed,
        skipped=result.skipped,
        num_trades=result.num_trades,
        permutation_tests=tuple(perm_dicts),
        bootstrap_cis=tuple(bs_dicts),
    )


# ---------------------------------------------------------------------------
# Trade analysis helpers
# ---------------------------------------------------------------------------


def _build_trade_analysis(trades: tuple[TradeRecord, ...]) -> TradeAnalysis:
    """Build derived trade statistics from trade records.

    Args:
        trades: Tuple of trade records.

    Returns:
        TradeAnalysis dataclass.
    """
    if not trades:
        return TradeAnalysis(
            total_trades=0,
            cumulative_pnl=(),
            exit_type_counts={},
            monthly_returns={},
            avg_return_winners=0.0,
            avg_return_losers=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            median_holding_days=0.0,
        )

    # Cumulative P&L
    running = 0.0
    cumulative: list[float] = []
    for t in trades:
        running += t.net_profit
        cumulative.append(round(running, 4))

    # Exit type counts
    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_type] = exit_counts.get(t.exit_type, 0) + 1

    # Monthly returns
    monthly: dict[str, float] = {}
    for t in trades:
        month_key = t.exit_date[:7]  # "YYYY-MM"
        monthly[month_key] = monthly.get(month_key, 0.0) + t.net_return_pct

    # Winner/loser stats
    winners = [t.net_return_pct for t in trades if t.is_win]
    losers = [t.net_return_pct for t in trades if not t.is_win]
    avg_winners = sum(winners) / len(winners) if winners else 0.0
    avg_losers = sum(losers) / len(losers) if losers else 0.0

    # Best/worst
    returns = [t.net_return_pct for t in trades]
    best = max(returns)
    worst = min(returns)

    # Median holding days
    holding_days = sorted(t.holding_days for t in trades)
    n = len(holding_days)
    if n % 2 == 1:
        median_hd = float(holding_days[n // 2])
    else:
        median_hd = (holding_days[n // 2 - 1] + holding_days[n // 2]) / 2.0

    return TradeAnalysis(
        total_trades=len(trades),
        cumulative_pnl=tuple(cumulative),
        exit_type_counts=exit_counts,
        monthly_returns={k: round(v, 6) for k, v in monthly.items()},
        avg_return_winners=avg_winners,
        avg_return_losers=avg_losers,
        best_trade_pct=best,
        worst_trade_pct=worst,
        median_holding_days=median_hd,
    )


# ---------------------------------------------------------------------------
# Metrics flattening
# ---------------------------------------------------------------------------


def _flatten_metrics(metrics: BacktestMetrics) -> dict[str, float | int]:
    """Convert BacktestMetrics to a flat dict.

    Args:
        metrics: Backtest metrics dataclass.

    Returns:
        Dict with all metrics as top-level keys.
    """
    result: dict[str, float | int] = {}
    for f in dataclasses.fields(metrics):
        result[f.name] = getattr(metrics, f.name)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    run_id: str,
    genome: StrategyGenome,
    backtest_result: BacktestResult | None = None,
    pipeline_result: PipelineResult | None = None,
    walkforward_result: WalkForwardResult | None = None,
    robustness_result: RobustnessResult | None = None,
) -> PerformanceReport:
    """Generate a comprehensive performance report.

    Args:
        run_id: Unique run identifier.
        genome: Strategy genome to describe.
        backtest_result: Backtest result with trades and metrics.
        pipeline_result: Optimization pipeline result.
        walkforward_result: Walk-forward validation result.
        robustness_result: Statistical robustness result.

    Returns:
        PerformanceReport dataclass.
    """
    strategy = _build_strategy_summary(genome)

    metrics: dict[str, float | int] = {}
    if backtest_result:
        metrics = _flatten_metrics(backtest_result.metrics)

    optimization: OptimizationSummary | None = None
    fitness_evolution: FitnessEvolution | None = None
    if pipeline_result:
        optimization = _build_optimization_summary(pipeline_result)
        fitness_evolution = _build_fitness_evolution(pipeline_result)

    walkforward: WalkForwardSummary | None = None
    if walkforward_result:
        walkforward = _build_walkforward_summary(walkforward_result)

    robustness: RobustnessSummary | None = None
    if robustness_result:
        robustness = _build_robustness_summary(robustness_result)

    trade_analysis: TradeAnalysis | None = None
    if backtest_result and backtest_result.trades:
        trade_analysis = _build_trade_analysis(backtest_result.trades)

    return PerformanceReport(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        strategy=strategy,
        metrics=metrics,
        optimization=optimization,
        fitness_evolution=fitness_evolution,
        walkforward=walkforward,
        robustness=robustness,
        trade_analysis=trade_analysis,
    )


def save_report(report: PerformanceReport, results_dir: Path) -> Path:
    """Save a performance report as JSON.

    Args:
        report: Performance report to save.
        results_dir: Root results directory.

    Returns:
        Path to the saved report file.
    """
    report_path = results_dir / "performance" / f"{report.run_id}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = dataclasses.asdict(report)
    report_path.write_text(json.dumps(data, indent=2))
    LOGGER.info("Saved performance report to %s", report_path)
    return report_path
