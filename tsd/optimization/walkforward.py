"""Anchored walk-forward validation engine.

Tests strategy robustness by re-optimizing on growing in-sample windows
and evaluating on unseen out-of-sample data. Supports configurable OOS
length, holdout period, and passing criteria on win rate and profitability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from tsd.config import env_float, env_int
from tsd.optimization.fitness import FitnessConfig
from tsd.optimization.metrics import aggregate_metrics
from tsd.optimization.pipeline import PipelineConfig, PipelineResult, run_pipeline
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, EvaluatorConfig, run_backtest
from tsd.strategy.genome import OutputMeta, StrategyGenome, StrategyMeta

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        oos_length_months: Length of each OOS window in months.
        final_holdout_months: Length of the final holdout period in months.
        slide_step_months: Step size for sliding the OOS window.
        min_is_months: Minimum in-sample period in months.
        min_oos_windows_win_rate: Minimum OOS windows passing win rate threshold.
        min_oos_windows_profitable: Minimum OOS windows that must be profitable.
        min_win_rate_threshold: Minimum win rate threshold for OOS windows.
        holdout_win_rate_tolerance: Allowed drop in win rate for holdout vs avg OOS.
        low_frequency_threshold: Minimum trades per window to not flag low frequency.
    """

    oos_length_months: int = 6
    final_holdout_months: int = 12
    slide_step_months: int = 6
    min_is_months: int = 36
    min_oos_windows_win_rate: int = 8
    min_oos_windows_profitable: int = 7
    min_win_rate_threshold: float = 0.80
    holdout_win_rate_tolerance: float = 0.10
    low_frequency_threshold: int = 5


@dataclass(frozen=True)
class WalkForwardWindow:
    """Boundaries for a single walk-forward window.

    Attributes:
        window_index: Zero-based index of this window.
        is_start: Start of in-sample period (inclusive).
        is_end: End of in-sample period (exclusive, equals oos_start).
        oos_start: Start of out-of-sample period (inclusive).
        oos_end: End of out-of-sample period (exclusive).
    """

    window_index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


@dataclass(frozen=True)
class WindowResult:
    """Result from a single walk-forward window.

    Attributes:
        window: Window boundaries.
        best_genome: Best genome from IS optimization.
        is_fitness: Best fitness on in-sample data.
        oos_metrics: Aggregated OOS backtest metrics.
        pipeline_result: Full pipeline result from IS optimization.
    """

    window: WalkForwardWindow
    best_genome: StrategyGenome
    is_fitness: float
    oos_metrics: BacktestMetrics
    pipeline_result: PipelineResult


@dataclass(frozen=True)
class HoldoutResult:
    """Result from holdout evaluation.

    Attributes:
        holdout_start: Start of holdout period.
        holdout_end: End of holdout period.
        genome: Genome evaluated on holdout.
        metrics: Aggregated holdout metrics.
        is_profitable: Whether holdout net profit is positive.
        win_rate_within_tolerance: Whether holdout win rate is within tolerance.
    """

    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    genome: StrategyGenome
    metrics: BacktestMetrics
    is_profitable: bool
    win_rate_within_tolerance: bool


@dataclass(frozen=True)
class WalkForwardResult:
    """Complete walk-forward validation result.

    Attributes:
        window_results: Results from each walk-forward window.
        holdout_result: Holdout evaluation result (None if not enough data).
        best_genome: Best genome selected from OOS performance.
        passed: Whether all passing criteria are met.
        win_rate_pass: Whether enough windows pass win rate threshold.
        profitability_pass: Whether enough windows are profitable.
        holdout_pass: Whether holdout evaluation passed.
        low_frequency: Whether any window had fewer trades than threshold.
        windows_with_trades: Number of windows with at least one trade.
        windows_passing_win_rate: Number of windows passing win rate threshold.
        windows_profitable: Number of windows with positive net profit.
        avg_oos_win_rate: Average OOS win rate across traded windows.
    """

    window_results: tuple[WindowResult, ...]
    holdout_result: HoldoutResult | None
    best_genome: StrategyGenome
    passed: bool
    win_rate_pass: bool
    profitability_pass: bool
    holdout_pass: bool
    low_frequency: bool
    windows_with_trades: int
    windows_passing_win_rate: int
    windows_profitable: int
    avg_oos_win_rate: float


def load_walkforward_config() -> WalkForwardConfig:
    """Load walk-forward configuration from environment variables.

    Returns:
        WalkForwardConfig with values from TSD_WF_* environment variables.
    """
    return WalkForwardConfig(
        oos_length_months=env_int("TSD_WF_OOS_LENGTH_MONTHS", 6),
        final_holdout_months=env_int("TSD_WF_FINAL_HOLDOUT_MONTHS", 12),
        slide_step_months=env_int("TSD_WF_SLIDE_STEP_MONTHS", 6),
        min_is_months=env_int("TSD_WF_MIN_IS_MONTHS", 36),
        min_oos_windows_win_rate=env_int("TSD_WF_MIN_OOS_WINDOWS_WIN_RATE", 8),
        min_oos_windows_profitable=env_int("TSD_WF_MIN_OOS_WINDOWS_PROFITABLE", 7),
        min_win_rate_threshold=env_float("TSD_WF_MIN_WIN_RATE_THRESHOLD", 0.80),
        holdout_win_rate_tolerance=env_float("TSD_WF_HOLDOUT_WIN_RATE_TOLERANCE", 0.10),
        low_frequency_threshold=env_int("TSD_WF_LOW_FREQUENCY_THRESHOLD", 5),
    )


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


def generate_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    wf_config: WalkForwardConfig,
) -> tuple[tuple[WalkForwardWindow, ...], pd.Timestamp, pd.Timestamp]:
    """Compute anchored walk-forward window boundaries and holdout period.

    Windows use anchored IS (always starting from data_start) with
    growing IS length. OOS windows slide by slide_step_months. The final
    holdout period is reserved at the end of the data.

    Args:
        data_start: Earliest date in the dataset.
        data_end: Latest date in the dataset.
        wf_config: Walk-forward configuration.

    Returns:
        Tuple of (windows, holdout_start, holdout_end).

    Raises:
        ValueError: If data is too short for minimum IS + one OOS + holdout.
    """
    holdout_end = data_end
    holdout_start = data_end - pd.DateOffset(months=wf_config.final_holdout_months)

    min_data_end = data_start + pd.DateOffset(
        months=wf_config.min_is_months + wf_config.oos_length_months + wf_config.final_holdout_months
    )
    if data_end < min_data_end:
        msg = (
            f"Data range ({data_start.date()} to {data_end.date()}) is too short. "
            f"Need at least {wf_config.min_is_months} IS + {wf_config.oos_length_months} OOS "
            f"+ {wf_config.final_holdout_months} holdout months."
        )
        raise ValueError(msg)

    windows: list[WalkForwardWindow] = []
    is_end = data_start + pd.DateOffset(months=wf_config.min_is_months)
    window_idx = 0

    while True:
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(months=wf_config.oos_length_months)

        if oos_end > holdout_start:
            break

        windows.append(
            WalkForwardWindow(
                window_index=window_idx,
                is_start=data_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
        window_idx += 1
        is_end = is_end + pd.DateOffset(months=wf_config.slide_step_months)

    return tuple(windows), holdout_start, holdout_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_walkforward(  # noqa: PLR0913
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    wf_config: WalkForwardConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    eval_config: EvaluatorConfig | None = None,
    fitness_config: FitnessConfig | None = None,
) -> WalkForwardResult:
    """Run anchored walk-forward validation.

    For each window: optimize on IS data, evaluate on OOS data.
    Select best genome across windows, evaluate on holdout,
    and check passing criteria.

    Args:
        meta: Strategy metadata describing the parameter space.
        stocks_data: Mapping of stock ticker to OHLCV DataFrame.
        indicator_outputs: Mapping of indicator name to output metadata.
        wf_config: Walk-forward configuration. Uses defaults if None.
        pipeline_config: Pipeline configuration for IS optimization.
        eval_config: Evaluator configuration. Uses defaults if None.
        fitness_config: Fitness configuration. Uses defaults if None.

    Returns:
        WalkForwardResult with window results, holdout, and pass/fail.
    """
    wf_config = wf_config or WalkForwardConfig()
    eval_config = eval_config or EvaluatorConfig()
    fitness_config = fitness_config or FitnessConfig()

    data_start, data_end = _find_data_range(stocks_data)
    LOGGER.info("Walk-forward data range: %s to %s", data_start.date(), data_end.date())

    windows, holdout_start, holdout_end = generate_windows(data_start, data_end, wf_config)
    LOGGER.info("Generated %d walk-forward windows", len(windows))

    window_results: list[WindowResult] = []
    for window in windows:
        LOGGER.info(
            "Running window %d: IS [%s, %s) OOS [%s, %s)",
            window.window_index, window.is_start.date(), window.is_end.date(),
            window.oos_start.date(), window.oos_end.date(),
        )
        result = _run_window(
            window, meta, stocks_data, indicator_outputs,
            pipeline_config, eval_config, fitness_config,
        )
        window_results.append(result)
        LOGGER.info(
            "Window %d: IS fitness=%.4f, OOS trades=%d, OOS win_rate=%.4f",
            window.window_index, result.is_fitness,
            result.oos_metrics.num_trades, result.oos_metrics.win_rate,
        )

    best_genome = _select_best_genome(window_results)

    traded_windows = [wr for wr in window_results if wr.oos_metrics.num_trades > 0]
    avg_oos_win_rate = (
        sum(wr.oos_metrics.win_rate for wr in traded_windows) / len(traded_windows)
        if traded_windows
        else 0.0
    )

    holdout_result = _evaluate_holdout(
        best_genome, stocks_data, indicator_outputs, eval_config,
        holdout_start, holdout_end, avg_oos_win_rate, wf_config,
    )

    return _evaluate_passing_criteria(
        tuple(window_results), holdout_result, best_genome,
        avg_oos_win_rate, wf_config,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_data_range(stocks_data: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the overall date range across all stock DataFrames.

    Args:
        stocks_data: Mapping of ticker to OHLCV DataFrame.

    Returns:
        Tuple of (earliest_date, latest_date).

    Raises:
        ValueError: If stocks_data is empty.
    """
    if not stocks_data:
        msg = "stocks_data is empty"
        raise ValueError(msg)

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for df in stocks_data.values():
        if len(df) > 0:
            starts.append(pd.Timestamp(df.index.min()))
            ends.append(pd.Timestamp(df.index.max()))

    if not starts:
        msg = "All DataFrames in stocks_data are empty"
        raise ValueError(msg)

    return min(starts), max(ends)


def _slice_stocks_data(
    stocks_data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Slice all stock DataFrames to a date range.

    Uses pandas loc with inclusive boundaries. Excludes stocks that
    have no data in the range.

    Args:
        stocks_data: Mapping of ticker to OHLCV DataFrame.
        start: Start date (inclusive).
        end: End date (exclusive).

    Returns:
        Dict of ticker to sliced DataFrame, excluding empty ones.
    """
    end_inclusive = end - pd.Timedelta(days=1)
    sliced: dict[str, pd.DataFrame] = {}
    for ticker, df in stocks_data.items():
        sub = df.loc[start:end_inclusive]
        if len(sub) > 0:
            sliced[ticker] = sub
    return sliced


def _run_window(  # noqa: PLR0913
    window: WalkForwardWindow,
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    pipeline_config: PipelineConfig | None,
    eval_config: EvaluatorConfig | None,
    fitness_config: FitnessConfig | None,
) -> WindowResult:
    """Run optimization on IS data and evaluate on OOS data for one window.

    Args:
        window: Window boundaries.
        meta: Strategy metadata.
        stocks_data: Full stock data (will be sliced).
        indicator_outputs: Indicator output metadata.
        pipeline_config: Pipeline configuration.
        eval_config: Evaluator configuration.
        fitness_config: Fitness configuration.

    Returns:
        WindowResult with IS fitness and OOS metrics.
    """
    is_data = _slice_stocks_data(stocks_data, window.is_start, window.is_end)
    oos_data = _slice_stocks_data(stocks_data, window.oos_start, window.oos_end)

    pipeline_result = run_pipeline(
        meta=meta,
        stocks_data=is_data,
        indicator_outputs=indicator_outputs,
        pipeline_config=pipeline_config,
        eval_config=eval_config,
        fitness_config=fitness_config,
    )

    oos_metrics = _evaluate_oos(
        pipeline_result.best_genome, oos_data, indicator_outputs, eval_config,
    )

    return WindowResult(
        window=window,
        best_genome=pipeline_result.best_genome,
        is_fitness=pipeline_result.best_fitness,
        oos_metrics=oos_metrics,
        pipeline_result=pipeline_result,
    )


def _evaluate_oos(
    genome: StrategyGenome,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    eval_config: EvaluatorConfig | None,
) -> BacktestMetrics:
    """Backtest a genome on OOS data and aggregate metrics.

    Args:
        genome: Strategy genome to evaluate.
        stocks_data: OOS stock data.
        indicator_outputs: Indicator output metadata.
        eval_config: Evaluator configuration.

    Returns:
        Aggregated BacktestMetrics across all stocks.
    """
    eval_config = eval_config or EvaluatorConfig()
    results: list[BacktestResult] = []
    for _ticker, df in stocks_data.items():
        try:
            result = run_backtest(genome, df, indicator_outputs, eval_config)
            results.append(result)
        except Exception:  # noqa: BLE001
            LOGGER.warning("OOS backtest failed for a stock, skipping")

    return aggregate_metrics(results)


def _select_best_genome(window_results: list[WindowResult]) -> StrategyGenome:
    """Select the genome with the highest OOS win rate.

    Only considers windows that had trades. Falls back to the first
    window's genome if no windows had trades.

    Args:
        window_results: Results from all walk-forward windows.

    Returns:
        Best genome based on OOS win rate.
    """
    traded = [wr for wr in window_results if wr.oos_metrics.num_trades > 0]
    if not traded:
        return window_results[0].best_genome

    return max(traded, key=lambda wr: wr.oos_metrics.win_rate).best_genome


def _evaluate_holdout(  # noqa: PLR0913
    genome: StrategyGenome,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    eval_config: EvaluatorConfig | None,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    avg_oos_win_rate: float,
    wf_config: WalkForwardConfig,
) -> HoldoutResult:
    """Evaluate a genome on the holdout period.

    Args:
        genome: Best genome to evaluate.
        stocks_data: Full stock data (will be sliced to holdout).
        indicator_outputs: Indicator output metadata.
        eval_config: Evaluator configuration.
        holdout_start: Start of holdout period.
        holdout_end: End of holdout period.
        avg_oos_win_rate: Average OOS win rate for tolerance check.
        wf_config: Walk-forward configuration.

    Returns:
        HoldoutResult with metrics and pass/fail flags.
    """
    holdout_data = _slice_stocks_data(stocks_data, holdout_start, holdout_end)
    metrics = _evaluate_oos(genome, holdout_data, indicator_outputs, eval_config)

    is_profitable = metrics.net_profit > 0
    win_rate_within_tolerance = (
        metrics.win_rate >= avg_oos_win_rate - wf_config.holdout_win_rate_tolerance
    )

    LOGGER.info(
        "Holdout: trades=%d, win_rate=%.4f, net_profit=%.2f, profitable=%s, tolerance=%s",
        metrics.num_trades, metrics.win_rate, metrics.net_profit,
        is_profitable, win_rate_within_tolerance,
    )

    return HoldoutResult(
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        genome=genome,
        metrics=metrics,
        is_profitable=is_profitable,
        win_rate_within_tolerance=win_rate_within_tolerance,
    )


def _evaluate_passing_criteria(
    window_results: tuple[WindowResult, ...],
    holdout_result: HoldoutResult,
    best_genome: StrategyGenome,
    avg_oos_win_rate: float,
    wf_config: WalkForwardConfig,
) -> WalkForwardResult:
    """Evaluate all walk-forward passing criteria.

    Criteria:
    1. Win rate: >= min_oos_windows_win_rate windows pass threshold.
    2. Profitability: >= min_oos_windows_profitable windows are profitable.
    3. Holdout: profitable and win rate within tolerance.

    Args:
        window_results: All window results.
        holdout_result: Holdout evaluation result.
        best_genome: Best genome across windows.
        avg_oos_win_rate: Average OOS win rate.
        wf_config: Walk-forward configuration.

    Returns:
        WalkForwardResult with all criteria evaluated.
    """
    traded = [wr for wr in window_results if wr.oos_metrics.num_trades > 0]
    windows_with_trades = len(traded)

    windows_passing_win_rate = sum(
        1 for wr in traded
        if wr.oos_metrics.win_rate >= wf_config.min_win_rate_threshold
    )
    windows_profitable = sum(
        1 for wr in traded
        if wr.oos_metrics.net_profit > 0
    )

    low_frequency = any(
        0 < wr.oos_metrics.num_trades < wf_config.low_frequency_threshold
        for wr in window_results
    )

    win_rate_pass = windows_passing_win_rate >= wf_config.min_oos_windows_win_rate
    profitability_pass = windows_profitable >= wf_config.min_oos_windows_profitable
    holdout_pass = holdout_result.is_profitable and holdout_result.win_rate_within_tolerance
    passed = win_rate_pass and profitability_pass and holdout_pass

    LOGGER.info(
        "Walk-forward result: passed=%s (win_rate=%s, profit=%s, holdout=%s, low_freq=%s)",
        passed, win_rate_pass, profitability_pass, holdout_pass, low_frequency,
    )

    return WalkForwardResult(
        window_results=window_results,
        holdout_result=holdout_result,
        best_genome=best_genome,
        passed=passed,
        win_rate_pass=win_rate_pass,
        profitability_pass=profitability_pass,
        holdout_pass=holdout_pass,
        low_frequency=low_frequency,
        windows_with_trades=windows_with_trades,
        windows_passing_win_rate=windows_passing_win_rate,
        windows_profitable=windows_profitable,
        avg_oos_win_rate=avg_oos_win_rate,
    )
