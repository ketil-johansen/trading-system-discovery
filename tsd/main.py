"""CLI entry point for Trading System Discovery.

Orchestrates the full pipeline: load data, optimize strategies,
validate with walk-forward analysis, test robustness, persist results,
and generate performance reports.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import pandas as pd

from tsd.analysis.reports import generate_report
from tsd.analysis.robustness import RobustnessConfig, assess_robustness
from tsd.config import CORE_INDICATORS, Config, load_config
from tsd.data.loader import load_market_data
from tsd.export.persistence import generate_run_id, save_run
from tsd.optimization.fitness import FitnessConfig
from tsd.optimization.ga import GAConfig, load_ga_config
from tsd.optimization.pipeline import PipelineConfig, run_pipeline
from tsd.strategy.evaluator import (
    BacktestMetrics,
    BacktestResult,
    EvaluatorConfig,
    TradeRecord,
    _compute_metrics,
    run_backtest,
)
from tsd.strategy.genome import (
    OutputMeta,
    StrategyGenome,
    StrategyMeta,
    load_strategy_config,
)

LOGGER = logging.getLogger(__name__)


def _filter_strategy_meta(meta: StrategyMeta, indicator_set: str) -> StrategyMeta:
    """Filter strategy metadata to only include the selected indicator set.

    Args:
        meta: Full strategy metadata.
        indicator_set: "core" or "full".

    Returns:
        Filtered StrategyMeta. Unchanged if indicator_set is "full".
    """
    if indicator_set == "full":
        return meta

    allowed = CORE_INDICATORS
    filtered_names = tuple(n for n in meta.indicator_names if n in allowed)
    filtered_params = {k: v for k, v in meta.indicator_params.items() if k in allowed}
    filtered_outputs: dict[str, tuple[OutputMeta, ...]] = {
        k: v for k, v in meta.indicator_outputs.items() if k in allowed or k in meta.filter_names
    }
    filtered_filter_names = tuple(n for n in meta.filter_names if n in allowed)
    filtered_filter_params: dict[str, Any] = {k: v for k, v in meta.filter_params.items() if k in allowed}

    max_ind_params = max((len(p) for p in filtered_params.values()), default=0)
    max_filter_params = max((len(p) for p in filtered_filter_params.values()), default=0)
    max_ind_outputs = max(
        (len(o) for k, o in filtered_outputs.items() if k in filtered_names and o),
        default=1,
    )

    LOGGER.info(
        "Indicator set '%s': %d indicators (%s), %d filters (%s)",
        indicator_set,
        len(filtered_names),
        ", ".join(filtered_names),
        len(filtered_filter_names),
        ", ".join(filtered_filter_names),
    )

    return StrategyMeta(
        num_entry_slots=meta.num_entry_slots,
        num_indicator_exit_slots=meta.num_indicator_exit_slots,
        num_filter_slots=meta.num_filter_slots,
        indicator_names=filtered_names,
        indicator_outputs=filtered_outputs,
        indicator_params=filtered_params,
        max_indicator_params=max_ind_params,
        max_indicator_outputs=max_ind_outputs,
        filter_names=filtered_filter_names,
        filter_params=filtered_filter_params,
        max_filter_params=max_filter_params,
        comparisons=meta.comparisons,
        exit_config=meta.exit_config,
        time_exit_config=meta.time_exit_config,
    )


def main() -> int:
    """Run the main optimization pipeline."""
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_id = generate_run_id()
    LOGGER.info("=" * 60)
    LOGGER.info("Trading System Discovery — run %s", run_id)
    LOGGER.info("=" * 60)
    ga_config = load_ga_config()
    LOGGER.info("Market: %s", config.market)
    LOGGER.info("Indicator set: %s", config.indicator_set)
    LOGGER.info("Pipeline mode: %s", config.pipeline_mode)
    LOGGER.info(
        "GA: pop=%d, gens=%d, workers=%d",
        ga_config.population_size,
        ga_config.max_generations,
        ga_config.n_workers,
    )

    try:
        return _run_pipeline(config, run_id, ga_config)
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1


def _run_pipeline(config: Config, run_id: str, ga_config: GAConfig) -> int:
    """Execute the full pipeline.

    Args:
        config: Application configuration.
        run_id: Unique run identifier.
        ga_config: GA configuration.

    Returns:
        Exit code (0 for success).
    """
    t0 = time.monotonic()

    # --- Step 1: Load market data ---
    LOGGER.info("[1/5] Loading market data for '%s'...", config.market)
    stocks_data = load_market_data(config.market, config.data_dir)
    LOGGER.info(
        "Loaded %d stocks, date range: %s to %s",
        len(stocks_data),
        min(df.index.min() for df in stocks_data.values()),
        max(df.index.max() for df in stocks_data.values()),
    )

    # --- Step 2: Load strategy config (with indicator filtering) ---
    LOGGER.info("[2/5] Loading strategy configuration...")
    meta = load_strategy_config(config.config_dir)
    meta = _filter_strategy_meta(meta, config.indicator_set)
    LOGGER.info(
        "Strategy space: %d entry slots, %d indicators, %d filters, %d comparisons",
        meta.num_entry_slots,
        len(meta.indicator_names),
        len(meta.filter_names),
        len(meta.comparisons),
    )

    # --- Step 3: Run optimization pipeline ---
    LOGGER.info("[3/5] Running optimization (mode=%s)...", config.pipeline_mode)
    pipeline_result = run_pipeline(
        meta=meta,
        stocks_data=stocks_data,
        indicator_outputs=meta.indicator_outputs,
        pipeline_config=PipelineConfig(mode=config.pipeline_mode),
        ga_config=ga_config,
        eval_config=EvaluatorConfig(),
        fitness_config=FitnessConfig(),
    )
    LOGGER.info(
        "Optimization complete: best_fitness=%.4f",
        pipeline_result.best_fitness,
    )

    # --- Step 4: Full backtest + robustness ---
    LOGGER.info("[4/5] Running full backtest and robustness testing...")
    backtest_result = _run_full_backtest(
        pipeline_result.best_genome,
        stocks_data,
        meta,
        EvaluatorConfig(),
    )
    LOGGER.info(
        "Full backtest: %d trades, win_rate=%.4f, net_profit=%.2f",
        backtest_result.metrics.num_trades,
        backtest_result.metrics.win_rate,
        backtest_result.metrics.net_profit,
    )

    robustness_result = assess_robustness(
        backtest_result,
        RobustnessConfig(mc_n_permutations=1000, bs_n_resamples=1000),
    )
    LOGGER.info("Robustness: passed=%s, skipped=%s", robustness_result.passed, robustness_result.skipped)

    # --- Step 5: Persist results and generate report ---
    LOGGER.info("[5/5] Saving results and generating report...")
    report = generate_report(
        run_id=run_id,
        genome=pipeline_result.best_genome,
        backtest_result=backtest_result,
        pipeline_result=pipeline_result,
        robustness_result=robustness_result,
    )

    manifest = save_run(
        run_id=run_id,
        results_dir=config.results_dir,
        pipeline_result=pipeline_result,
        robustness_result=robustness_result,
        backtest_result=backtest_result,
        report=report,
    )

    elapsed = time.monotonic() - t0
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline complete in %.1f seconds", elapsed)
    LOGGER.info("Run ID: %s", run_id)
    LOGGER.info("Report: %s", manifest.report_path)
    LOGGER.info("Strategy: %s", manifest.strategy_path)
    LOGGER.info(
        "Result: fitness=%.4f, trades=%d, win_rate=%.4f",
        pipeline_result.best_fitness,
        backtest_result.metrics.num_trades,
        backtest_result.metrics.win_rate,
    )
    LOGGER.info("=" * 60)
    return 0


def _run_full_backtest(
    genome: StrategyGenome,
    stocks_data: dict[str, pd.DataFrame],
    meta: StrategyMeta,
    eval_config: EvaluatorConfig,
) -> BacktestResult:
    """Run backtest across all stocks and merge trades.

    Args:
        genome: Best strategy genome.
        stocks_data: All stock DataFrames.
        meta: Strategy metadata.
        eval_config: Evaluator configuration.

    Returns:
        Combined BacktestResult with trades from all stocks.
    """
    all_trades: list[TradeRecord] = []
    for ticker, df in stocks_data.items():
        try:
            result = run_backtest(genome, df, meta.indicator_outputs, eval_config)
            all_trades.extend(result.trades)
        except Exception:
            LOGGER.warning("Backtest failed for %s, skipping", ticker, exc_info=True)

    if not all_trades:
        return BacktestResult(
            trades=(),
            metrics=BacktestMetrics(
                num_trades=0,
                num_wins=0,
                num_losses=0,
                win_rate=0.0,
                net_profit=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
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

    metrics = _compute_metrics(all_trades, eval_config)
    return BacktestResult(trades=tuple(all_trades), metrics=metrics)


if __name__ == "__main__":
    sys.exit(main())
