# Backlog

Ordered list of planned work. Each item becomes a TaskSpec in
`docs/tasks/` when work begins.
Items near the top are better defined; items further down are rougher.

## Data Foundation

- [x] **001 - Market data pipeline** — Fetch OHLCV data via yfinance,
  cache as Parquet, handle missing data and splits.
- [x] **002 - Data validation and preprocessing** — Validate downloaded
  data integrity, compute derived features via pandas_ta, handle edge
  cases (gaps, holidays, delistings).

## Indicators & Backtesting (Phase 2)

- [x] **003 - Indicator library (core subset)** — Base indicator
  interface, parameter metadata from `config/indicators.yaml`, and
  16 core indicators across all 6 categories (trend, momentum,
  volatility, volume, regime filters) using the `ta` library.
- [x] **004 - Strategy genome and signal generation** — Strategy DNA
  encoding with parameter genes and binary switches, signal generation
  from genome, exit types (all 3 categories), execution timing rules.
- [x] **005 - Backtest evaluator and fitness function** — Custom bar-by-bar
  backtest engine with correct execution timing (§3), per-trade metrics,
  20 aggregate BacktestMetrics, win-rate fitness with profitability gate
  and cost model.
- [ ] **005b - Remaining indicators** — Complete the full ~55 indicator
  universe. Can be done any time after 003.

## Optimization (Phase 3–4)

- [x] **006 - GA engine (DEAP)** — Genetic algorithm with tournament
  selection, crossover, mutation, elitism. Parallel fitness evaluation
  across CPU cores. Checkpointing and resume.
- [x] **007 - Bayesian optimizer (Optuna)** — Optuna-based parameter
  fine-tuning with TPE sampler, median pruner, study persistence.
- [x] **008 - Staged pipeline** — GA → Optuna pipeline with configurable
  modes (ga_only, bayesian_only, both).

## Validation (Phase 5)

- [x] **009 - Walk-forward validation** — Anchored walk-forward engine
  with growing in-sample, 6-month OOS windows, final holdout. Passing
  criteria evaluation.
- [x] **010 - Statistical significance testing** — Monte Carlo
  permutation tests, bootstrap confidence intervals on strategy metrics.

## Reporting (Phase 6)

- [x] **011 - Result persistence** — Save optimization results, best
  parameters, and validation metrics as structured Parquet/JSON files.
- [x] **012 - Performance reporting** — Generate summary reports: equity
  curves, parameter distributions, fitness evolution, validation results.
- [x] **013 - Pipeline runner** — Wire up full end-to-end pipeline with
  market data loading, indicator set filtering, progress logging, and
  Docker environment configuration.

## Export (Phase 7)

- [ ] **014 - Pine Script export** — Generate TradingView Pine Script
  from validated strategy configs.

## Future (not yet scoped)

- Portfolio simulation (Layer 2, Phase 7)
- Ensemble strategies
- Multi-asset optimization
- Live paper-trading integration
- Regime detection preprocessing
- Cross-market validation
