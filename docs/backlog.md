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

## Optimization Core

- [ ] **003 - GA engine (DEAP)** — Genetic algorithm with configurable
  chromosome encoding for strategy parameters, tournament selection,
  crossover, mutation. Parallel fitness evaluation across CPU cores.
- [ ] **004 - Bayesian optimizer (Optuna)** — Optuna-based parameter
  search with pruning, study persistence for checkpoint/resume.
- [ ] **005 - Strategy evaluation harness** — vectorbt-based backtest
  runner that takes parameter sets and returns fitness metrics (Sharpe,
  max drawdown, profit factor).

## Validation

- [ ] **006 - Walk-forward validation** — Rolling window out-of-sample
  validation to detect overfitting. Configurable window sizes and step
  sizes.
- [ ] **007 - Statistical significance testing** — Monte Carlo
  permutation tests, bootstrap confidence intervals on strategy metrics.

## Reporting

- [ ] **008 - Result persistence** — Save optimization results, best
  parameters, and validation metrics as structured Parquet files.
- [ ] **009 - Performance reporting** — Generate summary reports: equity
  curves, parameter distributions, fitness evolution, validation results.

## Future (not yet scoped)

- Ensemble strategies
- Multi-asset optimization
- Live paper-trading integration
- Regime detection preprocessing
