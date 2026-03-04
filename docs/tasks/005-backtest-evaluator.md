# TaskSpec: 005 - Backtest evaluator and fitness function

Status: implemented

## Objective

Build a bar-by-bar backtest engine that simulates trades from a
StrategyGenome + OHLCV DataFrame with correct execution timing per exit
category, computes 20 aggregate metrics, and provides a win-rate fitness
function with profitability gate.

## Context / References

- `docs/final_project_specification.md` §3 (execution timing, cost model)
- `tsd/strategy/genome.py` — StrategyGenome, all gene types
- `tsd/strategy/signals.py` — entry signal generation
- `tsd/strategy/exits.py` — all 3 exit categories
- `tsd/strategy/execution.py` — shift_to_next_open, check_limit_exit

## Scope

**In scope:**
- Custom bar-by-bar simulation loop (no vectorbt dependency)
- Correct execution timing: entries at next open, limit exits intraday,
  indicator/time exits at open
- Per-trade TradeRecord with cost model
- 20 aggregate BacktestMetrics (win rate, profit factor, Sharpe, Sortino,
  Calmar, drawdown, streaks, expectancy, etc.)
- Win-rate fitness function with 3 hard gates
- opposite_entry exit logic
- Re-exports in __init__.py files

**Out of scope:**
- vectorbt integration
- Portfolio-level simulation
- Walk-forward validation (TaskSpec 009)

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies.
- Custom simulator preferred over vectorbt for execution timing control.

## Acceptance Criteria (must be testable)

- [x] `run_backtest()` accepts a StrategyGenome + DataFrame and returns
  BacktestResult with trades and metrics
- [x] Entries execute at next bar's Open after signal
- [x] Limit exits check intraday with conservative same-bar rule
- [x] Indicator exits execute at next bar's Open after signal
- [x] Time exits execute at current bar's Open
- [x] Cost model deducts round_trip_cost_pct from gross returns
- [x] End-of-data closes open positions
- [x] Only one position at a time
- [x] ATR NaN bars skip entries
- [x] opposite_entry triggers exit when entry conditions no longer hold
- [x] BacktestMetrics has all 20 fields computed correctly
- [x] Zero-trade edge case returns all-zero metrics
- [x] `compute_fitness()` returns win_rate when all 3 gates pass, 0.0 otherwise
- [x] 36 unit tests pass
- [x] ruff check, ruff format, mypy all pass

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_evaluator.py tests/unit/test_fitness.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/strategy/evaluator.py` — backtest engine (new)
- `tsd/optimization/fitness.py` — fitness function (new)
- `tsd/strategy/__init__.py` — updated re-exports
- `tsd/optimization/__init__.py` — updated re-exports
- `tests/unit/test_evaluator.py` — 28 evaluator tests
- `tests/unit/test_fitness.py` — 8 fitness tests

## Risks / Open Questions

- None remaining.

## Learnings

- Custom bar-by-bar loop is simpler to verify than vectorbt for
  non-uniform execution timing across exit categories.
- Slicing DataFrames for per-trade exit levels (trailing, chandelier,
  breakeven) works well with relative indexing (rel = bar - entry_bar).
- Mock-based testing of run_backtest avoids dependency on real indicator
  computation, keeping unit tests fast and focused.

## Follow-ups / Backlog (if not done here)

- Walk-forward validation (TaskSpec 009) will call run_backtest
  on rolling windows.
- GA engine (TaskSpec 006) will use compute_fitness as the objective.
