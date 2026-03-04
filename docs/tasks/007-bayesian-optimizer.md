# TaskSpec: 007 - Bayesian optimizer (Optuna)

Status: implemented

## Objective

Implement Optuna-based Bayesian parameter optimization (Stage B) that
fine-tunes numeric parameters of a fixed genome structure discovered by
the GA engine.

## Context / References

- `docs/final_project_specification.md` §4 Stage B
- `tsd/optimization/ga.py` — GA engine (Stage A) pattern reference
- `tsd/strategy/genome.py` — StrategyGenome, StrategyMeta, gene dataclasses
- `tsd/strategy/evaluator.py` — run_backtest, BacktestResult, EvaluatorConfig
- `tsd/optimization/fitness.py` — compute_fitness, FitnessConfig

## Scope

**In scope:**
- `BayesianConfig` frozen dataclass with `TSD_BAYESIAN_*` env var loading
- `BayesianResult` frozen dataclass with best genome, fitness, trial info
- `run_bayesian()` public API taking a genome + meta + data
- TPE sampler with MedianPruner, SQLite persistence, resume support
- Parameter suggestion for all genome numeric fields (indicator params,
  thresholds, limit exit values, time exit values, filter params)
- Multi-stock evaluation with metric aggregation
- Periodic logging callback
- Unit tests with mocked backtests

**Out of scope:**
- Staged GA → Optuna pipeline (TaskSpec 008)
- Walk-forward validation integration
- Parallel trial evaluation

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- Optuna 3.6.0 already in requirements.txt.
- mypy ignores optuna.* (configured in pyproject.toml).
- Preserve genome structure (enabled flags, indicator names, comparisons)
  — only fine-tune numeric parameters.

## Acceptance Criteria (must be testable)

- [x] `BayesianConfig` loads from `TSD_BAYESIAN_*` env vars with sensible defaults
- [x] `BayesianConfig` is a frozen dataclass
- [x] `run_bayesian()` accepts a genome, meta, stocks_data, and config objects
- [x] Optuna study uses TPE sampler and MedianPruner
- [x] Study persisted to SQLite for resume support
- [x] `_suggest_genome()` preserves structure (enabled, names, comparisons)
- [x] `_suggest_genome()` respects parameter bounds from meta
- [x] Disabled slots keep original parameters unchanged
- [x] Multi-stock metric aggregation sums trades/wins/losses correctly
- [x] Objective returns fitness from `compute_fitness()`
- [x] Zero fitness when gates fail (insufficient trades)
- [x] `BayesianResult` contains best_genome, best_fitness, trials_run, best_params
- [x] Resume adds trials to existing study
- [x] Re-exports added to `tsd/optimization/__init__.py`
- [x] All 258 tests pass, ruff clean, mypy clean

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_bayesian.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/optimization/bayesian.py` — Bayesian optimizer implementation
- `tsd/optimization/__init__.py` — Updated re-exports
- `tests/unit/test_bayesian.py` — 13 unit tests
- `docs/tasks/007-bayesian-optimizer.md` — This TaskSpec
- `docs/backlog.md` — Updated status

## Risks / Open Questions

None remaining.

## Learnings

- `optuna.trial.FrozenTrial.value` is `float | None` — need to handle
  the None case for mypy strict mode.
- Optuna's `create_study(load_if_exists=True)` with SQLite storage
  cleanly handles resume by loading existing trials.
- Breaking `_suggest_genome` into per-section helpers keeps each function
  under ruff complexity limits.

## Follow-ups / Backlog (if not done here)

- TaskSpec 008: Staged GA → Optuna pipeline
