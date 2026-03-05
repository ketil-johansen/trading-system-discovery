# TaskSpec: 009 - Walk-Forward Validation

Status: implemented

## Objective

Implement an anchored walk-forward validation engine that tests strategy
robustness by re-optimizing on growing in-sample windows and evaluating
on unseen out-of-sample data, with a final holdout check.

## Context / References

- `docs/final_project_specification.md` §7 — Anchored WF with 6-month
  OOS windows, 12-month holdout, passing criteria on win rate and
  profitability.
- `tsd/optimization/pipeline.py` — Staged GA → Optuna pipeline (008)
- `tsd/optimization/ga.py`, `tsd/optimization/bayesian.py` — Shared
  `_aggregate_metrics` refactored to `tsd/optimization/metrics.py`.

## Scope

**In scope:**
- Extract shared `aggregate_metrics` / `empty_metrics` to
  `tsd/optimization/metrics.py`
- `WalkForwardConfig`, `WalkForwardWindow`, `WindowResult`,
  `HoldoutResult`, `WalkForwardResult` frozen dataclasses
- `load_walkforward_config()` from `TSD_WF_*` env vars
- `generate_windows()` for anchored window boundaries + holdout
- `run_walkforward()` orchestrator: per-window IS optimize + OOS eval,
  best genome selection, holdout evaluation, passing criteria
- Unit tests for all components (~28 tests)
- Re-exports in `tsd/optimization/__init__.py`

**Out of scope:**
- Integration with `main.py` (deferred)
- Monte Carlo / statistical significance tests (010)
- Parallel window processing

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies.

## Acceptance Criteria (must be testable)

- [x] `aggregate_metrics` and `empty_metrics` extracted to shared
  `tsd/optimization/metrics.py` and used by ga.py, bayesian.py,
  and walkforward.py
- [x] `WalkForwardConfig` dataclass with `load_walkforward_config()`
  reading `TSD_WF_*` env vars
- [x] `generate_windows()` produces anchored windows with no
  OOS/holdout overlap and IS_end == OOS_start
- [x] `run_walkforward()` calls pipeline per IS window, evaluates
  OOS, selects best genome, evaluates holdout
- [x] Passing criteria: win rate windows, profitable windows, holdout
  tolerance — all evaluated correctly
- [x] Low frequency flag set when any window has few trades
- [x] Short data raises `ValueError`
- [x] Re-exports added to `tsd/optimization/__init__.py`
- [x] All tests pass: `pytest tests/unit/test_metrics.py
  tests/unit/test_walkforward.py -v`
- [x] Existing GA and Bayesian tests still pass after metrics extraction
- [x] `ruff check`, `ruff format --check`, `mypy` all clean

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_metrics.py tests/unit/test_walkforward.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/optimization/metrics.py` — Shared metric aggregation
- `tsd/optimization/walkforward.py` — Walk-forward validation engine
- `tsd/optimization/__init__.py` — Updated re-exports
- `tsd/optimization/ga.py` — Uses shared metrics
- `tsd/optimization/bayesian.py` — Uses shared metrics
- `tests/unit/test_metrics.py` — 4 unit tests
- `tests/unit/test_walkforward.py` — 24 unit tests
- `tests/unit/test_ga.py` — Updated import path
- `tests/unit/test_bayesian.py` — Updated import path

## Risks / Open Questions

- None.

## Learnings

- Walk-forward is a thin orchestrator over pipeline + evaluator. The key
  complexity is in window generation (date arithmetic with DateOffset)
  and passing criteria evaluation.
- Extracting `aggregate_metrics` into a shared module eliminates
  duplication across three consumers (GA, Bayesian, WF).

## Follow-ups / Backlog (if not done here)

- 010: Statistical significance testing (Monte Carlo, bootstrap)
