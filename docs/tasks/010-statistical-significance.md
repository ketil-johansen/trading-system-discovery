# TaskSpec: 010 - Statistical Significance Testing

Status: implemented

## Objective

Add Monte Carlo permutation tests and bootstrap confidence intervals to
distinguish genuine trading edge from random chance, completing the
validation pipeline alongside walk-forward analysis (009).

## Context / References

- `tsd/analysis/robustness.py` — was a stub, now contains full implementation.
- `tsd/strategy/evaluator.py` — BacktestResult / TradeRecord used as input.
- `docs/final_project_specification.md` — robustness checks requirement.

## Scope

**In scope:**
- Sign-flip Monte Carlo permutation test on win_rate, net_profit, sharpe.
- Percentile bootstrap confidence intervals on win_rate, net_profit, sharpe, expectancy.
- Combined `assess_robustness()` function with pass/fail criteria.
- Configuration via `TSD_ROBUSTNESS_*` environment variables.
- Graceful skip when too few trades (< min_trades).
- Re-exports in `tsd/analysis/__init__.py`.

**Out of scope:**
- Integration with walk-forward pipeline (future TaskSpec).
- Cross-market validation.
- Monte Carlo equity curve simulation.

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies (numpy already available).
- Vectorized numpy operations for performance.

## Acceptance Criteria (must be testable)

- [x] `run_permutation_test()` accepts returns + callable, returns p-value and significance flag.
- [x] `run_bootstrap_ci()` accepts returns + callable, returns lower/upper bounds.
- [x] `assess_robustness()` runs 3 permutation tests + 4 bootstrap CIs, returns combined result.
- [x] Graceful skip when fewer than `min_trades` trades.
- [x] Configuration loads from `TSD_ROBUSTNESS_*` environment variables.
- [x] All result dataclasses are frozen.
- [x] Reproducible results with fixed random seed.
- [x] 31 unit tests pass.
- [x] `ruff check`, `ruff format --check`, and `mypy` all pass.

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_robustness.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/analysis/robustness.py` — Full implementation (~350 lines).
- `tsd/analysis/__init__.py` — Re-exports.
- `tests/unit/test_robustness.py` — 31 unit tests.
- `docs/tasks/010-statistical-significance.md` — This TaskSpec.
- `docs/backlog.md` — Updated status.

## Learnings

- `np.ndarray` requires `# type: ignore[type-arg]` for mypy strict mode,
  consistent with existing codebase pattern in `tsd/indicators/trend.py`.
- Sign-flip permutation test is more powerful than label shuffle for
  testing H0 of symmetric returns around zero.
- Continuity-corrected p-value `(count + 1) / (n + 1)` avoids p=0.

## Follow-ups / Backlog (if not done here)

- Integration of robustness checks into walk-forward pipeline.
- Monte Carlo equity curve simulation for drawdown analysis.
