# TaskSpec: 012 - Performance Reporting

Status: implemented

## Objective

Generate comprehensive JSON performance reports that aggregate strategy
descriptions, optimization summaries, walk-forward results, robustness
checks, and trade analysis into a single human-readable document.

## Context / References

- `docs/final_project_specification.md` — reporting requirements
- `tsd/export/persistence.py` — existing result persistence (TaskSpec 011)
- `tsd/analysis/robustness.py` — robustness types
- `tsd/optimization/pipeline.py` — pipeline result types
- `tsd/optimization/walkforward.py` — walk-forward result types

## Scope

**In scope:**
- Report dataclasses (StrategySummary, OptimizationSummary, FitnessEvolution,
  WalkForwardSummary, RobustnessSummary, TradeAnalysis, PerformanceReport)
- Human-readable strategy description from genome
- Derived trade analysis (cumulative P&L, exit type distribution, monthly
  returns, winner/loser stats)
- `generate_report()` and `save_report()` public API
- Integration with `RunManifest` and `save_run()` via `report_path`
- Unit tests for all builder functions and public API

**Out of scope:**
- HTML/PDF report rendering
- Equity curve charts (downstream concern)
- Interactive dashboards

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies — uses stdlib json + dataclasses only.
- Report saved as pretty-printed JSON for easy inspection.

## Acceptance Criteria (must be testable)

- [x] `PerformanceReport` frozen dataclass with all report sections
- [x] `_describe_indicator()` produces human-readable indicator strings
- [x] `_describe_exit()` produces exit type descriptions
- [x] `_describe_filter()` produces filter descriptions
- [x] `_build_strategy_summary()` produces StrategySummary from genome
- [x] `_build_optimization_summary()` handles ga_only, bayesian_only, both
- [x] `_build_fitness_evolution()` extracts GA logbook data
- [x] `_build_walkforward_summary()` includes window summaries and holdout
- [x] `_build_robustness_summary()` includes permutation and bootstrap results
- [x] `_build_trade_analysis()` computes cumulative P&L, exit counts, monthly returns
- [x] `generate_report()` assembles all sections
- [x] `save_report()` writes valid JSON to results/performance/
- [x] `RunManifest` has `report_path` field
- [x] `save_run()` accepts optional `report` parameter
- [x] 24 unit tests pass
- [x] All 385 tests pass
- [x] ruff check, ruff format, mypy all pass

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_reports.py -v
docker compose run --rm app pytest tests/unit/test_persistence.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/analysis/reports.py` — Report generation module (~400 lines)
- `tsd/analysis/__init__.py` — Updated re-exports
- `tsd/export/persistence.py` — `report_path` in RunManifest, `report` param in save_run
- `tests/unit/test_reports.py` — 24 unit tests
- `docs/tasks/012-performance-reporting.md` — This TaskSpec
- `docs/backlog.md` — Updated status

## Risks / Open Questions

None — straightforward aggregation of existing types.

## Learnings

- Ruff PLR0912 branch limit (12) requires splitting `_describe_exit()` into
  `_describe_limit_exits()` and `_describe_time_exits()` helpers.
- `save_run()` needed `# noqa: PLR0913` after adding the `report` parameter.

## Follow-ups / Backlog (if not done here)

- 013 - Pine Script export
