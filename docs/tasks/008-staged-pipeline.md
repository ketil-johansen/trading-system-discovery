# TaskSpec: 008 - Staged Pipeline

Status: implemented

## Objective

Wire together the GA engine (006) and Bayesian optimizer (007) into a
single staged pipeline orchestrator with configurable modes (ga_only,
bayesian_only, both).

## Context / References

- `tsd/optimization/ga.py` — GA engine (TaskSpec 006)
- `tsd/optimization/bayesian.py` — Bayesian optimizer (TaskSpec 007)
- `docs/final_project_specification.md` §4 — Stage A (GA) discovers
  structure, Stage B (Optuna) fine-tunes parameters

## Scope

**In scope:**
- `PipelineConfig` and `PipelineResult` frozen dataclasses
- `load_pipeline_config()` factory from `TSD_PIPELINE_MODE` and
  `TSD_PIPELINE_RESUME` env vars
- `run_pipeline()` public API with three modes
- Mode validation, logging, resume passthrough
- Re-exports in `tsd/optimization/__init__.py`
- Unit tests with mocked GA/Bayesian engines

**Out of scope:**
- `main.py` wiring (deferred to 009)
- Walk-forward validation integration

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies.

## Acceptance Criteria (must be testable)

- [x] `PipelineConfig` dataclass with `mode` and `resume` fields,
  frozen, with `load_pipeline_config()` reading env vars
- [x] `PipelineResult` dataclass holds `mode`, `best_genome`,
  `best_fitness`, optional `ga_result` and `bayesian_result`
- [x] `run_pipeline()` dispatches correctly for all three modes
- [x] `bayesian_only` mode raises `ValueError` if `seed_genome` is None
- [x] Invalid mode raises `ValueError`
- [x] Resume flag is forwarded to both engines
- [x] In `both` mode, GA best genome is passed to Bayesian stage
- [x] Re-exports added to `tsd/optimization/__init__.py`
- [x] All tests pass: `pytest tests/unit/test_pipeline.py -v`
- [x] `ruff check`, `ruff format --check`, `mypy` all clean

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_pipeline.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/optimization/pipeline.py` — Pipeline implementation
- `tsd/optimization/__init__.py` — Updated re-exports
- `tests/unit/test_pipeline.py` — 13 unit tests

## Risks / Open Questions

- None.

## Learnings

- Pipeline is a thin orchestrator — no complex logic needed, just
  dispatching and logging. Helper functions per mode keep `run_pipeline`
  clean and under statement limits.

## Follow-ups / Backlog (if not done here)

- 009: Walk-forward validation and main.py wiring
