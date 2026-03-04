# TaskSpec: 006 - GA Engine (DEAP)

Status: implemented

## Objective

Build the genetic algorithm engine that evolves strategy genomes to
discover profitable trading systems using DEAP for GA mechanics.

## Context / References

- `docs/final_project_specification.md` §4 (Optimization)
- `tsd/strategy/genome.py` — genome encoding, flat chromosome codec
- `tsd/strategy/evaluator.py` — backtest engine
- `tsd/optimization/fitness.py` — win-rate fitness with profitability gate

## Scope

**In scope:**
- GAConfig dataclass with `TSD_GA_*` env vars
- GenerationStats and GAResult dataclasses
- `run_ga()` public API: evolve strategies across multi-stock data
- Slot-level crossover (swap whole genome segments)
- Gaussian mutation (per-gene noise, clamped during decoding)
- Tournament selection with configurable size
- Elitism via DEAP HallOfFame
- Multi-stock evaluation with metric aggregation
- Parallel evaluation via `multiprocessing.Pool`
- Pickle-based checkpointing and resume
- Early stopping on fitness plateau
- Re-exports in `tsd/optimization/__init__.py`

**Out of scope:**
- Bayesian optimization (TaskSpec 007)
- Walk-forward validation (TaskSpec 009)
- Staged pipeline (TaskSpec 008)

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- DEAP already in requirements.txt; no new dependencies.
- `hasattr(creator, ...)` guard before creating DEAP types.
- Let `flat_to_genome()` handle clamping/rounding after mutation.

## Acceptance Criteria (must be testable)

- [x] `GAConfig` loads from `TSD_GA_*` env vars with sensible defaults
- [x] `run_ga()` returns `GAResult` with best genome, fitness, generation count, and logbook
- [x] Slot-level crossover swaps whole genome segments, preserving chromosome length
- [x] Gaussian mutation modifies gene values; prob=0 leaves individual unchanged
- [x] Multi-stock backtest results aggregated correctly (sum trades/wins/losses/profit)
- [x] Fitness gate failure (insufficient trades) returns 0.0
- [x] Checkpoint save/load round-trips all fields
- [x] Early stopping triggers when fitness plateaus
- [x] Resume from checkpoint continues evolution
- [x] All tests pass: `pytest tests/unit/test_ga.py -v` (19 tests)
- [x] `ruff check tsd/ tests/` passes
- [x] `ruff format --check tsd/ tests/` passes
- [x] `mypy tsd/` passes
- [x] Full test suite passes (245 tests)

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_ga.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/optimization/ga.py` — GA engine implementation
- `tsd/optimization/__init__.py` — updated re-exports
- `tests/unit/test_ga.py` — 19 unit tests
- `docs/tasks/006-ga-engine.md` — this TaskSpec
- `docs/backlog.md` — updated status

## Risks / Open Questions

- Parallel evaluation with `multiprocessing.Pool` requires picklable
  individuals and evaluation function. DEAP handles this natively.
- Pickle checkpoints are not portable across Python versions.

## Learnings

- `multiprocessing.Pool` must be imported from `multiprocessing.pool`
  (not `multiprocessing`) for mypy type annotation compatibility.
- DEAP types in `creator` module are global singletons — need
  `hasattr` guard to avoid re-creation errors in tests.
- `run_ga()` needed refactoring into helper functions
  (`_init_population`, `_assign_fitness`, `_evolve_loop`,
  `_breed_generation`) to stay under ruff's branch/statement limits
  (PLR0912, PLR0915).
- deap is already in mypy ignore list in pyproject.toml, so
  `# type: ignore[import-untyped]` comments are unnecessary.

## Follow-ups / Backlog (if not done here)

- TaskSpec 007: Bayesian optimizer (Optuna)
- TaskSpec 008: Staged GA → Optuna pipeline
