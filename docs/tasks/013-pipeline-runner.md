# TaskSpec: 013 - Pipeline Runner

Status: implemented

## Objective

Wire up the full end-to-end pipeline so users can run `python -m tsd.main`
inside Docker and see the complete optimization flow: load data, optimize
strategies, backtest, test robustness, persist results, and generate reports.

## Context / References

- `tsd/main.py` — CLI entry point (was a stub, now fully implemented).
- `tsd/data/loader.py` — Market data loader (new module).
- `tsd/config.py` — Added `market`, `indicator_set`, `pipeline_mode` fields
  and `CORE_INDICATORS` constant.
- `docker-compose.yml` — Added `TSD_MARKET`, `TSD_INDICATOR_SET`,
  `TSD_PIPELINE_MODE` environment variables.

## Scope

**In scope:**
- Full pipeline orchestration in `main.py` (5 steps with milestone logging).
- Market data loader (`tsd/data/loader.py`) reading cached Parquet files.
- Indicator set filtering (`_filter_strategy_meta()`) for "core" vs "full" mode.
- Three new config fields and corresponding env vars.
- Clear progress logging at each pipeline step.
- Unit tests for loader and main filtering logic.

**Out of scope:**
- Pine Script export (deferred).
- New indicators or strategy types.
- Portfolio simulation.

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies.
- Config directory is mounted read-only (`:ro`) — indicator filtering is
  in-memory, not by rewriting YAML files.

## Acceptance Criteria (must be testable)

- [x] `tsd/data/loader.py` loads all Parquet files for a market and returns
  `dict[str, pd.DataFrame]`.
- [x] Loader raises `FileNotFoundError` for missing market directories and
  `ValueError` for empty directories.
- [x] `_filter_strategy_meta()` reduces indicator set when `indicator_set="core"`.
- [x] `_filter_strategy_meta()` returns meta unchanged when `indicator_set="full"`.
- [x] `CORE_INDICATORS` contains the expected 7 indicators.
- [x] `Config` has `market`, `indicator_set`, `pipeline_mode` fields.
- [x] `docker-compose.yml` passes `TSD_MARKET`, `TSD_INDICATOR_SET`,
  `TSD_PIPELINE_MODE` to the container.
- [x] All tests pass: `pytest -v` (394 tests).
- [x] `ruff check`, `ruff format --check`, `mypy tsd/` all clean.

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_loader.py -v
docker compose run --rm app pytest tests/unit/test_main.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/main.py` — Full pipeline orchestration (rewrite from stub).
- `tsd/data/loader.py` — Market data loader (new).
- `tsd/config.py` — Added 3 config fields + `CORE_INDICATORS`.
- `docker-compose.yml` — Added 3 environment variables.
- `tests/unit/test_loader.py` — 4 unit tests (new).
- `tests/unit/test_main.py` — 5 unit tests (new).
- `tests/unit/test_config.py` — Updated Config constructors.
- `docs/tasks/013-pipeline-runner.md` — This TaskSpec.
- `docs/backlog.md` — Updated status.

## Risks / Open Questions

- Running the full pipeline requires downloaded market data in `data/raw/`.
  The pipeline will fail at step 1 if data hasn't been fetched first.

## Learnings

- Config directory is mounted as `:ro` in docker-compose.yml, so indicator
  filtering must happen in-memory via `_filter_strategy_meta()` rather than
  by rewriting config YAML files.
- `StrategyMeta` is a frozen dataclass, so filtering creates a new instance
  with filtered tuples/dicts rather than modifying in place.

## Follow-ups / Backlog (if not done here)

- Docker log tailing convenience commands for monitoring long pipeline runs.
- Pine Script export (original 013, deferred to future TaskSpec).
