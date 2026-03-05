# TaskSpec: 011 - Result Persistence

Status: implemented

Note: Update Status and check off completed acceptance criteria as part
of the feature branch —
never as a separate commit to `main`. The TaskSpec update must be
included in the PR
that delivers the work.

## Objective

Persist optimization results (genomes, pipeline results, walk-forward
results, robustness results, trade records) to structured JSON and
Parquet files so runs can be reviewed, compared, and fed into downstream
reporting and Pine Script export.

## Context / References

- `tsd/strategy/genome.py` — StrategyGenome dataclass
- `tsd/optimization/pipeline.py` — PipelineResult dataclass
- `tsd/optimization/walkforward.py` — WalkForwardResult dataclass
- `tsd/analysis/robustness.py` — RobustnessResult dataclass
- `tsd/strategy/evaluator.py` — BacktestResult, TradeRecord
- `tsd/data/quality.py:save_report()` — Existing JSON serialization pattern

## Scope

**In scope:**
- JSON serialization of frozen dataclasses with special float handling
  (inf, -inf, nan) and pd.Timestamp conversion
- Generic dict-to-dataclass reconstruction for round-trip loading
- Genome save/load
- PipelineResult save/load
- WalkForwardResult save (with stripped pipeline in windows)
- RobustnessResult save
- TradeRecord Parquet save/load
- Run manifest, run ID generation, JSONL event log
- Public API: save_run(), load_run(), generate_run_id(), save_run_log()

**Out of scope:**
- Performance reporting (TaskSpec 012)
- Pine Script export (TaskSpec 013)
- Database storage
- Run comparison or diffing

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies — uses stdlib json, dataclasses plus
  existing pyarrow/pandas.
- JSON for structured results, Parquet for tabular trade records.

## Acceptance Criteria (must be testable)

- [x] `generate_run_id()` produces YYYYMMDD_HHMMSS_hex8 format IDs
- [x] `_sanitize_dict()` handles inf, -inf, nan, pd.Timestamp, nested structures
- [x] `_dict_to_dataclass()` reconstructs nested frozen dataclasses, tuples, optionals
- [x] Genome round-trips through save/load (minimal and complex)
- [x] PipelineResult round-trips through save/load (including infinity values)
- [x] WalkForwardResult saves valid JSON with stripped pipeline in windows
- [x] RobustnessResult saves valid JSON (normal and skipped)
- [x] TradeRecord round-trips through Parquet (including empty case)
- [x] `save_run()` creates all expected files when all components provided
- [x] `save_run()` handles None components gracefully
- [x] `save_run_log()` appends JSONL lines with timestamps
- [x] `load_run()` loads saved manifests and raises on missing
- [x] `RunManifest` is frozen/immutable
- [x] All 31 unit tests pass
- [x] ruff check, ruff format, mypy all clean

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_persistence.py -v
docker compose run --rm app pytest -v
docker compose run --rm app ruff check tsd/ tests/
docker compose run --rm app ruff format --check tsd/ tests/
docker compose run --rm app mypy tsd/
```

## Deliverables

- `tsd/export/persistence.py` — New module (~360 lines)
- `tsd/export/__init__.py` — Updated with re-exports
- `tests/unit/test_persistence.py` — 31 unit tests
- `docs/tasks/011-result-persistence.md` — This TaskSpec
- `docs/backlog.md` — Updated status
- `pyproject.toml` — Added pyarrow to mypy ignore list
- `docker-compose.yml` — Mount pyproject.toml as volume

## Risks / Open Questions

- None identified.

## Learnings

- pyarrow needs to be in mypy ignore list (no py.typed marker)
- pyproject.toml was not volume-mounted in docker-compose.yml, causing
  mypy config changes to require a rebuild. Now mounted as read-only.
- `from __future__ import annotations` makes `X | None` syntax work
  in Python 3.10 for type hints at runtime, but `types.UnionType` is
  only available at runtime for actual union type objects.

## Follow-ups / Backlog (if not done here)

- Load helpers for walkforward/robustness results (not needed yet —
  only genome and pipeline results need round-trip loading currently)
