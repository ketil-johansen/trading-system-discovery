# TaskSpec: 002 - Data validation and quality checks

Status: draft

## Objective

Validate downloaded OHLCV data for integrity and completeness, generate
per-market quality reports, and provide a CLI script to run checks on
demand.

## Context / References

- `docs/final_project_specification.md` — Section 4 (Data Specification):
  split-adjusted only, gaps allowed (trades spanning gaps not generated),
  survivorship bias acknowledged, no corporate event modeling.
- `docs/tasks/001-market-data-pipeline.md` — delivers raw Parquet files
  in `data/raw/{market_key}/{ticker}.parquet` with columns Open, High,
  Low, Close, Volume and a DatetimeIndex named "Date".
- `tsd/data/quality.py` — placeholder module from TaskSpec 001.
- `CLAUDE.md` — directory layout shows `data/reports/` for quality reports
  and `scripts/check_data_quality.py` for the CLI.

## Scope

**In scope:**
- `tsd/data/quality.py` — validation functions:
  - Column validation: OHLCV columns present, correct dtypes (float64
    for prices, int64 for volume).
  - Index validation: DatetimeIndex, sorted ascending, no duplicates,
    named "Date".
  - Null detection: count and report any NaN/None values in OHLCV columns.
  - Price ordering: High >= max(Open, Close) and Low <= min(Open, Close)
    per bar. Report violations.
  - Gap detection: identify gaps longer than 1 trading day (weekends and
    exchange holidays excluded). Report gap count and max gap length.
  - Data coverage: percentage of expected trading days present in the
    date range.
  - Outlier flagging: detect single-day returns exceeding a configurable
    threshold (default: 50%). Log as warnings, do not remove data.
  - Per-stock quality score: PASS / WARN / FAIL based on configurable
    thresholds.
  - Per-market summary: total stocks, pass/warn/fail counts, overall
    data availability.
- `StockQualityResult` frozen dataclass — per-stock validation result.
- `MarketQualityReport` frozen dataclass — per-market summary.
- `validate_stock(path: Path) -> StockQualityResult` — validate one
  Parquet file.
- `validate_market(market_key: str, data_dir: Path) -> MarketQualityReport`
  — validate all stocks in a market.
- `save_report(report: MarketQualityReport, data_dir: Path) -> Path` —
  write JSON report to `data/reports/`.
- `scripts/check_data_quality.py` — CLI with `--market <key>` (optional,
  default: all). Prints summary to stdout, saves JSON report.
- Unit tests for all validation functions.
- Integration test: validate real OMXS30 data (requires prior download).

**Out of scope:**
- Data cleaning or repair (this task reports issues, does not fix them).
- Indicator computation (TaskSpec 003+).
- Exchange holiday calendars (use simple heuristic: gaps > 5 calendar
  days are flagged).
- Quality-based YAML config file (use env vars per CLAUDE.md convention).

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies — pandas, numpy, pyarrow already available.
- Follow CLAUDE.md conventions: `from __future__ import annotations`,
  type hints, Google-style docstrings, frozen dataclasses, stdlib logging.
- Quality thresholds configurable via environment variables (`TSD_*`).
- Reports written as JSON to `data/reports/` (gitignored).
- Do not modify or delete any downloaded Parquet files.

## Acceptance Criteria (must be testable)

- [ ] `validate_stock()` detects missing columns and returns FAIL.
- [ ] `validate_stock()` detects null values and reports count.
- [ ] `validate_stock()` detects OHLC ordering violations and reports count.
- [ ] `validate_stock()` detects gaps > threshold and reports them.
- [ ] `validate_stock()` flags outlier returns exceeding threshold.
- [ ] `validate_stock()` returns PASS for clean synthetic data.
- [ ] `validate_market()` aggregates per-stock results into a summary.
- [ ] `save_report()` writes valid JSON to `data/reports/`.
- [ ] `scripts/check_data_quality.py --market omxs30` runs and prints
  summary.
- [ ] All unit tests pass: `pytest tests/unit/test_quality.py -v`
- [ ] Integration test passes: `pytest tests/integration/test_quality_pipeline.py -v`
- [ ] `ruff check tsd/ scripts/ tests/` passes with no errors.
- [ ] `ruff format --check tsd/ scripts/ tests/` passes.
- [ ] `mypy tsd/` passes with no errors.

## Verification

```bash
# Full test suite
docker compose run --rm app pytest -v

# Lint and type-check
docker compose run --rm app ruff check tsd/ scripts/ tests/
docker compose run --rm app ruff format --check tsd/ scripts/ tests/
docker compose run --rm app mypy tsd/

# Smoke test: quality check on downloaded data
docker compose run --rm app python scripts/check_data_quality.py --market omxs30

# Verify JSON report output
docker compose run --rm app python -c "
import json, pathlib
reports = list(pathlib.Path('data/reports').glob('*.json'))
print(f'{len(reports)} report(s)')
r = json.loads(reports[0].read_text())
print(f'Market: {r[\"market_key\"]}, Stocks: {r[\"total_stocks\"]}')
"
```

Requires OMXS30 data to be downloaded first (TaskSpec 001).

## Deliverables

- `tsd/data/quality.py` — validation functions and dataclasses.
- `scripts/check_data_quality.py` — quality check CLI.
- `tests/unit/test_quality.py` — unit tests for validation logic.
- `tests/integration/test_quality_pipeline.py` — end-to-end quality check.
- `docs/backlog.md` — check off item 002.

## Risks / Open Questions

- **Gap detection without holiday calendars:** Using a simple heuristic
  (gaps > 5 calendar days) may produce false positives around long
  holidays (e.g., Christmas week in Nordic markets). Acceptable for
  Phase 1.
- **Outlier threshold:** 50% single-day return threshold may miss subtle
  data errors but avoids flagging legitimate moves (e.g., earnings gaps).
  Can be tuned later.
- **Performance:** Validating 500+ S&P 500 stocks should complete in
  under a minute. If slow, can parallelize per-stock validation.

## Learnings

(Populated during implementation.)

## Follow-ups / Backlog (if not done here)

- Exchange holiday calendar integration for precise gap detection.
- Data repair utilities (fill small gaps, fix ordering violations).
- Automated data freshness monitoring.
