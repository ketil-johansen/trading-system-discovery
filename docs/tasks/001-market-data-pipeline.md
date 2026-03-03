# TaskSpec: 001 - Market data pipeline

Status: implemented

## Objective

Build the foundational data pipeline: project package skeleton, configuration
system, constituent lists for all 6 markets, and a yfinance-based OHLCV
downloader that stores data as Parquet files.

## Context / References

- `docs/final_project_specification.md` — Section 2 (Target Markets),
  Section 4 (Data Specification), Section 11 (Project Structure),
  Section 12 Phase 1.
- `docs/backlog.md` — item 001.
- `CLAUDE.md` — directory layout, code conventions, configuration approach.

## Scope

**In scope:**
- `tsd/` package skeleton with `__init__.py` for all subpackages defined
  in the directory layout (data, indicators, strategy, optimization,
  analysis, portfolio, export). Placeholder modules with docstrings only
  for packages beyond Phase 1.
- `tsd/config.py` — frozen dataclass with `env_*()` helpers for runtime
  settings, plus a YAML loader for strategy/market definitions.
- `config/markets.yaml` — all 6 markets with index ticker, stock suffix,
  and market key as defined in spec Section 2.
- `tsd/data/constituents.py` — load constituent lists from CSV; for
  Nasdaq 100 and S&P 500, provide a scraper to refresh from Wikipedia.
  Nordic lists (OMXS30, OMXC25, OMXH25, OBX) are curated manually and
  committed as CSV files (small lists, change quarterly, no reliable
  public API).
- `data/constituents/` — CSV files for all 6 markets with columns:
  `ticker`, `name`, `yahoo_ticker`.
- `tsd/data/downloader.py` — download daily OHLCV from yfinance, store
  as Parquet in `data/raw/{market}/{ticker}.parquet`. Split-adjusted
  only (no dividend adjustment). Throttling (1–2s delay between stocks).
  Skip stocks whose existing Parquet is already up-to-date.
- `scripts/download_constituents.py` — CLI to refresh constituent lists.
  `--market <key>` to target one market, or all if omitted.
- `scripts/download_all_data.py` — CLI to download OHLCV data.
  `--market <key>` to target one market, or all if omitted.
- `tests/` directory structure (`conftest.py`, `unit/`, `integration/`).
- Unit tests for config loading, constituent parsing, downloader logic.
- Integration test: download OMXS30 constituent data and a small subset
  of OHLCV data to verify end-to-end flow.

**Out of scope:**
- Data quality checks and reporting (TaskSpec 002).
- Optimization/fitness/walk-forward YAML configs (not needed yet).
- Indicator computation, backtesting, or any optimization code.
- Notebooks.

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies — everything needed is already in
  requirements.txt.
- Follow CLAUDE.md conventions: `from __future__ import annotations`,
  type hints on all signatures, Google-style docstrings, frozen
  dataclasses, stdlib logging.
- Runtime config via environment variables (`TSD_*`). YAML only for
  market/strategy definitions.
- Split-adjusted prices only — use yfinance `auto_adjust=False` and
  select the appropriate columns.
- Throttle yfinance requests (1–2s between stocks) to avoid rate limits.
- Constituent CSVs are committed to git (`data/constituents/` is an
  exception to the `data/` gitignore rule).

## Acceptance Criteria (must be testable)

- [x] `tsd/` package imports successfully (`python -c "import tsd"`).
- [x] `config/markets.yaml` defines all 6 markets matching spec Section 2.
- [x] `tsd/config.py` loads runtime config from env vars with sensible
  defaults and loads YAML market definitions into frozen dataclasses.
- [x] `data/constituents/` contains a CSV for each of the 6 markets with
  columns `ticker`, `name`, `yahoo_ticker`.
- [x] `scripts/download_constituents.py --market omxs30` produces/updates
  the OMXS30 CSV.
- [x] `scripts/download_all_data.py --market omxs30` downloads OHLCV
  Parquet files to `data/raw/omxs30/`.
- [x] Downloaded Parquet files contain columns: Open, High, Low, Close,
  Volume with a DatetimeIndex.
- [x] Re-running the downloader skips stocks whose data is already
  up-to-date.
- [x] All unit tests pass: `pytest tests/unit/ -v`
- [x] All integration tests pass: `pytest tests/integration/ -v`
- [x] `ruff check tsd/ scripts/ tests/` passes with no errors.
- [x] `ruff format --check tsd/ scripts/ tests/` passes.
- [x] `mypy tsd/` passes with no errors.

## Verification

```bash
# Full test suite
docker compose run --rm app pytest -v

# Lint and type-check
docker compose run --rm app ruff check tsd/ scripts/ tests/
docker compose run --rm app ruff format --check tsd/ scripts/ tests/
docker compose run --rm app mypy tsd/

# Smoke test: constituent download
docker compose run --rm app python scripts/download_constituents.py --market omxs30

# Smoke test: data download (small market)
docker compose run --rm app python scripts/download_all_data.py --market omxs30

# Verify Parquet output
docker compose run --rm app python -c "
import pandas as pd, pathlib
files = list(pathlib.Path('data/raw/omxs30').glob('*.parquet'))
print(f'{len(files)} files')
df = pd.read_parquet(files[0])
print(df.columns.tolist())
print(df.head(2))
"
```

## Deliverables

- `tsd/` — full package skeleton with implemented data modules.
- `tsd/config.py` — configuration system.
- `tsd/data/constituents.py` — constituent list loading and scraping.
- `tsd/data/downloader.py` — yfinance OHLCV downloader.
- `config/markets.yaml` — market definitions.
- `data/constituents/*.csv` — constituent lists for all 6 markets.
- `scripts/download_constituents.py` — constituent download CLI.
- `scripts/download_all_data.py` — OHLCV download CLI.
- `tests/conftest.py` — shared fixtures.
- `tests/unit/test_config.py` — config loading tests.
- `tests/unit/test_constituents.py` — constituent parsing tests.
- `tests/unit/test_downloader.py` — downloader logic tests.
- `tests/integration/test_download_pipeline.py` — end-to-end download test.

## Risks / Open Questions

- **yfinance rate limiting:** If yfinance throttles aggressively, the
  1–2s delay may need increasing. The downloader should handle HTTP
  errors with retry + exponential backoff.
- **Nordic constituent accuracy:** Manually curated CSVs may drift from
  actual index composition. Acceptable for Phase 1 (survivorship bias
  already acknowledged in spec).
- **Wikipedia scraping fragility:** S&P 500 and Nasdaq 100 scrapers
  depend on Wikipedia table format. If scraping fails, fall back to
  existing CSV.

## Learnings

- `types-PyYAML` needed in requirements-dev.txt for mypy to accept `import yaml`.
- `PYTHONPATH=/app` needed in docker-compose.yml for `python scripts/*.py`
  to find the `tsd` package.
- yfinance `auto_adjust=False` returns MultiIndex columns `(Price, Ticker)`
  for single-ticker downloads — need to `droplevel("Ticker", axis=1)`.
- `is_up_to_date` freshness check uses last date in Parquet vs today with
  a 3-day window, which correctly accounts for weekends and holidays.
- PLR2004 (magic numbers) must be suppressed in tests via ruff per-file-ignores.
- PLR0913 (too many args) raised max-args to 7 for data pipeline functions.

## Follow-ups / Backlog (if not done here)

- TaskSpec 002: Data validation and quality checks.
- Refresh strategy for constituent lists (quarterly update process).
