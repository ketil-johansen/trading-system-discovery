# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project

Trading System Discovery is a framework for discovering profitable
algorithmic trading systems using genetic algorithms (DEAP) and Bayesian
optimization (Optuna). It evolves trading strategy parameters,
validates them with walk-forward analysis using vectorbt, and persists
results as Parquet files. See `docs/final_project_specification.md` for full details.

## Common Commands

### All commands run inside Docker — do not run Python on the host

```bash
# Build the container
docker compose build

# Run optimization job
docker compose run --rm app python -m tsd.main

# Interactive shell inside container
docker compose run --rm app bash

# Run all tests
docker compose run --rm app pytest

# Run tests with coverage
docker compose run --rm app pytest --cov=tsd --cov-report=term-missing

# Lint and type-check
docker compose run --rm app ruff check .
docker compose run --rm app ruff format --check .
docker compose run --rm app mypy tsd/
```

## Architecture

### Directory Layout

```
trading-system-discovery/
├── tsd/                     # Main package (source code)
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration (frozen dataclass + env helpers)
│   ├── data/                # Market data fetching, constituents, quality checks
│   │   ├── downloader.py    # Market data acquisition (yfinance)
│   │   ├── constituents.py  # Index constituent scrapers
│   │   └── quality.py       # Data validation & gap detection
│   ├── indicators/          # Technical indicator library
│   │   ├── base.py          # Indicator interface
│   │   ├── trend.py         # SMA, EMA, HMA, Ichimoku, etc.
│   │   ├── momentum.py      # RSI, Stochastic, MACD, etc.
│   │   ├── volatility.py    # ATR, Bollinger, Keltner, etc.
│   │   ├── volume.py        # OBV, CMF, Force Index, etc.
│   │   └── filters.py       # Regime filters, seasonality
│   ├── strategy/            # Strategy encoding and evaluation
│   │   ├── genome.py        # Strategy DNA encoding
│   │   ├── signals.py       # Signal generation from genome
│   │   ├── exits.py         # All exit types
│   │   ├── execution.py     # Execution timing rules
│   │   └── evaluator.py     # Backtest engine + metrics
│   ├── optimization/        # GA (DEAP) and Bayesian (Optuna) engines
│   │   ├── ga.py            # Genetic algorithm (DEAP)
│   │   ├── bayesian.py      # Bayesian optimization (Optuna)
│   │   ├── fitness.py       # Win-rate fitness with profitability gate
│   │   ├── walkforward.py   # Anchored walk-forward engine
│   │   └── pipeline.py      # Staged GA → Optuna pipeline
│   ├── analysis/            # Results analysis and robustness checks
│   │   ├── reports.py       # Strategy performance reports
│   │   ├── cross_market.py  # Cross-market validation (future)
│   │   └── robustness.py    # Monte Carlo, complexity checks
│   ├── portfolio/           # Portfolio simulation (Layer 2, future)
│   │   ├── simulator.py     # Portfolio equity curve simulation
│   │   ├── sizing.py        # Position sizing models
│   │   └── risk.py          # Portfolio-level risk controls
│   └── export/              # Output generation
│       └── pine_script.py   # Pine Script code generator
├── tests/                   # All tests (pytest)
│   ├── conftest.py          # Shared fixtures
│   ├── unit/                # Unit tests (fast, no I/O)
│   └── integration/         # Integration tests (with data, slower)
├── scripts/                 # Runner scripts
│   ├── download_constituents.py
│   ├── download_all_data.py
│   ├── check_data_quality.py
│   ├── run_optimization.py
│   └── generate_pine_scripts.py
├── config/                  # YAML configuration files
│   ├── markets.yaml         # Market definitions, tickers, suffixes
│   ├── indicators.yaml      # Indicator parameter ranges
│   ├── optimization.yaml    # GA + Optuna settings
│   ├── fitness.yaml         # Win rate target, cost model
│   └── walkforward.yaml     # WF windows, passing criteria
├── data/                    # Market data cache — gitignored
│   ├── raw/                 # Downloaded OHLCV parquet files
│   ├── constituents/        # Index constituent lists (CSV)
│   └── reports/             # Data quality reports
├── results/                 # Optimization outputs — gitignored
│   ├── strategies/          # Validated strategy configs (JSON)
│   ├── pine_scripts/        # Generated Pine Script files
│   ├── performance/         # Full backtest reports
│   ├── checkpoints/         # GA/Optuna checkpoints for resume
│   └── logs/                # Structured run logs (JSONL)
├── notebooks/               # Analysis notebooks
├── docs/                    # Documentation
│   ├── final_project_specification.md
│   ├── backlog.md           # Ordered list of planned work
│   └── tasks/               # TaskSpecs (one per unit of work)
│       └── TASK_TEMPLATE.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── CLAUDE.md
```

### Key Components

- `tsd/config.py` — All configuration via environment variables with
  frozen dataclass defaults. Uses `env_str()`, `env_int()`, `env_float()`,
  `env_bool()` helpers.
- `tsd/main.py` — Entry point following `def main() -> int:` /
  `sys.exit(main())` pattern.
- `tsd/data/` — Market data fetching (yfinance), constituent lists,
  quality checks. Caching via Parquet (pyarrow).
- `tsd/indicators/` — Technical indicator library with standardized interface.
- `tsd/strategy/` — Strategy genome encoding, signal generation, exit types,
  backtest evaluation.
- `tsd/optimization/` — DEAP genetic algorithm engine, Optuna Bayesian
  optimizer, walk-forward validation, staged pipeline.
- `tsd/analysis/` — Performance reports, cross-market validation, robustness
  checks (Monte Carlo, bootstrap).
- `tsd/portfolio/` — Portfolio simulation (Layer 2, built after core pipeline).
- `tsd/export/` — Pine Script code generator for TradingView.

## Project Organization

### Task-driven workflow

All work is organized as TaskSpecs in `docs/tasks/`, following the
template in `docs/tasks/TASK_TEMPLATE.md`. The backlog of planned work
lives in `docs/backlog.md`.

**TaskSpec lifecycle:** `draft` → `ready` → `in-progress` →
`implemented`

One TaskSpec at a time. One PR per TaskSpec. TaskSpec status updates and
acceptance criteria check-offs are part of the feature branch PR, never
committed separately to `main`.

### Workflow steps

When executing a TaskSpec, follow these steps:

| Step | What happens |
|------|-------------|
| **0 — Define** | Explore relevant code. Draft or refine the TaskSpec (objective, scope, acceptance criteria). Human reviews and approves. |
| **1 — Plan** | Enter plan mode. Investigate implementation approach. Present plan for human approval. |
| **2 — Implement** | Create feature branch (`feature/<id>-<short-title>`). Write code, commit incrementally. |
| **3 — Verify** | Run tests, lint, type-check. Ensure all acceptance criteria are met. |
| **4 — Fix** | If verification fails, fix issues and re-verify. |
| **5 — PR** | Update TaskSpec status to `implemented`, check off acceptance criteria, create PR. |

### Triage (before starting)

Before entering any step, check current state:
1. Does the TaskSpec exist? What is its status?
2. Does a feature branch or open PR already exist?
3. What's the right entry point (define, plan, implement, verify, or PR)?

### Documentation

- `docs/backlog.md` — Ordered list of planned work.
- `docs/tasks/` — TaskSpecs (one per unit of work).
- `docs/` — Any design docs, only when a feature needs architectural decisions across 3+ TaskSpecs.

## Code Conventions

### Python Version

- Python 3.10. Pinned in Dockerfile.

### Type Hints

- All function signatures must have type hints for all parameters and return values.
- Use `X | None` syntax, not `Optional[X]`.
- Use `from __future__ import annotations` at the top of every module.
- Import `Dict`, `List`, `Tuple`, `Any` from `typing` when needed.

### Docstrings

- Google-style docstrings.
- Every module must have a module-level docstring (one sentence describing purpose).
- Every class must have a class-level docstring.
- Public methods should have docstrings. Private methods need docstrings only when the logic is non-obvious.

### Naming

- Modules: `snake_case` (e.g., `walk_forward.py`, `ga_engine.py`)
- Classes: `PascalCase` (e.g., `GeneticOptimizer`, `ValidationResult`)
- Functions/methods: `snake_case` (e.g., `run_backtest`, `fetch_ohlcv`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_POPULATION_SIZE`, `MAX_GENERATIONS`)
- Private methods: prefix with `_` (e.g., `_evaluate_fitness`, `_crossover`)
- Environment variables: `UPPER_SNAKE_CASE` with `TSD_` prefix (e.g., `TSD_POPULATION_SIZE`)

### Imports

Order imports as follows, separated by blank lines:
1. `from __future__ import annotations`
2. Standard library (alphabetical)
3. Third-party packages (alphabetical)
4. Local/relative imports

### Structured Data

- Use `@dataclass(frozen=True)` for value objects, configuration, and results.
- Use plain `dict` for dynamic/flexible data (API responses, intermediate computation).
- Do not use Pydantic, TypedDict, or NamedTuple.

### Configuration

- All configuration via environment variables with sensible defaults.
- Configuration loaded into a frozen dataclass via `load_config()` factory function.
- Helper functions: `env_str()`, `env_int()`, `env_float()`, `env_bool()` in `config.py`.
- No YAML/JSON config files for runtime settings — environment variables only.
- YAML config files in `config/` for strategy definitions and parameter search spaces only.

### Entry Points

All entry points must follow this pattern:

```python
def main() -> int:
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # ... business logic ...
    return 0  # or 1 on failure

if __name__ == "__main__":
    sys.exit(main())
```

### Logging

- Use Python stdlib `logging`. Do not use loguru or structlog.
- Create loggers per module: `LOGGER = logging.getLogger(__name__)`
- Log format: `"%(asctime)s %(levelname)s %(message)s"`
- Log level set via `TSD_LOG_LEVEL` environment variable (default: INFO).
- Use `LOGGER.info()` for progress milestones (generation completed, validation started).
- Use `LOGGER.warning()` for recoverable issues (missing data, retries).
- Use `LOGGER.error()` for non-fatal errors.
- Use `LOGGER.exception()` for fatal crashes (includes traceback).
- Do not use `print()` for status output — use logging.

### Error Handling

- Catch library-specific exceptions first (e.g., `optuna.exceptions.TrialPruned`), then built-in types.
- For transient failures (network, API rate limits): exponential backoff with configurable max.
- For data pipeline errors: log warning, skip bad record, continue processing.
- Top-level `main()` catches `Exception` as a final guardrail, logs via `LOGGER.exception()`, returns exit code 1.
- Do not define custom exception classes unless there is a clear hierarchy need.

### Testing

- Framework: pytest.
- All tests run inside Docker: `docker compose run --rm app pytest`
- Test files: `tests/unit/test_*.py` and `tests/integration/test_*.py`
- Fixtures: shared fixtures in `tests/conftest.py`, module-specific fixtures in test files.
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`
- Unit tests: fast, no I/O, no network, test pure logic.
- Integration tests: may use data files, test end-to-end flows.
- Name test functions descriptively: `test_crossover_preserves_valid_genes`, not `test_crossover_1`.

### Code Quality

- Formatter: `ruff format` (compatible with black)
- Linter: `ruff` (rules: E, F, I, B, UP, PL)
- Type checker: `mypy` (strict mode)
- All configuration in `pyproject.toml`.

### Git Conventions

- Branch naming: `feature/<short-title>`, `fix/<short-title>`
- Commit messages: `<type>: <description>` (e.g., `feat: add walk-forward validation engine`)
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- One PR per feature, branched off main. Never commit directly to main.

### Docker

- All builds, tests, and tooling run in containers. Do not run host-native Python.
- Source code mounted as a volume for live development (`./tsd:/app/tsd`).
- Data directory (`./data`) and results directory (`./results`) are persistent volumes.
- Build context copies `requirements.txt` first for Docker layer caching.

### Things to Avoid

- Do not use `Optional[X]` — use `X | None`.
- Do not use `print()` for logging — use the `logging` module.
- Do not install packages on the host — everything runs in Docker.
- Do not use mutable dataclass defaults (use `field(default_factory=...)` if needed).
- Do not use Pydantic for config — use frozen dataclasses with env helpers.
- Do not create custom exception hierarchies unless strictly necessary.
- Do not use `*args` / `**kwargs` in public APIs — be explicit about parameters.
- Do not use global mutable state — pass configuration and state through function parameters.
