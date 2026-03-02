# Prompt for Claude Code Session — Extract Development Patterns for New Project

*Use this prompt in a Claude Code session inside your existing project repo.*

---

## Context

I'm starting a new Python project — an algorithmic trading system discovery framework using genetic algorithms and Bayesian optimization. The project will run in Docker containers on the same development host as this project.

This existing project has a proven, high-quality structure and workflow that I want to adapt for the new project. I need you to analyze this repo thoroughly and produce artifacts I can use as a foundation.

## What I need you to do

### Step 1: Analyze this project's patterns

Examine the following aspects of this repo. For each, note the specific pattern used and any configuration details:

**Project structure:**
- Directory layout and naming conventions
- How source code, tests, config, scripts, and docs are organized
- What lives at the repo root vs. nested directories

**Docker setup:**
- Dockerfile structure (base image, build stages, layer optimization)
- docker-compose.yml patterns (services, volumes, networks, environment variables)
- How dev vs. production containers differ (if applicable)
- How long-running processes are managed in containers
- How interactive development sessions work (exec into container? mounted volumes?)
- Any Makefile or shell scripts that wrap Docker commands

**Python conventions:**
- Python version and how it's pinned
- Type hints — how consistently used, any specific patterns (Optional vs. Union, etc.)
- Docstring format (Google, NumPy, Sphinx, or custom)
- Import ordering conventions
- Naming: modules, classes, functions, variables, constants
- Use of dataclasses, pydantic, TypedDict, or plain dicts for structured data
- How enums are used (if at all)
- Abstract base classes / interface patterns

**Dependency management:**
- requirements.txt vs. pyproject.toml vs. setup.py
- How dependencies are pinned (exact versions? ranges?)
- Dev vs. production dependencies separation

**Configuration management:**
- YAML, TOML, JSON, or .env files?
- How configs are loaded and validated
- How environment-specific overrides work
- Secrets management approach

**Testing:**
- Test framework (pytest, unittest, other)
- Test directory structure and naming
- Fixture patterns
- Mocking approach
- Coverage targets or enforcement
- Integration vs. unit test separation
- How tests run in Docker

**Logging:**
- Which library (stdlib logging, loguru, structlog, other)
- Log format and structure
- Log level conventions
- How logs are configured (code vs. config file)

**Error handling:**
- Custom exception classes or hierarchy
- How errors propagate
- Retry patterns (if any)

**Code quality tooling:**
- Linter (ruff, flake8, pylint)
- Formatter (black, ruff format)
- Type checker (mypy, pyright)
- Pre-commit hooks
- Any CI/CD configuration

**Git workflow:**
- Branch naming conventions
- Commit message format
- .gitignore patterns
- Any git hooks

**Scripts and automation:**
- Makefile targets
- Shell scripts
- Common commands for dev workflow

**Documentation:**
- README structure
- Inline documentation standards
- Any architecture decision records

### Step 2: Produce the following artifacts

Based on your analysis, create these files tailored for the new project:

#### 1. `CLAUDE.md`

This file will live in the root of the new repo and will be read automatically by Claude Code at the start of every session. It should contain:

- **Project overview** (1 paragraph): "This is a trading system discovery framework using GA + Bayesian optimization with walk-forward validation. See final_project_specification.md for full details."
- **Development environment**: Docker-based, Python 3.11, all commands run inside containers
- **Code conventions**: Every pattern you found in Step 1, stated as clear directives (e.g., "Use Google-style docstrings", "All functions must have type hints", "Use dataclasses for structured data")
- **Testing conventions**: Framework, patterns, how to run tests
- **Logging conventions**: Library, format, usage patterns
- **Git conventions**: Branch naming, commit format
- **Common commands**: Docker build, run, test, lint — whatever the equivalent workflow is
- **Architecture principles**: Any overarching patterns (separation of concerns, dependency injection, etc.)
- **Things to avoid**: Anti-patterns or mistakes you've seen in this project that should not be repeated

Be specific and prescriptive. "Use type hints" is too vague. "All function signatures must have type hints for all parameters and return values. Use `X | None` syntax over `Optional[X]`" is what I want.

#### 2. `Dockerfile`

A Dockerfile adapted for the new project:
- Based on patterns from this project
- Python 3.11 base
- Optimized layer caching for requirements.txt changes
- Working directory `/app`
- Appropriate for both interactive development and long-running optimization jobs

#### 3. `docker-compose.yml`

A docker-compose file adapted for the new project:
- Service for the main application
- Volume mounts for: source code (live reload during dev), data directory (persistent), results directory (persistent), config directory
- Environment variables for any runtime configuration
- Resource limits if this project uses them
- Based on patterns from this project

#### 4. `.gitignore`

A .gitignore based on this project's patterns, extended for the new project's needs:
- Python artifacts (__pycache__, .pyc, etc.)
- Docker artifacts
- Data files (Parquet, CSV downloads)
- Results that shouldn't be committed (checkpoints, logs, performance reports)
- IDE files
- Environment files with secrets

#### 5. `Makefile` (if this project uses one)

Common development commands:
- `make build` — build Docker image
- `make run` — start container
- `make test` — run tests inside container
- `make lint` — run linters
- `make shell` — interactive shell in container
- Any other patterns from this project

#### 6. Quality tooling config files

Whatever this project uses — `pyproject.toml`, `ruff.toml`, `.pre-commit-config.yaml`, `mypy.ini`, `pytest.ini`, etc. — adapted for the new project.

### Step 3: Summary of adaptations

After producing the artifacts, write a brief summary of:
- What you carried over directly from this project
- What you modified for the new project's specific needs (Docker-based Python data science vs. whatever this project is)
- Any patterns from this project that don't apply and were intentionally omitted
- Any gaps — things the new project needs that this project doesn't have patterns for

## Important notes

- The new project name is `trading-system-discovery`
- The new project uses: DEAP, Optuna, vectorbt, pandas, numpy, pandas_ta, yfinance, pyarrow, pyyaml
- The project has long-running CPU-intensive optimization jobs (hours to days)
- There are 16 CPU cores and ~25 GB RAM available
- Quality and maintainability matter — this is a serious project, not a prototype
- Favor explicit over implicit in all conventions
