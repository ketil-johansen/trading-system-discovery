# TaskSpec: 004 - Strategy genome and signal generation

Status: implemented

## Objective

Implement the strategy genome encoding (fixed-length chromosome with
parameter genes and binary switches), signal generation from genome,
exit type computation (all 3 categories from §3), and execution timing
helpers — providing the complete signal layer that the backtest
evaluator (005) will consume.

## Context / References

- `docs/final_project_specification.md` — §3 (Execution Timing Rules),
  §5 (Optimization Engine, genome encoding), §9 (Indicator Universe,
  entry/exit/filter design), §9.7/§9.8 (exit strategies, search space).
- `tsd/indicators/base.py` — `IndicatorResult`, `IndicatorMeta`,
  `ParamMeta`, `compute_indicator()`, `get_indicator_names()`,
  `load_indicator_config()`.
- `config/indicators.yaml` — Indicator parameter metadata (pattern for
  `config/strategy.yaml`).
- `tsd/strategy/` — 6 placeholder files (genome.py, signals.py,
  exits.py, execution.py, evaluator.py, \_\_init\_\_.py).
- `CLAUDE.md` — Frozen dataclasses, env-var config, YAML for strategy
  definitions and search spaces only.

## Scope

**In scope:**

### Indicator output type metadata (`config/indicators.yaml` update)

Add an `output_type` field to each indicator's outputs classifying them
as `oscillator` (bounded numeric range), `price_level` (tracks price),
`binary` (0 or 1), or `direction` (-1, 0, or 1). Also add
`threshold_min` and `threshold_max` for `oscillator` outputs (e.g.
RSI: 0–100). This metadata drives automatic comparison mode selection
in signal generation.

### Strategy config (`config/strategy.yaml`)

YAML file defining the genome structure and exit parameter search
spaces, following the same `{ type, min, max, default }` pattern as
`config/indicators.yaml`. Contains:

- Number of entry indicator slots (default: 4).
- Number of indicator exit slots (default: 1).
- Number of filter slots (default: 2).
- Available comparison operators (GT, LT, CROSS_ABOVE, CROSS_BELOW).
- Combination logic options (AND, OR).
- Exit parameter ranges for all Category 1 exit types (stop-loss,
  take-profit, trailing stop, chandelier, breakeven) — percentages and
  ATR-multiples with min/max/default per the spec.
- Category 3 time-exit parameter ranges (max holding days, stagnation
  days and threshold).

### Genome encoding (`tsd/strategy/genome.py`)

Frozen-dataclass genome representing a complete trading strategy:

- **`IndicatorGene`** — one entry indicator slot: `enabled` (bool),
  `indicator_name` (str), `output_key` (str), `comparison` (str),
  `threshold` (float), `params` (dict of indicator parameters).
- **`LimitExitGene`** — Category 1 exits: each exit type has an
  `enabled` switch plus its parameters (type selection percent/ATR,
  value). Covers stop-loss, take-profit, trailing stop, chandelier,
  breakeven.
- **`IndicatorExitGene`** — Category 2 exits: same shape as
  `IndicatorGene` (enabled, indicator, output_key, comparison,
  threshold, params). Also includes an `opposite_entry` switch (exit
  when entry condition reverses).
- **`TimeExitGene`** — Category 3 exits: `max_days_enabled` +
  `max_days`, `weekday_exit_enabled` + `weekday`, `eow_enabled`,
  `eom_enabled`, `stagnation_enabled` + `stagnation_days` +
  `stagnation_threshold`.
- **`FilterGene`** — one regime-filter slot: `enabled` (bool),
  `filter_name` (str), `params` (dict).
- **`StrategyGenome`** — top-level frozen dataclass: `entry_indicators`
  (tuple of `IndicatorGene`), `combination_logic` (str),
  `limit_exits` (`LimitExitGene`), `indicator_exits` (tuple of
  `IndicatorExitGene`), `time_exits` (`TimeExitGene`), `filters`
  (tuple of `FilterGene`).

Metadata and conversion functions:

- **`StrategyMeta`** — frozen dataclass describing the parameter space:
  number of slots, indicator names, output keys per indicator,
  comparison operators, exit parameter ranges. Loaded from
  `config/strategy.yaml` + `config/indicators.yaml`.
- `load_strategy_config(config_dir: Path) -> StrategyMeta` — parse
  YAML into structured metadata.
- `random_genome(meta: StrategyMeta) -> StrategyGenome` — generate a
  random valid genome (uniform random within parameter ranges,
  50/50 switch probability).
- `validate_genome(genome: StrategyGenome, meta: StrategyMeta) -> bool`
  — check structural validity (indicator names exist, params within
  ranges, at least one entry slot enabled, at least one exit type
  enabled).
- `genome_to_flat(genome: StrategyGenome, meta: StrategyMeta) -> list[float]`
  — serialize to flat chromosome for DEAP. Binary switches → 0.0/1.0,
  indicator name → index, comparison → index, params → raw values.
- `flat_to_genome(values: list[float], meta: StrategyMeta) -> StrategyGenome`
  — deserialize from flat chromosome. Clamp and round as needed.
- `genome_length(meta: StrategyMeta) -> int` — total chromosome length.

### Signal generation (`tsd/strategy/signals.py`)

Functions that convert a genome + OHLCV DataFrame into entry signals:

- `generate_entry_signals(genome: StrategyGenome, df: pd.DataFrame) -> pd.Series`
  — for each enabled entry indicator slot: compute indicator via
  `compute_indicator()`, extract the specified output_key, determine
  comparison mode from the output's `output_type` metadata
  (`oscillator` → compare against threshold, `price_level` → compare
  Close against indicator line, `binary`/`direction` → compare against
  threshold as a fixed value), apply the comparison condition,
  producing a boolean Series. Combine all active slot signals using
  AND/OR per `combination_logic`. Return a boolean Series aligned with
  the DataFrame index.
- `apply_condition(series: pd.Series, comparison: str, threshold: float) -> pd.Series`
  — apply a single comparison (GT, LT, CROSS_ABOVE, CROSS_BELOW) and
  return a boolean Series. CROSS_ABOVE/CROSS_BELOW detect transitions
  using `.shift(1)`.

### Exit computation (`tsd/strategy/exits.py`)

Functions that compute exit levels and signals for all 3 categories.
These produce the raw exit information; the evaluator (005) walks
through bars and applies them.

**Category 1 — Limit-based exits (return price levels):**
- `compute_stop_loss_level(entry_price: float, config: LimitExitGene, atr_at_entry: float) -> float`
  — fixed stop level (percent or ATR-based).
- `compute_take_profit_level(entry_price: float, config: LimitExitGene, atr_at_entry: float) -> float`
  — fixed target level (percent or ATR-based).
- `compute_trailing_stop_levels(entry_price: float, config: LimitExitGene, highs: pd.Series, atr: pd.Series) -> pd.Series`
  — trailing stop level per bar (ratchets up, never down).
- `compute_chandelier_levels(config: LimitExitGene, highs: pd.Series, atr: pd.Series, entry_bar: int) -> pd.Series`
  — N × ATR from highest high since entry.
- `compute_breakeven_level(entry_price: float, config: LimitExitGene, highs: pd.Series, atr_at_entry: float) -> pd.Series`
  — stop moves to entry price once trigger profit reached.

**Category 2 — Indicator-based exits (return boolean signals):**
- `generate_indicator_exit_signals(exit_genes: tuple[IndicatorExitGene, ...], df: pd.DataFrame) -> pd.Series`
  — compute indicator exits, OR them together. If `opposite_entry`
  is enabled, the caller (evaluator) handles reversal logic.

**Category 3 — Time/calendar-based exits (return boolean signals):**
- `generate_time_exit_signal(config: TimeExitGene, entry_bar: int, df: pd.DataFrame) -> pd.Series`
  — boolean Series marking which bars trigger a time-based exit.
  Checks max holding days, weekday, end-of-week, end-of-month, and
  stagnation (one-time check on day N: exit if price moved < ±X%
  from entry).

### Execution timing helpers (`tsd/strategy/execution.py`)

Building-block functions the evaluator (005) will call:

- `shift_to_next_open(signals: pd.Series) -> pd.Series` — shift a
  signal Series forward by 1 bar (signal on close T → execute at
  open T+1). Last bar's signal is dropped.
- `check_limit_exit(high: float, low: float, open_price: float, stop_level: float | None, target_level: float | None) -> tuple[str | None, float | None]`
  — check whether stop and/or target were hit intraday based on
  high/low. Returns `(exit_type, exit_price)` or `(None, None)`.
  Conservative rule: if both hit same day, stop wins unless open
  already exceeds target.
- Constants: `ENTRY_TIMING = "next_open"`,
  `EXIT_TIMING_LIMIT = "intraday"`,
  `EXIT_TIMING_INDICATOR = "next_open"`,
  `EXIT_TIMING_TIME = "at_open"`.

### Package re-exports (`tsd/strategy/__init__.py`)

Re-export key public types and functions.

**Out of scope:**

- Backtest evaluator / trade simulation (`tsd/strategy/evaluator.py` —
  TaskSpec 005).
- Fitness function and metrics (TaskSpec 005).
- Walk-forward validation (TaskSpec 009).
- GA engine and Optuna integration (TaskSpecs 006–008).
- Portfolio simulation (Layer 2).
- Remaining ~35 indicators (TaskSpec 005b) — the genome can reference
  any indicator registered in the indicator library; when 005b adds
  more indicators, they become available automatically.

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies — `pandas`, `numpy`, `pyyaml` already
  available.
- Follow CLAUDE.md conventions: `from __future__ import annotations`,
  type hints, Google-style docstrings, frozen dataclasses, stdlib
  logging.
- Strategy parameter ranges defined in `config/strategy.yaml`, not
  hardcoded in Python. Indicator parameter ranges come from the
  existing `config/indicators.yaml`.
- All genome components are frozen dataclasses. Genome is immutable
  once created.
- Signal generation functions are pure: genome + data → signals.
  No side effects, no global state.
- Exit computation functions take explicit inputs and return explicit
  outputs. No position tracking (that's the evaluator's job).
- The indicator exit uses `compute_indicator()` from the indicator
  library — no duplicate indicator computation logic.

## Acceptance Criteria (must be testable)

- [x] `config/indicators.yaml` updated with `output_type` and
  `threshold_min`/`threshold_max` for each indicator output.
- [x] `config/strategy.yaml` exists with entry slot count, exit
  parameter ranges, comparison operators, and time-exit ranges.
- [x] `load_strategy_config()` parses `config/strategy.yaml` +
  `config/indicators.yaml` and returns a `StrategyMeta`.
- [x] `StrategyGenome` and all sub-gene dataclasses are frozen.
- [x] `random_genome()` produces a valid genome within configured
  parameter ranges.
- [x] `validate_genome()` rejects invalid genomes (unknown indicator,
  params out of range, no entry slots enabled, no exit types enabled).
- [x] `genome_to_flat()` and `flat_to_genome()` round-trip correctly:
  `flat_to_genome(genome_to_flat(g, m), m)` reconstructs the genome.
- [x] `genome_length()` returns the correct chromosome length for the
  configured slot counts.
- [x] `generate_entry_signals()` returns a boolean Series aligned with
  the input DataFrame index.
- [x] `apply_condition()` correctly implements GT, LT, CROSS_ABOVE,
  CROSS_BELOW comparisons.
- [x] AND combination requires all enabled entry slots to agree;
  OR combination requires any one.
- [x] `compute_stop_loss_level()` and `compute_take_profit_level()`
  return correct price levels for both percent and ATR modes.
- [x] `compute_trailing_stop_levels()` produces a monotonically
  non-decreasing Series (for long positions).
- [x] `compute_chandelier_levels()` uses highest high since entry.
- [x] `compute_breakeven_level()` moves stop to entry price only after
  trigger profit is reached.
- [x] `generate_indicator_exit_signals()` returns a boolean Series
  using the indicator library.
- [x] `generate_time_exit_signal()` correctly marks bars for max-days,
  weekday, end-of-week, end-of-month, and stagnation exits.
- [x] `shift_to_next_open()` shifts signals forward by 1 bar.
- [x] `check_limit_exit()` returns stop-loss when both stop and target
  are hit on the same bar (conservative assumption), unless open
  exceeds target.
- [x] All unit tests pass: `pytest tests/unit/test_genome.py
  tests/unit/test_signals.py tests/unit/test_exits.py
  tests/unit/test_execution.py -v`
- [x] `ruff check tsd/ scripts/ tests/` passes with no errors.
- [x] `ruff format --check tsd/ scripts/ tests/` passes.
- [x] `mypy tsd/` passes with no errors.

## Verification

```bash
# Unit tests
docker compose run --rm app pytest tests/unit/test_genome.py -v
docker compose run --rm app pytest tests/unit/test_signals.py -v
docker compose run --rm app pytest tests/unit/test_exits.py -v
docker compose run --rm app pytest tests/unit/test_execution.py -v

# Full test suite (ensure no regressions)
docker compose run --rm app pytest -v

# Lint and type-check
docker compose run --rm app ruff check tsd/ scripts/ tests/
docker compose run --rm app ruff format --check tsd/ scripts/ tests/
docker compose run --rm app mypy tsd/

# Smoke test: create a random genome and generate signals
docker compose run --rm app python -c "
from pathlib import Path
from tsd.strategy.genome import load_strategy_config, random_genome, validate_genome, genome_to_flat, flat_to_genome, genome_length

meta = load_strategy_config(Path('config'))
genome = random_genome(meta)
print(f'Genome: {genome.combination_logic}, {len(genome.entry_indicators)} entry slots')
print(f'Valid: {validate_genome(genome, meta)}')
flat = genome_to_flat(genome, meta)
print(f'Chromosome length: {len(flat)} (expected: {genome_length(meta)})')
reconstructed = flat_to_genome(flat, meta)
print(f'Round-trip OK: {reconstructed == genome}')
"
```

## Deliverables

- `config/indicators.yaml` — Updated with `output_type` and threshold
  range metadata per indicator output.
- `config/strategy.yaml` — Strategy genome structure and exit parameter
  ranges.
- `tsd/strategy/genome.py` — `StrategyGenome`, sub-gene dataclasses,
  `StrategyMeta`, `load_strategy_config()`, `random_genome()`,
  `validate_genome()`, `genome_to_flat()`, `flat_to_genome()`,
  `genome_length()`.
- `tsd/strategy/signals.py` — `generate_entry_signals()`,
  `apply_condition()`.
- `tsd/strategy/exits.py` — Limit exit level functions, indicator exit
  signals, time exit signals.
- `tsd/strategy/execution.py` — `shift_to_next_open()`,
  `check_limit_exit()`, timing constants.
- `tsd/strategy/__init__.py` — Re-exports of key public types and
  functions.
- `tests/unit/test_genome.py` — Genome creation, validation,
  round-trip serialization, random generation.
- `tests/unit/test_signals.py` — Signal generation, condition
  application, AND/OR combination.
- `tests/unit/test_exits.py` — All exit type computations.
- `tests/unit/test_execution.py` — Timing shift, limit exit check,
  conservative same-day resolution.
- `docs/backlog.md` — Item 004 checked off.

## Design Decisions

- **Indicator output type classification:** Each indicator output is
  classified by `output_type` — one of `oscillator` (bounded range,
  e.g. RSI 0–100), `price_level` (tracks price, e.g. SMA),
  `binary` (0/1, e.g. price_vs_ma filter), or `direction` (-1/0/1,
  e.g. supertrend direction). This classification is added to
  `config/indicators.yaml` per output key.
- **Comparison mode determined by output type:** No extra gene needed.
  `oscillator` outputs are compared against a numeric threshold
  (e.g. `rsi < 30`); `price_level` outputs are compared against
  Close (e.g. `close > sma`); `binary` and `direction` outputs are
  compared against fixed values (e.g. `filter == 1`,
  `direction == 1`). The `threshold` gene in `IndicatorGene` holds
  the comparison value in all cases — for `price_level` it is unused
  (comparison is always against Close).
- **Cross-output comparisons deferred:** Each slot uses one
  `output_key` compared against threshold or Close. Cross-output
  comparisons (e.g. Ichimoku `conversion > base`) are deferred to a
  follow-up.

## Risks

- **Genome size with 4 entry slots:** With 4 entry slots × (1 switch
  + 1 indicator index + 1 output index + 1 comparison + 1 threshold
  + ~4 params) ≈ 32 entry genes, plus exit genes and filter genes,
  the chromosome could reach ~80–100 genes. This is manageable for
  DEAP but worth monitoring for GA convergence.

## Learnings

*Populated during implementation.*

## Follow-ups / Backlog (if not done here)

- **005 — Backtest evaluator and fitness function:** Consumes the
  genome, signal, and exit types defined here to run trade simulations
  and compute fitness metrics.
- **Cross-output comparisons:** Supporting "conversion > base" for
  multi-output indicators like Ichimoku. Would require a second
  `output_key` gene per slot and a new comparison mode.
