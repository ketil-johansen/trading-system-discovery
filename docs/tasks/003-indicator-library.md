# TaskSpec: 003 - Indicator library (core subset)

Status: implemented

## Objective

Implement the indicator base interface, parameter metadata loading from
`config/indicators.yaml`, and a core subset of ~15–20 indicators
spanning all 6 categories — enough to prove the architecture and
unblock downstream genome/signals/evaluator work.

## Context / References

- `docs/final_project_specification.md` — Section 9 (Indicator
  Universe): lists ~55 indicators with tunable parameter ranges across
  6 categories.
- `CLAUDE.md` — Directory layout shows `config/indicators.yaml` for
  indicator parameter ranges and `tsd/indicators/` with modules per
  category.
- `requirements.txt` — `ta>=0.11.0` is already a dependency. The `ta`
  library provides most indicator computations out of the box.
- `tsd/indicators/` — All 7 files are placeholders (docstring only).

## Scope

**In scope:**

### Base interface (`tsd/indicators/base.py`)
- `IndicatorResult(frozen=True)` dataclass — holds the computed
  indicator output: name, values (dict of column name → pd.Series),
  and the parameters used.
- `compute_indicator(name: str, df: pd.DataFrame, params: dict)
  -> IndicatorResult` — dispatcher that looks up the indicator by name
  and calls the appropriate computation function.
- `get_indicator_names() -> list[str]` — returns all registered
  indicator names.

### Parameter metadata (`config/indicators.yaml`)
- YAML file defining every indicator's tunable parameters with name,
  type (int/float), min, max, and default values.
- `load_indicator_config(config_dir: Path) -> dict` function in
  `tsd/indicators/base.py` to parse the YAML into a structured dict
  keyed by indicator name.
- The GA and Optuna will use this metadata to define search spaces.
  The indicator functions themselves use it only for documentation;
  they accept parameter values directly.

### Core indicators (one function per indicator, grouped by module)

**Trend (`tsd/indicators/trend.py`)** — 5 indicators:
- `sma` — Simple Moving Average (period)
- `ema` — Exponential Moving Average (period)
- `hma` — Hull Moving Average (period)
- `ichimoku` — Ichimoku Cloud (tenkan, kijun, senkou_b)
- `supertrend` — Supertrend (period, multiplier)

**Momentum (`tsd/indicators/momentum.py`)** — 4 indicators:
- `rsi` — Relative Strength Index (period)
- `stochastic` — Stochastic Oscillator (k_period, d_period, smooth_k)
- `macd` — MACD (fast, slow, signal)
- `williams_r` — Williams %R (period)

**Volatility (`tsd/indicators/volatility.py`)** — 3 indicators:
- `atr` — Average True Range (period)
- `bollinger` — Bollinger Bands (period, std_dev)
- `keltner` — Keltner Channels (ema_period, atr_period, multiplier)

**Volume (`tsd/indicators/volume.py`)** — 2 indicators:
- `obv` — On Balance Volume (no tunable params)
- `cmf` — Chaikin Money Flow (period)

**Filters (`tsd/indicators/filters.py`)** — 2 filters:
- `price_vs_ma` — Price vs. 200-day MA (ma_period)
- `volatility_regime` — ATR-based volatility regime
  (atr_period, lookback, low_threshold, high_threshold)

**Total: 16 indicators** covering all 6 spec categories (price action
indicators like Donchian breakout are deferred to 005b since they
overlap with exit logic).

### Each indicator function must:
- Accept a DataFrame with OHLCV columns and a DatetimeIndex.
- Accept explicit keyword parameters (not **kwargs).
- Return an `IndicatorResult` with named output Series.
- Use the `ta` library for computation where available.
- Handle edge cases (insufficient data → NaN-filled Series).

**Out of scope:**
- Signal generation (TaskSpec 004 — `tsd/strategy/signals.py`).
- Genome encoding (TaskSpec 004 — `tsd/strategy/genome.py`).
- Exit indicators (TaskSpec 004 — covered by exit types).
- Remaining ~35 indicators (TaskSpec 005b).
- Backtest evaluation (TaskSpec 005).

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies — `ta`, `pandas`, `numpy` already
  available.
- Follow CLAUDE.md conventions: `from __future__ import annotations`,
  type hints, Google-style docstrings, frozen dataclasses, stdlib
  logging.
- Indicator parameter ranges defined in `config/indicators.yaml`, not
  hardcoded in Python.
- Each indicator function is pure: takes data + params, returns result.
  No side effects, no global state.
- Use `ta` library for computation where possible. Only implement from
  scratch if `ta` doesn't provide the indicator.

## Acceptance Criteria (must be testable)

- [x] `config/indicators.yaml` exists with parameter metadata for all
  16 core indicators.
- [x] `load_indicator_config()` parses YAML and returns structured dict.
- [x] `compute_indicator()` dispatches to correct function by name.
- [x] `get_indicator_names()` returns all 16 registered names.
- [x] Each of the 16 indicator functions returns an `IndicatorResult`
  with correctly named output Series.
- [x] Indicator outputs have same index as input DataFrame.
- [x] Indicators handle short DataFrames gracefully (NaN-filled, no
  crash).
- [x] `IndicatorResult` is a frozen dataclass.
- [x] All unit tests pass: `pytest tests/unit/test_indicators.py -v`
- [x] `ruff check tsd/ scripts/ tests/` passes with no errors.
- [x] `ruff format --check tsd/ scripts/ tests/` passes.
- [x] `mypy tsd/` passes with no errors.

## Verification

```bash
# Unit tests
docker compose run --rm app pytest tests/unit/test_indicators.py -v

# Full test suite (ensure no regressions)
docker compose run --rm app pytest -v

# Lint and type-check
docker compose run --rm app ruff check tsd/ scripts/ tests/
docker compose run --rm app ruff format --check tsd/ scripts/ tests/
docker compose run --rm app mypy tsd/

# Smoke test: compute indicators on real OMXS30 data
docker compose run --rm app python -c "
import pandas as pd
from tsd.indicators.base import compute_indicator, get_indicator_names

df = pd.read_parquet('data/raw/omxs30/VOLV-B.parquet')
print(f'Registered indicators: {get_indicator_names()}')
for name in ['sma', 'rsi', 'atr', 'obv']:
    result = compute_indicator(name, df, {})
    print(f'{name}: {list(result.values.keys())}, len={len(list(result.values.values())[0])}')
"
```

## Deliverables

- `config/indicators.yaml` — Parameter metadata for 16 core indicators.
- `tsd/indicators/base.py` — `IndicatorResult` dataclass,
  `compute_indicator()` dispatcher, `load_indicator_config()`,
  `get_indicator_names()`.
- `tsd/indicators/trend.py` — SMA, EMA, HMA, Ichimoku, Supertrend.
- `tsd/indicators/momentum.py` — RSI, Stochastic, MACD, Williams %R.
- `tsd/indicators/volatility.py` — ATR, Bollinger Bands, Keltner
  Channels.
- `tsd/indicators/volume.py` — OBV, Chaikin Money Flow.
- `tsd/indicators/filters.py` — Price vs. MA, Volatility Regime.
- `tests/unit/test_indicators.py` — Unit tests for all indicators and
  the base interface.
- `docs/backlog.md` — Updated with reordered tasks.

## Risks / Open Questions

- **HMA and Supertrend not in `ta` library:** Hull Moving Average and
  Supertrend are not provided by `ta`. HMA can be computed from WMA
  components. Supertrend requires ATR + custom logic. Both are
  straightforward to implement with pandas/numpy.
- **`ta` library API stability:** The `ta` library uses class-based
  indicators with method accessors (e.g.,
  `RSIIndicator(close, window).rsi()`). Our wrapper functions
  abstract this, so if `ta` changes its API, only our wrappers need
  updating.
- **NaN handling at series start:** Most indicators produce NaN for the
  first N rows (warmup period). This is expected and must be handled
  downstream by the evaluator. Indicators should NOT drop or fill NaN
  rows.

## Learnings

- The `ta` library's `AverageTrueRange` crashes with `IndexError` when
  the DataFrame has fewer rows than the window period. All indicators
  using ATR internally (atr, keltner, supertrend, volatility_regime)
  need a short-data guard that returns NaN-filled Series.
- HMA implemented manually using pandas rolling WMA (weighted sum via
  `np.arange`). Supertrend implemented with ATR bands + direction flip
  loop, extracted into `_supertrend_step()` to satisfy ruff's branch
  limit (PLR0912).
- Lazy imports in `_get_registry()` require `# noqa: PLC0415` since
  they are intentionally deferred to avoid circular dependencies.
- `Callable` import must come from `collections.abc` (UP035), not
  `typing`.

## Follow-ups / Backlog (if not done here)

- **005b — Remaining indicators:** DEMA, TEMA, KAMA, ADX, Aroon,
  Parabolic SAR, Vortex, Linear Regression Slope, Connors RSI,
  Stochastic RSI, CCI, ROC, MFI, TSI, Ultimate Oscillator, Donchian
  Channels, Historical Volatility, Chaikin Volatility, Force Index,
  Volume SMA Ratio, Accumulation/Distribution, Pivot Points, Inside Bar,
  Gap Detection, Higher Highs/Lows, Day-of-Week filter, Month-of-Year
  filter, Distance from 52-week high, ADX trend strength filter.
