# Trading System Discovery — Final Project Specification

*All technical decisions locked. Ready for implementation in Claude Code.*

---

## 1. Objective

Build a Python framework that uses genetic algorithm and Bayesian optimization with walk-forward validation to discover **extremely high win-rate, profitable, long-only** trading systems on individual stocks across six markets. Winning strategies will be converted to TradingView Pine Script for live execution.

**Long only:** The system exclusively searches for buy (long) entry signals. Short selling is not modeled. All entry signals mean "buy this stock"; all exit signals mean "sell the existing long position."

**Note on indicator direction:** "Long only" means we never open short positions — it does NOT restrict which indicator conditions can trigger a buy entry. The GA is free to evolve any condition as a buy trigger, including conditions that might seem counterintuitive (e.g., "buy when RSI > 70" could be a valid momentum entry). The direction and threshold of each indicator signal are part of the genome and are optimized by the GA. The constraint is on position type (long only), not on signal logic.

---

## 2. Target Markets

| Market | Index | Yahoo Ticker | Constituents | YF Stock Suffix | Data Available |
|--------|-------|-------------|-------------|-----------------|----------------|
| Nasdaq 100 | NDX | `^NDX` | ~100 stocks | (none) | 10+ years |
| S&P 500 | SPX | `^GSPC` | ~500 stocks | (none) | 10+ years |
| Sweden | OMXS30 | `^OMXS30` | 30 stocks | `.ST` | 10+ years |
| Denmark | OMXC25 | `^OMXC25` | 25 stocks | `.CO` | ~9 years |
| Finland | OMXH25 | `^OMXH25` | 25 stocks | `.HE` | 10+ years |
| Norway | OBX | `OBX.OL` | 25 stocks | `.OL` | 10+ years |

**Total universe:** ~705 individual stocks

**STOXX Europe 600:** Excluded for now. May be added later.

**Survivorship bias:** Accepted for Phase 1 using current constituents. Documented as a known limitation.

---

## 3. Execution Timing Rules

These rules are critical and must be hardcoded into the backtesting engine:

### Entry
- **Signal generated:** After market close on day T, based on day T's completed daily bar (all indicators computed on closing data).
- **Position entered:** At the **open of day T+1**. This models the realistic scenario where the trader reviews signals in the evening, decides to enter, and submits a market-on-open (or pre-market) order for the next morning.

### Exit Category 1 — Limit-Based (Stop-Loss, Profit Target)
Exits where a specific price level triggers the exit during trading hours.

- **Executed:** During trading hours on the day the limit is hit.
- **Price used:** The limit price itself.
- **Intraday check:** On each bar, check if the day's High reached the profit target or the day's Low breached the stop-loss. If both are hit on the same day, assume the stop-loss was hit first (conservative assumption) unless the open already exceeds the profit target.
- **Note:** Limit levels are set relative to the entry price at the time of entry and remain active from the moment the position is opened.

**Exit types in this category:**
- Fixed stop-loss (percentage: 0.5%–10%)
- Fixed stop-loss (ATR-multiple: 1.0–4.0)
- Fixed profit target (percentage: 1%–15%)
- Fixed profit target (ATR-multiple: 1.0–6.0)
- Trailing stop (percentage-based: activates after X% profit, trails by Y%)
- Trailing stop (ATR-based: trails by N × ATR)
- Chandelier exit (N × ATR from highest high since entry)
- Breakeven stop (move stop to entry price after reaching X% or N × ATR profit)

**Note on trailing stops and chandelier exits:** These exits have a computed level that changes daily based on price action since entry (e.g., highest high). The level is **recalculated after each close** based on that day's data, and then **monitored intraday the following day** against High/Low. This means they straddle Category 1 and Category 2 behavior: the recalculation is indicator-like (after close), but the trigger is limit-like (intraday at the computed price).

### Exit Category 2 — Indicator-Based
Exits where a technical indicator value triggers the exit. Signal is evaluated after market close; execution happens at next open.

- **Signal generated:** After market close on day T, based on day T's completed daily bar.
- **Position exited:** At the **open of day T+1**.
- **Price used:** The next day's open price.

**Exit types in this category:**
- RSI crossing above/below threshold (e.g., RSI > 70 = exit long)
- MACD signal line cross
- Price crossing below/above a moving average
- Stochastic overbought/oversold cross
- Bollinger Band touch (price reaching upper/lower band)
- Supertrend flip
- Opposite entry signal (the entry indicator reverses)
- Any other indicator threshold or crossover condition

### Exit Category 3 — Time/Calendar-Based
Exits triggered by the passage of time or a calendar condition. Execution happens at the open of the specified day.

- **Trigger:** A temporal condition is met (N days elapsed, specific weekday reached, etc.).
- **Position exited:** At the **open of the day** the condition is met.
- **Price used:** That day's open price.

**Exit types in this category:**
- Exit after N trading days (N: 1–30, configurable). Counted from entry day. Exit at open of day N.
- Exit on next specific weekday (Monday, Tuesday, Wednesday, Thursday, Friday). Exit at open of the next occurrence of that weekday after entry.
- Exit at end of week (at Friday's open, or if entered on Friday, the following Friday's open).
- Exit at end of month (at the open of the last trading day of the month).
- Stagnation exit: On day N after entry, check if position has moved less than ±X% from entry price. If stagnant, exit at open of day N+1. This is a **one-time check** on day N only — not a rolling check. If the exit triggers, it fires regardless of what happens on day N+1 (even if a limit exit would also trigger on that day, the stagnation exit takes precedence since it fires at the open).

### Exit Combination Rules
A strategy can use **multiple exit types simultaneously** from different categories. The first exit condition to trigger wins. For example, a strategy might combine:
- A 2% profit target (Category 1 — fires intraday)
- A 5% stop-loss (Category 1 — fires intraday)
- An RSI > 70 exit (Category 2 — fires at next open)
- A maximum holding period of 10 days (Category 3 — fires at day 10 open)

Whichever condition is met first closes the position. This allows the GA to evolve complex exit logic by enabling/disabling individual exit components via genetic switches.

### Summary Table

| Event | Signal/Trigger Timing | Execution Price |
|-------|----------------------|-----------------|
| Entry | After close, day T | **Open of day T+1** |
| Exit (limit-based) | Intraday, day T | Limit price on day T |
| Exit (indicator-based) | After close, day T | Open of day T+1 |
| Exit (time/calendar) | Calendar condition met, day T | Open of day T |

### Execution Consistency
Entries and indicator-based exits follow the same pattern: signal after close → execute at next open. Limit-based exits fire intraday at their specified price levels. Time/calendar-based exits fire at the open of the day the temporal condition is met.

---

## 4. Data Specification

- **Frequency:** Daily OHLCV (Open, High, Low, Close, Volume)
- **Primary source:** `yfinance`
- **Price adjustment:** Split-adjusted only. Dividend adjustments are NOT applied. This means all OHLCV values reflect stock splits but not dividend payments. This is acceptable because our fitness function operates on percentage returns per trade over short holding periods, where dividend impact is negligible.
- **Minimum history required:** 10 years where available (2015–2025+)
- **Storage format:** Parquet, organized by market
- **Trading universe:** Individual constituent stocks of each index

**Corporate events:** Trading halts, suspensions, acquisitions, and delistings are NOT modeled. If a stock has missing data for a period, trades that would span the gap are simply not generated (the indicator signals won't fire on missing bars). Stocks that were delisted before the present day are not in our constituent lists due to accepted survivorship bias (see Section 2).

**Currency:** All calculations are performed in local currency (USD for US stocks, SEK for Swedish, DKK for Danish, EUR for Finnish, NOK for Norwegian). There is no currency conversion or FX adjustment. Win rate and profitability are measured as percentages, which are currency-agnostic. Currency exposure is a portfolio-level concern for Layer 2 (Phase 7+).

---

## 5. Optimization Engine

### Architecture: Staged, Configurable

The system supports two optimization modes, usable independently or in sequence:

**Stage A — Genetic Algorithm (DEAP)**
- **Purpose:** Structural optimization — which indicator categories to combine, which exit type, which regime filters to enable/disable.
- **Encoding:** Fixed-length chromosome with parameter genes and binary "switch" genes (inspired by Kör & Zengin 2026).
- **Population:** 200–500 individuals (configurable)
- **Generations:** 50–200 with early stopping (configurable)
- **Selection:** Tournament (size 3–5)
- **Crossover:** Uniform at indicator-slot level (Pc configurable, default 0.5)
- **Mutation:** Gaussian for parameters, bit-flip for switches (Pm configurable, default 0.2)
- **Elitism:** Top 5% preserved

**Stage B — Bayesian Optimization (Optuna)**
- **Purpose:** Parameter fine-tuning — given a strategy structure from Stage A, find optimal parameter values.
- **Method:** Tree-structured Parzen Estimator (TPE) — Optuna's default sampler
- **Pruning:** Optuna's MedianPruner to kill bad trials early
- **Trials:** 200–1000 per strategy structure (configurable)

**Configuration:** A YAML parameter controls the mode:
```yaml
optimization:
  mode: "both"  # Options: "ga_only", "bayesian_only", "both"
  ga:
    population_size: 300
    generations: 100
    crossover_prob: 0.5
    mutation_prob: 0.2
    elitism_pct: 0.05
    tournament_size: 3
  bayesian:
    n_trials: 500
    pruner: "median"
    sampler: "tpe"
```

### Logging, Checkpointing, and Reproducibility

Long-running optimization (potentially hours or days across 705 stocks) requires robust infrastructure for monitoring, recovery, and reproducibility.

**Logging:**
- Per-generation summary: best fitness, worst fitness, average fitness, population diversity metric
- Per-individual: genome hash, fitness score, number of trades, win rate, profitability (logged to file, not just console)
- Per-walk-forward-window: in-sample and OOS performance of top N strategies
- Log format: structured (JSON lines or CSV) for easy post-analysis
- Console output: progress bar with estimated time remaining

**Checkpointing:**
- Save full population state (all genomes + fitness scores) after every generation
- Save Optuna study object after every N trials (configurable, default every 50)
- Save walk-forward intermediate results (completed windows) incrementally
- Checkpoint format: pickle or JSON, stored in `results/checkpoints/`

**Restart / Resume:**
- If a run is interrupted (crash, manual stop, or intentional pause), the system must be able to resume from the last checkpoint
- Resume detects the most recent checkpoint and continues from that generation/trial/window
- Command-line flag: `--resume` to resume from checkpoint vs. `--fresh` to start clean

**Reproducibility:**
- Random seed: configurable in YAML (default: 42). Applied to GA initialization, Optuna sampler, and any random selection
- All configuration parameters are saved alongside results so any run can be reproduced
- Git commit hash of the codebase is logged with each run

```yaml
infrastructure:
  logging:
    level: "INFO"  # DEBUG for full detail
    output_dir: "results/logs/"
    format: "jsonl"
  checkpointing:
    enabled: true
    frequency_generations: 1     # save every generation
    frequency_optuna_trials: 50  # save every 50 trials
    output_dir: "results/checkpoints/"
  reproducibility:
    random_seed: 42
    log_git_hash: true
```

---

## 6. Fitness Function

### Design: Maximize Win Rate, Constrained to Profitability

```
Maximize:  win_rate

Subject to:
  - win_rate >= min_win_rate_threshold  (configurable, default 0.80)
  - net_profit > 0  (after transaction costs)
  - number_of_trades >= 30  (in-sample only — see note below)

If any constraint is violated, fitness = 0 (strategy is discarded)
```

**This is not a weighted composite.** Win rate is the sole optimization target. Profitability and the win rate threshold are hard gates — a strategy must be profitable after costs AND exceed the minimum win rate to even be considered. Among qualifying strategies, the one with the highest win rate wins.

**Win rate threshold:** The default is 80%, reflecting the goal of finding extremely high win-rate systems. This is configurable — set it lower (e.g., 65% or 70%) to widen the search if 80% produces too few results, or higher (e.g., 85%, 90%) to tighten it.

**Minimum trade count:** The 30-trade minimum applies **only to in-sample evaluation**. During walk-forward out-of-sample windows, a strategy may generate fewer trades (or even zero) and this is accepted — some strategies are selective by nature and may not trigger in every 6-month period.

**Definition of a win:** A trade is a **win** if its net return after transaction costs is strictly greater than zero. A trade that breaks even or loses money after costs is a **loss**. There is no "neutral" category.

```yaml
fitness:
  min_win_rate_threshold: 0.80  # configurable — lower to 0.65-0.70 for broader search
  min_trades_in_sample: 30
  require_net_profitable: true

### Metrics Tracked (for reporting, not optimization)

Every strategy that passes the profitability gate will have these metrics recorded for analysis:

- **Win rate** (primary, optimized)
- Net profit / loss (after costs)
- Profit factor (gross profit / gross loss)
- Average win size
- Average loss size
- Win/loss ratio (avg win / avg loss)
- Maximum drawdown
- Maximum drawdown duration
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Number of trades
- Average holding period (days)
- Longest winning streak
- Longest losing streak
- Expectancy per trade

These metrics provide the full picture even though optimization targets only win rate. When reviewing results, you'll be able to see the natural trade-off — extremely high win-rate strategies will likely show smaller average wins relative to average losses, and this data will help you make informed decisions.

### Transaction Costs in Fitness

- **Default round-trip cost:** 0.20% (0.10% per side)
- **Applied to:** Every entry and every exit
- **Configurable:** Per-market overrides possible in config
- **Effect:** Net profit calculation deducts costs before the profitability gate is evaluated

```yaml
costs:
  default_round_trip_pct: 0.20
  overrides:  # optional per-market
    nasdaq_100: 0.10   # lower for high-liquidity US stocks
    omxs30: 0.25       # slightly higher for Nordic
```

---

## 7. Walk-Forward Validation

### Design: Anchored, Growing In-Sample

```
Available data: 2015 ────────────────────────────────── 2026

Walk-forward windows (anchored — in-sample grows each step):

Window 1: Train [2015────────2020.0]  Test [2020.0──2020.6]
Window 2: Train [2015────────2020.6]  Test [2020.6──2021.0]
Window 3: Train [2015────────2021.0]  Test [2021.0──2021.6]
Window 4: Train [2015────────2021.6]  Test [2021.6──2022.0]
Window 5: Train [2015────────2022.0]  Test [2022.0──2022.6]
Window 6: Train [2015────────2022.6]  Test [2022.6──2023.0]
Window 7: Train [2015────────2023.0]  Test [2023.0──2023.6]
Window 8: Train [2015────────2023.6]  Test [2023.6──2024.0]
Window 9: Train [2015────────2024.0]  Test [2024.0──2024.6]
Window 10: Train [2015───────2024.6]  Test [2024.6──2025.0]

Final holdout (NEVER touched during any optimization):
             [2025.0 ─────── 2026.0]
```

- **In-sample:** Grows from ~5 years (window 1) to ~9.5 years (window 10)
- **Out-of-sample:** 6 months each (10 windows)
- **Final holdout:** 12 months, untouched until the very last validation step
- **Slide step:** 6 months

### Market Regime Awareness
The walk-forward windows intentionally span different market regimes (COVID crash 2020, bull market 2021, bear market 2022, recovery 2023–2024) without any regime-specific treatment. All OOS windows are weighted equally in the passing criteria. This means a strategy must perform across bull, bear, and sideways markets to pass — there is no concept of a "bull market only" strategy in this framework. This is a deliberate design choice: strategies that only work in favorable conditions are not robust enough for practical use.

### Passing Criteria for a Strategy

A strategy is considered walk-forward validated if:
1. Win rate ≥ `min_win_rate_threshold` in **at least 8 of 10** OOS windows (windows with zero trades are excluded from this count)
2. Net profitable (after costs) in **at least 7 of 10** OOS windows (windows with zero trades are excluded from this count)
3. Final holdout period: profitable with win rate within 10% of average OOS win rate

**Note on low-trade OOS windows:** Some strategies are selective and may generate few or zero trades in a given 6-month window. This is accepted. The passing criteria are evaluated only across windows where at least one trade occurred. However, if a strategy triggers trades in fewer than 5 of the 10 OOS windows, it is flagged as "low frequency" in the results — it may be valid but is too selective for practical use.

These thresholds are configurable.

```yaml
walk_forward:
  in_sample_start: "2015-01-01"
  oos_length_months: 6
  final_holdout_months: 12
  slide_step_months: 6
  anchored: true
  passing_criteria:
    min_win_rate_threshold: 0.80  # matches fitness threshold
    min_oos_windows_win_rate: 8   # out of 10 (excluding zero-trade windows)
    min_oos_windows_profitable: 7  # out of 10 (excluding zero-trade windows)
    holdout_win_rate_tolerance: 0.10  # within 10% of avg OOS
    low_frequency_flag_threshold: 5   # flag if trades in fewer than 5 windows
```

---

## 8. Portfolio Management & Risk Management

### Design Principle: Two Separate Layers

Signal discovery and portfolio management are intentionally separated:

| | Layer 1 — Signal Quality | Layer 2 — Portfolio Management |
|---|---|---|
| **Purpose** | Find the highest win-rate entry/exit combinations | Manage real capital using validated signals |
| **When** | During optimization (GA + Optuna + walk-forward) | After strategy validation, during paper/live trading |
| **Position sizing** | Fixed notional per trade (e.g., $10,000) | Risk-based sizing (e.g., 2% of capital per trade) |
| **Concurrent positions** | Not modeled — each trade evaluated independently | Limited by capital and max position rules |
| **Correlation** | Not modeled | Managed via sector/market exposure limits |
| **Capital curve** | Not tracked (only per-trade metrics) | Full equity curve simulation |

### Layer 1 — During Optimization (Included Now)

Each trade is backtested with a **fixed notional amount** (configurable, default $10,000). This is simply a scaling factor — it converts percentage returns into dollar returns for the profitability gate, and ensures transaction costs are applied correctly.

The optimizer evaluates:
- Win rate (primary objective)
- Net profit per trade after costs (must be > 0 on average for the strategy to pass)
- All other per-trade metrics (avg win, avg loss, etc.)

**What this does NOT model:**
- How many trades can be open simultaneously
- What happens when capital runs out because too many signals fire at once
- Correlation between concurrent positions
- Compounding of returns

**Why this is acceptable:** The optimizer is searching for signal quality. A signal that wins 80% of the time on independent trades will still win ~80% when deployed in a portfolio. The per-trade economics (average win, average loss, costs) are accurately captured. What changes in a portfolio context is how many of those trades you can take simultaneously and your aggregate P&L and drawdown — but that's a capital allocation decision, not a signal quality decision.

### Layer 2 — Post-Validation Portfolio Simulation (Built Later)

Once validated strategies are identified, a separate portfolio simulation module will model realistic capital deployment. This is Phase 7+ work, built after the core optimization pipeline is proven. The design is outlined here so the architecture accommodates it.

#### Capital and Position Sizing

```
Starting capital:           $1,000,000  (configurable)
Max risk per trade:         2% of current equity = $20,000 initial risk
Position size calculation:  risk_amount / stop_distance
Max concurrent positions:   10–20 (configurable)
Max positions per sector:   3–5 (configurable, prevents concentration)
Max positions per market:   configurable
```

**Example position sizing calculation:**
```
Capital:         $1,000,000
Risk per trade:  2% = $20,000
Stop-loss:       3% below entry
Entry price:     $100
Stop price:      $97

Position size = $20,000 / ($100 - $97) = 6,666 shares
Position value = 6,666 × $100 = $666,600

So 2% risk translates to a ~67% capital allocation on this trade
(because the stop is tight relative to the risk budget)
```

If the stop is wider (say 8%), the position would be smaller:
```
Stop price:      $92
Position size = $20,000 / ($100 - $92) = 2,500 shares
Position value = 2,500 × $100 = $250,000 (25% of capital)
```

This is standard fixed-fractional risk management — the dollar risk is constant, but position size adapts to the stop distance.

**Note:** This sizing formula requires a stop-loss to be defined. Strategies validated in Phases 1–6 that use only time-based or indicator-based exits (no stop-loss) will need a fallback sizing method. This is an acknowledged ambiguity to be resolved before Phase 7 implementation — options include using ATR-based implied risk or a fixed percentage of capital per position.

#### Signal Prioritization (When Too Many Signals Fire)

When more signals fire on a given day than available capital allows, a ranking mechanism is needed:

| Priority Method | Description | Best For |
|----------------|-------------|----------|
| Signal strength | Rank by how far the indicator has crossed the threshold | Highest conviction trades |
| Volatility-adjusted | Prefer lower-volatility stocks (smaller position = more trades) | Diversification |
| Historical win rate | Prefer stocks where this strategy has historically performed best | Per-stock optimization |
| Random | Random selection from valid signals | Avoiding selection bias |
| First-come | Process markets in order (e.g., Nordic first, then US) | Simplicity |

This is a configurable parameter. Default: signal strength.

#### Portfolio-Level Risk Controls

| Control | Description | Default |
|---------|-------------|---------|
| Max total exposure | Maximum % of capital deployed at any time | 80% |
| Max drawdown circuit breaker | Stop trading if portfolio draws down X% from peak | 15% |
| Max daily loss | Stop opening new positions if today's P&L exceeds -X% | 3% |
| Correlation filter | Avoid entering highly correlated positions simultaneously | Optional |
| Max positions in one stock | Prevent doubling down | 1 |

#### Portfolio Simulation Metrics (Layer 2 Output)

When the portfolio simulator is built, it will produce:
- Full equity curve with compounding
- Portfolio-level drawdown (different from per-trade drawdown)
- Capital utilization rate (% of time capital is deployed)
- Number of missed signals (due to capital constraints)
- Risk-adjusted returns (Sharpe, Sortino, Calmar at portfolio level)
- Monthly/yearly return breakdown

### Configuration

```yaml
portfolio:
  # Layer 1 (optimization phase)
  fixed_notional_per_trade: 10000

  # Layer 2 (post-validation simulation, built later)
  starting_capital: 1000000
  risk_per_trade_pct: 2.0
  max_concurrent_positions: 15
  max_positions_per_sector: 4
  max_total_exposure_pct: 80
  drawdown_circuit_breaker_pct: 15
  max_daily_loss_pct: 3
  signal_priority: "signal_strength"
```

### Summary

For the current implementation (Phases 1–6), portfolio management adds complexity without improving signal discovery. Each trade is evaluated independently with fixed notional sizing and transaction costs. This is standard practice in strategy research.

Portfolio simulation (Layer 2) is designed here and will be built as a separate module after validated strategies exist. The architecture supports it — the strategy configs produced by the optimizer contain everything needed (entry signals, exit rules, stop levels) to feed into a portfolio simulator.

---

## 9. Indicator Universe

### 9.1 Entry Indicators — Trend Identification

| Indicator | Tunable Parameters | Notes |
|-----------|-------------------|-------|
| SMA | period (5–200) | Simplest moving average |
| EMA | period (5–200) | Exponentially weighted |
| WMA | period (5–200) | Linearly weighted |
| DEMA | period (5–200) | Double exponential — less lag |
| TEMA | period (5–200) | Triple exponential — even less lag |
| Hull MA (HMA) | period (5–200) | Designed to eliminate lag |
| KAMA (Kaufman Adaptive) | period (5–200), fast_sc (2–30), slow_sc (10–50) | Adapts to volatility |
| MA Crossover Pairs | fast_period, slow_period | Signal on cross |
| ADX | period (7–28) | Trend strength, not direction |
| Aroon | period (14–50) | Trend identification via highs/lows timing |
| Ichimoku Cloud | tenkan (7–12), kijun (22–30), senkou_b (44–60) | Multiple signal types: cross, cloud position, cloud color |
| Parabolic SAR | af_start (0.01–0.03), af_max (0.15–0.25) | Trailing stop / trend |
| Supertrend | period (7–21), multiplier (1.5–4.0) | ATR-based trend |
| Linear Regression Slope | period (10–50) | Rate of price change |
| Vortex Indicator | period (14–28) | VI+ and VI- crossovers |

### 9.2 Entry Indicators — Momentum / Oscillators

| Indicator | Tunable Parameters | Notes |
|-----------|-------------------|-------|
| RSI | period (7–21) | Classic overbought/oversold |
| Connors RSI | rsi_period, streak_period, pct_rank_period | Composite momentum |
| Stochastic Oscillator | k_period (5–21), d_period (3–9), smooth_k (1–5) | %K and %D lines |
| Stochastic RSI | rsi_period, stoch_period, k_smooth, d_smooth | RSI fed into stochastic |
| MACD | fast (8–15), slow (21–30), signal (7–12) | Signal line cross, histogram, zero-line |
| CCI | period (14–28) | Commodity Channel Index |
| Williams %R | period (10–21) | Similar to stochastic, inverted |
| ROC (Rate of Change) | period (5–21) | Simple momentum |
| MFI (Money Flow Index) | period (10–21) | Volume-weighted RSI |
| TSI (True Strength Index) | long_period (20–30), short_period (10–15), signal (7–12) | Double-smoothed momentum |
| Ultimate Oscillator | period1 (5–10), period2 (10–20), period3 (20–30) | Multi-timeframe momentum |

### 9.3 Entry Indicators — Volatility

| Indicator | Tunable Parameters | Notes |
|-----------|-------------------|-------|
| ATR | period (7–21) | Average True Range |
| Bollinger Bands | period (15–25), std_dev (1.5–3.0) | Width, %B, squeeze detection |
| Keltner Channels | ema_period (15–25), atr_period (10–20), multiplier (1.0–2.5) | ATR-based bands |
| Donchian Channels | period (10–55) | N-period high/low |
| Historical Volatility | period (10–30) | Std dev of log returns |
| Chaikin Volatility | ema_period (10–20), change_period (10–20) | Volatility rate of change |
| VIX level | threshold values | US markets only — regime filter |

### 9.4 Entry Indicators — Volume-Based

| Indicator | Tunable Parameters | Notes |
|-----------|-------------------|-------|
| OBV | (none — derived) | On Balance Volume trend |
| Volume SMA Ratio | period (10–30) | Current vol vs. average |
| Accumulation/Distribution | (none — derived) | Money flow into/out of asset |
| Chaikin Money Flow | period (10–21) | Bounded A/D version |
| Force Index | period (2–13) | Price × volume momentum |

*Note: VWAP excluded — it requires intraday data and we are using daily bars only.*

### 9.5 Entry Indicators — Support/Resistance & Price Action

| Indicator | Tunable Parameters | Notes |
|-----------|-------------------|-------|
| Pivot Points | type (standard, fibonacci, camarilla) | Daily/weekly pivots |
| Donchian Breakout | period (10–55) | N-period high/low breakout |
| Price vs. N-period High/Low | period (20–252) | Percentage from extremes |
| Inside Bar Detection | (none) | Binary signal |
| Gap Detection | min_gap_pct (0.5–3.0) | Opening gap filter |
| Higher Highs/Lows | lookback (3–10) | Trend structure |

### 9.6 Regime / Context Filters

These are not entry signals themselves but act as filters that enable or disable entry signals based on market context. The GA's genetic switches can enable/disable each filter independently.

| Filter | Parameters | Notes |
|--------|-----------|-------|
| Price vs. 200-day MA | (none) | Only enter when price is above 200 MA (bull regime) |
| Volatility Regime | atr_period, lookback, thresholds | Only enter in low/medium vol environments |
| Day-of-Week | allowed_days set | Only enter on certain weekdays |
| Month-of-Year | allowed_months set | Seasonality filter |
| Distance from 52-week high | threshold_pct | Only enter within X% of 52-week high (momentum) |
| Trend Strength (ADX) | threshold (20–30) | Only enter when ADX confirms trending market |

### 9.7 Exit Strategies

*(Defined in detail in Section 3: Exit Categories 1, 2, and 3)*

### 9.8 Approximate Search Space

With ~55 entry indicators (each with 2–4 tunable parameters), 3 exit categories with ~20 exit types, 6 regime filters, and the combinatorial nature of mixing entry signals with exits and filters, the search space is enormous — easily **10^15+** possible configurations. This is precisely why a genetic algorithm is necessary rather than grid search.

---

## 10. Strategy Scope

- **Optimization:** Per-market (each market is optimized independently)
- **Cross-market validation:** Deferred to a later phase. When implemented, strategies found on one market will be tested on others without re-optimization to assess robustness.

### Constituent Overlap Between Markets
The S&P 500 and Nasdaq 100 share approximately 85–90 constituent stocks. These are treated as **separate markets** regardless of overlap. A stock like Apple will be included in both the Nasdaq 100 optimization and the S&P 500 optimization. This is intentional — the two markets have different compositions and characteristics, and a strategy optimized across all 500 S&P stocks may differ from one optimized across the 100 Nasdaq stocks.

### How the GA Evaluates Across Stocks
When optimizing for a market (e.g., Nasdaq 100), the GA evaluates each candidate strategy (genome) by running it across **all stocks in that market**. The fitness score (win rate) is the **aggregate win rate across all stocks** — i.e., the total number of winning trades across all stocks divided by the total number of trades across all stocks. This finds strategies that work broadly, not strategies overfit to a single stock.

```
Example: A strategy evaluated on Nasdaq 100 (100 stocks)
  - Stock AAPL: 12 trades, 10 wins
  - Stock MSFT: 8 trades, 7 wins
  - Stock GOOGL: 5 trades, 4 wins
  - ... (all 100 stocks)
  - Total: 340 trades, 278 wins
  - Aggregate win rate: 278 / 340 = 81.8%
  - Aggregate net profit: must be > 0 after costs
```

This approach ensures robustness — a strategy cannot achieve a high fitness score by performing well on one stock and poorly on others.

---

## 11. Project Structure (Updated)

### Development Environment

- **Host OS:** Linux Ubuntu (shared development host)
- **Execution:** All code runs inside Docker containers — nothing executes directly on the host
- **Python:** 3.10 (controlled via Docker image)
- **IDE:** VS Code with Remote-Containers or SSH + Docker exec
- **AI assistant:** Claude Code
- **Version control:** Git, synced to GitHub
- **Host resources:** 16 CPU cores, ~25 GB available RAM

### Docker Setup

The project requires a `Dockerfile` and `docker-compose.yml` at the repo root. Key requirements:

```
Base image:         python:3.10-slim (or similar)
Mounted volumes:    - ./data:/app/data        (persist downloaded data across container restarts)
                    - ./results:/app/results  (persist optimization results)
                    - ./config:/app/config    (configuration files)
Working directory:  /app
Dependencies:       Installed from requirements.txt at build time
```

Long-running optimization processes are handled naturally by Docker — containers persist independently of SSH sessions. Use `docker compose up -d` for detached runs, `docker compose logs -f` for monitoring.

Data and results volumes are mounted from the host filesystem so they survive container rebuilds.

### .gitignore Policy

The following directories contain generated or downloaded content and must NOT be committed to git:

```
data/raw/              # Downloaded market data (Parquet) — regenerated by downloader
data/reports/          # Generated quality reports
results/checkpoints/   # GA/Optuna checkpoint files (can be large)
results/logs/          # Run logs
results/performance/   # Generated backtest reports
*.pyc
__pycache__/
.venv/
*.egg-info/
```

The following SHOULD be committed:
```
Dockerfile
docker-compose.yml
config/                # All YAML configuration
data/constituents/     # Index constituent lists (small CSVs, curated)
tsd/                   # Main Python package (all source code)
scripts/               # Runner scripts
notebooks/             # Analysis notebooks
results/strategies/    # Validated strategy configs (the key output)
results/pine_scripts/  # Generated Pine Script files
requirements.txt       # Pinned dependencies
CLAUDE.md              # Claude Code context file (development preferences, conventions)
```

### CLAUDE.md Convention

A `CLAUDE.md` file in the repo root will be read automatically by Claude Code and should contain: project overview, development conventions, coding standards, and any cross-project learnings. This is populated during project setup, drawing on established patterns from existing projects on the host.

### Runtime Estimates (16 cores, population-level parallelism)

Assumptions: 300 individuals, 100 generations, ~50ms per stock per backtest, 16 cores used for parallel evaluation of individuals within each generation. Optuna stage adds ~50% to GA time. Figures are for a single complete run (GA + Optuna across all 10 walk-forward windows).

| Market | Stocks | Time per Generation | GA Only (100 gen × 10 windows) | GA + Optuna |
|--------|--------|--------------------|---------------------------------|-------------|
| **OMXS30** | 30 | ~30 seconds | ~8 hours | **~12 hours** |
| **OBX** | 25 | ~25 seconds | ~7 hours | **~10 hours** |
| **OMXC25** | 25 | ~25 seconds | ~7 hours | **~10 hours** |
| **OMXH25** | 25 | ~25 seconds | ~7 hours | **~10 hours** |
| **Nasdaq 100** | 100 | ~1.5 minutes | ~26 hours | **~40 hours** |
| **S&P 500** | 500 | ~8 minutes | ~130 hours (~5.5 days) | **~8 days** |

**Memory estimate:** All stock data in memory (705 stocks × 10 years daily OHLCV) fits in ~35 MB. With indicators, parallel evaluation overhead, and working copies, peak usage is estimated at 2–4 GB. Well within the 25 GB available.

**Recommendation:** Start development with OMXS30. A full run completes overnight, allowing daily iteration. Save the S&P 500 for final production runs after the pipeline is proven.

**Important caveat:** These are rough estimates. The 50ms-per-stock-per-backtest assumption depends heavily on strategy complexity (number of active indicators, exit combinations). Simple strategies may be 10ms; complex ones with multiple trailing stops and indicator exits could be 200ms. Early profiling in Phase 2 will give accurate numbers.

### Directory Layout

```
trading-system-discovery/
├── README.md
├── CLAUDE.md                  # Claude Code context: conventions, preferences, learnings
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── config/
│   ├── markets.yaml           # Market definitions, tickers, suffixes
│   ├── indicators.yaml        # Indicator parameter ranges
│   ├── optimization.yaml      # GA + Optuna settings
│   ├── fitness.yaml           # Win rate target, cost model
│   └── walkforward.yaml       # WF windows, passing criteria
├── data/
│   ├── raw/                   # Downloaded OHLCV parquet files
│   │   ├── nasdaq_100/
│   │   ├── sp500/
│   │   ├── omxs30/
│   │   ├── omxc25/
│   │   ├── omxh25/
│   │   └── obx/
│   ├── constituents/          # Index constituent lists (CSV)
│   └── reports/               # Data quality reports
├── tsd/                          # Main Python package
│   ├── __init__.py
│   ├── main.py                   # CLI entry point
│   ├── config.py                 # Configuration (frozen dataclass + env helpers)
│   ├── data/
│   │   ├── downloader.py         # Market data acquisition
│   │   ├── constituents.py       # Index constituent scrapers
│   │   └── quality.py            # Data validation & gap detection
│   ├── indicators/
│   │   ├── base.py               # Indicator interface
│   │   ├── trend.py              # SMA, EMA, HMA, Ichimoku, etc.
│   │   ├── momentum.py           # RSI, Stochastic, MACD, etc.
│   │   ├── volatility.py         # ATR, Bollinger, Keltner, etc.
│   │   ├── volume.py             # OBV, CMF, Force Index, etc.
│   │   └── filters.py            # Regime filters, seasonality
│   ├── strategy/
│   │   ├── genome.py             # Strategy DNA encoding
│   │   ├── signals.py            # Signal generation from genome
│   │   ├── exits.py              # All exit types
│   │   ├── execution.py          # Execution timing rules (§3)
│   │   └── evaluator.py          # Backtest engine + metrics
│   ├── optimization/
│   │   ├── ga.py                 # Genetic algorithm (DEAP)
│   │   ├── bayesian.py           # Bayesian optimization (Optuna)
│   │   ├── fitness.py            # Win-rate fitness with profitability gate
│   │   ├── walkforward.py        # Anchored walk-forward engine
│   │   └── pipeline.py           # Staged GA → Optuna pipeline
│   ├── analysis/
│   │   ├── reports.py            # Strategy performance reports
│   │   ├── cross_market.py       # Cross-market validation (future)
│   │   └── robustness.py         # Monte Carlo, complexity checks
│   ├── portfolio/
│   │   ├── simulator.py          # Portfolio equity curve simulation (Layer 2, later)
│   │   ├── sizing.py             # Position sizing models
│   │   └── risk.py               # Portfolio-level risk controls
│   └── export/
│       └── pine_script.py        # Pine Script code generator
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_indicator_testing.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/
│   ├── download_all_data.py
│   ├── run_optimization.py
│   └── generate_pine_scripts.py
└── results/
    ├── strategies/            # Validated strategy configs (JSON)
    ├── pine_scripts/          # Generated Pine Script files
    ├── performance/           # Full backtest reports
    ├── checkpoints/           # GA/Optuna checkpoints for resume
    └── logs/                  # Structured run logs (JSONL)
```

---

## 12. Implementation Phases

### Phase 1: Data Infrastructure
- Set up Git repo
- Constituent list scrapers for all 6 markets
- Data downloader with throttling and error handling
- Data quality verification report
- Parquet storage pipeline

### Phase 2: Indicator Library & Backtesting Engine
- Implement all indicators with standardized interface
- Build backtesting engine with correct execution timing (§3)
- Implement all exit strategy types
- Fitness function with profitability gate and cost model
- Performance benchmarks (target: single backtest < 100ms)

### Phase 3: Genetic Algorithm
- DEAP-based GA with genome encoding
- Genetic switch mechanism for structural optimization
- Population management, logging, parallelization
- **Parallelization strategy:** Needs investigation before implementation. Options include: (a) population-level parallelism (evaluate N individuals concurrently), (b) stock-level parallelism (evaluate one individual across N stocks concurrently), or (c) walk-forward-window-level parallelism. The choice affects memory usage, I/O patterns, and architecture. Investigate and decide during Phase 3 implementation.

### Phase 4: Bayesian Optimization
- Optuna integration for parameter fine-tuning
- Staged pipeline (GA → Optuna) with configurable modes
- Pruning configuration

### Phase 5: Walk-Forward Validation
- Anchored walk-forward engine
- Passing criteria evaluation
- Final holdout validation
- Result aggregation and reporting

### Phase 6: Results & Pine Script Export
- Strategy performance reports
- Pine Script generator
- TradingView verification

### Phase 7: Portfolio Simulation (Layer 2)
- Portfolio equity curve simulator with $1M starting capital
- Position sizing (2% risk per trade)
- Signal prioritization when more signals than capital
- Portfolio-level risk controls (max exposure, circuit breakers)
- Full equity curve reporting with monthly/yearly breakdowns

---

## 13. Decision Record

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Data frequency | **Daily** | Universal availability, less noise, suits swing trading |
| 2 | Trading universe | **Individual stocks** | Larger moves, better win-rate potential |
| 3 | Optimization engine | **Both (staged, configurable)** | GA for structure, Optuna for parameters |
| 4 | Fitness function | **Maximize win rate, profitability as hard gate** | Primary goal is extremely high win rate |
| 5 | Walk-forward design | **Anchored, growing, 6-month OOS, 12-month holdout** | Maximum use of data, rigorous validation |
| 6 | Transaction costs | **0.20% round trip, included in fitness** | Realistic from day one |
| 7 | Survivorship bias | **Accepted for Phase 1** | Documented limitation, can be addressed later |
| 8 | Strategy scope | **Per-market, cross-validation later** | Faster iteration, richer results |
| 9 | Portfolio management | **Layer 1 (fixed notional) now; Layer 2 (full simulation) later** | Signal quality doesn't depend on position sizing |
| 10 | Trade direction | **Long only** | Simplifies search space, matches trader profile |
| 11 | Win rate threshold | **80% default, configurable** | Extremely high win rate is primary goal |
| 12 | Price adjustment | **Split-adjusted only, no dividend adjustment** | Simplifies data handling, dividends negligible for short holds |
| 13 | GA evaluation scope | **All stocks in the market simultaneously** | Finds robust cross-stock strategies, prevents overfitting |
| 14 | S&P 500 / Nasdaq 100 overlap | **Treated as separate markets** | Different compositions and characteristics |
| 15 | Corporate events | **Ignored** | Accepted limitation for Phase 1 |
| 16 | Currency handling | **Local currency, no FX conversion** | All metrics are percentages; currency is a Layer 2 concern |
| 17 | Win/loss definition | **Win = net return > 0 after costs** | Clear binary definition |
| 18 | Checkpointing / restart | **Required — save per generation, resume from checkpoint** | Long-running optimization needs fault tolerance |
| 19 | Parallelization | **To be investigated in Phase 3** | Impact of different strategies needs analysis |
| 20 | Development environment | **Docker containers, Python 3.10, VS Code + Claude Code** | Isolation from shared host, reproducible environment |
| 21 | Initial test market | **OMXS30 (30 stocks)** | Small enough for overnight runs during development |
