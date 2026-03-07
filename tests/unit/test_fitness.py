"""Unit tests for the composite fitness function."""

from __future__ import annotations

import pytest

from tsd.optimization.fitness import (
    FitnessConfig,
    _compute_frequency_score,
    _compute_regularity_score,
    compute_fitness,
)
from tsd.strategy.evaluator import BacktestMetrics, TradeRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(
    num_trades: int = 50,
    win_rate: float = 0.85,
    net_profit: float = 5000.0,
) -> BacktestMetrics:
    """Build a BacktestMetrics with controllable gate-relevant fields."""
    num_wins = int(num_trades * win_rate)
    num_losses = num_trades - num_wins
    return BacktestMetrics(
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        net_profit=net_profit,
        gross_profit=max(net_profit, 0.0),
        gross_loss=min(net_profit, 0.0),
        profit_factor=2.0,
        avg_win_pct=0.03,
        avg_loss_pct=-0.02,
        win_loss_ratio=1.5,
        max_drawdown_pct=0.05,
        max_drawdown_duration=3,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        calmar_ratio=2.0,
        avg_holding_days=5.0,
        longest_win_streak=5,
        longest_loss_streak=2,
        expectancy_per_trade=0.01,
    )


def _make_trade(year: str, month: str = "06") -> TradeRecord:
    """Create a minimal TradeRecord with a given entry year."""
    return TradeRecord(
        entry_bar=0,
        entry_date=f"{year}-{month}-15",
        entry_price=100.0,
        exit_bar=5,
        exit_date=f"{year}-{month}-20",
        exit_price=102.0,
        exit_type="take_profit",
        gross_return_pct=0.02,
        cost_pct=0.002,
        net_return_pct=0.018,
        net_profit=180.0,
        is_win=True,
        holding_days=5,
    )


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


class TestFitnessGates:
    """Tests for the binary pass/fail gates."""

    def test_fails_min_trades(self) -> None:
        """Returns 0.0 when trades below minimum."""
        metrics = _make_metrics(num_trades=5, win_rate=0.90, net_profit=5000.0)
        assert compute_fitness(metrics) == 0.0

    def test_fails_net_profit(self) -> None:
        """Returns 0.0 when net profit is not positive."""
        metrics = _make_metrics(num_trades=50, win_rate=0.85, net_profit=-100.0)
        assert compute_fitness(metrics) == 0.0

    def test_fails_win_rate(self) -> None:
        """Returns 0.0 when win rate below minimum."""
        metrics = _make_metrics(num_trades=50, win_rate=0.60, net_profit=5000.0)
        assert compute_fitness(metrics) == 0.0

    def test_exactly_below_min_trades(self) -> None:
        """One below min_trades fails."""
        cfg = FitnessConfig(min_trades=30, min_win_rate=0.80)
        metrics = _make_metrics(num_trades=29, win_rate=0.80, net_profit=1.0)
        assert compute_fitness(metrics, cfg) == 0.0

    def test_zero_trades(self) -> None:
        """Zero trades returns 0.0."""
        metrics = _make_metrics(num_trades=0, win_rate=0.0, net_profit=0.0)
        assert compute_fitness(metrics) == 0.0


# ---------------------------------------------------------------------------
# Frequency score tests
# ---------------------------------------------------------------------------


class TestFrequencyScore:
    """Tests for _compute_frequency_score band-pass."""

    def test_in_band_scores_one(self) -> None:
        """Rate between min_rate and max_rate scores 1.0."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        # 10 stocks, 5 years, 10 trades → rate = 10/(10*5) = 0.2/yr... too low
        # 10 stocks, 5 years, 50 trades → rate = 50/(10*5) = 1.0/yr → in band
        trades = tuple(_make_trade(str(y), f"{m:02d}") for y in range(2019, 2024) for m in range(1, 11))
        score = _compute_frequency_score(trades, num_stocks=10, config=cfg)
        assert score == pytest.approx(1.0)

    def test_below_min_rate_ramps_up(self) -> None:
        """Below min_rate, score ramps linearly from 0."""
        cfg = FitnessConfig(min_rate=1.0, max_rate=2.0)
        # 10 stocks, 2 years, 10 trades → rate = 10/(10*2) = 0.5
        # score = 0.5 / 1.0 = 0.5
        trades = tuple(_make_trade("2020", f"{m:02d}") for m in range(1, 6))
        trades += tuple(_make_trade("2021", f"{m:02d}") for m in range(1, 6))
        score = _compute_frequency_score(trades, num_stocks=10, config=cfg)
        assert score == pytest.approx(0.5)

    def test_above_max_rate_ramps_down(self) -> None:
        """Above max_rate, score declines toward 0 at 2×max_rate."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        # 5 stocks, 2 years, 30 trades → rate = 30/(5*2) = 3.0
        # score = max(0, 2 - 3.0/2.0) = max(0, 0.5) = 0.5
        trades = tuple(_make_trade("2020", f"{m:02d}") for m in range(1, 11))
        trades += tuple(_make_trade("2021", f"{m:02d}") for m in range(1, 11))
        trades += tuple(_make_trade("2020", f"{m:02d}") for m in range(1, 11))
        score = _compute_frequency_score(trades, num_stocks=5, config=cfg)
        assert score == pytest.approx(0.5)

    def test_double_max_rate_scores_zero(self) -> None:
        """At 2×max_rate, score drops to 0.0."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        # 5 stocks, 2 years, 40 trades → rate = 40/(5*2) = 4.0 = 2×max_rate
        trades = tuple(_make_trade("2020", f"{m:02d}") for m in range(1, 11))
        trades += tuple(_make_trade("2021", f"{m:02d}") for m in range(1, 11))
        trades += tuple(_make_trade("2020", f"{m:02d}") for m in range(1, 11))
        trades += tuple(_make_trade("2021", f"{m:02d}") for m in range(1, 11))
        score = _compute_frequency_score(trades, num_stocks=5, config=cfg)
        assert score == pytest.approx(0.0)

    def test_no_trades_scores_zero(self) -> None:
        """No trades gives 0.0."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        assert _compute_frequency_score((), num_stocks=10, config=cfg) == 0.0

    def test_market_size_invariant(self) -> None:
        """Same per-stock rate yields same score regardless of universe size."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        # 1 trade/stock/year for 5 years → rate = 1.0, in band
        trades_small = tuple(_make_trade(str(y)) for y in range(2019, 2024) for _ in range(10))
        trades_large = tuple(_make_trade(str(y)) for y in range(2019, 2024) for _ in range(500))
        score_small = _compute_frequency_score(trades_small, num_stocks=10, config=cfg)
        score_large = _compute_frequency_score(trades_large, num_stocks=500, config=cfg)
        assert score_small == pytest.approx(score_large)


# ---------------------------------------------------------------------------
# Composite score tests
# ---------------------------------------------------------------------------


class TestCompositeScore:
    """Tests for the composite fitness = win_rate × frequency × regularity."""

    def test_no_trades_provided_scores_zero(self) -> None:
        """Without trades, frequency_score is 0.0."""
        metrics = _make_metrics(num_trades=50, win_rate=0.90, net_profit=5000.0)
        assert compute_fitness(metrics) == 0.0

    def test_in_band_with_even_trades(self) -> None:
        """Rate in band + even distribution → fitness ≈ win_rate."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        # 10 stocks, 5 years, 50 trades → rate = 1.0 (in band)
        # 10 per year → perfectly regular
        metrics = _make_metrics(num_trades=50, win_rate=0.95, net_profit=5000.0)
        trades = tuple(_make_trade(str(y), f"{m:02d}") for y in range(2019, 2024) for m in range(1, 11))
        fitness = compute_fitness(metrics, cfg, trades=trades, num_stocks=10)
        assert fitness == pytest.approx(0.95)

    def test_excessive_trades_penalized(self) -> None:
        """High frequency strategy scores poorly despite good win rate."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        # Simulate the previous run: ~4800 trades, 30 stocks, 11 years
        # rate = 4800/(30*11) = 14.5 → way above 2×max_rate → freq = 0.0
        metrics = _make_metrics(num_trades=4800, win_rate=0.88, net_profit=200000.0)
        trades = tuple(
            _make_trade(str(y), f"{m:02d}")
            for y in range(2015, 2026)
            for m in range(1, 13)
            for _ in range(40)  # ~40 trades/month
        )
        # Trim to 4800 trades
        trades = trades[:4800]
        fitness = compute_fitness(metrics, cfg, trades=trades, num_stocks=30)
        assert fitness == 0.0

    def test_ideal_rate_strategy(self) -> None:
        """2 trades/stock/year across 30 stocks over 5 years scores well."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        # 30 stocks, 5 years, 300 trades → rate = 300/(30*5) = 2.0 (at max_rate)
        metrics = _make_metrics(num_trades=300, win_rate=0.90, net_profit=10000.0)
        trades = tuple(
            _make_trade(str(y), f"{m:02d}") for y in range(2019, 2024) for m in range(1, 13) for _ in range(5)
        )
        trades = trades[:300]
        fitness = compute_fitness(metrics, cfg, trades=trades, num_stocks=30)
        # frequency = 1.0, regularity ≈ 1.0
        assert fitness == pytest.approx(0.90, abs=0.02)

    def test_previous_hyperactive_strategy_eliminated(self) -> None:
        """The 14.7 trades/stock/year strategy from last run gets fitness 0."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=2.0)
        metrics = _make_metrics(num_trades=4866, win_rate=0.878, net_profit=216000.0)
        trades = tuple(
            _make_trade(str(y), f"{(m % 12) + 1:02d}")
            for y in range(2015, 2026)
            for m in range(442)  # ~442/year
        )
        trades = trades[:4866]
        fitness = compute_fitness(metrics, cfg, trades=trades, num_stocks=30)
        assert fitness == 0.0

    def test_uneven_trades_penalized(self) -> None:
        """Uneven yearly distribution penalizes regularity."""
        cfg = FitnessConfig(min_rate=0.5, max_rate=4.0)
        metrics = _make_metrics(num_trades=10, win_rate=1.0, net_profit=5000.0)
        # 8 in 2019, 1 in 2020, 1 in 2021 → very uneven
        trades = tuple(_make_trade("2019", f"{m:02d}") for m in range(1, 9))
        trades += (_make_trade("2020"),)
        trades += (_make_trade("2021"),)
        fitness = compute_fitness(metrics, cfg, trades=trades, num_stocks=1)
        assert fitness < 0.05


# ---------------------------------------------------------------------------
# Regularity score tests
# ---------------------------------------------------------------------------


class TestRegularityScore:
    """Tests for _compute_regularity_score."""

    def test_perfectly_even(self) -> None:
        """Same count each year gives score 1.0."""
        trades = tuple(_make_trade(str(y)) for y in range(2018, 2024) for _ in range(5))
        assert _compute_regularity_score(trades) == pytest.approx(1.0)

    def test_single_year(self) -> None:
        """Only one year gives score 0.0 (can't assess regularity)."""
        trades = tuple(_make_trade("2020") for _ in range(10))
        assert _compute_regularity_score(trades) == 0.0

    def test_empty_trades(self) -> None:
        """No trades gives score 0.0."""
        assert _compute_regularity_score(()) == 0.0

    def test_moderately_uneven(self) -> None:
        """Moderate unevenness produces mid-range score."""
        # [10, 5, 5, 10] → mean=7.5, std≈2.5, cv≈0.33 → score≈0.67
        trades = tuple(_make_trade("2019") for _ in range(10))
        trades += tuple(_make_trade("2020") for _ in range(5))
        trades += tuple(_make_trade("2021") for _ in range(5))
        trades += tuple(_make_trade("2022") for _ in range(10))
        score = _compute_regularity_score(trades)
        assert 0.5 < score < 0.8
