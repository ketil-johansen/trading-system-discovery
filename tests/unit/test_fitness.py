"""Unit tests for the composite fitness function."""

from __future__ import annotations

import pytest

from tsd.optimization.fitness import FitnessConfig, _compute_regularity_score, compute_fitness
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
# Composite score tests
# ---------------------------------------------------------------------------


class TestCompositeScore:
    """Tests for the composite fitness = win_rate × volume × regularity."""

    def test_no_trades_provided_uses_regularity_one(self) -> None:
        """Without trades, regularity defaults to 1.0."""
        metrics = _make_metrics(num_trades=50, win_rate=0.90, net_profit=5000.0)
        # volume = 50/50 = 1.0, regularity = 1.0
        assert compute_fitness(metrics) == pytest.approx(0.90)

    def test_volume_score_scales_linearly(self) -> None:
        """Fewer trades reduce the volume component."""
        cfg = FitnessConfig(target_trades=50)
        metrics = _make_metrics(num_trades=25, win_rate=1.0, net_profit=5000.0)
        # volume = 25/50 = 0.5, regularity = 1.0 (no trades passed)
        assert compute_fitness(metrics, cfg) == pytest.approx(0.5)

    def test_volume_score_saturates(self) -> None:
        """Trades above target don't increase score beyond 1.0."""
        cfg = FitnessConfig(target_trades=50)
        metrics = _make_metrics(num_trades=100, win_rate=0.85, net_profit=5000.0)
        # volume = min(100/50, 1.0) = 1.0
        assert compute_fitness(metrics, cfg) == pytest.approx(0.85)

    def test_full_composite_with_even_trades(self) -> None:
        """All three components contribute with evenly distributed trades."""
        cfg = FitnessConfig(target_trades=50)
        metrics = _make_metrics(num_trades=50, win_rate=0.95, net_profit=5000.0)
        # 10 trades per year across 5 years → CV = 0, regularity = 1.0
        trades = tuple(_make_trade(str(y), f"{m:02d}") for y in range(2019, 2024) for m in range(1, 11))
        # volume = 1.0, regularity = 1.0
        assert compute_fitness(metrics, cfg, trades=trades) == pytest.approx(0.95)

    def test_full_composite_with_uneven_trades(self) -> None:
        """Uneven yearly distribution penalizes regularity."""
        cfg = FitnessConfig(target_trades=50)
        metrics = _make_metrics(num_trades=10, win_rate=1.0, net_profit=5000.0)
        # 8 trades in 2019, 1 in 2020, 1 in 2021 → very uneven
        trades = tuple(_make_trade("2019", f"{m:02d}") for m in range(1, 9))
        trades += (_make_trade("2020"),)
        trades += (_make_trade("2021"),)
        # volume = 10/50 = 0.2
        # CV of [8, 1, 1]: mean=3.33, std=3.30, cv=0.99 → regularity≈0.01
        fitness = compute_fitness(metrics, cfg, trades=trades)
        assert fitness < 0.05  # heavily penalised

    def test_previous_winners_score_poorly(self) -> None:
        """Strategies from prior runs should score low under composite."""
        cfg = FitnessConfig(target_trades=50)
        # "hold forever" run: 30 trades, 100% WR, all in 2016-2017
        metrics = _make_metrics(num_trades=30, win_rate=1.0, net_profit=50000.0)
        trades = tuple(_make_trade("2016", f"{m:02d}") for m in range(1, 12))
        trades += tuple(_make_trade("2016", f"{m:02d}") for m in range(1, 9))
        trades += tuple(_make_trade("2017", f"{m:02d}") for m in range(1, 8))
        trades += (_make_trade("2018"),)
        trades += (_make_trade("2019"),)
        trades += (_make_trade("2021"),)
        # volume = 30/50 = 0.6
        # counts: {2016: 19, 2017: 7, 2018: 1, 2019: 1, 2021: 1}
        # very uneven → low regularity
        fitness = compute_fitness(metrics, cfg, trades=trades)
        assert fitness < 0.15


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
