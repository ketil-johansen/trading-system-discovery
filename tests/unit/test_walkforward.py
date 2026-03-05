"""Unit tests for the walk-forward validation engine."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tsd.optimization.pipeline import PipelineResult
from tsd.optimization.walkforward import (
    HoldoutResult,
    WalkForwardConfig,
    WalkForwardWindow,
    WindowResult,
    _evaluate_oos,
    _evaluate_passing_criteria,
    _select_best_genome,
    _slice_stocks_data,
    generate_windows,
    load_walkforward_config,
    run_walkforward,
)
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, EvaluatorConfig
from tsd.strategy.genome import StrategyGenome, StrategyMeta, load_strategy_config, random_genome

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = "config"


@pytest.fixture
def meta() -> StrategyMeta:
    """Load strategy metadata from config files."""
    return load_strategy_config(Path(_CONFIG_DIR))


@pytest.fixture
def genome(meta: StrategyMeta) -> StrategyGenome:
    """Generate a deterministic genome."""
    return random_genome(meta, rng=random.Random(42))


@pytest.fixture
def wf_config() -> WalkForwardConfig:
    """Default walk-forward config."""
    return WalkForwardConfig()


def _make_ohlcv(start: str, periods: int) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame."""
    dates = pd.bdate_range(start=start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 105.0,
            "Low": 95.0,
            "Close": 102.0,
            "Volume": 1000,
        },
        index=dates,
    )


def _make_backtest_metrics(
    num_trades: int = 50,
    num_wins: int = 42,
    net_profit: float = 500.0,
) -> BacktestMetrics:
    """Create a BacktestMetrics with specified values."""
    num_losses = num_trades - num_wins
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0
    return BacktestMetrics(
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        net_profit=net_profit,
        gross_profit=max(net_profit, 0.0),
        gross_loss=min(net_profit, 0.0),
        profit_factor=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        win_loss_ratio=0.0,
        max_drawdown_pct=0.0,
        max_drawdown_duration=0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        avg_holding_days=0.0,
        longest_win_streak=0,
        longest_loss_streak=0,
        expectancy_per_trade=0.0,
    )


def _make_backtest_result(
    num_trades: int = 50,
    num_wins: int = 42,
    net_profit: float = 500.0,
) -> BacktestResult:
    """Create a BacktestResult."""
    return BacktestResult(
        trades=(),
        metrics=_make_backtest_metrics(num_trades, num_wins, net_profit),
    )


def _make_window_result(
    genome: StrategyGenome,
    window_index: int = 0,
    oos_trades: int = 50,
    oos_wins: int = 42,
    oos_profit: float = 500.0,
    is_fitness: float = 0.85,
) -> WindowResult:
    """Create a WindowResult for testing."""
    return WindowResult(
        window=WalkForwardWindow(
            window_index=window_index,
            is_start=pd.Timestamp("2015-01-01"),
            is_end=pd.Timestamp("2018-01-01"),
            oos_start=pd.Timestamp("2018-01-01"),
            oos_end=pd.Timestamp("2018-07-01"),
        ),
        best_genome=genome,
        is_fitness=is_fitness,
        oos_metrics=_make_backtest_metrics(oos_trades, oos_wins, oos_profit),
        pipeline_result=PipelineResult(
            mode="ga_only",
            best_genome=genome,
            best_fitness=is_fitness,
            ga_result=None,
            bayesian_result=None,
        ),
    )


# ---------------------------------------------------------------------------
# TestWalkForwardConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_defaults(self) -> None:
        """Verify default values."""
        cfg = WalkForwardConfig()
        assert cfg.oos_length_months == 6
        assert cfg.final_holdout_months == 12
        assert cfg.slide_step_months == 6
        assert cfg.min_is_months == 36
        assert cfg.min_oos_windows_win_rate == 8
        assert cfg.min_oos_windows_profitable == 7
        assert cfg.min_win_rate_threshold == 0.80
        assert cfg.holdout_win_rate_tolerance == 0.10
        assert cfg.low_frequency_threshold == 5

    def test_load_from_env(self) -> None:
        """Load config from TSD_WF_* environment variables."""
        env = {
            "TSD_WF_OOS_LENGTH_MONTHS": "3",
            "TSD_WF_FINAL_HOLDOUT_MONTHS": "6",
            "TSD_WF_SLIDE_STEP_MONTHS": "3",
            "TSD_WF_MIN_IS_MONTHS": "24",
            "TSD_WF_MIN_OOS_WINDOWS_WIN_RATE": "5",
            "TSD_WF_MIN_OOS_WINDOWS_PROFITABLE": "4",
            "TSD_WF_MIN_WIN_RATE_THRESHOLD": "0.75",
            "TSD_WF_HOLDOUT_WIN_RATE_TOLERANCE": "0.15",
            "TSD_WF_LOW_FREQUENCY_THRESHOLD": "3",
        }
        with patch.dict("os.environ", env):
            cfg = load_walkforward_config()
        assert cfg.oos_length_months == 3
        assert cfg.final_holdout_months == 6
        assert cfg.slide_step_months == 3
        assert cfg.min_is_months == 24
        assert cfg.min_oos_windows_win_rate == 5
        assert cfg.min_oos_windows_profitable == 4
        assert cfg.min_win_rate_threshold == 0.75
        assert cfg.holdout_win_rate_tolerance == 0.15
        assert cfg.low_frequency_threshold == 3

    def test_frozen(self) -> None:
        """Cannot mutate WalkForwardConfig."""
        cfg = WalkForwardConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.oos_length_months = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestGenerateWindows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateWindows:
    """Tests for generate_windows."""

    def test_correct_window_count(self) -> None:
        """Generate expected number of windows for known date range."""
        # 10 years of data: 36 IS + (window0 OOS=6) = 42 months used,
        # then slide by 6 each time. 120 total - 12 holdout = 108 usable.
        # Windows: IS_end at 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96 months
        # OOS ends at 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102 months
        # 102 <= 108 (holdout starts at 108). So 11 windows.
        start = pd.Timestamp("2010-01-01")
        end = pd.Timestamp("2020-01-01")
        cfg = WalkForwardConfig()
        windows, h_start, h_end = generate_windows(start, end, cfg)
        assert len(windows) >= 1
        assert h_end == end

    def test_anchored_is_starts_at_data_start(self) -> None:
        """All IS windows start at data_start (anchored)."""
        start = pd.Timestamp("2010-01-01")
        end = pd.Timestamp("2020-01-01")
        cfg = WalkForwardConfig()
        windows, _, _ = generate_windows(start, end, cfg)
        for w in windows:
            assert w.is_start == start

    def test_no_holdout_overlap(self) -> None:
        """No OOS window extends into the holdout period."""
        start = pd.Timestamp("2010-01-01")
        end = pd.Timestamp("2020-01-01")
        cfg = WalkForwardConfig()
        windows, h_start, _ = generate_windows(start, end, cfg)
        for w in windows:
            assert w.oos_end <= h_start

    def test_holdout_boundaries(self) -> None:
        """Holdout period occupies the last N months."""
        start = pd.Timestamp("2010-01-01")
        end = pd.Timestamp("2020-01-01")
        cfg = WalkForwardConfig(final_holdout_months=12)
        _, h_start, h_end = generate_windows(start, end, cfg)
        assert h_end == end
        expected_h_start = end - pd.DateOffset(months=12)
        assert h_start == expected_h_start

    def test_is_oos_contiguous(self) -> None:
        """IS end equals OOS start (no gap, no overlap)."""
        start = pd.Timestamp("2010-01-01")
        end = pd.Timestamp("2020-01-01")
        cfg = WalkForwardConfig()
        windows, _, _ = generate_windows(start, end, cfg)
        for w in windows:
            assert w.is_end == w.oos_start

    def test_short_data_raises(self) -> None:
        """Insufficient data raises ValueError."""
        start = pd.Timestamp("2019-01-01")
        end = pd.Timestamp("2020-01-01")
        cfg = WalkForwardConfig()
        with pytest.raises(ValueError, match="too short"):
            generate_windows(start, end, cfg)


# ---------------------------------------------------------------------------
# TestSliceStocksData
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSliceStocksData:
    """Tests for _slice_stocks_data."""

    def test_correct_slice(self) -> None:
        """Sliced data contains only rows in the date range."""
        df = _make_ohlcv("2020-01-01", 500)
        start = pd.Timestamp("2020-06-01")
        end = pd.Timestamp("2020-12-01")
        sliced = _slice_stocks_data({"AAPL": df}, start, end)
        assert "AAPL" in sliced
        sub = sliced["AAPL"]
        assert sub.index.min() >= start
        assert sub.index.max() < end

    def test_excludes_empty(self) -> None:
        """Stocks with no data in range are excluded."""
        df = _make_ohlcv("2020-01-01", 50)  # ends around Mar 2020
        start = pd.Timestamp("2021-01-01")
        end = pd.Timestamp("2021-06-01")
        sliced = _slice_stocks_data({"AAPL": df}, start, end)
        assert len(sliced) == 0


# ---------------------------------------------------------------------------
# TestEvaluateOos
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateOos:
    """Tests for _evaluate_oos."""

    def test_aggregates_across_stocks(self, genome: StrategyGenome) -> None:
        """OOS evaluation aggregates metrics from multiple stocks."""
        mock_result = _make_backtest_result(num_trades=20, num_wins=16, net_profit=100.0)
        stocks = {"AAPL": _make_ohlcv("2020-01-01", 100), "MSFT": _make_ohlcv("2020-01-01", 100)}
        with patch("tsd.optimization.walkforward.run_backtest", return_value=mock_result):
            metrics = _evaluate_oos(genome, stocks, {}, EvaluatorConfig())
        assert metrics.num_trades == 40  # 20 * 2

    def test_handles_failures(self, genome: StrategyGenome) -> None:
        """Failed backtests are skipped gracefully."""
        stocks = {"AAPL": _make_ohlcv("2020-01-01", 100)}
        with patch("tsd.optimization.walkforward.run_backtest", side_effect=RuntimeError("fail")):
            metrics = _evaluate_oos(genome, stocks, {}, EvaluatorConfig())
        assert metrics.num_trades == 0


# ---------------------------------------------------------------------------
# TestSelectBestGenome
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectBestGenome:
    """Tests for _select_best_genome."""

    def test_highest_win_rate(self, genome: StrategyGenome) -> None:
        """Selects genome with highest OOS win rate."""
        genome2 = random_genome(
            load_strategy_config(Path(_CONFIG_DIR)),
            rng=random.Random(99),
        )
        wr1 = _make_window_result(genome, oos_wins=40, oos_trades=50)  # 0.80
        wr2 = _make_window_result(genome2, window_index=1, oos_wins=45, oos_trades=50)  # 0.90
        best = _select_best_genome([wr1, wr2])
        assert best is genome2

    def test_skips_zero_trade_windows(self, genome: StrategyGenome) -> None:
        """Windows with zero trades are excluded from selection."""
        genome2 = random_genome(
            load_strategy_config(Path(_CONFIG_DIR)),
            rng=random.Random(99),
        )
        wr1 = _make_window_result(genome, oos_trades=0, oos_wins=0)
        wr2 = _make_window_result(genome2, window_index=1, oos_wins=40, oos_trades=50)
        best = _select_best_genome([wr1, wr2])
        assert best is genome2


# ---------------------------------------------------------------------------
# TestPassingCriteria
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPassingCriteria:
    """Tests for _evaluate_passing_criteria."""

    def _make_holdout(
        self,
        genome: StrategyGenome,
        profitable: bool = True,
        tolerance: bool = True,
    ) -> HoldoutResult:
        """Create a HoldoutResult."""
        return HoldoutResult(
            holdout_start=pd.Timestamp("2023-01-01"),
            holdout_end=pd.Timestamp("2024-01-01"),
            genome=genome,
            metrics=_make_backtest_metrics(
                num_trades=50,
                num_wins=42 if profitable else 8,
                net_profit=500.0 if profitable else -200.0,
            ),
            is_profitable=profitable,
            win_rate_within_tolerance=tolerance,
        )

    def test_all_pass(self, genome: StrategyGenome) -> None:
        """All criteria met returns passed=True."""
        # 10 windows all passing
        windows = tuple(
            _make_window_result(genome, window_index=i, oos_wins=45, oos_trades=50, oos_profit=100.0) for i in range(10)
        )
        holdout = self._make_holdout(genome)
        cfg = WalkForwardConfig(min_oos_windows_win_rate=8, min_oos_windows_profitable=7)
        result = _evaluate_passing_criteria(windows, holdout, genome, 0.90, cfg)
        assert result.passed is True
        assert result.win_rate_pass is True
        assert result.profitability_pass is True
        assert result.holdout_pass is True

    def test_win_rate_fail(self, genome: StrategyGenome) -> None:
        """Insufficient windows passing win rate fails."""
        # Only 3 windows passing win rate (0.80)
        passing = [
            _make_window_result(genome, window_index=i, oos_wins=45, oos_trades=50, oos_profit=100.0) for i in range(3)
        ]
        failing = [
            _make_window_result(genome, window_index=i + 3, oos_wins=30, oos_trades=50, oos_profit=100.0)
            for i in range(7)
        ]
        windows = tuple(passing + failing)
        holdout = self._make_holdout(genome)
        cfg = WalkForwardConfig(min_oos_windows_win_rate=8, min_oos_windows_profitable=3)
        result = _evaluate_passing_criteria(windows, holdout, genome, 0.70, cfg)
        assert result.win_rate_pass is False
        assert result.passed is False

    def test_profitability_fail(self, genome: StrategyGenome) -> None:
        """Insufficient profitable windows fails."""
        # All pass win rate but only 2 profitable
        profitable = [
            _make_window_result(genome, window_index=i, oos_wins=45, oos_trades=50, oos_profit=100.0) for i in range(2)
        ]
        unprofitable = [
            _make_window_result(genome, window_index=i + 2, oos_wins=45, oos_trades=50, oos_profit=-50.0)
            for i in range(8)
        ]
        windows = tuple(profitable + unprofitable)
        holdout = self._make_holdout(genome)
        cfg = WalkForwardConfig(min_oos_windows_win_rate=2, min_oos_windows_profitable=7)
        result = _evaluate_passing_criteria(windows, holdout, genome, 0.90, cfg)
        assert result.profitability_pass is False
        assert result.passed is False

    def test_holdout_fail(self, genome: StrategyGenome) -> None:
        """Holdout not profitable fails."""
        windows = tuple(
            _make_window_result(genome, window_index=i, oos_wins=45, oos_trades=50, oos_profit=100.0) for i in range(10)
        )
        holdout = self._make_holdout(genome, profitable=False)
        cfg = WalkForwardConfig(min_oos_windows_win_rate=8, min_oos_windows_profitable=7)
        result = _evaluate_passing_criteria(windows, holdout, genome, 0.90, cfg)
        assert result.holdout_pass is False
        assert result.passed is False

    def test_low_frequency_flag(self, genome: StrategyGenome) -> None:
        """Windows with few trades flag low_frequency."""
        normal = [
            _make_window_result(genome, window_index=i, oos_wins=45, oos_trades=50, oos_profit=100.0) for i in range(9)
        ]
        low = [_make_window_result(genome, window_index=9, oos_wins=3, oos_trades=4, oos_profit=10.0)]
        windows = tuple(normal + low)
        holdout = self._make_holdout(genome)
        cfg = WalkForwardConfig(
            min_oos_windows_win_rate=8,
            min_oos_windows_profitable=7,
            low_frequency_threshold=5,
        )
        result = _evaluate_passing_criteria(windows, holdout, genome, 0.90, cfg)
        assert result.low_frequency is True


# ---------------------------------------------------------------------------
# TestRunWalkforward
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunWalkforward:
    """Tests for run_walkforward end-to-end."""

    def _mock_pipeline(self, genome: StrategyGenome) -> PipelineResult:
        """Create a mock PipelineResult."""
        return PipelineResult(
            mode="ga_only",
            best_genome=genome,
            best_fitness=0.85,
            ga_result=None,
            bayesian_result=None,
        )

    def test_calls_pipeline_per_window(self, meta: StrategyMeta, genome: StrategyGenome) -> None:
        """Pipeline is called once per window."""
        # Create 10 years of data to get multiple windows
        stocks = {"AAPL": _make_ohlcv("2010-01-01", 2600)}
        mock_pipeline = self._mock_pipeline(genome)
        mock_bt = _make_backtest_result(num_trades=50, num_wins=45, net_profit=500.0)
        wf_cfg = WalkForwardConfig(min_is_months=36, oos_length_months=6, final_holdout_months=12)

        with (
            patch("tsd.optimization.walkforward.run_pipeline", return_value=mock_pipeline) as p_mock,
            patch("tsd.optimization.walkforward.run_backtest", return_value=mock_bt),
        ):
            result = run_walkforward(
                meta=meta,
                stocks_data=stocks,
                indicator_outputs={},
                wf_config=wf_cfg,
            )
        assert p_mock.call_count == len(result.window_results)
        assert p_mock.call_count >= 1

    def test_evaluates_oos_per_window(self, meta: StrategyMeta, genome: StrategyGenome) -> None:
        """Each window has OOS metrics."""
        stocks = {"AAPL": _make_ohlcv("2010-01-01", 2600)}
        mock_pipeline = self._mock_pipeline(genome)
        mock_bt = _make_backtest_result(num_trades=30, num_wins=25, net_profit=200.0)
        wf_cfg = WalkForwardConfig(min_is_months=36, oos_length_months=6, final_holdout_months=12)

        with (
            patch("tsd.optimization.walkforward.run_pipeline", return_value=mock_pipeline),
            patch("tsd.optimization.walkforward.run_backtest", return_value=mock_bt),
        ):
            result = run_walkforward(
                meta=meta,
                stocks_data=stocks,
                indicator_outputs={},
                wf_config=wf_cfg,
            )
        for wr in result.window_results:
            assert wr.oos_metrics.num_trades > 0

    def test_evaluates_holdout(self, meta: StrategyMeta, genome: StrategyGenome) -> None:
        """Holdout evaluation is performed."""
        stocks = {"AAPL": _make_ohlcv("2010-01-01", 2600)}
        mock_pipeline = self._mock_pipeline(genome)
        mock_bt = _make_backtest_result(num_trades=50, num_wins=45, net_profit=500.0)
        wf_cfg = WalkForwardConfig(min_is_months=36, oos_length_months=6, final_holdout_months=12)

        with (
            patch("tsd.optimization.walkforward.run_pipeline", return_value=mock_pipeline),
            patch("tsd.optimization.walkforward.run_backtest", return_value=mock_bt),
        ):
            result = run_walkforward(
                meta=meta,
                stocks_data=stocks,
                indicator_outputs={},
                wf_config=wf_cfg,
            )
        assert result.holdout_result is not None
        assert result.holdout_result.metrics.num_trades > 0

    def test_short_data_raises(self, meta: StrategyMeta) -> None:
        """Insufficient data raises ValueError."""
        stocks = {"AAPL": _make_ohlcv("2020-01-01", 100)}
        wf_cfg = WalkForwardConfig(min_is_months=36, oos_length_months=6, final_holdout_months=12)

        with pytest.raises(ValueError, match="too short"):
            run_walkforward(
                meta=meta,
                stocks_data=stocks,
                indicator_outputs={},
                wf_config=wf_cfg,
            )
