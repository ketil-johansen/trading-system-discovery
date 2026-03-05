"""Unit tests for the result persistence module."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from tsd.analysis.robustness import (
    BootstrapCIResult,
    PermutationTestResult,
    RobustnessResult,
)
from tsd.export.persistence import (
    RunManifest,
    _dict_to_dataclass,
    _load_genome,
    _load_pipeline_result,
    _load_trades_parquet,
    _sanitize_dict,
    _save_genome,
    _save_pipeline_result,
    _save_robustness_result,
    _save_trades_parquet,
    _save_walkforward_result,
    generate_run_id,
    load_run,
    save_run,
    save_run_log,
)
from tsd.optimization.bayesian import BayesianResult
from tsd.optimization.ga import GAResult, GenerationStats
from tsd.optimization.pipeline import PipelineResult
from tsd.optimization.walkforward import (
    HoldoutResult,
    WalkForwardResult,
    WalkForwardWindow,
    WindowResult,
)
from tsd.strategy.evaluator import BacktestMetrics, BacktestResult, TradeRecord
from tsd.strategy.genome import (
    BreakevenConfig,
    ChandelierConfig,
    FilterGene,
    IndicatorExitGene,
    IndicatorGene,
    LimitExitGene,
    StopLossConfig,
    StrategyGenome,
    TakeProfitConfig,
    TimeExitGene,
    TrailingStopConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disabled_limit_exits() -> LimitExitGene:
    """Return limit exits with everything disabled."""
    return LimitExitGene(
        stop_loss=StopLossConfig(enabled=False, mode="percent", percent=5.0, atr_multiple=2.0),
        take_profit=TakeProfitConfig(enabled=False, mode="percent", percent=10.0, atr_multiple=3.0),
        trailing_stop=TrailingStopConfig(
            enabled=False,
            mode="percent",
            percent=3.0,
            atr_multiple=2.0,
            activation_percent=5.0,
        ),
        chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
        breakeven=BreakevenConfig(
            enabled=False,
            mode="percent",
            trigger_percent=2.0,
            trigger_atr_multiple=1.0,
        ),
    )


def _disabled_time_exits() -> TimeExitGene:
    """Return time exits with everything disabled."""
    return TimeExitGene(
        max_days_enabled=False,
        max_days=10,
        weekday_exit_enabled=False,
        weekday=4,
        eow_enabled=False,
        eom_enabled=False,
        stagnation_enabled=False,
        stagnation_days=5,
        stagnation_threshold=0.5,
    )


def _minimal_genome() -> StrategyGenome:
    """Build a minimal genome with no indicators or filters."""
    return StrategyGenome(
        entry_indicators=(),
        combination_logic="AND",
        limit_exits=_disabled_limit_exits(),
        indicator_exits=(),
        time_exits=_disabled_time_exits(),
        filters=(),
    )


def _complex_genome() -> StrategyGenome:
    """Build a genome with entry indicators, exits, and filters."""
    return StrategyGenome(
        entry_indicators=(
            IndicatorGene(
                enabled=True,
                indicator_name="sma",
                output_key="sma",
                comparison="above",
                threshold=50.0,
                params={"period": 20},
            ),
            IndicatorGene(
                enabled=False,
                indicator_name="rsi",
                output_key="rsi",
                comparison="below",
                threshold=70.0,
                params={"period": 14},
            ),
        ),
        combination_logic="OR",
        limit_exits=LimitExitGene(
            stop_loss=StopLossConfig(enabled=True, mode="atr", percent=3.0, atr_multiple=2.5),
            take_profit=TakeProfitConfig(enabled=True, mode="percent", percent=8.0, atr_multiple=3.0),
            trailing_stop=TrailingStopConfig(
                enabled=False,
                mode="percent",
                percent=2.0,
                atr_multiple=1.5,
                activation_percent=4.0,
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(
                enabled=False,
                mode="percent",
                trigger_percent=2.0,
                trigger_atr_multiple=1.0,
            ),
        ),
        indicator_exits=(
            IndicatorExitGene(
                enabled=True,
                indicator_name="rsi",
                output_key="rsi",
                comparison="above",
                threshold=80.0,
                params={"period": 14},
                opposite_entry=False,
            ),
        ),
        time_exits=TimeExitGene(
            max_days_enabled=True,
            max_days=20,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=True,
            stagnation_days=7,
            stagnation_threshold=0.3,
        ),
        filters=(
            FilterGene(
                enabled=True,
                filter_name="sma_regime",
                params={"period": 200},
            ),
        ),
    )


def _make_trade(net_return_pct: float, idx: int = 0) -> TradeRecord:
    """Create a TradeRecord with the given net return."""
    return TradeRecord(
        entry_bar=idx,
        entry_date="2020-01-01",
        entry_price=100.0,
        exit_bar=idx + 1,
        exit_date="2020-01-02",
        exit_price=100.0 * (1 + net_return_pct),
        exit_type="take_profit",
        gross_return_pct=net_return_pct + 0.002,
        cost_pct=0.002,
        net_return_pct=net_return_pct,
        net_profit=net_return_pct * 10_000,
        is_win=net_return_pct > 0,
        holding_days=1,
    )


def _make_metrics() -> BacktestMetrics:
    """Create simple backtest metrics."""
    return BacktestMetrics(
        num_trades=10,
        num_wins=8,
        num_losses=2,
        win_rate=0.80,
        net_profit=500.0,
        gross_profit=700.0,
        gross_loss=-200.0,
        profit_factor=3.5,
        avg_win_pct=1.2,
        avg_loss_pct=-0.8,
        win_loss_ratio=1.5,
        max_drawdown_pct=5.0,
        max_drawdown_duration=3,
        sharpe_ratio=1.8,
        sortino_ratio=2.5,
        calmar_ratio=3.0,
        avg_holding_days=5.0,
        longest_win_streak=5,
        longest_loss_streak=2,
        expectancy_per_trade=50.0,
    )


def _make_ga_result(genome: StrategyGenome) -> GAResult:
    """Create a GAResult."""
    return GAResult(
        best_genome=genome,
        best_fitness=0.85,
        generations_run=50,
        logbook=(
            GenerationStats(
                generation=0,
                best_fitness=0.70,
                worst_fitness=0.10,
                avg_fitness=0.40,
                std_fitness=0.15,
                diversity=0.90,
            ),
            GenerationStats(
                generation=49,
                best_fitness=0.85,
                worst_fitness=0.30,
                avg_fitness=0.60,
                std_fitness=0.10,
                diversity=0.50,
            ),
        ),
    )


def _make_pipeline_result(
    genome: StrategyGenome | None = None,
    mode: str = "ga_only",
) -> PipelineResult:
    """Create a PipelineResult."""
    g = genome or _minimal_genome()
    ga_result = _make_ga_result(g) if mode in ("ga_only", "both") else None
    bay_result = None
    if mode in ("bayesian_only", "both"):
        bay_result = BayesianResult(
            best_genome=g,
            best_fitness=0.87,
            trials_run=100,
            trials_pruned=20,
            best_params={"period": 14.0, "threshold": 0.5},
        )
    return PipelineResult(
        mode=mode,
        best_genome=g,
        best_fitness=0.85 if ga_result else 0.87,
        ga_result=ga_result,
        bayesian_result=bay_result,
    )


def _make_walkforward_result(genome: StrategyGenome | None = None) -> WalkForwardResult:
    """Create a WalkForwardResult with one window."""
    g = genome or _minimal_genome()
    metrics = _make_metrics()
    window = WalkForwardWindow(
        window_index=0,
        is_start=pd.Timestamp("2018-01-01"),
        is_end=pd.Timestamp("2021-12-31"),
        oos_start=pd.Timestamp("2022-01-01"),
        oos_end=pd.Timestamp("2022-06-30"),
    )
    pipeline_result = _make_pipeline_result(g)
    window_result = WindowResult(
        window=window,
        best_genome=g,
        is_fitness=0.85,
        oos_metrics=metrics,
        pipeline_result=pipeline_result,
    )
    holdout = HoldoutResult(
        holdout_start=pd.Timestamp("2023-01-01"),
        holdout_end=pd.Timestamp("2023-12-31"),
        genome=g,
        metrics=metrics,
        is_profitable=True,
        win_rate_within_tolerance=True,
    )
    return WalkForwardResult(
        window_results=(window_result,),
        holdout_result=holdout,
        best_genome=g,
        passed=True,
        win_rate_pass=True,
        profitability_pass=True,
        holdout_pass=True,
        low_frequency=False,
        windows_with_trades=1,
        windows_passing_win_rate=1,
        windows_profitable=1,
        avg_oos_win_rate=0.80,
    )


def _make_robustness_result(skipped: bool = False) -> RobustnessResult:
    """Create a RobustnessResult."""
    if skipped:
        return RobustnessResult(
            permutation_tests=(),
            bootstrap_cis=(),
            passed=False,
            num_trades=5,
            skipped=True,
        )
    return RobustnessResult(
        permutation_tests=(
            PermutationTestResult(
                statistic_name="win_rate",
                actual_value=0.80,
                p_value=0.01,
                n_permutations=1000,
                significant=True,
            ),
        ),
        bootstrap_cis=(
            BootstrapCIResult(
                statistic_name="win_rate",
                actual_value=0.80,
                lower_bound=0.72,
                upper_bound=0.88,
                confidence_level=0.95,
                n_resamples=1000,
            ),
        ),
        passed=True,
        num_trades=50,
        skipped=False,
    )


def _make_backtest_result() -> BacktestResult:
    """Create a BacktestResult with sample trades."""
    trades = tuple(_make_trade(r, i) for i, r in enumerate([0.05, -0.02, 0.03]))
    return BacktestResult(trades=trades, metrics=_make_metrics())


# ---------------------------------------------------------------------------
# TestGenerateRunId
# ---------------------------------------------------------------------------


class TestGenerateRunId:
    """Tests for generate_run_id()."""

    def test_format(self) -> None:
        """Run ID matches YYYYMMDD_HHMMSS_hex8 format."""
        run_id = generate_run_id()
        parts = run_id.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 8  # hex chars
        # Verify hex part is valid hex
        int(parts[2], 16)

    def test_uniqueness(self) -> None:
        """Two consecutive run IDs differ."""
        id1 = generate_run_id()
        id2 = generate_run_id()
        assert id1 != id2


# ---------------------------------------------------------------------------
# TestSanitizeDict
# ---------------------------------------------------------------------------


class TestSanitizeDict:
    """Tests for _sanitize_dict()."""

    def test_positive_infinity(self) -> None:
        """float('inf') is converted to 'Infinity'."""
        result = _sanitize_dict({"x": float("inf")})
        assert result["x"] == "Infinity"

    def test_negative_infinity(self) -> None:
        """float('-inf') is converted to '-Infinity'."""
        result = _sanitize_dict({"x": float("-inf")})
        assert result["x"] == "-Infinity"

    def test_nan(self) -> None:
        """float('nan') is converted to 'NaN'."""
        result = _sanitize_dict({"x": float("nan")})
        assert result["x"] == "NaN"

    def test_timestamp(self) -> None:
        """pd.Timestamp is converted to ISO string."""
        ts = pd.Timestamp("2020-01-15 10:30:00")
        result = _sanitize_dict({"ts": ts})
        assert result["ts"] == ts.isoformat()

    def test_nested(self) -> None:
        """Nested dicts and lists are sanitized recursively."""
        data = {
            "outer": {
                "inner": float("inf"),
                "list": [float("-inf"), {"deep": float("nan")}],
            }
        }
        result = _sanitize_dict(data)
        assert result["outer"]["inner"] == "Infinity"
        assert result["outer"]["list"][0] == "-Infinity"
        assert result["outer"]["list"][1]["deep"] == "NaN"


# ---------------------------------------------------------------------------
# TestDictToDataclass
# ---------------------------------------------------------------------------


class TestDictToDataclass:
    """Tests for _dict_to_dataclass()."""

    def test_simple_dataclass(self) -> None:
        """Reconstruct a simple flat dataclass."""
        data = {
            "statistic_name": "win_rate",
            "actual_value": 0.80,
            "p_value": 0.01,
            "n_permutations": 1000,
            "significant": True,
        }
        result = _dict_to_dataclass(data, PermutationTestResult)
        assert result.statistic_name == "win_rate"
        assert result.actual_value == 0.80
        assert result.significant is True

    def test_nested_dataclass(self) -> None:
        """Reconstruct a dataclass with nested dataclass fields."""
        genome = _minimal_genome()
        data = dataclasses.asdict(genome)
        restored = _dict_to_dataclass(data, StrategyGenome)
        assert restored.combination_logic == genome.combination_logic
        assert restored.limit_exits.stop_loss.enabled is False

    def test_tuple_of_dataclass(self) -> None:
        """Reconstruct tuple[Dataclass, ...] fields from lists."""
        data = {
            "permutation_tests": [
                {
                    "statistic_name": "win_rate",
                    "actual_value": 0.8,
                    "p_value": 0.01,
                    "n_permutations": 1000,
                    "significant": True,
                }
            ],
            "bootstrap_cis": [],
            "passed": True,
            "num_trades": 50,
            "skipped": False,
        }
        result = _dict_to_dataclass(data, RobustnessResult)
        assert len(result.permutation_tests) == 1
        assert isinstance(result.permutation_tests, tuple)
        assert result.permutation_tests[0].statistic_name == "win_rate"

    def test_optional_none(self) -> None:
        """None fields for Optional dataclass are preserved."""
        data = {
            "mode": "ga_only",
            "best_genome": {},
            "best_fitness": 0.85,
            "ga_result": None,
            "bayesian_result": None,
        }
        # We need a valid genome dict
        data["best_genome"] = dataclasses.asdict(_minimal_genome())
        result = _dict_to_dataclass(data, PipelineResult)
        assert result.ga_result is None
        assert result.bayesian_result is None

    def test_dict_fields(self) -> None:
        """dict[str, int | float] fields are preserved."""
        data = {
            "enabled": True,
            "indicator_name": "sma",
            "output_key": "sma",
            "comparison": "above",
            "threshold": 50.0,
            "params": {"period": 20, "smooth": 2.5},
        }
        result = _dict_to_dataclass(data, IndicatorGene)
        assert result.params == {"period": 20, "smooth": 2.5}


# ---------------------------------------------------------------------------
# TestSaveLoadGenome
# ---------------------------------------------------------------------------


class TestSaveLoadGenome:
    """Tests for _save_genome / _load_genome round-trip."""

    def test_round_trip_minimal(self, tmp_path: Path) -> None:
        """Minimal genome survives save/load."""
        genome = _minimal_genome()
        path = tmp_path / "genome.json"
        _save_genome(genome, path)
        restored = _load_genome(path)
        assert restored.combination_logic == genome.combination_logic
        assert len(restored.entry_indicators) == 0
        assert restored.limit_exits.stop_loss.enabled is False

    def test_round_trip_complex(self, tmp_path: Path) -> None:
        """Complex genome with indicators/filters survives save/load."""
        genome = _complex_genome()
        path = tmp_path / "genome.json"
        _save_genome(genome, path)
        restored = _load_genome(path)
        assert len(restored.entry_indicators) == 2
        assert restored.entry_indicators[0].indicator_name == "sma"
        assert restored.entry_indicators[0].params == {"period": 20}
        assert restored.limit_exits.stop_loss.enabled is True
        assert restored.limit_exits.stop_loss.mode == "atr"
        assert len(restored.indicator_exits) == 1
        assert restored.indicator_exits[0].threshold == 80.0
        assert len(restored.filters) == 1
        assert restored.filters[0].filter_name == "sma_regime"
        assert restored.time_exits.max_days_enabled is True
        assert restored.time_exits.max_days == 20


# ---------------------------------------------------------------------------
# TestSaveLoadPipelineResult
# ---------------------------------------------------------------------------


class TestSaveLoadPipelineResult:
    """Tests for _save_pipeline_result / _load_pipeline_result."""

    def test_round_trip_ga_only(self, tmp_path: Path) -> None:
        """GA-only pipeline result survives save/load."""
        result = _make_pipeline_result(mode="ga_only")
        path = tmp_path / "pipeline.json"
        _save_pipeline_result(result, path)
        restored = _load_pipeline_result(path)
        assert restored.mode == "ga_only"
        assert restored.best_fitness == result.best_fitness
        assert restored.ga_result is not None
        assert restored.ga_result.generations_run == 50
        assert restored.bayesian_result is None

    def test_infinity_in_metrics(self, tmp_path: Path) -> None:
        """Pipeline result with infinity in metrics round-trips."""
        genome = _minimal_genome()
        ga_result = GAResult(
            best_genome=genome,
            best_fitness=float("inf"),
            generations_run=1,
            logbook=(),
        )
        result = PipelineResult(
            mode="ga_only",
            best_genome=genome,
            best_fitness=float("inf"),
            ga_result=ga_result,
            bayesian_result=None,
        )
        path = tmp_path / "pipeline.json"
        _save_pipeline_result(result, path)
        restored = _load_pipeline_result(path)
        assert math.isinf(restored.best_fitness)
        assert restored.best_fitness > 0


# ---------------------------------------------------------------------------
# TestSaveWalkForward
# ---------------------------------------------------------------------------


class TestSaveWalkForward:
    """Tests for _save_walkforward_result."""

    def test_valid_json(self, tmp_path: Path) -> None:
        """Walk-forward result saves as valid JSON."""
        result = _make_walkforward_result()
        path = tmp_path / "wf.json"
        _save_walkforward_result(result, path)
        data = json.loads(path.read_text())
        assert data["passed"] is True
        assert data["avg_oos_win_rate"] == 0.80

    def test_strips_pipeline_from_windows(self, tmp_path: Path) -> None:
        """Pipeline result in windows is stripped to summary."""
        result = _make_walkforward_result()
        path = tmp_path / "wf.json"
        _save_walkforward_result(result, path)
        data = json.loads(path.read_text())
        window_pr = data["window_results"][0]["pipeline_result"]
        # Should only have mode and best_fitness, not full GA/bayesian
        assert set(window_pr.keys()) == {"mode", "best_fitness"}

    def test_handles_timestamps(self, tmp_path: Path) -> None:
        """pd.Timestamp fields are serialized as ISO strings."""
        result = _make_walkforward_result()
        path = tmp_path / "wf.json"
        _save_walkforward_result(result, path)
        data = json.loads(path.read_text())
        window = data["window_results"][0]["window"]
        assert isinstance(window["is_start"], str)
        assert "2018" in window["is_start"]


# ---------------------------------------------------------------------------
# TestSaveRobustness
# ---------------------------------------------------------------------------


class TestSaveRobustness:
    """Tests for _save_robustness_result."""

    def test_valid_json(self, tmp_path: Path) -> None:
        """Robustness result saves as valid JSON."""
        result = _make_robustness_result()
        path = tmp_path / "rob.json"
        _save_robustness_result(result, path)
        data = json.loads(path.read_text())
        assert data["passed"] is True
        assert data["num_trades"] == 50
        assert len(data["permutation_tests"]) == 1

    def test_skipped_result(self, tmp_path: Path) -> None:
        """Skipped robustness result saves correctly."""
        result = _make_robustness_result(skipped=True)
        path = tmp_path / "rob.json"
        _save_robustness_result(result, path)
        data = json.loads(path.read_text())
        assert data["skipped"] is True
        assert data["passed"] is False
        assert len(data["permutation_tests"]) == 0


# ---------------------------------------------------------------------------
# TestSaveLoadTrades
# ---------------------------------------------------------------------------


class TestSaveLoadTrades:
    """Tests for _save_trades_parquet / _load_trades_parquet."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Trade records survive Parquet round-trip."""
        trades = tuple(_make_trade(r, i) for i, r in enumerate([0.05, -0.02, 0.03]))
        path = tmp_path / "trades.parquet"
        _save_trades_parquet(trades, path)
        restored = _load_trades_parquet(path)
        assert len(restored) == 3
        assert restored[0].entry_bar == 0
        assert restored[0].is_win is True
        assert restored[1].is_win is False
        assert abs(restored[2].net_return_pct - 0.03) < 1e-10

    def test_empty(self, tmp_path: Path) -> None:
        """Empty trade tuple produces valid empty Parquet file."""
        path = tmp_path / "empty.parquet"
        _save_trades_parquet((), path)
        restored = _load_trades_parquet(path)
        assert restored == ()


# ---------------------------------------------------------------------------
# TestSaveRun
# ---------------------------------------------------------------------------


class TestSaveRun:
    """Tests for save_run()."""

    def test_all_components(self, tmp_path: Path) -> None:
        """save_run with all components creates all files."""
        run_id = "20240101_120000_abcd1234"
        manifest = save_run(
            run_id=run_id,
            results_dir=tmp_path,
            pipeline_result=_make_pipeline_result(),
            walkforward_result=_make_walkforward_result(),
            robustness_result=_make_robustness_result(),
            backtest_result=_make_backtest_result(),
        )
        assert manifest.strategy_path is not None
        assert manifest.strategy_path.exists()
        assert manifest.pipeline_path is not None
        assert manifest.pipeline_path.exists()
        assert manifest.walkforward_path is not None
        assert manifest.walkforward_path.exists()
        assert manifest.robustness_path is not None
        assert manifest.robustness_path.exists()
        assert manifest.trades_path is not None
        assert manifest.trades_path.exists()

    def test_manifest_paths(self, tmp_path: Path) -> None:
        """Manifest contains correct run_id and file paths."""
        run_id = "20240101_120000_abcd1234"
        manifest = save_run(
            run_id=run_id,
            results_dir=tmp_path,
            pipeline_result=_make_pipeline_result(),
        )
        assert manifest.run_id == run_id
        assert "abcd1234" in str(manifest.strategy_path)
        assert "abcd1234" in str(manifest.pipeline_path)

    def test_none_components(self, tmp_path: Path) -> None:
        """save_run with no components creates only the log and manifest."""
        run_id = "20240101_120000_abcd1234"
        manifest = save_run(
            run_id=run_id,
            results_dir=tmp_path,
        )
        assert manifest.strategy_path is None
        assert manifest.pipeline_path is None
        assert manifest.walkforward_path is None
        assert manifest.robustness_path is None
        assert manifest.trades_path is None
        assert manifest.report_path is None


# ---------------------------------------------------------------------------
# TestSaveRunLog
# ---------------------------------------------------------------------------


class TestSaveRunLog:
    """Tests for save_run_log()."""

    def test_appends_jsonl(self, tmp_path: Path) -> None:
        """Multiple log calls append separate JSONL lines."""
        run_id = "20240101_120000_abcd1234"
        save_run_log(tmp_path, run_id, "start", {"step": 1})
        save_run_log(tmp_path, run_id, "end", {"step": 2})
        log_path = tmp_path / "logs" / f"{run_id}.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        line1 = json.loads(lines[0])
        line2 = json.loads(lines[1])
        assert line1["event"] == "start"
        assert line2["event"] == "end"

    def test_timestamp_in_each_line(self, tmp_path: Path) -> None:
        """Each log line includes a timestamp."""
        run_id = "20240101_120000_abcd1234"
        save_run_log(tmp_path, run_id, "test_event")
        log_path = tmp_path / "logs" / f"{run_id}.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# TestLoadRun
# ---------------------------------------------------------------------------


class TestLoadRun:
    """Tests for load_run()."""

    def test_loads_saved(self, tmp_path: Path) -> None:
        """load_run returns a manifest matching what save_run produced."""
        run_id = "20240101_120000_abcd1234"
        original = save_run(
            run_id=run_id,
            results_dir=tmp_path,
            pipeline_result=_make_pipeline_result(),
        )
        loaded = load_run(run_id, tmp_path)
        assert loaded.run_id == original.run_id
        assert loaded.pipeline_path is not None
        assert Path(loaded.pipeline_path).exists()

    def test_raises_on_missing(self, tmp_path: Path) -> None:
        """load_run raises FileNotFoundError for unknown run IDs."""
        with pytest.raises(FileNotFoundError, match="No manifest found"):
            load_run("nonexistent_run_id", tmp_path)


# ---------------------------------------------------------------------------
# TestFrozen
# ---------------------------------------------------------------------------


class TestFrozen:
    """Tests for immutability of RunManifest."""

    def test_run_manifest_immutable(self) -> None:
        """RunManifest is frozen and cannot be modified."""
        manifest = RunManifest(
            run_id="test",
            timestamp="2024-01-01T00:00:00",
            strategy_path=None,
            pipeline_path=None,
            walkforward_path=None,
            robustness_path=None,
            trades_path=None,
            report_path=None,
            log_path=Path("logs/test.jsonl"),
        )
        with pytest.raises(FrozenInstanceError):
            manifest.run_id = "changed"  # type: ignore[misc]
