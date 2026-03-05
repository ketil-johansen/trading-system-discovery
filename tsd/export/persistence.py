"""Persistence layer for optimization results.

Saves and loads optimization results (genomes, pipeline results,
walk-forward results, robustness results, trade records) to structured
JSON and Parquet files for downstream reporting and review.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_type_hints

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tsd.analysis.reports import PerformanceReport, save_report
from tsd.analysis.robustness import (
    BootstrapCIResult,
    PermutationTestResult,
    RobustnessResult,
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

LOGGER = logging.getLogger(__name__)

# Mapping from type name to actual class for _dict_to_dataclass reconstruction.
_DATACLASS_REGISTRY: dict[str, type] = {
    "StrategyGenome": StrategyGenome,
    "IndicatorGene": IndicatorGene,
    "IndicatorExitGene": IndicatorExitGene,
    "TimeExitGene": TimeExitGene,
    "FilterGene": FilterGene,
    "LimitExitGene": LimitExitGene,
    "StopLossConfig": StopLossConfig,
    "TakeProfitConfig": TakeProfitConfig,
    "TrailingStopConfig": TrailingStopConfig,
    "ChandelierConfig": ChandelierConfig,
    "BreakevenConfig": BreakevenConfig,
    "BacktestMetrics": BacktestMetrics,
    "BacktestResult": BacktestResult,
    "TradeRecord": TradeRecord,
    "GAResult": GAResult,
    "GenerationStats": GenerationStats,
    "BayesianResult": BayesianResult,
    "PipelineResult": PipelineResult,
    "WalkForwardWindow": WalkForwardWindow,
    "WindowResult": WindowResult,
    "HoldoutResult": HoldoutResult,
    "WalkForwardResult": WalkForwardResult,
    "PermutationTestResult": PermutationTestResult,
    "BootstrapCIResult": BootstrapCIResult,
    "RobustnessResult": RobustnessResult,
}


@dataclass(frozen=True)
class RunManifest:
    """Manifest describing the files saved for a single run."""

    run_id: str
    timestamp: str
    strategy_path: Path | None
    pipeline_path: Path | None
    walkforward_path: Path | None
    robustness_path: Path | None
    trades_path: Path | None
    report_path: Path | None
    log_path: Path


# ---------------------------------------------------------------------------
# Private helpers — sanitize / restore
# ---------------------------------------------------------------------------


def _sanitize_float(value: float) -> str | float:
    """Convert special float values to JSON-safe string representations."""
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if math.isnan(value):
        return "NaN"
    return value


def _sanitize_value(value: Any) -> Any:
    """Convert non-JSON-serializable values to JSON-safe representations."""
    if isinstance(value, float):
        return _sanitize_float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return value


def _sanitize_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize a dict (from dataclasses.asdict) for JSON."""
    return {k: _sanitize_value(v) for k, v in obj.items()}


def _restore_value(value: Any, target_type: type | str | None) -> Any:
    """Restore JSON-safe representations back to Python types."""
    if isinstance(value, str):
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
        if value == "NaN":
            return float("nan")
    return value


def _resolve_type_name(type_hint: Any) -> str | None:
    """Extract the class name from a type hint if it's a known dataclass."""
    if isinstance(type_hint, type) and type_hint.__name__ in _DATACLASS_REGISTRY:
        return type_hint.__name__
    return None


_TUPLE_ELLIPSIS_ARGS = 2  # tuple[X, ...] has exactly 2 type args


def _is_tuple_of_dataclass(type_hint: Any) -> str | None:
    """Check if type_hint is tuple[SomeDataclass, ...] and return class name."""
    origin = getattr(type_hint, "__origin__", None)
    if origin is tuple:
        args: tuple[Any, ...] = getattr(type_hint, "__args__", ())
        if len(args) == _TUPLE_ELLIPSIS_ARGS and args[1] is Ellipsis:
            inner = args[0]
            if isinstance(inner, type) and inner.__name__ in _DATACLASS_REGISTRY:
                return inner.__name__
    return None


def _is_optional_dataclass(type_hint: Any) -> str | None:
    """Check if type_hint is SomeDataclass | None and return class name."""
    origin = getattr(type_hint, "__origin__", None)
    # X | None compiles to types.UnionType in 3.10+
    import types as _types  # noqa: PLC0415

    if origin is _types.UnionType or (hasattr(type_hint, "__args__") and origin is not tuple):
        args = getattr(type_hint, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = non_none[0]
            if isinstance(inner, type) and inner.__name__ in _DATACLASS_REGISTRY:
                return inner.__name__
    return None


def _dict_to_dataclass(data: dict[str, Any], cls: type) -> Any:
    """Reconstruct a frozen dataclass from a dict.

    Handles nested dataclasses, tuple[DC, ...], DC | None, and
    dict[str, int | float] fields. Restores special float values.
    """
    hints = get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}

    for name in fields:
        if name not in data:
            continue
        value = data[name]
        hint = hints.get(name)

        # Check for tuple[Dataclass, ...]
        tuple_dc_name = _is_tuple_of_dataclass(hint) if hint else None
        if tuple_dc_name and isinstance(value, list):
            dc_cls = _DATACLASS_REGISTRY[tuple_dc_name]
            kwargs[name] = tuple(_dict_to_dataclass(item, dc_cls) for item in value)
            continue

        # Check for Optional[Dataclass] (DC | None)
        opt_dc_name = _is_optional_dataclass(hint) if hint else None
        if opt_dc_name and isinstance(value, dict):
            dc_cls = _DATACLASS_REGISTRY[opt_dc_name]
            kwargs[name] = _dict_to_dataclass(value, dc_cls)
            continue
        if opt_dc_name and value is None:
            kwargs[name] = None
            continue

        # Check for direct dataclass field
        dc_name = _resolve_type_name(hint) if hint else None
        if dc_name and isinstance(value, dict):
            dc_cls = _DATACLASS_REGISTRY[dc_name]
            kwargs[name] = _dict_to_dataclass(value, dc_cls)
            continue

        # Check for dict[str, ...] — keep as dict, just restore values
        if isinstance(value, dict):
            kwargs[name] = {k: _restore_value(v, None) for k, v in value.items()}
            continue

        # Check for tuple fields that aren't dataclass tuples (e.g. tuple[str, ...])
        origin = getattr(hint, "__origin__", None) if hint else None
        if origin is tuple and isinstance(value, list):
            kwargs[name] = tuple(_restore_value(item, None) for item in value)
            continue

        # Scalar restoration
        kwargs[name] = _restore_value(value, hint)

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Private helpers — JSON I/O
# ---------------------------------------------------------------------------


def _save_json(data: dict[str, Any], path: Path) -> None:
    """Write a dict as pretty-printed JSON, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return as dict."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Genome save / load
# ---------------------------------------------------------------------------


def _save_genome(genome: StrategyGenome, path: Path) -> None:
    """Save a StrategyGenome as JSON."""
    data = _sanitize_dict(dataclasses.asdict(genome))
    _save_json(data, path)


def _load_genome(path: Path) -> StrategyGenome:
    """Load a StrategyGenome from JSON."""
    data = _load_json(path)
    result: StrategyGenome = _dict_to_dataclass(data, StrategyGenome)
    return result


# ---------------------------------------------------------------------------
# Pipeline result save / load
# ---------------------------------------------------------------------------


def _save_pipeline_result(result: PipelineResult, path: Path) -> None:
    """Save a PipelineResult as JSON."""
    data = _sanitize_dict(dataclasses.asdict(result))
    _save_json(data, path)


def _load_pipeline_result(path: Path) -> PipelineResult:
    """Load a PipelineResult from JSON."""
    data = _load_json(path)
    result: PipelineResult = _dict_to_dataclass(data, PipelineResult)
    return result


# ---------------------------------------------------------------------------
# Walk-forward result save
# ---------------------------------------------------------------------------


def _strip_window_pipeline(window_dict: dict[str, Any]) -> dict[str, Any]:
    """Replace full pipeline_result in a window dict with a summary."""
    pr = window_dict.get("pipeline_result")
    if pr and isinstance(pr, dict):
        window_dict["pipeline_result"] = {
            "mode": pr.get("mode"),
            "best_fitness": pr.get("best_fitness"),
        }
    return window_dict


def _save_walkforward_result(result: WalkForwardResult, path: Path) -> None:
    """Save a WalkForwardResult as JSON, stripping pipeline from windows."""
    data = dataclasses.asdict(result)
    # Strip full pipeline results from each window to avoid duplication
    if "window_results" in data:
        data["window_results"] = [_strip_window_pipeline(w) for w in data["window_results"]]
    data = _sanitize_dict(data)
    _save_json(data, path)


# ---------------------------------------------------------------------------
# Robustness result save
# ---------------------------------------------------------------------------


def _save_robustness_result(result: RobustnessResult, path: Path) -> None:
    """Save a RobustnessResult as JSON."""
    data = _sanitize_dict(dataclasses.asdict(result))
    _save_json(data, path)


# ---------------------------------------------------------------------------
# Trade records — Parquet
# ---------------------------------------------------------------------------


def _save_trades_parquet(trades: tuple[TradeRecord, ...], path: Path) -> None:
    """Save trade records as a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trades:
        # Write an empty Parquet file with the correct schema
        schema = pa.schema(
            [
                ("entry_bar", pa.int64()),
                ("entry_date", pa.string()),
                ("entry_price", pa.float64()),
                ("exit_bar", pa.int64()),
                ("exit_date", pa.string()),
                ("exit_price", pa.float64()),
                ("exit_type", pa.string()),
                ("gross_return_pct", pa.float64()),
                ("cost_pct", pa.float64()),
                ("net_return_pct", pa.float64()),
                ("net_profit", pa.float64()),
                ("is_win", pa.bool_()),
                ("holding_days", pa.int64()),
            ]
        )
        table = pa.table({f.name: [] for f in schema}, schema=schema)
        pq.write_table(table, path)
        return

    rows = [dataclasses.asdict(t) for t in trades]
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def _load_trades_parquet(path: Path) -> tuple[TradeRecord, ...]:
    """Load trade records from a Parquet file."""
    table = pq.read_table(path)
    df = table.to_pandas()
    if df.empty:
        return ()
    return tuple(TradeRecord(**row) for row in df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------


def generate_run_id() -> str:
    """Generate a unique run ID in YYYYMMDD_HHMMSS_hex8 format."""
    now = datetime.now(tz=timezone.utc)
    hex_part = uuid.uuid4().hex[:8]
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{hex_part}"


# ---------------------------------------------------------------------------
# Run log (JSONL)
# ---------------------------------------------------------------------------


def save_run_log(
    results_dir: Path,
    run_id: str,
    event: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Append a timestamped JSONL line to the run's log file."""
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.jsonl"
    entry: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event": event,
    }
    if data:
        entry["data"] = data
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Public API — save_run / load_run
# ---------------------------------------------------------------------------


def save_run(  # noqa: PLR0913
    run_id: str,
    results_dir: Path,
    pipeline_result: PipelineResult | None = None,
    walkforward_result: WalkForwardResult | None = None,
    robustness_result: RobustnessResult | None = None,
    backtest_result: BacktestResult | None = None,
    report: PerformanceReport | None = None,
) -> RunManifest:
    """Save all results for a run and return a manifest.

    Args:
        run_id: Unique identifier for this run.
        results_dir: Root directory for results output.
        pipeline_result: Staged pipeline output (GA + Bayesian).
        walkforward_result: Walk-forward validation output.
        robustness_result: Statistical robustness output.
        backtest_result: Backtest with trade records.
        report: Performance report to save alongside results.

    Returns:
        RunManifest with paths to all saved files.
    """
    strategy_path: Path | None = None
    pipeline_path: Path | None = None
    walkforward_path: Path | None = None
    robustness_path: Path | None = None
    trades_path: Path | None = None
    report_path: Path | None = None
    log_path = results_dir / "logs" / f"{run_id}.jsonl"

    # Save genome from pipeline or walkforward best
    genome = None
    if pipeline_result:
        genome = pipeline_result.best_genome
    elif walkforward_result:
        genome = walkforward_result.best_genome

    if genome:
        strategy_path = results_dir / "strategies" / f"{run_id}_genome.json"
        _save_genome(genome, strategy_path)
        LOGGER.info("Saved genome to %s", strategy_path)

    if pipeline_result:
        pipeline_path = results_dir / "performance" / f"{run_id}_pipeline.json"
        _save_pipeline_result(pipeline_result, pipeline_path)
        LOGGER.info("Saved pipeline result to %s", pipeline_path)

    if walkforward_result:
        walkforward_path = results_dir / "performance" / f"{run_id}_walkforward.json"
        _save_walkforward_result(walkforward_result, walkforward_path)
        LOGGER.info("Saved walk-forward result to %s", walkforward_path)

    if robustness_result:
        robustness_path = results_dir / "performance" / f"{run_id}_robustness.json"
        _save_robustness_result(robustness_result, robustness_path)
        LOGGER.info("Saved robustness result to %s", robustness_path)

    if backtest_result:
        trades_path = results_dir / "performance" / f"{run_id}_trades.parquet"
        _save_trades_parquet(backtest_result.trades, trades_path)
        LOGGER.info("Saved %d trades to %s", len(backtest_result.trades), trades_path)

    if report:
        report_path = save_report(report, results_dir)

    # Log the save event
    save_run_log(
        results_dir,
        run_id,
        "run_saved",
        {
            "has_pipeline": pipeline_result is not None,
            "has_walkforward": walkforward_result is not None,
            "has_robustness": robustness_result is not None,
            "has_trades": backtest_result is not None,
            "has_report": report is not None,
        },
    )

    manifest = RunManifest(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        strategy_path=strategy_path,
        pipeline_path=pipeline_path,
        walkforward_path=walkforward_path,
        robustness_path=robustness_path,
        trades_path=trades_path,
        report_path=report_path,
        log_path=log_path,
    )

    # Save manifest itself
    manifest_path = results_dir / "performance" / f"{run_id}_manifest.json"
    manifest_dict = _sanitize_dict(dataclasses.asdict(manifest))
    _save_json(manifest_dict, manifest_path)
    LOGGER.info("Saved manifest to %s", manifest_path)

    return manifest


def load_run(run_id: str, results_dir: Path) -> RunManifest:
    """Load a run manifest by run ID.

    Args:
        run_id: The run identifier.
        results_dir: Root directory for results output.

    Returns:
        RunManifest with paths to all saved files.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
    manifest_path = results_dir / "performance" / f"{run_id}_manifest.json"
    if not manifest_path.exists():
        msg = f"No manifest found for run {run_id} at {manifest_path}"
        raise FileNotFoundError(msg)

    data = _load_json(manifest_path)

    # Reconstruct Path | None fields
    path_fields = {
        "strategy_path",
        "pipeline_path",
        "walkforward_path",
        "robustness_path",
        "trades_path",
        "report_path",
        "log_path",
    }
    for field_name in path_fields:
        val = data.get(field_name)
        if val is not None:
            data[field_name] = Path(val)
        elif field_name == "log_path":
            data[field_name] = Path(results_dir / "logs" / f"{run_id}.jsonl")
        else:
            data[field_name] = None

    return RunManifest(**data)
