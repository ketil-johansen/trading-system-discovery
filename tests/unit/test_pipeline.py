"""Unit tests for the staged optimization pipeline."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tsd.optimization.bayesian import BayesianResult
from tsd.optimization.ga import GAResult, GenerationStats
from tsd.optimization.pipeline import (
    VALID_MODES,
    PipelineConfig,
    load_pipeline_config,
    run_pipeline,
)
from tsd.strategy.genome import (
    StrategyGenome,
    StrategyMeta,
    load_strategy_config,
    random_genome,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir() -> Path:
    """Return the path to the config directory."""
    return Path("config")


@pytest.fixture
def meta(config_dir: Path) -> StrategyMeta:
    """Load strategy metadata from config files."""
    return load_strategy_config(config_dir)


@pytest.fixture
def genome(meta: StrategyMeta) -> StrategyGenome:
    """Generate a deterministic genome."""
    return random_genome(meta, rng=random.Random(42))


def _make_ga_result(genome: StrategyGenome) -> GAResult:
    """Create a mock GAResult."""
    return GAResult(
        best_genome=genome,
        best_fitness=0.75,
        generations_run=10,
        logbook=(
            GenerationStats(
                generation=0,
                best_fitness=0.75,
                worst_fitness=0.1,
                avg_fitness=0.4,
                std_fitness=0.15,
                diversity=0.8,
            ),
        ),
    )


def _make_bayesian_result(genome: StrategyGenome) -> BayesianResult:
    """Create a mock BayesianResult."""
    return BayesianResult(
        best_genome=genome,
        best_fitness=0.85,
        trials_run=50,
        trials_pruned=5,
        best_params={"entry_0_period": 14.0},
    )


# ---------------------------------------------------------------------------
# TestPipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self) -> None:
        """Default mode is 'both' and resume is False."""
        config = PipelineConfig()
        assert config.mode == "both"
        assert config.resume is False

    def test_load_from_env(self) -> None:
        """Config loads from environment variables."""
        with patch.dict(
            "os.environ",
            {"TSD_PIPELINE_MODE": "ga_only", "TSD_PIPELINE_RESUME": "true"},
        ):
            config = load_pipeline_config()
            assert config.mode == "ga_only"
            assert config.resume is True

    def test_frozen(self) -> None:
        """PipelineConfig is immutable."""
        config = PipelineConfig()
        with pytest.raises(FrozenInstanceError):
            config.mode = "ga_only"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRunPipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for run_pipeline orchestration."""

    @patch("tsd.optimization.pipeline.run_ga")
    def test_ga_only_mode(
        self,
        mock_run_ga: MagicMock,
        meta: StrategyMeta,
        genome: StrategyGenome,
    ) -> None:
        """GA-only mode runs GA and returns GA result."""
        ga_result = _make_ga_result(genome)
        mock_run_ga.return_value = ga_result

        result = run_pipeline(
            meta=meta,
            stocks_data={},
            indicator_outputs={},
            pipeline_config=PipelineConfig(mode="ga_only"),
        )

        assert result.mode == "ga_only"
        assert result.ga_result is ga_result
        assert result.bayesian_result is None
        assert result.best_genome is genome
        assert result.best_fitness == 0.75
        mock_run_ga.assert_called_once()

    @patch("tsd.optimization.pipeline.run_bayesian")
    def test_bayesian_only_mode(
        self,
        mock_run_bayesian: MagicMock,
        meta: StrategyMeta,
        genome: StrategyGenome,
    ) -> None:
        """Bayesian-only mode runs Bayesian and returns Bayesian result."""
        bayesian_result = _make_bayesian_result(genome)
        mock_run_bayesian.return_value = bayesian_result

        result = run_pipeline(
            meta=meta,
            stocks_data={},
            indicator_outputs={},
            pipeline_config=PipelineConfig(mode="bayesian_only"),
            seed_genome=genome,
        )

        assert result.mode == "bayesian_only"
        assert result.ga_result is None
        assert result.bayesian_result is bayesian_result
        assert result.best_genome is genome
        assert result.best_fitness == 0.85
        mock_run_bayesian.assert_called_once()

    def test_bayesian_only_requires_seed_genome(self, meta: StrategyMeta) -> None:
        """Bayesian-only mode raises ValueError without seed_genome."""
        with pytest.raises(ValueError, match="bayesian_only mode requires a seed_genome"):
            run_pipeline(
                meta=meta,
                stocks_data={},
                indicator_outputs={},
                pipeline_config=PipelineConfig(mode="bayesian_only"),
            )

    @patch("tsd.optimization.pipeline.run_bayesian")
    @patch("tsd.optimization.pipeline.run_ga")
    def test_both_mode(
        self,
        mock_run_ga: MagicMock,
        mock_run_bayesian: MagicMock,
        meta: StrategyMeta,
        genome: StrategyGenome,
    ) -> None:
        """Both mode runs GA then Bayesian."""
        ga_result = _make_ga_result(genome)
        bayesian_result = _make_bayesian_result(genome)
        mock_run_ga.return_value = ga_result
        mock_run_bayesian.return_value = bayesian_result

        result = run_pipeline(
            meta=meta,
            stocks_data={},
            indicator_outputs={},
            pipeline_config=PipelineConfig(mode="both"),
        )

        assert result.mode == "both"
        assert result.ga_result is ga_result
        assert result.bayesian_result is bayesian_result
        mock_run_ga.assert_called_once()
        mock_run_bayesian.assert_called_once()

    @patch("tsd.optimization.pipeline.run_bayesian")
    @patch("tsd.optimization.pipeline.run_ga")
    def test_both_mode_best_from_bayesian(
        self,
        mock_run_ga: MagicMock,
        mock_run_bayesian: MagicMock,
        meta: StrategyMeta,
        genome: StrategyGenome,
    ) -> None:
        """In 'both' mode, final best comes from Bayesian stage."""
        ga_result = _make_ga_result(genome)
        bayesian_result = _make_bayesian_result(genome)
        mock_run_ga.return_value = ga_result
        mock_run_bayesian.return_value = bayesian_result

        result = run_pipeline(
            meta=meta,
            stocks_data={},
            indicator_outputs={},
        )

        assert result.best_fitness == bayesian_result.best_fitness
        assert result.best_genome is bayesian_result.best_genome

    def test_invalid_mode_raises(self, meta: StrategyMeta) -> None:
        """Invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pipeline mode"):
            run_pipeline(
                meta=meta,
                stocks_data={},
                indicator_outputs={},
                pipeline_config=PipelineConfig(mode="invalid"),
            )

    @patch("tsd.optimization.pipeline.run_ga")
    def test_resume_passed_to_ga(
        self,
        mock_run_ga: MagicMock,
        meta: StrategyMeta,
        genome: StrategyGenome,
    ) -> None:
        """Resume flag is forwarded to GA engine."""
        mock_run_ga.return_value = _make_ga_result(genome)

        run_pipeline(
            meta=meta,
            stocks_data={},
            indicator_outputs={},
            pipeline_config=PipelineConfig(mode="ga_only", resume=True),
        )

        call_kwargs = mock_run_ga.call_args[1]
        assert call_kwargs["resume"] is True

    @patch("tsd.optimization.pipeline.run_bayesian")
    def test_resume_passed_to_bayesian(
        self,
        mock_run_bayesian: MagicMock,
        meta: StrategyMeta,
        genome: StrategyGenome,
    ) -> None:
        """Resume flag is forwarded to Bayesian engine."""
        mock_run_bayesian.return_value = _make_bayesian_result(genome)

        run_pipeline(
            meta=meta,
            stocks_data={},
            indicator_outputs={},
            pipeline_config=PipelineConfig(mode="bayesian_only", resume=True),
            seed_genome=genome,
        )

        call_kwargs = mock_run_bayesian.call_args[1]
        assert call_kwargs["resume"] is True

    @patch("tsd.optimization.pipeline.run_bayesian")
    @patch("tsd.optimization.pipeline.run_ga")
    def test_both_feeds_ga_genome_to_bayesian(
        self,
        mock_run_ga: MagicMock,
        mock_run_bayesian: MagicMock,
        meta: StrategyMeta,
        genome: StrategyGenome,
    ) -> None:
        """In 'both' mode, GA best genome is passed to Bayesian stage."""
        ga_result = _make_ga_result(genome)
        bayesian_result = _make_bayesian_result(genome)
        mock_run_ga.return_value = ga_result
        mock_run_bayesian.return_value = bayesian_result

        run_pipeline(
            meta=meta,
            stocks_data={},
            indicator_outputs={},
        )

        bayesian_call_kwargs = mock_run_bayesian.call_args[1]
        assert bayesian_call_kwargs["genome"] is ga_result.best_genome


class TestValidModes:
    """Tests for the VALID_MODES constant."""

    def test_valid_modes_tuple(self) -> None:
        """VALID_MODES contains exactly the three expected modes."""
        assert VALID_MODES == ("ga_only", "bayesian_only", "both")
