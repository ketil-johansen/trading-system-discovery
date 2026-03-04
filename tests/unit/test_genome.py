"""Unit tests for the strategy genome module."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tsd.strategy.genome import (
    BreakevenConfig,
    ChandelierConfig,
    FilterGene,
    IndicatorExitGene,
    IndicatorGene,
    LimitExitGene,
    OutputMeta,
    StopLossConfig,
    StrategyGenome,
    StrategyMeta,
    TakeProfitConfig,
    TimeExitGene,
    TrailingStopConfig,
    flat_to_genome,
    genome_length,
    genome_to_flat,
    load_strategy_config,
    random_genome,
    validate_genome,
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
    """Generate a deterministic random genome."""
    return random_genome(meta, rng=random.Random(42))


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    """All gene dataclasses must be immutable."""

    def test_output_meta_frozen(self) -> None:
        om = OutputMeta(key="rsi", output_type="oscillator", threshold_min=0, threshold_max=100)
        with pytest.raises(FrozenInstanceError):
            om.key = "other"  # type: ignore[misc]

    def test_indicator_gene_frozen(self) -> None:
        gene = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=30.0,
        )
        with pytest.raises(FrozenInstanceError):
            gene.enabled = False  # type: ignore[misc]

    def test_stop_loss_config_frozen(self) -> None:
        cfg = StopLossConfig(enabled=True, mode="percent", percent=3.0, atr_multiple=2.0)
        with pytest.raises(FrozenInstanceError):
            cfg.percent = 5.0  # type: ignore[misc]

    def test_limit_exit_gene_frozen(self) -> None:
        le = _make_limit_exits()
        with pytest.raises(FrozenInstanceError):
            le.stop_loss = None  # type: ignore[misc]

    def test_strategy_genome_frozen(self, genome: StrategyGenome) -> None:
        with pytest.raises(FrozenInstanceError):
            genome.combination_logic = "OR"  # type: ignore[misc]

    def test_time_exit_gene_frozen(self) -> None:
        te = TimeExitGene(
            max_days_enabled=True,
            max_days=10,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=False,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        with pytest.raises(FrozenInstanceError):
            te.max_days = 20  # type: ignore[misc]

    def test_filter_gene_frozen(self) -> None:
        fg = FilterGene(enabled=True, filter_name="price_vs_ma")
        with pytest.raises(FrozenInstanceError):
            fg.enabled = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_strategy_config
# ---------------------------------------------------------------------------


class TestLoadStrategyConfig:
    """Tests for loading strategy configuration."""

    def test_returns_strategy_meta(self, meta: StrategyMeta) -> None:
        assert isinstance(meta, StrategyMeta)

    def test_entry_slots_count(self, meta: StrategyMeta) -> None:
        assert meta.num_entry_slots == 4

    def test_indicator_exit_slots(self, meta: StrategyMeta) -> None:
        assert meta.num_indicator_exit_slots == 1

    def test_filter_slots(self, meta: StrategyMeta) -> None:
        assert meta.num_filter_slots == 2

    def test_separates_filters_from_indicators(self, meta: StrategyMeta) -> None:
        assert "price_vs_ma" in meta.filter_names
        assert "volatility_regime" in meta.filter_names
        assert "price_vs_ma" not in meta.indicator_names
        assert "volatility_regime" not in meta.indicator_names

    def test_indicator_names_sorted(self, meta: StrategyMeta) -> None:
        assert list(meta.indicator_names) == sorted(meta.indicator_names)

    def test_filter_names_sorted(self, meta: StrategyMeta) -> None:
        assert list(meta.filter_names) == sorted(meta.filter_names)

    def test_indicator_outputs_populated(self, meta: StrategyMeta) -> None:
        rsi_outputs = meta.indicator_outputs["rsi"]
        assert len(rsi_outputs) == 1
        assert rsi_outputs[0].key == "rsi"
        assert rsi_outputs[0].output_type == "oscillator"

    def test_multi_output_indicator(self, meta: StrategyMeta) -> None:
        ichimoku_outputs = meta.indicator_outputs["ichimoku"]
        assert len(ichimoku_outputs) == 4
        keys = {o.key for o in ichimoku_outputs}
        assert keys == {"conversion", "base", "span_a", "span_b"}

    def test_comparisons_loaded(self, meta: StrategyMeta) -> None:
        assert "GT" in meta.comparisons
        assert "LT" in meta.comparisons
        assert "CROSS_ABOVE" in meta.comparisons
        assert "CROSS_BELOW" in meta.comparisons

    def test_exit_config_populated(self, meta: StrategyMeta) -> None:
        assert "stop_loss" in meta.exit_config
        assert "take_profit" in meta.exit_config

    def test_time_exit_config_populated(self, meta: StrategyMeta) -> None:
        assert "max_days" in meta.time_exit_config
        assert "stagnation_days" in meta.time_exit_config

    def test_max_indicator_params(self, meta: StrategyMeta) -> None:
        # ichimoku or stochastic have 3 params — max should be >= 3
        assert meta.max_indicator_params >= 3

    def test_max_indicator_outputs(self, meta: StrategyMeta) -> None:
        # bollinger/keltner have 5 outputs
        assert meta.max_indicator_outputs >= 4


# ---------------------------------------------------------------------------
# random_genome
# ---------------------------------------------------------------------------


class TestRandomGenome:
    """Tests for random genome generation."""

    def test_returns_strategy_genome(self, meta: StrategyMeta) -> None:
        g = random_genome(meta)
        assert isinstance(g, StrategyGenome)

    def test_correct_entry_slot_count(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        assert len(genome.entry_indicators) == meta.num_entry_slots

    def test_correct_indicator_exit_count(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        assert len(genome.indicator_exits) == meta.num_indicator_exit_slots

    def test_correct_filter_count(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        assert len(genome.filters) == meta.num_filter_slots

    def test_at_least_one_entry_enabled(self, meta: StrategyMeta) -> None:
        for seed in range(20):
            g = random_genome(meta, rng=random.Random(seed))
            assert any(e.enabled for e in g.entry_indicators), f"Seed {seed}: no entry enabled"

    def test_at_least_one_exit_enabled(self, meta: StrategyMeta) -> None:
        for seed in range(20):
            g = random_genome(meta, rng=random.Random(seed))
            has_exit = (
                g.limit_exits.stop_loss.enabled
                or g.limit_exits.take_profit.enabled
                or g.limit_exits.trailing_stop.enabled
                or g.limit_exits.chandelier.enabled
                or g.limit_exits.breakeven.enabled
                or any(ie.enabled for ie in g.indicator_exits)
                or g.time_exits.max_days_enabled
                or g.time_exits.weekday_exit_enabled
                or g.time_exits.eow_enabled
                or g.time_exits.eom_enabled
                or g.time_exits.stagnation_enabled
            )
            assert has_exit, f"Seed {seed}: no exit enabled"

    def test_combination_logic_valid(self, meta: StrategyMeta) -> None:
        for seed in range(20):
            g = random_genome(meta, rng=random.Random(seed))
            assert g.combination_logic in ("AND", "OR")

    def test_params_within_range(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        for gene in genome.entry_indicators:
            if not gene.enabled:
                continue
            param_metas = meta.indicator_params.get(gene.indicator_name, ())
            for pm in param_metas:
                if pm.name in gene.params:
                    assert pm.min_value <= gene.params[pm.name] <= pm.max_value

    def test_deterministic_with_seed(self, meta: StrategyMeta) -> None:
        g1 = random_genome(meta, rng=random.Random(123))
        g2 = random_genome(meta, rng=random.Random(123))
        assert g1 == g2

    def test_different_seeds_differ(self, meta: StrategyMeta) -> None:
        g1 = random_genome(meta, rng=random.Random(1))
        g2 = random_genome(meta, rng=random.Random(2))
        assert g1 != g2

    def test_is_valid(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        assert validate_genome(genome, meta)


# ---------------------------------------------------------------------------
# validate_genome
# ---------------------------------------------------------------------------


class TestValidateGenome:
    """Tests for genome validation."""

    def test_valid_genome_passes(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        assert validate_genome(genome, meta)

    def test_unknown_indicator_rejected(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        bad_entry = IndicatorGene(
            enabled=True,
            indicator_name="nonexistent",
            output_key="x",
            comparison="GT",
            threshold=0.0,
        )
        bad = StrategyGenome(
            entry_indicators=(bad_entry,),
            combination_logic=genome.combination_logic,
            limit_exits=genome.limit_exits,
            indicator_exits=genome.indicator_exits,
            time_exits=genome.time_exits,
            filters=genome.filters,
        )
        assert not validate_genome(bad, meta)

    def test_invalid_output_key_rejected(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        bad_entry = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="nonexistent",
            comparison="GT",
            threshold=30.0,
        )
        bad = StrategyGenome(
            entry_indicators=(bad_entry,),
            combination_logic=genome.combination_logic,
            limit_exits=genome.limit_exits,
            indicator_exits=genome.indicator_exits,
            time_exits=genome.time_exits,
            filters=genome.filters,
        )
        assert not validate_genome(bad, meta)

    def test_out_of_range_params_rejected(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        bad_entry = IndicatorGene(
            enabled=True,
            indicator_name="rsi",
            output_key="rsi",
            comparison="GT",
            threshold=30.0,
            params={"period": 999},
        )
        bad = StrategyGenome(
            entry_indicators=(bad_entry,),
            combination_logic=genome.combination_logic,
            limit_exits=genome.limit_exits,
            indicator_exits=genome.indicator_exits,
            time_exits=genome.time_exits,
            filters=genome.filters,
        )
        assert not validate_genome(bad, meta)

    def test_no_entry_enabled_rejected(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        disabled = tuple(
            IndicatorGene(
                enabled=False,
                indicator_name=g.indicator_name,
                output_key=g.output_key,
                comparison=g.comparison,
                threshold=g.threshold,
                params=g.params,
            )
            for g in genome.entry_indicators
        )
        bad = StrategyGenome(
            entry_indicators=disabled,
            combination_logic=genome.combination_logic,
            limit_exits=genome.limit_exits,
            indicator_exits=genome.indicator_exits,
            time_exits=genome.time_exits,
            filters=genome.filters,
        )
        assert not validate_genome(bad, meta)

    def test_no_exit_enabled_rejected(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        no_limit = LimitExitGene(
            stop_loss=StopLossConfig(enabled=False, mode="percent", percent=3.0, atr_multiple=2.0),
            take_profit=TakeProfitConfig(enabled=False, mode="percent", percent=5.0, atr_multiple=3.0),
            trailing_stop=TrailingStopConfig(
                enabled=False,
                mode="percent",
                percent=2.0,
                atr_multiple=2.0,
                activation_percent=2.0,
            ),
            chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
            breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
        )
        no_ind_exit = tuple(
            IndicatorExitGene(
                enabled=False,
                indicator_name=g.indicator_name,
                output_key=g.output_key,
                comparison=g.comparison,
                threshold=g.threshold,
                params=g.params,
            )
            for g in genome.indicator_exits
        )
        no_time = TimeExitGene(
            max_days_enabled=False,
            max_days=10,
            weekday_exit_enabled=False,
            weekday=4,
            eow_enabled=False,
            eom_enabled=False,
            stagnation_enabled=False,
            stagnation_days=5,
            stagnation_threshold=1.0,
        )
        bad = StrategyGenome(
            entry_indicators=genome.entry_indicators,
            combination_logic=genome.combination_logic,
            limit_exits=no_limit,
            indicator_exits=no_ind_exit,
            time_exits=no_time,
            filters=genome.filters,
        )
        assert not validate_genome(bad, meta)

    def test_invalid_combination_logic_rejected(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        bad = StrategyGenome(
            entry_indicators=genome.entry_indicators,
            combination_logic="XOR",
            limit_exits=genome.limit_exits,
            indicator_exits=genome.indicator_exits,
            time_exits=genome.time_exits,
            filters=genome.filters,
        )
        assert not validate_genome(bad, meta)


# ---------------------------------------------------------------------------
# genome_to_flat / flat_to_genome round-trip
# ---------------------------------------------------------------------------


class TestFlatRoundTrip:
    """Tests for flat chromosome serialization round-trip."""

    def test_round_trip_deterministic(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        flat = genome_to_flat(genome, meta)
        reconstructed = flat_to_genome(flat, meta)
        assert reconstructed == genome

    def test_length_matches_genome_length(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        flat = genome_to_flat(genome, meta)
        assert len(flat) == genome_length(meta)

    def test_all_values_are_floats(self, genome: StrategyGenome, meta: StrategyMeta) -> None:
        flat = genome_to_flat(genome, meta)
        assert all(isinstance(v, float) for v in flat)

    def test_genome_length_value(self, meta: StrategyMeta) -> None:
        length = genome_length(meta)
        # With 4 entry slots, 1 exit slot, 2 filter slots — should be ~80-100
        assert 60 <= length <= 120

    def test_10_random_genomes_round_trip(self, meta: StrategyMeta) -> None:
        for seed in range(10):
            g = random_genome(meta, rng=random.Random(seed))
            flat = genome_to_flat(g, meta)
            g2 = flat_to_genome(flat, meta)
            assert g == g2, f"Round-trip failed for seed {seed}"

    def test_50_random_genomes_round_trip(self, meta: StrategyMeta) -> None:
        for seed in range(50):
            g = random_genome(meta, rng=random.Random(seed + 100))
            flat = genome_to_flat(g, meta)
            g2 = flat_to_genome(flat, meta)
            assert g == g2, f"Round-trip failed for seed {seed + 100}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_limit_exits() -> LimitExitGene:
    """Create a minimal LimitExitGene for testing."""
    return LimitExitGene(
        stop_loss=StopLossConfig(enabled=True, mode="percent", percent=3.0, atr_multiple=2.0),
        take_profit=TakeProfitConfig(enabled=False, mode="atr", percent=5.0, atr_multiple=3.0),
        trailing_stop=TrailingStopConfig(
            enabled=False,
            mode="percent",
            percent=2.0,
            atr_multiple=2.0,
            activation_percent=2.0,
        ),
        chandelier=ChandelierConfig(enabled=False, atr_multiple=3.0),
        breakeven=BreakevenConfig(enabled=False, mode="percent", trigger_percent=2.0, trigger_atr_multiple=1.5),
    )
