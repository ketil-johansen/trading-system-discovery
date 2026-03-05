"""Genetic algorithm engine using DEAP.

Evolves strategy genomes using tournament selection, slot-level crossover,
Gaussian mutation, and elitism. Supports parallel evaluation, checkpointing,
and early stopping.
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import pickle
import random
import statistics
import time
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from deap import base, creator, tools

from tsd.config import env_float, env_int
from tsd.optimization.fitness import FitnessConfig, compute_fitness
from tsd.optimization.metrics import aggregate_metrics
from tsd.strategy.evaluator import (
    BacktestResult,
    EvaluatorConfig,
    run_backtest,
)
from tsd.strategy.genome import (
    OutputMeta,
    StrategyGenome,
    StrategyMeta,
    flat_to_genome,
    genome_to_flat,
    random_genome,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GAConfig:
    """Configuration for the genetic algorithm engine.

    Attributes:
        population_size: Number of individuals in each generation.
        max_generations: Maximum number of generations to run.
        crossover_prob: Probability of swapping each segment during crossover.
        mutation_prob: Per-gene probability of mutation.
        mutation_sigma: Standard deviation of Gaussian noise for mutation.
        elitism_pct: Fraction of population preserved as elites.
        tournament_size: Number of individuals in tournament selection.
        random_seed: Random seed for reproducibility.
        early_stop_generations: Stop if best fitness unchanged for this many generations.
        n_workers: Number of parallel workers for evaluation.
        checkpoint_dir: Directory for saving checkpoints.
    """

    population_size: int = 300
    max_generations: int = 100
    crossover_prob: float = 0.5
    mutation_prob: float = 0.2
    mutation_sigma: float = 0.1
    elitism_pct: float = 0.05
    tournament_size: int = 3
    random_seed: int = 42
    early_stop_generations: int = 20
    n_workers: int = 1
    checkpoint_dir: Path = field(default_factory=lambda: Path("results/checkpoints"))


@dataclass(frozen=True)
class GenerationStats:
    """Statistics for a single generation.

    Attributes:
        generation: Generation number.
        best_fitness: Best fitness in the generation.
        worst_fitness: Worst fitness in the generation.
        avg_fitness: Average fitness in the generation.
        std_fitness: Standard deviation of fitness values.
        diversity: Fraction of unique fitness values.
    """

    generation: int
    best_fitness: float
    worst_fitness: float
    avg_fitness: float
    std_fitness: float
    diversity: float


@dataclass(frozen=True)
class GAResult:
    """Result of a GA optimization run.

    Attributes:
        best_genome: Best strategy genome found.
        best_fitness: Fitness of the best genome.
        generations_run: Number of generations completed.
        logbook: Statistics for each generation.
        top_genomes: Top N genomes from the hall of fame with fitness values.
    """

    best_genome: StrategyGenome
    best_fitness: float
    generations_run: int
    logbook: tuple[GenerationStats, ...]
    top_genomes: tuple[tuple[StrategyGenome, float], ...] = ()


def load_ga_config() -> GAConfig:
    """Load GA configuration from environment variables.

    Returns:
        GAConfig with values from TSD_GA_* environment variables.
    """
    return GAConfig(
        population_size=env_int("TSD_GA_POPULATION_SIZE", 300),
        max_generations=env_int("TSD_GA_MAX_GENERATIONS", 100),
        crossover_prob=env_float("TSD_GA_CROSSOVER_PROB", 0.5),
        mutation_prob=env_float("TSD_GA_MUTATION_PROB", 0.2),
        mutation_sigma=env_float("TSD_GA_MUTATION_SIGMA", 0.1),
        elitism_pct=env_float("TSD_GA_ELITISM_PCT", 0.05),
        tournament_size=env_int("TSD_GA_TOURNAMENT_SIZE", 3),
        random_seed=env_int("TSD_GA_RANDOM_SEED", 42),
        early_stop_generations=env_int("TSD_GA_EARLY_STOP_GENERATIONS", 20),
        n_workers=env_int("TSD_GA_N_WORKERS", 1),
        checkpoint_dir=Path(env_int.__module__ and "results/checkpoints"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ga(
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    ga_config: GAConfig | None = None,
    eval_config: EvaluatorConfig | None = None,
    fitness_config: FitnessConfig | None = None,
    resume: bool = False,
) -> GAResult:
    """Run the genetic algorithm to evolve trading strategies.

    Args:
        meta: Strategy metadata describing the parameter space.
        stocks_data: Mapping of stock ticker to OHLCV DataFrame.
        indicator_outputs: Mapping of indicator name to output metadata.
        ga_config: GA configuration. Uses defaults if None.
        eval_config: Evaluator configuration. Uses defaults if None.
        fitness_config: Fitness configuration. Uses defaults if None.
        resume: Whether to resume from a checkpoint.

    Returns:
        GAResult with the best genome, fitness, and generation log.
    """
    ga_config = ga_config or GAConfig()
    eval_config = eval_config or EvaluatorConfig()
    fitness_config = fitness_config or FitnessConfig()

    random.seed(ga_config.random_seed)
    np.random.seed(ga_config.random_seed)  # noqa: NPY002

    toolbox = _setup_deap(meta, ga_config)
    toolbox.register(
        "evaluate",
        _evaluate_individual,
        meta=meta,
        stocks_data=stocks_data,
        indicator_outputs=indicator_outputs,
        eval_config=eval_config,
        fitness_config=fitness_config,
    )

    pop_size = ga_config.population_size
    n_elites = max(1, int(pop_size * ga_config.elitism_pct))
    hof = tools.HallOfFame(n_elites)

    population, start_gen, logbook, hof = _init_population(
        toolbox,
        ga_config,
        hof,
        resume,
    )

    pool: Pool | None = None
    counter: Any = None
    total_val: Any = None
    if ga_config.n_workers > 1:
        counter = multiprocessing.Value(ctypes.c_int, 0)
        total_val = multiprocessing.Value(ctypes.c_int, len(population))
        pool = Pool(  # noqa: SIM115
            ga_config.n_workers,
            initializer=_init_worker,
            initargs=(counter, total_val),
        )
        toolbox.register("map", pool.map)
        LOGGER.info("Parallel evaluation: %d workers", ga_config.n_workers)
    else:
        LOGGER.info("Sequential evaluation (n_workers=1)")

    try:
        LOGGER.info("Evaluating initial population (%d individuals)...", len(population))
        init_t0 = time.monotonic()
        _assign_fitness(toolbox, population)
        hof.update(population)
        stats = _compute_generation_stats(0, population)
        logbook.append(stats)
        init_elapsed = time.monotonic() - init_t0
        _log_generation(stats, init_elapsed, len(population))

        best_fitness, logbook = _evolve_loop(
            toolbox,
            population,
            hof,
            meta,
            ga_config,
            stats.best_fitness,
            start_gen,
            logbook,
            counter,
            total_val,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    best_ind = hof[0]
    best_genome = flat_to_genome(list(best_ind), meta)
    generations_run = logbook[-1].generation + 1 if logbook else 0

    top_genomes = tuple((flat_to_genome(list(ind), meta), ind.fitness.values[0]) for ind in hof)
    LOGGER.info("Hall of fame: %d strategies saved", len(top_genomes))

    return GAResult(
        best_genome=best_genome,
        best_fitness=best_ind.fitness.values[0],
        generations_run=generations_run,
        logbook=tuple(logbook),
        top_genomes=top_genomes,
    )


def _init_population(
    toolbox: base.Toolbox,
    ga_config: GAConfig,
    hof: tools.HallOfFame,
    resume: bool,
) -> tuple[list[Any], int, list[GenerationStats], tools.HallOfFame]:
    """Load checkpoint or create a fresh population.

    Args:
        toolbox: Configured DEAP toolbox.
        ga_config: GA configuration.
        hof: Hall of fame instance.
        resume: Whether to attempt checkpoint resume.

    Returns:
        Tuple of (population, start_gen, logbook, hof).
    """
    checkpoint_path = ga_config.checkpoint_dir / "ga_checkpoint.pkl"
    if resume and checkpoint_path.exists():
        cp = _load_checkpoint(checkpoint_path)
        random.setstate(cp["rng_state"])
        LOGGER.info("Resumed from checkpoint at generation %d", cp["generation"])
        return cp["population"], cp["generation"] + 1, list(cp["logbook"]), cp["halloffame"]
    return toolbox.population(n=ga_config.population_size), 0, [], hof


def _assign_fitness(toolbox: base.Toolbox, population: list[Any]) -> None:
    """Evaluate and assign fitness to all individuals.

    Args:
        toolbox: DEAP toolbox with evaluate and map registered.
        population: List of individuals to evaluate.
    """
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses, strict=True):
        ind.fitness.values = fit


def _evolve_loop(  # noqa: PLR0913
    toolbox: base.Toolbox,
    population: list[Any],
    hof: tools.HallOfFame,
    meta: StrategyMeta,
    ga_config: GAConfig,
    best_fitness: float,
    start_gen: int,
    logbook: list[GenerationStats],
    counter: Any = None,
    total_val: Any = None,
) -> tuple[float, list[GenerationStats]]:
    """Run the main generational loop.

    Args:
        toolbox: Configured DEAP toolbox.
        population: Current population (mutated in place).
        hof: Hall of fame.
        meta: Strategy metadata.
        ga_config: GA configuration.
        best_fitness: Best fitness seen so far.
        start_gen: Generation to start from.
        logbook: Existing logbook entries.
        counter: Shared progress counter for parallel evaluation.
        total_val: Shared total for current evaluation batch.

    Returns:
        Tuple of (best_fitness, logbook).
    """
    pop_size = ga_config.population_size
    n_elites = max(1, int(pop_size * ga_config.elitism_pct))
    checkpoint_path = ga_config.checkpoint_dir / "ga_checkpoint.pkl"
    stale_count = 0

    for gen in range(max(1, start_gen), ga_config.max_generations):
        gen_t0 = time.monotonic()
        offspring = _breed_generation(toolbox, population, meta, ga_config, pop_size - n_elites)

        invalids = [ind for ind in offspring if not ind.fitness.valid]
        if counter is not None and total_val is not None:
            _reset_eval_progress(counter, total_val, len(invalids))
        _assign_fitness(toolbox, invalids)

        elites = [toolbox.clone(ind) for ind in hof]
        population[:] = offspring + elites
        hof.update(population)

        stats = _compute_generation_stats(gen, population)
        logbook.append(stats)
        gen_elapsed = time.monotonic() - gen_t0
        _log_generation(stats, gen_elapsed, len(invalids))

        _save_checkpoint(checkpoint_path, population, gen, hof, logbook, random.getstate())

        if stats.best_fitness > best_fitness:
            best_fitness = stats.best_fitness
            stale_count = 0
        else:
            stale_count += 1

        if stale_count >= ga_config.early_stop_generations:
            LOGGER.info(
                "Early stopping at generation %d (no improvement for %d generations)",
                gen,
                ga_config.early_stop_generations,
            )
            break

    return best_fitness, logbook


def _breed_generation(
    toolbox: base.Toolbox,
    population: list[Any],
    meta: StrategyMeta,
    ga_config: GAConfig,
    n_offspring: int,
) -> list[Any]:
    """Select, crossover, and mutate to produce offspring.

    Args:
        toolbox: DEAP toolbox.
        population: Current population.
        meta: Strategy metadata.
        ga_config: GA configuration.
        n_offspring: Number of offspring to produce.

    Returns:
        List of offspring individuals.
    """
    offspring = toolbox.select(population, n_offspring)
    offspring = [toolbox.clone(ind) for ind in offspring]

    for i in range(0, len(offspring) - 1, 2):
        offspring[i], offspring[i + 1] = _slot_crossover(
            offspring[i],
            offspring[i + 1],
            meta,
            ga_config.crossover_prob,
        )
        del offspring[i].fitness.values
        del offspring[i + 1].fitness.values

    for i in range(len(offspring)):
        (offspring[i],) = _gene_mutation(
            offspring[i],
            meta,
            ga_config.mutation_prob,
            ga_config.mutation_sigma,
        )
        del offspring[i].fitness.values

    return offspring


# ---------------------------------------------------------------------------
# DEAP setup
# ---------------------------------------------------------------------------


def _setup_deap(meta: StrategyMeta, ga_config: GAConfig) -> base.Toolbox:
    """Configure DEAP types and operators.

    Args:
        meta: Strategy metadata.
        ga_config: GA configuration.

    Returns:
        Configured DEAP Toolbox.
    """
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    rng = random.Random(ga_config.random_seed)  # noqa: S311

    toolbox.register(
        "individual",
        _create_individual,
        meta=meta,
        rng=rng,
        ind_cls=creator.Individual,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=ga_config.tournament_size)
    toolbox.register("map", map)

    return toolbox


def _create_individual(
    meta: StrategyMeta,
    rng: random.Random,
    ind_cls: type,
) -> Any:
    """Create a DEAP individual from a random genome.

    Args:
        meta: Strategy metadata.
        rng: Random number generator.
        ind_cls: DEAP Individual class.

    Returns:
        DEAP Individual wrapping a flat chromosome.
    """
    genome = random_genome(meta, rng)
    flat = genome_to_flat(genome, meta)
    return ind_cls(flat)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------


def _compute_segment_boundaries(meta: StrategyMeta) -> list[tuple[int, int]]:
    """Compute swap boundaries for slot-level crossover.

    Returns a list of (start, end) pairs, one per swappable unit:
    - N entry slots
    - 1 combination logic gene
    - 1 limit exits block (19 genes)
    - M indicator exit slots
    - 1 time exits block (9 genes)
    - K filter slots

    Args:
        meta: Strategy metadata.

    Returns:
        List of (start, end) tuples.
    """
    entry_width = 5 + meta.max_indicator_params
    exit_width = 6 + meta.max_indicator_params
    filter_width = 2 + meta.max_filter_params

    boundaries: list[tuple[int, int]] = []
    pos = 0

    # Entry slots
    for _ in range(meta.num_entry_slots):
        boundaries.append((pos, pos + entry_width))
        pos += entry_width

    # Combination logic
    boundaries.append((pos, pos + 1))
    pos += 1

    # Limit exits block
    boundaries.append((pos, pos + 19))
    pos += 19

    # Indicator exit slots
    for _ in range(meta.num_indicator_exit_slots):
        boundaries.append((pos, pos + exit_width))
        pos += exit_width

    # Time exits block
    boundaries.append((pos, pos + 9))
    pos += 9

    # Filter slots
    for _ in range(meta.num_filter_slots):
        boundaries.append((pos, pos + filter_width))
        pos += filter_width

    return boundaries


def _slot_crossover(
    ind1: list[float],
    ind2: list[float],
    meta: StrategyMeta,
    prob: float,
) -> tuple[list[float], list[float]]:
    """Swap whole genome segments between two individuals.

    For each segment boundary, swap the slice between parents with
    probability `prob`.

    Args:
        ind1: First parent individual.
        ind2: Second parent individual.
        meta: Strategy metadata.
        prob: Probability of swapping each segment.

    Returns:
        Tuple of two offspring individuals.
    """
    boundaries = _compute_segment_boundaries(meta)
    for start, end in boundaries:
        if random.random() < prob:
            ind1[start:end], ind2[start:end] = ind2[start:end], ind1[start:end]
    return ind1, ind2


def _gene_mutation(
    individual: list[float],
    meta: StrategyMeta,
    prob: float,
    sigma: float,
) -> tuple[list[float]]:
    """Apply Gaussian noise to individual genes.

    For each gene position, with probability `prob`, add noise drawn
    from N(0, sigma). Clamping is handled by flat_to_genome during
    decoding.

    Args:
        individual: The individual to mutate.
        meta: Strategy metadata (unused but kept for consistency).
        prob: Per-gene mutation probability.
        sigma: Standard deviation of Gaussian noise.

    Returns:
        Tuple containing the mutated individual (DEAP convention).
    """
    for i in range(len(individual)):
        if random.random() < prob:
            individual[i] += random.gauss(0, sigma)
    return (individual,)


# ---------------------------------------------------------------------------
# Shared progress counter for parallel evaluation
# ---------------------------------------------------------------------------

_eval_counter: Any = None
_eval_total: Any = None


def _init_worker(
    counter: Any,
    total: Any,
) -> None:
    """Initialize worker process with shared counter.

    Args:
        counter: Shared integer counter for completed evaluations.
        total: Shared integer for total evaluations in current batch.
    """
    global _eval_counter, _eval_total  # noqa: PLW0603
    _eval_counter = counter
    _eval_total = total


def _reset_eval_progress(
    counter: Any,
    total_val: Any,
    n: int,
) -> None:
    """Reset shared progress counter before a new evaluation batch.

    Args:
        counter: Shared counter to reset to 0.
        total_val: Shared total to set to n.
        n: Number of individuals in this batch.
    """
    with counter.get_lock():
        counter.value = 0
    with total_val.get_lock():
        total_val.value = n


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_individual(
    individual: list[float],
    meta: StrategyMeta,
    stocks_data: dict[str, pd.DataFrame],
    indicator_outputs: dict[str, tuple[OutputMeta, ...]],
    eval_config: EvaluatorConfig,
    fitness_config: FitnessConfig,
) -> tuple[float]:
    """Evaluate a single individual across all stocks.

    Decodes the flat chromosome, runs backtests per stock, aggregates
    metrics, and computes fitness.

    Args:
        individual: Flat chromosome values.
        meta: Strategy metadata.
        stocks_data: Mapping of ticker to OHLCV DataFrame.
        indicator_outputs: Mapping of indicator name to output metadata.
        eval_config: Evaluator configuration.
        fitness_config: Fitness configuration.

    Returns:
        Single-element tuple with fitness value.
    """
    genome = flat_to_genome(list(individual), meta)

    results: list[BacktestResult] = []
    for _ticker, df in stocks_data.items():
        try:
            result = run_backtest(genome, df, indicator_outputs, eval_config)
            results.append(result)
        except Exception:  # noqa: BLE001
            LOGGER.warning("Backtest failed for a stock, skipping")

    if not results:
        return (0.0,)

    aggregated = aggregate_metrics(results)
    fitness = compute_fitness(aggregated, fitness_config)

    if _eval_counter is not None and _eval_total is not None:
        with _eval_counter.get_lock():
            _eval_counter.value += 1
            done = _eval_counter.value
        total = _eval_total.value
        if total > 0 and (done % max(1, total // 10) == 0 or done == total):
            LOGGER.info("  eval %d/%d complete (fitness=%.4f)", done, total, fitness)

    return (fitness,)


# ---------------------------------------------------------------------------
# Generation stats
# ---------------------------------------------------------------------------


def _compute_generation_stats(generation: int, population: list[Any]) -> GenerationStats:
    """Compute statistics for a generation.

    Args:
        generation: Generation number.
        population: List of DEAP individuals.

    Returns:
        GenerationStats for the generation.
    """
    fits = [ind.fitness.values[0] for ind in population]
    unique_fits = len(set(fits))
    pop_size = len(population)

    return GenerationStats(
        generation=generation,
        best_fitness=max(fits),
        worst_fitness=min(fits),
        avg_fitness=statistics.mean(fits),
        std_fitness=statistics.stdev(fits) if len(fits) > 1 else 0.0,
        diversity=unique_fits / pop_size if pop_size > 0 else 0.0,
    )


def _log_generation(stats: GenerationStats, elapsed: float, n_evaluated: int) -> None:
    """Log generation statistics.

    Args:
        stats: Generation statistics to log.
        elapsed: Seconds elapsed for this generation.
        n_evaluated: Number of individuals evaluated.
    """
    LOGGER.info(
        "Gen %03d | best=%.4f avg=%.4f std=%.4f | evals=%d %.1fs",
        stats.generation,
        stats.best_fitness,
        stats.avg_fitness,
        stats.std_fitness,
        n_evaluated,
        elapsed,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: Path,
    population: list[Any],
    generation: int,
    halloffame: tools.HallOfFame,
    logbook: list[GenerationStats],
    rng_state: Any,
) -> None:
    """Save GA state to a checkpoint file.

    Args:
        path: Path to checkpoint file.
        population: Current population.
        generation: Current generation number.
        halloffame: DEAP HallOfFame.
        logbook: List of GenerationStats.
        rng_state: Random module state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "population": population,
        "generation": generation,
        "halloffame": halloffame,
        "logbook": logbook,
        "rng_state": rng_state,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load GA state from a checkpoint file.

    Args:
        path: Path to checkpoint file.

    Returns:
        Dictionary with population, generation, halloffame, logbook, rng_state.
    """
    with open(path, "rb") as f:
        result: dict[str, Any] = pickle.load(f)  # noqa: S301
    return result
