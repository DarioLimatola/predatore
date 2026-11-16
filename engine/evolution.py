"""Configuration and entry point for the persistent V6.04 vector engine."""
from dataclasses import dataclass
from .settings import Limits


@dataclass
class Config:
    population: int = Limits.DEFAULT_POPULATION
    generations: int = Limits.DEFAULT_GENERATIONS
    episodes: int = Limits.DEFAULT_EPISODES
    steps: int = Limits.DEFAULT_STEPS
    size: int = Limits.DEFAULT_WORLD_SIZE
    seed: int = Limits.DEFAULT_SEED
    initial_neurons: int = Limits.DEFAULT_BASE_NEURONS
    max_neurons: int = Limits.DEFAULT_MAX_NEURONS
    elite: int = Limits.DEFAULT_ELITE
    validation_episodes: int = Limits.DEFAULT_VALIDATION_EPISODES
    holdout_episodes: int = Limits.DEFAULT_HOLDOUT_EPISODES
    vector_width: int = Limits.DEFAULT_WIDTH
    vector_shape: str = ''
    memory_capacity: int = Limits.DEFAULT_MEMORY
    experience_capacity: int = Limits.DEFAULT_EXPERIENCES
    internal_ticks: int = Limits.DEFAULT_TICKS
    trace_depth: int = Limits.DEFAULT_TRACE
    circuit_plasticity_enabled: bool = True
    contextual_distillation_enabled: bool = True
    minimum_active_neurons: int = Limits.MIN_NEURONS
    retention_tasks: int = Limits.DEFAULT_RETENTION_TASKS
    smith_profile: str = Limits.DEFAULT_SMITH_PROFILE
    novelty_archive_size: int = Limits.DEFAULT_NOVELTY_ARCHIVE


def run(root, config=None, resume=False, progress=None, should_stop=None,
        observer=None, monitor=None):
    config = config or Config()
    from .continual import run_vector
    return run_vector(root, config, resume=resume, progress=progress,
                      should_stop=should_stop, observer=observer, monitor=monitor)
