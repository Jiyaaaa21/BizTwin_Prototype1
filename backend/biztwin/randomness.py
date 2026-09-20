"""
RandomnessProvider - Strategy pattern (Final Design, Task 8 / Phase 8's OOP
Architecture).

This is the single mechanism that lets one simulation pipeline
(SimulationEngine._compute_step) serve as both the deterministic engine
(Phases 4-5) and the stochastic synthetic-data generator (Phase 7) - there is
only one set of formulas, never two that could drift apart.

DeterministicProvider returns exactly the expected value/probability handed
to it - this is what produces the reproducible, auditable expected-value
math in the Static/Dynamic engines and in scenario comparisons.

MonteCarloProvider draws from the distribution appropriate to what each
quantity physically is (Final Design, Section 8 - Synthetic Data Framework):
    - counts (e.g. Leads arriving)      -> Poisson
    - single yes/no outcomes            -> Bernoulli
    - a sum of many yes/no outcomes     -> Binomial (the aggregate form)
    - positive, right-skewed magnitudes -> Lognormal
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RandomnessProvider(ABC):
    """Abstract Strategy: how the engine turns an expected value into a draw."""

    @abstractmethod
    def draw_count(self, expected_value: float) -> float:
        """An arrival-count process (e.g. Leads(t))."""

    @abstractmethod
    def draw_binomial_count(self, n_trials: float, p: float) -> float:
        """The number of successes out of n_trials, each with probability p
        (e.g. how many of this month's Qualified Opportunities convert)."""

    @abstractmethod
    def draw_event(self, probability: float) -> bool:
        """A single yes/no outcome (e.g. did this one customer churn)."""

    @abstractmethod
    def draw_magnitude(self, mean: float, sigma: float = 0.4) -> float:
        """A positive, right-skewed continuous magnitude (e.g. deal size)."""


class DeterministicProvider(RandomnessProvider):
    """
    Returns exact expected values - no sampling. This is what makes the
    Static and Dynamic engines reproduce the Final Design's worked numeric
    examples (Phase 4/5/9) exactly, and what makes same-seed-independent
    reproducibility trivial: there is no seed, because there is no randomness.
    """

    def draw_count(self, expected_value: float) -> float:
        return expected_value

    def draw_binomial_count(self, n_trials: float, p: float) -> float:
        return n_trials * p

    def draw_event(self, probability: float) -> bool:
        # Expected-value engines don't resolve individual binary events;
        # callers that need a hard yes/no under determinism should instead
        # use draw_binomial_count on the relevant population.
        raise NotImplementedError(
            "DeterministicProvider has no notion of a single resolved event; "
            "use draw_binomial_count on the population instead."
        )

    def draw_magnitude(self, mean: float, sigma: float = 0.4) -> float:
        return mean


class MonteCarloProvider(RandomnessProvider):
    """
    Draws from real distributions, seeded explicitly and recorded (Final
    Design Section 8: 'every run is seeded explicitly ... so any specific
    run is exactly reproducible'). Used by SyntheticDataGenerator and by
    ScenarioRunner's uncertainty-analysis mode.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def draw_count(self, expected_value: float) -> float:
        expected_value = max(0.0, expected_value)
        return float(self._rng.poisson(expected_value))

    def draw_binomial_count(self, n_trials: float, p: float) -> float:
        n = max(0, int(round(n_trials)))
        p = min(max(p, 0.0), 1.0)
        return float(self._rng.binomial(n, p))

    def draw_event(self, probability: float) -> bool:
        return bool(self._rng.random() < probability)

    def draw_magnitude(self, mean: float, sigma: float = 0.4) -> float:
        # Lognormal parameterized so that E[X] = mean exactly, regardless of sigma.
        mu = np.log(mean) - 0.5 * sigma ** 2
        return float(self._rng.lognormal(mean=mu, sigma=sigma))
