"""
ScenarioRunner / SensitivityAnalyzer - Final Design Section 9. Both are
compositions over the same SimulationEngine; neither is a separate pipeline.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Callable

from .controllers import Controller, IdentityController
from .engine import DynamicSimulationEngine, StepResult, initial_state
from .parameters import ParameterSet
from .pricing import PricingStrategy
from .randomness import DeterministicProvider, RandomnessProvider


@dataclass
class ScenarioSpec:
    name: str
    overrides: dict = field(default_factory=dict)  # e.g. {"price": 14400.0}
    marketing_controller: Controller | None = None
    capacity_controller: Controller | None = None
    price_controller: Controller | None = None


@dataclass
class ScenarioResult:
    name: str
    history: list[StepResult]


class ScenarioRunner:
    """
    Runs one or more ScenarioSpecs through an identical DynamicSimulationEngine
    configuration, differing only in initial lever overrides and/or which
    Controller each lever uses (open-loop IdentityController by default,
    matching every Phase 9 scenario; closed-loop when a real Controller is
    supplied).
    """

    def __init__(
        self,
        params: ParameterSet,
        pricing_factory: Callable[[ParameterSet], PricingStrategy],
        provider: RandomnessProvider | None = None,
        enable_backlog: bool = True,
        enable_referral: bool = True,
    ):
        self.params = params
        self.pricing_factory = pricing_factory
        self.provider = provider or DeterministicProvider()
        self.enable_backlog = enable_backlog
        self.enable_referral = enable_referral

    def run_scenario(
        self,
        spec: ScenarioSpec,
        existing_active_base: float,
        base_marketing_budget: float,
        base_price: float,
        base_capacity: float,
        horizon: int,
    ) -> ScenarioResult:
        """
        All scenarios share one observed month-0 snapshot (Final Design,
        Phase 9: "All four start from the identical Phase 4/5 baseline at
        month 0") and diverge only from month 1 onward, once the scenario's
        lever change has actually taken effect. Concretely: month 0 is
        computed with the BASE lever values (identical across every
        scenario, reproducing the shared Active(0)=158 baseline exactly),
        and the scenario's overrides are then substituted as the lever
        values that Controllers carry forward into month 1..horizon. This
        avoids retroactively applying a decision (e.g. a price increase) to
        a month that was already observed under the old price.
        """
        params = copy.deepcopy(self.params)
        pricing = self.pricing_factory(params)

        marketing_budget = spec.overrides.get("marketing_budget", base_marketing_budget)
        price = spec.overrides.get("price", base_price)
        capacity = spec.overrides.get("capacity", base_capacity)

        engine = DynamicSimulationEngine(
            params=params, pricing=pricing, provider=self.provider,
            marketing_controller=spec.marketing_controller or IdentityController(),
            capacity_controller=spec.capacity_controller or IdentityController(),
            price_controller=spec.price_controller or IdentityController(),
            enable_backlog=self.enable_backlog, enable_referral=self.enable_referral,
        )

        # Month 0: shared baseline levers, identical across all scenarios.
        baseline_seed = initial_state(existing_active_base, base_marketing_budget, base_price, base_capacity)
        month_zero = engine._compute_step(baseline_seed, is_first_step=True)

        # From month 1 onward: substitute this scenario's lever values as
        # the "current" values Controllers carry forward (IdentityController
        # just returns current_value unchanged, so this is what makes the
        # override actually take hold starting next month).
        month_zero_adjusted = replace(
            month_zero, marketing_budget=marketing_budget, price=price, capacity=capacity,
        )

        history = [month_zero]
        current = month_zero_adjusted
        for _ in range(horizon):
            current = engine._compute_step(current, is_first_step=False)
            history.append(current)
        return ScenarioResult(name=spec.name, history=history)

    def compare(self, specs: list[ScenarioSpec], **kwargs) -> dict[str, ScenarioResult]:
        return {spec.name: self.run_scenario(spec, **kwargs) for spec in specs}


class SensitivityAnalyzer:
    """
    One-at-a-Time (OAT) sensitivity sweep (Final Design Section 9 / Task 7's
    Validation Framework) - composition over ScenarioRunner, zero
    optimization or AI content. Determines whether a scenario ranking is
    robust or an artifact of one illustrative default.
    """

    def __init__(self, runner: ScenarioRunner):
        self.runner = runner

    def sweep(
        self,
        parameter_name: str,
        low_high_steps: tuple[float, float, int],
        existing_active_base: float,
        base_marketing_budget: float,
        base_price: float,
        base_capacity: float,
        horizon: int,
        output_fn: Callable[[list[StepResult]], float],
    ) -> list[tuple[float, float]]:
        """Returns [(parameter_value, output_value), ...] across the swept range."""
        low, high, steps = low_high_steps
        results = []
        for i in range(steps):
            value = low + (high - low) * i / max(1, steps - 1)
            params = copy.deepcopy(self.runner.params)
            getattr(params, parameter_name).value = value
            trial_runner = ScenarioRunner(
                params, self.runner.pricing_factory, self.runner.provider,
                self.runner.enable_backlog, self.runner.enable_referral,
            )
            spec = ScenarioSpec(name=f"{parameter_name}={value:.4f}")
            result = trial_runner.run_scenario(
                spec, existing_active_base, base_marketing_budget, base_price,
                base_capacity, horizon,
            )
            results.append((value, output_fn(result.history)))
        return results

    def sweep_2d(
        self,
        param_x: str,
        range_x: tuple[float, float, int],
        param_y: str,
        range_y: tuple[float, float, int],
        existing_active_base: float,
        base_marketing_budget: float,
        base_price: float,
        base_capacity: float,
        horizon: int,
        output_fn: Callable[[list[StepResult]], float],
    ) -> list[tuple[float, float, float]]:
        """
        Two-parameter interaction sweep - answers a question the one-at-a-
        time sweep() above cannot: not just "how much does X move the
        output", but "does X's effect on the output change depending on Y".
        Runs the full grid of param_x times param_y combinations and
        records the output at every (x, y) cell.

        Interaction is present when X's marginal effect (the output's
        change across the X range) differs across different fixed values of
        Y; if that marginal effect is the same at every Y, X and Y move the
        output independently and additively - no true interaction, just two
        separate one-at-a-time effects layered on top of each other.
        """
        low_x, high_x, steps_x = range_x
        low_y, high_y, steps_y = range_y
        results = []
        for j in range(steps_y):
            y_value = low_y + (high_y - low_y) * j / max(1, steps_y - 1)
            for i in range(steps_x):
                x_value = low_x + (high_x - low_x) * i / max(1, steps_x - 1)
                params = copy.deepcopy(self.runner.params)
                getattr(params, param_x).value = x_value
                getattr(params, param_y).value = y_value
                trial_runner = ScenarioRunner(
                    params, self.runner.pricing_factory, self.runner.provider,
                    self.runner.enable_backlog, self.runner.enable_referral,
                )
                spec = ScenarioSpec(name=f"{param_x}={x_value:.4f},{param_y}={y_value:.4f}")
                result = trial_runner.run_scenario(
                    spec, existing_active_base, base_marketing_budget, base_price,
                    base_capacity, horizon,
                )
                results.append((x_value, y_value, output_fn(result.history)))
        return results

    @staticmethod
    def interaction_strength(results: list[tuple[float, float, float]]) -> float:
        """
        A simple, honest interaction diagnostic over a sweep_2d() grid: the
        spread (max - min) of param_x's marginal effect (output at X-high
        minus output at X-low), measured separately at each fixed Y value.
        Near zero = no interaction (X's effect on the output doesn't depend
        on Y). Large relative to the output's own scale = a real
        interaction - X and Y's combined effect is not just the sum of
        their two separate one-at-a-time effects.
        """
        by_y: dict[float, list[tuple[float, float]]] = {}
        for x, y, out in results:
            by_y.setdefault(y, []).append((x, out))
        x_effects = []
        for y, xs in by_y.items():
            xs.sort(key=lambda t: t[0])
            x_effects.append(xs[-1][1] - xs[0][1])
        return (max(x_effects) - min(x_effects)) if x_effects else 0.0
