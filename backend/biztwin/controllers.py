"""
Controllers - first-class policy objects (Final Design, Section 9 Scenario
Engine Design / Architectural Reframing). A Controller observes State(t) and
produces Inputs(t+1); it is never an optimizer, only a documented
threshold/proportional rule.

Every Phase-9-style open-loop scenario is the special case where each
controller runs the identity policy (hold the input constant) - that mode is
IdentityController below, used by default so nothing already validated
breaks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Controller(ABC):
    @abstractmethod
    def next_value(self, state: dict, current_value: float) -> float:
        """Given the current system State(t) and this lever's current value,
        return its value for t+1."""


class IdentityController(Controller):
    """Open-loop: hold the input constant. This is what every Phase 9
    scenario used implicitly."""

    def next_value(self, state: dict, current_value: float) -> float:
        return current_value


@dataclass
class MarketingController(Controller):
    """
    Observes Backlog(t) and capacity utilization. If there is no backlog and
    utilization is below target, step Marketing Budget up; if oversubscribed
    (Backlog > 0), hold or step down. A threshold rule, not an optimizer
    (Final Design Section 4 of the Architectural Reframing).
    """

    step_up: float = 5_000.0
    step_down: float = 5_000.0
    target_utilization: float = 0.85

    def next_value(self, state: dict, current_value: float) -> float:
        backlog = state.get("backlog", 0.0)
        qo = state.get("qualified_opportunity", 0.0)
        capacity = state.get("capacity", 1.0)
        utilization = qo / capacity if capacity else 0.0

        if backlog > 0:
            return max(0.0, current_value - self.step_down)
        if utilization < self.target_utilization:
            return current_value + self.step_up
        return current_value


@dataclass
class CapacityController(Controller):
    """
    Observes sustained Backlog over a hysteresis window (avoids a hiring
    trigger on one noisy month) and raises Capacity once the threshold is
    crossed. Carries real memory (the consecutive-months counter) - the one
    Controller with its own state (Final Design, Task 3 Entity Catalog).
    """

    hysteresis_months: int = 3
    hiring_increment: float = 20.0
    _consecutive_backlog_months: int = 0

    def next_value(self, state: dict, current_value: float) -> float:
        backlog = state.get("backlog", 0.0)
        if backlog > 0:
            self._consecutive_backlog_months += 1
        else:
            self._consecutive_backlog_months = 0

        if self._consecutive_backlog_months >= self.hysteresis_months:
            self._consecutive_backlog_months = 0  # reset after acting
            return current_value + self.hiring_increment
        return current_value


@dataclass
class PricingController(Controller):
    """
    Open-loop by design (Final Design, Section 4 of the Architectural
    Reframing): Price is treated as a slow, strategic decision, not a
    monthly feedback lever. This is a stated modeling choice, not an
    oversight - a scenario may still override Price directly.
    """

    def next_value(self, state: dict, current_value: float) -> float:
        return current_value


@dataclass
class RevenueReinvestmentMarketingController(Controller):
    """
    Loop C - Revenue -> Marketing Budget -> Leads -> Funnel -> Revenue
    (reinforcing), added per project-lead review: the causal graph needs at
    least one genuine feedback loop whose output visibly becomes the next
    month's input, closing back on itself.

    Deliberately NOT wired through metrics.py: this controller computes its
    own "implied MRR" directly from state fields the engine already carries
    (`active`, `price`), rather than importing a metrics.py function. That
    preserves the Architectural Reframing's rule that metrics are read-only,
    emergent outputs that never feed back into state (metrics.py itself is
    never called from inside the state-transition pipeline) - Revenue Engine
    remains an aggregation layer, not a business entity with its own state;
    this controller just happens to compute the same arithmetic locally, on
    purpose, so the boundary stays intact even while demonstrating the loop.

    Off by default (IdentityController is still what every Phase 9 scenario
    and the shipped dashboard use) - this is an opt-in demonstration
    Controller, not a change to any previously-verified scenario.
    """

    reinvestment_rate: float = 0.15  # fraction of implied monthly revenue reinvested into next month's budget
    floor: float = 0.0

    def next_value(self, state: dict, current_value: float) -> float:
        active = state.get("active", 0.0)
        price = state.get("price", 0.0)
        implied_mrr = active * price / 12.0
        return max(self.floor, implied_mrr * self.reinvestment_rate)
