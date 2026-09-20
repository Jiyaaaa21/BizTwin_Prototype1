"""
FunnelState - State pattern (Final Design, Section 5 State Transition Design;
this is the concrete answer to the project lead's "State Change Techniques"
requirement).

A Customer never has its stage set directly from outside; it holds a
FunnelState object and delegates `advance()` to it. Each concrete state
carries its own entry condition, exit condition, transition rule, and
transition probability, exactly as tabulated in the Final Design's Section 5.

This module is the individual-agent (ABM) layer used by
SyntheticDataGenerator (biztwin/synthetic.py) to produce record-level
Customer/Subscription/Revenue-Event data. The aggregate engine
(biztwin/engine.py) uses the same transition rules at the population level
via RandomnessProvider, so both layers are faithful to one specification.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .pricing import PricingStrategy
from .randomness import RandomnessProvider


@dataclass
class Customer:
    """An individual agent - the unit whose state actually evolves
    (Final Design, Section 4 Entity Catalog)."""

    customer_id: str
    segment: str
    cohort_id: str  # acquisition month x segment (Decision 4)
    deal_size: float
    pricing: PricingStrategy
    months_in_queue: int = 0  # only meaningful while state is "backlog"
    tenure_months: int = 0
    state_name: str = "lead"

    def advance(self, month: int, params, provider: RandomnessProvider) -> "Customer":
        """Delegate to the current FunnelState; returns self (mutated) for chaining."""
        state = STATES[self.state_name]
        state.advance(self, month, params, provider)
        return self


class FunnelState(ABC):
    """Abstract State: entry condition, exit condition, transition rule,
    and transition probability are documented per concrete subclass,
    matching the Final Design's Section 5 table row for row."""

    name: str

    @abstractmethod
    def entry_condition(self) -> str: ...

    @abstractmethod
    def exit_condition(self) -> str: ...

    @abstractmethod
    def transition_probability(self, customer: Customer, params) -> float: ...

    @abstractmethod
    def advance(self, customer: Customer, month: int, params, provider: RandomnessProvider) -> None:
        """Mutate customer.state_name (and any state-specific fields) in place."""


class LeadState(FunnelState):
    name = "lead"

    def entry_condition(self) -> str:
        return "Generated this month from Marketing Budget / referral (Section 5)"

    def exit_condition(self) -> str:
        return "Same month: converts to qualified_opportunity or is lost"

    def transition_probability(self, customer: Customer, params) -> float:
        return params["r1_lead_to_qo"]

    def advance(self, customer: Customer, month: int, params, provider: RandomnessProvider) -> None:
        p = self.transition_probability(customer, params)
        converted = provider.draw_event(p) if hasattr(provider, "draw_event") else (p >= 0.5)
        customer.state_name = "qualified_opportunity" if converted else "lost"


class QualifiedOpportunityState(FunnelState):
    name = "qualified_opportunity"

    def entry_condition(self) -> str:
        return "Leads(t) x r1, up to Capacity(t) (Section 5)"

    def exit_condition(self) -> str:
        return "Converts to active, or is lost"

    def transition_probability(self, customer: Customer, params) -> float:
        base_rate = params["r2_qo_to_active_base"]
        price = params.get("price", customer.pricing.reference_price)
        return customer.pricing.compute_effective_conversion_rate(base_rate, price)

    def advance(self, customer: Customer, month: int, params, provider: RandomnessProvider) -> None:
        p = self.transition_probability(customer, params)
        converted = provider.draw_event(p)
        customer.state_name = "active" if converted else "lost"


class BacklogState(FunnelState):
    name = "backlog"

    def entry_condition(self) -> str:
        return "Overflow(t) = max(0, Leads(t)*r1 - Capacity(t)) (Section 5)"

    def exit_condition(self) -> str:
        return "Converts at a discounted rate, exits (goes cold), or persists"

    def transition_probability(self, customer: Customer, params) -> float:
        base_rate = params["r2_qo_to_active_base"]
        price = params.get("price", customer.pricing.reference_price)
        discount = params["backlog_staleness_discount"]
        return customer.pricing.compute_effective_conversion_rate(base_rate, price) * discount

    def advance(self, customer: Customer, month: int, params, provider: RandomnessProvider) -> None:
        p_convert = self.transition_probability(customer, params)
        p_exit = params["backlog_exit_rate"]
        draw = provider.draw_event(p_convert)
        if draw:
            customer.state_name = "active"
        elif provider.draw_event(p_exit):
            customer.state_name = "lost"
        else:
            customer.months_in_queue += 1  # persists, still in backlog


class ActiveState(FunnelState):
    name = "active"

    def entry_condition(self) -> str:
        return "New conversion from qualified_opportunity or backlog, or persisted from prior month"

    def exit_condition(self) -> str:
        return "Remains active, or exits to churned"

    def transition_probability(self, customer: Customer, params) -> float:
        return params["monthly_churn_hazard"]

    def advance(self, customer: Customer, month: int, params, provider: RandomnessProvider) -> None:
        h = self.transition_probability(customer, params)
        churned = provider.draw_event(h)
        customer.tenure_months += 1
        customer.state_name = "churned" if churned else "active"


class ChurnedState(FunnelState):
    name = "churned"

    def entry_condition(self) -> str:
        return "Active(t) x h(t) exits this month"

    def exit_condition(self) -> str:
        return "Terminal in Prototype 1 (no win-back)"

    def transition_probability(self, customer: Customer, params) -> float:
        return 0.0

    def advance(self, customer: Customer, month: int, params, provider: RandomnessProvider) -> None:
        pass  # absorbing state


STATES: dict[str, FunnelState] = {
    "lead": LeadState(),
    "qualified_opportunity": QualifiedOpportunityState(),
    "backlog": BacklogState(),
    "active": ActiveState(),
    "churned": ChurnedState(),
}
