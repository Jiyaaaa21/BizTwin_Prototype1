"""
SimulationEngine - Final Design Section 10 (Final Architecture Diagram):
"State & Transition Core". One deterministic pipeline (`_compute_step`),
shared by StaticSimulationEngine and DynamicSimulationEngine (Template
Method pattern) and reused unchanged by ScenarioRunner and
SyntheticDataGenerator via different RandomnessProvider / Controller
configurations (Strategy pattern) - there is exactly one set of formulas.

StepResult holds ONLY true dynamical state (Final Design's Architectural
Reframing: "Metrics Are Emergent, Not Modeled"). ARR, churn rate, conversion
rate, and pipeline value are never computed here - see metrics.py, which
reads a StepResult after the fact and writes nothing back.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from .controllers import Controller, IdentityController
from .market import MarketConditions
from .parameters import ParameterSet
from .pricing import PricingStrategy
from .randomness import RandomnessProvider
from .segment import SegmentMix


@dataclass
class StepResult:
    """True state at the end of month `month` (Final Design, Section 4:
    'A quantity belongs in State(t) only if it is carried forward')."""

    month: int
    leads: float
    qo_raw: float               # Leads(t) x r1, before the capacity cap
    qualified_opportunity: float  # after the capacity cap
    overflow: float              # qo_raw - qualified_opportunity, if positive
    backlog: float               # persists across months
    new_active_from_qo: float
    new_active_from_backlog: float
    churned: float
    active: float
    # Controllable lever values actually used this month (needed by the
    # next step's Controllers, and by MetricsObserver - not "state" the
    # transition function itself evolves, but inputs it consumed):
    marketing_budget: float
    price: float
    capacity: float

    @property
    def new_active(self) -> float:
        return self.new_active_from_qo + self.new_active_from_backlog


def initial_state(existing_active_base: float, marketing_budget: float, price: float, capacity: float) -> StepResult:
    """
    Pre-simulation seed (month=-1), NOT itself a reportable state. It carries
    only the business-supplied Existing Active Base and the starting lever
    values. Month 0 (the actual static snapshot) is produced by running this
    seed through one step of `_compute_step` with churn disabled - see
    `is_first_step` below - because Phase 4's worked example adds this
    month's new conversions to the existing base WITHOUT yet applying churn
    to it (churn only acts on a population that has had a full month to
    attrite, starting at the month 0 -> month 1 transition).
    """
    return StepResult(
        month=-1, leads=0.0, qo_raw=0.0, qualified_opportunity=0.0, overflow=0.0,
        backlog=0.0, new_active_from_qo=0.0, new_active_from_backlog=0.0,
        churned=0.0, active=existing_active_base,
        marketing_budget=marketing_budget, price=price, capacity=capacity,
    )


class SimulationEngine(ABC):
    """
    Abstract base. `_compute_step` is the ONE state-transition pipeline
    (Final Design Sections 4-6); `run()` is overridden by subclasses to
    orchestrate it differently (Template Method pattern, Task 8).
    """

    def __init__(
        self,
        params: ParameterSet,
        pricing: PricingStrategy,
        provider: RandomnessProvider,
        marketing_controller: Controller | None = None,
        capacity_controller: Controller | None = None,
        price_controller: Controller | None = None,
        enable_backlog: bool = True,
        enable_referral: bool = True,
        market: MarketConditions | None = None,
        segment_mix: SegmentMix | None = None,
    ):
        self.params = params
        self.pricing = pricing
        self.provider = provider
        self.marketing_controller = marketing_controller or IdentityController()
        self.capacity_controller = capacity_controller or IdentityController()
        self.price_controller = price_controller or IdentityController()
        self.enable_backlog = enable_backlog
        self.enable_referral = enable_referral
        # Market - Final Design domain component (Market -> Customer Segment
        # -> Sales Funnel). Default built from the same demand_index /
        # seasonality_index ParameterSet fields the pipeline always used, so
        # an engine built without an explicit `market` reproduces every
        # previously-verified number exactly (see MarketConditions.demand_multiplier).
        self.market = market or MarketConditions(
            demand_index=params.demand_index.value,
            seasonality_index=params.seasonality_index.value,
        )
        # Customer Segment, now reachable from the AGGREGATE engine too
        # (Architecture Review follow-up: the causal graph's Segment ->
        # Funnel edge previously existed only in the individual-level
        # SyntheticDataGenerator, never here - so a reviewer asking "does
        # Segment affect the numbers this dashboard actually simulates
        # forward with" had no honest yes). segment_mix is opt-in and None
        # by default: with no mix supplied, both blended multipliers below
        # are exactly 1.0, so an engine built without one reproduces every
        # previously-verified number exactly bit-for-bit - identical to the
        # pre-existing behavior, nothing already verified can regress.
        # When a mix IS supplied, `_compute_step` applies one
        # population-weighted (by catalog_weight) blended conversion and
        # churn multiplier to the whole population each step. This is an
        # honest approximation of the mix's net effect on the aggregate
        # numbers, not a replacement for true per-segment cohort tracking
        # (splitting the single Active population into per-segment
        # sub-populations remains a real Prototype 2 extension - see
        # SegmentMix.blended_multipliers's own docstring).
        self.segment_mix = segment_mix
        if segment_mix is not None:
            blended = segment_mix.blended_multipliers()
            self._segment_conversion_multiplier = blended["conversion"]
            self._segment_churn_multiplier = blended["churn"]
        else:
            self._segment_conversion_multiplier = 1.0
            self._segment_churn_multiplier = 1.0

    def _compute_step(self, prev: StepResult, is_first_step: bool = False) -> StepResult:
        p = self.params.snapshot_values()
        month = prev.month + 1

        # 1. Controllers observe State(t) and set Inputs(t+1) (Final Design
        #    Sec. 4 of the Architectural Reframing / Sec. 9 Scenario Engine).
        #    IdentityController reproduces every open-loop Phase 9 scenario.
        state_view = {
            "backlog": prev.backlog,
            "qualified_opportunity": prev.qualified_opportunity,
            "capacity": prev.capacity,
            "active": prev.active,
            "price": prev.price,  # needed by RevenueReinvestmentMarketingController (Loop C)
        }
        marketing_budget = self.marketing_controller.next_value(state_view, prev.marketing_budget)
        price = self.price_controller.next_value(state_view, prev.price)
        capacity = self.capacity_controller.next_value(state_view, prev.capacity)

        # 2. Leads: Marketing-sourced + referral (Loop A, Final Design Sec. 6).
        #    Marketing-sourced volume is scaled by the Market component's
        #    demand multiplier (Market -> Customer Segment -> Sales Funnel);
        #    with the default MarketConditions this is numerically identical
        #    to the old inline `demand_index * seasonality_index` term.
        referral_leads_expected = (prev.active * p["referral_rate"]) if self.enable_referral else 0.0
        marketing_leads_expected = (
            marketing_budget / p["cost_per_lead"] * self.market.demand_multiplier(month)
        )
        leads = self.provider.draw_count(marketing_leads_expected + referral_leads_expected)

        # 3. Qualified Opportunity, capacity-capped (the one hard nonlinearity
        #    besides Backlog decay - Final Design Sec. 4/5):
        qo_raw = self.provider.draw_binomial_count(leads, p["r1_lead_to_qo"])
        qualified_opportunity = min(qo_raw, capacity)
        overflow = max(0.0, qo_raw - capacity)
        slack = max(0.0, capacity - qo_raw)

        # 4. Conversion at the (price-adjusted, segment-mix-adjusted) rate:
        effective_r2 = self.pricing.compute_effective_conversion_rate(p["r2_qo_to_active_base"], price)
        effective_r2 = min(effective_r2 * self._segment_conversion_multiplier, 1.0)
        new_active_from_qo = self.provider.draw_binomial_count(qualified_opportunity, effective_r2)

        # 5. Backlog (Loop B, Final Design Sec. 6) - overflow queues instead
        #    of being discarded (replaces the old Assumption A9):
        new_active_from_backlog = 0.0
        remaining_backlog = 0.0
        next_backlog = 0.0
        if self.enable_backlog:
            backlog_prev = prev.backlog
            discounted_r2 = effective_r2 * p["backlog_staleness_discount"]
            if backlog_prev > 0:
                new_active_from_backlog = self.provider.draw_binomial_count(backlog_prev, discounted_r2)
            remaining_backlog = max(0.0, backlog_prev - new_active_from_backlog)
            next_backlog = max(0.0, remaining_backlog + overflow - slack) * (1 - p["backlog_exit_rate"])

        # 6. Active stock update (System Dynamics layer, Final Design Sec. 5).
        #    Month 0 adds this month's conversions to the existing base
        #    without yet applying churn (see initial_state's docstring).
        if is_first_step:
            churned = 0.0
        else:
            churn_hazard = min(p["monthly_churn_hazard"] * self._segment_churn_multiplier, 1.0)
            churned = self.provider.draw_binomial_count(prev.active, churn_hazard)
        active = prev.active - churned + new_active_from_qo + new_active_from_backlog

        return StepResult(
            month=month, leads=leads, qo_raw=qo_raw,
            qualified_opportunity=qualified_opportunity, overflow=overflow,
            backlog=next_backlog, new_active_from_qo=new_active_from_qo,
            new_active_from_backlog=new_active_from_backlog, churned=churned,
            active=active, marketing_budget=marketing_budget, price=price,
            capacity=capacity,
        )

    @abstractmethod
    def run(self, seed: StepResult, horizon: int = 0) -> list[StepResult]:
        ...


class StaticSimulationEngine(SimulationEngine):
    """Evaluates the pipeline exactly once (month 0 only), no state retained
    and no further time progression (Final Design Section 4 - a single
    frozen snapshot)."""

    def run(self, seed: StepResult, horizon: int = 0) -> list[StepResult]:
        month_zero = self._compute_step(seed, is_first_step=True)
        return [month_zero]


class DynamicSimulationEngine(SimulationEngine):
    """Loops the identical pipeline across months 0..T, retaining state
    between calls (Final Design Section 5). Month 0 is produced the same
    way as the static engine (churn disabled); months 1..T apply churn
    normally against the prior month's Active stock."""

    def run(self, seed: StepResult, horizon: int = 6) -> list[StepResult]:
        month_zero = self._compute_step(seed, is_first_step=True)
        history = [month_zero]
        current = month_zero
        for _ in range(horizon):
            current = self._compute_step(current, is_first_step=False)
            history.append(current)
        return history
