"""
MetricsObserver - Final Design, Architectural Reframing: "Metrics Are
Emergent, Not Modeled." Every function here is Metrics(t) = g(State(t)),
evaluated strictly after the transition, and NONE of it is ever written back
into StepResult or consumed by SimulationEngine._compute_step. This module
is the entire "Visualization Layer" data source (Final Design Section 10) -
a dashboard reads only what's here, never the engine's internals directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine import StepResult


@dataclass
class PeriodMetrics:
    month: int
    arr: float
    mrr: float
    churn_rate: float | None
    conversion_rate: float | None
    pipeline_value: float
    active: float
    backlog: float


def arr(step: StepResult) -> float:
    """ARR(t) = Active(t) x Price(t) - a pure read of the Active stock and
    the Price input, never a separately modeled quantity."""
    return step.active * step.price


def mrr(step: StepResult) -> float:
    return arr(step) / 12.0


def churn_rate(step: StepResult, prev: StepResult | None) -> float | None:
    """Churn Rate(t) = Churned(t) / Active(t-1) - a read of a transition
    that already happened, never a driver within the same timestep."""
    if prev is None or prev.active <= 0:
        return None
    return step.churned / prev.active


def conversion_rate(step: StepResult) -> float | None:
    """Conversion Rate(t) = NewActive(t) / QO(t)."""
    if step.qualified_opportunity <= 0:
        return None
    return step.new_active / step.qualified_opportunity


def pipeline_value(step: StepResult, win_prob_lead: float, win_prob_qo: float,
                    backlog_discount: float = 1.0) -> float:
    """Weighted-pipeline valuation, now including Backlog (Final Design's
    Architectural Reframing note: a real business's 'pipeline' balloons with
    backlog, not just fresh leads, while ARR can stay flat - Phase 9's
    Scenario C finding, sharpened)."""
    deal_size = step.price
    value = step.leads * deal_size * win_prob_lead
    value += step.qualified_opportunity * deal_size * win_prob_qo
    value += step.backlog * deal_size * win_prob_qo * backlog_discount
    return value


def expected_sales_cycle_length(qo_exit_prob: float, backlog_exit_prob: float,
                                 overflow_fraction: float) -> float:
    """
    Derived from the transition structure itself (geometric holding-time
    result for a Markov state with per-period exit probability p: expected
    periods spent = 1/p) - Final Design, Architectural Reframing:
    'this was never an independently modeled number.'

    Base cycle = 1 month in Lead + 1 month in Qualified Opportunity
    (both resolve same-month, Section 5). The fraction of units that
    overflow into Backlog additionally spend an expected
    1 / (qo_exit_prob + backlog_exit_prob) months queued.
    """
    base = 2.0
    if overflow_fraction <= 0:
        return base
    denom = qo_exit_prob + backlog_exit_prob
    extra = (1.0 / denom) if denom > 0 else 0.0
    return base + overflow_fraction * extra


def summarize(step: StepResult, prev: StepResult | None, params_snapshot: dict) -> PeriodMetrics:
    return PeriodMetrics(
        month=step.month,
        arr=arr(step),
        mrr=mrr(step),
        churn_rate=churn_rate(step, prev),
        conversion_rate=conversion_rate(step),
        pipeline_value=pipeline_value(
            step, params_snapshot["win_prob_lead"], params_snapshot["win_prob_qo"],
            params_snapshot["backlog_staleness_discount"],
        ),
        active=step.active,
        backlog=step.backlog,
    )
