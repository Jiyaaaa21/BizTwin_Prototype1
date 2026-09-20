"""
Parameter registry - Final Prototype 1 Design, Section 7 (Calibration Framework).

Every rate/behavioral parameter in the system is declared here, exactly once,
with its documented range, default (prior) value, and confidence level. No
parameter value appears anywhere else in the codebase as a bare literal -
engines, controllers, and generators all read from this registry so that a
calibration update (CalibrationEngine, calibration.py) changes behavior
everywhere at once, and every value in the system remains traceable to a
cited source.

Confidence levels (Design Review, Task 6 / Section 9 of the Final Design):
    HIGH        - sourced directly from a system of record, not estimated.
    MEDIUM      - a directly countable operational metric with limited history.
    LOW         - industry benchmark only, no internal signal yet.
    VERY_LOW    - the functional form itself is uncertain, not just the magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Confidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"


@dataclass
class Parameter:
    """One calibratable (or fixed) quantity, with full provenance."""

    name: str
    meaning: str
    value: float
    range: tuple[float, float]
    confidence: Confidence
    source: str
    calibration_method: str
    validation_method: str

    def clamp(self, x: float) -> float:
        """Keep a proposed value inside the documented plausible range."""
        lo, hi = self.range
        return max(lo, min(hi, x))


@dataclass
class ParameterSet:
    """
    The complete Phase 6 / Final Design Section 7 parameter table, illustrative
    defaults per Assumption A8 - not measurements. Instantiate one of these per
    simulation run; CalibrationEngine returns updated copies, never mutates a
    shared global registry in place, so a run's parameters are always a known,
    frozen snapshot (reproducibility, Guiding Principle 7).
    """

    # --- Marketing / acquisition ---
    cost_per_lead: Parameter = field(default_factory=lambda: Parameter(
        name="cost_per_lead", meaning="Marketing spend to generate one lead",
        value=200.0, range=(50.0, 500.0), confidence=Confidence.LOW,
        source="Industry benchmark",
        calibration_method="Credibility blend vs. observed spend / leads generated",
        validation_method="Held-out likelihood on next observation window",
    ))
    referral_rate: Parameter = field(default_factory=lambda: Parameter(
        name="referral_rate", meaning="New leads generated per Active customer per month via referral",
        value=0.02, range=(0.01, 0.05), confidence=Confidence.LOW,
        source="Industry benchmark (B2B SaaS referral rates)",
        calibration_method="Credibility blend vs. observed referral-sourced lead volume",
        validation_method="Held-out likelihood on next observation window",
    ))
    demand_index: Parameter = field(default_factory=lambda: Parameter(
        name="demand_index", meaning="Exogenous multiplier scaling lead volume for macro demand",
        value=1.0, range=(0.7, 1.3), confidence=Confidence.LOW,
        source="Neutral default, no benchmark yet",
        calibration_method="Classical seasonal/trend decomposition once >=12 months of data exists",
        validation_method="Out-of-sample lead-volume comparison",
    ))
    seasonality_index: Parameter = field(default_factory=lambda: Parameter(
        name="seasonality_index", meaning="Exogenous multiplier scaling lead volume for calendar seasonality",
        value=1.0, range=(0.7, 1.3), confidence=Confidence.LOW,
        source="Neutral default, no benchmark yet",
        calibration_method="Classical seasonal decomposition (e.g. STL)",
        validation_method="Out-of-sample lead-volume comparison",
    ))

    # --- Funnel conversion ---
    r1_lead_to_qo: Parameter = field(default_factory=lambda: Parameter(
        name="r1_lead_to_qo", meaning="Lead -> Qualified Opportunity conversion rate",
        value=0.30, range=(0.10, 0.40), confidence=Confidence.LOW,
        source="Industry benchmark",
        calibration_method="Credibility blend (Bayesian Beta-Binomial)",
        validation_method="Held-out likelihood on next observation window",
    ))
    r2_qo_to_active_base: Parameter = field(default_factory=lambda: Parameter(
        name="r2_qo_to_active_base", meaning="Qualified Opportunity -> Active conversion rate at reference price",
        value=0.20, range=(0.10, 0.35), confidence=Confidence.LOW,
        source="Industry benchmark",
        calibration_method="Credibility blend vs. observed win rate",
        validation_method="Held-out likelihood on next observation window",
    ))
    price_elasticity_beta: Parameter = field(default_factory=lambda: Parameter(
        name="price_elasticity_beta",
        meaning="Elasticity linking price deviation from reference to change in r2 "
                 "(log-linear: r2(Price) = r2_base * (Price/Price_ref)^-beta)",
        value=0.5, range=(0.1, 2.0), confidence=Confidence.VERY_LOW,
        source="First-principles, deliberately conservative pending evidence",
        calibration_method="Logistic regression of observed conversion outcomes on price",
        validation_method="Out-of-sample conversion prediction",
    ))

    # --- Capacity / Backlog ---
    sales_team_capacity: Parameter = field(default_factory=lambda: Parameter(
        name="sales_team_capacity", meaning="Maximum Qualified Opportunities workable per month",
        value=40.0, range=(20.0, 200.0), confidence=Confidence.MEDIUM,
        source="Operational headcount x industry per-rep range (2 reps x 20)",
        calibration_method="Direct measurement from CRM workload logs, no blending",
        validation_method="Direct comparison to logged workload",
    ))
    backlog_exit_rate: Parameter = field(default_factory=lambda: Parameter(
        name="backlog_exit_rate", meaning="Fraction of queued Backlog that goes cold and exits each month",
        value=0.15, range=(0.10, 0.30), confidence=Confidence.LOW,
        source="First-principles (no benchmark exists)",
        calibration_method="Survival analysis on backlog-aging data",
        validation_method="Out-of-sample backlog-conversion comparison",
    ))
    backlog_staleness_discount: Parameter = field(default_factory=lambda: Parameter(
        name="backlog_staleness_discount", meaning="Conversion-rate multiplier for Backlog-originated conversions vs. fresh QO",
        value=0.75, range=(0.5, 0.9), confidence=Confidence.LOW,
        source="First-principles",
        calibration_method="Compare realized win rate, aged vs. fresh opportunities",
        validation_method="Held-out comparison",
    ))

    # --- Churn ---
    monthly_churn_hazard: Parameter = field(default_factory=lambda: Parameter(
        name="monthly_churn_hazard", meaning="Probability an Active customer churns in a given month",
        value=0.02, range=(0.005, 0.05), confidence=Confidence.LOW,
        source="Industry benchmark",
        calibration_method="Kaplan-Meier estimator on cohort retention curves, credibility-blended",
        validation_method="Out-of-sample cohort retention comparison",
    ))

    # --- Contract / renewal ---
    renewal_probability: Parameter = field(default_factory=lambda: Parameter(
        name="renewal_probability",
        meaning="Probability a Contract renews at its term boundary (Contract's own state "
                 "transition, distinct from monthly_churn_hazard's mid-term churn draw)",
        value=0.90, range=(0.60, 0.98), confidence=Confidence.MEDIUM,
        source="Comparable B2B SaaS net-renewal-rate benchmarks",
        calibration_method="Credibility blend (Bayesian Beta-Binomial) vs. observed renewal outcomes "
                             "at term boundaries - see CalibrationEngine.worked_example_renewal()",
        validation_method="Held-out likelihood on the next cohort's renewal boundary",
    ))

    # --- Pipeline valuation ---
    win_prob_lead: Parameter = field(default_factory=lambda: Parameter(
        name="win_prob_lead", meaning="Stage win-probability discount applied to Lead-stage pipeline value",
        value=0.10, range=(0.05, 0.15), confidence=Confidence.LOW,
        source="Industry benchmark",
        calibration_method="Historical stage-to-close hit rate, credibility-blended",
        validation_method="Held-out likelihood",
    ))
    win_prob_qo: Parameter = field(default_factory=lambda: Parameter(
        name="win_prob_qo", meaning="Stage win-probability discount applied to Qualified Opportunity pipeline value",
        value=0.25, range=(0.15, 0.35), confidence=Confidence.LOW,
        source="Industry benchmark",
        calibration_method="Historical stage-to-close hit rate, credibility-blended",
        validation_method="Held-out likelihood",
    ))

    # --- Controller policy (business decisions, not statistically calibrated) ---
    capacity_controller_hysteresis_months: Parameter = field(default_factory=lambda: Parameter(
        name="capacity_controller_hysteresis_months",
        meaning="Consecutive months of positive Backlog before CapacityController triggers a hire",
        value=3.0, range=(2.0, 4.0), confidence=Confidence.MEDIUM,
        source="Business policy choice",
        calibration_method="Set directly by the business, not statistically estimated",
        validation_method="Reviewed by the business, not validated statistically",
    ))
    capacity_hiring_increment: Parameter = field(default_factory=lambda: Parameter(
        name="capacity_hiring_increment", meaning="Capacity added per hiring trigger (one rep)",
        value=20.0, range=(10.0, 30.0), confidence=Confidence.MEDIUM,
        source="Business policy choice / per-rep capacity benchmark",
        calibration_method="Set directly by the business",
        validation_method="Reviewed by the business, not validated statistically",
    ))

    def as_dict(self) -> dict[str, Parameter]:
        return {p.name: p for p in self.__dict__.values() if isinstance(p, Parameter)}

    def snapshot_values(self) -> dict[str, float]:
        """A plain {name: value} view - what the engine actually consumes each step."""
        return {name: p.value for name, p in self.as_dict().items()}
