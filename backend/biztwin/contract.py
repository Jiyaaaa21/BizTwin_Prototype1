"""
Contract - a business-domain component (node) in the Final Design's causal
graph: Sales Funnel -> Contract/Subscription -> Revenue Engine. Governs
what happens to a customer AFTER conversion, which the funnel's own State
pattern (funnel_state.py) deliberately does not model - FunnelState governs
pre-purchase progression (Lead -> ... -> Active), Contract governs the
post-purchase lifecycle (Active -> Renewal Due -> Renewed | Churned).

Before this module, a Subscription record carried only a flat `term`
string ("monthly"/"annual") with no actual renewal or lifecycle behavior -
there was nothing a reviewer could point to as "the contract" and ask how
it evolves over time. This is a small, explicit state machine, not a
duplicate of FunnelState: it is stepped once per month per active customer,
independently of the funnel.

Renewal is now a real calibratable draw (`renewal_probability`,
parameters.py), not a hardcoded boolean - this is the exact behavior
CalibrationEngine.worked_example_renewal() demonstrates calibrating.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContractStatus(Enum):
    ACTIVE = "active"
    RENEWAL_DUE = "renewal_due"   # term boundary reached, no renewal draw was supplied
    RENEWED = "renewed"           # term boundary reached and the renewal draw succeeded
    CHURNED = "churned"           # absorbing - includes both mid-term churn AND a failed
                                   # renewal draw (a non-renewal is a churn event in Prototype 1's scope)


@dataclass
class Contract:
    contract_id: str
    customer_id: str
    term_length_months: int   # 1 = monthly, 12 = annual
    start_month: int
    status: ContractStatus = ContractStatus.ACTIVE
    renewals: int = 0

    def months_elapsed(self, current_month: int) -> int:
        return current_month - self.start_month

    def is_up_for_renewal(self, current_month: int) -> bool:
        """
        Monthly contracts (term_length_months == 1) never hit a renewal
        boundary here by design: month-to-month attrition is already fully
        captured by the engine's monthly_churn_hazard draw, so treating
        every single month as a separate renewal decision would silently
        double-count churn for most of the customer base. A genuine
        renewal decision - the one renewal_probability actually calibrates
        - exists only for a term longer than one month (e.g. annual).
        """
        if self.term_length_months <= 1:
            return False
        elapsed = self.months_elapsed(current_month)
        return elapsed > 0 and elapsed % self.term_length_months == 0

    def advance(
        self,
        current_month: int,
        churned: bool,
        renewal_probability: float | None = None,
        provider=None,
    ) -> None:
        """
        One state-machine step. `churned` is supplied by the caller -
        Contract does not draw its own mid-term churn; monthly_churn_hazard
        and the engine's Bernoulli draw remain the single source of truth
        for whether a customer left this month (Final Design: one
        mechanism, never duplicated).

        At a term boundary, if both `renewal_probability` and `provider`
        are supplied, renewal is a real Bernoulli draw on the calibrated
        `renewal_probability` parameter (optionally segment-adjusted by the
        caller before it's passed in) - this is the intended path.
        Without them (e.g. a caller that hasn't wired calibration/a
        provider through yet), the contract is left RENEWAL_DUE rather than
        silently assumed to renew, so a missing wire-up is visible in the
        data rather than hidden behind an optimistic default.
        """
        if self.status == ContractStatus.CHURNED:
            return
        if churned:
            self.status = ContractStatus.CHURNED
            return
        if self.is_up_for_renewal(current_month):
            if renewal_probability is not None and provider is not None:
                renewed = provider.draw_event(renewal_probability)
                if renewed:
                    self.status = ContractStatus.RENEWED
                    self.renewals += 1
                else:
                    self.status = ContractStatus.CHURNED
            else:
                self.status = ContractStatus.RENEWAL_DUE
        else:
            self.status = ContractStatus.ACTIVE
