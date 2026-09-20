"""
PricingStrategy - Strategy pattern (Final Design, Section 4 Entity Catalog /
Task 8 OOP Architecture). Two concrete pricing structures, matching the
Domain Knowledge Document's "Tiered Packages" vs "Dynamic Pricing Models".

TieredPricingStrategy implements the calibrated elasticity relationship
(Final Design Section 7 / parameters.price_elasticity_beta):

    r2(Price) = r2_base * (Price / Price_ref) ** (-beta)

DynamicPricingStrategy (usage-based) is included for architectural
completeness per the Entity Catalog, but the Design Review named this branch
as ahead of demonstrated need (no calibrated usage parameters exist yet) -
it is deliberately minimal, not a hidden complexity trap.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class PricingStrategy(ABC):
    reference_price: float

    @abstractmethod
    def compute_effective_conversion_rate(self, base_rate: float, price: float) -> float:
        """Given a base (reference-price) conversion rate, return the rate
        actually in effect at `price`."""

    @abstractmethod
    def compute_per_customer_arr(self, price: float) -> float:
        """Annualized revenue contributed by one customer at `price`."""


class TieredPricingStrategy(PricingStrategy):
    """Fixed package tiers at a fixed price point; conversion responds to
    price via the calibrated log-linear elasticity (Final Design Sec. 7)."""

    def __init__(self, reference_price: float, elasticity_beta: float):
        self.reference_price = reference_price
        self.elasticity_beta = elasticity_beta

    def compute_effective_conversion_rate(self, base_rate: float, price: float) -> float:
        if self.reference_price <= 0:
            return base_rate
        ratio = price / self.reference_price
        return base_rate * (ratio ** (-self.elasticity_beta))

    def compute_per_customer_arr(self, price: float) -> float:
        return price


class DynamicPricingStrategy(PricingStrategy):
    """Usage-based pricing: revenue scales with a usage quantity rather than
    a fixed tier. Structurally present per the Entity Catalog; Generation 1
    carries no calibrated usage parameters, so this defaults to behaving
    like a flat rate until real usage data justifies more."""

    def __init__(self, reference_price: float, rate_per_unit: float, expected_usage_units: float = 1.0):
        self.reference_price = reference_price
        self.rate_per_unit = rate_per_unit
        self.expected_usage_units = expected_usage_units

    def compute_effective_conversion_rate(self, base_rate: float, price: float) -> float:
        # No calibrated usage-elasticity exists yet (Architecture Review, Task 11):
        # fall back to the same log-linear form with a conservative default beta.
        if self.reference_price <= 0:
            return base_rate
        ratio = price / self.reference_price
        return base_rate * (ratio ** -0.5)

    def compute_per_customer_arr(self, price: float) -> float:
        return self.rate_per_unit * self.expected_usage_units
