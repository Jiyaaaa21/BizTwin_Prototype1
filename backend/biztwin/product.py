"""
Product - a business-domain component (node) in the Final Design's causal
graph, not merely a software-architecture module: Product -> Sales Funnel,
because which SKU/tier a Qualified Opportunity is pursuing determines the
price it converts at (via PricingStrategy) and therefore its conversion
probability. Before this module, `synthetic.py` emitted a single hardcoded
"Core Subscription" row - there was no real Product catalog to point to in
a review ("what product did this customer buy?" had no honest answer).

Kept deliberately small: a Product's price is expressed as a multiplier on
the prevailing Price lever (`price_multiplier`), not an independent price,
so the existing PricingStrategy / elasticity mechanism still governs actual
dollars - Product selects WHICH tier a customer lands in, it does not
duplicate pricing logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    tier: str  # "self_serve" | "mid_market" | "enterprise"
    features: tuple[str, ...]
    value_proposition: str
    price_multiplier: float   # relative to the Price lever, e.g. 0.7 = 70% of list price
    catalog_weight: float     # relative frequency this SKU is sold at, for synthetic sampling


@dataclass
class ProductCatalog:
    products: list[Product] = field(default_factory=list)

    def by_sku(self, sku: str) -> Product:
        for p in self.products:
            if p.sku == sku:
                return p
        raise KeyError(f"Unknown SKU: {sku}")

    def weighted_average_multiplier(self) -> float:
        """Sanity check: the catalog's demand-weighted average price
        multiplier, used to keep synthetic revenue reconciled against the
        single Price lever (Final Design Section 8's built-in consistency
        check) even though individual customers now buy different tiers."""
        total_weight = sum(p.catalog_weight for p in self.products)
        if total_weight <= 0:
            return 1.0
        return sum(p.price_multiplier * p.catalog_weight for p in self.products) / total_weight

    def sample(self, rng) -> Product:
        """Draw one SKU, weighted by `catalog_weight` (Strategy-free: a
        plain categorical draw, not a decision the model has to justify)."""
        weights = [p.catalog_weight for p in self.products]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        idx = rng.choice(len(self.products), p=probabilities)
        return self.products[idx]


def default_catalog() -> ProductCatalog:
    """
    Three tiers, weighted so the catalog's average price multiplier is
    1.0 - the same dollar volume the model produced before Product existed,
    just now attributable to a real SKU instead of one hardcoded row.
    Confidence: LOW (feature lists and tier names are illustrative,
    matching the project's Customer Segment mix conceptually - self_serve
    customers skew Starter, enterprise customers skew Enterprise - but
    have not been validated against a real product catalog).
    """
    return ProductCatalog(products=[
        Product(
            sku="STARTER-1", name="Starter", tier="self_serve",
            features=("Core workflows", "Email support"),
            value_proposition="Low-friction entry for small teams",
            price_multiplier=0.70, catalog_weight=0.50,
        ),
        Product(
            sku="GROWTH-1", name="Growth", tier="mid_market",
            features=("Core workflows", "Priority support", "API access"),
            value_proposition="Scales with a growing team's usage",
            price_multiplier=1.00, catalog_weight=0.30,
        ),
        Product(
            sku="ENTERPRISE-1", name="Enterprise", tier="enterprise",
            features=("Core workflows", "Dedicated CSM", "SSO/SAML", "Custom SLAs"),
            value_proposition="Governance and support for large organizations",
            price_multiplier=1.75, catalog_weight=0.20,
        ),
    ])
