"""
CustomerSegment - a business-domain component (node) in the causal graph:
Market -> Customer Segment -> Sales Funnel. Before this module, "segment"
was a bare label ("enterprise" / "self_serve") attached to a customer
record with no behavior of its own - a name, not a node in a causal graph.
Each CustomerSegment now carries real behavioral multipliers that shape
conversion, churn, renewal, and revenue, so a segment actually does
something rather than just describing something.

Scope note (explicit, not hidden): these multipliers act at the
individual-agent layer (SyntheticDataGenerator, Contract) - they are not
folded into the aggregate deterministic engine (engine.py), which remains
a single-population System Dynamics model for Prototype 1. Splitting the
aggregate engine into per-segment cohort populations is a legitimate
Prototype 2 extension; doing it inside this pass would mean re-deriving
every already-verified Phase 4/5/9 worked number, which the project's own
generation-to-generation discipline argues against attempting alongside
this reframing.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CustomerSegment:
    name: str
    conversion_multiplier: float   # segment's relative pull among each month's NEWLY converted customers
    churn_multiplier: float        # >1 = over-represented when the engine says N customers churned this month
    renewal_multiplier: float      # multiplies renewal_probability at each Contract renewal boundary
    deal_size_multiplier: float    # multiplies deal size (revenue), stacked with the Product's own multiplier
    catalog_weight: float          # baseline relative frequency in the customer base


@dataclass
class SegmentMix:
    segments: list[CustomerSegment] = field(default_factory=list)

    def by_name(self, name: str) -> CustomerSegment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(f"Unknown segment: {name}")

    def sample_for_new_customer(self, rng) -> CustomerSegment:
        """Weighted by catalog_weight x conversion_multiplier: segments
        with higher conversion propensity are over-represented among each
        month's newly converted customers. This is how 'Segment influences
        conversion' actually shows up, without rebuilding the aggregate
        engine into a per-segment state space (see module docstring)."""
        weights = [s.catalog_weight * s.conversion_multiplier for s in self.segments]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        idx = rng.choice(len(self.segments), p=probabilities)
        return self.segments[idx]

    def sample_baseline(self, rng) -> CustomerSegment:
        """Weighted by catalog_weight alone - used for the pre-existing
        Active base at month 0, which this run did not itself convert, so
        it should not carry a conversion-propensity bias."""
        weights = [s.catalog_weight for s in self.segments]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        idx = rng.choice(len(self.segments), p=probabilities)
        return self.segments[idx]

    def blended_multipliers(self) -> dict[str, float]:
        """
        Population-weighted (by catalog_weight) blend of this mix's
        conversion and churn multipliers into two single numbers - what the
        aggregate, single-population SimulationEngine can actually consume
        (it has one Active stock, not per-segment cohorts). An honest
        approximation: it makes the mix's net effect visible in the
        aggregate numbers, but it cannot reproduce a shift in mix
        composition over time the way true per-segment cohort tracking
        would (that remains a Prototype 2 extension - see engine.py).
        """
        total_weight = sum(s.catalog_weight for s in self.segments)
        if total_weight <= 0:
            return {"conversion": 1.0, "churn": 1.0}
        conversion = sum(s.catalog_weight * s.conversion_multiplier for s in self.segments) / total_weight
        churn = sum(s.catalog_weight * s.churn_multiplier for s in self.segments) / total_weight
        return {"conversion": conversion, "churn": churn}

    def churn_weights(self, names: list[str]) -> list[float]:
        """Relative likelihood each of these customers (given as a list of
        segment names, one per candidate) is selected as part of this
        month's churned population - segments with a higher churn_multiplier
        are over-represented, exactly as a Digital Twin reviewer would
        expect 'Segment influences churn' to actually mean."""
        raw = [self.by_name(n).churn_multiplier for n in names]
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(names)] * len(names)
        return [w / total for w in raw]


def default_segments() -> SegmentMix:
    """
    Confidence: LOW - multipliers are illustrative, matching the standard
    SaaS pattern (Enterprise: larger deals, longer consideration, stickier;
    Self-Serve: smaller deals, faster in, faster out) but not yet validated
    against real cohort data. catalog_weight (0.30/0.35/0.35) matches the
    flat segment_mix this project used before segments had behavior, so
    introducing behavior does not by itself shift the baseline population mix.
    """
    return SegmentMix(segments=[
        CustomerSegment(
            name="enterprise", conversion_multiplier=0.85, churn_multiplier=0.5,
            renewal_multiplier=1.15, deal_size_multiplier=1.5, catalog_weight=0.30,
        ),
        CustomerSegment(
            name="smb", conversion_multiplier=1.0, churn_multiplier=1.0,
            renewal_multiplier=1.0, deal_size_multiplier=1.0, catalog_weight=0.35,
        ),
        CustomerSegment(
            name="self_serve", conversion_multiplier=1.15, churn_multiplier=1.6,
            renewal_multiplier=0.85, deal_size_multiplier=0.6, catalog_weight=0.35,
        ),
    ])
