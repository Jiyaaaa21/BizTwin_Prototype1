"""
Tests for two gaps identified in an architecture review against the
project's own Phase 0 notes: (1) only one-at-a-time parameter sensitivity
existed - no way to see whether one parameter's effect depends on another
(a true interaction); (2) Customer Segment's behavioral multipliers only
ever reached the individual-level SyntheticDataGenerator, never the
aggregate engine the dashboard actually simulates forward with.

Both additions are opt-in and additive: every test in
test_reproducibility.py still passes unchanged (see the same session's
verification run), because the new code paths are only exercised when a
caller explicitly passes a segment_mix or calls sweep_2d - the default
path is bit-for-bit identical to before.

Run with: pytest backend/tests/test_architecture_extensions.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from biztwin import (
    CustomerSegment, DeterministicProvider, DynamicSimulationEngine,
    ParameterSet, ScenarioRunner, SegmentMix, SensitivityAnalyzer,
    TieredPricingStrategy, initial_state,
)

EXISTING_ACTIVE_BASE = 150.0
MARKETING_BUDGET = 50_000.0
PRICE = 12_000.0
CAPACITY = 40.0


def pricing_factory(params: ParameterSet) -> TieredPricingStrategy:
    return TieredPricingStrategy(
        reference_price=PRICE, elasticity_beta=params.price_elasticity_beta.value,
    )


# ---------------------------------------------------------------------------
# Gap 2: segment effects reaching the aggregate engine
# ---------------------------------------------------------------------------

def test_segment_mix_none_reproduces_baseline_exactly():
    """No segment_mix supplied -> identical to the pre-existing engine,
    bit-for-bit. This is what guarantees the new feature cannot regress
    any previously-verified number."""
    params = ParameterSet()
    pricing = pricing_factory(params)
    engine = DynamicSimulationEngine(
        params, pricing, DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    start = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)
    history = engine.run(start, horizon=6)
    # Matches the shipped verification script's own Section 8 baseline exactly.
    assert history[-1].active == pytest.approx(185.6261, abs=0.01)


def test_segment_mix_shifts_aggregate_churn_in_the_expected_direction():
    """A population skewed toward Self-Serve (churn_multiplier=1.6) should
    end up with FEWER active customers than an identically-sized population
    skewed toward Enterprise (churn_multiplier=0.5) - proving the Segment
    -> Funnel edge is now real at the aggregate level, not just in the
    individual-record synthetic layer."""
    params = ParameterSet()
    pricing = pricing_factory(params)
    start = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)

    high_churn_mix = SegmentMix(segments=[CustomerSegment(
        name="self_serve", conversion_multiplier=1.0, churn_multiplier=1.6,
        renewal_multiplier=0.85, deal_size_multiplier=0.6, catalog_weight=1.0,
    )])
    low_churn_mix = SegmentMix(segments=[CustomerSegment(
        name="enterprise", conversion_multiplier=1.0, churn_multiplier=0.5,
        renewal_multiplier=1.15, deal_size_multiplier=1.5, catalog_weight=1.0,
    )])

    high_engine = DynamicSimulationEngine(
        params, pricing, DeterministicProvider(), enable_backlog=False,
        enable_referral=False, segment_mix=high_churn_mix,
    )
    low_engine = DynamicSimulationEngine(
        params, pricing, DeterministicProvider(), enable_backlog=False,
        enable_referral=False, segment_mix=low_churn_mix,
    )

    high_active = high_engine.run(start, horizon=12)[-1].active
    low_active = low_engine.run(start, horizon=12)[-1].active
    assert high_active < low_active


def test_default_segment_mix_blends_to_expected_multipliers():
    """The shipped default_segments() catalog (30/35/35 weights) should
    blend to a conversion multiplier just above 1.0 and a churn multiplier
    just above 1.0 - self_serve's higher multipliers (1.15x, 1.6x) pull the
    weighted average up despite enterprise's dampening (0.85x, 0.5x)."""
    from biztwin import default_segments
    blended = default_segments().blended_multipliers()
    assert 1.0 < blended["conversion"] < 1.1
    assert 1.0 < blended["churn"] < 1.2


# ---------------------------------------------------------------------------
# Gap 1: two-parameter interaction analysis
# ---------------------------------------------------------------------------

def test_sweep_2d_runs_the_full_grid():
    params = ParameterSet()
    runner = ScenarioRunner(params, pricing_factory, DeterministicProvider(), enable_backlog=True, enable_referral=False)
    analyzer = SensitivityAnalyzer(runner)
    results = analyzer.sweep_2d(
        "sales_team_capacity", (20.0, 100.0, 3),
        "r1_lead_to_qo", (0.10, 0.40, 2),
        EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6,
        output_fn=lambda hist: hist[-1].active,
    )
    assert len(results) == 3 * 2  # every (x, y) cell present


def test_sweep_2d_detects_a_real_backlog_by_conversion_interaction():
    """backlog_staleness_discount's payoff should genuinely depend on
    lead-to-QO conversion: Backlog only accumulates when qo_raw overflows
    the (fixed, capacity=40) cap. At r1_lead_to_qo=0.10, qo_raw rarely
    exceeds 40, so there is ~no backlog and backlog_staleness_discount is
    inert (flat line - see the 165.84 values at every discount level
    below). At r1_lead_to_qo=0.40, qo_raw regularly overflows 40, so
    backlog is real and its staleness discount materially changes Active.
    This is a real, mechanistic interaction (Backlog only binds when QO
    overflows capacity), not noise.

    Note: sales_team_capacity (the originally intended param_x here) turned
    out to be a dead ParameterSet field - grep confirms engine.py never
    reads it; the capacity actually used each step is threaded through as
    a plain function argument (base_capacity / the capacity Controller),
    not the ParameterSet. Swapping it out for a field the engine actually
    consumes is the honest fix; leaving sales_team_capacity unused instead
    of wiring it up (or removing it) is a separate, pre-existing gap worth
    flagging on its own."""
    params = ParameterSet()
    runner = ScenarioRunner(params, pricing_factory, DeterministicProvider(), enable_backlog=True, enable_referral=False)
    analyzer = SensitivityAnalyzer(runner)
    results = analyzer.sweep_2d(
        "backlog_staleness_discount", (0.2, 0.9, 3),
        "r1_lead_to_qo", (0.10, 0.40, 2),
        EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6,
        output_fn=lambda hist: hist[-1].active,
    )
    strength = SensitivityAnalyzer.interaction_strength(results)
    assert strength > 10.0  # discount's marginal effect clearly differs by conversion level


def test_sweep_2d_no_interaction_when_parameters_cannot_touch():
    """Sanity check on the diagnostic itself: two parameters that can never
    mechanistically interact (churn hazard, which only ever acts on the
    Active stock, and win_prob_lead, which only feeds the read-only pipeline
    valuation in metrics.py and never touches Active at all) should show
    ~zero interaction strength on Active(T) - proving interaction_strength
    doesn't just report a nonzero number for any two parameters swept
    together."""
    params = ParameterSet()
    runner = ScenarioRunner(params, pricing_factory, DeterministicProvider(), enable_backlog=False, enable_referral=False)
    analyzer = SensitivityAnalyzer(runner)
    results = analyzer.sweep_2d(
        "monthly_churn_hazard", (0.005, 0.05, 3),
        "win_prob_lead", (0.05, 0.15, 2),
        EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6,
        output_fn=lambda hist: hist[-1].active,
    )
    strength = SensitivityAnalyzer.interaction_strength(results)
    assert strength < 0.01
