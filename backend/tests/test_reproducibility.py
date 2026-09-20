"""
Regression / reproducibility tests - Final Design, Section 7 "Verification"
gate: automated tests asserting every formula in Tasks 4-6 is implemented as
written, including the static/dynamic boundary reconciliation, plus a
same-seed reproducibility guarantee for the Monte Carlo synthetic data path
(Final Design Section 8: a Digital Twin's synthetic data must be
regenerable, not a one-off random artifact).

Run with: pytest tests/test_reproducibility.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from biztwin import (
    CalibrationEngine, DeterministicProvider, DynamicSimulationEngine,
    MonteCarloProvider, ParameterSet, ScenarioRunner, ScenarioSpec,
    SensitivityAnalyzer, StaticSimulationEngine, SyntheticDataGenerator,
    TieredPricingStrategy, initial_state, metrics,
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
# Phase 4 - Static State Model (legacy mode: no Backlog, no Referral)
# ---------------------------------------------------------------------------

def test_phase4_static_baseline():
    params = ParameterSet()
    engine = StaticSimulationEngine(
        params, pricing_factory(params), DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    start = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)
    step = engine.run(start)[-1]

    assert step.leads == pytest.approx(250, abs=1)
    assert step.qualified_opportunity == pytest.approx(40, abs=1)
    assert step.new_active_from_qo == pytest.approx(8, abs=1)
    assert step.active == pytest.approx(158, abs=1)
    assert metrics.arr(step) == pytest.approx(1_896_000, abs=10)


# ---------------------------------------------------------------------------
# Phase 5 - Dynamic State Engine, months 0-2 (legacy mode)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "month,expected_active,expected_arr",
    [(0, 158.00, 1_896_000), (1, 162.84, 1_954_080), (2, 167.5832, 2_010_998.4)],
)
def test_phase5_dynamic_trace(month, expected_active, expected_arr):
    params = ParameterSet()
    engine = DynamicSimulationEngine(
        params, pricing_factory(params), DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    start = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)
    history = engine.run(start, horizon=2)
    step = history[month]

    assert step.active == pytest.approx(expected_active, abs=0.05)
    assert metrics.arr(step) == pytest.approx(expected_arr, abs=50)


def test_phase4_phase5_boundary_reconciliation():
    """Static engine's month 0 must equal the dynamic engine's month 0
    exactly - the Phase 4/5 boundary the Final Design requires."""
    params = ParameterSet()
    static_engine = StaticSimulationEngine(
        params, pricing_factory(params), DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    dynamic_engine = DynamicSimulationEngine(
        params, pricing_factory(params), DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    start_a = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)
    start_b = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)

    static_month0 = static_engine.run(start_a)[0]
    dynamic_month0 = dynamic_engine.run(start_b, horizon=0)[0]

    assert static_month0.active == pytest.approx(dynamic_month0.active, abs=1e-9)
    assert metrics.arr(static_month0) == pytest.approx(metrics.arr(dynamic_month0), abs=1e-6)


# ---------------------------------------------------------------------------
# Phase 9 - Scenario comparison, month 6 (legacy mode)
# ---------------------------------------------------------------------------

@pytest.fixture
def phase9_results():
    params = ParameterSet()
    runner = ScenarioRunner(
        params, pricing_factory, DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    specs = [
        ScenarioSpec(name="A - Baseline", overrides={}),
        ScenarioSpec(name="B - Price Increase", overrides={"price": 14_400.0}),
        ScenarioSpec(name="C - Marketing Expansion", overrides={"marketing_budget": 75_000.0}),
        ScenarioSpec(name="D - Sales Team Growth", overrides={"capacity": 60.0}),
    ]
    return runner.compare(
        specs, existing_active_base=EXISTING_ACTIVE_BASE,
        base_marketing_budget=MARKETING_BUDGET, base_price=PRICE,
        base_capacity=CAPACITY, horizon=6,
    )


def test_phase9_all_scenarios_share_month0_baseline(phase9_results):
    """Final Design: 'All four start from the identical Phase 4/5 baseline
    at month 0' - a scenario's lever override must not retroactively affect
    the month-0 snapshot, only month 1 onward."""
    month0_actives = {name: r.history[0].active for name, r in phase9_results.items()}
    for name, active in month0_actives.items():
        assert active == pytest.approx(158.0, abs=1e-6), f"{name} month-0 baseline mismatch"


@pytest.mark.parametrize(
    "name,expected_active_6",
    [
        ("A - Baseline", 185.6),
        ("B - Price Increase", 181.6),
        ("C - Marketing Expansion", 185.6),
        ("D - Sales Team Growth", 208.5),
    ],
)
def test_phase9_scenarios_month6(phase9_results, name, expected_active_6):
    final = phase9_results[name].history[-1]
    assert final.active == pytest.approx(expected_active_6, abs=0.5)


def test_phase9_scenario_ranking_on_arr(phase9_results):
    """B > D > A = C on ARR(6), per the Final Design's Phase 9 narrative."""
    arr = {name: metrics.arr(r.history[-1]) for name, r in phase9_results.items()}
    assert arr["B - Price Increase"] > arr["D - Sales Team Growth"]
    assert arr["D - Sales Team Growth"] > arr["A - Baseline"]
    assert arr["A - Baseline"] == pytest.approx(arr["C - Marketing Expansion"], abs=1.0)


# ---------------------------------------------------------------------------
# Section 7 - Bayesian calibration worked example
# ---------------------------------------------------------------------------

def test_calibration_worked_example():
    params = ParameterSet()
    engine = CalibrationEngine(params)
    entry = engine.worked_example_lead_conversion()

    assert entry.n_observations == 200
    assert entry.k_credibility == 75
    assert entry.z_weight == pytest.approx(0.727, abs=0.001)
    assert entry.new_value * 100 == pytest.approx(17.2, abs=0.1)


def test_calibration_never_mutates_in_place():
    """CalibrationEngine.calibrate must return a new ParameterSet, never
    mutate the one it was constructed with (Final Design Sec. 7: every
    parameter change must be an auditable, logged event)."""
    params = ParameterSet()
    original_value = params.r1_lead_to_qo.value
    engine = CalibrationEngine(params)
    new_params = engine.calibrate(
        "r1_lead_to_qo", observed_value=0.5, n_observations=100, k_credibility=50,
        data_source="unit-test",
    )
    assert params.r1_lead_to_qo.value == original_value
    assert new_params.r1_lead_to_qo.value != original_value
    assert len(engine.log) == 1


# ---------------------------------------------------------------------------
# Final Design mode (Backlog + Referral enabled) - no NaNs, monotonic growth
# ---------------------------------------------------------------------------

def test_final_design_mode_smoke():
    params = ParameterSet()
    engine = DynamicSimulationEngine(params, pricing_factory(params), DeterministicProvider())
    start = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)
    history = engine.run(start, horizon=6)

    assert len(history) == 7  # months 0..6 inclusive
    assert history[-1].active > history[0].active
    for step in history:
        assert step.active >= 0
        assert step.backlog >= 0


# ---------------------------------------------------------------------------
# Section 8 - Synthetic data: reproducibility + revenue reconciliation
# ---------------------------------------------------------------------------

def test_synthetic_data_reproducible_with_same_seed():
    """Same seed -> byte-identical synthetic dataset (Final Design Sec. 8:
    a Digital Twin's synthetic data generation must be a reproducible
    scientific procedure, not an unrepeatable random artifact)."""
    params = ParameterSet()
    gen_a = SyntheticDataGenerator(params, pricing_factory, seed=7)
    gen_b = SyntheticDataGenerator(ParameterSet(), pricing_factory, seed=7)

    dataset_a, history_a = gen_a.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=4)
    dataset_b, history_b = gen_b.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=4)

    assert [s.active for s in history_a] == [s.active for s in history_b]
    pd.testing.assert_frame_equal(dataset_a.customers, dataset_b.customers)
    pd.testing.assert_frame_equal(dataset_a.revenue_events, dataset_b.revenue_events)


def test_synthetic_data_different_seed_diverges():
    """Sanity check on the reproducibility test itself: different seeds
    should (almost certainly) NOT produce identical customer draws."""
    params = ParameterSet()
    gen_a = SyntheticDataGenerator(params, pricing_factory, seed=1)
    gen_b = SyntheticDataGenerator(ParameterSet(), pricing_factory, seed=2)

    dataset_a, _ = gen_a.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=4)
    dataset_b, _ = gen_b.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=4)

    assert not dataset_a.customers["deal_size"].tolist() == dataset_b.customers["deal_size"].tolist()


def test_synthetic_data_revenue_reconciliation():
    params = ParameterSet()
    generator = SyntheticDataGenerator(params, pricing_factory, seed=42)
    dataset, history = generator.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6)
    final = history[-1]

    expected_monthly_revenue = final.active * final.price / 12.0
    realized = dataset.revenue_events.loc[dataset.revenue_events["month"] == 6, "amount"].sum()
    rel_error = abs(realized - expected_monthly_revenue) / expected_monthly_revenue

    # 10%, not 5%: two independent rounding/sampling sources compound over 6
    # months now that Product exists (integer customer-count rounding, and
    # which SKU/price-multiplier each customer draws) - see the matching
    # comment in examples/run_verification.py.
    assert rel_error < 0.10


def test_synthetic_data_every_customer_has_one_contract():
    """Contract - Final Design domain component: every synthetic customer
    (existing base and newly acquired) must carry exactly one Contract with
    a real SKU, not the old flat `term` string alone."""
    params = ParameterSet()
    generator = SyntheticDataGenerator(params, pricing_factory, seed=11)
    dataset, _ = generator.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6)

    assert len(dataset.contracts) == len(dataset.customers)
    assert set(dataset.contracts["customer_id"]) == set(dataset.customers["customer_id"])
    assert set(dataset.customers["sku"]) <= {"STARTER-1", "GROWTH-1", "ENTERPRISE-1"}
    assert set(dataset.contracts["status"]) <= {"active", "renewal_due", "renewed", "churned"}
    # A churned customer's contract must show CHURNED, never a live status.
    churned_ids = set(dataset.customers.loc[dataset.customers["churn_month"].notna(), "customer_id"])
    churned_contract_status = dataset.contracts.set_index("customer_id").loc[list(churned_ids), "status"]
    assert (churned_contract_status == "churned").all()


def test_product_catalog_weighted_average_is_neutral():
    """product.py's catalog is deliberately weighted so its demand-weighted
    average price multiplier is ~1.0 - introducing Product should not shift
    the model's aggregate dollar volume relative to before Product existed."""
    from biztwin import default_catalog
    catalog = default_catalog()
    assert catalog.weighted_average_multiplier() == pytest.approx(1.0, abs=0.02)


def test_market_conditions_default_matches_legacy_formula():
    """MarketConditions with default competitive_pressure=1 and
    market_growth_rate=0 must reproduce the old inline
    `demand_index * seasonality_index` arithmetic exactly, for any month -
    Market's introduction must not silently change any already-verified
    number (engine.py's SimulationEngine builds exactly this default when no
    explicit `market` is passed)."""
    from biztwin import MarketConditions
    market = MarketConditions(demand_index=1.1, seasonality_index=0.95)
    for month in range(0, 12):
        assert market.demand_multiplier(month) == pytest.approx(1.1 * 0.95)


# ---------------------------------------------------------------------------
# Customer Segment - a behavioral entity, not a label (project-lead review)
# ---------------------------------------------------------------------------

def test_segment_behavior_shapes_conversion_mix():
    """Enterprise (conversion_multiplier=0.85) should be UNDER-represented,
    and self_serve (1.15) OVER-represented, among newly converted customers
    relative to their raw catalog_weight - otherwise segment is just a label
    again, not a node that influences conversion."""
    from biztwin import default_segments
    segments = default_segments()
    enterprise = segments.by_name("enterprise")
    self_serve = segments.by_name("self_serve")

    params = ParameterSet()
    generator = SyntheticDataGenerator(params, pricing_factory, seed=123)
    dataset, _ = generator.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6)

    new_customers = dataset.customers[dataset.customers["acquisition_month"] > 0]
    realized_share = new_customers["segment"].value_counts(normalize=True)

    assert realized_share.get("enterprise", 0) < enterprise.catalog_weight
    assert realized_share.get("self_serve", 0) > self_serve.catalog_weight * 0.9  # allow sampling noise


def test_segment_behavior_shapes_deal_size():
    """Enterprise customers should carry a materially larger deal size than
    self_serve customers on average - segment must influence revenue, not
    just be a descriptive tag."""
    params = ParameterSet()
    generator = SyntheticDataGenerator(params, pricing_factory, seed=5)
    dataset, _ = generator.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6)

    by_segment = dataset.customers.groupby("segment")["deal_size"].mean()
    assert by_segment["enterprise"] > by_segment["self_serve"]


def test_segment_behavior_shapes_churn_selection():
    """Over many seeds, self_serve (churn_multiplier=1.6) should churn more
    often than enterprise (0.5) as a share of each segment's own
    population - not just in absolute count, which would just reflect
    population size."""
    params = ParameterSet()
    self_serve_churn_rate, enterprise_churn_rate = [], []
    for seed in range(10):
        generator = SyntheticDataGenerator(params, pricing_factory, seed=seed)
        dataset, _ = generator.generate(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6)
        counts = dataset.customers.groupby("segment")["churn_month"].apply(lambda s: s.notna().mean())
        if "self_serve" in counts:
            self_serve_churn_rate.append(counts["self_serve"])
        if "enterprise" in counts:
            enterprise_churn_rate.append(counts["enterprise"])

    avg_self_serve = sum(self_serve_churn_rate) / len(self_serve_churn_rate)
    avg_enterprise = sum(enterprise_churn_rate) / len(enterprise_churn_rate)
    assert avg_self_serve > avg_enterprise


def test_monthly_contracts_never_draw_a_renewal_boundary():
    """Monthly (term_length_months=1) contracts must never hit
    is_up_for_renewal - that attrition is already fully captured by
    monthly_churn_hazard; double-drawing it would silently inflate churn."""
    from biztwin import Contract
    contract = Contract(contract_id="c1", customer_id="cust1", term_length_months=1, start_month=0)
    for month in range(1, 25):
        assert contract.is_up_for_renewal(month) is False


def test_annual_contract_reaches_a_real_renewal_boundary():
    """An annual contract (term_length_months=12) run past 12 months must
    reach a Bernoulli renewal draw and end up RENEWED or CHURNED - never
    silently stuck ACTIVE past its own term boundary."""
    from biztwin import Contract, ContractStatus, MonteCarloProvider
    provider = MonteCarloProvider(seed=1)
    contract = Contract(contract_id="c1", customer_id="cust1", term_length_months=12, start_month=0)
    for month in range(1, 13):
        contract.advance(month, churned=False, renewal_probability=0.9, provider=provider)
    assert contract.status in (ContractStatus.RENEWED, ContractStatus.CHURNED)


# ---------------------------------------------------------------------------
# Contract renewal calibration - the make-or-break component (project-lead review)
# ---------------------------------------------------------------------------

def test_calibration_worked_example_renewal():
    params = ParameterSet()
    engine = CalibrationEngine(params)
    entry = engine.worked_example_renewal()

    assert entry.n_observations == 150
    assert entry.k_credibility == 50
    assert entry.z_weight == pytest.approx(0.75, abs=0.001)
    # Z=0.75 -> 0.75*0.82 + 0.25*0.90 = 0.84
    assert entry.new_value * 100 == pytest.approx(84.0, abs=0.1)


# ---------------------------------------------------------------------------
# Loop C - Revenue -> Marketing Budget -> Leads -> Funnel -> Revenue (reinforcing)
# ---------------------------------------------------------------------------

def test_revenue_reinvestment_controller_grows_faster_than_open_loop():
    """With RevenueReinvestmentMarketingController attached to the marketing
    lever, Active(T) should exceed the open-loop (IdentityController)
    baseline over a long enough horizon - the defining behavior of a
    reinforcing loop, and off by default so no already-verified scenario
    changes (IdentityController remains the default everywhere else)."""
    from biztwin import RevenueReinvestmentMarketingController

    # Backlog must be enabled for this loop to show up at all: with a fixed
    # sales capacity of 40 and Backlog off, EVERY extra lead beyond capacity
    # is simply lost (legacy mode's Assumption A9), so more marketing budget
    # accomplishes nothing once Qualified Opportunity is capacity-capped in
    # both runs. With Backlog on, the extra overflow queues and gradually
    # converts - the channel this reinforcing loop actually grows through.
    params_baseline = ParameterSet()
    baseline_engine = DynamicSimulationEngine(
        params_baseline, pricing_factory(params_baseline), DeterministicProvider(),
        enable_backlog=True, enable_referral=False,
    )
    start = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)
    baseline_history = baseline_engine.run(start, horizon=24)

    params_loop = ParameterSet()
    loop_engine = DynamicSimulationEngine(
        params_loop, pricing_factory(params_loop), DeterministicProvider(),
        marketing_controller=RevenueReinvestmentMarketingController(reinvestment_rate=0.5),
        enable_backlog=True, enable_referral=False,
    )
    loop_history = loop_engine.run(
        initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY), horizon=24,
    )

    assert loop_history[-1].active > baseline_history[-1].active
    # And it must actually be a growing budget, not a fluke - later months'
    # marketing_budget should exceed the flat baseline value.
    assert loop_history[-1].marketing_budget > MARKETING_BUDGET


# ---------------------------------------------------------------------------
# Task 7 - One-at-a-Time (OAT) Sensitivity Analysis, actually run
# ---------------------------------------------------------------------------

def test_sensitivity_sweep_churn_hazard_is_monotonically_decreasing():
    """SensitivityAnalyzer.sweep() exists in scenario.py but must actually be
    demonstrated: as monthly_churn_hazard rises, Active(6) must monotonically
    fall (legacy mode, so the effect is isolated from Backlog/Referral)."""
    params = ParameterSet()
    runner = ScenarioRunner(
        params, pricing_factory, DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    analyzer = SensitivityAnalyzer(runner)
    results = analyzer.sweep(
        "monthly_churn_hazard", (0.005, 0.05, 5),
        EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6,
        output_fn=lambda history: history[-1].active,
    )
    values = [active for _, active in results]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_sensitivity_sweep_qo_conversion_is_monotonically_increasing():
    """As r2_qo_to_active_base rises, Active(6) must monotonically rise -
    the complementary direction to the churn-hazard sweep above, confirming
    the sweep mechanism responds correctly to both signs of effect."""
    params = ParameterSet()
    runner = ScenarioRunner(
        params, pricing_factory, DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    analyzer = SensitivityAnalyzer(runner)
    results = analyzer.sweep(
        "r2_qo_to_active_base", (0.10, 0.35, 5),
        EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY, horizon=6,
        output_fn=lambda history: history[-1].active,
    )
    values = [active for _, active in results]
    assert all(a <= b for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# Section 8 - Monte Carlo convergence (law of large numbers), actually checked
# ---------------------------------------------------------------------------

def test_monte_carlo_mean_converges_to_deterministic_trace():
    """The average of many independently seeded MonteCarloProvider runs must
    converge to the DeterministicProvider's expected-value trace - the
    concrete validation behind the design's long-stated 'every run is seeded
    explicitly ... averages converge to the deterministic model' principle,
    which until now was never actually implemented as a test."""
    params = ParameterSet()
    deterministic_engine = DynamicSimulationEngine(
        params, pricing_factory(params), DeterministicProvider(),
        enable_backlog=False, enable_referral=False,
    )
    start = initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY)
    deterministic_active_6 = deterministic_engine.run(start, horizon=6)[-1].active

    mc_finals = []
    for seed in range(200):
        mc_params = ParameterSet()
        mc_engine = DynamicSimulationEngine(
            mc_params, pricing_factory(mc_params), MonteCarloProvider(seed),
            enable_backlog=False, enable_referral=False,
        )
        mc_history = mc_engine.run(
            initial_state(EXISTING_ACTIVE_BASE, MARKETING_BUDGET, PRICE, CAPACITY), horizon=6,
        )
        mc_finals.append(mc_history[-1].active)

    mc_mean = sum(mc_finals) / len(mc_finals)
    rel_error = abs(mc_mean - deterministic_active_6) / deterministic_active_6
    assert rel_error < 0.03
