# biztwin - Prototype 1 (Commercial & Sales Domain)

A deterministic-first, scientifically calibrated Business Model Digital Twin
for the Commercial & Sales domain of a SaaS subscription business. This
implements the **Final Prototype 1 Design (Definitive Specification)** in
the project's Claude Docs artifact exactly as approved, section by section.

AI is deliberately absent from this codebase. Every state transition,
controller, and calibration rule is closed-form and explainable - no
optimization engine (no pymoo/NSGA-II), no reinforcement learning, no
neural surrogates. This is a hard exclusion the Final Design carries
forward from the project's Guiding Principles, not an oversight.

## Project layout

```
biztwin_project/
├── backend/                     # the deterministic Digital Twin core - everything the design doc specifies
│   ├── biztwin/                 # the package: engine, state machines, calibration, scenarios, metrics
│   │   ├── engine.py            # SimulationEngine (Static + Dynamic), StepResult, initial_state
│   │   ├── funnel_state.py      # Sales Funnel state machine (Lead -> QO -> Backlog -> Active -> Churned)
│   │   ├── contract.py          # Contract/Subscription state machine (Active -> RenewalDue/Renewed -> Churned)
│   │   ├── market.py            # Market component (demand/seasonality/competition)
│   │   ├── segment.py           # Customer Segment component (Enterprise/SMB/Self-Serve behavior)
│   │   ├── product.py           # Product component (SKU catalog)
│   │   ├── controllers.py       # Feedback-loop Controllers (Marketing, Capacity, Pricing, Revenue Reinvestment)
│   │   ├── pricing.py           # Pricing strategies
│   │   ├── randomness.py        # Deterministic vs. Monte Carlo provider (Strategy pattern)
│   │   ├── parameters.py        # Every model coefficient, sourced and confidence-tagged
│   │   ├── calibration.py       # Bayesian credibility-weighted CalibrationEngine
│   │   ├── scenario.py          # ScenarioRunner + SensitivityAnalyzer
│   │   ├── synthetic.py         # SyntheticDataGenerator (same engine, Monte Carlo mode)
│   │   └── metrics.py           # Read-only emergent metrics (ARR, MRR, churn rate, ...)
│   ├── examples/
│   │   └── run_verification.py  # narrated walkthrough, eleven checks - `python backend/examples/run_verification.py`
│   ├── tests/
│   │   └── test_reproducibility.py   # 30 automated regression tests - `pytest backend/tests/ -v`
│   └── requirements.txt         # numpy, pandas, pytest
├── frontend/
│   ├── dashboard_app.py         # Streamlit UI - wires widgets to the backend engine, no logic of its own
│   └── requirements.txt         # streamlit, pandas, numpy
└── README.md
```

The split is a packaging convenience, not an architectural one: `frontend/`
contains zero business logic - every number it shows comes from calling
into `backend/biztwin/` live. This keeps the hard boundary the project
mandate requires (deterministic core vs. any presentation/future-AI layer)
visible in the folder structure itself, not just in code comments.

### Quickstart

```bash
# from the project root
pip install -r backend/requirements.txt -r frontend/requirements.txt

# backend: run the narrated verification walkthrough
python backend/examples/run_verification.py

# backend: run the automated regression suite
pytest backend/tests/test_reproducibility.py -v

# frontend: launch the live dashboard (imports backend/biztwin directly)
streamlit run frontend/dashboard_app.py
```

## The six business-domain components

Per the project lead's explicit guidance, Prototype 1's "5-6 components"
are business-domain nodes (Market, Customer Segment, Product, Sales
Funnel, Contract/Subscription, Revenue Engine), not software-architecture
modules - see the "Business-Domain Component View" section in the Claude
Docs design doc for the full causal graph and direct/indirect effects.
The module map below still lists every file; these three are the ones
this reframing added or upgraded from a stub:

| Module | Component | What it implements |
|---|---|---|
| `market.py` | Market | `MarketConditions` - gives `demand_index`/`seasonality_index` (previously two bare multipliers inline in `engine.py`) a first-class home, plus `competitive_pressure` and `market_growth_rate` hooks. Defaults reproduce the old inline arithmetic exactly - introducing Market changed no previously-verified number. |
| `product.py` | Product | `ProductCatalog` of three real SKUs (Starter/Growth/Enterprise), each a `price_multiplier` on the Price lever rather than an independent price. Replaces the single hardcoded "Core Subscription" row `synthetic.py` used to emit. |
| `contract.py` | Contract / Subscription | `Contract` - a real renewal state machine (`ACTIVE → RENEWAL_DUE / RENEWED → CHURNED`), driven by a calibrated `renewal_probability` Bernoulli draw at genuine term boundaries (annual only - see below), stepped monthly per active customer in `synthetic.py`. Replaces the flat `term` string that used to be the only trace of a subscription's lifecycle. |
| `segment.py` | Customer Segment | `SegmentMix` - Enterprise/SMB/Self-Serve each carry real conversion, churn, renewal, and deal-size multipliers (not just a descriptive label). See "Customer Segment is a behavioral entity" below. |

Dynamic Pricing (`pricing.py`'s `DynamicPricingStrategy`) remains an
explicit, unbuilt placeholder - deprioritized by the project lead as an
optimization-flavored component, not a state/causality one.

### Second review round - five challenges addressed

A follow-up architecture review challenged five things about the six-node
reframing above. All five are addressed in code (not just prose) and
covered by `backend/tests/test_reproducibility.py` (30 tests) and
`backend/examples/run_verification.py`:

1. **Customer Segment is a behavioral entity, not a label.** `segment.py`'s
   `SegmentMix` gives Enterprise/SMB/Self-Serve real multipliers - Enterprise
   converts more slowly (0.85x) but churns less (0.5x) and renews more
   (1.15x) at 1.5x the deal size; Self-Serve is the mirror image. These
   multipliers actually shape which segment each month's new customers come
   from, which segment is over-represented among churn, each segment's
   renewal odds, and deal size - not just a stored string. Scope note,
   stated explicitly: this acts at the individual-agent synthetic-data
   layer; the aggregate deterministic engine stays one population for
   Prototype 1 (splitting it into per-segment cohorts is a Prototype 2
   extension, not attempted here to avoid re-deriving every already-verified
   Phase 4/5/9 number).
2. **Revenue Engine is documented as an aggregation layer, not a business
   entity.** `metrics.py` holds no state; if asked "is Revenue a stateful
   business entity," the documented answer is no, by construction.
3. **A genuine feedback loop was added - Loop C (reinforcing).**
   `controllers.py`'s `RevenueReinvestmentMarketingController` closes
   Revenue → Marketing Budget → Leads → Funnel → Revenue, computing its own
   "implied MRR" from `active x price` rather than calling `metrics.py` (so
   the "metrics never feed back into state" boundary stays intact even while
   demonstrating the loop). Off by default - every scenario and the
   dashboard still use the open-loop `IdentityController`.
4. **Calibration extended to a second parameter - Contract's renewal.**
   `renewal_probability` is now a real calibrated `Parameter`, and
   `CalibrationEngine.worked_example_renewal()` reproduces the requested
   story with the same Bayesian mechanism as lead conversion: prior 90%,
   observed 82% (n=150, k=50), Z=0.75 → calibrated 84.0%. Fixed along the
   way: a monthly contract must never draw a renewal boundary at all
   (`term_length_months <= 1` → `is_up_for_renewal()` is always `False`) -
   otherwise every month would double-count churn on top of
   `monthly_churn_hazard`; only a longer-than-monthly term (annual) has a
   genuine, separately-calibrated renewal decision.
5. **Sales Funnel and Contract are two separate state diagrams**, never
   merged - see the "Second Project-Lead Review - Response" section of the
   Claude Docs design doc for both `stateDiagram-v2` blocks side by side
   with the reasoning for keeping acquisition and retention lifecycles apart.

The 10% synthetic-data revenue-reconciliation tolerance is unchanged and,
per this review, considered right-sized rather than loose: it reflects
compounding stochastic variance from integer rounding, SKU draw, and
segment draw - not model error - and is documented inline wherever it's
asserted.

### Two closed gaps - Sensitivity Analysis and Monte Carlo convergence

A checklist pass against the project lead's own meeting notes surfaced two
places where a validation method was designed and named but never actually
run. Both are now demonstrated with real numbers, not just present as
unused classes:

1. **One-at-a-Time (OAT) Sensitivity Analysis, actually run.**
   `scenario.py`'s `SensitivityAnalyzer.sweep()` existed since the design
   phase but had never been exercised. `backend/examples/run_verification.py` now
   sweeps two parameters in legacy mode: `monthly_churn_hazard` over
   (0.5%, 5%, 5 steps) - Active(6) monotonically *decreases*, 200.7 → 158.5
   - and `r2_qo_to_active_base` over (10%, 35%, 5 steps) - Active(6)
   monotonically *increases*, 159.3 → 225.2. (Lead-volume parameters like
   `r1_lead_to_qo` and `referral_rate` were deliberately not chosen for this
   demonstration: at the default capacity=40, Qualified Opportunity is
   already capacity-capped at month 0 - 250 leads x 30% = 75 > 40 - so a
   lead-volume sweep would show no effect on Active(T) regardless of the
   true parameter sensitivity, an artifact of the capacity ceiling rather
   than evidence the parameter doesn't matter.) `backend/tests/test_reproducibility.py`
   asserts both monotonicity directions as regression tests.
2. **Monte Carlo convergence validation, actually checked.** The design has
   stated since early phases that "every run is seeded explicitly ...
   averages converge to the deterministic model" - this was a principle,
   never a test. `backend/examples/run_verification.py` now runs 200 independently
   seeded `MonteCarloProvider` simulations (legacy mode, horizon 6), takes
   the mean of their final Active values, and compares it against the
   `DeterministicProvider` trace's Active(6): the two agree to within 0.12%
   in a representative run, well inside the asserted 3% bound - the law of
   large numbers demonstrably holds for this model, not just asserted in
   prose. Covered by a matching regression test.

## Module map - code to Final Design section

| Module | Final Design section | What it implements |
|---|---|---|
| `parameters.py` | Section 7 (Calibration Framework), Entity Catalog | `Parameter` (value + range + confidence + source + calibration/validation method) and `ParameterSet`, the single source of truth for every coefficient in the model. Nothing is a magic number - every field cites where it came from and how confident we are in it. |
| `randomness.py` | Modeling Paradigm Selection | `RandomnessProvider` (Strategy pattern): `DeterministicProvider` runs the exact expected-value arithmetic (used for every worked example and scenario comparison); `MonteCarloProvider` draws from Poisson/Binomial/Lognormal distributions matched to each quantity's type (used only for synthetic data generation). One set of formulas, two ways of evaluating them. |
| `pricing.py` | Entity Catalog / Causal Graph (price -> conversion) | `PricingStrategy` (Strategy pattern): `TieredPricingStrategy` implements the price-elasticity conversion-rate adjustment `r2(Price) = base_rate * (Price/reference_price)^(-beta)`; `DynamicPricingStrategy` is a usage-based placeholder, flagged in the Architecture Review as ahead-of-need. |
| `funnel_state.py` | State Transition Design | `FunnelState` (State pattern): `LeadState`, `QualifiedOpportunityState`, `BacklogState`, `ActiveState`, `ChurnedState`, each owning its own entry/exit condition and transition probability. `Customer` is the individual agent that carries a `cohort_id` for cohort-level reconciliation (Decision 4). |
| `controllers.py` | Directed Causal Graph (feedback loops), Architectural Reframing | `Controller` (Strategy pattern): `IdentityController` (open-loop, reproduces every original Phase 9 scenario), `MarketingController` (threshold/proportional rule on Backlog/utilization), `CapacityController` (hysteresis-based hiring trigger with real memory), `PricingController` (open-loop by design). Controllers are first-class feedback *policies*, never an optimizer searching for a "best" lever value. |
| `engine.py` | Static State Model, Dynamic State Model, Final Architecture Diagram | `SimulationEngine` (Template Method pattern): one `_compute_step` pipeline shared by `StaticSimulationEngine` (single frozen month-0 snapshot) and `DynamicSimulationEngine` (loops the identical pipeline across months). `StepResult` holds *only* true dynamical state - no metric is stored here (see the Architectural Reframing: "Metrics Are Emergent, Not Modeled"). |
| `metrics.py` | Architectural Reframing (metrics as emergent outputs) | Pure, read-only functions over a `StepResult`: `arr`, `mrr`, `churn_rate`, `conversion_rate`, `pipeline_value`, `expected_sales_cycle_length`. Nothing here ever writes back into the state - metrics cannot influence the simulation, by construction. |
| `calibration.py` | Section 7 (Calibration Framework) | `CalibrationEngine`: Bayesian credibility-weighted blending, `Z = n/(n+k)`, mathematically identical to Beta-Binomial conjugate posterior updating. Every adjustment returns a *new* `ParameterSet` and is logged (`CalibrationLogEntry`) - never a silent in-place overwrite. |
| `scenario.py` | Section 9 (Scenario Engine Design) | `ScenarioRunner` (composition over `SimulationEngine`, not a new pipeline): every scenario shares an identical month-0 baseline snapshot, and lever overrides (price/marketing/capacity) take effect starting month 1 - a decision does not retroactively re-price a month already observed. `SensitivityAnalyzer` runs One-at-a-Time (OAT) sweeps, with zero optimization content. |
| `synthetic.py` | Section 8 (Synthetic Data Framework) | `SyntheticDataGenerator`: the *same* `DynamicSimulationEngine`, run with `MonteCarloProvider`, materializing individual customer/subscription/revenue/pipeline records from the aggregate draws the engine already makes each month. Not a separate data-faking pipeline - this is the Digital Twin's own mechanism used generatively. |

## Two operating modes

Every check in `backend/examples/run_verification.py` and `backend/tests/test_reproducibility.py`
exercises both:

- **Legacy mode** (`enable_backlog=False, enable_referral=False`) reproduces
  the exact worked numeric examples from the Final Design's Phase 4/5/9
  sections, computed before the Backlog (balancing loop) and Referral
  (reinforcing loop) mechanics existed. This is the regression test that
  the core arithmetic - leads, capacity cap, conversion, churn - is
  implemented exactly as documented.
- **Final Design mode** (both enabled, the shipped default) is the actual
  Prototype 1 model: overflow queues into a decaying Backlog instead of
  being discarded, and the existing Active base regenerates new leads via
  referral. Numbers here intentionally differ from legacy mode - that's the
  two closed feedback loops (Sections 5-6) doing their job.

## Verification

```bash
python backend/examples/run_verification.py   # narrated walkthrough, eleven checks
pytest backend/tests/test_reproducibility.py -v   # 30 automated regression tests
```

Both must pass before any change to `backend/biztwin/` is considered complete. The
tests assert, in order: the Phase 4 static baseline; the Phase 5 dynamic
trace (months 0-2); the Phase 4/5 static-vs-dynamic boundary reconciliation
(ARR(0) must match exactly across both engines); the Phase 9 four-scenario
comparison at month 6, including that all four scenarios share an identical
month-0 baseline; the Section 7 Bayesian calibration worked example
(15% prior, 18% observed, n=200, k=75 -> 17.2% calibrated) and that
calibration never mutates a `ParameterSet` in place; the Contract renewal
calibration worked example (90% prior, 82% observed, n=150, k=50 -> 84.0%
calibrated); that a monthly contract never draws a renewal boundary while
an annual one does; that Loop C (revenue reinvestment) genuinely outgrows
the open-loop baseline; that Customer Segment multipliers actually shift
conversion mix, deal size, and churn selection; a Final Design mode smoke
test; Monte Carlo synthetic-data reproducibility (same seed -> identical
dataset) plus revenue reconciliation against `Active x Price / 12`; the OAT
sensitivity sweep's two monotonicity directions (churn hazard down-shifts
Active(6), QO->Active conversion up-shifts it); and that the mean of 200
seeded Monte Carlo runs converges to the deterministic expected-value trace
within 3%.

## Live dashboard (real engine, not a mockup)

`dashboard_app.py` is a Streamlit front end that imports `biztwin` unchanged
and calls the same `DynamicSimulationEngine` / `ScenarioRunner` /
`CalibrationEngine` / `SyntheticDataGenerator` / `SensitivityAnalyzer`
classes the verification script does. Every chip, chart and table on it
recomputes live from whatever inputs you set - it is the actual Digital
Twin, viewed, not a static picture of one. (A separate, purely visual
wireframe of this dashboard's layout also exists as a Claude Artifact, for
design review before this app was built; this file is what you actually
run.)

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/dashboard_app.py
```

Five tabs cover the full Prototype 1 surface, not just the original core:

- **Overview** - KPIs, the Active/ARR trace, and the month-N funnel, all
  driven by the sidebar's business inputs, including live Market
  conditions (demand/seasonality/competitive pressure/growth) and an
  optional Loop C (revenue reinvestment) toggle.
- **Components** - a behavior lab, not a set of static tables: turn a knob
  on Market, Product, or Customer Segment and watch the same seeded run
  react downstream, end to end. Market compares the current sidebar
  conditions against a neutral market on two full engine runs. Product
  re-prices one SKU and shows the resulting revenue-by-SKU mix from a live
  `SyntheticDataGenerator` run. Customer Segment exposes one segment's
  conversion/churn/renewal/deal-size multipliers as sliders and shows a
  before/after scoreboard (customers, revenue share, churn rate) for every
  segment, proving the change is population-wide, not local. Contract
  joins that same adjusted run's Contract records back to Segment and
  reports the renewal rate by segment at actual renewal boundaries
  (`elapsed % term_length_months == 0`), the same test `Contract.
  is_up_for_renewal()` uses - so an ordinary mid-term churn from
  `monthly_churn_hazard` is never miscounted as a failed renewal.
- **Scenario Comparison** - define B/C/D lever overrides and see all four
  scenarios re-run live against a shared month-0 baseline, with Loop C
  applied consistently across all four when enabled.
- **Calibration** - both worked examples side by side: lead conversion AND
  Contract renewal, the same Bayesian credibility-blend mechanism on two
  parameters, plus the full live `ParameterSet` with its confidence tags.
- **Validation** - Sensitivity Analysis (pick a parameter, sweep a range,
  see the monotonic Active(T) response) and the Monte Carlo convergence
  check (run N seeded simulations live and watch the mean converge to the
  deterministic trace), both run interactively rather than only in
  `run_verification.py`.

No AI, optimizer, or search selects or ranks anything here - it is the
deterministic core, wired to widgets.

## What this is not

No forecasting, no genetic algorithms, no reinforcement learning, no neural
surrogate models, and no code here treats ARR/churn/conversion as anything
other than a read of `StepResult` taken after the fact. If a future phase
adds AI, it is restricted by the project's own AI Requirements to
forecasting or pattern discovery *in addition to* this deterministic core -
it may never replace or control a state transition.