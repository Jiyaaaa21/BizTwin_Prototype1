"""
biztwin - Prototype 1 Commercial & Sales simulation core.

Implements the Final Prototype 1 Design (Definitive Specification) exactly:
one deterministic state-transition core (engine.py), a State-pattern funnel
(funnel_state.py), Strategy-pattern pricing and randomness (pricing.py,
randomness.py), first-class feedback-aware Controllers (controllers.py), a
Bayesian credibility-weighted CalibrationEngine (calibration.py), a
ScenarioRunner/SensitivityAnalyzer (scenario.py), a SyntheticDataGenerator
that reuses the same engine (synthetic.py), and a strictly read-only
MetricsObserver (metrics.py).

Also implements the six business-domain components (nodes) of the causal
graph, per the project lead's guidance: Market (market.py), Customer
Segment (segment.py - a behavioral entity, not a label: it shapes
conversion, churn, renewal, and revenue), Product (product.py), Sales
Funnel (funnel_state.py), Contract/Subscription (contract.py), Revenue
Engine (metrics.py - explicitly an aggregation layer, never a stateful
business entity in its own right). Market -> Customer Segment -> Sales
Funnel and Product -> Sales Funnel are the two inbound edges; Sales Funnel
-> Contract/Subscription -> Revenue Engine is the outbound chain. A third,
reinforcing loop (Revenue -> Marketing Budget -> Leads -> Funnel -> Revenue)
is available as an opt-in Controller (controllers.py's
RevenueReinvestmentMarketingController) - off by default so no
previously-verified scenario changes.

No optimization, no genetic algorithms, no reinforcement learning, no
neural surrogates - per the project's Guiding Principles and Decision 1/2.
Dynamic Pricing (pricing.py's DynamicPricingStrategy) remains an unbuilt
placeholder by explicit instruction - it is an optimization-flavored
component the project lead deprioritized in favor of Product, Contract,
and Market.
"""
from .calibration import CalibrationEngine, CalibrationLogEntry
from .contract import Contract, ContractStatus
from .controllers import (CapacityController, Controller, IdentityController,
                           MarketingController, PricingController,
                           RevenueReinvestmentMarketingController)
from .engine import (DynamicSimulationEngine, SimulationEngine,
                      StaticSimulationEngine, StepResult, initial_state)
from .funnel_state import Customer, FunnelState, STATES
from .market import MarketConditions
from .parameters import Confidence, Parameter, ParameterSet
from .pricing import DynamicPricingStrategy, PricingStrategy, TieredPricingStrategy
from .product import Product, ProductCatalog, default_catalog
from .randomness import DeterministicProvider, MonteCarloProvider, RandomnessProvider
from .scenario import ScenarioResult, ScenarioRunner, ScenarioSpec, SensitivityAnalyzer
from .segment import CustomerSegment, SegmentMix, default_segments
from .synthetic import SyntheticDataGenerator, SyntheticDataset
from . import metrics

__all__ = [
    "CalibrationEngine", "CalibrationLogEntry",
    "Contract", "ContractStatus",
    "CapacityController", "Controller", "IdentityController",
    "MarketingController", "PricingController", "RevenueReinvestmentMarketingController",
    "DynamicSimulationEngine", "SimulationEngine", "StaticSimulationEngine",
    "StepResult", "initial_state",
    "Customer", "FunnelState", "STATES",
    "MarketConditions",
    "Confidence", "Parameter", "ParameterSet",
    "DynamicPricingStrategy", "PricingStrategy", "TieredPricingStrategy",
    "Product", "ProductCatalog", "default_catalog",
    "DeterministicProvider", "MonteCarloProvider", "RandomnessProvider",
    "ScenarioResult", "ScenarioRunner", "ScenarioSpec", "SensitivityAnalyzer",
    "CustomerSegment", "SegmentMix", "default_segments",
    "SyntheticDataGenerator", "SyntheticDataset",
    "metrics",
]
