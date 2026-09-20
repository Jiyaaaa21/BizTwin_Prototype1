"""
SyntheticDataGenerator - Final Design Section 8. Not a separate pipeline:
the identical DynamicSimulationEngine, configured with MonteCarloProvider,
producing the required outputs (Customers, Products, Sales Pipeline,
Subscriptions, Revenue Events, Contracts) by materializing individual
records from the same aggregate draws the engine already makes each month.

Also where the Product, Contract, and Customer Segment domain components
(product.py, contract.py, segment.py) actually get exercised: each
synthetic customer buys a real SKU from a ProductCatalog, carries a real
Contract with a calibrated, probabilistic renewal state machine, and
belongs to a CustomerSegment whose multipliers actually shape which
customers convert, which churn, whether they renew, and how much revenue
they carry - not just a descriptive label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .contract import Contract, ContractStatus
from .engine import DynamicSimulationEngine, initial_state
from .parameters import ParameterSet
from .pricing import PricingStrategy
from .product import ProductCatalog, default_catalog
from .randomness import MonteCarloProvider
from .segment import SegmentMix, default_segments


@dataclass
class SyntheticDataset:
    customers: pd.DataFrame
    products: pd.DataFrame
    sales_pipeline: pd.DataFrame
    subscriptions: pd.DataFrame
    revenue_events: pd.DataFrame
    contracts: pd.DataFrame


class SyntheticDataGenerator:
    def __init__(
        self,
        params: ParameterSet,
        pricing_factory: Callable[[ParameterSet], PricingStrategy],
        seed: int,
        enable_backlog: bool = True,
        enable_referral: bool = True,
        segments: SegmentMix | None = None,
        term_mix: dict[str, float] | None = None,
        catalog: ProductCatalog | None = None,
    ):
        self.params = params
        self.pricing_factory = pricing_factory
        self.provider = MonteCarloProvider(seed)
        self.enable_backlog = enable_backlog
        self.enable_referral = enable_referral
        self.segments = segments or default_segments()
        self.term_mix = term_mix or {"monthly": 0.7, "annual": 0.3}
        self.catalog = catalog or default_catalog()

    def _draw_categorical(self, mix: dict[str, float]) -> str:
        labels, weights = zip(*mix.items())
        return str(self.provider._rng.choice(labels, p=weights))

    def generate(
        self,
        existing_active_base: float,
        marketing_budget: float,
        price: float,
        capacity: float,
        horizon: int,
    ) -> tuple[SyntheticDataset, list]:
        pricing = self.pricing_factory(self.params)
        engine = DynamicSimulationEngine(
            params=self.params, pricing=pricing, provider=self.provider,
            enable_backlog=self.enable_backlog, enable_referral=self.enable_referral,
        )
        start = initial_state(existing_active_base, marketing_budget, price, capacity)
        history = engine.run(start, horizon=horizon)

        customers: list[dict] = []
        subscriptions: list[dict] = []
        revenue_events: list[dict] = []
        pipeline_rows: list[dict] = []
        active_pool: list[dict] = []
        contracts: dict[str, Contract] = {}
        p = self.params.snapshot_values()
        renewal_probability_base = p["renewal_probability"]

        def make_contract(cid: str, term: str, start_month: int) -> Contract:
            term_length = 12 if term == "annual" else 1
            contract = Contract(
                contract_id=f"contract-{cid}", customer_id=cid,
                term_length_months=term_length, start_month=start_month,
            )
            contracts[cid] = contract
            return contract

        def segment_renewal_probability(segment_name: str) -> float:
            multiplier = self.segments.by_name(segment_name).renewal_multiplier
            return max(0.0, min(1.0, renewal_probability_base * multiplier))

        # Month-0 existing base: individually materialized so revenue events
        # reconcile against Active(0) x Price / 12 exactly (Sec. 8's built-in
        # consistency check). Product price multipliers and Segment deal-size
        # multipliers are each chosen so their demand-weighted averages land
        # close to 1.0 (product.py, segment.py), so this reconciliation still
        # holds within its documented tolerance even though customers now buy
        # different SKUs, in different segments, at different price points.
        for i in range(int(round(existing_active_base))):
            cid = f"existing-{i}"
            segment = self.segments.sample_baseline(self.provider._rng)
            sku = self.catalog.sample(self.provider._rng)
            deal_size = self.provider.draw_magnitude(
                mean=price * sku.price_multiplier * segment.deal_size_multiplier
            )
            term = self._draw_categorical(self.term_mix)
            record = {"customer_id": cid, "segment": segment.name, "cohort_id": "pre-existing",
                      "acquisition_month": 0, "deal_size": deal_size, "term": term,
                      "sku": sku.sku, "product_tier": sku.tier, "churn_month": None}
            customers.append(record)
            subscriptions.append({
                "subscription_id": f"sub-{cid}", "customer_id": cid, "mrr": deal_size / 12,
                "arr": deal_size, "start_month": 0, "term_length": term, "status": "active",
            })
            active_pool.append({"id": cid, "deal_size": deal_size, "segment": segment.name})
            make_contract(cid, term, start_month=0)

        for step in history[1:]:
            month = step.month
            n_churn = min(int(round(step.churned)), len(active_pool))
            if n_churn > 0:
                # Segment influences churn: candidates are selected weighted
                # by their segment's churn_multiplier, not uniformly - a
                # higher-churn segment is genuinely over-represented among
                # this month's churned population.
                churn_weights = self.segments.churn_weights([m["segment"] for m in active_pool])
                idxs = self.provider._rng.choice(
                    len(active_pool), size=n_churn, replace=False, p=churn_weights,
                )
                for idx in sorted(idxs, reverse=True):
                    member = active_pool.pop(idx)
                    for c in customers:
                        if c["customer_id"] == member["id"]:
                            c["churn_month"] = month
                    for s in subscriptions:
                        if s["customer_id"] == member["id"]:
                            s["status"] = "churned"
                    contracts[member["id"]].advance(month, churned=True)

            n_new = int(round(step.new_active))
            for j in range(n_new):
                cid = f"c-{month}-{j}"
                # Segment influences conversion: new customers are drawn
                # weighted by catalog_weight x conversion_multiplier, so a
                # segment with an easier/faster path to conversion is
                # genuinely over-represented among this month's new Active
                # customers (see segment.py's sample_for_new_customer).
                segment = self.segments.sample_for_new_customer(self.provider._rng)
                sku = self.catalog.sample(self.provider._rng)
                deal_size = self.provider.draw_magnitude(
                    mean=step.price * sku.price_multiplier * segment.deal_size_multiplier
                )
                term = self._draw_categorical(self.term_mix)
                customers.append({"customer_id": cid, "segment": segment.name,
                                   "cohort_id": f"{month}-{segment.name}", "acquisition_month": month,
                                   "deal_size": deal_size, "term": term,
                                   "sku": sku.sku, "product_tier": sku.tier, "churn_month": None})
                subscriptions.append({
                    "subscription_id": f"sub-{cid}", "customer_id": cid, "mrr": deal_size / 12,
                    "arr": deal_size, "start_month": month, "term_length": term, "status": "active",
                })
                active_pool.append({"id": cid, "deal_size": deal_size, "segment": segment.name})
                make_contract(cid, term, start_month=month)

            renewal_churned_ids: list[str] = []
            for member in active_pool:
                # Segment influences renewal: the Contract's own calibrated
                # renewal_probability is scaled by this customer's segment
                # before the Bernoulli draw at each term boundary. Only
                # annual contracts ever reach a boundary here (see
                # Contract.is_up_for_renewal) - a failed renewal is a real
                # churn event, handled exactly like one below.
                contracts[member["id"]].advance(
                    month, churned=False,
                    renewal_probability=segment_renewal_probability(member["segment"]),
                    provider=self.provider,
                )
                if contracts[member["id"]].status == ContractStatus.CHURNED:
                    renewal_churned_ids.append(member["id"])
                else:
                    revenue_events.append({
                        "event_id": f"rev-{member['id']}-{month}", "customer_id": member["id"],
                        "month": month, "amount": member["deal_size"] / 12,
                    })

            if renewal_churned_ids:
                churned_id_set = set(renewal_churned_ids)
                for c in customers:
                    if c["customer_id"] in churned_id_set:
                        c["churn_month"] = month
                for s in subscriptions:
                    if s["customer_id"] in churned_id_set:
                        s["status"] = "churned"
                active_pool[:] = [m for m in active_pool if m["id"] not in churned_id_set]

            pipeline_rows.extend([
                {"month": month, "stage": "lead", "count": step.leads,
                 "win_probability": p["win_prob_lead"],
                 "value": step.leads * step.price * p["win_prob_lead"]},
                {"month": month, "stage": "qualified_opportunity", "count": step.qualified_opportunity,
                 "win_probability": p["win_prob_qo"],
                 "value": step.qualified_opportunity * step.price * p["win_prob_qo"]},
                {"month": month, "stage": "backlog", "count": step.backlog,
                 "win_probability": p["win_prob_qo"] * p["backlog_staleness_discount"],
                 "value": step.backlog * step.price * p["win_prob_qo"] * p["backlog_staleness_discount"]},
            ])

        products = pd.DataFrame([
            {"product_id": sku.sku, "name": sku.name, "tier": sku.tier,
             "features": ", ".join(sku.features), "value_proposition": sku.value_proposition,
             "price_multiplier": sku.price_multiplier, "catalog_weight": sku.catalog_weight,
             "pricing_strategy": type(pricing).__name__}
            for sku in self.catalog.products
        ])

        contract_rows = [
            {"contract_id": c.contract_id, "customer_id": c.customer_id,
             "term_length_months": c.term_length_months, "start_month": c.start_month,
             "status": c.status.value, "renewals": c.renewals}
            for c in contracts.values()
        ]

        dataset = SyntheticDataset(
            customers=pd.DataFrame(customers),
            products=products,
            sales_pipeline=pd.DataFrame(pipeline_rows),
            subscriptions=pd.DataFrame(subscriptions),
            revenue_events=pd.DataFrame(revenue_events),
            contracts=pd.DataFrame(contract_rows),
        )
        return dataset, history
