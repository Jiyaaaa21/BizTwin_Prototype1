"""
Market - a business-domain component (node) in the Final Design's causal
graph: Market -> Customer Segment -> Sales Funnel. Before this module,
external demand and seasonality were two bare multipliers
(`demand_index`, `seasonality_index`) used directly inline inside
`engine.py`'s Leads formula - there was no Market a reviewer could point to
as its own node, only two loose parameters. This gives them a first-class
home, and a place for market growth / competitive pressure to live, without
changing any number the model already produces (see `demand_multiplier`'s
default arguments below).

Market is intentionally one-directional in Prototype 1: it influences Leads,
nothing in the business feeds back into it. A real competitive-response
loop (the business's own growth attracting competitors, which then erodes
demand) is a genuine future-phase extension, not fabricated here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketConditions:
    demand_index: float        # exogenous baseline demand multiplier (Final Design Sec. 7)
    seasonality_index: float   # exogenous seasonal multiplier (Final Design Sec. 7)
    competitive_pressure: float = 1.0  # >1 dampens demand (more competition); 1.0 = neutral, unmodeled by default
    market_growth_rate: float = 0.0    # monthly compounding drift; 0.0 = flat market, unmodeled by default

    def demand_multiplier(self, months_elapsed: int) -> float:
        """
        The combined exogenous multiplier applied to marketing-sourced lead
        volume this month. With the defaults above (competitive_pressure=1,
        market_growth_rate=0) this returns exactly `demand_index *
        seasonality_index` for every month - identical to the pre-Market
        arithmetic - so introducing this component changes nothing about
        any already-verified number until a scenario deliberately sets a
        non-default growth rate or competitive pressure.
        """
        growth = (1.0 + self.market_growth_rate) ** months_elapsed
        return (self.demand_index * self.seasonality_index * growth) / self.competitive_pressure
