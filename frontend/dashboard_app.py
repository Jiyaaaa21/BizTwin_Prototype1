"""
Live Digital Twin dashboard - Streamlit front end over the real `biztwin`
engine (not a mockup: every number on screen comes from re-running
DynamicSimulationEngine / ScenarioRunner / CalibrationEngine / Synthetic-
DataGenerator / SensitivityAnalyzer against whatever inputs you set, live).

This is a thin viewer: it imports `biztwin` unchanged and calls the same
classes `backend/examples/run_verification.py` and
`backend/tests/test_reproducibility.py` exercise. No business logic lives
in this file - only widgets, layout, and plotting.

Run with (from the project root):
    pip install -r frontend/requirements.txt
    streamlit run frontend/dashboard_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import dataclasses

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from biztwin import (
    CalibrationEngine, CustomerSegment, DeterministicProvider,
    DynamicSimulationEngine, MarketConditions, MonteCarloProvider,
    ParameterSet, ProductCatalog, RevenueReinvestmentMarketingController,
    ScenarioRunner, ScenarioSpec, SegmentMix, SensitivityAnalyzer,
    SyntheticDataGenerator, TieredPricingStrategy, default_catalog,
    default_segments, initial_state, metrics,
)

# ---------------------------------------------------------------------------
# Palette - fixed categorical order + a sequential blue ramp for ordered
# stages (funnel). Kept in one place so every chart on the page draws from
# the same small, validated set of colors instead of framework defaults.
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, RED = "#4a3aa7", "#e34948"
INK, MUTED, GRID = "#0f0f10", "#7a7873", "#e9e8e3"
CARD_BORDER = "rgba(15, 15, 16, 0.08)"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]
FUNNEL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]  # light -> dark, ordered

st.set_page_config(page_title="Commercial & Sales Digital Twin", page_icon="📈", layout="wide")

st.markdown(f"""
<style>
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
    html, body, [class*="css"] {{
        font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    }}
    h1, h2 {{ font-weight: 700; letter-spacing: -0.01em; color: {INK}; }}
    .twin-section-title {{
        font-size: 1.05rem !important; font-weight: 700 !important; letter-spacing: -0.01em;
        color: {INK}; margin-bottom: 0.2rem; line-height: 1.3;
    }}

    /* Header ------------------------------------------------------------ */
    .twin-header-row {{ display: flex; align-items: center; justify-content: space-between; }}
    .twin-title {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: {INK}; margin: 0; }}
    .twin-subtitle {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.15rem; }}
    .twin-pill {{
        display: inline-block; font-size: 0.72rem; font-weight: 600; color: {BLUE};
        background: rgba(42, 120, 214, 0.08); border-radius: 999px;
        padding: 5px 12px; white-space: nowrap;
    }}

    /* Section headings inside cards -------------------------------------- */
    .twin-eyebrow {{
        color: {MUTED}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 2px;
    }}
    .twin-hint {{ color: {MUTED}; font-size: 0.83rem; margin-top: 2px; }}

    /* Cards --------------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px !important;
        border: 1px solid {CARD_BORDER} !important;
        box-shadow: 0 1px 3px rgba(15, 15, 16, 0.04);
        background: #ffffff;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {{
        gap: 0.6rem;
    }}

    /* Metrics --------------------------------------------------------------- */
    div[data-testid="stMetric"] {{
        background: rgba(42, 120, 214, 0.045);
        border: 1px solid {CARD_BORDER};
        border-radius: 10px;
        padding: 12px 16px 8px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.8rem; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; color: {INK}; }}

    /* Tabs -------------------------------------------------------------- */
    button[data-testid="stTab"] {{ font-weight: 600; font-size: 0.95rem; }}

    /* Sidebar ------------------------------------------------------------ */
    section[data-testid="stSidebar"] {{ background: #fafaf9; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.8rem; }}
    .twin-sidebar-label {{
        color: {MUTED}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; margin: 1.1rem 0 0.3rem 0;
    }}
    .twin-badge {{
        display: block; text-align: center; font-size: 0.72rem; color: {MUTED};
        border: 1px solid {CARD_BORDER}; border-radius: 8px;
        padding: 7px 10px; margin-top: 1.4rem; background: #fff;
    }}
</style>
""", unsafe_allow_html=True)


def styled_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    """One shared look for every chart: thin lines, quiet gridlines, no
    chart-junk. Kept as a single function so every chart stays consistent."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def section(title: str, hint: str | None = None) -> None:
    """A consistent card section heading: bold title + one short optional hint."""
    st.markdown(f'<div class="twin-section-title">{title}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<div class="twin-hint">{hint}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar - the "current state of the business" every tab reads
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Commercial & Sales Twin")

st.sidebar.markdown('<div class="twin-sidebar-label">Business inputs</div>', unsafe_allow_html=True)
existing_active_base = st.sidebar.number_input("Existing active base", 0.0, 5000.0, 150.0, step=10.0)
marketing_budget = st.sidebar.number_input("Marketing budget ($/mo)", 0.0, 1_000_000.0, 50_000.0, step=5_000.0)
price = st.sidebar.number_input("Price ($/yr)", 100.0, 500_000.0, 12_000.0, step=500.0)
capacity = st.sidebar.number_input("Sales team capacity (QO/mo)", 1.0, 1000.0, 40.0, step=5.0)
horizon = st.sidebar.slider("Horizon (months)", 1, 24, 6)

st.sidebar.markdown('<div class="twin-sidebar-label">Growth loops</div>', unsafe_allow_html=True)
enable_backlog = st.sidebar.checkbox("Backlog (Loop B)", value=True)
enable_referral = st.sidebar.checkbox("Referral (Loop A)", value=True)
enable_loop_c = st.sidebar.checkbox("Revenue reinvestment (Loop C)", value=False)
reinvestment_rate = st.sidebar.slider("Reinvestment rate", 0.0, 1.0, 0.15, step=0.05) if enable_loop_c else 0.15

st.sidebar.markdown('<div class="twin-sidebar-label">Simulation mode</div>', unsafe_allow_html=True)
provider_mode = st.sidebar.radio("Randomness provider", ["Deterministic", "Monte Carlo"], label_visibility="collapsed")
seed = st.sidebar.number_input("Monte Carlo seed", 0, 10_000, 42) if provider_mode == "Monte Carlo" else None

with st.sidebar.expander("Market conditions"):
    demand_index = st.slider("Demand index", 0.7, 1.3, 1.0, step=0.05)
    seasonality_index = st.slider("Seasonality index", 0.7, 1.3, 1.0, step=0.05)
    competitive_pressure = st.slider("Competitive pressure", 0.7, 1.5, 1.0, step=0.05)
    market_growth_rate = st.slider("Market growth rate (monthly)", 0.0, 0.05, 0.0, step=0.005)

st.sidebar.markdown('<span class="twin-badge">Deterministic core · no AI in the loop</span>', unsafe_allow_html=True)

params = ParameterSet()
market = MarketConditions(
    demand_index=demand_index, seasonality_index=seasonality_index,
    competitive_pressure=competitive_pressure, market_growth_rate=market_growth_rate,
)


def make_provider():
    return MonteCarloProvider(seed) if provider_mode == "Monte Carlo" else DeterministicProvider()


def make_pricing(p: ParameterSet) -> TieredPricingStrategy:
    return TieredPricingStrategy(reference_price=price, elasticity_beta=p.price_elasticity_beta.value)


def make_marketing_controller():
    return RevenueReinvestmentMarketingController(reinvestment_rate=reinvestment_rate) if enable_loop_c else None


def run_trace() -> list:
    engine = DynamicSimulationEngine(
        params, make_pricing(params), make_provider(),
        marketing_controller=make_marketing_controller(),
        enable_backlog=enable_backlog, enable_referral=enable_referral, market=market,
    )
    start = initial_state(existing_active_base, marketing_budget, price, capacity)
    return engine.run(start, horizon=horizon)


st.markdown(
    '<div class="twin-header-row">'
    '<div><p class="twin-title">Commercial & Sales Digital Twin</p>'
    '<p class="twin-subtitle">A live simulation of your funnel, revenue, and renewals</p></div>'
    '<span class="twin-pill">Live</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.write("")

tab_overview, tab_components, tab_scenarios, tab_calibration, tab_validation = st.tabs(
    ["Overview", "Components", "Scenario Comparison", "Calibration", "Validation"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    history = run_trace()
    final = history[-1]
    m0 = history[0]

    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(f"ARR (month {final.month})", f"${metrics.arr(final):,.0f}", f"{metrics.arr(final) - metrics.arr(m0):,.0f} vs month 0")
        col2.metric("MRR", f"${metrics.mrr(final):,.0f}")
        col3.metric("Active customers", f"{final.active:,.1f}", f"{final.active - m0.active:+.1f}")
        col4.metric("Backlog", f"{final.backlog:,.1f}")
        churn_rate = metrics.churn_rate(final, history[-2]) if len(history) > 1 else 0.0
        col5.metric("Churn rate", f"{churn_rate * 100:.2f}%")
        if enable_loop_c:
            st.caption("Loop C is on - marketing budget is reinvested from revenue each month.")

    df = pd.DataFrame({
        "Month": [s.month for s in history],
        "Active": [s.active for s in history],
        "ARR": [metrics.arr(s) for s in history],
        "Backlog": [s.backlog for s in history],
        "Leads": [s.leads for s in history],
        "Qualified Opportunity": [s.qualified_opportunity for s in history],
    }).set_index("Month")

    st.write("")
    with st.container(border=True):
        left, mid, right = st.columns([1, 1, 1])
        with left:
            section("Active customers")
            fig = go.Figure(go.Scatter(
                x=df.index, y=df["Active"], mode="lines", line=dict(color=BLUE, width=2.5),
                fill="tozeroy", fillcolor="rgba(42, 120, 214, 0.08)", name="Active",
            ))
            st.plotly_chart(styled_fig(fig, height=260), use_container_width=True)
        with mid:
            section("ARR")
            fig = go.Figure(go.Scatter(
                x=df.index, y=df["ARR"], mode="lines", line=dict(color=ORANGE, width=2.5),
                fill="tozeroy", fillcolor="rgba(235, 104, 52, 0.08)", name="ARR",
            ))
            st.plotly_chart(styled_fig(fig, height=260), use_container_width=True)
        with right:
            section(f"Funnel - month {final.month}")
            stages = ["Leads", "Qualified Opp.", "Backlog", "Active"]
            counts = [final.leads, final.qualified_opportunity, final.backlog, final.active]
            fig = go.Figure(go.Bar(x=stages, y=counts, marker_color=FUNNEL_RAMP))
            st.plotly_chart(styled_fig(fig, height=260), use_container_width=True)

    with st.expander("Month-by-month data"):
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# Components - a behavior lab: every knob here feeds a live
# SyntheticDataGenerator run, and every chart shows the downstream effect
# on real generated records (revenue mix, churn, renewal outcomes).
# ---------------------------------------------------------------------------
with tab_components:
    st.markdown(
        '<div class="twin-hint">Adjust Market, Product, or Customer Segment inputs and see the effect '
        "flow through to revenue, churn, and renewals.</div>",
        unsafe_allow_html=True,
    )
    lab_seed = st.number_input("Seed (shared by every experiment below)", 0, 10_000, 42, key="lab_seed")
    st.write("")

    # -- Market: does changing conditions actually move Active/ARR? -----------
    with st.container(border=True):
        section("Market: current conditions vs. a neutral market")
        neutral_market = MarketConditions(demand_index=1.0, seasonality_index=1.0, competitive_pressure=1.0, market_growth_rate=0.0)
        current_engine = DynamicSimulationEngine(
            ParameterSet(), make_pricing(ParameterSet()), DeterministicProvider(),
            enable_backlog=enable_backlog, enable_referral=enable_referral, market=market,
        )
        neutral_engine = DynamicSimulationEngine(
            ParameterSet(), make_pricing(ParameterSet()), DeterministicProvider(),
            enable_backlog=enable_backlog, enable_referral=enable_referral, market=neutral_market,
        )
        current_hist = current_engine.run(initial_state(existing_active_base, marketing_budget, price, capacity), horizon=horizon)
        neutral_hist = neutral_engine.run(initial_state(existing_active_base, marketing_budget, price, capacity), horizon=horizon)
        market_delta = current_hist[-1].active - neutral_hist[-1].active
        market_delta_pct = (market_delta / neutral_hist[-1].active * 100) if neutral_hist[-1].active else 0.0

        mkt_col, chart_col = st.columns([1, 2])
        with mkt_col:
            st.metric(f"Active({horizon}) with current Market", f"{current_hist[-1].active:,.1f}",
                       f"{market_delta:+.1f} ({market_delta_pct:+.1f}%) vs neutral")
            st.caption(f"demand {demand_index:.2f} · seasonality {seasonality_index:.2f} · "
                       f"competition {competitive_pressure:.2f} · growth {market_growth_rate:.1%}/mo")
        with chart_col:
            months = [s.month for s in current_hist]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=[s.active for s in current_hist], mode="lines",
                                      line=dict(color=BLUE, width=2.5), name="Current Market"))
            fig.add_trace(go.Scatter(x=months, y=[s.active for s in neutral_hist], mode="lines",
                                      line=dict(color=MUTED, width=2, dash="dot"), name="Neutral Market"))
            st.plotly_chart(styled_fig(fig, height=220), use_container_width=True)

    st.write("")

    # -- Product: does re-pricing a SKU shift revenue mix? ---------------------
    with st.container(border=True):
        section("Product: re-price a SKU, watch revenue mix shift")
        default_products = {p.sku: p for p in default_catalog().products}
        sku_choice = st.selectbox("SKU to adjust", list(default_products.keys()), key="sku_choice")
        base_product = default_products[sku_choice]
        new_multiplier = st.slider(
            f"{base_product.name} price multiplier", 0.3, 3.0, base_product.price_multiplier, step=0.05, key="sku_multiplier",
        )
        modified_products = [
            dataclasses.replace(p, price_multiplier=new_multiplier) if p.sku == sku_choice else p
            for p in default_catalog().products
        ]
        modified_catalog = ProductCatalog(products=modified_products)

        product_generator = SyntheticDataGenerator(
            ParameterSet(), make_pricing, seed=int(lab_seed), catalog=modified_catalog,
        )
        product_dataset, _ = product_generator.generate(existing_active_base, marketing_budget, price, capacity, horizon=horizon)

        pcol1, pcol2 = st.columns([1, 2])
        with pcol1:
            st.metric("Catalog weighted-avg multiplier", f"{modified_catalog.weighted_average_multiplier():.3f}",
                       f"{modified_catalog.weighted_average_multiplier() - default_catalog().weighted_average_multiplier():+.3f} vs default")
            total_revenue = product_dataset.customers["deal_size"].sum()
            sku_share = product_dataset.customers.groupby("sku")["deal_size"].sum() / total_revenue * 100
            st.caption(f"{sku_choice} carries {sku_share.get(sku_choice, 0):.1f}% of revenue "
                       f"(catalog weight alone predicts {base_product.catalog_weight * 100:.0f}%).")
        with pcol2:
            st.markdown("**Revenue share by SKU**")
            rev_by_sku = product_dataset.customers.groupby("sku")["deal_size"].sum().sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=rev_by_sku.index, y=rev_by_sku.values, marker_color=CATEGORICAL[:len(rev_by_sku)]))
            st.plotly_chart(styled_fig(fig, height=220), use_container_width=True)

    st.write("")

    # -- Segment: turn a behavioral dial, watch the downstream population -----
    with st.container(border=True):
        section("Customer Segment: change one segment's behavior")
        default_seg_map = {s.name: s for s in default_segments().segments}
        seg_choice = st.selectbox("Segment to adjust", list(default_seg_map.keys()), key="seg_choice")
        base_seg = default_seg_map[seg_choice]

        seg1, seg2, seg3, seg4 = st.columns(4)
        conv_x = seg1.slider("Conversion x", 0.3, 2.0, base_seg.conversion_multiplier, step=0.05, key="conv_x")
        churn_x = seg2.slider("Churn x", 0.1, 3.0, base_seg.churn_multiplier, step=0.05, key="churn_x")
        renew_x = seg3.slider("Renewal x", 0.3, 1.5, base_seg.renewal_multiplier, step=0.05, key="renew_x")
        deal_x = seg4.slider("Deal size x", 0.3, 3.0, base_seg.deal_size_multiplier, step=0.05, key="deal_x")

        modified_segments = [
            dataclasses.replace(s, conversion_multiplier=conv_x, churn_multiplier=churn_x,
                                 renewal_multiplier=renew_x, deal_size_multiplier=deal_x)
            if s.name == seg_choice else s
            for s in default_segments().segments
        ]
        modified_mix = SegmentMix(segments=modified_segments)

        baseline_generator = SyntheticDataGenerator(ParameterSet(), make_pricing, seed=int(lab_seed), segments=default_segments())
        baseline_dataset, _ = baseline_generator.generate(existing_active_base, marketing_budget, price, capacity, horizon=horizon)
        modified_generator = SyntheticDataGenerator(ParameterSet(), make_pricing, seed=int(lab_seed), segments=modified_mix)
        modified_dataset, _ = modified_generator.generate(existing_active_base, marketing_budget, price, capacity, horizon=horizon)

        def segment_scoreboard(dataset) -> pd.DataFrame:
            customers = dataset.customers
            total_revenue = customers["deal_size"].sum()
            rows = []
            for name in default_seg_map:
                seg_rows = customers[customers["segment"] == name]
                if len(seg_rows) == 0:
                    rows.append({"Segment": name, "Customers": 0, "Revenue share %": 0.0, "Churn rate %": 0.0})
                    continue
                churn_rate_pct = seg_rows["churn_month"].notna().mean() * 100
                revenue_share = seg_rows["deal_size"].sum() / total_revenue * 100 if total_revenue else 0.0
                rows.append({
                    "Segment": name, "Customers": len(seg_rows),
                    "Revenue share %": round(revenue_share, 1), "Churn rate %": round(churn_rate_pct, 1),
                })
            return pd.DataFrame(rows).set_index("Segment")

        baseline_score = segment_scoreboard(baseline_dataset)
        modified_score = segment_scoreboard(modified_dataset)

        st.markdown(f"**Before vs. after adjusting `{seg_choice}`**")
        bcol, acol = st.columns(2)
        with bcol:
            st.caption("Baseline")
            st.dataframe(baseline_score, use_container_width=True)
        with acol:
            st.caption("After your adjustment")
            st.dataframe(modified_score, use_container_width=True)

        delta_customers = modified_score.loc[seg_choice, "Customers"] - baseline_score.loc[seg_choice, "Customers"]
        delta_churn = modified_score.loc[seg_choice, "Churn rate %"] - baseline_score.loc[seg_choice, "Churn rate %"]
        delta_revenue = modified_score.loc[seg_choice, "Revenue share %"] - baseline_score.loc[seg_choice, "Revenue share %"]
        d1, d2, d3 = st.columns(3)
        d1.metric(f"{seg_choice} customers", int(modified_score.loc[seg_choice, "Customers"]), f"{delta_customers:+.0f} vs baseline")
        d2.metric(f"{seg_choice} churn rate", f"{modified_score.loc[seg_choice, 'Churn rate %']:.1f}%", f"{delta_churn:+.1f} pts")
        d3.metric(f"{seg_choice} revenue share", f"{modified_score.loc[seg_choice, 'Revenue share %']:.1f}%", f"{delta_revenue:+.1f} pts")

    st.write("")

    # -- Contract: does the Segment dial actually change renewal outcomes? ----
    with st.container(border=True):
        section("Contract: renewal outcomes by segment", "Annual contracts only")

        joined = modified_dataset.contracts.merge(
            modified_dataset.customers[["customer_id", "segment", "churn_month"]], on="customer_id", how="left",
        )

        # A renewal-boundary outcome is either a successful RENEWED, or a
        # CHURNED whose churn_month lands exactly on a renewal boundary
        # (elapsed % term_length_months == 0), matching
        # Contract.is_up_for_renewal(). An ordinary CHURNED elsewhere in the
        # term is mid-term attrition (monthly_churn_hazard), not a failed
        # renewal draw, and must not be counted here.
        def _at_renewal_boundary(row) -> bool:
            if row["term_length_months"] <= 1:
                return False
            if row["status"] == "renewed":
                return True
            if row["status"] == "churned" and pd.notna(row["churn_month"]):
                elapsed = row["churn_month"] - row["start_month"]
                return elapsed > 0 and elapsed % row["term_length_months"] == 0
            return False

        joined["at_boundary"] = joined.apply(_at_renewal_boundary, axis=1)
        annual_boundary = joined[joined["at_boundary"]]
        if len(annual_boundary) > 0:
            renewal_by_segment = annual_boundary.groupby("segment")["status"].apply(
                lambda s: (s == "renewed").mean() * 100
            ).reindex(default_seg_map.keys())
            rcol1, rcol2 = st.columns([1, 2])
            with rcol1:
                st.dataframe(renewal_by_segment.round(1).rename("Renewal rate %").to_frame(), use_container_width=True)
            with rcol2:
                fig = go.Figure(go.Bar(
                    x=renewal_by_segment.index, y=renewal_by_segment.values,
                    marker_color=CATEGORICAL[:len(renewal_by_segment)],
                ))
                st.plotly_chart(styled_fig(fig, height=200), use_container_width=True)
        else:
            st.info(f"No annual contracts have reached a renewal boundary within {horizon} months yet - "
                    f"increase the horizon (sidebar) to see renewal outcomes.")

        status_by_segment = modified_dataset.customers[["customer_id", "segment"]].merge(
            modified_dataset.contracts[["customer_id", "status"]], on="customer_id",
        )
        pivot = status_by_segment.groupby(["segment", "status"]).size().unstack(fill_value=0)
        st.markdown("**Contract status by segment (all contracts)**")
        fig = go.Figure()
        for i, status in enumerate(pivot.columns):
            fig.add_trace(go.Bar(x=pivot.index, y=pivot[status], name=status, marker_color=CATEGORICAL[i % len(CATEGORICAL)]))
        fig.update_layout(barmode="stack")
        st.plotly_chart(styled_fig(fig, height=240), use_container_width=True)

# ---------------------------------------------------------------------------
# Scenario Comparison
# ---------------------------------------------------------------------------
with tab_scenarios:
    with st.container(border=True):
        section("Compare levers against a shared month-0 baseline")

        sc1, sc2, sc3 = st.columns(3)
        price_b = sc1.number_input("B - Price increase ($/yr)", 100.0, 500_000.0, price * 1.2, step=500.0)
        budget_c = sc2.number_input("C - Marketing expansion ($/mo)", 0.0, 1_000_000.0, marketing_budget * 1.5, step=5_000.0)
        capacity_d = sc3.number_input("D - Sales team growth (QO/mo)", 1.0, 1000.0, capacity * 1.5, step=5.0)

        runner = ScenarioRunner(
            params, make_pricing, make_provider(),
            enable_backlog=enable_backlog, enable_referral=enable_referral,
        )
        shared_controller = make_marketing_controller()
        specs = [
            ScenarioSpec(name="A - Baseline", overrides={}, marketing_controller=shared_controller),
            ScenarioSpec(name="B - Price Increase", overrides={"price": price_b}, marketing_controller=shared_controller),
            ScenarioSpec(name="C - Marketing Expansion", overrides={"marketing_budget": budget_c}, marketing_controller=shared_controller),
            ScenarioSpec(name="D - Sales Team Growth", overrides={"capacity": capacity_d}, marketing_controller=shared_controller),
        ]
        results = runner.compare(
            specs, existing_active_base=existing_active_base,
            base_marketing_budget=marketing_budget, base_price=price,
            base_capacity=capacity, horizon=horizon,
        )
        if enable_loop_c:
            st.caption("Loop C is on - all four scenarios reinvest revenue into marketing.")

        rows = []
        for name, result in results.items():
            f = result.history[-1]
            rows.append({"Scenario": name, "Active": round(f.active, 1), "ARR": round(metrics.arr(f), 0)})
        scenario_df = pd.DataFrame(rows).set_index("Scenario")
        st.dataframe(scenario_df, use_container_width=True)

    st.write("")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            section(f"Active customers - month {horizon}")
            fig = go.Figure(go.Bar(x=scenario_df.index, y=scenario_df["Active"], marker_color=CATEGORICAL))
            st.plotly_chart(styled_fig(fig, height=280), use_container_width=True)
        with right:
            section(f"ARR - month {horizon}")
            fig = go.Figure(go.Bar(x=scenario_df.index, y=scenario_df["ARR"], marker_color=CATEGORICAL))
            st.plotly_chart(styled_fig(fig, height=280), use_container_width=True)

# ---------------------------------------------------------------------------
# Calibration - both worked examples, the same Bayesian mechanism on two
# parameters (lead conversion AND Contract renewal).
# ---------------------------------------------------------------------------
with tab_calibration:
    with st.container(border=True):
        section("Bayesian credibility-weighted calibration")

        cal1, cal2 = st.columns(2)

        with cal1:
            st.markdown("**Lead conversion**")
            c1, c2 = st.columns(2)
            prior = c1.number_input("Prior (%)", 0.0, 100.0, 15.0, key="lead_prior") / 100.0
            observed = c2.number_input("Observed (%)", 0.0, 100.0, 18.0, key="lead_observed") / 100.0
            c3, c4 = st.columns(2)
            n_obs = c3.number_input("n observations", 0, 100_000, 200, key="lead_n")
            k_cred = c4.number_input("k (prior's sample size)", 0, 100_000, 75, key="lead_k")

            calib_params = ParameterSet()
            calib_params.r1_lead_to_qo.value = prior
            engine = CalibrationEngine(calib_params)
            engine.calibrate(
                "r1_lead_to_qo", observed_value=observed, n_observations=n_obs,
                k_credibility=k_cred, data_source="dashboard live input",
            )
            entry = engine.log[-1]
            z = engine.credibility_weight(n_obs, k_cred)
            st.latex(r"Z = \frac{n}{n+k} = %.3f" % z)
            st.metric("Calibrated value", f"{entry.new_value * 100:.2f}%", f"{(entry.new_value - prior) * 100:+.2f} pts vs prior")

        with cal2:
            st.markdown("**Contract renewal**")
            r1, r2 = st.columns(2)
            renewal_prior = r1.number_input("Prior (%)", 0.0, 100.0, 90.0, key="renewal_prior") / 100.0
            renewal_observed = r2.number_input("Observed (%)", 0.0, 100.0, 82.0, key="renewal_observed") / 100.0
            r3, r4 = st.columns(2)
            renewal_n = r3.number_input("n observations", 0, 100_000, 150, key="renewal_n")
            renewal_k = r4.number_input("k (prior's sample size)", 0, 100_000, 50, key="renewal_k")

            renewal_params = ParameterSet()
            renewal_params.renewal_probability.value = renewal_prior
            renewal_engine = CalibrationEngine(renewal_params)
            renewal_engine.calibrate(
                "renewal_probability", observed_value=renewal_observed, n_observations=renewal_n,
                k_credibility=renewal_k, data_source="dashboard live input",
            )
            renewal_entry = renewal_engine.log[-1]
            renewal_z = renewal_engine.credibility_weight(renewal_n, renewal_k)
            st.latex(r"Z = \frac{n}{n+k} = %.3f" % renewal_z)
            st.metric("Calibrated value", f"{renewal_entry.new_value * 100:.2f}%",
                       f"{(renewal_entry.new_value - renewal_prior) * 100:+.2f} pts vs prior")
            st.caption("Only annual contracts reach a renewal boundary - monthly attrition is captured separately.")

    st.write("")
    with st.container(border=True):
        section("Current parameter set")
        param_rows = []
        for name, p in params.as_dict().items():
            param_rows.append({
                "Parameter": name, "Value": p.value, "Range": f"{p.range[0]}-{p.range[1]}",
                "Confidence": p.confidence.name, "Source": p.source,
            })
        st.dataframe(pd.DataFrame(param_rows).set_index("Parameter"), use_container_width=True)

# ---------------------------------------------------------------------------
# Validation - Sensitivity Analysis (OAT sweeps) and Monte Carlo convergence,
# run live rather than only in run_verification.py.
# ---------------------------------------------------------------------------
with tab_validation:
    with st.container(border=True):
        section("Sensitivity analysis", "One-at-a-time sweep, growth loops off to isolate the swept parameter")

        sweep_choice = st.selectbox(
            "Parameter to sweep",
            ["monthly_churn_hazard (expect Active(T) to fall)", "r2_qo_to_active_base (expect Active(T) to rise)"],
        )
        sweep_param = "monthly_churn_hazard" if sweep_choice.startswith("monthly_churn_hazard") else "r2_qo_to_active_base"
        default_range = (0.005, 0.05) if sweep_param == "monthly_churn_hazard" else (0.10, 0.35)

        sw1, sw2, sw3 = st.columns(3)
        low = sw1.number_input("Low", value=default_range[0], format="%.3f")
        high = sw2.number_input("High", value=default_range[1], format="%.3f")
        steps = sw3.slider("Steps", 3, 10, 5)

        if st.button("Run sensitivity sweep"):
            sweep_params = ParameterSet()
            sweep_runner = ScenarioRunner(
                sweep_params, make_pricing, DeterministicProvider(),
                enable_backlog=False, enable_referral=False,
            )
            analyzer = SensitivityAnalyzer(sweep_runner)
            sweep_results = analyzer.sweep(
                sweep_param, (low, high, steps),
                existing_active_base, marketing_budget, price, capacity, horizon=horizon,
                output_fn=lambda hist: hist[-1].active,
            )
            sweep_df = pd.DataFrame(sweep_results, columns=[sweep_param, f"Active({horizon})"])
            fig = go.Figure(go.Scatter(
                x=sweep_df[sweep_param], y=sweep_df[f"Active({horizon})"],
                mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=8),
            ))
            st.plotly_chart(styled_fig(fig, height=280), use_container_width=True)
            st.dataframe(sweep_df.set_index(sweep_param), use_container_width=True)
            values = sweep_df[f"Active({horizon})"].tolist()
            monotonic_down = all(a >= b for a, b in zip(values, values[1:]))
            monotonic_up = all(a <= b for a, b in zip(values, values[1:]))
            if monotonic_down or monotonic_up:
                st.success(f"Monotonic {'decrease' if monotonic_down else 'increase'} confirmed across the sweep.")
            else:
                st.warning("Not monotonic across this range - try a narrower range or check for capacity capping.")

    st.write("")
    with st.container(border=True):
        section("Monte Carlo convergence check", "Averages seeded runs and compares the mean to the deterministic trace")
        n_runs = st.slider("Number of Monte Carlo runs", 20, 500, 200, step=20)

        if st.button("Run Monte Carlo convergence check"):
            det_params = ParameterSet()
            det_engine = DynamicSimulationEngine(
                det_params, make_pricing(det_params), DeterministicProvider(),
                enable_backlog=False, enable_referral=False,
            )
            det_start = initial_state(existing_active_base, marketing_budget, price, capacity)
            deterministic_active = det_engine.run(det_start, horizon=horizon)[-1].active

            mc_finals = []
            progress = st.progress(0)
            for i, s in enumerate(range(n_runs)):
                mc_params = ParameterSet()
                mc_engine = DynamicSimulationEngine(
                    mc_params, make_pricing(mc_params), MonteCarloProvider(s),
                    enable_backlog=False, enable_referral=False,
                )
                mc_history = mc_engine.run(
                    initial_state(existing_active_base, marketing_budget, price, capacity), horizon=horizon,
                )
                mc_finals.append(mc_history[-1].active)
                if i % max(1, n_runs // 20) == 0:
                    progress.progress(min(1.0, (i + 1) / n_runs))
            progress.progress(1.0)

            mc_mean = sum(mc_finals) / len(mc_finals)
            rel_error = abs(mc_mean - deterministic_active) / deterministic_active if deterministic_active else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric(f"Deterministic Active({horizon})", f"{deterministic_active:,.2f}")
            m2.metric(f"Mean of {n_runs} Monte Carlo runs", f"{mc_mean:,.2f}")
            m3.metric("Relative error", f"{rel_error * 100:.2f}%")

            fig = go.Figure(go.Histogram(x=mc_finals, marker_color=AQUA, nbinsx=30))
            fig.add_vline(x=deterministic_active, line_color=ORANGE, line_width=2.5)
            st.plotly_chart(styled_fig(fig, height=260), use_container_width=True)

            if rel_error < 0.03:
                st.success(f"Converged within 3% (relative error {rel_error:.2%}).")
            else:
                st.warning(f"Relative error {rel_error:.2%} exceeds 3% - try more runs.")