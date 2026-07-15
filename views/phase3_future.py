"""Phase 3 — bounded cost, yield and financial scenario analysis."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

from core.phase3_model import (
    MODEL_KEY_TO_SPEC,
    available_accounts,
    calculate_phase3_scenario,
    clean_relationship_data,
    normalise_cultivar,
    normalise_text,
    prepare_farmer_costs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PALETTE = {
    "teal": "#2A7F87",
    "teal_light": "#E4F1F2",
    "green": "#1CBA59",
    "red": "#D64545",
    "blue": "#5B7DB1",
    "grey": "#8A93A6",
    "light_grey": "#EEF1F5",
    "ink": "#253247",
}


def resolve_data_file(filename: str, required: bool = True) -> Path | None:
    candidates = [
        DATA_DIR / filename,
        PROJECT_ROOT / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find {filename}. Checked: {searched}")
    return None


@st.cache_data
def load_phase3_relationships() -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(resolve_data_file("phase3_significant_inputs.csv"), encoding="utf-8-sig")
    yields = pd.read_csv(resolve_data_file("phase3_significant_yield.csv"), encoding="utf-8-sig")
    return clean_relationship_data(inputs), clean_relationship_data(yields)


@st.cache_data
def load_yield_benchmarks() -> pd.DataFrame:
    path = resolve_data_file("yield.csv", required=False)
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    yield_columns = [column for column in frame.columns if "yield" in column.casefold()]
    if not yield_columns:
        return pd.DataFrame()
    value_column = yield_columns[0]
    frame["__yield"] = pd.to_numeric(frame[value_column], errors="coerce")
    if frame["__yield"].median(skipna=True) > 1000:
        frame["__yield"] = frame["__yield"] / 1000.0
    return frame


@st.cache_data
def load_cost_change_benchmarks() -> pd.DataFrame:
    path = resolve_data_file("costs_2024_2025_with_blended_growth_percent.csv", required=False)
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def lookup_industry_yield(base: dict[str, Any]) -> tuple[float | None, float | None]:
    low = base.get("industry_yield_low")
    high = base.get("industry_yield_high")
    if low is not None or high is not None:
        return (
            float(low) if low is not None else None,
            float(high) if high is not None else None,
        )

    frame = load_yield_benchmarks()
    if frame.empty:
        return None, None

    sub = frame.copy()
    if "Wine Class" in sub.columns:
        sub = sub[sub["Wine Class"].map(normalise_text) == normalise_text(base.get("wine_class"))]
    if "Region" in sub.columns:
        sub = sub[sub["Region"].map(normalise_text) == normalise_text(base.get("region"))]
    if "Grape Variety" in sub.columns:
        exact = sub[sub["Grape Variety"].map(normalise_cultivar) == normalise_cultivar(base.get("grape_variety"))]
        if not exact.empty:
            sub = exact
        else:
            broader = "Other White" if normalise_text(base.get("wine_class")) == "white" else "Other Red"
            other = sub[sub["Grape Variety"].map(normalise_text) == normalise_text(broader)]
            if not other.empty:
                sub = other

    if "Band" not in sub.columns:
        return None, None
    values = {
        normalise_text(row["Band"]): float(row["__yield"])
        for _, row in sub.dropna(subset=["__yield"]).iterrows()
    }
    return values.get("low"), values.get("high")


def lookup_cost_change_benchmark(region: str, item: str) -> float | None:
    frame = load_cost_change_benchmarks()
    required = {"Region", "Item", "model_ready_blended_growth_percent"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    sub = frame[
        (frame["Region"].map(normalise_text) == normalise_text(region))
        & (frame["Item"].map(normalise_text) == normalise_text(item))
    ].copy()
    if "Year" in sub.columns:
        years = pd.to_numeric(sub["Year"], errors="coerce")
        if years.notna().any():
            sub = sub[years == years.max()]
    values = pd.to_numeric(sub["model_ready_blended_growth_percent"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def money(value: Any) -> str:
    try:
        return f"R {float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def plot_financial_outcomes(summary: pd.DataFrame):
    labels = [
        label.replace("Phase 1 baseline", "Baseline").replace(" response", "")
        for label in summary["Case"]
    ]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.1), dpi=145)
    ax.bar(x - width, summary["Revenue (R/ha)"], width, label="Revenue", color=PALETTE["blue"])
    ax.bar(x, summary["Total cost (R/ha)"], width, label="Total cost", color=PALETTE["grey"])
    nfi_colors = [PALETTE["green"] if value >= 0 else PALETTE["red"] for value in summary["NFI (R/ha)"]]
    ax.bar(x + width, summary["NFI (R/ha)"], width, label="NFI", color=nfi_colors)
    ax.axhline(0, color="#C7CBD1", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R/ha")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"R {value:,.0f}"))
    ax.grid(axis="y", alpha=0.16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    return fig


def plot_cost_changes(cost_details: pd.DataFrame):
    central = cost_details[cost_details["Case"] == "Central response"].copy()
    central = central[central["Change (R/ha)"].abs() > 0.01]
    central = central.reindex(central["Change (R/ha)"].abs().sort_values(ascending=False).index).head(10)
    if central.empty:
        return None

    central = central.sort_values("Scenario cost (R/ha)")
    labels = central["Item"].str.replace(" en ", " & ", regex=False)
    y = np.arange(len(central))
    fig, ax = plt.subplots(figsize=(8.4, max(3.4, len(central) * 0.42)), dpi=145)
    ax.barh(y - 0.17, central["Baseline cost (R/ha)"], height=0.32, color=PALETTE["light_grey"], label="Phase 1")
    ax.barh(y + 0.17, central["Scenario cost (R/ha)"], height=0.32, color=PALETTE["teal"], label="Central response")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("R/ha")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"R {value:,.0f}"))
    ax.grid(axis="x", alpha=0.14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=8.5)
    fig.tight_layout()
    return fig


st.markdown(
    """
    <style>
      .main .block-container { padding-top: 0.7rem; }
      .phase3-note {
        border-left: 4px solid #2A7F87; background: #F2F8F8;
        padding: 0.75rem 0.95rem; border-radius: 0.3rem; margin: 0.25rem 0 1rem 0;
      }
      .phase3-note strong { color: #253247; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Phase 3 — Cost & Yield Scenarios")
st.markdown(
    "<p style='color:#6b7280; margin-top:-10px;'><em>"
    "Bounded scenario analysis using the farmer’s current 2025 Phase 1 position."
    "</em></p>",
    unsafe_allow_html=True,
)

base = st.session_state.get("phase1_current_base")
if not base:
    st.warning(
        "Phase 3 needs the current farmer baseline. Open Phase 1, confirm the farmer inputs, "
        "and then return to this page. No Phase 2 forecast values are used here."
    )
    st.stop()

try:
    input_relationships, yield_relationships = load_phase3_relationships()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Copy `phase3_significant_inputs.csv` and `phase3_significant_yield.csv` into the project data folder."
    )
    st.stop()

farmer_costs = prepare_farmer_costs(base.get("farmer_costs", []))
accounts = available_accounts(farmer_costs)
if not accounts:
    st.error("No Phase 1 cost lines could be mapped to the Phase 3 coefficient accounts.")
    st.stop()
selectable_accounts = [account for account in accounts if account["baseline_cost"] > 0]
if not selectable_accounts:
    st.error("The mapped Phase 1 cost lines are all zero, so a percentage cost adjustment cannot be calculated.")
    st.stop()

yield_low, yield_high = lookup_industry_yield(base)
if yield_low is not None and yield_high is not None:
    industry_yield_midpoint = (yield_low + yield_high) / 2.0
else:
    industry_yield_midpoint = yield_high if yield_high is not None else yield_low
effective_yield_cap = (
    max(float(base.get("farmer_yield", 0.0)), float(industry_yield_midpoint))
    if industry_yield_midpoint is not None
    else None
)

account_keys = [account["key"] for account in selectable_accounts]
default_key = "fertilizer" if "fertilizer" in account_keys else account_keys[0]

with st.sidebar:
    st.header("Phase 3 scenario")
    st.caption("The farmer changes one current Phase 1 cost. Related effects then run for a fixed number of rounds and stop.")
    selected_key = st.selectbox(
        "Cost item to adjust",
        options=account_keys,
        index=account_keys.index(default_key),
        format_func=lambda key: MODEL_KEY_TO_SPEC[key].label,
        key="phase3_selected_key",
    )
    selected_account = next(account for account in selectable_accounts if account["key"] == selected_key)
    st.caption(f"Phase 1 value: {money(selected_account['baseline_cost'])}/ha")
    adjustment_percent = st.number_input(
        "Percentage adjustment",
        min_value=-90.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        format="%.1f",
        help="A positive value increases the selected cost; a negative value reduces it.",
        key="phase3_adjustment_percent",
    )
    rounds = st.radio(
        "Relationship rounds",
        options=[1, 2],
        index=0,
        horizontal=True,
        format_func=lambda value: f"{value} round" if value == 1 else f"{value} rounds",
        help="The calculation stops after the selected round. It never runs an unrestricted loop.",
        key="phase3_rounds",
    )
    st.markdown("---")
    st.caption("Coefficient cases: C2.5, posterior mean and C97.5. Grape price remains constant.")

regional_change = lookup_cost_change_benchmark(base.get("region", ""), selected_account["item"])
if regional_change is not None:
    if adjustment_percent > regional_change and adjustment_percent > 0:
        st.warning(
            f"The selected increase ({adjustment_percent:.1f}%) is above the 2025 area benchmark change "
            f"for this cost item ({regional_change:.1f}%). The scenario will still calculate."
        )
    else:
        st.caption(
            f"2025 area benchmark change for the selected cost item: {regional_change:.1f}%."
        )

result = calculate_phase3_scenario(
    farmer_costs=farmer_costs,
    provision_for_renewal=float(base.get("provision_for_renewal", 0.0)),
    grape_price_per_tonne=float(base.get("farmer_income_rt", 0.0)),
    baseline_yield=float(base.get("farmer_yield", 0.0)),
    industry_yield_cap=industry_yield_midpoint,
    area=str(base.get("region", "")),
    cultivar=str(base.get("grape_variety", "")),
    selected_key=selected_key,
    adjustment_percent=float(adjustment_percent),
    rounds=int(rounds),
    input_relationships=input_relationships,
    yield_relationships=yield_relationships,
)

summary = result["summary"]
baseline_row = summary.iloc[0]
central_row = summary.loc[summary["Case"] == "Central response"].iloc[0]

st.markdown(
    f"""
    <div class="phase3-note">
      <strong>Current scenario:</strong> {escape(str(base.get('region', '')))} ·
      {escape(str(base.get('grape_variety', '')))} ·
      {escape(MODEL_KEY_TO_SPEC[selected_key].label)} {adjustment_percent:+.1f}% ·
      {rounds} bounded relationship round{'s' if rounds == 2 else ''}.<br>
      <span style="color:#5f6b7a;">This page uses the Phase 1 current-year baseline; it does not use the 2030 forecast.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
metric_columns[0].metric("Baseline NFI", money(baseline_row["NFI (R/ha)"]))
metric_columns[1].metric(
    "Central NFI",
    money(central_row["NFI (R/ha)"]),
    delta=money(central_row["Change in NFI (R/ha)"]),
)
metric_columns[2].metric(
    "Central yield",
    f"{central_row['Yield (t/ha)']:.2f} t/ha",
    delta=f"{central_row['Yield (t/ha)'] - baseline_row['Yield (t/ha)']:+.2f} t/ha",
)
metric_columns[3].metric("Central total cost", money(central_row["Total cost (R/ha)"]))
metric_columns[4].metric(
    "Yield increase cap",
    f"{effective_yield_cap:.2f} t/ha" if effective_yield_cap is not None else "Not available",
)

if result["yield_details"].empty:
    st.info(
        "No applicable yield coefficient was found for the selected and linked cost changes for this area–cultivar combination. "
        "Costs and NFI are still recalculated, while yield remains at the Phase 1 level."
    )
elif bool(summary["Yield cap applied"].fillna(False).any()):
    st.info(
        "At least one calculated yield response reached the agreed cap. Further cost increases can still affect costs, "
        "but they do not increase yield above the 2025 industry midpoint. If the current farmer yield is already higher, "
        "the current baseline is preserved and becomes the cap."
    )

st.subheader("Scenario outcomes")
st.caption(
    "Lower, central and upper are coefficient-response cases. They are not labelled best or worst because that economic ranking remains open for discussion."
)

display_summary = summary[
    [
        "Case",
        "Selected input cost (R/ha)",
        "Associated cost change (R/ha)",
        "Yield (t/ha)",
        "Revenue (R/ha)",
        "Total cost (R/ha)",
        "NFI (R/ha)",
        "Change in NFI (R/ha)",
    ]
].copy()
st.dataframe(
    display_summary,
    hide_index=True,
    width="stretch",
    column_config={
        "Selected input cost (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
        "Associated cost change (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
        "Yield (t/ha)": st.column_config.NumberColumn(format="%.2f"),
        "Revenue (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
        "Total cost (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
        "NFI (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
        "Change in NFI (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
    },
)

chart_columns = st.columns(2)
with chart_columns[0]:
    st.markdown("#### Financial outcomes")
    financial_figure = plot_financial_outcomes(summary)
    st.pyplot(financial_figure, width="stretch")
    plt.close(financial_figure)

with chart_columns[1]:
    st.markdown("#### Cost composition change")
    cost_figure = plot_cost_changes(result["cost_details"])
    if cost_figure is None:
        st.info("No material cost changes to chart.")
    else:
        st.pyplot(cost_figure, width="stretch")
        plt.close(cost_figure)

central_change = float(central_row["Change in NFI (R/ha)"])
direction = "improves" if central_change >= 0 else "reduces"
st.markdown("#### Automatic interpretation")
st.write(
    f"For the central coefficient response, the {adjustment_percent:+.1f}% change in "
    f"{MODEL_KEY_TO_SPEC[selected_key].label.lower()} {direction} NFI by {money(abs(central_change))}/ha "
    f"relative to the Phase 1 baseline. The adjusted yield is {number(central_row['Yield (t/ha)'])} t/ha, "
    f"with grape price held constant at {money(base.get('farmer_income_rt', 0.0))}/t."
)

with st.expander("Detailed cost calculations", expanded=False):
    cost_details = result["cost_details"].copy()
    st.dataframe(
        cost_details,
        hide_index=True,
        width="stretch",
        column_config={
            "Baseline cost (R/ha)": st.column_config.NumberColumn(format="R %.2f"),
            "Scenario cost (R/ha)": st.column_config.NumberColumn(format="R %.2f"),
            "Change (R/ha)": st.column_config.NumberColumn(format="R %.2f"),
            "Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

with st.expander("Relationship and coefficient audit trail", expanded=False):
    if result["propagation_details"].empty:
        st.info("No mapped input-to-input relationship was activated by this selection.")
    else:
        st.markdown("**Associated-cost pathways**")
        st.dataframe(result["propagation_details"], hide_index=True, width="stretch")
    if result["yield_details"].empty:
        st.info("No mapped yield relationship was activated by this selection.")
    else:
        st.markdown("**Yield-response pathways**")
        st.dataframe(result["yield_details"], hide_index=True, width="stretch")

with st.expander("Matrix suitability and safeguards", expanded=False):
    diagnostic = result["matrix_diagnostic"]
    if diagnostic["spectral_radius"] is None:
        st.warning("No input-cost matrix is available for the selected area.")
    else:
        dcols = st.columns(4)
        dcols[0].metric("Cost accounts", diagnostic["nodes"])
        dcols[1].metric("Directional links", diagnostic["edges"])
        dcols[2].metric("Spectral radius", f"{diagnostic['spectral_radius']:.3f}")
        dcols[3].metric("Condition number", f"{diagnostic['condition_number']:.1f}")
        if diagnostic["invertible"] and diagnostic["iterative_stable"]:
            st.success(
                "The posterior-mean area matrix is algebraically invertible and its unrestricted iteration test is below 1. "
                "The dashboard nevertheless remains bounded because the equations were estimated pairwise rather than confirmed as one joint system."
            )
        else:
            st.warning(
                "The full area network is not suitable for an unrestricted feedback loop. The dashboard therefore applies the selected one or two rounds and stops."
            )
    coverage = result["coverage"].get("Central response", {})
    st.caption(
        f"Mapped directional relationships used for this area: {coverage.get('mapped_relationships', 0)} of "
        f"{coverage.get('area_relationships', 0)}. Aggregate coefficient concepts without an unambiguous Phase 1 line are not forced into the calculation."
    )
    st.markdown(
        """
        - Rows previously marked as logically invalid are absent from the runtime coefficient files.
        - Gross income, cross margin and NFI are not used to predict yield.
        - Multiple effects on the same target are added in log-change space.
        - Grape price is held constant.
        - Yield increases stop at the higher of the current farmer yield and the 2025 industry midpoint; negative yield responses are not hidden.
        - The original Phase 1 record is never overwritten.
        """
    )

download_columns = st.columns(2)
download_columns[0].download_button(
    "Download scenario summary (CSV)",
    data=summary.to_csv(index=False).encode("utf-8-sig"),
    file_name="phase3_scenario_summary.csv",
    mime="text/csv",
    width="stretch",
)
audit_download = pd.concat(
    [
        result["propagation_details"].assign(Audit_type="Associated cost"),
        result["yield_details"].assign(Audit_type="Yield response"),
    ],
    ignore_index=True,
    sort=False,
)
download_columns[1].download_button(
    "Download coefficient audit trail (CSV)",
    data=audit_download.to_csv(index=False).encode("utf-8-sig"),
    file_name="phase3_coefficient_audit.csv",
    mime="text/csv",
    width="stretch",
)

# Keep a compact result for a later holistic summary page without replacing
# any of the detail shown above.
st.session_state["phase3_latest_result"] = {
    "settings": {
        "selected_key": selected_key,
        "adjustment_percent": float(adjustment_percent),
        "rounds": int(rounds),
        "area": base.get("region"),
        "cultivar": base.get("grape_variety"),
    },
    "summary": summary.to_dict("records"),
}
