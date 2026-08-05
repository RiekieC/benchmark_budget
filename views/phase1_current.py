# Vineyard Production Tool (VPT) — Phase 1
# Run with: streamlit run f1.py
# Includes Organic, Fairtrade, and Organic + Fairtrade scenarios.

import base64
import io
import os
import re
import shutil
from datetime import datetime
from html import escape
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# ---- Visual palette ----
PALETTE = {
    "green_pos": "#1CBA59",
    "red_neg": "#D64545",
    "neutral": "#9AA3B2",
    "neutral_dark": "#6B7280",
    "industry_fill": "#E8EEF6",
    "industry_edge": "#8A93A6",
    "scenario_edge": "#9EA3AA",  # grey dashed
    "zero_line": "#C7CBD1",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def resolve_data_file(filename: str) -> Path:
    """Find project data without depending on the terminal's working folder."""
    candidates = [DATA_DIR / filename, PROJECT_ROOT / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {searched}")


def resolve_first_data_file(*filenames: str) -> Path:
    """Return the first available preferred data file."""
    searched: list[str] = []
    for filename in filenames:
        for candidate in [DATA_DIR / filename, PROJECT_ROOT / filename]:
            searched.append(str(candidate))
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Could not find a suitable data file. Checked: " + ", ".join(searched))

# --- Streamlit layout ---
st.markdown("""
<style>
.main .block-container{
  padding-top: 8px;
}
[data-testid="stAppViewContainer"] .main .block-container {
  padding-left: 8px;
  padding-right: 8px;
}
section[data-testid="stSidebar"] > div:first-child {
  padding-right: 6px;
  padding-left: 6px;
}
[data-testid="stAppViewContainer"] .main {
  margin-left: -6px;
}
section[data-testid="stSidebar"] {
  min-width: 290px;
}
</style>
""", unsafe_allow_html=True)

# --- Intro section ---
st.title("Vineyard Production Tool (VPT)")
st.markdown(
    "<p style='color:#6b7280; margin-top:-10px;'><em>A scenario-based model for benchmarking and decision support.</em></p>",
    unsafe_allow_html=True)

# --- Compact intro (Quick Guide + Contacts combined) ---
with st.expander("Quick Guide", expanded=False):
    st.markdown("""
**Purpose**
- Compare your **vineyard’s production budget and performance** with **regional industry ranges**, from *Low to High*.
- Explore three comparison options:
  - **Scenario 1 – Organic**
  - **Scenario 2 – Fairtrade**
  - **Scenario 3 – Organic + Fairtrade**

**How to use**
1. In the **sidebar**, select **2024 or 2025** and enter the vineyard profile, income, yield and costs.
2. Click a **Scenario** above to switch the comparison mode.
3. Review the **tables and charts** below.
4. **Export** your results as **CSV**, **Excel**, or **PDF**  
   (If PDF isn’t available, export to **HTML** and print to PDF.)

<div style="font-size:0.87rem; color:#555; margin:14px 0 8px 0; line-height:1.45;">
  <em style="color:#444;">Note:</em><br>
  Unless indicated otherwise, monetary values are in Rand (R).<br>
  Scenario and comparison columns reflect the lowest and highest borders of each regional profile.
</div>

<hr style="margin:12px 0 10px 0; border:0; border-top:1px solid #e7e7e7;">

<p style="margin:6px 0 6px 0; font-weight:600;">Questions or feedback?</p>
<ul style="font-size:0.93rem; line-height:1.5; list-style-type:disc; margin:0 0 0 20px;">
  <li>
    <strong>Riekie Cloete</strong>
    <span style="font-variant:small-caps; color:#999;">design</span> :
    <a href="mailto:riekiec@jerryanalytics.co.za" style="text-decoration:none; color:#145fd8;">
      riekiec@jerryanalytics.co.za
    </a>
  </li>
  <li>
    <strong>Petri de Beer</strong>
    <span style="font-variant:small-caps; color:#999;">content</span> :
    <a href="mailto:petridb@gmail.com" style="text-decoration:none; color:#145fd8;">
      petridb@gmail.com
    </a>
  </li>
</ul>
""", unsafe_allow_html=True)


# -----------------------
# Data loading
# -----------------------
@st.cache_data
def load_data():
    try:
        income = pd.read_csv(
            resolve_first_data_file(
                "income_2024_2025_with_average_annual_growth_percent.csv",
                "income.csv",
            ),
            encoding="utf-8-sig",
        )
        yield_df = pd.read_csv(
            resolve_first_data_file("yield.csv"),
            encoding="utf-8-sig",
        )
        costs = pd.read_csv(
            resolve_first_data_file(
                "costs_2024_2025_with_blended_growth_percent.csv",
                "costs.csv",
            ),
            encoding="utf-8-sig",
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Copy the combined 2024/2025 income and cost files, together with yield.csv, "
            "into the project data folder and rerun the app."
        )
        st.stop()

    for df in (income, yield_df, costs):
        df.columns = [str(c).strip() for c in df.columns]

    def norm(df):
        for col in ["Wine Class", "Grape Variety", "Region", "Band"]:
            if col in df.columns:
                if col in {"Band", "Wine Class"}:
                    df[col] = df[col].astype(str).str.strip().str.title()
                else:
                    df[col] = df[col].astype(str).str.strip()
        return df

    income = norm(income)
    yield_df = norm(yield_df)
    costs = norm(costs)

    if "Income_R_per_t" not in income.columns:
        guess = [c for c in income.columns if "income" in c.lower()]
        if guess:
            income = income.rename(columns={guess[0]: "Income_R_per_t"})

    if "Yield_t_per_ha" not in yield_df.columns:
        guess = [c for c in yield_df.columns if "yield" in c.lower()]
        if guess:
            yield_df = yield_df.rename(columns={guess[0]: "Yield_t_per_ha"})

    if "YEAR" not in income.columns and "Year" in income.columns:
        income = income.rename(columns={"Year": "YEAR"})
    if "Year" not in costs.columns and "YEAR" in costs.columns:
        costs = costs.rename(columns={"YEAR": "Year"})

    # If yield came in kg/ha, normalise to t/ha.
    med = pd.to_numeric(yield_df["Yield_t_per_ha"], errors="coerce").median()
    if pd.notna(med) and med > 1000:
        yield_df["Yield_t_per_ha"] = (
            pd.to_numeric(yield_df["Yield_t_per_ha"], errors="coerce") / 1000.0
        )

    if "Avg_Cost" not in costs.columns:
        guesses = [c for c in costs.columns if "cost" in c.lower()]
        if guesses:
            costs = costs.rename(columns={guesses[0]: "Avg_Cost"})
        else:
            costs["Avg_Cost"] = 0.0

    for col in ["Category", "Item", "Region"]:
        if col not in costs.columns:
            costs[col] = "Unknown"

    income["Income_R_per_t"] = pd.to_numeric(income["Income_R_per_t"], errors="coerce")
    yield_df["Yield_t_per_ha"] = pd.to_numeric(yield_df["Yield_t_per_ha"], errors="coerce")
    costs["Avg_Cost"] = pd.to_numeric(costs["Avg_Cost"], errors="coerce").fillna(0.0)
    if "YEAR" in income.columns:
        income["YEAR"] = pd.to_numeric(income["YEAR"], errors="coerce").astype("Int64")
    if "Year" in costs.columns:
        costs["Year"] = pd.to_numeric(costs["Year"], errors="coerce").astype("Int64")

    return income, yield_df, costs


income_df, yield_df, costs_df = load_data()


# -----------------------
# Helpers
# -----------------------
def money(x):
    if x is None or pd.isna(x): return ""
    return f"R {x:,.2f}"


def num(x):
    if x is None or pd.isna(x): return ""
    return f"{x:,.2f}"


def money_n(x):  # no symbol (for the report table)
    return f"{x:,.2f}"


def lookup_band(df, wine_class, grape, region, band, col, year=None):
    mask = (
            (df["Wine Class"].str.lower() == wine_class.lower()) &
            (df["Grape Variety"].str.lower() == grape.lower()) &
            (df["Region"].str.lower() == region.lower()) &
            (df["Band"].str.lower() == band.lower())
    )
    if year is not None and "YEAR" in df.columns:
        mask &= (df["YEAR"].astype("Int64") == year)
    vals = df.loc[mask, col]
    if not vals.empty and not pd.isna(vals.iloc[0]):
        return float(vals.iloc[0])

    other = "Other White" if wine_class.lower() == "white" else "Other Red"
    mask = (
            (df["Wine Class"].str.lower() == wine_class.lower()) &
            (df["Grape Variety"].str.lower() == other.lower()) &
            (df["Region"].str.lower() == region.lower()) &
            (df["Band"].str.lower() == band.lower())
    )
    if year is not None and "YEAR" in df.columns:
        mask &= (df["YEAR"].astype("Int64") == year)
    vals = df.loc[mask, col]
    return float(vals.iloc[0]) if not vals.empty and not pd.isna(vals.iloc[0]) else None


FORBIDDEN = {"provision for renewal", "total expenditure", "totale kontant uitgawes",
             "net farm income (r/ha)", "income (r/ha)"}


def is_forbidden(text): return str(text).strip().lower() in FORBIDDEN


def filter_costs_for_year(costs: pd.DataFrame, selected_year: int | None) -> pd.DataFrame:
    """Return cost rows for the selected benchmark year."""
    out = costs.copy()
    if selected_year is not None and "Year" in out.columns:
        year_values = pd.to_numeric(out["Year"], errors="coerce")
        out = out[year_values == int(selected_year)].copy()
    return out


def lookup_industry_provision(
    costs: pd.DataFrame,
    region_name: str,
    selected_year: int | None,
    default: float = 20125.0,
) -> float:
    """Read the regional provision-for-renewal benchmark from the cost data."""
    sub = filter_costs_for_year(costs, selected_year)
    sub = sub[sub["Region"].astype(str).str.strip().str.lower() == str(region_name).strip().lower()]
    provision_mask = (
        sub["Category"].astype(str).str.strip().str.lower().eq("provision for renewal")
        | sub["Item"].astype(str).str.strip().str.lower().eq("provision for renewal")
    )
    values = pd.to_numeric(sub.loc[provision_mask, "Avg_Cost"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float(default)


def build_scenarios(selected_year: int | None) -> dict:
    """Return the sustainability assumptions belonging to the selected year."""
    year_key = 2024 if int(selected_year or 2025) <= 2024 else 2025
    assumptions = {
        2024: {
            "organic_material": 13349,
            "crop_protection": 1053,
            "fuel": 110,
            "organic_admin": 555,
            "fairtrade_labour_pct": 7.9,
            "fairtrade_admin": 1666,
            "combined_admin": 2221,
        },
        2025: {
            "organic_material": 16081,
            "crop_protection": 1053,
            "fuel": 150,
            "organic_admin": 610,
            "fairtrade_labour_pct": 34.4,
            "fairtrade_admin": 1830,
            "combined_admin": 2440,
        },
    }[year_key]

    return {
        "Scenario 1 — Organic": {
            "label": "Organic",
            "diff_multiplier": 1.41,
            "yield_high_pct": 0.0,
            "income_abs": 0.0,
            "cost_rules": [
                ("item", "Saad, organiese bemesting en materiaal", "set", assumptions["organic_material"]),
                ("item", "Gewasbeskerming (swam- en insekbeheer)", "set", assumptions["crop_protection"]),
                ("item", "Brandstof (petrol en diesel) en smeermiddels", "abs", assumptions["fuel"]),
                ("item", "Administrasie", "abs", assumptions["organic_admin"]),
                ("item", "Kunsmis, blaar- en grondontledings", "set", 0),
                ("item", "Onkruiddoder", "set", 0),
            ],
        },
        "Scenario 2 — Fairtrade": {
            "label": "Fairtrade",
            "yield_low_pct": 0.0,
            "yield_high_pct": 0.0,
            "income_abs": 30.0,
            "cost_rules": [
                ("item", "Permanente arbeid", "pct", assumptions["fairtrade_labour_pct"]),
                ("item", "Seisoensarbeid en kontrakwerk", "pct", assumptions["fairtrade_labour_pct"]),
                ("item", "Administrasie", "abs", assumptions["fairtrade_admin"]),
            ],
        },
        "Scenario 3 — Organic + Fairtrade": {
            "label": "Organic + Fairtrade",
            "diff_multiplier": 1.41,
            "yield_high_pct": 0.0,
            "income_abs": 30.0,
            "cost_rules": [
                ("item", "Saad, organiese bemesting en materiaal", "set", assumptions["organic_material"]),
                ("item", "Gewasbeskerming (swam- en insekbeheer)", "set", assumptions["crop_protection"]),
                ("item", "Brandstof (petrol en diesel) en smeermiddels", "abs", assumptions["fuel"]),
                ("item", "Administrasie", "abs", assumptions["combined_admin"]),
                ("item", "Permanente arbeid", "pct", assumptions["fairtrade_labour_pct"]),
                ("item", "Seisoensarbeid en kontrakwerk", "pct", assumptions["fairtrade_labour_pct"]),
                ("item", "Kunsmis, blaar- en grondontledings", "set", 0),
                ("item", "Onkruiddoder", "set", 0),
                ("item", "Veevoer en medisyne / Droogmiddels", "set", 0),
            ],
        },
    }


# Fixed Step-1 category order
CATEGORY_ORDER = ["Direkte koste", "Arbeid", "Meganisasie", "Vaste verbeteringe", "Algemene uitgawes"]

# -----------------------
# Sidebar — farmer inputs
# -----------------------
with st.sidebar:
    st.title("Farmer Inputs")
    st.caption("Inputs start with default vineyard values. Adjust as needed for your case.")
    st.subheader("Vineyard Profile")

    wine_classes = sorted(yield_df["Wine Class"].dropna().unique().tolist())
    wine_class = st.selectbox("Wine Class", wine_classes,
                              index=(wine_classes.index("White") if "White" in wine_classes else 0))
    grapes = sorted(yield_df.loc[yield_df["Wine Class"] == wine_class, "Grape Variety"].dropna().unique().tolist())
    grape_variety = st.selectbox("Grape Variety", grapes,
                                 index=(grapes.index("Chenin blanc") if "Chenin blanc" in grapes else 0))
    regions = sorted(set(
        yield_df.loc[(yield_df["Wine Class"] == wine_class) & (yield_df["Grape Variety"] == grape_variety), "Region"]
    ).union(set(income_df.loc[
                    (income_df["Wine Class"] == wine_class) & (income_df["Grape Variety"] == grape_variety), "Region"]))
                     .union(set(costs_df["Region"])))
    region = st.selectbox("Region", regions, index=(regions.index("Breedekloof") if "Breedekloof" in regions else 0))

    years = sorted(income_df.get("YEAR", pd.Series(dtype=float)).dropna().astype(int).unique().tolist())
    year = (
        st.selectbox(
            "Benchmark data year",
            years,
            index=(len(years) - 1 if years else 0),
            help="Choose the historical 2024 benchmark or the latest 2025 benchmark.",
        )
        if years else None
    )
    selected_costs_df = filter_costs_for_year(costs_df, int(year) if year is not None else None)
    industry_provision_default = lookup_industry_provision(
        costs_df, region, int(year) if year is not None else None
    )

    st.markdown("---")
    st.header("Farmer Income")
    gross_income_current = st.number_input("Gross Income (R/t)", min_value=0.0, step=50.0, value=5000.0, format="%.2f")
    yield_current = st.number_input("Yield (t/ha)", min_value=0.0, step=0.1, value=24.6, format="%.2f")

    st.markdown("### Farmer Costs")
    st.caption("Enter costs by component")
    region_costs_default = selected_costs_df[selected_costs_df["Region"].str.lower() == region.lower()].copy()
    if region_costs_default.empty:
        st.warning("No costs found for this region. All costs default to 0. Edit as needed.")

    cost_inputs = []
    cats_seen = region_costs_default["Category"].tolist()
    cats_in_order = [c for c in CATEGORY_ORDER if c in cats_seen]
    cats_in_order += [c for c in region_costs_default["Category"].unique() if c not in cats_in_order]

    for cat in cats_in_order:
        if is_forbidden(cat): continue
        with st.expander(f"{cat}", expanded=False):
            sub = region_costs_default[
                (region_costs_default["Category"] == cat) &
                (~region_costs_default["Item"].apply(is_forbidden))
                ]
            for _, r in sub.iterrows():
                # Include the region so its default costs do not leak into a
                # newly selected region through Streamlit session state.
                key = f"cost::{year}::{region}::{cat}::{r['Item']}"
                default_val = float(r["Avg_Cost"])

                val = st.number_input(
                    f"{r['Item']}",
                    min_value=0.0, step=10.0,
                    value=default_val,
                    format="%.2f", key=key
                )
                cost_inputs.append((cat, r["Item"], val))

    provision_for_renewal = st.number_input(
        "Provision for renewal (R/ha)",
        min_value=0.0,
        step=100.0,
        value=float(industry_provision_default),
        format="%.2f",
        key=f"provision::{year}::{region}",
    )

# -----------------------
# Align lists and enforce ordering
# -----------------------
costs_df_user = pd.DataFrame(cost_inputs, columns=["Category", "Item", "Cost"])
region_costs_avg = selected_costs_df[
    (selected_costs_df["Region"].str.lower() == region.lower()) &
    (~selected_costs_df["Category"].apply(is_forbidden)) &
    (~selected_costs_df["Item"].apply(is_forbidden))
    ].copy()
avg_items = region_costs_avg[["Category", "Item", "Avg_Cost"]].rename(columns={"Avg_Cost": "Avg"}).copy()

farmer_items = costs_df_user.copy()
key_all = (pd.DataFrame({
    "Category": pd.concat([farmer_items["Category"], avg_items["Category"]], ignore_index=True),
    "Item": pd.concat([farmer_items["Item"], avg_items["Item"]], ignore_index=True)
}).drop_duplicates())

aligned = key_all.merge(farmer_items, on=["Category", "Item"], how="left").merge(avg_items, on=["Category", "Item"],
                                                                                 how="left")
aligned["Cost"] = pd.to_numeric(aligned["Cost"], errors="coerce")
aligned["Avg"] = pd.to_numeric(aligned["Avg"], errors="coerce")

cat_order_map = {c: i for i, c in enumerate(CATEGORY_ORDER)}
aligned["__cat_order"] = aligned["Category"].map(lambda c: cat_order_map.get(c, 999))
order_src = selected_costs_df[
    (selected_costs_df["Region"].str.lower() == region.lower()) &
    (~selected_costs_df["Category"].apply(is_forbidden)) &
    (~selected_costs_df["Item"].apply(is_forbidden))
    ].copy()
order_src["__item_idx"] = order_src.groupby("Category").cumcount()
aligned = aligned.merge(order_src[["Category", "Item", "__item_idx"]], on=["Category", "Item"], how="left")
aligned["__item_idx"] = aligned["__item_idx"].fillna(999999).astype(int)
aligned = aligned.sort_values(["__cat_order", "__item_idx"], kind="stable").reset_index(drop=True)

# -----------------------
# Baselines
# -----------------------
INDUSTRY_PROVISION = float(industry_provision_default)

industry_total_cash = float(pd.to_numeric(aligned["Avg"], errors="coerce").fillna(0).sum())
industry_total_expenditure = industry_total_cash + INDUSTRY_PROVISION

total_cash_user = float(pd.to_numeric(aligned["Cost"], errors="coerce").fillna(0).sum())
total_expenditure_user = total_cash_user + provision_for_renewal

# Income & yield bands
income_low = lookup_band(income_df, wine_class, grape_variety, region, "Low", "Income_R_per_t", year)
income_high = lookup_band(income_df, wine_class, grape_variety, region, "High", "Income_R_per_t", year)
yield_low = lookup_band(yield_df, wine_class, grape_variety, region, "Low", "Yield_t_per_ha", None)
yield_high = lookup_band(yield_df, wine_class, grape_variety, region, "High", "Yield_t_per_ha", None)

income_current_rha = gross_income_current * yield_current
income_low_rha = (income_low * yield_low) if (income_low is not None and yield_low is not None) else None
income_high_rha = (income_high * yield_high) if (income_high is not None and yield_high is not None) else None

net_current = income_current_rha - total_expenditure_user
net_low = (income_low_rha - industry_total_expenditure) if income_low_rha is not None else None
net_high = (income_high_rha - industry_total_expenditure) if income_high_rha is not None else None

# -----------------------
# Scenarios (Organic, Fairtrade)
# -----------------------
SCENARIOS = build_scenarios(int(year) if year is not None else 2025)


def _apply_abs(x, inc):
    return (0.0 if pd.isna(x) else float(x)) + float(inc)


def apply_cost_rules(base_df: pd.DataFrame, rules):
    """
    rules: list of (itype, label, mode, value)
      itype:  "category" | "item"
      mode:   "abs" (add R/ha), "pct" (add %), "set" (replace with this R/ha)
    """
    out = base_df.copy()
    for itype, label, mode, val in rules:
        if itype == "category":
            mask = out["Category"].str.strip().str.lower() == label.strip().lower()
        else:
            mask = out["Item"].str.strip().str.lower() == label.strip().lower()

        if mode == "set":
            out.loc[mask, "Value"] = float(val)
        elif mode == "abs":
            out.loc[mask, "Value"] = out.loc[mask, "Value"].apply(
                lambda v: (0.0 if pd.isna(v) else float(v)) + float(val)
            )
        elif mode == "pct":
            out.loc[mask, "Value"] = out.loc[mask, "Value"].apply(
                lambda v: (0.0 if pd.isna(v) else float(v)) * (1.0 + float(val) / 100.0)
            )
    return out


def build_year_profile_result(compare_year: int, cfg_key: str) -> dict:
    """Calculate one comparable industry/scenario position for a historical year."""
    year_costs = filter_costs_for_year(costs_df, compare_year)
    year_costs = year_costs[
        (year_costs["Region"].astype(str).str.strip().str.lower() == region.strip().lower())
        & (~year_costs["Category"].apply(is_forbidden))
        & (~year_costs["Item"].apply(is_forbidden))
    ].copy()
    base_costs = year_costs[["Category", "Item", "Avg_Cost"]].rename(
        columns={"Avg_Cost": "Value"}
    )
    base_costs["Value"] = pd.to_numeric(base_costs["Value"], errors="coerce").fillna(0.0)

    provision = lookup_industry_provision(costs_df, region, compare_year)
    industry_cash = float(base_costs["Value"].sum())
    industry_total = industry_cash + provision

    cfg = build_scenarios(compare_year)[cfg_key]
    scenario_costs = apply_cost_rules(base_costs, cfg["cost_rules"])
    scenario_cash = float(pd.to_numeric(scenario_costs["Value"], errors="coerce").fillna(0.0).sum())
    scenario_total = scenario_cash + provision

    inc_low = lookup_band(
        income_df, wine_class, grape_variety, region, "Low", "Income_R_per_t", compare_year
    )
    inc_high = lookup_band(
        income_df, wine_class, grape_variety, region, "High", "Income_R_per_t", compare_year
    )
    y_low_base = lookup_band(
        yield_df, wine_class, grape_variety, region, "Low", "Yield_t_per_ha", None
    )
    y_high_base = lookup_band(
        yield_df, wine_class, grape_variety, region, "High", "Yield_t_per_ha", None
    )

    industry_income_low = (
        inc_low * y_low_base if inc_low is not None and y_low_base is not None else None
    )
    industry_income_high = (
        inc_high * y_high_base if inc_high is not None and y_high_base is not None else None
    )
    industry_net_low = (
        industry_income_low - industry_total if industry_income_low is not None else None
    )
    industry_net_high = (
        industry_income_high - industry_total if industry_income_high is not None else None
    )

    scenario_income_low_rt = _apply_abs(inc_low, cfg.get("income_abs", 0.0)) if inc_low is not None else None
    scenario_income_high_rt = _apply_abs(inc_high, cfg.get("income_abs", 0.0)) if inc_high is not None else None

    scenario_yield_low = None
    scenario_yield_high = None
    if y_low_base is not None and y_high_base is not None:
        if cfg.get("diff_multiplier") is not None:
            difference = float(y_high_base) - float(y_low_base)
            scenario_yield_high = float(y_high_base)
            scenario_yield_low = float(y_high_base) - difference * float(cfg["diff_multiplier"])
        else:
            scenario_yield_low = float(y_low_base) * (
                1.0 + float(cfg.get("yield_low_pct", 0.0)) / 100.0
            )
            scenario_yield_high = float(y_high_base) * (
                1.0 + float(cfg.get("yield_high_pct", 0.0)) / 100.0
            )

    scenario_income_low = (
        scenario_income_low_rt * scenario_yield_low
        if scenario_income_low_rt is not None and scenario_yield_low is not None
        else None
    )
    scenario_income_high = (
        scenario_income_high_rt * scenario_yield_high
        if scenario_income_high_rt is not None and scenario_yield_high is not None
        else None
    )
    scenario_net_low = (
        scenario_income_low - scenario_total if scenario_income_low is not None else None
    )
    scenario_net_high = (
        scenario_income_high - scenario_total if scenario_income_high is not None else None
    )

    return {
        "year": compare_year,
        "industry_cash": industry_cash,
        "industry_provision": provision,
        "industry_total": industry_total,
        "scenario_cash": scenario_cash,
        "scenario_total": scenario_total,
        "income_low_rt": inc_low,
        "income_high_rt": inc_high,
        "industry_net_low": industry_net_low,
        "industry_net_high": industry_net_high,
        "scenario_net_low": scenario_net_low,
        "scenario_net_high": scenario_net_high,
    }


def historical_cost_change_frame() -> pd.DataFrame:
    """Return comparable regional line-item changes between 2024 and 2025."""
    frames = []
    for compare_year in [2024, 2025]:
        sub = filter_costs_for_year(costs_df, compare_year)
        sub = sub[
            (sub["Region"].astype(str).str.strip().str.lower() == region.strip().lower())
            & (~sub["Category"].apply(is_forbidden))
            & (~sub["Item"].apply(is_forbidden))
        ][["Category", "Item", "Avg_Cost"]].copy()
        sub = sub.rename(columns={"Avg_Cost": str(compare_year)})
        frames.append(sub)

    if len(frames) != 2:
        return pd.DataFrame()

    comparison = frames[0].merge(frames[1], on=["Category", "Item"], how="outer")
    comparison["2024"] = pd.to_numeric(comparison["2024"], errors="coerce").fillna(0.0)
    comparison["2025"] = pd.to_numeric(comparison["2025"], errors="coerce").fillna(0.0)
    comparison["Change (R/ha)"] = comparison["2025"] - comparison["2024"]
    comparison["Change (%)"] = comparison.apply(
        lambda row: (
            (row["Change (R/ha)"] / row["2024"]) * 100.0
            if abs(float(row["2024"])) > 1e-9
            else (0.0 if abs(float(row["2025"])) <= 1e-9 else None)
        ),
        axis=1,
    )
    comparison["__abs_change"] = comparison["Change (R/ha)"].abs()
    return comparison.sort_values("__abs_change", ascending=False).drop(columns="__abs_change")


def build_scenario(cfg_key: str):
    cfg = SCENARIOS[cfg_key]

    # Costs start from region AVG (fixed per region)
    base_costs = aligned[["Category", "Item", "Avg"]].rename(columns={"Avg": "Value"})
    sc_costs = apply_cost_rules(base_costs, cfg["cost_rules"])
    sc_total_cash = float(pd.to_numeric(sc_costs["Value"], errors="coerce").fillna(0).sum())
    sc_provision = INDUSTRY_PROVISION  # keep industry provision for scenario
    sc_total_exp = sc_total_cash + sc_provision

    # Income: start from bands, then apply scenario modifiers
    income_low_rt = _apply_abs(income_low, cfg["income_abs"]) if income_low is not None else None
    income_high_rt = _apply_abs(income_high, cfg["income_abs"]) if income_high is not None else None

    # Per-band yield adjustments
    # --- Yield logic ---
    # Default: pct adjustments (your existing behavior)
    y_low = None
    y_high = None

    if (yield_low is not None) and (yield_high is not None):
        diff_mult = cfg.get("diff_multiplier", None)

        if diff_mult is not None:
            # Organic logic: widen the downside band by multiplier, keep upper unchanged
            diff = float(yield_high) - float(yield_low)
            y_high = float(yield_high)
            y_low = float(yield_high) - (diff * float(diff_mult))
        else:
            # Original per-band % logic
            y_low_pct = cfg.get("yield_low_pct", 0.0)
            y_high_pct = cfg.get("yield_high_pct", 0.0)
            y_low = float(yield_low) * (1.0 + float(y_low_pct) / 100.0)
            y_high = float(yield_high) * (1.0 + float(y_high_pct) / 100.0)

    inc_low_rha = (income_low_rt * y_low) if (income_low_rt is not None and y_low is not None) else None
    inc_high_rha = (income_high_rt * y_high) if (income_high_rt is not None and y_high is not None) else None

    net_low_sc = (inc_low_rha - sc_total_exp) if inc_low_rha is not None else None
    net_high_sc = (inc_high_rha - sc_total_exp) if inc_high_rha is not None else None

    return {
        "label": cfg["label"],
        "costs": sc_costs,
        "total_cash": sc_total_cash,
        "provision": sc_provision,
        "total_exp": sc_total_exp,
        "income_rt_low": income_low_rt,
        "income_rt_high": income_high_rt,
        "yield_low": y_low,
        "yield_high": y_high,
        "income_rha_low": inc_low_rha,
        "income_rha_high": inc_high_rha,
        "net_low": net_low_sc,
        "net_high": net_high_sc,
    }


scenario_options = list(SCENARIOS.keys())

scenario_key = st.segmented_control(
    "Select Scenario",
    options=scenario_options,
    default=scenario_options[0],
    help="Click to switch scenarios.",
    key="scenario_label",
)

SC = build_scenario(scenario_key)
SC_LABEL = SC["label"]

# -----------------------
# Share Phase 1 current/base inputs with Phases 2 and 3
# -----------------------
# In the multipage app, Streamlit keeps st.session_state while the user moves
# between pages. Phase 2 uses these values as the forecast base year, while
# Phase 3 uses the unchanged current-year position for its cost scenarios.
st.session_state["phase1_current_base"] = {
    "wine_class": wine_class,
    "grape_variety": grape_variety,
    "region": region,
    "year": int(year) if year is not None else None,
    "scenario_key": scenario_key,
    "scenario_label": SC_LABEL,
    "farmer_income_rt": float(gross_income_current),
    "farmer_yield": float(yield_current),
    "industry_yield_low": float(yield_low) if yield_low is not None else None,
    "industry_yield_high": float(yield_high) if yield_high is not None else None,
    "industry_income_rt_low": float(income_low) if income_low is not None else None,
    "industry_income_rt_high": float(income_high) if income_high is not None else None,
    "provision_for_renewal": float(provision_for_renewal),
    "farmer_total_cost": float(total_expenditure_user),
    "farmer_revenue_rha": float(income_current_rha),
    "farmer_nfi_rha": float(net_current),
    "farmer_costs": aligned[["Category", "Item", "Cost"]].to_dict("records"),
    "last_updated": datetime.now().isoformat(timespec="seconds"),
}

# -----------------------
# Build rows
# -----------------------
rows = []
rows += [
    # ["", "", "", ""],
    ["Year", str(year) if year is not None else "", "", ""],
    ["Wine Class", wine_class, "", ""],
    ["Grape Variety", grape_variety, "", ""],
    ["Region", region, "", ""],
    ["", "", "", ""],
    ["Vineyard Performance", "", "", ""],
]

# Top numbers
rows += [
    ["Gross Income (R/t)",
     money(gross_income_current),
     f"{money(income_low)} / {money(income_high)}",
     f"{money(SC['income_rt_low'])} / {money(SC['income_rt_high'])}"],
    ["Yield (t/ha)",
     num(yield_current),
     f"{num(yield_low)} / {num(yield_high)}",
     f"{num(SC['yield_low'])} / {num(SC['yield_high'])}"],
    ["Income (R/ha)",
     money(income_current_rha),
     f"{money(income_low_rha)} / {money(income_high_rha)}",
     f"{money(SC['income_rha_low'])} / {money(SC['income_rha_high'])}"],
    ["", "", "", ""],
]

# --- Safety: if any first-column cell is a list/tuple, stringify it (prevents Arrow error) ---
for i, r in enumerate(rows):
    if isinstance(r[0], (list, tuple)):
        rows[i][0] = " ".join(map(str, r[0]))

# Totals per category for Farmer/Avg/Scenario
cat_tot_farmer = aligned.groupby("Category", as_index=False)["Cost"].sum()
cat_tot_avg = aligned.groupby("Category", as_index=False)["Avg"].sum()
sc_map = {(r.Category, r.Item): float(r.Value) if pd.notna(r.Value) else 0.0
          for r in SC["costs"].itertuples(index=False)}
sc_cat = SC["costs"].groupby("Category", as_index=False)["Value"].sum()

for cat in aligned["Category"].unique():
    if is_forbidden(cat): continue
    f_total = float(cat_tot_farmer.loc[cat_tot_farmer["Category"] == cat, "Cost"].fillna(0).iloc[0]) if cat in \
                                                                                                        cat_tot_farmer[
                                                                                                            "Category"].values else 0.0
    a_total = float(cat_tot_avg.loc[cat_tot_avg["Category"] == cat, "Avg"].fillna(0).iloc[0]) if cat in cat_tot_avg[
        "Category"].values else 0.0
    s_total = float(sc_cat.loc[sc_cat["Category"] == cat, "Value"].fillna(0).iloc[0]) if cat in sc_cat[
        "Category"].values else 0.0
    rows.append([cat, money(f_total), money(a_total), money(s_total)])

    sub = aligned[aligned["Category"] == cat]
    for _, r in sub.iterrows():
        s_val = sc_map.get((r['Category'], r['Item']), None)
        rows.append([f"  {r['Item']}", money(r["Cost"]), money(r["Avg"]), money(s_val)])

# Totals (no R symbol in cells)
rows += [
    ["Totale kontant uitgawes", money_n(total_cash_user), money_n(industry_total_cash), money_n(SC["total_cash"])],
    ["Provision for renewal", money_n(provision_for_renewal), money_n(INDUSTRY_PROVISION), money_n(SC["provision"])],
    ["Total Expenditure", money_n(total_expenditure_user), money_n(industry_total_expenditure),
     money_n(SC["total_exp"])],
    ["Net Farm Income (R/ha)", money_n(net_current),
     f"{money_n(net_low)} / {money_n(net_high)}",
     f"{money_n(SC['net_low'])} / {money_n(SC['net_high'])}"],
]

# Build DataFrame for export/render (internal names, then rename for display)
cols_long = ["Section / Item", "Farmer Inputs / Current", "Industry Comparison", f"Scenario — {SC_LABEL} (rules)"]
table = pd.DataFrame(rows, columns=cols_long)


# ---------- Styled HTML report (profile split + sticky bottom, aligned headers) ----------
def build_html(table_long, label_short):
    # 1) Shorter column headers
    t = table_long.rename(columns={
        f"Scenario — {label_short} (rules)": label_short
    }).copy()

    # 2) Remove "R" currency symbol from all VALUE columns (not the first label column)
    def strip_currency_R(x: object) -> str:
        s = str(x)
        # normalise non-breaking spaces: \u00A0 (NBSP) and \u202F (narrow NBSP)
        s = s.replace('\u00A0', ' ').replace('\u202F', ' ')
        # delete "R" only when it precedes a number, e.g. "R 1,234.56", "R123" … but
        # do NOT touch words like "Region", "Red", etc.
        # works inside pairs too: "R 132,734.88 / R 164,025.78"
        return re.sub(r'(?<![A-Za-z])R\s*(?=\d)', '', s)

    # apply to every column except the first (labels)
    if t.shape[1] > 1:
        t.iloc[:, 1:] = t.iloc[:, 1:].map(strip_currency_R)

    # Row → CSS class (simple & robust)
    def classify_row(row):
        first = str(row.iloc[0]).strip().lower()
        if first == "vineyard performance":
            return "row-inline-title"
        if first.startswith("totale kontant uitgawes"):
            return "row-total"
        if first.startswith("provision for renewal"):
            return "row-provision"
        if first.startswith("total expenditure") or first.startswith("net farm income"):
            return "row-grand"
        if str(row.iloc[0]).startswith("  "):
            return "row-item"
        return "row-cat"

    # 2) Shared header + colgroup (SAME for top & bottom)
    headers = list(t.columns)

    # Set a generous first-column width so "Section / Item" and long items don’t wrap awkwardly.
    FIRST_W = 220  # px
    colgroup_html = (
        f'<colgroup>'
        f'<col style="width:{FIRST_W}px">'
        f'<col style="width:26%"><col style="width:26%"><col style="width:26%">'
        f'</colgroup>'
    )

    default_thead = "<tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr>"

    # 3) Split right AFTER the “Income (R/ha)” line
    first_col_norm = t.iloc[:, 0].astype(str).str.strip().str.lower()
    income_aliases = {"income (r/ha)", "income r/ha", "inkomste (r/ha)", "inkomste r/ha"}
    if first_col_norm.isin(income_aliases).any():
        split_at = first_col_norm[first_col_norm.isin(income_aliases)].index[0] + 1
    else:
        cand = first_col_norm[(first_col_norm.str.contains("income|inkomste")) &
                              (first_col_norm.str.contains("r/ha"))]
        split_at = cand.index[0] + 1 if len(cand) else None

    if split_at is not None:
        top_df = t.iloc[:split_at, :]
        bottom_df = t.iloc[split_at:, :].copy()
    else:
        top_df = t.copy()
        bottom_df = t.iloc[0:0, :].copy()

    # 4) Render TOP: profile mini-grid, then the “Vineyard Performance” title + inline headers
    def rows_to_html_top(df):
        html_rows, i, n = [], 0, len(df)

        def _is_profile_label(s):
            s = str(s).strip().lower()
            return s in {"year", "wine class", "grape variety", "region"}

        prof_labels, prof_values = [], []
        while i < n and _is_profile_label(df.iloc[i, 0]):
            prof_labels.append(str(df.iloc[i, 0]).strip())
            prof_values.append(str(df.iloc[i, 1]).strip() if df.shape[1] > 1 else "")
            i += 1

        # Compact profile grid
        if prof_labels:
            inner_heads = "".join(f"<th>{escape(h)}</th>" for h in prof_labels)
            inner_vals = "".join(f"<td>{escape(v)}</td>" for v in prof_values)
            inner_html = (
                '<table class="profile-grid">'
                f'<thead><tr>{inner_heads}</tr></thead>'
                f'<tbody><tr>{inner_vals}</tr></tbody>'
                '</table>'
            )
            html_rows.append(f'<tr class="row-profile"><td colspan="{len(headers)}">{inner_html}</td></tr>')

        # Title + inline colheads (these colheads will match the same right-align as bottom)
        while i < n:
            row = df.iloc[i]
            if classify_row(row) == "row-inline-title":
                title = escape(str(row.iloc[0]))
                html_rows.append(f'<tr class="row-inline-title"><td colspan="{len(headers)}">{title}</td></tr>')
                hdr_tds = "".join(f"<td>{escape(str(h))}</td>" for h in headers)
                html_rows.append(f'<tr class="row-colheads">{hdr_tds}</tr>')
            else:
                tds = "".join(f"<td>{escape(str(v))}</td>" for v in row.values)
                html_rows.append(f"<tr>{tds}</tr>")
            i += 1

        return "\n".join(html_rows)

    def rows_to_html_bottom(df):
        html_rows = []
        for _, row in df.iterrows():
            css = classify_row(row)

            # --- build each cell, allowing wrap when needed ---
            cells = []
            for v in row.values:
                s = str(v)
                # detect long or paired values like "R 132,734.88 / R 164,025.78"
                needs_wrap = ("/" in s) or (len(s) > 22)
                cls = ' class="allow-wrap"' if needs_wrap else ""
                cells.append(f"<td{cls}>{escape(s)}</td>")

            tds = "".join(cells)
            html_rows.append(f'<tr class="{css}">{tds}</tr>')

        return "\n".join(html_rows)

    tbody_html_top = rows_to_html_top(top_df)

    # If bottom_df starts with a copy of the headers, use it as the real thead
    thead_html_bottom = default_thead
    if len(bottom_df):
        first_row = bottom_df.iloc[0].astype(str).tolist()
        if [s.strip().lower() for s in first_row[1:]] == [s.strip().lower() for s in headers[1:]]:
            thead_html_bottom = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in first_row) + "</tr>"
            bottom_df = bottom_df.iloc[1:, :]

    tbody_html_bottom = rows_to_html_bottom(bottom_df)

    # 5) Full HTML
    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:8px; }}
        .report {{ font-variant-numeric: tabular-nums; }}

        /* Base tables */
        .tbl {{ border-collapse: collapse; border-spacing: 0; width: 100%; font-size: 0.95rem; table-layout: fixed; }}
        .tbl th, .tbl td {{ padding: 6px 10px; border-bottom: 1px solid #e8e8e8; vertical-align: top; }}
        .tbl td:nth-child(n+2) {{ text-align: right; white-space: nowrap; }}   /* numeric cells */
        .tbl th:first-child, .tbl td:first-child {{ padding-right: 6px; }}

        /* Allow wrapping for long pair values like "R 132,734.88 / R 164,025.78" */
        .tbl td.allow-wrap {{
          white-space: normal !important;
          overflow-wrap: anywhere;
          word-break: break-word;
        }}

        @media (max-width: 1400px) {{
          .tbl td:nth-child(n+2) {{
            white-space: normal !important;
            overflow-wrap: anywhere;
            word-break: break-word;
          }}
        }}


        /* Titles */
        .section-title {{ font-size: 1.08rem; font-weight: 700; color:#333; margin:12px 0 6px; letter-spacing:0.2px; text-align:center; position:relative; }}
        .section-title::after {{ content:""; display:block; height:1px; background:#e6e6e6; margin-top:8px; }}

        /* Profile mini-grid */
        .profile-grid {{ width:100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; font-size:0.95rem; }}
        .profile-grid thead th {{
          text-align: right;
          font-weight: 700;
          color: #374151;
          background: #f8fafc;
          border: 1px solid #e8e8e8;
          padding: 6px 10px;
        }}

        /* Override for the first column (Year) */
        .profile-grid thead th:first-child,
        .profile-grid tbody td:first-child {{
          text-align: left !important;
        }}
        .profile-grid th:first-child,
        .profile-grid td:first-child {{
          width: 20%;
        }}


        .profile-grid tbody td {{ border:1px solid #e8e8e8; padding:8px 10px; background:#fff; }}
        .row-profile td {{ padding:0; }}

        /* Vineyard Performance title + inline headers */
        .row-inline-title td {{ font-weight:700; font-size:1.05rem; text-align:center; color:#333; background:#fafafa; border-top:1px solid #e6e6e6; border-bottom:1px solid #e6e6e6; padding:8px 0; letter-spacing:0.2px; }}
        .row-colheads td {{
          font-weight:700;
          border-top:1px solid #e8e8e8;
          /* background:#f6f8fb; */  /* removed shading */
        }}

        .row-colheads td:nth-child(1) {{ text-align:left; }}
        .row-colheads td:nth-child(n+2) {{ text-align:right; }}  /* numeric headers (top) */

        /* Category / totals styling */
        .row-cat td {{ background:#fff; border-top:none; }}
        .row-cat td:first-child {{ font-weight:700; letter-spacing:0.2px; }}
        .row-item td:first-child {{ padding-left:22px; color:#444; }}
        .row-total td {{ background:#f8fafc; border-top:1px solid #e5e7eb; font-weight:600; }}
        .row-provision td {{ background:#fff; border-top:1px solid #e5e7eb; font-weight:600; }}
        .row-grand td {{ background:#f3f4f6; border-top:1px solid #e5e7eb; font-weight:700; font-size:1rem; }}
        .row-total td:nth-child(n+2), .row-provision td:nth-child(n+2), .row-grand td:nth-child(n+2) {{ font-weight:500; }}


        /* Add full outer border to Vineyard Profile and Performance tables */
        .tbl,
        .profile-grid {{
          border: 1px solid #e8e8e8;
          border-collapse: separate;
          border-spacing: 0;
          border-radius: 6px;
        }}

        /* Ensure the right edge line shows up cleanly */
        .tbl td:last-child,
        .tbl th:last-child,
        .profile-grid td:last-child,
        .profile-grid th:last-child {{
          border-right: 1px solid #e8e8e8;
        }}


        /* Scroll area for the big table */
        .tbl-wrap-bottom {{ max-height:50vh; overflow:auto; border:1px solid #e8e8e8; border-radius:6px; -webkit-overflow-scrolling:touch; }}

        /* Sticky header — RIGHT-ALIGN numeric headers to mirror the data */
        .tbl.sticky thead th {{ position:sticky; top:0; z-index:5; background:#fff; font-weight:700; box-shadow:0 2px 0 rgba(0,0,0,0.04); white-space:normal; }}
        .tbl.sticky thead th:nth-child(1) {{ text-align:left; }}
        .tbl.sticky thead th:nth-child(n+2) {{ text-align:right; }}
        /* Keep the first header cell above the sticky first column cells */
        .tbl.sticky thead th:first-child {{ left:0; z-index:6; background:#fff; background-clip:padding-box; }}

        /* Sticky first column (tbody only) — use same width as colgroup */
        .tbl-body tbody td:first-child {{ position:sticky; left:0; z-index:2; background:#fff; background-clip:padding-box; box-shadow:2px 0 0 #e8e8e8 inset; width:{FIRST_W}px; }}

        /* Nice vertical centering */
        .tbl-body tr td {{ vertical-align: middle; }}

        /* Print */
        @media print {{
          .tbl-wrap-bottom {{ max-height:none; overflow:visible; border:1px solid #e8e8e8; border-radius:6px; }}
          .tbl.sticky thead th {{ position:static; box-shadow:none; }}
        }}

        .footer {{
          text-align:center;
          font-size:0.85rem;
          color:#6b7280;
          margin-top:14px;
          padding-top:6px;
          border-top:1px solid #e5e7eb;
          font-style:italic;
          letter-spacing:0.3px;
          transition:color 0.3s ease;
        }}

        @media screen {{
          .footer:hover {{ color:#374151; }}
        }}

        @media print {{
          .footer {{
            position:static !important;
            text-align:center !important;
            color:#6b7280 !important;
            border-top:1px solid #d1d5db !important;
            margin-top:10px !important;
          }}
        }}
      </style>


    </head>
    <body>

            <div class="section-title">Vineyard Profile</div>
      <table class="tbl tbl-body vprofile">
        {colgroup_html}
        <tbody>{tbody_html_top}</tbody>
      </table>

      {"" if not len(bottom_df) else f'''
      <div class="print-gap"></div> <!-- prevents table header clipping at page top -->
      <div class="section-title">Costs and Totals</div>
      <div class="tbl-wrap-bottom">
        <table class="tbl tbl-body sticky">
          {colgroup_html}
          <thead>{thead_html_bottom}</thead>
          <tbody>{tbody_html_bottom}</tbody>
        </table>
      </div>
      '''}


      <div class="footer">© <strong>Jerry Analytics</strong> — Data with Interest</div>
    </body>
    </html>
    """
    return html


st.iframe(build_html(table, SC_LABEL), height=740, width="stretch")

with st.expander("Historical comparison — 2024 to 2025", expanded=False):
    st.caption(
        "This additional view compares the same region, cultivar and selected sustainability "
        "scenario across the two benchmark years. It does not replace the detailed selected-year view above."
    )
    available_years = set(
        pd.to_numeric(income_df.get("YEAR", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    cost_years = set(
        pd.to_numeric(costs_df.get("Year", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    if {2024, 2025}.issubset(available_years) and {2024, 2025}.issubset(cost_years):
        result_2024 = build_year_profile_result(2024, scenario_key)
        result_2025 = build_year_profile_result(2025, scenario_key)

        def _change_pct(old_value: float, new_value: float) -> float | None:
            return ((new_value / old_value) - 1.0) * 100.0 if abs(old_value) > 1e-9 else None

        industry_cash_change = _change_pct(
            result_2024["industry_cash"], result_2025["industry_cash"]
        )
        scenario_total_change = _change_pct(
            result_2024["scenario_total"], result_2025["scenario_total"]
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Industry cash costs",
            money(result_2025["industry_cash"]),
            delta=(
                f"{industry_cash_change:+.1f}% vs 2024"
                if industry_cash_change is not None else "No 2024 base"
            ),
        )
        metric_cols[1].metric(
            f"{SC_LABEL} total expenditure",
            money(result_2025["scenario_total"]),
            delta=(
                f"{scenario_total_change:+.1f}% vs 2024"
                if scenario_total_change is not None else "No 2024 base"
            ),
        )
        metric_cols[2].metric(
            "2025 provision for renewal",
            money(result_2025["industry_provision"]),
            delta=money(
                result_2025["industry_provision"] - result_2024["industry_provision"]
            ),
        )

        historical_summary = pd.DataFrame(
            [
                {
                    "Indicator": "Industry income range (R/t)",
                    "2024": f"{money(result_2024['income_low_rt'])} / {money(result_2024['income_high_rt'])}",
                    "2025": f"{money(result_2025['income_low_rt'])} / {money(result_2025['income_high_rt'])}",
                },
                {
                    "Indicator": "Industry total expenditure (R/ha)",
                    "2024": money(result_2024["industry_total"]),
                    "2025": money(result_2025["industry_total"]),
                },
                {
                    "Indicator": f"{SC_LABEL} total expenditure (R/ha)",
                    "2024": money(result_2024["scenario_total"]),
                    "2025": money(result_2025["scenario_total"]),
                },
                {
                    "Indicator": "Industry NFI range (R/ha)",
                    "2024": f"{money(result_2024['industry_net_low'])} / {money(result_2024['industry_net_high'])}",
                    "2025": f"{money(result_2025['industry_net_low'])} / {money(result_2025['industry_net_high'])}",
                },
                {
                    "Indicator": f"{SC_LABEL} NFI range (R/ha)",
                    "2024": f"{money(result_2024['scenario_net_low'])} / {money(result_2024['scenario_net_high'])}",
                    "2025": f"{money(result_2025['scenario_net_low'])} / {money(result_2025['scenario_net_high'])}",
                },
            ]
        )
        st.dataframe(historical_summary, hide_index=True, width="stretch")

        cost_changes = historical_cost_change_frame()
        if not cost_changes.empty:
            largest = cost_changes.iloc[0]
            movement_text = (
                f"For {region}, total industry cash costs changed by "
                f"{industry_cash_change:+.1f}% from 2024 to 2025. "
                f"The largest absolute line-item movement was {largest['Item']}: "
                f"{money(largest['2024'])} to {money(largest['2025'])} "
                f"({money(largest['Change (R/ha)'])} per hectare)."
                if industry_cash_change is not None else
                "The 2024 and 2025 line-item values are shown below."
            )
            if industry_cash_change is not None and abs(industry_cash_change) >= 20:
                movement_text += (
                    " This is a large year-on-year movement and should be interpreted "
                    "with the source-year context rather than as a normal long-run trend."
                )
            st.info(movement_text)

            st.markdown("#### Detailed industry cost changes")
            st.dataframe(
                cost_changes,
                hide_index=True,
                width="stretch",
                column_config={
                    "2024": st.column_config.NumberColumn(format="R %.2f"),
                    "2025": st.column_config.NumberColumn(format="R %.2f"),
                    "Change (R/ha)": st.column_config.NumberColumn(format="R %.2f"),
                    "Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )
    else:
        st.info("Both 2024 and 2025 income and cost records are required for this comparison.")


# ---------- Helper: build a safe filename ----------
def make_filename(prefix="Budget_Comparison", ext="csv"):
    safe = lambda v: (str(v).strip().replace(" ", "") if v is not None else "NA")
    parts = [
        prefix,
        SC_LABEL,  # e.g. "Organic"
        str(year) if year is not None else "NA",
        safe(wine_class), safe(grape_variety), safe(region),
        datetime.now().strftime("%Y%m%d_%H%M")
    ]
    return "_".join(parts) + f".{ext}"


# ---------- Net summary + charts ----------

def net_summary_for_charts(sc_label: str):
    """Return a tidy list of (label, value) for the bar chart."""
    items = []
    items.append(("Farmer", net_current if net_current is not None else 0.0))
    if net_low is not None:
        items.append(("Industry Low", net_low))
    if net_high is not None:
        items.append(("Industry High", net_high))
    # Scenario low/high can be a pair; show both if available
    if SC.get("net_low") is not None:
        items.append((f"{sc_label} Low", SC["net_low"]))
    if SC.get("net_high") is not None:
        items.append((f"{sc_label} High", SC["net_high"]))
    return items


def plot_net_summary_bar(sc_label: str, show_yield: bool = True):
    data = net_summary_for_charts(sc_label)
    labels = [k for k, _ in data]
    values = [v for _, v in data]

    # Keep the full labels for the calculations, but wrap the displayed
    # X-axis labels so longer scenario names remain readable.
    def axis_label(label: str) -> str:
        if label == "Farmer":
            return label
        if label.startswith("Industry "):
            return label.replace(" ", "\n", 1)

        scenario_name, band = label.rsplit(" ", 1)
        if " + " in scenario_name:
            scenario_name = scenario_name.replace(" + ", " +\n")
        return f"{scenario_name}\n{band}"

    display_labels = [axis_label(label) for label in labels]

    # A balanced figure size keeps the comparison readable without allowing
    # the bars to dominate the page.
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)

    # --- Bars: Net Farm Income (left Y) ---

    # Find the farmer bar (fallback to 0 if not present)
    try:
        farmer_idx = labels.index("Farmer")
    except ValueError:
        farmer_idx = 0

    farmer_nfi = values[farmer_idx]

    benchmark_bar_color = "#B0B8C4"
    colors = [benchmark_bar_color] * len(values)
    # Make farmer conditional: green if ≥ 0, red if < 0
    colors[farmer_idx] = "#1CBA59" if farmer_nfi >= 0 else "#D64545"

    # Plot
    x = range(len(labels))
    ax.bar(
        x,
        values,
        width=0.56,
        color=colors,
        alpha=0.90,
        label="Net Farm Income (bars)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        display_labels,
        rotation=0,
        ha="center",
        fontsize=8.5,
        linespacing=1.15,
    )
    ax.tick_params(axis="x", pad=7)
    ax.margins(x=0.08)

    ax.set_ylabel("Net Farm Income (R/ha)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"R {v:,.0f}"))
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=1.2)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(axis="y", colors="#6b7280")

    # --- Optional: Yield markers only (right Y), NO connecting line, NO point labels ---
    if show_yield:
        yield_map = {
            "Farmer": yield_current,
            "Industry Low": yield_low,
            "Industry High": yield_high,
            f"{sc_label} Low": SC.get("yield_low"),
            f"{sc_label} High": SC.get("yield_high"),
        }
        yield_vals = [yield_map.get(lbl, None) for lbl in labels]

        ax2 = ax.twinx()
        ax2.set_ylabel("Yield (t/ha)")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:,.1f}"))
        ax2.grid(False)
        ax2.tick_params(axis="y", colors="#6b7280")

        for i, (yval, bar_color) in enumerate(zip(yield_vals, colors)):
            if yval is None:
                continue
            # marker only (no line), match bar color
            ax2.scatter(i, yval, s=58, marker="o",
                        facecolor=bar_color, edgecolor="#1f2937", linewidths=0.8, zorder=5)

    # --- Compact legend explaining the two encodings ---
    legend_handles = [
        Patch(facecolor=benchmark_bar_color, edgecolor="#4b5563",
              label="Net Farm Income (bars)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=7,
               markerfacecolor="#1f2937", markeredgecolor="#1f2937",
               label="Yield (markers)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),  # clear space between legend and plot
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        fontsize=8.5,
        borderpad=0.4,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.76, bottom=0.24)  # legend and wrapped labels
    return fig


def fig_to_png_bytes(fig) -> bytes:
    """Return PNG bytes from a Matplotlib figure without touching disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ---------- Yield and Net Farm Income plot (with scenario overlay) ----------
st.markdown(
    """
    <style>
    /* Tighten gap above chart heading */
    div[data-testid="stMarkdownContainer"] h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.4rem !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("### Regional Competitiveness Map — Yield and Net Farm Income")
st.caption(
    "This chart shows how the farmer’s yield and profitability compare with "
    "regional industry ranges and optional scenario adjustments. It highlights "
    "both competitive position and financial sustainability."
)


def build_range_plot():
    # Industry box corners
    if (yield_low is not None and yield_high is not None and
            net_low is not None and net_high is not None):
        y0, y1 = (min(yield_low, yield_high), max(yield_low, yield_high))
        n0, n1 = (min(net_low, net_high), max(net_low, net_high))
    else:
        y0 = y1 = yield_current
        n0 = n1 = net_current

    # Scenario box corners (may be None if data incomplete)
    sy0 = sy1 = sn0 = sn1 = None
    if (SC["yield_low"] is not None and SC["yield_high"] is not None and
            SC["net_low"] is not None and SC["net_high"] is not None):
        sy0, sy1 = (min(SC["yield_low"], SC["yield_high"]), max(SC["yield_low"], SC["yield_high"]))
        sn0, sn1 = (min(SC["net_low"], SC["net_high"]), max(SC["net_low"], SC["net_high"]))

    # Signals
    in_range = (y0 <= yield_current <= y1) and (n0 <= net_current <= n1)
    sign_color = PALETTE["green_pos"] if (net_current is not None and net_current >= 0) else PALETTE["red_neg"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)

    # Industry range: keep the boundaries accurate, but use a lighter fill so
    # the range supports the message without dominating the chart.
    ax.fill_between([y0, y1], n0, n1, step="pre", alpha=0.28,
                    color=PALETTE["industry_fill"], zorder=1)
    ax.plot([y0, y0, y1, y1, y0], [n0, n1, n1, n0, n0],
            color=PALETTE["industry_edge"], linewidth=1.1, zorder=2,
            label="Industry range")

    # Scenario range (grey dashed outline)
    if sy0 is not None and sy1 is not None and sn0 is not None and sn1 is not None:
        ax.plot([sy0, sy0, sy1, sy1, sy0], [sn0, sn1, sn1, sn0, sn0],
                color=PALETTE["scenario_edge"], linewidth=1.3, linestyle="--",
                alpha=0.9, zorder=2.5, label=f"{SC_LABEL} range")

    # Break-even line (R/ha = 0)
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=1.2, zorder=0)
    # ax.text(x=y0, y=0, s="Break-even", va="bottom", ha="left",
    #      fontsize=8, color=PALETTE["neutral_dark"], alpha=0.9)

    # Farmer point: filled if in range, hollow if out of range
    if in_range:
        ax.scatter(yield_current, net_current, s=54,
                   facecolor=sign_color, edgecolor="white", linewidths=0.9, zorder=3)
    else:
        ax.scatter(yield_current, net_current, s=62,
                   facecolor="none", edgecolor=sign_color, linewidths=1.6, zorder=3)

    ax.annotate("Farmer", (yield_current, net_current), xytext=(5, 7),
                textcoords="offset points", fontsize=8.5,
                color=sign_color, weight="bold")

    # Dynamic axes
    x_candidates = [v for v in [y0, y1, sy0, sy1, yield_current] if v is not None]
    y_candidates = [v for v in [n0, n1, sn0, sn1, net_current] if v is not None]
    x_min = min(x_candidates) if x_candidates else 0
    x_max = max(x_candidates) if x_candidates else 1
    y_min = min(y_candidates) if y_candidates else 0
    y_max = max(y_candidates) if y_candidates else 1
    xspan = max(0.1, (x_max - x_min) or 1.0)
    yspan = max(0.1, (y_max - y_min) or 1.0)
    ax.set_xlim(x_min - xspan * 0.18, x_max + xspan * 0.18)
    ax.set_ylim(y_min - yspan * 0.20, y_max + yspan * 0.20)

    ax.set_xlabel("Yield (t/ha)", fontsize=9)
    ax.set_ylabel("Net Farm Income (R/ha)", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"R {v:,.0f}"))
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(alpha=0.12)

    # The status chips already explain profit/loss and in/out of range. The
    # legend therefore shows only the three elements currently on the map.
    farmer_face = sign_color if in_range else "none"
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=8,
               markerfacecolor=PALETTE["industry_fill"], markeredgecolor=PALETTE["industry_edge"],
               label="Industry range"),
        Line2D([0], [0], color=PALETTE["scenario_edge"], linestyle="--", linewidth=1.3,
               label=f"{SC_LABEL} range"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6.5,
               markerfacecolor=farmer_face, markeredgecolor=sign_color,
               markeredgewidth=1.4, label="Farmer"),
    ]
    ax.legend(
        handles=legend_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        fontsize=8,
        borderpad=0.45,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    return fig


# Build both figures
fig = build_range_plot()
fig_bar = plot_net_summary_bar(SC_LABEL)

# Compute status text
if net_current is not None:
    profit_loss = "🟢 Profit" if net_current >= 0 else "🔴 Loss"
else:
    profit_loss = "?"

if (yield_low is not None and yield_high is not None and
        net_low is not None and net_high is not None):
    in_range = (yield_low <= yield_current <= yield_high) and (net_low <= net_current <= net_high)
    range_text = "• In range" if in_range else "• Out of range"
else:
    range_text = ""


# st.caption(f"{profit_loss} {range_text}")

# --- Helpers for chips + range math (place once, near your other helpers) ---
def _chip(txt, tone="neutral"):
    colors = {
        "good": ("#0f5132", "#d1e7dd", "#badbcc"),
        "bad": ("#842029", "#f8d7da", "#f5c2c7"),
        "warn": ("#664d03", "#fff3cd", "#ffe69c"),
        "info": ("#055160", "#cff4fc", "#b6effb"),
        "neutral": ("#374151", "#f3f4f6", "#e5e7eb"),
    }
    fg, bg, bd = colors.get(tone, colors["neutral"])
    return (
        f'<span style="display:inline-block;padding:4px 10px;'
        f'border:1px solid {bd};border-radius:999px;'
        f'background:{bg};color:{fg};font-size:12px;'
        f'line-height:1;">{txt}</span>'
    )


def _dist_to_band(val, lo, hi):
    """Return (status, delta, where) where delta>0 means outside the band."""
    if lo is None or hi is None or val is None:
        return ("unknown", 0.0, "")
    lo, hi = min(lo, hi), max(lo, hi)
    if lo <= val <= hi:
        # distance to nearest edge (inside)
        d = min(val - lo, hi - val)
        return ("inside", d, "inside")
    # outside
    if val < lo:
        return ("below", lo - val, "below")
    else:
        return ("above", val - hi, "above")


# --- Snapshot under the chart ---
# industry band
y0, y1 = (min(yield_low, yield_high), max(yield_low, yield_high)) if (
            yield_low is not None and yield_high is not None) else (yield_current, yield_current)
n0, n1 = (min(net_low, net_high), max(net_low, net_high)) if (net_low is not None and net_high is not None) else (
    net_current, net_current)

profit = (net_current is not None and net_current >= 0)
profit_txt = f"{'🟢 Profit' if profit else '🔴 Loss'} · R {0 if net_current is None else round(net_current):,}/ha"
profit_tone = "good" if profit else "bad"

status_y, dy, where_y = _dist_to_band(yield_current, y0, y1)
status_n, dn, where_n = _dist_to_band(net_current, n0, n1)

range_txt = "In industry range" if (status_y == "inside" and status_n == "inside") else "Out of industry range"
range_tone = "good" if (status_y == "inside" and status_n == "inside") else "warn"


# wording for distance
def fmt_delta(label, status, d, unit):
    if status == "inside":
        return f"{label}: inside band"
    elif status == "below":
        return f"{label}: {unit} {d:,.2f} below"
    elif status == "above":
        return f"{label}: {unit} {d:,.2f} above"
    else:
        return f"{label}: n/a"


chip1 = _chip(profit_txt, profit_tone)
chip2 = _chip(range_txt, range_tone)
chip3 = _chip(fmt_delta("Yield", status_y, dy, "t/ha"), "info")
chip4 = _chip(fmt_delta("NFI", status_n, dn, "R/ha"), "info")

st.markdown(
    f'<div style="margin:6px 0 2px 0;display:flex;gap:8px;flex-wrap:wrap;">'
    f'{chip1}{chip2}{chip3}{chip4}'
    f'</div>',
    unsafe_allow_html=True
)

# Show on screen
st.markdown("<div style='margin-bottom:-8px;'></div>", unsafe_allow_html=True)

# Centre the map at about 75% of the page width so it feels lighter than the
# detailed comparison chart below it.
map_left, map_centre, map_right = st.columns([1, 6, 1])
with map_centre:
    st.pyplot(fig, width="stretch")
st.markdown("### Regional Competitiveness Chart — Yield and Net Farm Income")
st.caption(
    "This snapshot compares the farmer’s profitability against industry and scenario benchmarks."
)
# Display the bar chart at 80% of the page width to maintain comfortable
# spacing around the narrower bars.
bar_left, bar_centre, bar_right = st.columns([1, 8, 1])
with bar_centre:
    st.pyplot(fig_bar, width="stretch")

# Divider

st.markdown("""
<hr style='border: 1px solid #e5e7eb; margin: 16px 0 4px 0;' />
""", unsafe_allow_html=True)  # ⬅️ reduced bottom margin from 16px → 12px

# =========================
# Export (CSV / Excel / PDF)
# =========================
st.markdown("### Results Export")

st.markdown(
    "<p style='margin-top:-6px; font-style:italic; color:#4b5563;'>"
    "Download the current analysis in CSV, Excel, or PDF format below."
    "</p>",
    unsafe_allow_html=True,
)

# --- CSV ---
csv_buf = StringIO()
table.to_csv(csv_buf, index=False)
st.download_button(
    label="CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=make_filename(ext="csv"),
    mime="text/csv",
)


# --- Excel (prefer XlsxWriter; fallback to openpyxl) ---
def to_excel_bytes(df, fig_range=None, fig_bar=None):
    """
    df: table DataFrame
    fig_range: Matplotlib figure for the 'Yield vs NFI' range plot (optional)
    fig_bar:   Matplotlib figure for the 'Net summary' bar (optional)
    """
    out = BytesIO()
    try:
        import xlsxwriter
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(out, engine=engine) as writer:
        # Sheet 1 — the main table
        df.to_excel(writer, index=False, sheet_name="Comparison")

        if engine == "xlsxwriter":
            wb = writer.book
            ws = writer.sheets["Comparison"]
            hdr = wb.add_format({"bold": True})
            for c, name in enumerate(df.columns):
                ws.write(0, c, name, hdr)
            ws.set_column(0, 0, 28)
            ws.set_column(1, len(df.columns) - 1, 18)

            # Sheet 2 — Charts
            ws2 = wb.add_worksheet("Charts")
            row = 0
            ws2.write(row, 0, "Wine Class:");
            ws2.write(row, 1, str(wine_class));
            row += 1
            ws2.write(row, 0, "Grape Variety:");
            ws2.write(row, 1, str(grape_variety));
            row += 1
            ws2.write(row, 0, "Region:");
            ws2.write(row, 1, str(region));
            row += 1
            ws2.write(row, 0, "Scenario:");
            ws2.write(row, 1, str(SC_LABEL));
            row += 2

            if fig_bar is not None:
                png_bar = fig_to_png_bytes(fig_bar)
                ws2.insert_image(row, 0, "net_summary.png",
                                 {"image_data": BytesIO(png_bar)})
                row += 20
            if fig_range is not None:
                png_rng = fig_to_png_bytes(fig_range)
                ws2.insert_image(row, 0, "yield_vs_nfi.png",
                                 {"image_data": BytesIO(png_rng)})
        else:
            # openpyxl fallback (no images)
            try:
                wb = writer.book
                ws2 = wb.create_sheet("Charts")
                info = {
                    "Wine Class": wine_class,
                    "Grape Variety": grape_variety,
                    "Region": region,
                    "Scenario": SC_LABEL,
                }
                r = 1
                for k, v in info.items():
                    ws2.cell(row=r, column=1, value=k)
                    ws2.cell(row=r, column=2, value=str(v))
                    r += 1
                r += 1
                rows = net_summary_for_charts(SC_LABEL)
                ws2.cell(row=r, column=1, value="Label")
                ws2.cell(row=r, column=2, value="Net Farm Income (R/ha)")
                r += 1
                for k, v in rows:
                    ws2.cell(row=r, column=1, value=k)
                    ws2.cell(row=r, column=2, value=float(v))
                    r += 1
            except Exception:
                pass

    return out.getvalue()


# Build Excel bytes and show the button (outside the function)
xlsx_bytes = to_excel_bytes(table, fig_range=fig, fig_bar=fig_bar)
st.download_button(
    label="Excel (.xlsx)",
    data=xlsx_bytes,
    file_name=make_filename(ext="xlsx"),
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dl_xlsx"
)


# --- Robust PDF export (wkhtmltopdf if available; else Printable HTML) ---
def _wkhtmltopdf_config():
    """Return a pdfkit configuration if wkhtmltopdf is available, else None."""
    try:
        import pdfkit
    except Exception:
        return None

    exe = shutil.which("wkhtmltopdf")
    if exe is None:
        # common Windows locations (optional)
        candidates = [
            r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
            r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                exe = c
                break
    if exe is None:
        return None

    return pdfkit, pdfkit.configuration(wkhtmltopdf=exe)


# 1) Convert figures to base64 PNGs
png_range_b64 = base64.b64encode(fig_to_png_bytes(fig)).decode("ascii")
png_bar_b64 = base64.b64encode(fig_to_png_bytes(fig_bar)).decode("ascii")

# 2) Compose HTML for a clean two-chart report + table
html_str = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Regional Competitiveness — Yield & Net Farm Income</title>
<style>
  @media print {{
    @page {{ size: A4; margin: 16mm; }}
  }}
  body {{
    font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
    color: #111827;
    margin: 0;
    padding: 8px 20px 6px;
  }}
  h2 {{ margin: 0 0 10px 0; font-weight: 700; }}
  h3 {{ margin: 18px 0 8px 0; font-weight: 600; }}
  .card {{
    border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px;
    margin: 8px 0 12px 0;
  }}
  .chart-img {{
    width: 100%; height: auto; display: block;
  }}
  .break {{ page-break-after: always; }}

/* ===== Print rules (single, consolidated) ===== */
@media print {{
  /* Page setup */
  @page {{ size: A4; margin: 20mm 12mm 18mm 12mm; }} /* top right bottom left */

  /* Let long tables flow; remove scroll boxes & sticky bits for print */
  .tbl-wrap-bottom {{
    max-height: none !important;
    overflow: visible !important;
    border-top: 1px solid #e5e7eb; /* keeps a top rule when a table starts on a new page */
  }}
  .tbl.sticky thead th,
  .tbl-head thead th,
  .tbl-body td:first-child {{
    position: static !important;
    left: auto !important;
    box-shadow: none !important;
  }}

  /* Allow long tables to flow, repeat headers, and keep each row intact. */
  table {{
    break-inside: auto !important;
    page-break-inside: auto !important;
  }}
  thead {{
    display: table-header-group !important;
    break-inside: avoid !important;
    page-break-inside: avoid !important;
  }}
  tbody {{
    display: table-row-group !important;
    break-inside: auto !important;
    page-break-inside: auto !important;
  }}
  tbody tr {{
    break-inside: avoid-page !important;
    page-break-inside: avoid !important;
    page-break-after: auto !important;
  }}
  tbody td {{
    break-inside: avoid !important;
    page-break-inside: avoid !important;
  }}
  tfoot {{ display: table-footer-group !important; }}

  /* Keep section headings off the very bottom of a page */
  .section-title {{ page-break-after: avoid !important; }}

  /* Keep repeated headers compact and clear of the page edge. */
  .tbl thead tr:first-child th {{
    padding-top: 7px !important;
    padding-bottom: 7px !important;
  }}
  .print-gap {{ height: 4mm; }}
}}



/* --- Footer: screen + print unified --- */

/* Base footer style */
.footer {{
  text-align: center;
  font-size: 0.85rem;
  color: #6b7280;                  /* soft gray */
  margin-top: 14px;                /* space above footer */
  padding-top: 6px;                /* small top padding */
  border-top: 1px solid #e5e7eb;   /* gentle separator */
  font-style: italic;
  letter-spacing: 0.3px;
  transition: color 0.3s ease;
}}

/* Hover effect (screen only) */
@media screen {{
  .footer:hover {{
    color: #374151;                /* darker gray on hover */
  }}
}}

/* Print adjustments */
@media print {{
  .footer {{
    position: static !important;
    text-align: center !important;
    color: #6b7280 !important;
    border-top: 1px solid #d1d5db !important;
    margin-top: 10px !important;
    padding-top: 4px !important;
    font-style: italic;
  }}
}}

/* Optional: tighten spacing before next section titles */
.section-title {{
  margin-top: 6px !important;
}}



</style>

</head>

<body>

   <div class="card">
    <h3>Regional Competitiveness <strong>Map</strong> — Yield and Net Farm Income</h3>
    <img class="chart-img" alt="Regional Competitiveness Map — Yield and Net Farm Income"
         src="data:image/png;base64,{png_range_b64}" />
  </div>

  <div class="card">
    <h3>Regional Competitiveness <strong>Chart</strong> — Yield and Net Farm Income</h3>
    <img class="chart-img" alt="Regional Competitiveness Chart — Yield and Net Farm Income"
         src="data:image/png;base64,{png_bar_b64}" />
  </div>

  <div class="break"></div>

  <!-- Your styled comparison table -->
  {build_html(table, SC_LABEL)}

</body>
</html>
"""

# 3) Try PDF first; if wkhtmltopdf missing, offer Printable HTML
_cfg = _wkhtmltopdf_config()
if _cfg is not None:
    pdfkit, cfg = _cfg
    try:
        pdf_bytes = pdfkit.from_string(
            html_str, False, configuration=cfg,
            options={"quiet": ""}  # suppress wkhtmltopdf chatter
        )
        st.download_button(
            "PDF",
            data=pdf_bytes,
            file_name=make_filename(ext="pdf"),
            mime="application/pdf",
        )
    except Exception:
        st.download_button(
            "Printable HTML",
            data=html_str,
            file_name=make_filename(ext="html"),
            mime="text/html",
        )
else:
    st.download_button(
        "Printable HTML",
        data=html_str,
        file_name=make_filename(ext="html"),
        mime="text/html",
    )

    st.caption("💡 Tip: Open the file → Ctrl+P → Save as PDF (uncheck 'Headers and footers').")

# Release Matplotlib resources after Streamlit and the exports have consumed them.
plt.close(fig)
plt.close(fig_bar)

