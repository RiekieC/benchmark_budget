# Phase 2 — Outlook to 2030
# Replacement file: views/phase2_forecast.py
#
# Purpose:
# - Page 2A gives a farmer-friendly 2025/current vs 2030 forecast summary.
# - Page 2B keeps the detailed year-by-year view.
#
# This file is intentionally self-contained so it can be dropped into the
# multipage Streamlit structure without needing changes to Phase 1.

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


# -----------------------
# Basic settings / helpers
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"

PALETTE = {
    "green_pos": "#1CBA59",
    "red_neg": "#D64545",
    "neutral": "#9AA3B2",
    "neutral_dark": "#6B7280",
    "industry_fill": "#E8EEF6",
    "industry_edge": "#8A93A6",
    "scenario_edge": "#9EA3AA",
    "zero_line": "#C7CBD1",
}

CATEGORY_ORDER = [
    "Direkte koste",
    "Arbeid",
    "Meganisasie",
    "Vaste verbeteringe",
    "Algemene uitgawes",
    "Kontantuitgawes",
    "Totale kontantuitgawes",
]

FORBIDDEN_LABELS = {
    "",
    "nan",
    "none",
    "subtotal",
    "sub-total",
    "total",
    "totale",
    "grand total",
}


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    return _clean_text(value).lower().replace("_", " ").replace("-", " ").strip()


def _fmt_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"R {float(value):,.0f}"
    except Exception:
        return str(value)


def _fmt_num(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def _fmt_range(low: Any, high: Any, money: bool = False, decimals: int = 2) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "—"
    if money:
        return f"{_fmt_money(low)} / {_fmt_money(high)}"
    return f"{_fmt_num(low, decimals)} / {_fmt_num(high, decimals)}"


def _pct_change(value: float, pct: float, periods: int) -> float:
    return float(value) * ((1.0 + float(pct) / 100.0) ** int(periods))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        found = lookup.get(_norm(cand))
        if found is not None:
            return found
    return None


def _read_csv(name: str) -> pd.DataFrame:
    # Helpful fallback while testing suggested/new data files.
    # For income, prefer a newer file that includes
    # `average_annual_growth_percent` when it is present in the data folder.
    candidates: list[Path] = []
    if name == "income.csv":
        for folder in [DATA_DIR, PROJECT_ROOT, Path.cwd(), Path.cwd() / "data"]:
            if folder.exists():
                candidates.extend(sorted(folder.glob("income*growth*.csv")))
                candidates.extend(sorted(folder.glob("income_2024_2025*.csv")))

    candidates.extend([
        DATA_DIR / name,
        PROJECT_ROOT / name,
        Path.cwd() / name,
        Path.cwd() / "data" / name,
    ])

    if name == "yield.csv":
        for folder in [DATA_DIR, PROJECT_ROOT, Path.cwd(), Path.cwd() / "data"]:
            if folder.exists():
                candidates.extend(sorted(folder.glob("yield*.csv")))

    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            try:
                df = pd.read_csv(p)
            except UnicodeDecodeError:
                df = pd.read_csv(p, encoding="latin1")
            df.columns = [str(c).strip() for c in df.columns]
            return df
    st.error(
        f"Could not find `{name}`. Place it either in the project root or in a `data` folder."
    )
    return pd.DataFrame()


def _is_forbidden(label: Any) -> bool:
    text = _norm(label)
    return text in FORBIDDEN_LABELS or text.startswith("unnamed")


def _choose_default(options: list[str], preferred: str | None = None) -> int:
    if not options:
        return 0
    if preferred:
        for i, opt in enumerate(options):
            if _norm(opt) == _norm(preferred):
                return i
    return 0


# -----------------------
# Data normalisation
# -----------------------

@dataclass
class DataColumns:
    wine_class: str | None = None
    grape_variety: str | None = None
    region: str | None = None
    year: str | None = None
    band: str | None = None
    income_rt: str | None = None
    income_growth: str | None = None
    yield_tha: str | None = None
    category: str | None = None
    item: str | None = None
    avg_cost: str | None = None


def income_cols(df: pd.DataFrame) -> DataColumns:
    return DataColumns(
        wine_class=_find_col(df, ["Wine Class", "Wine_Class", "Class"]),
        grape_variety=_find_col(df, ["Grape Variety", "Grape_Variety", "Variety", "Cultivar"]),
        region=_find_col(df, ["Region", "Area"]),
        year=_find_col(df, ["YEAR", "Year"]),
        band=_find_col(df, ["Band", "Level", "Type", "Range"]),
        income_rt=_find_col(df, ["Income_R_per_t", "Income R/t", "Income (R/t)", "R/t", "Income"]),
        income_growth=_find_col(df, [
            "average_annual_growth_percent",
            "Average Annual Growth Percent",
            "Average annual growth %",
            "Income Growth %",
            "Growth %",
        ]),
    )


def yield_cols(df: pd.DataFrame) -> DataColumns:
    return DataColumns(
        wine_class=_find_col(df, ["Wine Class", "Wine_Class", "Class"]),
        grape_variety=_find_col(df, ["Grape Variety", "Grape_Variety", "Variety", "Cultivar"]),
        region=_find_col(df, ["Region", "Area"]),
        band=_find_col(df, ["Band", "Level", "Type", "Range"]),
        yield_tha=_find_col(df, [
            "Yield_t_per_ha",
            "Yield_R_per_ha",  # source file label; values are converted from kg/ha to t/ha when needed
            "Yield kg/ha",
            "Yield_kg_per_ha",
            "Yield t/ha",
            "Yield (t/ha)",
            "t/ha",
            "Yield",
        ]),
    )


def cost_cols(df: pd.DataFrame) -> DataColumns:
    return DataColumns(
        region=_find_col(df, ["Region", "Area"]),
        category=_find_col(df, ["Category", "Cost Category", "Group"]),
        item=_find_col(df, ["Item", "Description", "Cost Item"]),
        avg_cost=_find_col(df, ["Avg_Cost", "Avg Cost", "Average Cost", "Avg", "Cost", "Value"]),
    )


def lookup_band(
    df: pd.DataFrame,
    cols: DataColumns,
    wine_class: str,
    grape_variety: str,
    region: str,
    band: str,
    value_col: str | None,
    year: int | None = None,
) -> float | None:
    """Look up Low/High benchmark values with transparent fallbacks.

    Order used:
    1. Exact cultivar + exact region.
    2. Other White / Other Red + exact region.
    3. Any cultivar in the same wine class + exact region, averaged by band.
    4. Exact cultivar across all regions, averaged by band.
    5. Other White / Other Red across all regions, averaged by band.
    6. Any cultivar in the same wine class across all regions, averaged by band.

    This is especially important for yield.csv, where a specific cultivar/region
    combination may not exist even though a usable broader benchmark does exist.
    """
    if df.empty or value_col is None or value_col not in df.columns:
        return None

    def _base() -> pd.DataFrame:
        sub = df.copy()
        if cols.wine_class is not None and cols.wine_class in sub.columns:
            sub = sub[sub[cols.wine_class].astype(str).str.strip().str.lower() == str(wine_class).strip().lower()]
        if year is not None and cols.year is not None and cols.year in sub.columns:
            year_num = pd.to_numeric(sub[cols.year], errors="coerce")
            sub = sub[year_num == int(year)]
        if cols.band is not None and cols.band in sub.columns:
            sub = sub[sub[cols.band].astype(str).str.strip().str.lower() == band.lower()]
        return sub

    def _value(sub: pd.DataFrame) -> float | None:
        if sub.empty:
            return None
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else None

    def _filter_grape(sub: pd.DataFrame, grape_name: str) -> pd.DataFrame:
        if cols.grape_variety is not None and cols.grape_variety in sub.columns:
            return sub[sub[cols.grape_variety].astype(str).str.strip().str.lower() == str(grape_name).strip().lower()]
        return sub

    def _filter_region(sub: pd.DataFrame, region_name: str) -> pd.DataFrame:
        if cols.region is not None and cols.region in sub.columns:
            return sub[sub[cols.region].astype(str).str.strip().str.lower() == str(region_name).strip().lower()]
        return sub

    base = _base()
    other_grape = "Other White" if _norm(wine_class) == "white" else "Other Red"

    # 1. Exact cultivar + exact region
    val = _value(_filter_region(_filter_grape(base, grape_variety), region))
    if val is not None:
        return val

    # 2. Broader Other White / Other Red + exact region
    if _norm(grape_variety) != _norm(other_grape):
        val = _value(_filter_region(_filter_grape(base, other_grape), region))
        if val is not None:
            return val

    # 3. Same wine class + exact region, average by band
    val = _value(_filter_region(base, region))
    if val is not None:
        return val

    # 4. Exact cultivar across all regions
    val = _value(_filter_grape(base, grape_variety))
    if val is not None:
        return val

    # 5. Other White / Other Red across all regions
    if _norm(grape_variety) != _norm(other_grape):
        val = _value(_filter_grape(base, other_grape))
        if val is not None:
            return val

    # 6. Same wine class across all regions, average by band
    val = _value(base)
    if val is not None:
        return val

    # Wide-format fallback, where Low/High may be column names rather than rows.
    wide_col = _find_col(df, [band])
    if wide_col:
        vals = pd.to_numeric(df[wide_col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else None

    return None


def lookup_income_growth_percent(
    df: pd.DataFrame,
    cols: DataColumns,
    wine_class: str,
    grape_variety: str,
    region: str,
    year: int | None,
) -> float | None:
    """Return the suggested annual income-growth coefficient from income data.

    The new income file contains `average_annual_growth_percent`. This helper
    uses the same matching logic as the benchmark: exact cultivar/region first,
    then broader fallbacks. It is only a default assumption; the user can still
    edit it in the sidebar.
    """
    growth_col = cols.income_growth
    if df.empty or growth_col is None or growth_col not in df.columns:
        return None

    # Prefer the selected base year. If no value is stored there, use any
    # available growth value for the same profile.
    candidates: list[pd.DataFrame] = []

    def _prep(base_year_only: bool) -> pd.DataFrame:
        sub = df.copy()
        if cols.wine_class and cols.wine_class in sub.columns:
            sub = sub[sub[cols.wine_class].astype(str).str.strip().str.lower() == str(wine_class).strip().lower()]
        if base_year_only and year is not None and cols.year and cols.year in sub.columns:
            sub = sub[pd.to_numeric(sub[cols.year], errors="coerce") == int(year)]
        return sub

    for year_only in [True, False]:
        base = _prep(year_only)
        if base.empty:
            continue

        other_grape = "Other White" if _norm(wine_class) == "white" else "Other Red"

        def _fg(sub: pd.DataFrame, grape: str) -> pd.DataFrame:
            if cols.grape_variety and cols.grape_variety in sub.columns:
                return sub[sub[cols.grape_variety].astype(str).str.strip().str.lower() == str(grape).strip().lower()]
            return sub

        def _fr(sub: pd.DataFrame, reg: str) -> pd.DataFrame:
            if cols.region and cols.region in sub.columns:
                return sub[sub[cols.region].astype(str).str.strip().str.lower() == str(reg).strip().lower()]
            return sub

        candidates.extend([
            _fr(_fg(base, grape_variety), region),
            _fr(_fg(base, other_grape), region),
            _fr(base, region),
            _fg(base, grape_variety),
            _fg(base, other_grape),
            base,
        ])

    for sub in candidates:
        vals = pd.to_numeric(sub[growth_col], errors="coerce").dropna()
        if not vals.empty:
            return float(vals.mean())

    return None


def prepare_costs(costs_df: pd.DataFrame, cols: DataColumns, region: str) -> pd.DataFrame:
    if costs_df.empty or not all([cols.region, cols.category, cols.item, cols.avg_cost]):
        return pd.DataFrame(columns=["Category", "Item", "Avg"])

    sub = costs_df[
        costs_df[cols.region].astype(str).str.strip().str.lower() == region.strip().lower()
    ].copy()

    sub = sub.rename(
        columns={
            cols.category: "Category",
            cols.item: "Item",
            cols.avg_cost: "Avg",
        }
    )[["Category", "Item", "Avg"]]

    sub["Category"] = sub["Category"].apply(_clean_text)
    sub["Item"] = sub["Item"].apply(_clean_text)
    sub["Avg"] = pd.to_numeric(sub["Avg"], errors="coerce").fillna(0.0)

    sub = sub[~sub["Category"].apply(_is_forbidden)]
    sub = sub[~sub["Item"].apply(_is_forbidden)]

    cat_order_map = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    sub["__cat_order"] = sub["Category"].map(lambda c: cat_order_map.get(c, 999))
    sub["__item_order"] = sub.groupby("Category").cumcount()
    sub = sub.sort_values(["__cat_order", "__item_order"], kind="stable").drop(
        columns=["__cat_order", "__item_order"]
    )
    return sub.reset_index(drop=True)


# -----------------------
# Scenario assumptions
# -----------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "Scenario 1 — Organic": {
        "label": "Organic",
        "diff_multiplier": 1.41,
        "yield_high_pct": 0.0,
        "income_abs": 0.0,
        "cost_rules": [
            ("item", "Saad, organiese bemesting en materiaal", "set", 13349),
            ("item", "Gewasbeskerming (swam- en insekbeheer)", "set", 1053),
            ("category", "Meganisasie", "abs", 110),
            ("item", "Administrasie", "abs", 555),
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
            ("item", "Permanente arbeid", "pct", 7.9),
            ("item", "Seisoensarbeid en kontrakwerk", "pct", 7.9),
            ("item", "Administrasie", "abs", 1666),
        ],
    },
    "Scenario 3 — Organic + Fairtrade": {
        "label": "Organic + Fairtrade",
        "diff_multiplier": 1.41,
        "yield_high_pct": 0.0,
        "income_abs": 30.0,
        "cost_rules": [
            ("item", "Saad, organiese bemesting en materiaal", "set", 13349),
            ("item", "Gewasbeskerming (swam- en insekbeheer)", "set", 1053),
            ("category", "Meganisasie", "abs", 110),
            ("item", "Administrasie", "abs", 2221),
            ("item", "Permanente arbeid", "pct", 7.9),
            ("item", "Seisoensarbeid en kontrakwerk", "pct", 7.9),
            ("item", "Kunsmis, blaar- en grondontledings", "set", 0),
            ("item", "Onkruiddoder", "set", 0),
            ("item", "Veevoer en medisyne / Droogmiddels", "set", 0),
        ],
    },
}


def apply_cost_rules(base_df: pd.DataFrame, rules: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    out = base_df.copy()
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce").fillna(0.0)

    for itype, label, mode, value in rules:
        if itype == "category":
            mask = out["Category"].astype(str).str.strip().str.lower() == label.strip().lower()
        else:
            mask = out["Item"].astype(str).str.strip().str.lower() == label.strip().lower()

        if mode == "set":
            out.loc[mask, "Value"] = float(value)
        elif mode == "abs":
            out.loc[mask, "Value"] = out.loc[mask, "Value"] + float(value)
        elif mode == "pct":
            out.loc[mask, "Value"] = out.loc[mask, "Value"] * (1.0 + float(value) / 100.0)

    return out


def scenario_values(
    scenario_key: str,
    base_costs: pd.DataFrame,
    income_low_rt: float | None,
    income_high_rt: float | None,
    yield_low: float | None,
    yield_high: float | None,
    provision: float,
) -> dict[str, Any]:
    cfg = SCENARIOS[scenario_key]
    base = base_costs[["Category", "Item", "Avg"]].rename(columns={"Avg": "Value"}).copy()
    sc_costs = apply_cost_rules(base, cfg["cost_rules"])
    total_cash = float(sc_costs["Value"].sum())
    total_exp = total_cash + provision

    inc_low_rt = None if income_low_rt is None else float(income_low_rt) + float(cfg.get("income_abs", 0.0))
    inc_high_rt = None if income_high_rt is None else float(income_high_rt) + float(cfg.get("income_abs", 0.0))

    y_low = None
    y_high = None
    if yield_low is not None and yield_high is not None:
        if cfg.get("diff_multiplier") is not None:
            diff = float(yield_high) - float(yield_low)
            y_high = float(yield_high)
            y_low = float(yield_high) - diff * float(cfg["diff_multiplier"])
        else:
            y_low = float(yield_low) * (1.0 + float(cfg.get("yield_low_pct", 0.0)) / 100.0)
            y_high = float(yield_high) * (1.0 + float(cfg.get("yield_high_pct", 0.0)) / 100.0)

    income_low_rha = inc_low_rt * y_low if inc_low_rt is not None and y_low is not None else None
    income_high_rha = inc_high_rt * y_high if inc_high_rt is not None and y_high is not None else None
    net_low = income_low_rha - total_exp if income_low_rha is not None else None
    net_high = income_high_rha - total_exp if income_high_rha is not None else None

    return {
        "label": cfg["label"],
        "costs": sc_costs,
        "income_low_rt": inc_low_rt,
        "income_high_rt": inc_high_rt,
        "yield_low": y_low,
        "yield_high": y_high,
        "income_low_rha": income_low_rha,
        "income_high_rha": income_high_rha,
        "total_cash": total_cash,
        "provision": provision,
        "total_exp": total_exp,
        "net_low": net_low,
        "net_high": net_high,
    }


# -----------------------
# HTML table styling
# -----------------------

def build_comparison_html(profile_rows: list[tuple[str, str]], rows: list[dict[str, str]]) -> str:
    profile_html = "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in profile_rows
    )

    body = []
    for r in rows:
        row_class = r.get("_class", "")
        body.append(
            f"""
            <tr class="{row_class}">
                <td>{r.get("Section / Item", "")}</td>
                <td>{r.get("2025 / Current", "")}</td>
                <td>{r.get("2030 Farmer Forecast", "")}</td>
                <td>{r.get("2030 Industry Range", "")}</td>
                <td>{r.get("2030 Scenario Range", "")}</td>
            </tr>
            """
        )

    return f"""
    <style>
      .phase2-wrap {{
        font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
        color: #111827;
      }}
      .phase2-profile,
      .phase2-table {{
        width: 100%;
        font-size: 0.88rem;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        background: #fff;
        margin-bottom: 14px;
      }}
      .phase2-profile th {{
        width: 26%;
        text-align: left;
        background: #f8fafc;
        border-bottom: 1px solid #e5e7eb;
        padding: 6px 8px;
        color: #374151;
      }}
      .phase2-profile td {{
        border-bottom: 1px solid #e5e7eb;
        padding: 6px 8px;
      }}
      .phase2-table th {{
        background: #f8fafc;
        color: #374151;
        padding: 6px 8px;
        border-bottom: 1px solid #e5e7eb;
        font-weight: 700;
      }}
      .phase2-table th:first-child,
      .phase2-table td:first-child {{
        text-align: left;
        width: 27%;
      }}
      .phase2-table th:not(:first-child),
      .phase2-table td:not(:first-child) {{
        text-align: right;
      }}
      .phase2-table td {{
        padding: 6px 8px;
        border-bottom: 1px solid #e5e7eb;
        vertical-align: middle;
      }}
      .row-heading td {{
        background: #fafafa;
        color: #111827;
        font-weight: 700;
        text-align: left !important;
      }}
      .row-total td {{
        background: #f8fafc;
        font-weight: 700;
      }}
      .row-grand td {{
        background: #f3f4f6;
        font-weight: 800;
      }}
      .phase2-note {{
        font-size: 0.88rem;
        color: #6b7280;
        margin-top: -4px;
        margin-bottom: 10px;
      }}
    </style>

    <div class="phase2-wrap">
      <table class="phase2-profile">
        <tbody>{profile_html}</tbody>
      </table>

      <table class="phase2-table">
        <thead>
          <tr>
            <th>Section / Item</th>
            <th>2025 / Current</th>
            <th>2030 Farmer Forecast</th>
            <th>2030 Industry Range</th>
            <th>2030 Scenario Range</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body)}
        </tbody>
      </table>
      <div class="phase2-note">
        Industry and scenario columns use Low / High benchmark ranges where available. The forecast uses Phase 1 farmer values as the 2025 base.
      </div>
    </div>
    """


def build_cost_detail_html(profile_rows: list[tuple[str, str]], rows: list[dict[str, str]]) -> str:
    """Build a Phase-1-style detailed cost table for the 2030 forecast view."""
    profile_html = "".join(
        f"<tr><th>{escape(str(label))}</th><td>{escape(str(value))}</td></tr>" for label, value in profile_rows
    )

    body = []
    for r in rows:
        row_class = escape(str(r.get("_class", "")))
        body.append(
            f"""
            <tr class="{row_class}">
                <td>{escape(str(r.get("Section / Item", "")))}</td>
                <td>{escape(str(r.get("2025 Farmer Current", "")))}</td>
                <td>{escape(str(r.get("2030 Farmer Forecast", "")))}</td>
                <td>{escape(str(r.get("2030 Industry Forecast", "")))}</td>
                <td>{escape(str(r.get("2030 Scenario Forecast", "")))}</td>
            </tr>
            """
        )

    return f"""
    <style>
      .phase2-detail-wrap {{
        font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
        color: #111827;
      }}
      .phase2-detail-profile,
      .phase2-detail-table {{
        width: 100%;
        font-size: 0.86rem;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        background: #fff;
        margin-bottom: 14px;
      }}
      .phase2-detail-profile th {{
        width: 26%;
        text-align: left;
        background: #f8fafc;
        border-bottom: 1px solid #e5e7eb;
        padding: 8px 10px;
        color: #374151;
      }}
      .phase2-detail-profile td {{
        border-bottom: 1px solid #e5e7eb;
        padding: 8px 10px;
      }}
      .phase2-detail-scroll {{
        max-height: 560px;
        overflow: auto;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
      }}
      .phase2-detail-table {{
        border: none;
        border-radius: 0;
        margin-bottom: 0;
      }}
      .phase2-detail-table th {{
        position: sticky;
        top: 0;
        z-index: 5;
        background: #f8fafc;
        color: #374151;
        padding: 6px 8px;
        border-bottom: 1px solid #e5e7eb;
        font-weight: 700;
      }}
      .phase2-detail-table th:first-child,
      .phase2-detail-table td:first-child {{
        text-align: left;
        width: 30%;
      }}
      .phase2-detail-table th:not(:first-child),
      .phase2-detail-table td:not(:first-child) {{
        text-align: right;
      }}
      .phase2-detail-table td {{
        padding: 6px 8px;
        border-bottom: 1px solid #e5e7eb;
        vertical-align: middle;
      }}
      .row-cat td {{
        background: #fff;
      }}
      .row-cat td:first-child {{
        font-weight: 800;
      }}
      .row-item td:first-child {{
        padding-left: 26px;
        color: #374151;
      }}
      .row-total td {{
        background: #f8fafc;
        font-weight: 700;
      }}
      .row-grand td {{
        background: #f3f4f6;
        font-weight: 800;
      }}
      .phase2-detail-note {{
        font-size: 0.88rem;
        color: #6b7280;
        margin-top: 8px;
        margin-bottom: 4px;
      }}
    </style>

    <div class="phase2-detail-wrap">
      <table class="phase2-detail-profile">
        <tbody>{profile_html}</tbody>
      </table>
      <div class="phase2-detail-scroll">
        <table class="phase2-detail-table">
          <thead>
            <tr>
              <th>Section / Item</th>
              <th>2025 Farmer Current</th>
              <th>2030 Farmer Forecast</th>
              <th>2030 Industry Forecast</th>
              <th>2030 Scenario Forecast</th>
            </tr>
          </thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
      <div class="phase2-detail-note">
        Detail view uses the Phase 1 farmer values as the 2025 base and applies the selected forecast assumptions to show the 2030 position.
      </div>
    </div>
    """


def build_forecast_cost_detail_rows(
    farmer_costs: pd.DataFrame,
    base_costs: pd.DataFrame,
    scenario_costs: pd.DataFrame,
    cost_growth: float,
    periods: int,
    provision_current: float,
    provision_2030: float,
    farmer_net_current: float,
    farmer_net_2030: float,
    industry_net_low_2030: float | None,
    industry_net_high_2030: float | None,
    scenario_net_low_2030: float | None,
    scenario_net_high_2030: float | None,
) -> list[dict[str, str]]:
    """Create detailed cost rows so Page 2A can mirror the Phase 1 cost detail."""
    rows: list[dict[str, str]] = []

    merged = base_costs[["Category", "Item", "Avg"]].merge(
        farmer_costs[["Category", "Item", "Cost"]],
        on=["Category", "Item"],
        how="left",
    )
    merged["Cost"] = pd.to_numeric(merged["Cost"], errors="coerce").fillna(0.0)
    merged["Avg"] = pd.to_numeric(merged["Avg"], errors="coerce").fillna(0.0)
    merged["Farmer2030"] = merged["Cost"].apply(lambda v: _pct_change(v, cost_growth, periods))
    merged["Industry2030"] = merged["Avg"].apply(lambda v: _pct_change(v, cost_growth, periods))

    sc = scenario_costs[["Category", "Item", "Value"]].copy() if not scenario_costs.empty else pd.DataFrame(columns=["Category", "Item", "Value"])
    sc["Value"] = pd.to_numeric(sc["Value"], errors="coerce").fillna(0.0)
    sc_lookup = {
        (_clean_text(r["Category"]).lower(), _clean_text(r["Item"]).lower()): float(r["Value"])
        for _, r in sc.iterrows()
    }
    merged["Scenario2030"] = merged.apply(
        lambda r: sc_lookup.get((_clean_text(r["Category"]).lower(), _clean_text(r["Item"]).lower()), 0.0),
        axis=1,
    )

    for cat in merged["Category"].drop_duplicates().tolist():
        sub = merged[merged["Category"] == cat]
        rows.append(
            {
                "Section / Item": cat,
                "2025 Farmer Current": _fmt_money(sub["Cost"].sum()),
                "2030 Farmer Forecast": _fmt_money(sub["Farmer2030"].sum()),
                "2030 Industry Forecast": _fmt_money(sub["Industry2030"].sum()),
                "2030 Scenario Forecast": _fmt_money(sub["Scenario2030"].sum()),
                "_class": "row-cat",
            }
        )
        for _, r in sub.iterrows():
            rows.append(
                {
                    "Section / Item": "  " + _clean_text(r["Item"]),
                    "2025 Farmer Current": _fmt_money(r["Cost"]),
                    "2030 Farmer Forecast": _fmt_money(r["Farmer2030"]),
                    "2030 Industry Forecast": _fmt_money(r["Industry2030"]),
                    "2030 Scenario Forecast": _fmt_money(r["Scenario2030"]),
                    "_class": "row-item",
                }
            )

    farmer_cash_current = float(pd.to_numeric(merged["Cost"], errors="coerce").fillna(0.0).sum())
    farmer_cash_2030 = float(pd.to_numeric(merged["Farmer2030"], errors="coerce").fillna(0.0).sum())
    industry_cash_2030 = float(pd.to_numeric(merged["Industry2030"], errors="coerce").fillna(0.0).sum())
    scenario_cash_2030 = float(pd.to_numeric(merged["Scenario2030"], errors="coerce").fillna(0.0).sum())

    rows.extend(
        [
            {
                "Section / Item": "Totale kontant uitgawes",
                "2025 Farmer Current": _fmt_money(farmer_cash_current),
                "2030 Farmer Forecast": _fmt_money(farmer_cash_2030),
                "2030 Industry Forecast": _fmt_money(industry_cash_2030),
                "2030 Scenario Forecast": _fmt_money(scenario_cash_2030),
                "_class": "row-total",
            },
            {
                "Section / Item": "Provision for renewal",
                "2025 Farmer Current": _fmt_money(provision_current),
                "2030 Farmer Forecast": _fmt_money(provision_2030),
                "2030 Industry Forecast": _fmt_money(provision_2030),
                "2030 Scenario Forecast": _fmt_money(provision_2030),
                "_class": "row-total",
            },
            {
                "Section / Item": "Total Expenditure",
                "2025 Farmer Current": _fmt_money(farmer_cash_current + provision_current),
                "2030 Farmer Forecast": _fmt_money(farmer_cash_2030 + provision_2030),
                "2030 Industry Forecast": _fmt_money(industry_cash_2030 + provision_2030),
                "2030 Scenario Forecast": _fmt_money(scenario_cash_2030 + provision_2030),
                "_class": "row-total",
            },
            {
                "Section / Item": "Net Farm Income (R/ha)",
                "2025 Farmer Current": _fmt_money(farmer_net_current),
                "2030 Farmer Forecast": _fmt_money(farmer_net_2030),
                "2030 Industry Forecast": _fmt_range(industry_net_low_2030, industry_net_high_2030, money=True),
                "2030 Scenario Forecast": _fmt_range(scenario_net_low_2030, scenario_net_high_2030, money=True),
                "_class": "row-grand",
            },
        ]
    )
    return rows


def _rand_axis(value: float, _position: int) -> str:
    return f"R {value:,.0f}"


def _dist_to_band(value: float | None, low: float | None, high: float | None) -> tuple[str, float]:
    if value is None or low is None or high is None:
        return "unknown", 0.0
    low, high = min(low, high), max(low, high)
    if low <= value <= high:
        return "inside", min(value - low, high - value)
    if value < low:
        return "below", low - value
    return "above", value - high


def _chip(text: str, tone: str = "neutral") -> str:
    colours = {
        "good": ("#0f5132", "#d1e7dd", "#badbcc"),
        "bad": ("#842029", "#f8d7da", "#f5c2c7"),
        "warn": ("#664d03", "#fff3cd", "#ffe69c"),
        "info": ("#055160", "#cff4fc", "#b6effb"),
        "neutral": ("#374151", "#f3f4f6", "#e5e7eb"),
    }
    fg, bg, border = colours.get(tone, colours["neutral"])
    return (
        f'<span style="display:inline-block;padding:4px 10px;border:1px solid {border};'
        f'border-radius:999px;background:{bg};color:{fg};font-size:12px;line-height:1;">'
        f'{escape(text)}</span>'
    )


def build_forecast_range_plot(
    farmer_yield: float,
    farmer_net: float,
    industry_yield_low: float | None,
    industry_yield_high: float | None,
    industry_net_low: float | None,
    industry_net_high: float | None,
    scenario_yield_low: float | None,
    scenario_yield_high: float | None,
    scenario_net_low: float | None,
    scenario_net_high: float | None,
    scenario_label: str,
):
    """Phase-1-style forecast competitiveness map for the selected target year."""
    if all(v is not None for v in [industry_yield_low, industry_yield_high, industry_net_low, industry_net_high]):
        y0, y1 = min(industry_yield_low, industry_yield_high), max(industry_yield_low, industry_yield_high)
        n0, n1 = min(industry_net_low, industry_net_high), max(industry_net_low, industry_net_high)
    else:
        y0 = y1 = farmer_yield
        n0 = n1 = farmer_net

    sy0 = sy1 = sn0 = sn1 = None
    if all(v is not None for v in [scenario_yield_low, scenario_yield_high, scenario_net_low, scenario_net_high]):
        sy0, sy1 = min(scenario_yield_low, scenario_yield_high), max(scenario_yield_low, scenario_yield_high)
        sn0, sn1 = min(scenario_net_low, scenario_net_high), max(scenario_net_low, scenario_net_high)

    in_range = y0 <= farmer_yield <= y1 and n0 <= farmer_net <= n1
    point_colour = PALETTE["green_pos"] if farmer_net >= 0 else PALETTE["red_neg"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.fill_between([y0, y1], n0, n1, step="pre", alpha=0.28, color=PALETTE["industry_fill"], zorder=1)
    ax.plot(
        [y0, y0, y1, y1, y0],
        [n0, n1, n1, n0, n0],
        color=PALETTE["industry_edge"],
        linewidth=1.1,
        zorder=2,
    )

    if sy0 is not None and sy1 is not None and sn0 is not None and sn1 is not None:
        ax.plot(
            [sy0, sy0, sy1, sy1, sy0],
            [sn0, sn1, sn1, sn0, sn0],
            color=PALETTE["scenario_edge"],
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            zorder=2.5,
        )

    ax.axhline(0, color=PALETTE["zero_line"], linewidth=1.2, zorder=0)
    if in_range:
        ax.scatter(
            farmer_yield,
            farmer_net,
            s=54,
            facecolor=point_colour,
            edgecolor="white",
            linewidths=0.9,
            zorder=3,
        )
    else:
        ax.scatter(
            farmer_yield,
            farmer_net,
            s=62,
            facecolor="none",
            edgecolor=point_colour,
            linewidths=1.6,
            zorder=3,
        )
    ax.annotate(
        "Farmer",
        (farmer_yield, farmer_net),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=8.5,
        color=point_colour,
        weight="bold",
    )

    x_candidates = [v for v in [y0, y1, sy0, sy1, farmer_yield] if v is not None]
    y_candidates = [v for v in [n0, n1, sn0, sn1, farmer_net] if v is not None]
    x_min, x_max = min(x_candidates), max(x_candidates)
    y_min, y_max = min(y_candidates), max(y_candidates)
    x_span = max(0.1, (x_max - x_min) or 1.0)
    y_span = max(0.1, (y_max - y_min) or 1.0)
    ax.set_xlim(x_min - x_span * 0.18, x_max + x_span * 0.18)
    ax.set_ylim(y_min - y_span * 0.20, y_max + y_span * 0.20)

    ax.set_xlabel("Yield (t/ha)", fontsize=9)
    ax.set_ylabel("Net Farm Income (R/ha)", fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(_rand_axis))
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(alpha=0.12)

    farmer_face = point_colour if in_range else "none"
    legend_handles = [
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=8,
            markerfacecolor=PALETTE["industry_fill"],
            markeredgecolor=PALETTE["industry_edge"],
            label="Industry range",
        ),
        Line2D(
            [0], [0], color=PALETTE["scenario_edge"], linestyle="--",
            linewidth=1.3, label=f"{scenario_label} range",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6.5,
            markerfacecolor=farmer_face, markeredgecolor=point_colour,
            markeredgewidth=1.4, label="Farmer",
        ),
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


def plot_forecast_net_summary_bar(
    farmer_yield: float,
    farmer_net: float,
    industry_yield_low: float | None,
    industry_yield_high: float | None,
    industry_net_low: float | None,
    industry_net_high: float | None,
    scenario_yield_low: float | None,
    scenario_yield_high: float | None,
    scenario_net_low: float | None,
    scenario_net_high: float | None,
    scenario_label: str,
):
    """Phase-1-style bar-and-marker chart for the selected forecast year."""
    items: list[tuple[str, float, float | None]] = [("Farmer", farmer_net, farmer_yield)]
    for label, net_value, yield_value in [
        ("Industry Low", industry_net_low, industry_yield_low),
        ("Industry High", industry_net_high, industry_yield_high),
        (f"{scenario_label} Low", scenario_net_low, scenario_yield_low),
        (f"{scenario_label} High", scenario_net_high, scenario_yield_high),
    ]:
        if net_value is not None:
            items.append((label, net_value, yield_value))

    labels = [label for label, _, _ in items]
    net_values = [value for _, value, _ in items]
    yield_values = [value for _, _, value in items]

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
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)

    benchmark_colour = "#B0B8C4"
    colours = [benchmark_colour] * len(net_values)
    colours[0] = PALETTE["green_pos"] if farmer_net >= 0 else PALETTE["red_neg"]

    x = list(range(len(labels)))
    ax.bar(x, net_values, width=0.56, color=colours, alpha=0.90)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=8.5, linespacing=1.15)
    ax.tick_params(axis="x", pad=7)
    ax.margins(x=0.08)
    ax.set_ylabel("Net Farm Income (R/ha)")
    ax.yaxis.set_major_formatter(FuncFormatter(_rand_axis))
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=1.2)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(axis="y", colors="#6b7280")

    ax2 = ax.twinx()
    ax2.set_ylabel("Yield (t/ha)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.1f}"))
    ax2.grid(False)
    ax2.tick_params(axis="y", colors="#6b7280")
    for i, (yield_value, bar_colour) in enumerate(zip(yield_values, colours)):
        if yield_value is None:
            continue
        ax2.scatter(
            i,
            yield_value,
            s=58,
            marker="o",
            facecolor=bar_colour,
            edgecolor="#1f2937",
            linewidths=0.8,
            zorder=5,
        )

    legend_handles = [
        Patch(
            facecolor=benchmark_colour,
            edgecolor="#4b5563",
            label="Net Farm Income (bars)",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="#1f2937", markeredgecolor="#1f2937",
            label="Yield (markers)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
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
    fig.subplots_adjust(top=0.76, bottom=0.24)
    return fig


# -----------------------
# App rendering
# -----------------------

st.title("Phase 2 — Outlook to 2030")
st.markdown(
    "<p style='color:#6b7280; margin-top:-10px;'><em>Consistent forecast view for farmer-friendly benchmarking and decision support.</em></p>",
    unsafe_allow_html=True,
)

with st.expander("Quick Guide", expanded=False):
    st.markdown(
        """
**Purpose**
- Compare the **current/base year position** with the **2030 forecast position**.
- Keep the forecast layout consistent with the benchmark page.
- Retain a detailed year-by-year view for users who want the full pathway.

**How to use**
1. Select the vineyard profile and scenario in the sidebar.
2. Check the growth assumptions.
3. Use **Page 2A** for the main farmer-friendly view.
4. Use **Page 2B** for the detailed year-by-year forecast.
        """
    )

income_df = _read_csv("income.csv")
yield_df = _read_csv("yield.csv")
costs_df = _read_csv("costs.csv")

ic = income_cols(income_df)
yc = yield_cols(yield_df)
cc = cost_cols(costs_df)

missing = []
if income_df.empty:
    missing.append("income.csv")
if yield_df.empty:
    missing.append("yield.csv")
if costs_df.empty:
    missing.append("costs.csv")

if missing:
    st.stop()

# Keep Phase 2 consistent with Phase 1: if yield values were supplied in kg/ha,
# convert them to t/ha before any benchmark calculations are made.
if yc.yield_tha and yc.yield_tha in yield_df.columns:
    _yield_median = pd.to_numeric(yield_df[yc.yield_tha], errors="coerce").median()
    if pd.notna(_yield_median) and _yield_median > 1000:
        yield_df[yc.yield_tha] = pd.to_numeric(yield_df[yc.yield_tha], errors="coerce") / 1000.0

phase1_base = st.session_state.get("phase1_current_base", {})
if not isinstance(phase1_base, dict):
    phase1_base = {}

def _base_default(key: str, fallback: Any) -> Any:
    value = phase1_base.get(key, fallback)
    return fallback if value is None else value


def _phase1_cost_lookup() -> dict[tuple[str, str], float]:
    rows = phase1_base.get("farmer_costs", [])
    lookup: dict[tuple[str, str], float] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            cat = _clean_text(row.get("Category"))
            item = _clean_text(row.get("Item"))
            if cat and item:
                lookup[(cat.lower(), item.lower())] = _safe_float(row.get("Cost"), 0.0)
    return lookup


def _select_index(options: list[Any], preferred: Any, fallback_index: int = 0) -> int:
    if not options:
        return 0
    if preferred is not None:
        for i, opt in enumerate(options):
            if _norm(opt) == _norm(preferred):
                return i
    return max(0, min(fallback_index, len(options) - 1))


phase1_cost_lookup = _phase1_cost_lookup()

with st.sidebar:
    st.header("Forecast Inputs")
    if phase1_base:
        st.success("Using Phase 1 current values as the base year. You can still adjust the forecast assumptions below.")
    else:
        st.info("Open Phase 1 first if you want this page to use the farmer's current inputs as the base year.")
    st.caption("Defaults are loaded from the benchmark data. Adjust as needed.")

    # Selection lists
    if ic.wine_class and ic.wine_class in income_df.columns:
        wine_classes = sorted(income_df[ic.wine_class].dropna().astype(str).str.strip().unique().tolist())
    elif yc.wine_class and yc.wine_class in yield_df.columns:
        wine_classes = sorted(yield_df[yc.wine_class].dropna().astype(str).str.strip().unique().tolist())
    else:
        wine_classes = ["White", "Red"]

    wine_class = st.selectbox(
        "Wine Class",
        wine_classes,
        index=_choose_default(wine_classes, _base_default("wine_class", "White")),
        key="phase2_wine_class",
    )

    grape_source = income_df
    grape_col = ic.grape_variety
    if grape_col and ic.wine_class:
        grape_filter = grape_source[ic.wine_class].astype(str).str.strip().str.lower() == wine_class.lower()
        grapes = sorted(grape_source.loc[grape_filter, grape_col].dropna().astype(str).str.strip().unique().tolist())
    else:
        grapes = []
    if not grapes and yc.grape_variety:
        grapes = sorted(yield_df[yc.grape_variety].dropna().astype(str).str.strip().unique().tolist())

    grape_variety = st.selectbox(
        "Grape Variety",
        grapes if grapes else ["Bukettraube"],
        index=_choose_default(grapes if grapes else ["Bukettraube"], _base_default("grape_variety", "Bukettraube")),
        key="phase2_grape_variety",
    )

    regions: set[str] = set()
    if ic.region and ic.grape_variety:
        sub = income_df.copy()
        if ic.wine_class:
            sub = sub[sub[ic.wine_class].astype(str).str.strip().str.lower() == wine_class.lower()]
        sub = sub[sub[ic.grape_variety].astype(str).str.strip().str.lower() == grape_variety.lower()]
        regions.update(sub[ic.region].dropna().astype(str).str.strip().unique().tolist())
    if yc.region and yc.grape_variety:
        sub = yield_df.copy()
        if yc.wine_class:
            sub = sub[sub[yc.wine_class].astype(str).str.strip().str.lower() == wine_class.lower()]
        sub = sub[sub[yc.grape_variety].astype(str).str.strip().str.lower() == grape_variety.lower()]
        regions.update(sub[yc.region].dropna().astype(str).str.strip().unique().tolist())
    if cc.region:
        regions.update(costs_df[cc.region].dropna().astype(str).str.strip().unique().tolist())

    regions_list = sorted(regions) if regions else ["BREEDEKLOOF"]
    region = st.selectbox(
        "Region",
        regions_list,
        index=_choose_default(regions_list, _base_default("region", "BREEDEKLOOF")),
        key="phase2_region",
    )

    years = []
    if ic.year and ic.year in income_df.columns:
        years = sorted(pd.to_numeric(income_df[ic.year], errors="coerce").dropna().astype(int).unique().tolist())
    base_year_options = years if years else [2025]
    base_year = st.selectbox(
        "Base year",
        base_year_options,
        index=_select_index(base_year_options, _base_default("year", None), len(base_year_options) - 1),
        key="phase2_base_year",
    )

    target_year = st.selectbox(
        "Forecast year",
        [2030, 2029, 2028, 2027, 2026],
        index=0,
        key="phase2_target_year",
    )

    scenario_options = list(SCENARIOS.keys())
    scenario_key = st.selectbox(
        "Scenario",
        scenario_options,
        index=_select_index(scenario_options, _base_default("scenario_key", scenario_options[0]), 0),
        key="phase2_scenario",
    )

    st.markdown("---")
    st.header("Farmer Current Inputs")

    # Industry defaults
    income_low = lookup_band(income_df, ic, wine_class, grape_variety, region, "Low", ic.income_rt, int(base_year))
    income_high = lookup_band(income_df, ic, wine_class, grape_variety, region, "High", ic.income_rt, int(base_year))
    yield_low = lookup_band(yield_df, yc, wine_class, grape_variety, region, "Low", yc.yield_tha, None)
    yield_high = lookup_band(yield_df, yc, wine_class, grape_variety, region, "High", yc.yield_tha, None)

    if yield_low is None or yield_high is None:
        st.warning(
            "No Low/High yield benchmark was found for this selection. "
            "Industry and scenario income/NFI ranges will show as dashes until yield data is available."
        )

    default_income = (income_low + income_high) / 2 if income_low is not None and income_high is not None else 5000.0
    default_yield = (yield_low + yield_high) / 2 if yield_low is not None and yield_high is not None else 24.6
    default_income = _safe_float(_base_default("farmer_income_rt", default_income), default_income)
    default_yield = _safe_float(_base_default("farmer_yield", default_yield), default_yield)

    farmer_income_rt = st.number_input(
        "Gross Income (R/t)",
        min_value=0.0,
        value=float(default_income),
        step=50.0,
        format="%.2f",
        key="phase2_farmer_income_rt",
    )

    farmer_yield = st.number_input(
        "Yield (t/ha)",
        min_value=0.0,
        value=float(default_yield),
        step=0.1,
        format="%.2f",
        key="phase2_farmer_yield",
    )

    provision_current = st.number_input(
        "Provision for renewal (R/ha)",
        min_value=0.0,
        value=float(_safe_float(_base_default("provision_for_renewal", 20125.0), 20125.0)),
        step=100.0,
        format="%.2f",
        key="phase2_provision",
    )

    st.markdown("---")
    st.header("Growth Assumptions")

    suggested_income_growth = lookup_income_growth_percent(
        income_df, ic, wine_class, grape_variety, region, int(base_year)
    )
    if suggested_income_growth is not None:
        st.caption(
            f"Suggested income growth from the income data: {suggested_income_growth:.2f}% per year. "
            "You can adjust it if needed."
        )
    else:
        st.caption("No income-growth coefficient was found in the income data. Using the default assumption.")

    income_growth = st.number_input(
        "Income growth (% per year)",
        value=float(suggested_income_growth) if suggested_income_growth is not None else 5.0,
        step=0.5,
        format="%.2f",
        key="phase2_income_growth",
    )
    cost_growth = st.number_input(
        "Cost growth (% per year)",
        value=5.0,
        step=0.5,
        format="%.2f",
        key="phase2_cost_growth",
    )
    yield_growth = st.number_input(
        "Yield growth (% per year)",
        value=0.0,
        step=0.25,
        format="%.2f",
        key="phase2_yield_growth",
    )
    provision_growth = st.number_input(
        "Provision growth (% per year)",
        value=5.0,
        step=0.5,
        format="%.2f",
        key="phase2_provision_growth",
    )

base_costs = prepare_costs(costs_df, cc, region)
if base_costs.empty:
    st.warning("No cost data found for this selected region. The forecast can still run, but costs will be zero.")

# Farmer cost inputs, kept compact but editable.
with st.sidebar.expander("Farmer Costs", expanded=False):
    st.caption("Loaded from regional average costs. Edit where the farmer differs.")
    farmer_cost_rows = []
    for _, row in base_costs.iterrows():
        key = f"phase2_cost::{region}::{row['Category']}::{row['Item']}"
        default_cost = phase1_cost_lookup.get(
            (_clean_text(row["Category"]).lower(), _clean_text(row["Item"]).lower()),
            float(row["Avg"]),
        )
        val = st.number_input(
            f"{row['Category']} — {row['Item']}",
            min_value=0.0,
            value=float(default_cost),
            step=10.0,
            format="%.2f",
            key=key,
        )
        farmer_cost_rows.append((row["Category"], row["Item"], val))

farmer_costs = pd.DataFrame(farmer_cost_rows, columns=["Category", "Item", "Cost"])
periods = max(0, int(target_year) - int(base_year))

# Current values
farmer_cash_current = float(pd.to_numeric(farmer_costs["Cost"], errors="coerce").fillna(0.0).sum())
farmer_exp_current = farmer_cash_current + float(provision_current)
farmer_income_rha_current = float(farmer_income_rt) * float(farmer_yield)
farmer_net_current = farmer_income_rha_current - farmer_exp_current

industry_cash_current = float(pd.to_numeric(base_costs["Avg"], errors="coerce").fillna(0.0).sum())
industry_exp_current = industry_cash_current + float(provision_current)
industry_income_low_rha = income_low * yield_low if income_low is not None and yield_low is not None else None
industry_income_high_rha = income_high * yield_high if income_high is not None and yield_high is not None else None
industry_net_low_current = industry_income_low_rha - industry_exp_current if industry_income_low_rha is not None else None
industry_net_high_current = industry_income_high_rha - industry_exp_current if industry_income_high_rha is not None else None

# Forecast values
farmer_income_rt_2030 = _pct_change(farmer_income_rt, income_growth, periods)
farmer_yield_2030 = _pct_change(farmer_yield, yield_growth, periods)
farmer_cash_2030 = _pct_change(farmer_cash_current, cost_growth, periods)
provision_2030 = _pct_change(provision_current, provision_growth, periods)
farmer_exp_2030 = farmer_cash_2030 + provision_2030
farmer_income_rha_2030 = farmer_income_rt_2030 * farmer_yield_2030
farmer_net_2030 = farmer_income_rha_2030 - farmer_exp_2030

industry_low_rt_2030 = _pct_change(income_low, income_growth, periods) if income_low is not None else None
industry_high_rt_2030 = _pct_change(income_high, income_growth, periods) if income_high is not None else None
industry_yield_low_2030 = _pct_change(yield_low, yield_growth, periods) if yield_low is not None else None
industry_yield_high_2030 = _pct_change(yield_high, yield_growth, periods) if yield_high is not None else None
industry_cash_2030 = _pct_change(industry_cash_current, cost_growth, periods)
industry_exp_2030 = industry_cash_2030 + provision_2030
industry_income_low_2030 = (
    industry_low_rt_2030 * industry_yield_low_2030
    if industry_low_rt_2030 is not None and industry_yield_low_2030 is not None
    else None
)
industry_income_high_2030 = (
    industry_high_rt_2030 * industry_yield_high_2030
    if industry_high_rt_2030 is not None and industry_yield_high_2030 is not None
    else None
)
industry_net_low_2030 = industry_income_low_2030 - industry_exp_2030 if industry_income_low_2030 is not None else None
industry_net_high_2030 = industry_income_high_2030 - industry_exp_2030 if industry_income_high_2030 is not None else None

scenario_current = scenario_values(
    scenario_key,
    base_costs,
    income_low,
    income_high,
    yield_low,
    yield_high,
    float(provision_current),
)
scenario_2030 = scenario_values(
    scenario_key,
    base_costs.assign(Avg=base_costs["Avg"].apply(lambda v: _pct_change(v, cost_growth, periods))),
    industry_low_rt_2030,
    industry_high_rt_2030,
    industry_yield_low_2030,
    industry_yield_high_2030,
    provision_2030,
)

scenario_label = scenario_2030["label"]

profile_rows = [
    ("Base Year", str(base_year)),
    ("Forecast Year", str(target_year)),
    ("Wine Class", wine_class),
    ("Grape Variety", grape_variety),
    ("Region", region),
    ("Scenario", scenario_label),
]

summary_rows = [
    {"Section / Item": "Vineyard Performance", "_class": "row-heading"},
    {
        "Section / Item": "Gross Income (R/t)",
        "2025 / Current": _fmt_money(farmer_income_rt),
        "2030 Farmer Forecast": _fmt_money(farmer_income_rt_2030),
        "2030 Industry Range": _fmt_range(industry_low_rt_2030, industry_high_rt_2030, money=True),
        "2030 Scenario Range": _fmt_range(scenario_2030["income_low_rt"], scenario_2030["income_high_rt"], money=True),
    },
    {
        "Section / Item": "Yield (t/ha)",
        "2025 / Current": _fmt_num(farmer_yield, 2),
        "2030 Farmer Forecast": _fmt_num(farmer_yield_2030, 2),
        "2030 Industry Range": _fmt_range(industry_yield_low_2030, industry_yield_high_2030, money=False, decimals=2),
        "2030 Scenario Range": _fmt_range(scenario_2030["yield_low"], scenario_2030["yield_high"], money=False, decimals=2),
    },
    {
        "Section / Item": "Income (R/ha)",
        "2025 / Current": _fmt_money(farmer_income_rha_current),
        "2030 Farmer Forecast": _fmt_money(farmer_income_rha_2030),
        "2030 Industry Range": _fmt_range(industry_income_low_2030, industry_income_high_2030, money=True),
        "2030 Scenario Range": _fmt_range(scenario_2030["income_low_rha"], scenario_2030["income_high_rha"], money=True),
    },
    {"Section / Item": "Costs and Totals", "_class": "row-heading"},
    {
        "Section / Item": "Total cash costs (R/ha)",
        "2025 / Current": _fmt_money(farmer_cash_current),
        "2030 Farmer Forecast": _fmt_money(farmer_cash_2030),
        "2030 Industry Range": _fmt_money(industry_cash_2030),
        "2030 Scenario Range": _fmt_money(scenario_2030["total_cash"]),
        "_class": "row-total",
    },
    {
        "Section / Item": "Provision for renewal (R/ha)",
        "2025 / Current": _fmt_money(provision_current),
        "2030 Farmer Forecast": _fmt_money(provision_2030),
        "2030 Industry Range": _fmt_money(provision_2030),
        "2030 Scenario Range": _fmt_money(scenario_2030["provision"]),
    },
    {
        "Section / Item": "Total expenditure (R/ha)",
        "2025 / Current": _fmt_money(farmer_exp_current),
        "2030 Farmer Forecast": _fmt_money(farmer_exp_2030),
        "2030 Industry Range": _fmt_money(industry_exp_2030),
        "2030 Scenario Range": _fmt_money(scenario_2030["total_exp"]),
        "_class": "row-total",
    },
    {
        "Section / Item": "Net Farm Income (R/ha)",
        "2025 / Current": _fmt_money(farmer_net_current),
        "2030 Farmer Forecast": _fmt_money(farmer_net_2030),
        "2030 Industry Range": _fmt_range(industry_net_low_2030, industry_net_high_2030, money=True),
        "2030 Scenario Range": _fmt_range(scenario_2030["net_low"], scenario_2030["net_high"], money=True),
        "_class": "row-grand",
    },
]

cost_detail_rows = build_forecast_cost_detail_rows(
    farmer_costs=farmer_costs,
    base_costs=base_costs,
    scenario_costs=scenario_2030.get("costs", pd.DataFrame(columns=["Category", "Item", "Value"])),
    cost_growth=cost_growth,
    periods=periods,
    provision_current=float(provision_current),
    provision_2030=float(provision_2030),
    farmer_net_current=float(farmer_net_current),
    farmer_net_2030=float(farmer_net_2030),
    industry_net_low_2030=industry_net_low_2030,
    industry_net_high_2030=industry_net_high_2030,
    scenario_net_low_2030=scenario_2030["net_low"],
    scenario_net_high_2030=scenario_2030["net_high"],
)



tab_a, tab_b = st.tabs(
    [
        "Page 2A — 2030 Forecast Benchmark",
        "Page 2B — Detailed Year-by-Year Forecast",
    ]
)

with tab_a:
    st.subheader("Page 2A — 2030 Forecast Benchmark")
    st.caption(
        "Main client-friendly view: Phase 1 current values are carried forward and shown alongside the 2030 forecast position."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Net Farm Income", _fmt_money(farmer_net_current))
    c2.metric("2030 Forecast Net Farm Income", _fmt_money(farmer_net_2030), delta=_fmt_money(farmer_net_2030 - farmer_net_current))
    c3.metric("Scenario", scenario_label)

    st.iframe(
        build_comparison_html(profile_rows, summary_rows),
        height=610,
        width="stretch",
    )

    st.markdown("### Detailed Costs and Totals — 2025 Base to 2030 Forecast")
    st.caption(
        "This mirrors the detailed Phase 1 cost section, but shows the Phase 1 base values next to the projected 2030 values."
    )
    st.iframe(
        build_cost_detail_html(profile_rows, cost_detail_rows),
        height=760,
        width="stretch",
    )

    st.markdown("### Regional Competitiveness Map — 2030 Yield and Net Farm Income")
    st.caption(
        "This forecast map shows how the farmer's projected yield and profitability compare with the projected regional industry and scenario ranges."
    )

    _industry_y0 = min(industry_yield_low_2030, industry_yield_high_2030) if industry_yield_low_2030 is not None and industry_yield_high_2030 is not None else None
    _industry_y1 = max(industry_yield_low_2030, industry_yield_high_2030) if industry_yield_low_2030 is not None and industry_yield_high_2030 is not None else None
    _industry_n0 = min(industry_net_low_2030, industry_net_high_2030) if industry_net_low_2030 is not None and industry_net_high_2030 is not None else None
    _industry_n1 = max(industry_net_low_2030, industry_net_high_2030) if industry_net_low_2030 is not None and industry_net_high_2030 is not None else None

    _yield_status, _yield_distance = _dist_to_band(farmer_yield_2030, _industry_y0, _industry_y1)
    _nfi_status, _nfi_distance = _dist_to_band(farmer_net_2030, _industry_n0, _industry_n1)
    _profit = farmer_net_2030 >= 0
    _inside = _yield_status == "inside" and _nfi_status == "inside"

    def _distance_text(label: str, status: str, distance: float, unit: str) -> str:
        if status == "inside":
            return f"{label}: inside band"
        if status == "below":
            return f"{label}: {unit} {distance:,.2f} below"
        if status == "above":
            return f"{label}: {unit} {distance:,.2f} above"
        return f"{label}: n/a"

    st.markdown(
        '<div style="margin:6px 0 2px 0;display:flex;gap:8px;flex-wrap:wrap;">'
        + _chip(f"{'Profit' if _profit else 'Loss'} · R {farmer_net_2030:,.0f}/ha", "good" if _profit else "bad")
        + _chip("In industry range" if _inside else "Out of industry range", "good" if _inside else "warn")
        + _chip(_distance_text("Yield", _yield_status, _yield_distance, "t/ha"), "info")
        + _chip(_distance_text("NFI", _nfi_status, _nfi_distance, "R/ha"), "info")
        + "</div>",
        unsafe_allow_html=True,
    )

    forecast_map_fig = build_forecast_range_plot(
        farmer_yield=float(farmer_yield_2030),
        farmer_net=float(farmer_net_2030),
        industry_yield_low=industry_yield_low_2030,
        industry_yield_high=industry_yield_high_2030,
        industry_net_low=industry_net_low_2030,
        industry_net_high=industry_net_high_2030,
        scenario_yield_low=scenario_2030["yield_low"],
        scenario_yield_high=scenario_2030["yield_high"],
        scenario_net_low=scenario_2030["net_low"],
        scenario_net_high=scenario_2030["net_high"],
        scenario_label=scenario_label,
    )
    map_left, map_centre, map_right = st.columns([1, 6, 1])
    with map_centre:
        st.pyplot(forecast_map_fig, width="stretch")

    st.markdown("### Regional Competitiveness Chart — 2030 Yield and Net Farm Income")
    st.caption(
        "This forecast snapshot compares the farmer's projected profitability with projected industry and scenario benchmarks."
    )
    forecast_bar_fig = plot_forecast_net_summary_bar(
        farmer_yield=float(farmer_yield_2030),
        farmer_net=float(farmer_net_2030),
        industry_yield_low=industry_yield_low_2030,
        industry_yield_high=industry_yield_high_2030,
        industry_net_low=industry_net_low_2030,
        industry_net_high=industry_net_high_2030,
        scenario_yield_low=scenario_2030["yield_low"],
        scenario_yield_high=scenario_2030["yield_high"],
        scenario_net_low=scenario_2030["net_low"],
        scenario_net_high=scenario_2030["net_high"],
        scenario_label=scenario_label,
    )
    bar_left, bar_centre, bar_right = st.columns([1, 8, 1])
    with bar_centre:
        st.pyplot(forecast_bar_fig, width="stretch")

    plt.close(forecast_map_fig)
    plt.close(forecast_bar_fig)

    export_df = pd.DataFrame([{k: v for k, v in row.items() if k != "_class"} for row in summary_rows])
    detail_export_df = pd.DataFrame([{k: v for k, v in row.items() if k != "_class"} for row in cost_detail_rows])
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download Page 2A summary as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Phase2A_2030_summary_{region}_{base_year}_{target_year}.csv",
            mime="text/csv",
        )
    with col_dl2:
        st.download_button(
            "Download Page 2A detailed costs as CSV",
            data=detail_export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Phase2A_2030_detailed_costs_{region}_{base_year}_{target_year}.csv",
            mime="text/csv",
        )

with tab_b:
    st.subheader("Page 2B — Detailed Year-by-Year Forecast")
    st.caption(
        "Optional detailed view showing the pathway between the base year and the 2030 forecast."
    )

    years_path = list(range(int(base_year), int(target_year) + 1))
    detailed_rows = []

    for yr in years_path:
        p = yr - int(base_year)

        f_income_rt = _pct_change(farmer_income_rt, income_growth, p)
        f_yield = _pct_change(farmer_yield, yield_growth, p)
        f_cash = _pct_change(farmer_cash_current, cost_growth, p)
        f_provision = _pct_change(provision_current, provision_growth, p)
        f_income_rha = f_income_rt * f_yield
        f_exp = f_cash + f_provision
        f_net = f_income_rha - f_exp

        i_low_rt = _pct_change(income_low, income_growth, p) if income_low is not None else None
        i_high_rt = _pct_change(income_high, income_growth, p) if income_high is not None else None
        i_y_low = _pct_change(yield_low, yield_growth, p) if yield_low is not None else None
        i_y_high = _pct_change(yield_high, yield_growth, p) if yield_high is not None else None
        i_cash = _pct_change(industry_cash_current, cost_growth, p)
        i_provision = _pct_change(provision_current, provision_growth, p)
        i_exp = i_cash + i_provision
        i_income_low = i_low_rt * i_y_low if i_low_rt is not None and i_y_low is not None else None
        i_income_high = i_high_rt * i_y_high if i_high_rt is not None and i_y_high is not None else None
        i_net_low = i_income_low - i_exp if i_income_low is not None else None
        i_net_high = i_income_high - i_exp if i_income_high is not None else None

        sc = scenario_values(
            scenario_key,
            base_costs.assign(Avg=base_costs["Avg"].apply(lambda v: _pct_change(v, cost_growth, p))),
            i_low_rt,
            i_high_rt,
            i_y_low,
            i_y_high,
            i_provision,
        )

        detailed_rows.append(
            {
                "Year": yr,
                "Farmer Gross Income (R/t)": round(f_income_rt, 2),
                "Farmer Yield (t/ha)": round(f_yield, 2),
                "Farmer Income (R/ha)": round(f_income_rha, 2),
                "Farmer Total Expenditure (R/ha)": round(f_exp, 2),
                "Farmer Net Farm Income (R/ha)": round(f_net, 2),
                "Industry Net Low (R/ha)": None if i_net_low is None else round(i_net_low, 2),
                "Industry Net High (R/ha)": None if i_net_high is None else round(i_net_high, 2),
                f"{scenario_label} Net Low (R/ha)": None if sc["net_low"] is None else round(sc["net_low"], 2),
                f"{scenario_label} Net High (R/ha)": None if sc["net_high"] is None else round(sc["net_high"], 2),
            }
        )

    detailed_df = pd.DataFrame(detailed_rows)

    chart_df = detailed_df.set_index("Year")[
        [
            "Farmer Net Farm Income (R/ha)",
            "Industry Net Low (R/ha)",
            "Industry Net High (R/ha)",
            f"{scenario_label} Net Low (R/ha)",
            f"{scenario_label} Net High (R/ha)",
        ]
    ]
    st.line_chart(chart_df, width="stretch")

    detail_formatters = {
        "Farmer Gross Income (R/t)": _fmt_money,
        "Farmer Yield (t/ha)": lambda x: _fmt_num(x, 2),
        "Farmer Income (R/ha)": _fmt_money,
        "Farmer Total Expenditure (R/ha)": _fmt_money,
        "Farmer Net Farm Income (R/ha)": _fmt_money,
        "Industry Net Low (R/ha)": _fmt_money,
        "Industry Net High (R/ha)": _fmt_money,
        f"{scenario_label} Net Low (R/ha)": _fmt_money,
        f"{scenario_label} Net High (R/ha)": _fmt_money,
    }

    st.dataframe(
        detailed_df.style.format(detail_formatters),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download Page 2B detailed forecast as CSV",
        data=detailed_df.to_csv(index=False).encode("utf-8"),
        file_name=f"Phase2B_year_by_year_{region}_{base_year}_{target_year}.csv",
        mime="text/csv",
    )

st.info(
    "Page 2A retains the full detailed forecast and adds Phase-1-style forecast visuals. Page 2B keeps the complete year-by-year pathway."
)
