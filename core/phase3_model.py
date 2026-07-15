"""Pure calculation helpers for the Phase 3 vineyard scenario page.

The model deliberately uses a bounded propagation sequence.  It never loops
until convergence and it does not treat the separately estimated pairwise
coefficients as a jointly estimated simultaneous system.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log1p
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


CASE_COEFFICIENTS = {
    "Lower response": "ci_2_5",
    "Central response": "posterior_mean",
    "Upper response": "ci_97_5",
}


@dataclass(frozen=True)
class AccountSpec:
    key: str
    item: str
    label: str


# These are the coefficient accounts that map unambiguously to a single
# Phase 1 cost line.  Aggregate coefficient concepts such as vineyards, fixed
# improvements and moveable production assets are intentionally not forced
# onto a Phase 1 line because doing so would double count existing items.
ACCOUNT_SPECS = (
    AccountSpec("crop_protection", "Gewasbeskerming (swam- en insekbeheer)", "Crop protection"),
    AccountSpec("repair_binding_material", "Herstel- en bindmateriaal", "Repair and binding material"),
    AccountSpec("fertilizer", "Kunsmis, blaar- en grondontledings", "Fertiliser"),
    AccountSpec("herbicide_control", "Onkruiddoder", "Herbicide control"),
    AccountSpec("organic_material", "Saad, organiese bemesting en materiaal", "Seed and organic material"),
    AccountSpec("permanent_labour", "Permanente arbeid", "Permanent labour"),
    AccountSpec("seasonal_labour_contract_work", "Seisoensarbeid en kontrakwerk", "Seasonal labour and contract work"),
    AccountSpec("supervision", "Toesig en bestuurshulp", "Supervision and management assistance"),
    AccountSpec("fuel", "Brandstof (petrol en diesel) en smeermiddels", "Fuel and lubricants"),
    AccountSpec("repair_parts_maintenance", "Herstel. onderdele en reparasies", "Machinery repairs, parts and maintenance"),
    AccountSpec("hired_transport", "Vervoer gehuur", "Hired transport"),
    AccountSpec("repairs_and_maintenance", "Herstelwerk en onderhoud (vaste verbeteringe)", "Fixed-improvement repairs and maintenance"),
    AccountSpec("administration", "Administrasie", "Administration"),
    AccountSpec("electricity", "Elektrisiteit", "Electricity"),
    AccountSpec("water_costs", "Waterkoste / belasting", "Water costs"),
)

MODEL_KEY_TO_SPEC = {spec.key: spec for spec in ACCOUNT_SPECS}

# The Phase 1 source combines seed and organic material in one cost line.  The
# yield coefficient data never use both for the same area/cultivar target, so
# seed can safely point to the combined organic-material account for yield.
COEFFICIENT_KEY_ALIASES = {"seed": "organic_material"}


def normalise_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalise_cultivar(value: Any) -> str:
    cultivar = normalise_text(value)
    aliases = {
        "colombar": "colombard",
        "syrah": "shiraz",
    }
    return aliases.get(cultivar, cultivar)


def normalise_area(value: Any) -> str:
    """Match source variants such as OLIFANTSRIVER and Olifants River."""
    area = normalise_text(value).replace(" ", "")
    aliases = {
        "olifantsrivier": "olifantsriver",
    }
    return aliases.get(area, area)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def prepare_farmer_costs(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    costs = pd.DataFrame(list(records))
    for column in ("Category", "Item", "Cost"):
        if column not in costs.columns:
            costs[column] = "" if column != "Cost" else 0.0
    costs = costs[["Category", "Item", "Cost"]].copy()
    costs["Category"] = costs["Category"].astype(str).str.strip()
    costs["Item"] = costs["Item"].astype(str).str.strip()
    costs["Cost"] = pd.to_numeric(costs["Cost"], errors="coerce").fillna(0.0)
    return costs


def available_accounts(costs: pd.DataFrame) -> list[dict[str, Any]]:
    item_lookup = {normalise_text(item): i for i, item in enumerate(costs["Item"].tolist())}
    accounts: list[dict[str, Any]] = []
    for spec in ACCOUNT_SPECS:
        index = item_lookup.get(normalise_text(spec.item))
        if index is None:
            continue
        row = costs.iloc[index]
        accounts.append(
            {
                "key": spec.key,
                "label": spec.label,
                "item": spec.item,
                "category": row["Category"],
                "baseline_cost": _safe_float(row["Cost"]),
                "order": index,
            }
        )
    return sorted(accounts, key=lambda account: account["order"])


def _account_baselines(costs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {account["key"]: account for account in available_accounts(costs)}


def clean_relationship_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    for column in ("posterior_mean", "posterior_median", "ci_2_5", "ci_97_5", "n_obs", "start_year", "end_year"):
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def _area_rows(df: pd.DataFrame, area_column: str, area: str) -> pd.DataFrame:
    wanted = normalise_area(area)
    return df[df[area_column].map(normalise_area) == wanted].copy()


def matrix_diagnostic(input_relationships: pd.DataFrame, area: str) -> dict[str, Any]:
    """Test the full posterior-mean area matrix without claiming joint validity."""
    rows = _area_rows(input_relationships, "area", area)
    rows = rows.dropna(subset=["target_cost_key", "predictor_cost_key", "posterior_mean"])
    nodes = sorted(set(rows["target_cost_key"].astype(str)) | set(rows["predictor_cost_key"].astype(str)))
    if not nodes:
        return {
            "area": area,
            "nodes": 0,
            "edges": 0,
            "spectral_radius": None,
            "condition_number": None,
            "invertible": False,
            "iterative_stable": False,
        }

    positions = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for row in rows.itertuples(index=False):
        target = str(row.target_cost_key)
        predictor = str(row.predictor_cost_key)
        matrix[positions[target], positions[predictor]] = _safe_float(row.posterior_mean)

    system = np.eye(len(nodes)) - matrix
    eigenvalues = np.linalg.eigvals(matrix)
    radius = float(np.max(np.abs(eigenvalues)))
    rank = int(np.linalg.matrix_rank(system))
    invertible = rank == len(nodes)
    condition = float(np.linalg.cond(system)) if invertible else float("inf")
    return {
        "area": area,
        "nodes": len(nodes),
        "edges": len(rows),
        "spectral_radius": radius,
        "condition_number": condition,
        "invertible": invertible,
        "iterative_stable": bool(radius < 1.0),
    }


def _propagate_cost_changes(
    input_relationships: pd.DataFrame,
    area: str,
    coefficient_column: str,
    selected_key: str,
    selected_change: float,
    rounds: int,
    supported_keys: set[str],
) -> tuple[dict[str, float], pd.DataFrame, dict[str, int]]:
    rows = _area_rows(input_relationships, "area", area)
    rows = rows.dropna(subset=["target_cost_key", "predictor_cost_key", coefficient_column]).copy()
    rows = rows[
        rows["target_cost_key"].astype(str).isin(supported_keys)
        & rows["predictor_cost_key"].astype(str).isin(supported_keys)
    ].copy()

    current = {key: 0.0 for key in supported_keys}
    current[selected_key] = selected_change
    total = dict(current)
    audit_rows: list[dict[str, Any]] = []

    for round_number in range(1, int(rounds) + 1):
        next_round = {key: 0.0 for key in supported_keys}
        for row in rows.itertuples(index=False):
            predictor = str(row.predictor_cost_key)
            target = str(row.target_cost_key)
            predictor_change = current.get(predictor, 0.0)
            if abs(predictor_change) < 1e-14:
                continue
            coefficient = _safe_float(getattr(row, coefficient_column))
            contribution = coefficient * predictor_change
            if abs(contribution) < 1e-14:
                continue
            next_round[target] += contribution
            audit_rows.append(
                {
                    "Round": round_number,
                    "Predictor key": predictor,
                    "Predictor": MODEL_KEY_TO_SPEC[predictor].label,
                    "Target key": target,
                    "Target": MODEL_KEY_TO_SPEC[target].label,
                    "Coefficient": coefficient,
                    "Predictor log change": predictor_change,
                    "Target log contribution": contribution,
                }
            )
        for key, contribution in next_round.items():
            total[key] = total.get(key, 0.0) + contribution
        current = next_round

    coverage = {
        "area_relationships": len(_area_rows(input_relationships, "area", area)),
        "mapped_relationships": len(rows),
    }
    return total, pd.DataFrame(audit_rows), coverage


def _adjust_cost_table(costs: pd.DataFrame, total_log_changes: dict[str, float]) -> pd.DataFrame:
    adjusted = costs.copy()
    item_to_key = {normalise_text(spec.item): spec.key for spec in ACCOUNT_SPECS}
    adjusted["Model key"] = adjusted["Item"].map(lambda value: item_to_key.get(normalise_text(value), ""))
    adjusted["Baseline cost (R/ha)"] = adjusted["Cost"]
    adjusted["Scenario cost (R/ha)"] = adjusted.apply(
        lambda row: row["Cost"]
        * exp(float(np.clip(total_log_changes.get(row["Model key"], 0.0), -20.0, 20.0))),
        axis=1,
    )
    adjusted["Change (R/ha)"] = adjusted["Scenario cost (R/ha)"] - adjusted["Baseline cost (R/ha)"]
    adjusted["Change (%)"] = np.where(
        adjusted["Baseline cost (R/ha)"].abs() > 1e-12,
        adjusted["Change (R/ha)"] / adjusted["Baseline cost (R/ha)"] * 100.0,
        0.0,
    )
    return adjusted


def _yield_response(
    yield_relationships: pd.DataFrame,
    area: str,
    cultivar: str,
    coefficient_column: str,
    total_log_changes: dict[str, float],
    baseline_yield: float,
    industry_yield_cap: float | None,
) -> tuple[float, float, bool, pd.DataFrame]:
    rows = _area_rows(yield_relationships, "target_area", area)
    wanted_cultivar = normalise_cultivar(cultivar)
    rows = rows[rows["target_cultivar"].map(normalise_cultivar) == wanted_cultivar].copy()
    rows = rows.dropna(subset=["predictor_input_key", coefficient_column])

    audit_rows: list[dict[str, Any]] = []
    total_yield_log_change = 0.0
    for row in rows.itertuples(index=False):
        source_key = str(row.predictor_input_key)
        model_key = COEFFICIENT_KEY_ALIASES.get(source_key, source_key)
        predictor_change = total_log_changes.get(model_key, 0.0)
        if abs(predictor_change) < 1e-14:
            continue
        coefficient = _safe_float(getattr(row, coefficient_column))
        contribution = coefficient * predictor_change
        total_yield_log_change += contribution
        audit_rows.append(
            {
                "Predictor key": source_key,
                "Predictor": str(row.predictor_input_label),
                "Coefficient": coefficient,
                "Predictor log change": predictor_change,
                "Yield log contribution": contribution,
                "Observations": int(row.n_obs) if not pd.isna(row.n_obs) else None,
                "Data period": (
                    f"{int(row.start_year)}–{int(row.end_year)}"
                    if not pd.isna(row.start_year) and not pd.isna(row.end_year)
                    else ""
                ),
            }
        )

    raw_yield = baseline_yield * exp(float(np.clip(total_yield_log_change, -20.0, 20.0)))
    adjusted_yield = raw_yield
    cap_applied = False
    if industry_yield_cap is not None and raw_yield > baseline_yield:
        effective_cap = max(float(baseline_yield), float(industry_yield_cap))
        if raw_yield > effective_cap:
            adjusted_yield = effective_cap
            cap_applied = True
    return adjusted_yield, raw_yield, cap_applied, pd.DataFrame(audit_rows)


def calculate_phase3_scenario(
    *,
    farmer_costs: pd.DataFrame,
    provision_for_renewal: float,
    grape_price_per_tonne: float,
    baseline_yield: float,
    industry_yield_cap: float | None,
    area: str,
    cultivar: str,
    selected_key: str,
    adjustment_percent: float,
    rounds: int,
    input_relationships: pd.DataFrame,
    yield_relationships: pd.DataFrame,
) -> dict[str, Any]:
    if selected_key not in MODEL_KEY_TO_SPEC:
        raise ValueError(f"Unsupported Phase 3 account: {selected_key}")
    if adjustment_percent <= -100.0:
        raise ValueError("The selected cost adjustment must be greater than -100%.")
    if rounds not in (1, 2):
        raise ValueError("Phase 3 permits one or two bounded propagation rounds.")

    costs = prepare_farmer_costs(farmer_costs.to_dict("records"))
    account_baselines = _account_baselines(costs)
    if selected_key not in account_baselines:
        raise ValueError("The selected coefficient account is not available in the Phase 1 cost lines.")

    input_relationships = clean_relationship_data(input_relationships)
    yield_relationships = clean_relationship_data(yield_relationships)
    supported_keys = set(account_baselines)
    selected_log_change = log1p(float(adjustment_percent) / 100.0)

    baseline_cash_cost = float(costs["Cost"].sum())
    baseline_total_cost = baseline_cash_cost + float(provision_for_renewal)
    baseline_revenue = float(grape_price_per_tonne) * float(baseline_yield)
    baseline_nfi = baseline_revenue - baseline_total_cost
    selected_baseline_cost = account_baselines[selected_key]["baseline_cost"]
    direct_cost_change = selected_baseline_cost * (exp(selected_log_change) - 1.0)

    summary_rows = [
        {
            "Case": "Phase 1 baseline",
            "Selected input cost (R/ha)": selected_baseline_cost,
            "Associated cost change (R/ha)": 0.0,
            "Yield (t/ha)": float(baseline_yield),
            "Revenue (R/ha)": baseline_revenue,
            "Total cost (R/ha)": baseline_total_cost,
            "NFI (R/ha)": baseline_nfi,
            "Change in NFI (R/ha)": 0.0,
            "Yield cap applied": False,
        }
    ]
    cost_details: list[pd.DataFrame] = []
    propagation_details: list[pd.DataFrame] = []
    yield_details: list[pd.DataFrame] = []
    coverage_by_case: dict[str, dict[str, int]] = {}

    for case, coefficient_column in CASE_COEFFICIENTS.items():
        total_log_changes, propagation_audit, coverage = _propagate_cost_changes(
            input_relationships=input_relationships,
            area=area,
            coefficient_column=coefficient_column,
            selected_key=selected_key,
            selected_change=selected_log_change,
            rounds=rounds,
            supported_keys=supported_keys,
        )
        adjusted_costs = _adjust_cost_table(costs, total_log_changes)
        scenario_cash_cost = float(adjusted_costs["Scenario cost (R/ha)"].sum())
        scenario_total_cost = scenario_cash_cost + float(provision_for_renewal)
        selected_scenario_cost = float(
            adjusted_costs.loc[
                adjusted_costs["Model key"] == selected_key, "Scenario cost (R/ha)"
            ].sum()
        )
        associated_cost_change = scenario_total_cost - baseline_total_cost - direct_cost_change

        adjusted_yield, raw_yield, cap_applied, yield_audit = _yield_response(
            yield_relationships=yield_relationships,
            area=area,
            cultivar=cultivar,
            coefficient_column=coefficient_column,
            total_log_changes=total_log_changes,
            baseline_yield=float(baseline_yield),
            industry_yield_cap=industry_yield_cap,
        )
        revenue = float(grape_price_per_tonne) * adjusted_yield
        nfi = revenue - scenario_total_cost
        summary_rows.append(
            {
                "Case": case,
                "Selected input cost (R/ha)": selected_scenario_cost,
                "Associated cost change (R/ha)": associated_cost_change,
                "Yield (t/ha)": adjusted_yield,
                "Revenue (R/ha)": revenue,
                "Total cost (R/ha)": scenario_total_cost,
                "NFI (R/ha)": nfi,
                "Change in NFI (R/ha)": nfi - baseline_nfi,
                "Yield cap applied": cap_applied,
                "Raw yield before cap (t/ha)": raw_yield,
            }
        )

        adjusted_costs.insert(0, "Case", case)
        cost_details.append(adjusted_costs)
        if not propagation_audit.empty:
            propagation_audit.insert(0, "Case", case)
            propagation_details.append(propagation_audit)
        if not yield_audit.empty:
            yield_audit.insert(0, "Case", case)
            yield_details.append(yield_audit)
        coverage_by_case[case] = coverage

    return {
        "summary": pd.DataFrame(summary_rows),
        "cost_details": pd.concat(cost_details, ignore_index=True) if cost_details else pd.DataFrame(),
        "propagation_details": (
            pd.concat(propagation_details, ignore_index=True) if propagation_details else pd.DataFrame()
        ),
        "yield_details": pd.concat(yield_details, ignore_index=True) if yield_details else pd.DataFrame(),
        "matrix_diagnostic": matrix_diagnostic(input_relationships, area),
        "coverage": coverage_by_case,
        "selected_account": account_baselines[selected_key],
        "direct_cost_change": direct_cost_change,
        "industry_yield_cap": industry_yield_cap,
    }
