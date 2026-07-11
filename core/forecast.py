from collections.abc import Iterable

import pandas as pd

from core.scenarios import (
    SCENARIOS,
    SCENARIO_ORDER,
    apply_cost_rules,
    scenario_yield_band,
)


BASE_YEAR = 2025
FORECAST_YEARS = list(range(2025, 2031))
PHASE1_PROVISION_BASE = 20125.0


def compound_value(
    base_value: float, growth_percent: float, year: int, base_year: int = BASE_YEAR
) -> float:
    """Compound a 2025 value to the selected forecast year."""
    periods = int(year) - int(base_year)
    return float(base_value) * (1.0 + float(growth_percent) / 100.0) ** periods


def _matches(series: pd.Series, value: str) -> pd.Series:
    return series.astype(str).str.strip().str.casefold() == str(value).strip().casefold()


def _income_profile(
    income: pd.DataFrame,
    wine_class: str,
    grape_variety: str,
    region: str,
) -> pd.DataFrame:
    profile = income[
        income["YEAR"].eq(BASE_YEAR)
        & _matches(income["Wine Class"], wine_class)
        & _matches(income["Grape Variety"], grape_variety)
        & _matches(income["Region"], region)
    ].copy()
    if profile.empty:
        fallback_grape = "Other White" if wine_class.casefold() == "white" else "Other Red"
        profile = income[
            income["YEAR"].eq(BASE_YEAR)
            & _matches(income["Wine Class"], wine_class)
            & _matches(income["Grape Variety"], fallback_grape)
            & _matches(income["Region"], region)
        ].copy()
    return profile


def _yield_value(
    yield_df: pd.DataFrame,
    wine_class: str,
    grape_variety: str,
    region: str,
    band: str,
) -> float:
    mask = (
        _matches(yield_df["Wine Class"], wine_class)
        & _matches(yield_df["Grape Variety"], grape_variety)
        & _matches(yield_df["Region"], region)
        & _matches(yield_df["Band"], band)
    )
    values = yield_df.loc[mask, "Yield_t_per_ha"].dropna()
    if values.empty:
        fallback_grape = "Other White" if wine_class.casefold() == "white" else "Other Red"
        fallback_mask = (
            _matches(yield_df["Wine Class"], wine_class)
            & _matches(yield_df["Grape Variety"], fallback_grape)
            & _matches(yield_df["Region"], region)
            & _matches(yield_df["Band"], band)
        )
        values = yield_df.loc[fallback_mask, "Yield_t_per_ha"].dropna()
    if values.empty:
        raise ValueError(
            f"No {band.lower()} yield was found for {grape_variety}, {region}."
        )
    return float(values.iloc[0])


def _cost_profile(costs: pd.DataFrame, region: str) -> pd.DataFrame:
    profile = costs[costs["Year"].eq(BASE_YEAR) & _matches(costs["Region"], region)].copy()
    total_mask = _matches(profile["Category"], "Total Expenditure") | _matches(
        profile["Item"], "Total Expenditure"
    )
    profile = profile.loc[~total_mask].copy()
    profile = profile.rename(
        columns={
            "Avg_Cost": "Scenario_Base_2025",
            "model_ready_blended_growth_percent": "Growth_Percent",
        }
    )
    profile["Growth_Percent"] = pd.to_numeric(
        profile["Growth_Percent"], errors="coerce"
    ).fillna(0.0)

    # Phase 1 uses a common R20,125 provision. Keep that base consistent while
    # applying the region-specific provision growth coefficient in Phase 2.
    provision_mask = _matches(profile["Item"], "Provision for renewal")
    profile.loc[provision_mask, "Scenario_Base_2025"] = PHASE1_PROVISION_BASE
    return profile[["Region", "Category", "Item", "Scenario_Base_2025", "Growth_Percent"]]


def build_forecast(
    income: pd.DataFrame,
    costs: pd.DataFrame,
    yield_df: pd.DataFrame,
    wine_class: str,
    grape_variety: str,
    region: str,
    years: Iterable[int] = FORECAST_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scenario summaries and component-level cost forecasts."""
    years = [int(year) for year in years]
    income_profile = _income_profile(income, wine_class, grape_variety, region)
    if income_profile.empty:
        raise ValueError(
            f"No 2025 income profile was found for {grape_variety}, {region}."
        )

    income_by_band = {}
    for band in ("Low", "High"):
        row = income_profile[_matches(income_profile["Band"], band)]
        if row.empty:
            raise ValueError(f"The income profile does not contain a {band.lower()} band.")
        income_by_band[band] = {
            "base": float(row.iloc[0]["Income_R_per_t"]),
            "growth": float(row.iloc[0]["average_annual_growth_percent"]),
        }

    yield_low = _yield_value(yield_df, wine_class, grape_variety, region, "Low")
    yield_high = _yield_value(yield_df, wine_class, grape_variety, region, "High")
    cost_profile = _cost_profile(costs, region)
    if cost_profile.empty:
        raise ValueError(f"No 2025 cost profile was found for {region}.")

    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []

    for scenario_name in SCENARIO_ORDER:
        config = SCENARIOS[scenario_name]
        scenario_costs = cost_profile.copy()
        scenario_costs = apply_cost_rules(scenario_costs, config["cost_rules"])
        scenario_yield_low, scenario_yield_high = scenario_yield_band(
            yield_low, yield_high, scenario_name
        )

        for year in years:
            detail = scenario_costs.copy()
            detail["Forecast_Cost"] = detail.apply(
                lambda row: compound_value(
                    row["Scenario_Base_2025"], row["Growth_Percent"], year
                ),
                axis=1,
            )
            detail.insert(0, "Scenario", scenario_name)
            detail.insert(1, "Year", year)
            detail_frames.append(detail)

            total_cost = float(detail["Forecast_Cost"].sum())
            income_low_rt = compound_value(
                income_by_band["Low"]["base"] + float(config["income_abs"]),
                income_by_band["Low"]["growth"],
                year,
            )
            income_high_rt = compound_value(
                income_by_band["High"]["base"] + float(config["income_abs"]),
                income_by_band["High"]["growth"],
                year,
            )
            income_low_rha = income_low_rt * scenario_yield_low
            income_high_rha = income_high_rt * scenario_yield_high
            nfi_low = income_low_rha - total_cost
            nfi_high = income_high_rha - total_cost

            summary_rows.append(
                {
                    "Scenario": scenario_name,
                    "Year": year,
                    "Income_R_per_t_Low": income_low_rt,
                    "Income_R_per_t_High": income_high_rt,
                    "Yield_t_per_ha_Low": scenario_yield_low,
                    "Yield_t_per_ha_High": scenario_yield_high,
                    "Income_R_per_ha_Low": income_low_rha,
                    "Income_R_per_ha_High": income_high_rha,
                    "Total_Cost_R_per_ha": total_cost,
                    "NFI_R_per_ha_Low": nfi_low,
                    "NFI_R_per_ha_High": nfi_high,
                    "NFI_R_per_ha_Midpoint": (nfi_low + nfi_high) / 2.0,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["Scenario"] = pd.Categorical(
        summary["Scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    summary = summary.sort_values(["Scenario", "Year"]).reset_index(drop=True)

    cost_detail = pd.concat(detail_frames, ignore_index=True)
    cost_detail["Scenario"] = pd.Categorical(
        cost_detail["Scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    return summary, cost_detail
