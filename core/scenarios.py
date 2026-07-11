import pandas as pd


SCENARIOS = {
    "Industry baseline": {
        "label": "Industry baseline",
        "income_abs": 0.0,
        "yield_low_pct": 0.0,
        "yield_high_pct": 0.0,
        "cost_rules": [],
    },
    "Organic": {
        "label": "Organic",
        "income_abs": 0.0,
        "diff_multiplier": 1.41,
        "cost_rules": [
            ("item", "Saad, organiese bemesting en materiaal", "set", 13349),
            ("item", "Gewasbeskerming (swam- en insekbeheer)", "set", 1053),
            ("category", "Meganisasie", "abs", 110),
            ("item", "Administrasie", "abs", 555),
            ("item", "Kunsmis, blaar- en grondontledings", "set", 0),
            ("item", "Onkruiddoder", "set", 0),
        ],
    },
    "Fairtrade": {
        "label": "Fairtrade",
        "income_abs": 30.0,
        "yield_low_pct": 0.0,
        "yield_high_pct": 0.0,
        "cost_rules": [
            ("item", "Permanente arbeid", "pct", 7.9),
            ("item", "Seisoensarbeid en kontrakwerk", "pct", 7.9),
            ("item", "Administrasie", "abs", 1666),
        ],
    },
    "Organic + Fairtrade": {
        "label": "Organic + Fairtrade",
        "income_abs": 30.0,
        "diff_multiplier": 1.41,
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


SCENARIO_ORDER = list(SCENARIOS)


def apply_cost_rules(base_costs: pd.DataFrame, rules: list[tuple]) -> pd.DataFrame:
    """Apply the existing Phase 1 rules to 2025 cost components."""
    result = base_costs.copy()
    for rule_type, label, mode, adjustment in rules:
        column = "Category" if rule_type == "category" else "Item"
        mask = result[column].str.casefold() == str(label).strip().casefold()

        if mode == "set":
            result.loc[mask, "Scenario_Base_2025"] = float(adjustment)
        elif mode == "abs":
            result.loc[mask, "Scenario_Base_2025"] = (
                result.loc[mask, "Scenario_Base_2025"].fillna(0.0) + float(adjustment)
            )
        elif mode == "pct":
            result.loc[mask, "Scenario_Base_2025"] = (
                result.loc[mask, "Scenario_Base_2025"].fillna(0.0)
                * (1.0 + float(adjustment) / 100.0)
            )
    return result


def scenario_yield_band(
    yield_low: float, yield_high: float, scenario_name: str
) -> tuple[float, float]:
    config = SCENARIOS[scenario_name]
    difference_multiplier = config.get("diff_multiplier")
    if difference_multiplier is not None:
        difference = float(yield_high) - float(yield_low)
        return (
            float(yield_high) - difference * float(difference_multiplier),
            float(yield_high),
        )

    low_pct = float(config.get("yield_low_pct", 0.0))
    high_pct = float(config.get("yield_high_pct", 0.0))
    return (
        float(yield_low) * (1.0 + low_pct / 100.0),
        float(yield_high) * (1.0 + high_pct / 100.0),
    )
