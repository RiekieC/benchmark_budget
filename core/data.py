from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def resolve_data_file(filename: str) -> Path:
    """Resolve a data file from either data/ or the project root."""
    candidates = [DATA_DIR / filename, PROJECT_ROOT / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {searched}")


def _read_csv(filename: str) -> pd.DataFrame:
    df = pd.read_csv(resolve_data_file(filename), encoding="utf-8-sig")
    df.columns = [str(column).strip() for column in df.columns]
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()
    return df


def load_forecast_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the income, cost and yield inputs required by Phase 2."""
    income = _read_csv("income_2024_2025_with_average_annual_growth_percent.csv")
    costs = _read_csv("costs_2024_2025_with_blended_growth_percent.csv")
    yield_df = _read_csv("yield.csv")

    income["YEAR"] = pd.to_numeric(income["YEAR"], errors="coerce").astype("Int64")
    income["Income_R_per_t"] = pd.to_numeric(income["Income_R_per_t"], errors="coerce")
    income["average_annual_growth_percent"] = pd.to_numeric(
        income["average_annual_growth_percent"], errors="coerce"
    )
    income["Wine Class"] = income["Wine Class"].str.title()
    income["Band"] = income["Band"].str.title()

    costs["Year"] = pd.to_numeric(costs["Year"], errors="coerce").astype("Int64")
    costs["Avg_Cost"] = pd.to_numeric(costs["Avg_Cost"], errors="coerce").fillna(0.0)
    costs["model_ready_blended_growth_percent"] = pd.to_numeric(
        costs["model_ready_blended_growth_percent"], errors="coerce"
    )

    yield_column = "Yield_t_per_ha"
    if yield_column not in yield_df.columns:
        guesses = [column for column in yield_df.columns if "yield" in column.lower()]
        if not guesses:
            raise ValueError("yield.csv does not contain a yield column.")
        yield_df = yield_df.rename(columns={guesses[0]: yield_column})

    yield_df[yield_column] = pd.to_numeric(yield_df[yield_column], errors="coerce")
    median_yield = yield_df[yield_column].median()
    if pd.notna(median_yield) and median_yield > 1000:
        yield_df[yield_column] = yield_df[yield_column] / 1000.0
    yield_df["Wine Class"] = yield_df["Wine Class"].str.title()
    yield_df["Band"] = yield_df["Band"].str.title()

    return income, costs, yield_df
