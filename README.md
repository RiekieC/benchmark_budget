# Vineyard Production Tool — Multipage Version

This project preserves the completed Phase 1 benchmark dashboard and adds a working Phase 2 outlook to 2030. Phase 3 is included as a placeholder so it can be developed without restructuring the application again.

## Pages

- **Phase 1 — Current Benchmark:** the existing dashboard, scenario analysis, charts and exports.
- **Phase 2 — Outlook to 2030:** compound projections for income, cost components, total expenditure and Net Farm Income from 2025 to 2030.
- **Phase 3 — Future Analysis:** a reserved page for the next analytical layer.

## Required data

The following files must be in `data/` or in the project root:

- `income.csv`
- `costs.csv`
- `yield.csv`
- `income_2024_2025_with_average_annual_growth_percent.csv`
- `costs_2024_2025_with_blended_growth_percent.csv`

`yield.csv` is not included in this package because it was not attached with the Phase 2 source files. Copy the existing `yield.csv` from the current `benchmark_budget` project into the new `data` folder before launching the application.

## Run locally

Create or activate the virtual environment, install the dependencies, and start the master application:

```powershell
pip install -r requirements.txt
streamlit run .\app.py
```

The compatibility commands also work:

```powershell
streamlit run .\f1.py
streamlit run .\s7.py
```

## Forecast formula

For each year from 2026 to 2030:

```text
Forecast value = 2025 value × (1 + growth percentage / 100)^(year − 2025)
```

Income and cost components use their supplied annual coefficients. Total expenditure is recalculated from the forecast components. Yield remains at the Phase 1 low/high benchmark until a yield-growth assumption is supplied.

## Existing deployment

Both `f1.py` and `s7.py` now launch the master application. This allows the existing local command or an existing Streamlit Cloud main-file path to continue working after the full project is copied into the repository.
