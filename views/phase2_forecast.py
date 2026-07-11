from io import BytesIO

import pandas as pd
import streamlit as st

from core.data import load_forecast_data
from core.forecast import FORECAST_YEARS, build_forecast
from core.scenarios import SCENARIO_ORDER


@st.cache_data
def cached_forecast_data():
    return load_forecast_data()


def money(value: float) -> str:
    return f"R {value:,.2f}"


def forecast_excel(summary: pd.DataFrame, cost_detail: pd.DataFrame) -> bytes:
    output = BytesIO()
    try:
        engine = "xlsxwriter"
        __import__(engine)
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(output, engine=engine) as writer:
        summary.to_excel(writer, index=False, sheet_name="Scenario Forecast")
        cost_detail.to_excel(writer, index=False, sheet_name="Cost Detail")
    return output.getvalue()


st.title("Phase 2 — Outlook to 2030")
st.markdown(
    "<p style='color:#6b7280; margin-top:-10px;'><em>"
    "A coefficient-based outlook for income, costs and Net Farm Income."
    "</em></p>",
    unsafe_allow_html=True,
)

try:
    income_growth, cost_growth, yield_df = cached_forecast_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info("Copy yield.csv into the project root or the data folder, then rerun the app.")
    st.stop()

income_2025 = income_growth[income_growth["YEAR"].eq(2025)].copy()

with st.sidebar:
    st.markdown("---")
    st.header("Phase 2 Inputs")
    st.caption("Select the 2025 profile that will be projected to 2030.")

    wine_classes = sorted(income_2025["Wine Class"].dropna().unique().tolist())
    wine_class = st.selectbox(
        "Wine Class",
        wine_classes,
        index=wine_classes.index("White") if "White" in wine_classes else 0,
        key="phase2_wine_class",
    )

    grape_options = sorted(
        income_2025.loc[
            income_2025["Wine Class"].eq(wine_class), "Grape Variety"
        ].dropna().unique().tolist()
    )
    grape_variety = st.selectbox(
        "Grape Variety",
        grape_options,
        index=grape_options.index("Chenin Blanc") if "Chenin Blanc" in grape_options else 0,
        key="phase2_grape_variety",
    )

    region_options = sorted(
        income_2025.loc[
            income_2025["Wine Class"].eq(wine_class)
            & income_2025["Grape Variety"].eq(grape_variety),
            "Region",
        ].dropna().unique().tolist()
    )
    region = st.selectbox(
        "Region",
        region_options,
        index=region_options.index("BREEDEKLOOF") if "BREEDEKLOOF" in region_options else 0,
        key="phase2_region",
    )

    headline_year = st.selectbox(
        "Headline Year",
        FORECAST_YEARS[1:],
        index=len(FORECAST_YEARS[1:]) - 1,
        key="phase2_headline_year",
    )

try:
    summary, cost_detail = build_forecast(
        income_growth,
        cost_growth,
        yield_df,
        wine_class,
        grape_variety,
        region,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    f"**Selected profile:** {wine_class} · {grape_variety} · {region}"
)

st.subheader(f"{headline_year} Scenario Outlook")
headline = summary[summary["Year"].eq(headline_year)].copy()
headline["Scenario"] = headline["Scenario"].astype(str)
headline_display = headline[
    [
        "Scenario",
        "Income_R_per_ha_Low",
        "Income_R_per_ha_High",
        "Total_Cost_R_per_ha",
        "NFI_R_per_ha_Low",
        "NFI_R_per_ha_High",
    ]
].rename(
    columns={
        "Income_R_per_ha_Low": "Income Low (R/ha)",
        "Income_R_per_ha_High": "Income High (R/ha)",
        "Total_Cost_R_per_ha": "Total Cost (R/ha)",
        "NFI_R_per_ha_Low": "NFI Low (R/ha)",
        "NFI_R_per_ha_High": "NFI High (R/ha)",
    }
)
st.dataframe(
    headline_display.style.format(
        {column: money for column in headline_display.columns if column != "Scenario"}
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("Net Farm Income Path: 2025–2030")
chart_data = summary.pivot(
    index="Year", columns="Scenario", values="NFI_R_per_ha_Midpoint"
)
chart_data = chart_data[[scenario for scenario in SCENARIO_ORDER if scenario in chart_data]]
st.line_chart(chart_data, width="stretch")
st.caption(
    "The line chart uses the midpoint between the low and high NFI estimates. "
    "The detailed table below retains both boundaries."
)

st.subheader("Annual Scenario Detail")
annual_display = summary[
    [
        "Scenario",
        "Year",
        "Income_R_per_t_Low",
        "Income_R_per_t_High",
        "Total_Cost_R_per_ha",
        "NFI_R_per_ha_Low",
        "NFI_R_per_ha_High",
    ]
].copy()
annual_display["Scenario"] = annual_display["Scenario"].astype(str)
annual_display = annual_display.rename(
    columns={
        "Income_R_per_t_Low": "Income Low (R/t)",
        "Income_R_per_t_High": "Income High (R/t)",
        "Total_Cost_R_per_ha": "Total Cost (R/ha)",
        "NFI_R_per_ha_Low": "NFI Low (R/ha)",
        "NFI_R_per_ha_High": "NFI High (R/ha)",
    }
)
money_columns = [
    column for column in annual_display.columns if column not in {"Scenario", "Year"}
]
st.dataframe(
    annual_display.style.format({column: money for column in money_columns}),
    width="stretch",
    hide_index=True,
    height=480,
)

st.subheader("Detailed Cost Forecast")
detail_scenario = st.selectbox(
    "Scenario shown in the detailed cost table",
    SCENARIO_ORDER,
    index=0,
    key="phase2_detail_scenario",
)
selected_detail = cost_detail[
    cost_detail["Scenario"].astype(str).eq(detail_scenario)
].copy()
cost_pivot = selected_detail.pivot_table(
    index=["Category", "Item"],
    columns="Year",
    values="Forecast_Cost",
    aggfunc="first",
    sort=False,
).reset_index()
# Streamlit/PyArrow requires one consistent column-name type. The pivot creates
# text identifiers plus integer years, so make every column label a string.
cost_pivot.columns = [str(column) for column in cost_pivot.columns]
st.dataframe(
    cost_pivot.style.format(
        {str(year): money for year in FORECAST_YEARS if str(year) in cost_pivot.columns}
    ),
    width="stretch",
    hide_index=True,
    height=560,
)

download_left, download_right = st.columns(2)
with download_left:
    st.download_button(
        "Download forecast CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"VPT_Forecast_{region}_{grape_variety.replace(' ', '')}_2030.csv",
        mime="text/csv",
        width="stretch",
    )
with download_right:
    st.download_button(
        "Download forecast Excel",
        data=forecast_excel(summary, cost_detail),
        file_name=f"VPT_Forecast_{region}_{grape_variety.replace(' ', '')}_2030.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

with st.expander("Forecast method and assumptions", expanded=False):
    st.markdown(
        """
**Formula**

Forecast value = 2025 value × (1 + annual growth coefficient)^(year − 2025)

**Current Phase 2 assumptions**

- Income and each cost component are compounded annually using their supplied coefficients.
- Yield remains at its Phase 1 low/high benchmark because no yield-growth coefficient was supplied.
- Phase 1 scenario adjustments are applied to the 2025 base and then compounded using the relevant item coefficient.
- Provision for renewal starts from the Phase 1 value of R20,125 and uses the selected region's provision-growth coefficient.
- Total expenditure is recalculated from forecast components; it is not forecast as a separate total.
"""
    )
