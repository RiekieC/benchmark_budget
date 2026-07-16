"""Phase 3 — bounded cost, yield and financial scenario analysis."""

from __future__ import annotations

import base64
import os
import shutil
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    "green_pos": "#1CBA59",
    "red_neg": "#D64545",
    "industry_fill": "#E8EEF6",
    "industry_edge": "#8A93A6",
    "scenario_edge": "#9EA3AA",
    "zero_line": "#C7CBD1",
}

INDUSTRY_PROVISION = 20125.0


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


@st.cache_data
def load_industry_cost_benchmarks() -> pd.DataFrame:
    path = resolve_data_file("costs.csv", required=False)
    if path is None:
        return pd.DataFrame(columns=["Region", "Category", "Item", "Industry cost (R/ha)"])
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Avg_Cost" in frame.columns:
        frame = frame.rename(columns={"Avg_Cost": "Industry cost (R/ha)"})
    required = {"Region", "Category", "Item", "Industry cost (R/ha)"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=list(required))
    frame["Industry cost (R/ha)"] = pd.to_numeric(
        frame["Industry cost (R/ha)"], errors="coerce"
    ).fillna(0.0)
    return frame[["Region", "Category", "Item", "Industry cost (R/ha)"]].copy()


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


def format_range(low: Any, high: Any, *, monetary: bool = False, decimals: int = 2) -> str:
    try:
        if low is None or high is None or pd.isna(low) or pd.isna(high):
            return "—"
        if monetary:
            return f"{money(low)} / {money(high)}"
        return f"{number(low, decimals)} / {number(high, decimals)}"
    except (TypeError, ValueError):
        return "—"


def lookup_industry_costs(region: str) -> pd.DataFrame:
    frame = load_industry_cost_benchmarks()
    if frame.empty:
        return frame
    return frame[frame["Region"].map(normalise_text) == normalise_text(region)].copy()


def figure_to_png_bytes(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def build_benchmark_table_html(
    profile_rows: list[tuple[str, str]],
    rows: list[dict[str, str]],
    *,
    detailed: bool = False,
) -> str:
    if not rows:
        return ""
    columns = [column for column in rows[0] if column != "_class"]
    profile_html = "".join(
        f"<div class='p3-profile-cell'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
        for label, value in profile_rows
    )
    header_html = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body_rows: list[str] = []
    for row in rows:
        css_class = escape(str(row.get("_class", "")))
        cells = "".join(
            f"<td>{escape(str(row.get(column, '')))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr class='{css_class}'>{cells}</tr>")
    min_width = "980px" if detailed else "860px"
    return f"""
    <style>
      .p3-benchmark-wrap {{
        border:1px solid #e5e7eb; border-radius:9px; background:#fff;
        padding:12px; margin:4px 0 12px 0;
      }}
      .p3-profile-grid {{
        display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
        gap:8px; margin-bottom:12px;
      }}
      .p3-profile-cell {{background:#f8fafc;border:1px solid #e5e7eb;border-radius:7px;padding:7px 9px;}}
      .p3-profile-cell span {{display:block;color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:.03em;}}
      .p3-profile-cell strong {{display:block;color:#253247;font-size:13px;margin-top:2px;}}
      .p3-table-scroll {{overflow-x:auto;}}
      .p3-benchmark-table {{border-collapse:collapse;width:100%;min-width:{min_width};font-size:13px;}}
      .p3-benchmark-table th {{background:#253247;color:white;text-align:left;padding:9px 10px;white-space:normal;}}
      .p3-benchmark-table td {{border-bottom:1px solid #edf0f3;padding:7px 10px;vertical-align:top;}}
      .p3-benchmark-table td:first-child {{min-width:220px;white-space:pre-wrap;}}
      .p3-benchmark-table td:not(:first-child) {{text-align:right;}}
      .p3-benchmark-table .row-heading td {{background:#e4f1f2;color:#253247;font-weight:800;text-align:left;}}
      .p3-benchmark-table .row-category td {{background:#f8fafc;font-weight:700;}}
      .p3-benchmark-table .row-item td:first-child {{padding-left:24px;color:#4b5563;}}
      .p3-benchmark-table .row-total td {{background:#f8fafc;font-weight:700;}}
      .p3-benchmark-table .row-grand td {{background:#eef1f5;font-weight:800;}}
      @media print {{
        .p3-benchmark-wrap {{border:0;padding:0;}}
        .p3-table-scroll {{overflow:visible;}}
        .p3-benchmark-table {{min-width:0;font-size:9px;}}
        .p3-benchmark-table th,.p3-benchmark-table td {{padding:4px 5px;}}
        .p3-benchmark-table tr {{break-inside:avoid;}}
      }}
    </style>
    <div class="p3-benchmark-wrap">
      <div class="p3-profile-grid">{profile_html}</div>
      <div class="p3-table-scroll">
        <table class="p3-benchmark-table">
          <thead><tr>{header_html}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
      </div>
    </div>
    """


def distance_to_band(
    value: float | None,
    low: float | None,
    high: float | None,
) -> tuple[str, float]:
    if value is None or low is None or high is None:
        return "unknown", 0.0
    low, high = min(low, high), max(low, high)
    if low <= value <= high:
        return "inside", min(value - low, high - value)
    if value < low:
        return "below", low - value
    return "above", value - high


def status_chip(text: str, tone: str = "neutral") -> str:
    colours = {
        "good": ("#0f5132", "#d1e7dd", "#badbcc"),
        "bad": ("#842029", "#f8d7da", "#f5c2c7"),
        "warn": ("#664d03", "#fff3cd", "#ffe69c"),
        "info": ("#055160", "#cff4fc", "#b6effb"),
        "neutral": ("#374151", "#f3f4f6", "#e5e7eb"),
    }
    foreground, background, border = colours.get(tone, colours["neutral"])
    return (
        f'<span style="display:inline-block;padding:4px 10px;border:1px solid {border};'
        f'border-radius:999px;background:{background};color:{foreground};font-size:12px;line-height:1;">'
        f'{escape(text)}</span>'
    )


def distance_text(label: str, status: str, distance: float, unit: str) -> str:
    if status == "inside":
        return f"{label}: inside band"
    if status == "below":
        return f"{label}: {unit} {distance:,.2f} below"
    if status == "above":
        return f"{label}: {unit} {distance:,.2f} above"
    return f"{label}: n/a"


def build_phase3_range_plot(
    *,
    baseline_yield: float,
    baseline_nfi: float,
    central_yield: float,
    central_nfi: float,
    industry_yield_low: float | None,
    industry_yield_high: float | None,
    industry_nfi_low: float | None,
    industry_nfi_high: float | None,
    response_yield_low: float,
    response_yield_high: float,
    response_nfi_low: float,
    response_nfi_high: float,
):
    industry_available = all(
        value is not None
        for value in [industry_yield_low, industry_yield_high, industry_nfi_low, industry_nfi_high]
    )
    if industry_available:
        y0, y1 = sorted([float(industry_yield_low), float(industry_yield_high)])
        n0, n1 = sorted([float(industry_nfi_low), float(industry_nfi_high)])
    else:
        y0 = y1 = baseline_yield
        n0 = n1 = baseline_nfi

    sy0, sy1 = min(response_yield_low, central_yield, response_yield_high), max(
        response_yield_low, central_yield, response_yield_high
    )
    sn0, sn1 = min(response_nfi_low, central_nfi, response_nfi_high), max(
        response_nfi_low, central_nfi, response_nfi_high
    )
    point_colour = PALETTE["green_pos"] if central_nfi >= 0 else PALETTE["red_neg"]
    in_range = industry_available and y0 <= central_yield <= y1 and n0 <= central_nfi <= n1

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=150)
    if industry_available:
        ax.fill_between([y0, y1], n0, n1, step="pre", alpha=0.30, color=PALETTE["industry_fill"])
        ax.plot(
            [y0, y0, y1, y1, y0], [n0, n1, n1, n0, n0],
            color=PALETTE["industry_edge"], linewidth=1.1,
        )
    ax.plot(
        [sy0, sy0, sy1, sy1, sy0], [sn0, sn1, sn1, sn0, sn0],
        color=PALETTE["scenario_edge"], linewidth=1.4, linestyle="--",
    )
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=1.1)
    ax.scatter(
        baseline_yield, baseline_nfi, s=58, marker="X", color=PALETTE["grey"],
        edgecolor="white", linewidth=0.8, zorder=4,
    )
    ax.scatter(
        central_yield, central_nfi, s=68,
        facecolor=point_colour if in_range else "none",
        edgecolor=point_colour, linewidth=1.6, zorder=5,
    )
    ax.annotate("Phase 1", (baseline_yield, baseline_nfi), xytext=(5, 7), textcoords="offset points", fontsize=8)
    ax.annotate(
        "Phase 3 central", (central_yield, central_nfi), xytext=(5, 7),
        textcoords="offset points", fontsize=8.5, color=point_colour, weight="bold",
    )

    x_values = [baseline_yield, central_yield, sy0, sy1]
    y_values = [baseline_nfi, central_nfi, sn0, sn1]
    if industry_available:
        x_values.extend([y0, y1])
        y_values.extend([n0, n1])
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(0.1, (x_max - x_min) or 1.0)
    y_span = max(1.0, (y_max - y_min) or 1.0)
    ax.set_xlim(x_min - x_span * 0.18, x_max + x_span * 0.18)
    ax.set_ylim(y_min - y_span * 0.20, y_max + y_span * 0.20)
    ax.set_xlabel("Yield (t/ha)")
    ax.set_ylabel("Net Farm Income (R/ha)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"R {value:,.0f}"))
    ax.grid(alpha=0.13)

    handles = []
    if industry_available:
        handles.append(
            Patch(
                facecolor=PALETTE["industry_fill"], edgecolor=PALETTE["industry_edge"],
                label="Industry range",
            )
        )
    handles.extend(
        [
            Line2D([0], [0], color=PALETTE["scenario_edge"], linestyle="--", label="Phase 3 response range"),
            Line2D([0], [0], marker="X", linestyle="none", color=PALETTE["grey"], label="Phase 1 baseline"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=point_colour, label="Phase 3 central"),
        ]
    )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.20),
        ncol=2, frameon=True, fontsize=8,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    return fig


def build_phase3_summary_chart(
    *,
    baseline_row: pd.Series,
    lower_row: pd.Series,
    central_row: pd.Series,
    upper_row: pd.Series,
    industry_yield_low: float | None,
    industry_yield_high: float | None,
    industry_nfi_low: float | None,
    industry_nfi_high: float | None,
):
    items: list[tuple[str, float, float | None]] = [
        ("Phase 1", float(baseline_row["NFI (R/ha)"]), float(baseline_row["Yield (t/ha)"])),
        ("Lower response", float(lower_row["NFI (R/ha)"]), float(lower_row["Yield (t/ha)"])),
        ("Central response", float(central_row["NFI (R/ha)"]), float(central_row["Yield (t/ha)"])),
        ("Upper response", float(upper_row["NFI (R/ha)"]), float(upper_row["Yield (t/ha)"])),
    ]
    if industry_nfi_low is not None:
        items.append(("Industry Low", float(industry_nfi_low), industry_yield_low))
    if industry_nfi_high is not None:
        items.append(("Industry High", float(industry_nfi_high), industry_yield_high))

    labels = [label.replace(" response", "\nresponse").replace("Industry ", "Industry\n") for label, _, _ in items]
    nfi_values = [nfi for _, nfi, _ in items]
    yield_values = [yield_value for _, _, yield_value in items]
    colours = [PALETTE["grey"], "#8CB9BD", PALETTE["teal"], "#4D9BA2"]
    colours.extend(["#B0B8C4"] * max(0, len(items) - len(colours)))
    colours = colours[: len(items)]

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    positions = np.arange(len(items))
    ax.bar(positions, nfi_values, width=0.58, color=colours, alpha=0.92)
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=1.1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Net Farm Income (R/ha)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"R {value:,.0f}"))
    ax.grid(axis="y", alpha=0.16)

    ax2 = ax.twinx()
    for position, (yield_value, colour) in enumerate(zip(yield_values, colours)):
        if yield_value is not None:
            ax2.scatter(
                position, yield_value, s=58, marker="o", facecolor=colour,
                edgecolor="#1f2937", linewidth=0.8, zorder=5,
            )
    ax2.set_ylabel("Yield (t/ha)")
    ax2.grid(False)
    ax.legend(
        handles=[
            Patch(facecolor="#B0B8C4", edgecolor="#4b5563", label="NFI (bars)"),
            Line2D([0], [0], marker="o", linestyle="none", color="#1f2937", label="Yield (markers)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=8.5,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.78, bottom=0.18)
    return fig


def phase3_excel_bytes(
    summary_frame: pd.DataFrame,
    detail_frame: pd.DataFrame,
    profile_rows: list[tuple[str, str]],
    range_figure,
    summary_figure,
) -> bytes:
    output = BytesIO()
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(output, engine=engine) as writer:
        summary_frame.to_excel(writer, index=False, sheet_name="Scenario Summary")
        detail_frame.to_excel(writer, index=False, sheet_name="Detailed Costs")
        if engine == "xlsxwriter":
            workbook = writer.book
            header_format = workbook.add_format({"bold": True, "bg_color": "#253247", "font_color": "#FFFFFF"})
            for sheet_name, frame in [("Scenario Summary", summary_frame), ("Detailed Costs", detail_frame)]:
                worksheet = writer.sheets[sheet_name]
                for column_index, column_name in enumerate(frame.columns):
                    worksheet.write(0, column_index, column_name, header_format)
                worksheet.set_column(0, 0, 34)
                worksheet.set_column(1, len(frame.columns) - 1, 22)
            charts = workbook.add_worksheet("Charts")
            row = 0
            for label, value in profile_rows:
                charts.write(row, 0, label)
                charts.write(row, 1, value)
                row += 1
            row += 1
            charts.insert_image(row, 0, "phase3_range.png", {"image_data": BytesIO(figure_to_png_bytes(range_figure)), "x_scale": 0.75, "y_scale": 0.75})
            row += 23
            charts.insert_image(row, 0, "phase3_summary.png", {"image_data": BytesIO(figure_to_png_bytes(summary_figure)), "x_scale": 0.75, "y_scale": 0.75})
    return output.getvalue()


def wkhtmltopdf_configuration():
    try:
        import pdfkit
    except Exception:
        return None
    executable = shutil.which("wkhtmltopdf")
    if executable is None:
        for candidate in [
            r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
            r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
        ]:
            if os.path.exists(candidate):
                executable = candidate
                break
    if executable is None:
        return None
    return pdfkit, pdfkit.configuration(wkhtmltopdf=executable)


def build_printable_report(
    *,
    profile_rows: list[tuple[str, str]],
    summary_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
    range_figure,
    summary_figure,
) -> str:
    range_image = base64.b64encode(figure_to_png_bytes(range_figure)).decode("ascii")
    summary_image = base64.b64encode(figure_to_png_bytes(summary_figure)).decode("ascii")
    profile_line = " · ".join(f"{escape(label)}: {escape(value)}" for label, value in profile_rows)
    return f"""
    <!doctype html>
    <html><head><meta charset="utf-8"><title>Phase 3A Scenario Benchmark</title>
    <style>
      @page {{size:A4 landscape;margin:12mm;}}
      body {{font-family:Inter,Segoe UI,Arial,sans-serif;color:#253247;margin:18px;}}
      h1 {{margin-bottom:4px;}} h2 {{margin-top:22px;}}
      .meta {{color:#6b7280;margin-bottom:14px;}}
      .chart {{border:1px solid #e5e7eb;border-radius:8px;padding:8px;margin:10px 0;}}
      .chart img {{width:100%;height:auto;display:block;}}
      .page-break {{page-break-before:always;}}
    </style></head><body>
      <h1>Phase 3A — Scenario Benchmark</h1>
      <div class="meta">{profile_line}</div>
      <h2>Income, Costs and Summary</h2>
      {build_benchmark_table_html(profile_rows, summary_rows)}
      <div class="chart"><h2>Regional Competitiveness Map</h2><img src="data:image/png;base64,{range_image}"></div>
      <div class="chart"><h2>Regional Competitiveness Chart</h2><img src="data:image/png;base64,{summary_image}"></div>
      <div class="page-break"></div>
      <h2>Detailed Costs and Totals</h2>
      {build_benchmark_table_html(profile_rows, detail_rows, detailed=True)}
    </body></html>
    """


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
account_by_key = {account["key"]: account for account in selectable_accounts}

stored_adjustments = st.session_state.get("phase3_applied_adjustments", {})
if not isinstance(stored_adjustments, dict):
    stored_adjustments = {}
applied_adjustments = {
    key: float(value)
    for key, value in stored_adjustments.items()
    if key in account_keys
}
if not applied_adjustments:
    applied_adjustments = {default_key: 10.0}
    st.session_state["phase3_applied_adjustments"] = dict(applied_adjustments)

with st.sidebar:
    st.header("Phase 3 scenario")
    st.caption("Select one or more current Phase 1 costs and specify the change for each item.")
    selected_keys = st.multiselect(
        "Cost items to adjust",
        options=account_keys,
        default=list(applied_adjustments),
        format_func=lambda key: MODEL_KEY_TO_SPEC[key].label,
        key="phase3_selected_keys",
    )
    draft_adjustments: dict[str, float] = {}
    for key in selected_keys:
        account = account_by_key[key]
        draft_adjustments[key] = float(
            st.number_input(
                f"{MODEL_KEY_TO_SPEC[key].label} adjustment (%)",
                min_value=-90.0,
                max_value=100.0,
                value=float(applied_adjustments.get(key, 10.0)),
                step=1.0,
                format="%.1f",
                help=(
                    f"Phase 1 value: {money(account['baseline_cost'])}/ha. "
                    "A positive value increases this cost; a negative value reduces it."
                ),
                key=f"phase3_adjustment_{key}",
            )
        )

    run_scenario = st.button(
        "Run scenario",
        type="primary",
        width="stretch",
        disabled=not selected_keys,
    )
    if run_scenario:
        applied_adjustments = dict(draft_adjustments)
        st.session_state["phase3_applied_adjustments"] = dict(applied_adjustments)
    elif draft_adjustments != applied_adjustments:
        st.info("Selections changed — click **Run scenario** to update the results.")
    if not selected_keys:
        st.caption("Select at least one cost item to run a new scenario.")
    st.caption("Selected changes are calculated together. Grape price remains constant.")

benchmark_warnings: list[str] = []
for key, adjustment_percent in applied_adjustments.items():
    account = account_by_key[key]
    regional_change = lookup_cost_change_benchmark(base.get("region", ""), account["item"])
    if regional_change is not None and adjustment_percent > regional_change and adjustment_percent > 0:
        benchmark_warnings.append(
            f"{MODEL_KEY_TO_SPEC[key].label}: {adjustment_percent:.1f}% selected versus "
            f"a {regional_change:.1f}% 2025 area benchmark change"
        )
if benchmark_warnings:
    st.warning(
        "The following selected increases are above the available area benchmarks: "
        + "; ".join(benchmark_warnings)
        + ". The scenario has still been calculated."
    )

result = calculate_phase3_scenario(
    farmer_costs=farmer_costs,
    provision_for_renewal=float(base.get("provision_for_renewal", 0.0)),
    grape_price_per_tonne=float(base.get("farmer_income_rt", 0.0)),
    baseline_yield=float(base.get("farmer_yield", 0.0)),
    industry_yield_cap=industry_yield_midpoint,
    area=str(base.get("region", "")),
    cultivar=str(base.get("grape_variety", "")),
    adjustments=applied_adjustments,
    input_relationships=input_relationships,
    yield_relationships=yield_relationships,
)

summary = result["summary"]
baseline_row = summary.iloc[0]
lower_row = summary.loc[summary["Case"] == "Lower response"].iloc[0]
central_row = summary.loc[summary["Case"] == "Central response"].iloc[0]
upper_row = summary.loc[summary["Case"] == "Upper response"].iloc[0]
scenario_text = "; ".join(
    f"{MODEL_KEY_TO_SPEC[key].label} {value:+.1f}%"
    for key, value in applied_adjustments.items()
)

industry_costs = lookup_industry_costs(str(base.get("region", "")))
industry_cash_cost = (
    float(industry_costs["Industry cost (R/ha)"].sum())
    if not industry_costs.empty
    else None
)
industry_total_cost = (
    industry_cash_cost + INDUSTRY_PROVISION
    if industry_cash_cost is not None
    else None
)
industry_income_low = base.get("industry_income_rt_low")
industry_income_high = base.get("industry_income_rt_high")
industry_yield_low = base.get("industry_yield_low")
industry_yield_high = base.get("industry_yield_high")
industry_revenue_low = (
    float(industry_income_low) * float(industry_yield_low)
    if industry_income_low is not None and industry_yield_low is not None
    else None
)
industry_revenue_high = (
    float(industry_income_high) * float(industry_yield_high)
    if industry_income_high is not None and industry_yield_high is not None
    else None
)
industry_nfi_low = (
    industry_revenue_low - industry_total_cost
    if industry_revenue_low is not None and industry_total_cost is not None
    else None
)
industry_nfi_high = (
    industry_revenue_high - industry_total_cost
    if industry_revenue_high is not None and industry_total_cost is not None
    else None
)

profile_rows = [
    ("Year", str(base.get("year") or "2025")),
    ("Wine Class", str(base.get("wine_class", ""))),
    ("Grape Variety", str(base.get("grape_variety", ""))),
    ("Region", str(base.get("region", ""))),
    ("Selected Changes", scenario_text),
]

summary_columns = [
    "Section / Item",
    "Phase 1 Current",
    "Phase 3 Central Scenario",
    "2025 Industry Comparison",
    "Phase 3 Response Range (Lower / Upper)",
]


def benchmark_row(
    item: str,
    current: str = "",
    central: str = "",
    industry: str = "",
    response: str = "",
    css_class: str = "",
) -> dict[str, str]:
    return {
        summary_columns[0]: item,
        summary_columns[1]: current,
        summary_columns[2]: central,
        summary_columns[3]: industry,
        summary_columns[4]: response,
        "_class": css_class,
    }


baseline_cash_cost = float(baseline_row["Total cost (R/ha)"]) - float(base.get("provision_for_renewal", 0.0))
lower_cash_cost = float(lower_row["Total cost (R/ha)"]) - float(base.get("provision_for_renewal", 0.0))
central_cash_cost = float(central_row["Total cost (R/ha)"]) - float(base.get("provision_for_renewal", 0.0))
upper_cash_cost = float(upper_row["Total cost (R/ha)"]) - float(base.get("provision_for_renewal", 0.0))
grape_price = float(base.get("farmer_income_rt", 0.0))

benchmark_summary_rows = [
    benchmark_row("Vineyard Performance", css_class="row-heading"),
    benchmark_row(
        "Gross Income (R/t)", money(grape_price), money(grape_price),
        format_range(industry_income_low, industry_income_high, monetary=True),
        format_range(grape_price, grape_price, monetary=True),
    ),
    benchmark_row(
        "Yield (t/ha)", number(baseline_row["Yield (t/ha)"]), number(central_row["Yield (t/ha)"]),
        format_range(industry_yield_low, industry_yield_high),
        format_range(lower_row["Yield (t/ha)"], upper_row["Yield (t/ha)"]),
    ),
    benchmark_row(
        "Income (R/ha)", money(baseline_row["Revenue (R/ha)"]), money(central_row["Revenue (R/ha)"]),
        format_range(industry_revenue_low, industry_revenue_high, monetary=True),
        format_range(lower_row["Revenue (R/ha)"], upper_row["Revenue (R/ha)"], monetary=True),
    ),
    benchmark_row("Costs and Totals", css_class="row-heading"),
    benchmark_row(
        "Total cash costs (R/ha)", money(baseline_cash_cost), money(central_cash_cost),
        money(industry_cash_cost) if industry_cash_cost is not None else "—",
        format_range(lower_cash_cost, upper_cash_cost, monetary=True), "row-total",
    ),
    benchmark_row(
        "Direct selected-cost change (R/ha)", money(0), money(central_row["Direct cost change (R/ha)"]),
        "—", format_range(lower_row["Direct cost change (R/ha)"], upper_row["Direct cost change (R/ha)"], monetary=True),
    ),
    benchmark_row(
        "Associated cost change (R/ha)", money(0), money(central_row["Associated cost change (R/ha)"]),
        "—", format_range(lower_row["Associated cost change (R/ha)"], upper_row["Associated cost change (R/ha)"], monetary=True),
    ),
    benchmark_row(
        "Provision for renewal (R/ha)", money(base.get("provision_for_renewal", 0.0)),
        money(base.get("provision_for_renewal", 0.0)), money(INDUSTRY_PROVISION),
        format_range(base.get("provision_for_renewal", 0.0), base.get("provision_for_renewal", 0.0), monetary=True),
    ),
    benchmark_row(
        "Total Expenditure (R/ha)", money(baseline_row["Total cost (R/ha)"]), money(central_row["Total cost (R/ha)"]),
        money(industry_total_cost) if industry_total_cost is not None else "—",
        format_range(lower_row["Total cost (R/ha)"], upper_row["Total cost (R/ha)"], monetary=True), "row-total",
    ),
    benchmark_row(
        "Net Farm Income (R/ha)", money(baseline_row["NFI (R/ha)"]), money(central_row["NFI (R/ha)"]),
        format_range(industry_nfi_low, industry_nfi_high, monetary=True),
        format_range(lower_row["NFI (R/ha)"], upper_row["NFI (R/ha)"], monetary=True), "row-grand",
    ),
]

detail_cases = {
    case: result["cost_details"][result["cost_details"]["Case"] == case].copy()
    for case in ["Lower response", "Central response", "Upper response"]
}
scenario_cost_lookup: dict[str, dict[tuple[str, str], float]] = {}
for case, frame in detail_cases.items():
    scenario_cost_lookup[case] = {
        (normalise_text(row["Category"]), normalise_text(row["Item"])): float(row["Scenario cost (R/ha)"])
        for _, row in frame.iterrows()
    }
industry_cost_lookup = {
    (normalise_text(row["Category"]), normalise_text(row["Item"])): float(row["Industry cost (R/ha)"])
    for _, row in industry_costs.iterrows()
}

detail_rows: list[dict[str, str]] = []
for category in farmer_costs["Category"].drop_duplicates().tolist():
    category_rows = farmer_costs[farmer_costs["Category"] == category]
    keys = [
        (normalise_text(row["Category"]), normalise_text(row["Item"]))
        for _, row in category_rows.iterrows()
    ]
    current_total = float(pd.to_numeric(category_rows["Cost"], errors="coerce").fillna(0.0).sum())
    central_total = sum(scenario_cost_lookup["Central response"].get(key, 0.0) for key in keys)
    lower_total = sum(scenario_cost_lookup["Lower response"].get(key, 0.0) for key in keys)
    upper_total = sum(scenario_cost_lookup["Upper response"].get(key, 0.0) for key in keys)
    industry_total = sum(industry_cost_lookup.get(key, 0.0) for key in keys)
    detail_rows.append(
        benchmark_row(
            str(category), money(current_total), money(central_total), money(industry_total),
            format_range(lower_total, upper_total, monetary=True), "row-category",
        )
    )
    for _, item_row in category_rows.iterrows():
        key = (normalise_text(item_row["Category"]), normalise_text(item_row["Item"]))
        detail_rows.append(
            benchmark_row(
                f"  {item_row['Item']}", money(item_row["Cost"]),
                money(scenario_cost_lookup["Central response"].get(key, item_row["Cost"])),
                money(industry_cost_lookup.get(key, 0.0)),
                format_range(
                    scenario_cost_lookup["Lower response"].get(key, item_row["Cost"]),
                    scenario_cost_lookup["Upper response"].get(key, item_row["Cost"]),
                    monetary=True,
                ),
                "row-item",
            )
        )

detail_rows.extend(
    [
        benchmark_row(
            "Totale kontant uitgawes", money(baseline_cash_cost), money(central_cash_cost),
            money(industry_cash_cost) if industry_cash_cost is not None else "—",
            format_range(lower_cash_cost, upper_cash_cost, monetary=True), "row-total",
        ),
        benchmark_row(
            "Provision for renewal", money(base.get("provision_for_renewal", 0.0)),
            money(base.get("provision_for_renewal", 0.0)), money(INDUSTRY_PROVISION),
            format_range(base.get("provision_for_renewal", 0.0), base.get("provision_for_renewal", 0.0), monetary=True),
            "row-total",
        ),
        benchmark_row(
            "Total Expenditure", money(baseline_row["Total cost (R/ha)"]), money(central_row["Total cost (R/ha)"]),
            money(industry_total_cost) if industry_total_cost is not None else "—",
            format_range(lower_row["Total cost (R/ha)"], upper_row["Total cost (R/ha)"], monetary=True), "row-total",
        ),
        benchmark_row(
            "Net Farm Income (R/ha)", money(baseline_row["NFI (R/ha)"]), money(central_row["NFI (R/ha)"]),
            format_range(industry_nfi_low, industry_nfi_high, monetary=True),
            format_range(lower_row["NFI (R/ha)"], upper_row["NFI (R/ha)"], monetary=True), "row-grand",
        ),
    ]
)

benchmark_range_figure = build_phase3_range_plot(
    baseline_yield=float(baseline_row["Yield (t/ha)"]),
    baseline_nfi=float(baseline_row["NFI (R/ha)"]),
    central_yield=float(central_row["Yield (t/ha)"]),
    central_nfi=float(central_row["NFI (R/ha)"]),
    industry_yield_low=float(industry_yield_low) if industry_yield_low is not None else None,
    industry_yield_high=float(industry_yield_high) if industry_yield_high is not None else None,
    industry_nfi_low=industry_nfi_low,
    industry_nfi_high=industry_nfi_high,
    response_yield_low=float(lower_row["Yield (t/ha)"]),
    response_yield_high=float(upper_row["Yield (t/ha)"]),
    response_nfi_low=float(lower_row["NFI (R/ha)"]),
    response_nfi_high=float(upper_row["NFI (R/ha)"]),
)
benchmark_summary_figure = build_phase3_summary_chart(
    baseline_row=baseline_row,
    lower_row=lower_row,
    central_row=central_row,
    upper_row=upper_row,
    industry_yield_low=float(industry_yield_low) if industry_yield_low is not None else None,
    industry_yield_high=float(industry_yield_high) if industry_yield_high is not None else None,
    industry_nfi_low=industry_nfi_low,
    industry_nfi_high=industry_nfi_high,
)

with st.expander("Quick Guide", expanded=False):
    st.markdown(
        """
        1. Select one or more cost items in the sidebar and enter a percentage for each.
        2. Click **Run scenario**. The same combined scenario updates both Phase 3 pages.
        3. Use **Page 3A** for the complete client-facing benchmark, graphs and report downloads.
        4. Use **Page 3B** to inspect the response cases and detailed scenario calculations.
        """
    )

tab_a, tab_b = st.tabs(
    [
        "Page 3A — Scenario Benchmark",
        "Page 3B — Detailed Scenario Analysis",
    ]
)

with tab_a:
    st.subheader("Page 3A — Multi-Input Scenario Benchmark")
    st.caption(
        "Main client-facing view: the selected cost changes are calculated together and presented in the same benchmark format as Phases 1 and 2A."
    )
    st.markdown(
        f"""
        <div class="phase3-note">
          <strong>Current multi-input scenario:</strong> {escape(str(base.get('region', '')))} ·
          {escape(str(base.get('grape_variety', '')))} · {escape(scenario_text)}.<br>
          <span style="color:#5f6b7a;">This page uses the Phase 1 current-year position and does not use the Phase 2 forecast.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Phase 1 Net Farm Income", money(baseline_row["NFI (R/ha)"]))
    metric_columns[1].metric(
        "Phase 3 Central NFI", money(central_row["NFI (R/ha)"]),
        delta=money(central_row["Change in NFI (R/ha)"]),
    )
    metric_columns[2].metric(
        "Phase 3 Central Yield", f"{central_row['Yield (t/ha)']:.2f} t/ha",
        delta=f"{central_row['Yield (t/ha)'] - baseline_row['Yield (t/ha)']:+.2f} t/ha",
    )
    metric_columns[3].metric("Cost items adjusted", len(applied_adjustments))

    if result["yield_details"].empty:
        st.info(
            "No applicable yield relationship was found for the selected and linked cost changes for this area–cultivar combination. "
            "Costs and NFI are still recalculated, while yield remains at the Phase 1 level."
        )
    elif bool(summary["Yield cap applied"].fillna(False).any()):
        st.info(
            "At least one response reached the available 2025 yield benchmark. Further cost increases can still affect costs, "
            f"but the calculated yield is limited to {effective_yield_cap:.2f} t/ha."
        )

    st.markdown("### Income, Costs and Summary")
    st.caption(
        "Phase 1 current values, the central combined scenario, regional 2025 benchmarks and the lower-to-upper response range are shown together."
    )
    st.html(build_benchmark_table_html(profile_rows, benchmark_summary_rows))

    st.markdown("### Detailed Costs and Totals — Phase 1 to Phase 3 Scenario")
    st.caption(
        "The farmer's current line items are shown alongside the central multi-input scenario, regional industry values and the response range."
    )
    st.html(build_benchmark_table_html(profile_rows, detail_rows, detailed=True))

    st.markdown("### Regional Competitiveness Map — Yield and Net Farm Income")
    st.caption(
        "This shows where the central combined scenario sits relative to the Phase 1 baseline, the regional industry range and the Phase 3 response range."
    )
    yield_status, yield_distance = distance_to_band(
        float(central_row["Yield (t/ha)"]),
        float(industry_yield_low) if industry_yield_low is not None else None,
        float(industry_yield_high) if industry_yield_high is not None else None,
    )
    nfi_status, nfi_distance = distance_to_band(
        float(central_row["NFI (R/ha)"]), industry_nfi_low, industry_nfi_high,
    )
    central_profit = float(central_row["NFI (R/ha)"]) >= 0
    central_in_range = yield_status == "inside" and nfi_status == "inside"
    st.markdown(
        '<div style="margin:6px 0 2px 0;display:flex;gap:8px;flex-wrap:wrap;">'
        + status_chip(
            f"{'Profit' if central_profit else 'Loss'} · {money(central_row['NFI (R/ha)'])}/ha",
            "good" if central_profit else "bad",
        )
        + status_chip("In industry range" if central_in_range else "Out of industry range", "good" if central_in_range else "warn")
        + status_chip(distance_text("Yield", yield_status, yield_distance, "t/ha"), "info")
        + status_chip(distance_text("NFI", nfi_status, nfi_distance, "R/ha"), "info")
        + "</div>",
        unsafe_allow_html=True,
    )
    map_left, map_centre, map_right = st.columns([1, 6, 1])
    with map_centre:
        st.pyplot(benchmark_range_figure, width="stretch")

    st.markdown("### Regional Competitiveness Chart — Scenario Outcomes")
    st.caption(
        "Net Farm Income is shown as bars and yield as markers for Phase 1, all three response cases and the regional boundaries."
    )
    bar_left, bar_centre, bar_right = st.columns([1, 8, 1])
    with bar_centre:
        st.pyplot(benchmark_summary_figure, width="stretch")

    summary_export = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "_class"} for row in benchmark_summary_rows]
    )
    detail_export = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "_class"} for row in detail_rows]
    )
    safe_region = str(base.get("region", "Region")).replace(" ", "_").replace("/", "-")
    report_base_name = f"Phase3A_Scenario_Benchmark_{safe_region}_{base.get('year') or 2025}"
    excel_bytes = phase3_excel_bytes(
        summary_export, detail_export, profile_rows, benchmark_range_figure, benchmark_summary_figure
    )
    printable_report = build_printable_report(
        profile_rows=profile_rows,
        summary_rows=benchmark_summary_rows,
        detail_rows=detail_rows,
        range_figure=benchmark_range_figure,
        summary_figure=benchmark_summary_figure,
    )

    st.markdown("### Results Export")
    st.caption("Download the complete Phase 3A scenario benchmark and supporting detail.")
    export_columns = st.columns(4)
    with export_columns[0]:
        st.download_button(
            "Summary CSV", summary_export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{report_base_name}_summary.csv", mime="text/csv", width="stretch",
        )
    with export_columns[1]:
        st.download_button(
            "Detailed costs CSV", detail_export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{report_base_name}_detailed_costs.csv", mime="text/csv", width="stretch",
        )
    with export_columns[2]:
        st.download_button(
            "Excel report", excel_bytes, file_name=f"{report_base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
        )
    with export_columns[3]:
        pdf_configuration = wkhtmltopdf_configuration()
        if pdf_configuration is not None:
            pdfkit, configuration = pdf_configuration
            try:
                pdf_bytes = pdfkit.from_string(
                    printable_report, False, configuration=configuration,
                    options={"quiet": "", "orientation": "Landscape"},
                )
                st.download_button(
                    "PDF report", pdf_bytes, file_name=f"{report_base_name}.pdf",
                    mime="application/pdf", width="stretch",
                )
            except Exception:
                st.download_button(
                    "Printable report", printable_report.encode("utf-8"),
                    file_name=f"{report_base_name}.html", mime="text/html", width="stretch",
                )
        else:
            st.download_button(
                "Printable report", printable_report.encode("utf-8"),
                file_name=f"{report_base_name}.html", mime="text/html", width="stretch",
            )
            st.caption("Open the report and use Ctrl+P → Save as PDF.")

with tab_b:
    st.subheader("Page 3B — Detailed Multi-Input Scenario Analysis")
    st.caption(
        "Supporting detail for the same multi-input scenario shown on Page 3A. No second set of inputs is required."
    )
    st.markdown(
        f"""
        <div class="phase3-note">
          <strong>Current scenario:</strong> {escape(str(base.get('region', '')))} ·
          {escape(str(base.get('grape_variety', '')))} · {escape(scenario_text)}.<br>
          <span style="color:#5f6b7a;">Results show the estimated response range around the central result.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Baseline NFI", money(baseline_row["NFI (R/ha)"]))
    metric_columns[1].metric(
        "Central NFI", money(central_row["NFI (R/ha)"]),
        delta=money(central_row["Change in NFI (R/ha)"]),
    )
    metric_columns[2].metric(
        "Central yield", f"{central_row['Yield (t/ha)']:.2f} t/ha",
        delta=f"{central_row['Yield (t/ha)'] - baseline_row['Yield (t/ha)']:+.2f} t/ha",
    )
    metric_columns[3].metric("Central total cost", money(central_row["Total cost (R/ha)"]))

    st.markdown("### Scenario Outcomes")
    display_summary = summary[
        [
            "Case", "Selected costs (R/ha)", "Direct cost change (R/ha)",
            "Associated cost change (R/ha)", "Yield (t/ha)", "Revenue (R/ha)",
            "Total cost (R/ha)", "NFI (R/ha)", "Change in NFI (R/ha)",
        ]
    ].copy()
    st.dataframe(
        display_summary, hide_index=True, width="stretch",
        column_config={
            "Selected costs (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
            "Direct cost change (R/ha)": st.column_config.NumberColumn(format="R %.0f"),
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
    selected_labels = [MODEL_KEY_TO_SPEC[key].label.lower() for key in applied_adjustments]
    selection_description = (
        selected_labels[0]
        if len(selected_labels) == 1
        else ", ".join(selected_labels[:-1]) + f" and {selected_labels[-1]}"
    )
    st.markdown("#### Automatic interpretation")
    st.write(
        f"For the central result, the combined adjustments to {selection_description} "
        f"{direction} NFI by {money(abs(central_change))}/ha relative to the Phase 1 baseline. "
        f"The adjusted yield is {number(central_row['Yield (t/ha)'])} t/ha, "
        f"with grape price held constant at {money(base.get('farmer_income_rt', 0.0))}/t."
    )

    with st.expander("Detailed cost calculations", expanded=False):
        cost_details = result["cost_details"][
            [
                "Case", "Category", "Item", "Baseline cost (R/ha)",
                "Scenario cost (R/ha)", "Change (R/ha)", "Change (%)",
            ]
        ].copy()
        st.dataframe(
            cost_details, hide_index=True, width="stretch",
            column_config={
                "Baseline cost (R/ha)": st.column_config.NumberColumn(format="R %.2f"),
                "Scenario cost (R/ha)": st.column_config.NumberColumn(format="R %.2f"),
                "Change (R/ha)": st.column_config.NumberColumn(format="R %.2f"),
                "Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    st.download_button(
        "Download detailed scenario summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="phase3b_scenario_summary.csv", mime="text/csv", width="content",
    )

plt.close(benchmark_range_figure)
plt.close(benchmark_summary_figure)

# Keep a compact result for a later holistic summary page without replacing
# any of the detail shown above.
st.session_state["phase3_latest_result"] = {
    "settings": {
        "adjustments": dict(applied_adjustments),
        "rounds": int(result["rounds"]),
        "area": base.get("region"),
        "cultivar": base.get("grape_variety"),
    },
    "summary": summary.to_dict("records"),
}
