from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    st.set_page_config(
        page_title="Vineyard Production Tool",
        page_icon="🍇",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = [
        st.Page(
            str(PROJECT_ROOT / "views" / "phase1_current.py"),
            title="Phase 1 — Current Benchmark",
            icon="📊",
            default=True,
        ),
        st.Page(
            str(PROJECT_ROOT / "views" / "phase2_forecast.py"),
            title="Phase 2 — Outlook to 2030",
            icon="📈",
        ),
        st.Page(
            str(PROJECT_ROOT / "views" / "phase3_future.py"),
            title="Phase 3 — Cost & Yield Scenarios",
            icon="🧭",
        ),
    ]

    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
