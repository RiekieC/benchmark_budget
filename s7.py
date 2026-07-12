"""Compatibility launcher for the deployed Streamlit app.

The multipage application itself is defined in app.py.
Locally, run: streamlit run app.py
Streamlit Community Cloud may continue using s7.py as its entrypoint.
"""

from app import main

main()
