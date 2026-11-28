import re
import warnings
from calendar import month_name

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Load Forecast Studio | LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- DESIGN -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f5f7fa;}
[data-testid="stSidebar"] {background:#10243a;}
[data-testid="stSidebar"] * {color:#f4f7fb !important;}
.hero {background:linear-gradient(115deg,#10243a,#1b4965);padding:26px 30px;
       border-radius:17px;color:white;margin-bottom:20px;}
.hero h1 {color:white;margin:0 0 7px 0;font-size:29px;}
.hero p {color:#d7e5ef;margin:0;font-size:14px;}
div[data-testid="stMetric"] {background:white;padding:13px;border:1px solid #e5eaf0;border-radius:12px;}
.section-card {background:white;padding:18px;border:1px solid #e5eaf0;border-radius:14px;}
.stButton>button,.stDownloadButton>button {border-radius:9px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

APP_VERSION = "2.2"
MONTH_LOOKUP = {m.lower(): i for i, m in enumerate(month_name) if m}
MONTH_LOOKUP.update({m[:3].lower(): i for i, m in enumerate(month_name) if m})


# ----------------------------- DATA HELPERS -----------------------------
def clean_columns(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def read_files(files):
    """Read one or more CSV/Excel files and combine every Excel worksheet.

    If a worksheet or filename is a four-digit year (e.g. 2024.xlsx or
    a sheet named 2025), that year is attached to month-name-only records.
    """
    if not files:
        return None

    if not isinstance(files, (list, tuple)):
        files = [files]

    frames = []
    for uploaded in files:
        filename = uploaded.name
        lower_name = filename.lower()
        filename_year_match = re.search(r"(?<!\\d)(19\\d{2}|20\\d{2}|21\\d{2})(?!\\d)", filename)
        filename_year = int(filename_year_match.group(1)) if filename_year_match else None

        if lower_name.endswith(".csv"):
            frame = pd.read_csv(uploaded)
            frame["_Source_File"] = filename
            if filename_year is not None and not any("year" in str(c).lower() for c in frame.columns):
                frame["Year"] = filename_year
            frames.append(frame)
        elif lower_name.endswith((".xlsx", ".xls")):
            # sheet_name=None reads ALL worksheets, not only the first one.
            sheets = pd.read_excel(uploaded, sheet_name=None)
            for sheet_name, sheet_df in sheets.items():
                if sheet_df.empty:
                    continue
                frame = sheet_df.copy()
                has_year = any("year" in str(c).lower() for c in frame.columns)
                sheet_year_match = re.fullmatch(r"(19\\d{2}|20\\d{2}|21\\d{2})", str(sheet_name).strip())
                sheet_year = int(sheet_year_match.group(1)) if sheet_year_match else None

                if not has_year:
                    inferred_year = sheet_year if sheet_year is not None else filename_year
                    if inferred_year is not None:
                        frame["Year"] = inferred_year

                frame["_Source_File"] = filename
                frame["_Source_Sheet"] = str(sheet_name)
                frames.append(frame)
        else:
            raise ValueError(f"Unsupported file type: {filename}. Upload CSV or Excel files.")
