import re
from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
import streamlit as st

st.set_page_config(page_title="Imperial Endowment Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_endowment_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [col.strip() for col in df.columns]

    def parse_date(value: str) -> pd.Timestamp:
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
        if pd.isna(parsed):
            match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(value))
            if match:
                month = int(match.group(2))
                year = int(match.group(3))
                parsed = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        return parsed

    df["Date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    return df


@st.cache_data(show_spinner=False)
def load_sp500(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sp500 = web.DataReader("SP500", "fred", start, end)
    sp500 = sp500.dropna().sort_index()
    return sp500["SP500"]


@st.cache_data(show_spinner=False)
def load_cpi_series() -> pd.Series:
    url = "https://www.ons.gov.uk/generator?uri=/economy/inflationandpriceindices/timeseries/l522/mm23&format=csv"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    data_start = next(i for i, line in enumerate(lines) if re.match(r'"\d{4}(?: [A-Z]{3})?"', line))
    data_lines = lines[data_start:]
    data = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=["date", "cpi"])
    data = data[data["date"].str.contains(r"^\d{4} [A-Z]{3}$", regex=True)]
    data["date"] = pd.to_datetime(data["date"], format="%Y %b") + pd.offsets.MonthEnd(0)
    data["cpi"] = pd.to_numeric(data["cpi"], errors="coerce")
    data = data.dropna().set_index("date").sort_index()
    return data["cpi"]


def safe_fetch_sp500(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    try:
        sp500 = load_sp500(start, end)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"S&P 500 data unavailable: {exc}")
        return None
    if sp500.empty:
        st.warning("S&P 500 data unavailable for the selected dates.")
        return None
    return sp500


def safe_fetch_cpi() -> pd.Series | None:
    try:
        cpi = load_cpi_series()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Inflation data unavailable: {exc}")
        return None
    if cpi.empty:
        st.warning("Inflation data unavailable for the selected dates.")
        return None
    return cpi


def align_next_trading_day(series: pd.Series, dates: pd.Series) -> pd.Series:
    aligned = series.reindex(dates, method="bfill")
    aligned.index = dates
    return aligned


def performance_metrics(values: pd.Series) -> dict:
    returns = values.pct_change(fill_method=None).dropna()
    if returns.empty:
        return {
            "Average annual return": np.nan,
            "Annualized volatility": np.nan,
            "Sharpe ratio": np.nan,
            "Maximum drawdown": np.nan,
        }

    avg_annual = (1 + returns).prod() ** (12 / len(returns)) - 1
    vol_annual = returns.std() * np.sqrt(12)
    sharpe = np.nan if vol_annual == 0 else avg_annual / vol_annual
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Average annual return": avg_annual,
        "Annualized volatility": vol_annual,
        "Sharpe ratio": sharpe,
        "Maximum drawdown": max_drawdown,
    }


st.markdown(
    """
    <style>
        .main {background-color: #f8f9fb;}
        .block-container {padding-top: 2rem;}
        h1, h2, h3 {font-family: "Segoe UI", sans-serif;}
        .stMetric {background-color: #ffffff; border-radius: 12px; padding: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Imperial College London Endowment Dashboard")
st.caption("Composition and performance overview with inflation-adjusted metrics.")

endowment = load_endowment_data("data/data.xlsx")
asset_columns = [col for col in endowment.columns if col != "Date"]
endowment["Total"] = endowment[asset_columns].sum(axis=1)

start_date = endowment["Date"].min() - pd.offsets.MonthBegin(1)
end_date = endowment["Date"].max() + pd.offsets.MonthEnd(1)

sp500_raw = safe_fetch_sp500(start_date, end_date)
sp500 = align_next_trading_day(sp500_raw, endowment["Date"]) if sp500_raw is not None else None

cpi = safe_fetch_cpi()
if cpi is not None:
    endowment_cpi = cpi.reindex(endowment["Date"], method="ffill")
    if endowment_cpi.notna().any():
        valid_cpi = endowment_cpi[endowment_cpi.notna()]
        real_total = endowment.loc[valid_cpi.index, "Total"] / (valid_cpi / valid_cpi.iloc[0])
    else:
        real_total = None
else:
    real_total = None

composition = endowment.melt(id_vars="Date", value_vars=asset_columns, var_name="Asset Class", value_name="Value")
composition["QuarterLabel"] = composition["Date"].dt.to_period("Q").astype(str).str.replace("Q", " Q", regex=False)
chart_data = (
    composition.groupby(["Date", "QuarterLabel", "Asset Class"], as_index=False)["Value"].sum()
)

cumulative_endowment = endowment["Total"] / endowment["Total"].iloc[0] - 1
cumulative_sp500 = sp500 / sp500.iloc[0] - 1 if sp500 is not None else None

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("Asset Allocation by Quarter")
    bar_chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "QuarterLabel:O",
                title="Quarter",
                sort=alt.SortField("Date", order="ascending"),
            ),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Asset Class:N", legend=alt.Legend(title="Asset Class")),
            tooltip=["QuarterLabel:O", "Asset Class:N", "Value:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(bar_chart, use_container_width=True)

with col_right:
    st.subheader("Cumulative Return: Endowment vs S&P 500")
    line_data = pd.DataFrame(
        {"Endowment": cumulative_endowment.values},
        index=endowment["Date"],
    )
    if sp500 is not None:
        line_data["S&P 500"] = cumulative_sp500.values
    else:
        st.caption("S&P 500 series is temporarily unavailable; showing endowment only.")
    line_chart = (
        alt.Chart(line_data.reset_index().melt("Date", var_name="Series", value_name="Return"))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Date:T",
                title="Date",
                axis=alt.Axis(format="%Y Q%q", labelAngle=-45),
            ),
            y=alt.Y("Return:Q", title="Cumulative return"),
            color=alt.Color("Series:N", legend=alt.Legend(title="Series")),
            tooltip=["Date:T", "Series:N", "Return:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(line_chart, use_container_width=True)

st.subheader("Performance Metrics")
nominal_metrics = performance_metrics(endowment["Total"])
if real_total is not None:
    real_metrics = performance_metrics(real_total)
else:
    real_metrics = {key: np.nan for key in nominal_metrics}

metrics_df = pd.DataFrame(
    {
        "Nominal": nominal_metrics,
        "Inflation-adjusted": real_metrics,
    }
)

format_pct = {
    "Average annual return": "{:.2%}",
    "Annualized volatility": "{:.2%}",
    "Sharpe ratio": "{:.2f}",
    "Maximum drawdown": "{:.2%}",
}

metrics_display = metrics_df.copy().astype(object)
for metric, fmt in format_pct.items():
    metrics_display.loc[metric] = metrics_df.loc[metric].map(lambda x: "—" if pd.isna(x) else fmt.format(x))

st.dataframe(metrics_display, width="stretch")
st.caption("Sources: Imperial endowment data (internal), S&P 500 (FRED), CPIH (ONS).")
