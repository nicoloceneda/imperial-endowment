import re
import shutil
import subprocess
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import streamlit as st

st.set_page_config(page_title="Imperial Endowment Dashboard", layout="wide")

ASSET_CLASS_HEADING_PATTERN = re.compile(r"Breakdown\s+by\s+asset\s+class", flags=re.IGNORECASE)
DATE_TAG_PATTERN = re.compile(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}")
SECTION_END_PATTERN = re.compile(
    r"\n\s*Direct Holdings, Collective Investment Vehicles, Property",
    flags=re.IGNORECASE,
)
TOTAL_ROW_PATTERN = re.compile(r"Endowment\s+Total\s+([\d,\s]+|-)$", flags=re.IGNORECASE)
ASSET_ROW_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z &/,\-]+?)\s+([\d,\s]+|-)\s*$")
DIRECT_HOLDINGS_HEADER_PATTERN = re.compile(r"^\s*Direct Holdings\s+£", flags=re.MULTILINE)
TOTAL_FUNDS_PATTERN = re.compile(r"^\s*.*Total\s+Funds\s+[\d,\s]+$", flags=re.MULTILINE)
HOLDING_AMOUNT_PATTERN = re.compile(r"\d[\d\s]*,\d{3}(?:,\d{3})*")
COLLECTIVE_HOLDINGS_HEADER_PATTERN = re.compile(
    r"^\s*Direct Holdings in Collective Investment Vehicles\s*$",
    flags=re.MULTILINE,
)
TOTAL_COLLECTIVE_VEHICLES_PATTERN = re.compile(
    r"Total\s+Collective\s+Vehicles",
    flags=re.IGNORECASE,
)
DIRECT_PROPERTY_HEADER_PATTERN = re.compile(
    r"^\s*Direct Holdings in Property\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
INDIRECT_HOLDINGS_COMPANIES_HEADER_PATTERN = re.compile(
    r"^\s*Indirect Holdings in Companies\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
COLLECTIVE_RANGE_PATTERN = re.compile(
    r"^\s*(<\s*£?\s*\d+(?:\.\d+)?m|£?\s*\d+(?:\.\d+)?m\s*<\s*£?\s*\d+(?:\.\d+)?m)\s*(.*)$",
    flags=re.IGNORECASE,
)
NAME_FRAGMENT_TOKENS = {
    "A",
    "PLC",
    "PLC*",
    "LTD",
    "INC",
    "COM",
    "CO",
    "CORP",
    "NPV",
    "ORD",
}


def normalize_pdf_text(text: str) -> str:
    for symbol in ("\u00a0", "\u2007", "\u202f"):
        text = text.replace(symbol, " ")
    for symbol in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(symbol, "-")
    return text


def parse_pdf_number(value: str) -> float:
    value = value.strip()
    if value in {"", "-"}:
        return 0.0
    cleaned = re.sub(r"[^\d,\s]", "", value).replace(" ", "")
    if not cleaned:
        raise ValueError(f"Unable to parse numeric value from: {value!r}")
    return float(cleaned.replace(",", ""))


@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_path: str) -> str:
    gs_path = shutil.which("gs")
    if gs_path is None:
        raise RuntimeError("Ghostscript (gs) is required to extract text from PDFs.")
    result = subprocess.run(
        [gs_path, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=txtwrite", "-sOutputFile=-", pdf_path],
        capture_output=True,
        text=True,
        check=True,
    )
    return normalize_pdf_text(result.stdout)


def parse_asset_class_section(text: str, pdf_name: str) -> dict:
    heading = ASSET_CLASS_HEADING_PATTERN.search(text)
    if heading is None:
        raise ValueError(f"{pdf_name}: could not find 'Breakdown by asset class' section.")
    section = text[heading.start() :]
    section_end = SECTION_END_PATTERN.search(section)
    if section_end is not None:
        section = section[: section_end.start()]

    date_match = DATE_TAG_PATTERN.search(section)
    if date_match is None:
        raise ValueError(f"{pdf_name}: could not find quarter tag in asset class section.")
    date_value = pd.to_datetime(date_match.group(), format="%b-%y") + pd.offsets.MonthEnd(0)

    assets: dict[str, float] = {}
    total_value = None
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or "£ 000s" in line:
            continue
        if ASSET_CLASS_HEADING_PATTERN.search(line):
            continue

        total_match = TOTAL_ROW_PATTERN.search(line)
        if total_match:
            total_value = parse_pdf_number(total_match.group(1))
            continue

        asset_match = ASSET_ROW_PATTERN.match(line)
        if asset_match:
            asset_name = re.sub(r"\s+", " ", asset_match.group(1)).strip()
            assets[asset_name] = parse_pdf_number(asset_match.group(2))

    if not assets:
        raise ValueError(f"{pdf_name}: no asset class rows were parsed.")
    if total_value is None:
        total_value = float(sum(assets.values()))

    return {"Date": date_value, "Total": total_value, **assets}


def clean_company_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -")


def is_incomplete_company_name(value: str) -> bool:
    normalized = clean_company_name(value)
    if not normalized:
        return False
    token = normalized.upper()
    if token in NAME_FRAGMENT_TOKENS:
        return True
    return len(re.sub(r"[^A-Za-z]", "", token)) <= 3


def parse_direct_holdings_section(text: str, pdf_name: str) -> list[dict]:
    start_match = DIRECT_HOLDINGS_HEADER_PATTERN.search(text)
    if start_match is None:
        raise ValueError(f"{pdf_name}: could not find Direct Holdings table.")

    section = text[start_match.end() :]
    end_match = TOTAL_FUNDS_PATTERN.search(section)
    if end_match is not None:
        section = section[: end_match.start()]

    records: list[dict] = []
    pending_prefix = ""
    unresolved_fragment_index: int | None = None

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("direct holdings"):
            continue
        if line.startswith("*"):
            continue

        amount_matches = list(HOLDING_AMOUNT_PATTERN.finditer(line))

        if not amount_matches:
            if unresolved_fragment_index is not None and re.search(r"[A-Za-z]", line):
                joined = clean_company_name(f"{line} {records[unresolved_fragment_index]['Company']}")
                records[unresolved_fragment_index]["Company"] = joined
                unresolved_fragment_index = None
                continue
            if re.search(r"[A-Za-z]", line):
                pending_prefix = clean_company_name(f"{pending_prefix} {line}")
            continue

        if len(amount_matches) == 1:
            match = amount_matches[0]
            company_name = clean_company_name(line[: match.start()])
            if pending_prefix:
                company_name = clean_company_name(f"{pending_prefix} {company_name}")
                pending_prefix = ""
            amount_value = parse_pdf_number(match.group())
            if company_name:
                records.append({"Company": company_name, "Amount": amount_value})
                if is_incomplete_company_name(company_name):
                    unresolved_fragment_index = len(records) - 1
            trailing_text = clean_company_name(line[match.end() :])
            if trailing_text and unresolved_fragment_index is not None:
                joined = clean_company_name(f"{trailing_text} {records[unresolved_fragment_index]['Company']}")
                records[unresolved_fragment_index]["Company"] = joined
                unresolved_fragment_index = None
            continue

        left_match, right_match = amount_matches[0], amount_matches[1]

        left_name = clean_company_name(line[: left_match.start()])
        if pending_prefix:
            left_name = clean_company_name(f"{pending_prefix} {left_name}")
            pending_prefix = ""
        right_name = clean_company_name(line[left_match.end() : right_match.start()])

        if left_name:
            left_amount = parse_pdf_number(left_match.group())
            records.append({"Company": left_name, "Amount": left_amount})
            if is_incomplete_company_name(left_name):
                unresolved_fragment_index = len(records) - 1
        if right_name:
            right_amount = parse_pdf_number(right_match.group())
            records.append({"Company": right_name, "Amount": right_amount})
            if is_incomplete_company_name(right_name):
                unresolved_fragment_index = len(records) - 1

    if not records:
        raise ValueError(f"{pdf_name}: no direct holdings rows were parsed.")
    return records


def normalize_collective_range(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    normalized = re.sub(r"\s*<\s*", " < ", normalized)
    normalized = normalized.replace("£ ", "£")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized.startswith("<") and not normalized.startswith("< "):
        normalized = f"< {normalized[1:].strip()}"
    return normalized


def parse_collective_range_bounds_million(range_label: str) -> tuple[float, float] | None:
    values = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", range_label)]
    if not values:
        return None

    if range_label.strip().startswith("<"):
        lower_bound = 0.0
        upper_bound = values[-1]
    elif len(values) >= 2:
        lower_bound = values[0]
        upper_bound = values[1]
    else:
        lower_bound = values[0]
        upper_bound = values[0]

    if upper_bound < lower_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    return lower_bound, upper_bound


def parse_collective_holdings_section(text: str, pdf_name: str) -> list[dict]:
    start_match = COLLECTIVE_HOLDINGS_HEADER_PATTERN.search(text)
    if start_match is None:
        raise ValueError(
            f"{pdf_name}: could not find 'Direct Holdings in Collective Investment Vehicles' table."
        )

    section = text[start_match.end() :]
    end_positions: list[int] = []
    total_match = TOTAL_COLLECTIVE_VEHICLES_PATTERN.search(section)
    if total_match is not None:
        end_positions.append(total_match.start())
    property_match = DIRECT_PROPERTY_HEADER_PATTERN.search(section)
    if property_match is not None:
        end_positions.append(property_match.start())
    if end_positions:
        section = section[: min(end_positions)]

    ranges: list[dict] = []
    active_range_index: int | None = None
    pending_vehicles_text = ""

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("*"):
            continue

        range_match = COLLECTIVE_RANGE_PATTERN.match(line)
        if range_match:
            range_label = normalize_collective_range(range_match.group(1))
            vehicles_text = clean_company_name(f"{pending_vehicles_text} {range_match.group(2)}")
            ranges.append({"Range": range_label, "VehiclesText": vehicles_text})
            active_range_index = len(ranges) - 1
            pending_vehicles_text = ""
            continue

        if active_range_index is not None and re.search(r"[A-Za-z]", line):
            current_text = ranges[active_range_index]["VehiclesText"]
            ranges[active_range_index]["VehiclesText"] = clean_company_name(f"{current_text} {line}")
            continue

        if active_range_index is None and re.search(r"[A-Za-z]", line):
            pending_vehicles_text = clean_company_name(f"{pending_vehicles_text} {line}")

    if not ranges:
        raise ValueError(
            f"{pdf_name}: no ranges were parsed from direct holdings in collective investment vehicles."
        )

    records: list[dict] = []
    for range_row in ranges:
        vehicles = [
            clean_company_name(vehicle)
            for vehicle in range_row["VehiclesText"].split(",")
            if clean_company_name(vehicle)
        ]
        for vehicle in vehicles:
            records.append({"Range": range_row["Range"], "Vehicle": vehicle})

    if not records:
        raise ValueError(
            f"{pdf_name}: no vehicle rows were parsed from direct holdings in collective investment vehicles."
        )
    return records


def parse_indirect_holdings_companies_section(text: str, pdf_name: str) -> list[str]:
    start_match = INDIRECT_HOLDINGS_COMPANIES_HEADER_PATTERN.search(text)
    if start_match is None:
        raise ValueError(f"{pdf_name}: could not find 'Indirect Holdings in Companies' section.")

    section = text[start_match.end() :]
    parsed_rows: list[tuple[str, str]] = []
    baseline_indent: int | None = None

    for raw_line in section.splitlines():
        leading_spaces = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("indirect holdings in companies"):
            continue
        if lowered.startswith("please refer"):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            continue
        if line.startswith("*"):
            continue
        if not re.search(r"[A-Za-z]", line):
            continue

        entries = [clean_company_name(part) for part in re.split(r"\s{2,}", line) if clean_company_name(part)]
        if not entries:
            continue

        left_entry = entries[0]
        right_entry = entries[1] if len(entries) > 1 else ""
        if len(entries) > 2:
            right_entry = clean_company_name(" ".join(entries[1:]))

        if len(entries) > 1:
            baseline_indent = leading_spaces if baseline_indent is None else min(baseline_indent, leading_spaces)

        # Handle wrapped left-column labels, e.g. "Inst Inv" on the next line.
        words = re.findall(r"[A-Za-z]+", left_entry)
        looks_like_wrapped_fragment = words and len(words) <= 3 and all(len(word) <= 4 for word in words)
        if (
            len(entries) > 1
            and parsed_rows
            and baseline_indent is not None
            and baseline_indent + 3 <= leading_spaces <= baseline_indent + 8
            and looks_like_wrapped_fragment
        ):
            prev_left, prev_right = parsed_rows[-1]
            parsed_rows[-1] = (clean_company_name(f"{prev_left} {left_entry}"), prev_right)
            left_entry = ""

        parsed_rows.append((left_entry, right_entry))

    companies: list[str] = []
    for left_entry, right_entry in parsed_rows:
        if left_entry:
            companies.append(left_entry)
        if right_entry:
            companies.append(right_entry)

    if not companies:
        raise ValueError(f"{pdf_name}: no companies were parsed from indirect holdings in companies.")
    return companies


def build_two_column_company_table(companies: list[str]) -> pd.DataFrame:
    rows: list[list[str]] = []
    for index in range(0, len(companies), 2):
        left_company = companies[index]
        right_company = companies[index + 1] if index + 1 < len(companies) else ""
        rows.append([left_company, right_company])

    table = pd.DataFrame(rows, columns=["Company", "Company "])
    table.index = range(1, len(table) * 2, 2)
    return table


@st.cache_data(show_spinner=False)
def load_endowment_data(pdf_dir: str) -> pd.DataFrame:
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}.")

    rows = []
    errors = []
    for pdf_path in pdf_paths:
        try:
            text = extract_pdf_text(str(pdf_path))
            rows.append(parse_asset_class_section(text, pdf_path.name))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{pdf_path.name}: {exc}")

    if errors:
        error_preview = "; ".join(errors[:3])
        raise RuntimeError(f"Failed to parse one or more PDFs ({len(errors)} files). {error_preview}")

    df = pd.DataFrame(rows).sort_values("Date")
    asset_columns = [column for column in df.columns if column not in {"Date", "Total"}]
    df[asset_columns] = df[asset_columns].fillna(0.0)
    df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp("M")
    return df


@st.cache_data(show_spinner=False)
def load_direct_holdings_data(pdf_dir: str) -> pd.DataFrame:
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}.")

    rows = []
    errors = []
    for pdf_path in pdf_paths:
        try:
            text = extract_pdf_text(str(pdf_path))
            statement_date = parse_asset_class_section(text, pdf_path.name)["Date"]
            statement_label = statement_date.strftime("%b-%y")
            holdings = parse_direct_holdings_section(text, pdf_path.name)
            for row_order, holding in enumerate(holdings, start=1):
                rows.append(
                    {
                        "StatementDate": statement_date,
                        "Statement": statement_label,
                        "StatementFile": pdf_path.name,
                        "RowOrder": row_order,
                        "Company": holding["Company"],
                        "Amount": holding["Amount"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{pdf_path.name}: {exc}")

    if errors:
        error_preview = "; ".join(errors[:3])
        raise RuntimeError(
            f"Failed to parse direct holdings from one or more PDFs ({len(errors)} files). {error_preview}"
        )

    return pd.DataFrame(rows).sort_values(["StatementDate", "RowOrder"])


@st.cache_data(show_spinner=False)
def load_collective_holdings_data(pdf_dir: str) -> pd.DataFrame:
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}.")

    rows = []
    errors = []
    for pdf_path in pdf_paths:
        try:
            text = extract_pdf_text(str(pdf_path))
            statement_date = parse_asset_class_section(text, pdf_path.name)["Date"]
            statement_label = statement_date.strftime("%b-%y")
            holdings = parse_collective_holdings_section(text, pdf_path.name)
            for row_order, holding in enumerate(holdings, start=1):
                rows.append(
                    {
                        "StatementDate": statement_date,
                        "Statement": statement_label,
                        "StatementFile": pdf_path.name,
                        "RowOrder": row_order,
                        "Range": holding["Range"],
                        "Vehicle": holding["Vehicle"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{pdf_path.name}: {exc}")

    if errors:
        error_preview = "; ".join(errors[:3])
        raise RuntimeError(
            "Failed to parse direct holdings in collective investment vehicles "
            f"from one or more PDFs ({len(errors)} files). {error_preview}"
        )

    return pd.DataFrame(rows).sort_values(["StatementDate", "RowOrder"])


@st.cache_data(show_spinner=False)
def load_indirect_holdings_data(pdf_dir: str) -> pd.DataFrame:
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}.")

    rows = []
    errors = []
    for pdf_path in pdf_paths:
        try:
            text = extract_pdf_text(str(pdf_path))
            statement_date = parse_asset_class_section(text, pdf_path.name)["Date"]
            statement_label = statement_date.strftime("%b-%y")
            companies = parse_indirect_holdings_companies_section(text, pdf_path.name)
            for row_order, company in enumerate(companies, start=1):
                rows.append(
                    {
                        "StatementDate": statement_date,
                        "Statement": statement_label,
                        "StatementFile": pdf_path.name,
                        "RowOrder": row_order,
                        "Company": company,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{pdf_path.name}: {exc}")

    if errors:
        error_preview = "; ".join(errors[:3])
        raise RuntimeError(
            f"Failed to parse indirect holdings from one or more PDFs ({len(errors)} files). {error_preview}"
        )

    return pd.DataFrame(rows).sort_values(["StatementDate", "RowOrder"])


@st.cache_data(show_spinner=False)
def load_sp500(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sp500 = web.DataReader("SP500", "fred", start, end)
    sp500 = sp500.dropna().sort_index()
    return sp500["SP500"]


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

st.title("Imperial College Endowment")
st.caption("The Endowment publishes details of its Unitised Scheme investment holdings on a quarterly basis.")

try:
    endowment = load_endowment_data("data")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load endowment data from quarterly PDFs: {exc}")
    st.stop()

asset_columns = [col for col in endowment.columns if col not in {"Date", "Total"}]

start_date = endowment["Date"].min() - pd.offsets.MonthBegin(1)
end_date = endowment["Date"].max() + pd.offsets.MonthEnd(1)

sp500_raw = safe_fetch_sp500(start_date, end_date)
sp500 = align_next_trading_day(sp500_raw, endowment["Date"]) if sp500_raw is not None else None

composition = endowment.melt(id_vars="Date", value_vars=asset_columns, var_name="Asset Class", value_name="Value")
composition["QuarterLabel"] = composition["Date"].dt.to_period("Q").astype(str).str.replace("Q", " Q", regex=False)
chart_data = (
    composition.groupby(["Date", "QuarterLabel", "Asset Class"], as_index=False)["Value"].sum()
)

cumulative_endowment = endowment["Total"] / endowment["Total"].iloc[0] - 1
cumulative_sp500 = sp500 / sp500.iloc[0] - 1 if sp500 is not None else None

overview_tab, direct_tab, indirect_tab = st.tabs(["Overview", "Direct Holdings", "Indirect Holdings"])

with overview_tab:
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Asset Allocation")
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
        st.subheader("Cumulative Return")
        line_data = pd.DataFrame(
            {"Endowment": cumulative_endowment.values},
            index=endowment["Date"],
        )
        if sp500 is not None:
            line_data["S&P 500"] = cumulative_sp500.values
        else:
            st.caption("S&P 500 series is temporarily unavailable; showing endowment only.")
        line_chart_data = line_data.reset_index().melt("Date", var_name="Series", value_name="Return")
        line_chart = (
            alt.Chart(line_chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%Y Q%q",
                        labelAngle=-90,
                        values=endowment["Date"].dt.to_pydatetime().tolist(),
                    ),
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

    metrics_df = pd.DataFrame(
        {
            "Nominal": nominal_metrics,
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
        metrics_display.loc[metric] = metrics_df.loc[metric].map(
            lambda x: "—" if pd.isna(x) else fmt.format(x)
        )

    st.dataframe(metrics_display, width="stretch")
    st.caption("Sources: Imperial endowment quarterly PDF disclosures, S&P 500 (FRED).")

with direct_tab:
    try:
        direct_holdings = load_direct_holdings_data("data")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load direct holdings from quarterly PDFs: {exc}")
    else:
        statements = (
            direct_holdings[["StatementDate", "Statement", "StatementFile"]]
            .drop_duplicates()
            .sort_values("StatementDate", ascending=False)
        )
        selected_statement = st.selectbox(
            "Statement",
            statements["Statement"].tolist(),
            index=0,
        )
        sort_option = st.radio(
            "Sort holdings by",
            ["Amount (High-Low)", "Name (A-Z)"],
            index=0,
            horizontal=True,
        )
        selected_metadata = statements[statements["Statement"] == selected_statement].iloc[0]

        st.subheader("Direct Holdings")
        selected_holdings = (
            direct_holdings[direct_holdings["Statement"] == selected_statement]
            .assign(
                _cash_first=lambda df: df["Company"].str.strip().str.upper().ne("CASH").astype(int),
                _company_sort=lambda df: df["Company"].str.upper(),
            )
            .sort_values(by=["_cash_first", "_company_sort", "RowOrder"])
            .drop(columns=["_cash_first", "_company_sort"])
        )

        cash_rows = selected_holdings[selected_holdings["Company"].str.strip().str.upper() == "CASH"]
        non_cash_rows = selected_holdings[selected_holdings["Company"].str.strip().str.upper() != "CASH"]

        if sort_option == "Name (A-Z)":
            non_cash_rows = non_cash_rows.sort_values(
                by=["Company", "RowOrder"],
                key=lambda column: column.str.upper() if column.name == "Company" else column,
            )
        else:
            non_cash_rows = non_cash_rows.sort_values(by=["Amount", "Company"], ascending=[False, True])

        holdings_display = pd.concat([cash_rows, non_cash_rows], ignore_index=True)
        holdings_table = holdings_display[["Company", "Amount"]].copy()
        holdings_table["Investment Amount (£)"] = holdings_table["Amount"].map(lambda value: f"{value:,.0f}")
        holdings_table = holdings_table.drop(columns=["Amount"])

        st.table(holdings_table)

        st.subheader("Direct Holdings - Allocation")
        holdings_chart_data = holdings_display[["Company", "Amount"]].copy()
        holdings_chart_data["IsCash"] = holdings_chart_data["Company"].str.strip().str.upper().eq("CASH")
        holding_order = holdings_display["Company"].drop_duplicates().tolist()
        holdings_chart = (
            alt.Chart(holdings_chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Amount:Q", title="Investment Amount (£)", axis=alt.Axis(format=",.0f")),
                y=alt.Y("Company:N", sort=holding_order, title="Holding"),
                color=alt.condition(alt.datum.IsCash, alt.value("#2a9d8f"), alt.value("#4e79a7")),
                tooltip=[
                    "Company:N",
                    alt.Tooltip("Amount:Q", title="Investment Amount (£)", format=",.0f"),
                ],
            )
            .properties(height=max(320, len(holdings_chart_data) * 16))
        )
        st.altair_chart(holdings_chart, use_container_width=True)

        st.subheader("Direct Holdings in Collective Investment Vehicles")
        try:
            collective_holdings = load_collective_holdings_data("data")
        except Exception as exc:  # noqa: BLE001
            st.error(
                "Could not load direct holdings in collective investment vehicles "
                f"from quarterly PDFs: {exc}"
            )
        else:
            selected_collective_holdings = collective_holdings[
                collective_holdings["Statement"] == selected_statement
            ].copy()
            if selected_collective_holdings.empty:
                st.info("No collective investment vehicle details found for the selected statement.")
            else:
                range_bounds = selected_collective_holdings["Range"].map(parse_collective_range_bounds_million)
                selected_collective_holdings["LowerBoundM"] = range_bounds.map(
                    lambda value: np.nan if value is None else value[0]
                )
                selected_collective_holdings["UpperBoundM"] = range_bounds.map(
                    lambda value: np.nan if value is None else value[1]
                )

                if sort_option == "Name (A-Z)":
                    selected_collective_holdings = selected_collective_holdings.sort_values(
                        by=["Vehicle", "RowOrder"],
                        key=lambda column: column.str.upper() if column.name == "Vehicle" else column,
                    )
                else:
                    selected_collective_holdings = selected_collective_holdings.sort_values(
                        by=["UpperBoundM", "LowerBoundM", "Vehicle"],
                        ascending=[False, False, True],
                        na_position="last",
                        key=lambda column: column.str.upper() if column.name == "Vehicle" else column,
                    )

                collective_table = selected_collective_holdings[["Vehicle", "Range"]].rename(
                    columns={
                        "Range": "Investment Range",
                        "Vehicle": "Collective Investment Vehicle",
                    }
                )
                collective_table = collective_table.reset_index(drop=True)
                collective_table.index = collective_table.index + 1
                st.table(collective_table)

                st.subheader("Direct Holdings in Collective Investment Vehicles - Allocation")
                collective_chart_data = selected_collective_holdings.copy()
                collective_chart_data = collective_chart_data.dropna(
                    subset=["LowerBoundM", "UpperBoundM"]
                ).copy()
                collective_chart_data["RangeLabel"] = collective_chart_data["Range"]

                if collective_chart_data.empty:
                    st.info("No investment range data available to display for the selected statement.")
                else:
                    vehicle_order = collective_chart_data["Vehicle"].drop_duplicates().tolist()
                    y_encoding = alt.Y(
                        "Vehicle:N",
                        sort=vehicle_order,
                        title="Collective Investment Vehicle",
                    )
                    range_line = (
                        alt.Chart(collective_chart_data)
                        .mark_rule(color="#4e79a7", strokeWidth=2)
                        .encode(
                            x=alt.X(
                                "LowerBoundM:Q",
                                title="Investment Range (£m)",
                                axis=alt.Axis(format=",.1f"),
                            ),
                            x2="UpperBoundM:Q",
                            y=y_encoding,
                            tooltip=[
                                alt.Tooltip("Vehicle:N", title="Collective Investment Vehicle"),
                                alt.Tooltip("RangeLabel:N", title="Investment Range"),
                                alt.Tooltip("LowerBoundM:Q", title="Lower Bound (£m)", format=",.1f"),
                                alt.Tooltip("UpperBoundM:Q", title="Upper Bound (£m)", format=",.1f"),
                            ],
                        )
                    )
                    lower_marker = (
                        alt.Chart(collective_chart_data)
                        .mark_point(shape="circle", filled=True, size=80, color="#d73027")
                        .encode(
                            x=alt.X(
                                "LowerBoundM:Q",
                                title="Investment Range (£m)",
                                axis=alt.Axis(format=",.1f"),
                            ),
                            y=y_encoding,
                            tooltip=[
                                alt.Tooltip("Vehicle:N", title="Collective Investment Vehicle"),
                                alt.Tooltip("RangeLabel:N", title="Investment Range"),
                                alt.Tooltip("LowerBoundM:Q", title="Lower Bound (£m)", format=",.1f"),
                                alt.Tooltip("UpperBoundM:Q", title="Upper Bound (£m)", format=",.1f"),
                            ],
                        )
                    )
                    upper_marker = (
                        alt.Chart(collective_chart_data)
                        .mark_point(shape="circle", filled=True, size=80, color="#1a9850")
                        .encode(
                            x=alt.X(
                                "UpperBoundM:Q",
                                title="Investment Range (£m)",
                                axis=alt.Axis(format=",.1f"),
                            ),
                            y=y_encoding,
                            tooltip=[
                                alt.Tooltip("Vehicle:N", title="Collective Investment Vehicle"),
                                alt.Tooltip("RangeLabel:N", title="Investment Range"),
                                alt.Tooltip("LowerBoundM:Q", title="Lower Bound (£m)", format=",.1f"),
                                alt.Tooltip("UpperBoundM:Q", title="Upper Bound (£m)", format=",.1f"),
                            ],
                        )
                    )
                    collective_chart = (range_line + lower_marker + upper_marker).properties(
                        height=max(320, len(collective_chart_data) * 16)
                    )
                    st.altair_chart(collective_chart, use_container_width=True)

        st.caption(f"Source statement: {selected_metadata['StatementFile']}")

with indirect_tab:
    try:
        indirect_holdings = load_indirect_holdings_data("data")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load indirect holdings from quarterly PDFs: {exc}")
    else:
        statements = (
            indirect_holdings[["StatementDate", "Statement", "StatementFile"]]
            .drop_duplicates()
            .sort_values("StatementDate", ascending=False)
        )
        selected_statement = st.selectbox(
            "Statement",
            statements["Statement"].tolist(),
            index=0,
            key="indirect_statement_selector",
        )
        selected_metadata = statements[statements["Statement"] == selected_statement].iloc[0]

        st.subheader("Indirect Holdings")
        selected_indirect_holdings = indirect_holdings[
            indirect_holdings["Statement"] == selected_statement
        ].sort_values("RowOrder")

        if selected_indirect_holdings.empty:
            st.info("No indirect holdings in companies found for the selected statement.")
        else:
            indirect_companies = selected_indirect_holdings["Company"].tolist()
            indirect_table = build_two_column_company_table(indirect_companies)
            st.table(indirect_table)

        st.caption(f"Source statement: {selected_metadata['StatementFile']}")
