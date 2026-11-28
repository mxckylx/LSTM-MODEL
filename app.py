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

APP_VERSION = "2.0"
MONTH_LOOKUP = {m.lower(): i for i, m in enumerate(month_name) if m}
MONTH_LOOKUP.update({m[:3].lower(): i for i, m in enumerate(month_name) if m})


# ----------------------------- DATA HELPERS -----------------------------
def clean_columns(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def read_file(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Upload a CSV or Excel file.")


def numeric_clean(series):
    """Convert common spreadsheet number formats into numeric values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    values = series.astype(str).str.strip()
    values = values.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    values = values.str.replace(",", "", regex=False)
    values = values.str.replace(r"[₱$]", "", regex=True)
    values = values.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(values, errors="coerce")


def month_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (pd.Timestamp,)):
        return value.month
    token = str(value).strip().lower()
    token_clean = re.sub(r"[^a-z]", "", token)
    if token_clean in MONTH_LOOKUP:
        return MONTH_LOOKUP[token_clean]
    try:
        number = int(float(token))
        return number if 1 <= number <= 12 else np.nan
    except Exception:
        parsed = pd.to_datetime(token, errors="coerce")
        return parsed.month if not pd.isna(parsed) else np.nan


def parse_months(df, month_col, year_col=None):
    values = df[month_col]
    if year_col and year_col in df.columns:
        months = values.map(month_number)
        years = numeric_clean(df[year_col])
        if months.isna().any() or years.isna().any():
            raise ValueError("Some Month or Year entries could not be recognized. Check the date columns.")
        if not years.mod(1).eq(0).all():
            raise ValueError("The Year column must contain whole years, such as 2022 or 2025.")
        if not months.between(1, 12).all():
            raise ValueError("Month values must be month names or numbers from 1 to 12.")
        return pd.PeriodIndex(
            [f"{int(y):04d}-{int(m):02d}" for y, m in zip(years, months)],
            freq="M",
        )

    parsed = []
    for value in values:
        if pd.isna(value):
            parsed.append(pd.NaT)
            continue
        if isinstance(value, (pd.Timestamp,)):
            parsed.append(pd.Period(value, freq="M"))
            continue
        text = str(value).strip()
        # Month names without a year are ambiguous; require a separate Year column.
        token = re.sub(r"[^a-z]", "", text.lower())
        if token in MONTH_LOOKUP:
            parsed.append(pd.NaT)
            continue
        dt = pd.to_datetime(text, errors="coerce")
        parsed.append(pd.Period(dt, freq="M") if not pd.isna(dt) else pd.NaT)

    result = pd.PeriodIndex(parsed, freq="M")
    if result.isna().any():
        raise ValueError(
            "The Month column contains values without a year or unrecognized dates. "
            "Use dates such as 2022-01, Jan-2022, or add a separate Year column."
        )
    return result


def prepare_dataset(main_df, month_col, target_col, year_col=None,
                    extra_df=None, extra_month_col=None, extra_year_col=None):
    main = clean_columns(main_df)
    periods = parse_months(main, month_col, year_col)
    target = numeric_clean(main[target_col])

    data = pd.DataFrame({
        "Month": periods.to_timestamp(),
        "Consumption_kWh": target.to_numpy(),
    })

    # Preserve the other columns in the main workbook as candidate predictors.
    excluded = {month_col, target_col}
    if year_col:
        excluded.add(year_col)
    for col in main.columns:
        if col not in excluded and col not in data.columns:
            data[col] = main[col].to_numpy()

    if data["Consumption_kWh"].isna().any():
        bad = list(np.where(data["Consumption_kWh"].isna())[0] + 2)
        raise ValueError(f"The selected kWh column has blank or invalid values near spreadsheet rows: {bad[:8]}.")

    if (data["Consumption_kWh"] < 0).any():
        raise ValueError("Negative kWh values were found. Check the selected target column.")

    if extra_df is not None:
        extra = clean_columns(extra_df)
        if extra_month_col:
            extra_periods = parse_months(extra, extra_month_col, extra_year_col)
            extra = extra.drop(columns=[extra_month_col] + ([extra_year_col] if extra_year_col else []))
            extra.insert(0, "_MonthKey", extra_periods.astype(str))
            data["_MonthKey"] = periods.astype(str)
            if extra["_MonthKey"].duplicated().any():
                raise ValueError("Additional Variables contains duplicate months.")
            data = data.merge(extra, on="_MonthKey", how="left", validate="one_to_one")
            data.drop(columns="_MonthKey", inplace=True)
        else:
            if len(extra) != len(data):
                raise ValueError("Without a Month column, the additional file must have the same number of rows as the main file.")
            for col in extra.columns:
                if col not in data.columns:
                    data[col] = extra[col].to_numpy()

    data = data.sort_values("Month").reset_index(drop=True)
    if data["Month"].duplicated().any():
        duplicates = data.loc[data["Month"].duplicated(), "Month"].dt.strftime("%b %Y").tolist()
        raise ValueError(f"Duplicate monthly records found: {duplicates[:8]}.")

    return data


def metric_values(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def make_sequences(features, target_scaled, lookback, start, end):
    X, y, positions = [], [], []
    for pos in range(max(lookback, start), end):
        X.append(features[pos - lookback:pos])
        y.append(target_scaled[pos, 0])
        positions.append(pos)
    return np.asarray(X), np.asarray(y), np.asarray(positions)


def train_lstm(data, selected_features, lookback, test_size, epochs, units, dropout, horizon):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be loaded. Add a compatible tensorflow version to requirements.txt "
            "and check the deployment environment."
        ) from exc

    target_col = "Consumption_kWh"
    feature_cols = [target_col] + selected_features
    work = data[["Month"] + feature_cols].copy()
    for col in feature_cols:
        work[col] = numeric_clean(work[col])

    n = len(work)
    split = n - int(test_size)
    if split <= lookback + 5:
        raise ValueError(
            f"Not enough training history for these settings. There are {n} months, "
            f"with {test_size} reserved for testing and a lookback of {lookback}. "
            "Reduce the lookback/test size or provide more records."
        )

    # Fit imputation values on training data only to avoid using future/test information.
    train_medians = {}
    for col in selected_features:
        work[col] = work[col].ffill()
        median_value = work.loc[:split - 1, col].median()
        if pd.isna(median_value):
            raise ValueError(f"Selected predictor '{col}' has no usable numeric training values.")
        train_medians[col] = float(median_value)
        work[col] = work[col].fillna(median_value)

    if work[target_col].isna().any():
        raise ValueError("The kWh target contains missing values. Correct these before training.")

    raw_x = work[feature_cols].to_numpy(dtype=float)
    raw_y = work[[target_col]].to_numpy(dtype=float)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(raw_x[:split])
    y_scaler.fit(raw_y[:split])
    x_scaled = x_scaler.transform(raw_x)
    y_scaled = y_scaler.transform(raw_y)

    X_train, y_train, _ = make_sequences(x_scaled, y_scaled, lookback, lookback, split)
    X_test, _, test_positions = make_sequences(x_scaled, y_scaled, lookback, split, n)
    if len(X_train) < 5 or len(X_test) == 0:
        raise ValueError("There are too few usable training or test sequences. Adjust the test size/lookback.")

    model = Sequential([
        Input(shape=(lookback, len(feature_cols))),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(max(8, units // 2)),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    callbacks = [EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]
    fit_args = {
        "epochs": int(epochs),
        "batch_size": min(8, len(X_train)),
        "verbose": 0,
        "shuffle": False,
        "callbacks": callbacks,
    }
    if len(X_train) >= 12:
        fit_args["validation_split"] = 0.15
    else:
        fit_args.pop("callbacks")
        fit_args["validation_split"] = 0.0

    history = model.fit(X_train, y_train, **fit_args)

    test_scaled_pred = model.predict(X_test, verbose=0)
    test_pred = y_scaler.inverse_transform(test_scaled_pred).ravel()
    test_actual = work[target_col].to_numpy()[test_positions]
    metrics = metric_values(test_actual, test_pred)

    # Future predictor scenario: hold each selected predictor at its latest known value.
    # This is a baseline assumption, not a claim about actual future operations.
    sequence_history = x_scaled.copy()
    future_predictions = []
    last_date = pd.Timestamp(work["Month"].iloc[-1])
    future_months = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=int(horizon), freq="MS")

    for _ in range(int(horizon)):
        seq = sequence_history[-lookback:].reshape(1, lookback, len(feature_cols))
        next_scaled = float(model.predict(seq, verbose=0)[0, 0])
        next_kwh = float(y_scaler.inverse_transform([[next_scaled]])[0, 0])

        next_row = sequence_history[-1].copy()
        next_row[0] = next_scaled
        # Keep predictor columns at the latest observed values.
        sequence_history = np.vstack([sequence_history, next_row])
        future_predictions.append(next_kwh)

    evaluation = pd.DataFrame({
        "Month": work["Month"].iloc[test_positions].to_numpy(),
        "Actual_kWh": test_actual,
        "Predicted_kWh": test_pred,
        "Absolute_Error_kWh": np.abs(test_actual - test_pred),
    })
    forecast = pd.DataFrame({"Month": future_months, "Forecast_kWh": future_predictions})
    return model, metrics, evaluation, forecast, history


def header(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("## 📈 Load Forecast Studio")
    st.caption("LSTM Edition")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Home", "Forecast & Evaluation", "How to Use", "About"])
    st.markdown("---")
    st.caption("Monthly electricity consumption forecasting")
    st.caption(f"Version {APP_VERSION}")


if "prepared_df" not in st.session_state:
    st.session_state.prepared_df = None
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None


# ----------------------------- HOME -----------------------------
if page == "Home":
    header("Understand your electricity consumption.",
           "Prepare monthly electricity and operational records, then generate an LSTM-based forecast.")

    st.write(
        "Upload one combined spreadsheet containing monthly kWh and operational variables, "
        "or upload a separate file for additional variables. The app uses kWh as its forecasting target."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("### 1. Main dataset")
        main_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="main_upload",
            help="Required: month/date, kWh consumption, and any other columns you want to keep in the same file.",
        )
    with right:
        st.markdown("### 2. Additional variables (optional)")
        extra_file = st.file_uploader(
            "Upload a second CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="extra_upload",
            help="Optional. If uploaded, include Month and optionally Year so rows can be matched by month.",
        )

    if main_file:
        try:
            main_df = clean_columns(read_file(main_file))
            st.session_state.main_df = main_df
            st.success(f"Main dataset loaded: {len(main_df)} rows and {len(main_df.columns)} columns.")
        except Exception as exc:
            st.error(f"Could not read the main dataset: {exc}")
            st.session_state.main_df = None

    if extra_file:
        try:
            st.session_state.extra_df = clean_columns(read_file(extra_file))
            st.success(f"Additional dataset loaded: {len(st.session_state.extra_df)} rows.")
        except Exception as exc:
            st.error(f"Could not read the additional dataset: {exc}")
            st.session_state.extra_df = None

    main_df = st.session_state.get("main_df")
    if main_df is not None:
        st.markdown("### Preview main dataset")
        st.caption(f"Showing all {len(main_df):,} uploaded rows and {len(main_df.columns)} columns.")
        st.dataframe(main_df, use_container_width=True, height=420)

        columns = list(main_df.columns)
        month_candidates = [c for c in columns if "month" in c.lower() or "date" in c.lower()]
        year_candidates = [c for c in columns if c.lower() == "year" or "year" in c.lower()]

        # Show the year coverage so multi-year files can be checked before preparation.
        detected_years = None
        if year_candidates:
            year_values = numeric_clean(main_df[year_candidates[0]]).dropna().astype(int)
            if not year_values.empty:
                detected_years = year_values.value_counts().sort_index()
        elif month_candidates:
            parsed_dates = pd.to_datetime(main_df[month_candidates[0]], errors="coerce")
            date_years = parsed_dates.dt.year.dropna()
            if not date_years.empty:
                detected_years = date_years.value_counts().sort_index()

        if detected_years is not None:
            st.markdown("#### Uploaded year coverage")
            coverage = pd.DataFrame({
                "Year": detected_years.index.astype(str),
                "Rows": detected_years.values
            })
            st.dataframe(coverage, hide_index=True, use_container_width=True)
            st.caption("Check that every expected year (for example, 2024 and 2025) appears before preparing the dataset.")
        else:
            st.info("Year coverage will be confirmed after you select the Month and Year columns below.")

        target_candidates = [c for c in columns if "kwh" in c.lower() or "consumption" in c.lower()]

        c1, c2, c3 = st.columns(3)
        with c1:
            month_col = st.selectbox("Month/date column", columns,
                                     index=columns.index(month_candidates[0]) if month_candidates else 0)
        with c2:
            year_options = ["None"] + columns
            year_default = year_candidates[0] if year_candidates else "None"
            year_col_choice = st.selectbox("Separate Year column (if needed)", year_options,
                                           index=year_options.index(year_default))
            year_col = None if year_col_choice == "None" else year_col_choice
        with c3:
            target_default = target_candidates[0] if target_candidates else columns[-1]
            target_col = st.selectbox("Electricity consumption (kWh)", columns,
                                      index=columns.index(target_default))

        extra_df = st.session_state.get("extra_df")
        extra_month_col = None
        extra_year_col = None
        if extra_df is not None:
            st.markdown("### Preview additional variables")
            st.dataframe(extra_df.head(8), use_container_width=True)
            extra_cols = list(extra_df.columns)
            extra_month_candidates = [c for c in extra_cols if "month" in c.lower() or "date" in c.lower()]
            extra_year_candidates = [c for c in extra_cols if "year" in c.lower()]
            if extra_month_candidates:
                ec1, ec2 = st.columns(2)
                with ec1:
                    extra_month_col = st.selectbox("Additional file Month/date column", extra_cols,
                                                    index=extra_cols.index(extra_month_candidates[0]))
                with ec2:
                    ey_options = ["None"] + extra_cols
                    ey_default = extra_year_candidates[0] if extra_year_candidates else "None"
                    ey_choice = st.selectbox("Additional file Year column", ey_options,
                                             index=ey_options.index(ey_default))
                    extra_year_col = None if ey_choice == "None" else ey_choice
            else:
                st.warning("No Month/date column found in the additional file. Rows will be matched by row order.")

        if st.button("Validate and Prepare Dataset", type="primary", use_container_width=True):
            try:
                prepared = prepare_dataset(main_df, month_col, target_col, year_col,
                                           extra_df, extra_month_col, extra_year_col)
                st.session_state.prepared_df = prepared
                st.session_state.evaluation_result = None
                st.session_state.forecast_result = None
                st.success("Dataset successfully prepared.")
            except Exception as exc:
                st.session_state.prepared_df = None
                st.error(f"Dataset validation failed: {exc}")

    prepared = st.session_state.prepared_df
    if prepared is not None:
        st.markdown("### Prepared monthly dataset")
        st.dataframe(prepared, use_container_width=True)
        st.success(
            f"{len(prepared)} monthly records | "
            f"{prepared['Month'].min():%b %Y} to {prepared['Month'].max():%b %Y}"
        )
        expected = pd.date_range(prepared["Month"].min(), prepared["Month"].max(), freq="MS")
        missing = expected.difference(pd.DatetimeIndex(prepared["Month"]))
        if len(missing):
            st.warning("Missing months detected: " + ", ".join(x.strftime("%b %Y") for x in missing))
        else:
            st.success("No missing months detected in the covered period.")
        st.info("Next: open Forecast & Evaluation, choose predictors, and configure the LSTM.")


# ----------------------------- FORECAST -----------------------------
elif page == "Forecast & Evaluation":
    header("Forecast & Evaluation",
           "Train the LSTM, inspect historical test performance, and estimate future monthly kWh.")

    data = st.session_state.prepared_df
    if data is None:
        st.warning("Prepare your dataset first on the Home page.")
        st.stop()

    possible_features = [
        c for c in data.columns
        if c not in ["Month", "Consumption_kWh"]
        and pd.api.types.is_numeric_dtype(numeric_clean(data[c]))
        and numeric_clean(data[c]).notna().any()
    ]

    st.markdown("### 1. Select operational predictors")
    st.caption(
        "Choose only variables that are relevant and consistently recorded. "
        "Avoid using electricity amount or cost per kWh as predictors unless you have a clear research reason."
    )
    selected_features = []
    if possible_features:
        st.caption("Tick the variables you want the LSTM to use. You can select all, a few, or just one.")
        select_all_col, clear_col, count_col = st.columns([1, 1, 2])

        # Set checkbox states before creating the checkbox widgets.
        with select_all_col:
            if st.button("Select all", use_container_width=True, key="select_all_predictors"):
                for feature in possible_features:
                    st.session_state[f"predictor_choice_{feature}"] = True
        with clear_col:
            if st.button("Deselect all", use_container_width=True, key="clear_all_predictors"):
                for feature in possible_features:
                    st.session_state[f"predictor_choice_{feature}"] = False

        feature_cols = st.columns(3)
        for i, feature in enumerate(possible_features):
            state_key = f"predictor_choice_{feature}"
            if state_key not in st.session_state:
                st.session_state[state_key] = False
            with feature_cols[i % 3]:
                checked = st.checkbox(str(feature), key=state_key)
                if checked:
                    selected_features.append(feature)

        with count_col:
            st.metric("Selected variables", f"{len(selected_features)} / {len(possible_features)}")
        if selected_features:
            st.caption("Included: " + ", ".join(map(str, selected_features)))
        else:
            st.info("No additional predictors selected. The model will use historical kWh only.")
    else:
        st.info("No numeric predictors found. The model will use historical kWh only.")

    st.markdown("### 2. LSTM settings")
    a, b, c = st.columns(3)
    with a:
        lookback = st.number_input("Lookback (months)", min_value=2, max_value=24, value=12, step=1)
    with b:
        default_test = min(6, max(2, len(data) // 5))
        test_size = st.number_input("Test months", min_value=1, max_value=max(1, len(data) - 8),
                                    value=min(default_test, max(1, len(data) - 8)), step=1)
    with c:
        horizon = st.selectbox("Forecast horizon", [1, 2, 3, 6, 9, 12], index=5)

    d, e, f = st.columns(3)
    with d:
        epochs = st.number_input("Maximum epochs", min_value=10, max_value=300, value=100, step=10)
    with e:
        units = st.selectbox("LSTM units", [16, 24, 32, 48, 64], index=3)
    with f:
        dropout = st.slider("Dropout rate", 0.0, 0.5, 0.1, 0.05)

    st.markdown("### 3. Evaluation and future-data assumptions")
    st.info(
        "Historical evaluation is a rolling one-step-ahead test: each test month is predicted using "
        "the preceding observed months. For the future forecast, selected operational predictors are "
        "held at their latest observed values. This is a baseline scenario only; it does not predict "
        "future manpower, production, operating hours, LPG, or diesel consumption."
    )

    if st.button("Train LSTM and Generate Forecast", type="primary", use_container_width=True):
        try:
            if len(data) <= int(lookback) + int(test_size) + 5:
                raise ValueError("There are too few records for the selected lookback and test size. Reduce one or upload more months.")
            with st.spinner("Training LSTM model..."):
                model, metrics, evaluation, forecast, history = train_lstm(
                    data, selected_features, int(lookback), int(test_size),
                    int(epochs), int(units), float(dropout), int(horizon)
                )
            st.session_state.evaluation_result = {
                "metrics": metrics,
                "evaluation": evaluation,
                "features": selected_features,
                "lookback": int(lookback),
                "test_size": int(test_size),
            }
            st.session_state.forecast_result = forecast
            st.success("LSTM training and forecasting completed.")
        except Exception as exc:
            st.error(f"Unable to complete forecasting: {exc}")

    result = st.session_state.evaluation_result
    forecast = st.session_state.forecast_result
    if result is not None and forecast is not None:
        st.markdown("---")
        st.markdown("### Model performance on the historical test period")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{result['metrics']['MAE']:,.2f} kWh")
        m2.metric("RMSE", f"{result['metrics']['RMSE']:,.2f} kWh")
        m3.metric("MAPE", "N/A" if np.isnan(result["metrics"]["MAPE"]) else f"{result['metrics']['MAPE']:.2f}%")

        with st.expander("How should I interpret these metrics?"):
            st.markdown("""
            - **MAE:** average absolute prediction error, expressed in kWh. Lower is better.
            - **RMSE:** gives greater weight to larger errors. Lower is better.
            - **MAPE:** average absolute percentage error. It is not defined for actual values equal to zero.

            These scores describe the selected historical test period. They do not guarantee future accuracy.
            """)

        evaluation = result["evaluation"].copy()
        st.markdown("### Historical test results")
        display_eval = evaluation.copy()
        display_eval["Month"] = pd.to_datetime(display_eval["Month"]).dt.strftime("%B %Y")
        st.dataframe(display_eval, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=evaluation["Month"], y=evaluation["Actual_kWh"],
                                 mode="lines+markers", name="Actual kWh"))
        fig.add_trace(go.Scatter(x=evaluation["Month"], y=evaluation["Predicted_kWh"],
                                 mode="lines+markers", name="Predicted kWh"))
        fig.update_layout(title="Actual vs. Predicted — Historical Test Period",
                          xaxis_title="Month", yaxis_title="Consumption (kWh)",
                          template="plotly_white", height=410,
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Future monthly forecast")
        display_forecast = forecast.copy()
        display_forecast["Month"] = pd.to_datetime(display_forecast["Month"]).dt.strftime("%B %Y")
        st.dataframe(display_forecast, use_container_width=True)

        future_fig = go.Figure()
        future_fig.add_trace(go.Scatter(x=data["Month"], y=data["Consumption_kWh"],
                                        mode="lines+markers", name="Historical kWh"))
        future_fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["Forecast_kWh"],
                                        mode="lines+markers", name="Forecast kWh",
                                        line=dict(dash="dash")))
        future_fig.update_layout(title="Historical Consumption and Future Forecast",
                                 xaxis_title="Month", yaxis_title="Consumption (kWh)",
                                 template="plotly_white", height=430,
                                 legend=dict(orientation="h", y=1.12))
        st.plotly_chart(future_fig, use_container_width=True)

        st.download_button(
            "Download forecast (CSV)",
            forecast.to_csv(index=False).encode("utf-8"),
            "lstm_monthly_forecast.csv",
            "text/csv",
        )
        st.download_button(
            "Download test results (CSV)",
            evaluation.to_csv(index=False).encode("utf-8"),
            "lstm_test_results.csv",
            "text/csv",
        )


# ----------------------------- USER GUIDE -----------------------------
elif page == "How to Use":
    header("User Guide", "Prepare your monthly records and run an LSTM forecast.")
    st.markdown("""
    ### Recommended dataset structure
    Use one row per month. Recommended columns include:
    - `Year` and `Month` (or one date column such as `2022-01`)
    - Electricity consumption in kWh (required target)
    - Total manpower
    - Production volume (kg)
    - LPG consumption (kg)
    - Operating time (hours)
    - Non-operating time (hours)
    - Diesel consumption
    - Peak demand (kW), if available

    The app accepts CSV, XLSX, and XLS files. A separate additional-variables file is optional.

    ### Important preparation checks
    1. Use one record per month.
    2. Keep the records chronological or provide clear Month and Year columns.
    3. Check that kWh values are numeric and nonnegative.
    4. Check for missing months, duplicate months, and inconsistent units.
    5. Select only predictors with enough reliable historical observations.

    ### Forecasting assumptions
    The historical test uses rolling one-step-ahead predictions. For future forecasting, selected predictors
    are held at their latest observed values. Therefore, future operational changes are not represented unless
    future predictor scenarios are added to the application.
    """)


# ----------------------------- ABOUT -----------------------------
elif page == "About":
    header("About Load Forecast Studio", "LSTM edition for monthly electricity consumption forecasting.")
    st.markdown("""
    ### Purpose
    This application prepares monthly electricity and operational data, trains a Long Short-Term Memory
    (LSTM) model, evaluates historical predictions, and generates future monthly kWh estimates.

    ### Forecasting target
    The target variable is monthly electricity consumption in kWh. Peak demand in kW, electricity cost,
    manpower, production, operating hours, and fuel consumption are separate variables and are not
    interchangeable with the kWh target.

    ### Responsible use
    Forecasts are estimates and should be interpreted with the selected test period, data quality,
    operational assumptions, and limitations in mind. Do not upload confidential information unless
    you are authorized to use it.
    """)
    st.caption("Load Forecast Studio — LSTM Edition")
