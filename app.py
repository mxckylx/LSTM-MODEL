import re
import warnings
from calendar import month_name
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Load Forecast Studio | LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# DESIGN
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7fa; }
    [data-testid="stSidebar"] { background: #10243a; }
    [data-testid="stSidebar"] * { color: #f4f7fb !important; }
    [data-testid="stSidebar"] .stRadio label { padding: 7px 10px; }
    .hero {
        background: linear-gradient(115deg, #10243a, #183b55);
        padding: 28px 30px; border-radius: 18px; color: white;
        margin-bottom: 20px;
    }
    .hero h1 { color: white; margin: 0 0 8px 0; font-size: 30px; }
    .hero p { color: #d7e5ef; margin: 0; font-size: 15px; }
    .section-card {
        background: white; padding: 20px; border: 1px solid #e5eaf0;
        border-radius: 15px; margin-bottom: 14px;
    }
    .small-muted { color: #657589; font-size: 13px; }
    div[data-testid="stMetric"] {
        background: white; padding: 14px; border: 1px solid #e5eaf0;
        border-radius: 13px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# CONSTANTS / SESSION STATE
# ------------------------------------------------------------
APP_VERSION = "1.0"
MONTH_LOOKUP = {m.lower(): i for i, m in enumerate(month_name) if m}
MONTH_LOOKUP.update({m[:3].lower(): i for i, m in enumerate(month_name) if m})

for key, default in {
    "main_df": None,
    "extra_df": None,
    "prepared_df": None,
    "forecast_result": None,
    "evaluation_result": None,
    "trained_model": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------
# DATA HELPERS
# ------------------------------------------------------------
def read_uploaded_file(uploaded_file):
    """Read CSV or Excel upload and return a DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS.")


def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def month_to_period(value, previous_month=None, previous_year=2000):
    """
    Convert common month/date values into a monthly Period.
    If only month names are supplied, infer year rollover from row order.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Period(value, freq="M")

    text = str(value).strip()
    if not text:
        return None

    # Full date / year-month formats
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        # Avoid pandas interpreting a bare month name using today's year.
        if re.search(r"\d{4}", text) or re.search(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", text):
            return pd.Period(parsed, freq="M")

    # Month name or abbreviation, e.g. January / Jan
    token = re.sub(r"[^a-z]", "", text.lower())
    if token in MONTH_LOOKUP:
        month_num = MONTH_LOOKUP[token]
        year = previous_year
        if previous_month is not None and month_num < previous_month:
            year += 1
        return pd.Period(year=year, month=month_num, freq="M")

    # Numeric month, e.g. 1 through 12
    if re.fullmatch(r"\d{1,2}", text):
        month_num = int(text)
        if 1 <= month_num <= 12:
            year = previous_year
            if previous_month is not None and month_num < previous_month:
                year += 1
            return pd.Period(year=year, month=month_num, freq="M")

    return None


def build_month_periods(series):
    values = []
    previous_month = None
    current_year = 2000
    for value in series:
        period = month_to_period(value, previous_month, current_year)
        if period is None:
            values.append(None)
            continue
        if previous_month is not None and period.month < previous_month and period.year == current_year:
            current_year += 1
            period = pd.Period(year=current_year, month=period.month, freq="M")
        else:
            current_year = period.year
        previous_month = period.month
        values.append(period)
    return pd.PeriodIndex(values, freq="M")


def numeric_clean(series):
    """Convert numeric-like text to numeric values."""
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
        cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def make_monthly_dataset(main_df, month_col, target_col, extra_df=None, extra_month_col=None):
    main = clean_column_names(main_df)
    month_col = str(month_col)
    target_col = str(target_col)

    periods = build_month_periods(main[month_col])
    if periods.isna().any():
        bad_rows = list(np.where(periods.isna())[0] + 1)
        raise ValueError(
            "Some Month values could not be recognized. Check these spreadsheet row numbers: "
            + ", ".join(map(str, bad_rows[:10]))
        )

    data = pd.DataFrame({
        "Month": periods.astype(str),
        "Consumption_kWh": numeric_clean(main[target_col]).values,
    })

    if data["Consumption_kWh"].isna().any():
        raise ValueError("The kWh column contains blank or non-numeric values. Please correct them first.")

    if extra_df is not None and len(extra_df.columns) > 0:
        extra = clean_column_names(extra_df)
        if extra_month_col and extra_month_col in extra.columns:
            extra_periods = build_month_periods(extra[extra_month_col])
            if extra_periods.isna().any():
                raise ValueError("The Month column in Additional Variables contains unrecognized values.")
            extra = extra.drop(columns=[extra_month_col])
            extra.insert(0, "_MonthKey", extra_periods.astype(str))
            data["_MonthKey"] = periods.astype(str)
            if extra["_MonthKey"].duplicated().any():
                raise ValueError("Additional Variables has duplicate Month entries. Each month should appear once.")
            data = data.merge(extra, left_on="_MonthKey", right_on="_MonthKey", how="left", validate="one_to_one")
            data = data.drop(columns=["_MonthKey"])
        else:
            if len(extra) != len(data):
                raise ValueError(
                    "The number of rows in Additional Variables must match the Main Dataset "
                    "when no Month column is available for matching."
                )
            # Row-order matching is permitted only when there is no Month column.
            for col in extra.columns:
                data[col] = extra[col].values

    data["Month"] = pd.PeriodIndex(data["Month"], freq="M").to_timestamp()
    data = data.sort_values("Month").reset_index(drop=True)

    if data["Month"].duplicated().any():
        raise ValueError("The Main Dataset contains duplicate months. Please keep one record per month.")

    return data


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def build_sequences(features, target_scaled, lookback, target_start=0, target_end=None):
    X, y, target_positions = [], [], []
    if target_end is None:
        target_end = len(features)
    for target_pos in range(max(lookback, target_start), target_end):
        X.append(features[target_pos - lookback:target_pos])
        y.append(target_scaled[target_pos, 0])
        target_positions.append(target_pos)
    return np.asarray(X), np.asarray(y), np.asarray(target_positions)


def train_lstm(data, target_col, selected_features, lookback, test_size, epochs, units, dropout):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be loaded. Check that tensorflow is included in requirements.txt "
            "and that the deployment environment supports it."
        ) from exc

    working = data[["Month", target_col] + selected_features].copy()
    working[target_col] = numeric_clean(working[target_col])
    for col in selected_features:
        working[col] = numeric_clean(working[col])

    # Fill gaps in optional predictors using time-based interpolation, then median.
    for col in selected_features:
        working[col] = working[col].interpolate(limit_direction="both")
        working[col] = working[col].fillna(working[col].median())

    if working[[target_col] + selected_features].isna().any().any():
        raise ValueError("The selected variables contain missing or invalid values that could not be filled.")

    n = len(working)
    split = n - test_size
    if split <= lookback:
        raise ValueError(
            f"Not enough training history. With {n} rows and a lookback of {lookback}, "
            "increase the dataset size or reduce the lookback/test size."
        )

    feature_cols = [target_col] + selected_features
    raw_features = working[feature_cols].to_numpy(dtype=float)
    target_values = working[[target_col]].to_numpy(dtype=float)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(raw_features[:split])
    target_scaler.fit(target_values[:split])

    scaled_features = feature_scaler.transform(raw_features)
    scaled_target = target_scaler.transform(target_values)

    X_train, y_train, _ = build_sequences(scaled_features, scaled_target, lookback, lookback, split)
    X_test, y_test, test_positions = build_sequences(scaled_features, scaled_target, lookback, split, n)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Unable to create training and test sequences. Adjust the lookback or test observations.")

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(lookback, len(feature_cols))),
        Dropout(dropout),
        LSTM(max(8, units // 2)),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    validation_split = 0.15 if len(X_train) >= 10 else 0.0
    fit_kwargs = {
        "epochs": epochs,
        "batch_size": min(16, max(1, len(X_train))),
        "verbose": 0,
        "callbacks": [early_stop],
    }
    if validation_split > 0:
        fit_kwargs["validation_split"] = validation_split

    model.fit(X_train, y_train, **fit_kwargs)

    test_scaled_pred = model.predict(X_test, verbose=0)
    test_pred = target_scaler.inverse_transform(test_scaled_pred).flatten()
    test_actual = working[target_col].to_numpy()[test_positions]
    metrics = calculate_metrics(test_actual, test_pred)

    # Recursive multi-step forecast. Optional future predictors are held at their latest
    # observed values; this assumption is clearly disclosed in the interface.
    history = scaled_features.copy()
    future_values = []
    last_date = pd.Timestamp(working["Month"].iloc[-1])
    future_months = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=12, freq="MS")

    for _ in range(12):
        sequence = history[-lookback:].reshape(1, lookback, len(feature_cols))
        next_scaled_target = float(model.predict(sequence, verbose=0)[0, 0])
        next_target = float(target_scaler.inverse_transform([[next_scaled_target]])[0, 0])

        next_row = history[-1].copy()
        next_row[0] = next_scaled_target
        history = np.vstack([history, next_row])
        future_values.append(next_target)

    forecast_df = pd.DataFrame({
        "Month": future_months,
        "Forecast_kWh": future_values,
    })
    evaluation_df = pd.DataFrame({
        "Month": working["Month"].iloc[test_positions].values,
        "Actual_kWh": test_actual,
        "Predicted_kWh": test_pred,
    })

    return model, metrics, evaluation_df, forecast_df, feature_scaler, target_scaler


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Load Forecast Studio")
    st.caption("LSTM Edition")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Home", "Forecast & Evaluation", "How to Use", "About"],
        label_visibility="visible",
        key="navigation_page",
    )
    st.markdown("---")
    st.caption("Monthly electricity consumption forecasting")
    st.caption(f"Version {APP_VERSION}")


# ------------------------------------------------------------
# REUSABLE HEADER
# ------------------------------------------------------------
def page_header(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# HOME / UPLOAD
# ------------------------------------------------------------
if page == "Home":
    page_header(
        "Understand your electricity consumption.",
        "Prepare historical monthly data, explore patterns, and generate an LSTM-based forecast.",
    )

    st.markdown("### Getting started")
    st.write(
        "Load Forecast Studio is a forecasting tool for users who want to examine historical "
        "electricity consumption and estimate future monthly consumption. You do not need to "
        "know how to program. Prepare your spreadsheet, upload it, identify the correct columns, "
        "and follow the instructions shown in each section."
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Main Dataset")
        st.write("Upload one CSV or Excel file containing:")
        st.markdown("- A **Month** column (for example, January, February, or 2024-01)")
        st.markdown("- An **Electricity Consumption (kWh)** column")
        st.caption("Keep one record per month and arrange the records chronologically.")
        main_file = st.file_uploader(
            "Upload Main Dataset",
            type=["csv", "xlsx", "xls"],
            key="main_dataset_upload",
            help="This file must contain both the month and the monthly kWh consumption.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 2. Additional Variables (Optional)")
        st.write("Upload a separate file for other monthly information, such as:")
        st.markdown("- Manpower or production volume")
        st.markdown("- Operating hours or non-operating hours")
        st.markdown("- Peak demand (kW), electricity cost, or fuel consumption")
        st.caption("Include a Month column to match each row by month. Otherwise, rows must be in the same order as the Main Dataset.")
        extra_file = st.file_uploader(
            "Upload Additional Variables",
            type=["csv", "xlsx", "xls"],
            key="additional_variables_upload",
            help="Optional. You can choose which numeric variables to use later.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Read files only when uploaded; widget keys are never reassigned.
    if main_file is not None:
        try:
            st.session_state["main_df"] = clean_column_names(read_uploaded_file(main_file))
            st.success(f"Main Dataset loaded: {len(st.session_state['main_df'])} rows.")
        except Exception as exc:
            st.error(f"Unable to read the Main Dataset: {exc}")
            st.session_state["main_df"] = None

    if extra_file is not None:
        try:
            st.session_state["extra_df"] = clean_column_names(read_uploaded_file(extra_file))
            st.success(f"Additional Variables loaded: {len(st.session_state['extra_df'])} rows.")
        except Exception as exc:
            st.error(f"Unable to read Additional Variables: {exc}")
            st.session_state["extra_df"] = None

    main_df = st.session_state["main_df"]
    if main_df is not None:
        st.markdown("### Review your Main Dataset")
        st.dataframe(main_df.head(10), use_container_width=True)
        st.caption(f"Detected columns: {', '.join(map(str, main_df.columns))}")

        col1, col2 = st.columns(2)
        with col1:
            month_default = next((i for i, c in enumerate(main_df.columns) if "month" in str(c).lower() or "date" in str(c).lower()), 0)
            month_col = st.selectbox(
                "Select the Month column",
                options=list(main_df.columns),
                index=month_default,
                key="main_month_column_choice",
                help="Choose the column containing month names or dates.",
            )
        with col2:
            target_candidates = [c for c in main_df.columns if "kwh" in str(c).lower() or "consumption" in str(c).lower()]
            target_default = list(main_df.columns).index(target_candidates[0]) if target_candidates else min(1, len(main_df.columns)-1)
            target_col = st.selectbox(
                "Select the electricity consumption column",
                options=list(main_df.columns),
                index=target_default,
                key="main_consumption_column_choice",
                help="Choose the column containing monthly electricity consumption in kWh.",
            )

        extra_df = st.session_state["extra_df"]
        extra_month_col = None
        if extra_df is not None:
            st.markdown("### Review Additional Variables")
            st.dataframe(extra_df.head(10), use_container_width=True)
            extra_month_candidates = [c for c in extra_df.columns if "month" in str(c).lower() or "date" in str(c).lower()]
            if extra_month_candidates:
                extra_month_col = st.selectbox(
                    "Select the Month column in Additional Variables",
                    options=list(extra_df.columns),
                    index=list(extra_df.columns).index(extra_month_candidates[0]),
                    key="extra_month_column_choice",
                    help="This allows the app to match the additional variables to the correct month.",
                )
            else:
                st.info("No Month column was detected in Additional Variables. The app will match rows by their order.")

        if st.button("Validate and Prepare Dataset", type="primary", use_container_width=True):
            try:
                prepared = make_monthly_dataset(main_df, month_col, target_col, extra_df, extra_month_col)
                st.session_state["prepared_df"] = prepared
                st.session_state["forecast_result"] = None
                st.session_state["evaluation_result"] = None
                st.success("Your dataset has been validated and prepared.")
            except Exception as exc:
                st.session_state["prepared_df"] = None
                st.error(f"Dataset validation failed: {exc}")

    if st.session_state["prepared_df"] is not None:
        prepared = st.session_state["prepared_df"]
        st.markdown("### Prepared dataset")
        st.dataframe(prepared, use_container_width=True)
        st.success(
            f"Ready for forecasting: {len(prepared)} monthly records, "
            f"from {prepared['Month'].min():%b %Y} to {prepared['Month'].max():%b %Y}."
        )
        st.info("Next step: open **Forecast & Evaluation** from the sidebar to select variables and configure the LSTM model.")


# ------------------------------------------------------------
# FORECAST / EVALUATION
# ------------------------------------------------------------
elif page == "Forecast & Evaluation":
    page_header(
        "Forecast & Evaluation",
        "Select the variables to include, configure the model, and review its test performance.",
    )

    data = st.session_state["prepared_df"]
    if data is None:
        st.warning("Your dataset has not been prepared yet. Go to Home, upload your files, and select Validate and Prepare Dataset.")
        st.stop()

    st.markdown("### 1. Select additional variables")
    st.write(
        "The kWh consumption column is the forecasting target. Additional variables are optional. "
        "Select only the numeric variables that are relevant to your dataset and available consistently over time."
    )
    possible_features = [
        c for c in data.columns
        if c != "Month" and c != "Consumption_kWh"
        and pd.api.types.is_numeric_dtype(numeric_clean(data[c]))
        and numeric_clean(data[c]).notna().any()
    ]

    selected_features = []
    if possible_features:
        feature_cols = st.columns(2)
        for i, feature in enumerate(possible_features):
            with feature_cols[i % 2]:
                if st.checkbox(str(feature), value=True, key=f"feature_select_{i}"):
                    selected_features.append(feature)
    else:
        st.info("No additional numeric variables are available. The LSTM will use historical kWh consumption only.")

    st.markdown("### 2. Model settings")
    a, b, c = st.columns(3)
    with a:
        lookback = st.number_input(
            "Lookback (months)", min_value=2, max_value=36, value=12, step=1,
            help="Number of previous months the model examines to estimate the next month.",
        )
    with b:
        test_size = st.number_input(
            "Test observations", min_value=1, max_value=24, value=min(3, max(1, len(data)//5)), step=1,
            help="Number of latest historical months reserved to evaluate the model.",
        )
    with c:
        forecast_horizon = st.selectbox(
            "Forecast horizon", options=[1, 2, 3, 6, 9, 12], index=5,
            help="Number of future months to forecast.",
        )

    d, e, f = st.columns(3)
    with d:
        epochs = st.number_input("Training epochs", min_value=10, max_value=300, value=100, step=10,
                                 help="Maximum number of learning cycles. Early stopping may end training sooner.")
    with e:
        units = st.selectbox("LSTM units", options=[16, 24, 32, 48, 64, 96], index=3,
                             help="Number of units in the first LSTM layer. Larger values may require more data and computing time.")
    with f:
        dropout = st.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05,
                            help="Regularization setting that helps reduce overfitting.")

    st.markdown("### 3. Important forecasting assumptions")
    st.info(
        "The evaluation uses the latest selected historical observations as a holdout test set. "
        "For each test month, the model predicts that month using the preceding observed history. "
        "For future months, optional variables are assumed to remain at their latest observed values "
        "because future values have not been supplied. These assumptions should be considered when interpreting results."
    )

    if st.button("Train LSTM and Generate Forecast", type="primary", use_container_width=True):
        try:
            if len(data) <= int(lookback) + int(test_size):
                raise ValueError(
                    f"Not enough monthly records for the selected settings. You have {len(data)} records. "
                    "Reduce Lookback or Test observations, or upload more historical records."
                )
            with st.spinner("Training the LSTM model. This may take a few minutes..."):
                model, metrics, evaluation_df, forecast_df, feature_scaler, target_scaler = train_lstm(
                    data=data,
                    target_col="Consumption_kWh",
                    selected_features=selected_features,
                    lookback=int(lookback),
                    test_size=int(test_size),
                    epochs=int(epochs),
                    units=int(units),
                    dropout=float(dropout),
                )
                forecast_df = forecast_df.head(int(forecast_horizon))
                st.session_state["trained_model"] = model
                st.session_state["evaluation_result"] = {
                    "metrics": metrics,
                    "evaluation_df": evaluation_df,
                    "selected_features": selected_features,
                    "lookback": int(lookback),
                    "test_size": int(test_size),
                }
                st.session_state["forecast_result"] = forecast_df
            st.success("Model training and forecasting completed.")
        except Exception as exc:
            st.error(f"Unable to complete forecasting: {exc}")

    evaluation = st.session_state["evaluation_result"]
    forecast = st.session_state["forecast_result"]
    if evaluation is not None and forecast is not None:
        st.markdown("---")
        st.markdown("### Model evaluation")
        st.write("The following metrics compare the model's predictions with actual values in the historical test set.")
        m1, m2, m3 = st.columns(3)
        metrics = evaluation["metrics"]
        m1.metric("MAE", f"{metrics['MAE']:,.2f} kWh")
        m2.metric("RMSE", f"{metrics['RMSE']:,.2f} kWh")
        m3.metric("MAPE", "N/A" if np.isnan(metrics["MAPE"]) else f"{metrics['MAPE']:.2f}%")

        with st.expander("What do these evaluation metrics mean?"):
            st.markdown(
                """
                - **MAE (Mean Absolute Error):** the average absolute difference between actual and predicted kWh values. Lower is better.
                - **RMSE (Root Mean Squared Error):** measures prediction error while giving larger errors more influence. Lower is better.
                - **MAPE (Mean Absolute Percentage Error):** expresses the average absolute prediction error as a percentage of actual values. Lower is better. It is not calculated for actual values equal to zero.

                These metrics describe performance on the selected historical test observations. They do not guarantee the same level of accuracy for future months.
                """
            )

        st.markdown("### Historical test results")
        eval_df = evaluation["evaluation_df"].copy()
        eval_df["Month"] = pd.to_datetime(eval_df["Month"])
        st.dataframe(eval_df.assign(Month=eval_df["Month"].dt.strftime("%B %Y")), use_container_width=True)

        chart = go.Figure()
        chart.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["Actual_kWh"], mode="lines+markers", name="Actual kWh"))
        chart.add_trace(go.Scatter(x=eval_df["Month"], y=eval_df["Predicted_kWh"], mode="lines+markers", name="Predicted kWh"))
        chart.update_layout(
            title="Actual vs Predicted Consumption — Test Period",
            xaxis_title="Month", yaxis_title="Electricity Consumption (kWh)",
            template="plotly_white", height=400, legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(chart, use_container_width=True)

        st.markdown("### Future monthly forecast")
        forecast_display = forecast.copy()
        forecast_display["Month"] = pd.to_datetime(forecast_display["Month"]).dt.strftime("%B %Y")
        st.dataframe(forecast_display, use_container_width=True)

        future_chart = go.Figure()
        historical = data[["Month", "Consumption_kWh"]].copy()
        future_chart.add_trace(go.Scatter(
            x=historical["Month"], y=historical["Consumption_kWh"],
            mode="lines+markers", name="Historical kWh"
        ))
        future_chart.add_trace(go.Scatter(
            x=forecast["Month"], y=forecast["Forecast_kWh"],
            mode="lines+markers", name="Forecast kWh", line=dict(dash="dash")
        ))
        future_chart.update_layout(
            title="Historical Consumption and Future Forecast",
            xaxis_title="Month", yaxis_title="Electricity Consumption (kWh)",
            template="plotly_white", height=430, legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(future_chart, use_container_width=True)

        csv_forecast = forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast (CSV)",
            data=csv_forecast,
            file_name="lstm_monthly_forecast.csv",
            mime="text/csv",
        )
        csv_eval = evaluation["evaluation_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Test Results (CSV)",
            data=csv_eval,
            file_name="lstm_test_results.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# HOW TO USE
# ------------------------------------------------------------
elif page == "How to Use":
    page_header(
        "User Guide",
        "A step-by-step reference for preparing data, running a forecast, and interpreting the results.",
    )

    st.markdown("### Before you begin")
    st.markdown(
        """
        1. Prepare a historical monthly dataset in CSV, XLSX, or XLS format.
        2. Ensure that each month has one electricity consumption value in kWh.
        3. Arrange the records from oldest to newest.
        4. Check that there are no duplicate months and that kWh values are numeric.
        5. If you have additional variables, prepare them in a separate file. Include a Month column so the app can match records by month.
        """
    )

    st.markdown("### Upload file structure")
    st.markdown("**Main Dataset — required**")
    st.code(
        "Month,ENERGY CONSUMPTION (kWh)\n"
        "January,58880\n"
        "February,69440\n"
        "March,99360",
        language="text",
    )
    st.markdown("**Additional Variables — optional**")
    st.code(
        "Month,TOTAL MANPOWER,OPERATING TIME (hrs),PEAK DEMAND (kW)\n"
        "January,119,462,372.16\n"
        "February,121,396,350.007\n"
        "March,119,330,381.12",
        language="text",
    )
    st.caption("The values above are illustrative examples only. Use your own verified records.")

    st.markdown("### Running a forecast")
    st.markdown(
        """
        1. Open **Home** and upload the Main Dataset.
        2. Upload Additional Variables if available.
        3. Select the Month and electricity consumption columns.
        4. If the additional file has a Month column, select it so records can be matched by month.
        5. Select **Validate and Prepare Dataset**.
        6. Open **Forecast & Evaluation**.
        7. Select the additional numeric variables you want the model to use.
        8. Set the lookback, test observations, forecast horizon, and model settings.
        9. Select **Train LSTM and Generate Forecast**.
        10. Review the evaluation metrics, test results, forecast chart, and downloadable files.
        """
    )

    st.markdown("### Understanding the model settings")
    st.markdown(
        """
        - **Lookback (months):** how many previous months are used as input for the next prediction.
        - **Test observations:** how many of the most recent historical months are held out to evaluate the model.
        - **Forecast horizon:** how many future months are estimated.
        - **Training epochs:** the maximum number of times the model can learn from the training data. Early stopping may stop training earlier.
        - **LSTM units:** controls the capacity of the LSTM layers.
        - **Dropout rate:** a regularization setting intended to reduce overfitting.
        """
    )

    st.markdown("### Important limitations")
    st.warning(
        """
        - Forecast quality depends on the amount, quality, and consistency of the historical data.
        - A forecast is an estimate, not a guaranteed future electricity bill or demand value.
        - The test results are based on a holdout period and may not represent every future operating condition.
        - When future values for additional variables are not provided, the app carries forward their latest observed values.
        - Sudden changes in production, operating schedules, equipment, tariffs, weather, or other conditions may affect actual consumption.
        - The app forecasts monthly electricity consumption in kWh. It does not forecast peak demand in kW unless a separate target and model setup is developed for that purpose.
        """
    )


# ------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------
elif page == "About":
    page_header(
        "About Load Forecast Studio",
        "A practical interface for monthly electricity consumption forecasting using Long Short-Term Memory (LSTM).",
    )
    st.markdown(
        """
        ### Purpose
        Load Forecast Studio helps users organize historical monthly electricity consumption data,
        train an LSTM forecasting model, evaluate its performance, and view estimates for upcoming months.

        ### What is LSTM?
        Long Short-Term Memory (LSTM) is a type of recurrent neural network designed to learn patterns
        from sequential data. In this application, the sequence consists of monthly electricity
        consumption records and any selected additional numeric variables.

        ### Intended use
        The application can support preliminary analysis, planning, and research. Results should be
        reviewed alongside operational knowledge and other relevant information before being used
        for financial, engineering, or business decisions.

        ### Data responsibility
        Users are responsible for ensuring that uploaded data is accurate and that they have permission
        to use it. Avoid uploading confidential or personally identifiable information unless you are
        authorized to do so.
        """
    )
    st.markdown("---")
    st.caption("Load Forecast Studio — LSTM Edition")
