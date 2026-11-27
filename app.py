
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Load Forecast Studio | LSTM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F5F8FC;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102542 0%, #173B59 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.sidebar-brand {
    text-align: center;
    padding: 20px 5px 30px 5px;
}

.sidebar-brand h2 {
    color: white;
    font-size: 23px;
    font-weight: 800;
    line-height: 1.2;
}

.sidebar-brand p {
    color: #A9DDE0 !important;
    font-size: 12px;
}

.hero {
    background: linear-gradient(115deg, #FFFFFF 45%, #DDF8F7 100%);
    border: 1px solid #E4EDF4;
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #102542;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #53677C;
    font-size: 15px;
    margin-top: 8px;
}

.section-card {
    background: #FFFFFF;
    border: 1px solid #E5ECF3;
    border-radius: 17px;
    padding: 23px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(16, 37, 66, 0.04);
}

.section-title {
    color: #102542;
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-caption {
    color: #64788C;
    font-size: 13px;
    margin-bottom: 18px;
}

div[data-testid="stFileUploader"] {
    background: #F1F7FE;
    border: 1.5px dashed #8DB5D8;
    border-radius: 13px;
    padding: 12px;
}

div.stButton > button {
    background: #087F8C;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 650;
}

div.stButton > button:hover {
    background: #066875;
    color: white;
    border: none;
}

.metric-card {
    background: white;
    border: 1px solid #E5ECF3;
    border-radius: 14px;
    padding: 18px;
}

.metric-label {
    color: #718397;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #102542;
    font-size: 25px;
    font-weight: 800;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5ECF3;
    border-radius: 14px;
    padding: 15px;
}

h2, h3 {
    color: #102542;
}

small {
    color: #64788C;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def read_file(uploaded_file):
    if uploaded_file is None:
        return None

    extension = uploaded_file.name.lower().split(".")[-1]

    if extension == "csv":
        return pd.read_csv(uploaded_file)
    elif extension in ["xlsx", "xls"]:
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format.")


def make_sequences(features, target, lookback):
    X, y = [], []

    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = np.abs(actual) > 1e-8
    if np.any(nonzero):
        mape = np.mean(
            np.abs((actual[nonzero] - predicted[nonzero]) /
                   actual[nonzero])
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


def build_lstm(lookback, feature_count, units, dropout):
    model = Sequential([
        LSTM(
            units,
            input_shape=(lookback, feature_count),
            return_sequences=True
        ),
        Dropout(dropout),
        LSTM(max(8, units // 2)),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def make_chart(dates, actual, predicted=None, future_dates=None,
               future_values=None, title="Electricity Consumption"):
    fig = go.Figure()

    if actual is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode="lines+markers",
            name="Actual Consumption",
            line=dict(color="#087F8C", width=3)
        ))

    if predicted is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted,
            mode="lines+markers",
            name="Predicted Consumption",
            line=dict(color="#173B59", width=2, dash="dash")
        ))

    if future_dates is not None and future_values is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_values,
            mode="lines+markers",
            name="Future Forecast",
            line=dict(color="#00A6A6", width=3, dash="dot")
        ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(orientation="h", y=-0.2),
        xaxis_title="Date",
        yaxis_title="Electricity Consumption (kWh)",
        hovermode="x unified"
    )

    return fig


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:48px;">⚡</div>
        <h2>Load Forecast<br>Studio</h2>
        <p>Smarter Forecasts. A More Efficient Tomorrow.</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Forecast",
            "Rolling Test",
            "Model Comparison",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("LSTM EDITION")
    st.caption("Monthly and time-series forecasting")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div style="color:#087F8C;font-weight:700;font-size:12px;
                letter-spacing:2px;">LOAD FORECAST STUDIO</div>
    <h1>Electrical Load Forecasting</h1>
    <p>Upload your historical data, configure your model, and generate forecasts.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HOME / DATA UPLOAD
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="section-card">
        <div class="section-title">01 — Main Dataset</div>
        <div class="section-caption">
            Upload one file containing both your date/month and
            electricity consumption columns.
        </div>
    </div>
    """, unsafe_allow_html=True)

    main_file = st.file_uploader(
        "Upload Main Dataset",
        type=["csv", "xlsx", "xls"],
        key="main_upload",
        help="The file must contain a date/month column and a consumption column."
    )

    main_df = None

    if main_file is not None:
        try:
            main_df = read_file(main_file)
            st.success(f"Main dataset uploaded: {main_file.name}")

            st.markdown("#### Main Dataset Preview")
            st.dataframe(main_df, use_container_width=True)

            columns = main_df.columns.tolist()

            st.markdown("#### Identify Main Columns")

            col1, col2 = st.columns(2)

            with col1:
                date_column = st.selectbox(
                    "Select Date / Month Column",
                    columns,
                    key="date_column"
                )

            with col2:
                possible_targets = [
                    c for c in columns if c != date_column
                ]

                target_column = st.selectbox(
                    "Select Electricity Consumption Column (kWh)",
                    possible_targets,
                    key="target_column"
                )

            st.session_state["main_df"] = main_df
            st.session_state["date_column"] = date_column
            st.session_state["target_column"] = target_column

        except Exception as e:
            st.error(f"Unable to read the main dataset: {e}")

    st.markdown("""
    <div class="section-card">
        <div class="section-title">02 — Additional Variables (Optional)</div>
        <div class="section-caption">
            Upload one file containing other variables such as manpower,
            operating hours, production, peak demand, and diesel consumption.
        </div>
    </div>
    """, unsafe_allow_html=True)

    other_file = st.file_uploader(
        "Upload Additional Variables",
        type=["csv", "xlsx", "xls"],
        key="other_upload",
        help="Rows must follow the same date order as the main dataset."
    )

    if other_file is not None:
        try:
            other_df = read_file(other_file)
            st.success(f"Additional variables uploaded: {other_file.name}")

            st.markdown("#### Additional Variables Preview")
            st.dataframe(other_df, use_container_width=True)

            st.session_state["other_df"] = other_df

        except Exception as e:
            st.error(f"Unable to read the additional variables file: {e}")
    else:
        st.session_state["other_df"] = None

    st.info(
        "After uploading, go to Forecast to configure your dataset, "
        "choose optional variables, and train the LSTM model."
    )


# --------------------------------------------------
# FORECAST PAGE
# --------------------------------------------------

elif page == "Forecast":

    if "main_df" not in st.session_state:
        st.warning("Please upload your Main Dataset first on the Home page.")
        st.stop()

    main_df = st.session_state["main_df"].copy()
    date_column = st.session_state["date_column"]
    target_column = st.session_state["target_column"]

    st.markdown("## Dataset Configuration")

    dates = pd.to_datetime(main_df[date_column], errors="coerce")
    consumption = pd.to_numeric(
        main_df[target_column], errors="coerce"
    )

    data = pd.DataFrame({
        "Date": dates,
        "Consumption": consumption
    })

    other_df = st.session_state.get("other_df")

    if other_df is not None:
        if len(other_df) != len(data):
            st.error(
                "The number of rows in the Additional Variables file "
                "must match the Main Dataset."
            )
            st.stop()

        for column in other_df.columns:
            data[column] = pd.to_numeric(
                other_df[column], errors="coerce"
            )

    if data[["Date", "Consumption"]].isna().any().any():
        st.error(
            "The selected date or consumption column contains invalid "
            "or missing values. Please clean the file and upload it again."
        )
        st.stop()

    data = data.sort_values("Date").reset_index(drop=True)

    st.markdown("### Dataset Preview")
    st.dataframe(data, use_container_width=True)

    st.markdown("### Select Additional Variables")

    available_features = [
        c for c in data.columns
        if c not in ["Date", "Consumption"]
        and data[c].notna().any()
    ]

    selected_features = []

    if available_features:
        st.caption(
            "Select only the variables you want the LSTM model to use."
        )

        checkbox_columns = st.columns(3)

        for i, feature in enumerate(available_features):
            with checkbox_columns[i % 3]:
                if st.checkbox(feature, value=True, key=f"feature_{feature}"):
                    selected_features.append(feature)
    else:
        st.info("No additional variables uploaded. The model will use kWh only.")

    st.markdown("---")
    st.markdown("### LSTM Model Settings")

    c1, c2, c3 = st.columns(3)

    with c1:
        lookback = st.number_input(
            "Lookback Period",
            min_value=2,
            max_value=36,
            value=3,
            step=1,
            help="Number of previous time steps used to predict the next value."
        )

    with c2:
        horizon = st.number_input(
            "Forecast Horizon",
            min_value=1,
            max_value=36,
            value=12,
            step=1
        )

    with c3:
        test_size = st.number_input(
            "Test Observations",
            min_value=1,
            max_value=100,
            value=max(1, min(3, len(data) // 5)),
            step=1
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        epochs = st.number_input(
            "Epochs",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

    with c5:
        units = st.selectbox(
            "LSTM Units",
            [16, 32, 48, 64, 96, 128],
            index=2
        )

    with c6:
        frequency_label = st.selectbox(
            "Data Frequency",
            ["Monthly", "Quarterly", "Weekly", "Daily"]
        )

    frequency_map = {
        "Monthly": "MS",
        "Quarterly": "QS",
        "Weekly": "W",
        "Daily": "D"
    }

    frequency = frequency_map[frequency_label]

    dropout = st.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05
    )

    st.caption(
        "Future forecasts reuse the latest available values of optional "
        "variables unless future values are supplied. This is a persistence assumption."
    )

    if st.button("Train LSTM and Generate Forecast", use_container_width=True):

        if len(data) <= lookback + test_size:
            st.error(
                "Not enough observations for the selected lookback and test size. "
                "Reduce these settings or upload more historical data."
            )
            st.stop()

        with st.spinner("Training the LSTM model. Please wait..."):

            try:
                feature_columns = ["Consumption"] + selected_features

                model_data = data[feature_columns].copy()

                for col in feature_columns:
                    model_data[col] = model_data[col].fillna(
                        model_data[col].median()
                    )

                values = model_data.values.astype(float)

                split = len(values) - int(test_size)

                if split <= lookback + 1:
                    st.error("Not enough training observations.")
                    st.stop()

                x_scaler = MinMaxScaler()
                y_scaler = MinMaxScaler()

                x_scaler.fit(values[:split])
                y_scaler.fit(values[:split, [0]])

                x_scaled = x_scaler.transform(values)
                y_scaled = y_scaler.transform(values[:, [0]])

                X_train, y_train = make_sequences(
                    x_scaled[:split],
                    y_scaled[:split],
                    int(lookback)
                )

                if len(X_train) < 2:
                    st.error("Not enough training sequences. Reduce lookback.")
                    st.stop()

                model = build_lstm(
                    int(lookback),
                    len(feature_columns),
                    int(units),
                    float(dropout)
                )

                early_stop = EarlyStopping(
                    monitor="loss",
                    patience=12,
                    restore_best_weights=True
                )

                model.fit(
                    X_train,
                    y_train,
                    epochs=int(epochs),
                    batch_size=min(16, len(X_train)),
                    callbacks=[early_stop],
                    verbose=0
                )

                # One-step-ahead test using actual prior observations
                test_predictions_scaled = []

                for i in range(split, len(values)):
                    sequence = x_scaled[i - int(lookback):i]
                    prediction = model.predict(
                        sequence.reshape(1, int(lookback), len(feature_columns)),
                        verbose=0
                    )
                    test_predictions_scaled.append(prediction[0, 0])

                test_predictions = y_scaler.inverse_transform(
                    np.array(test_predictions_scaled).reshape(-1, 1)
                ).flatten()

                actual_test = values[split:, 0]

                mae, rmse, mape = calculate_metrics(
                    actual_test,
                    test_predictions
                )

                # Recursive future forecast
                history = x_scaled.copy()
                future_predictions = []

                for _ in range(int(horizon)):
                    sequence = history[-int(lookback):]

                    prediction_scaled = model.predict(
                        sequence.reshape(1, int(lookback), len(feature_columns)),
                        verbose=0
                    )[0, 0]

                    prediction_actual = y_scaler.inverse_transform(
                        [[prediction_scaled]]
                    )[0, 0]

                    future_predictions.append(prediction_actual)

                    next_row = history[-1].copy()
                    next_row[0] = prediction_scaled
                    history = np.vstack([history, next_row])

                future_dates = pd.date_range(
                    start=data["Date"].iloc[-1],
                    periods=int(horizon) + 1,
                    freq=frequency
                )[1:]

                test_dates = data["Date"].iloc[split:]

                st.session_state["model"] = model
                st.session_state["mae"] = mae
                st.session_state["rmse"] = rmse
                st.session_state["mape"] = mape
                st.session_state["test_dates"] = test_dates
                st.session_state["actual_test"] = actual_test
                st.session_state["test_predictions"] = test_predictions
                st.session_state["future_dates"] = future_dates
                st.session_state["future_predictions"] = future_predictions
                st.session_state["selected_features"] = selected_features
                st.session_state["data"] = data

                st.success("LSTM training and forecasting completed!")

            except Exception as e:
                st.error(f"Model training failed: {e}")

    # DISPLAY RESULTS
    if "future_predictions" in st.session_state:

        st.markdown("---")
        st.markdown("## Model Evaluation")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("MAE", f"{st.session_state['mae']:,.2f}")

        with m2:
            st.metric("RMSE", f"{st.session_state['rmse']:,.2f}")

        with m3:
            mape_value = st.session_state["mape"]
            st.metric(
                "MAPE",
                "N/A" if np.isnan(mape_value) else f"{mape_value:.2f}%"
            )

        st.caption(
            "MAE measures average absolute error. RMSE gives greater weight "
            "to large errors. MAPE expresses error as a percentage and may "
            "be unavailable when actual values are zero."
        )

        st.markdown("## Forecast Results")

        forecast_df = pd.DataFrame({
            "Date": st.session_state["future_dates"],
            "Forecasted Consumption (kWh)": st.session_state["future_predictions"]
        })

        st.dataframe(forecast_df, use_container_width=True)

        fig = make_chart(
            dates=st.session_state["test_dates"],
            actual=st.session_state["actual_test"],
            predicted=st.session_state["test_predictions"],
            future_dates=st.session_state["future_dates"],
            future_values=st.session_state["future_predictions"],
            title="LSTM Test Results and Future Forecast"
        )

        st.plotly_chart(fig, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name="lstm_forecast.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# ROLLING TEST PAGE
# --------------------------------------------------

elif page == "Rolling Test":

    st.markdown("## Rolling Test")
    st.info(
        "This section is reserved for rolling-origin evaluation. "
        "It will be implemented separately from the one-step test evaluation "
        "shown on the Forecast page."
    )


# --------------------------------------------------
# MODEL COMPARISON PAGE
# --------------------------------------------------

elif page == "Model Comparison":

    st.markdown("## Model Comparison")
    st.info(
        "This is the LSTM-only edition. XGBoost will have its own separate app "
        "and model workflow."
    )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.markdown("## About Load Forecast Studio")

    st.write("""
    Load Forecast Studio is a general-purpose electrical load forecasting
    application that allows users to upload their own historical data and
    generate forecasts using a Long Short-Term Memory (LSTM) neural network.

    The application supports optional external variables that may help
    describe changes in electricity consumption.

    **Model:** Long Short-Term Memory (LSTM)

    **Target variable:** Electricity Consumption (kWh)

    **Evaluation metrics:** MAE, RMSE, and MAPE
    """)

