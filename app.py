"Scroll inside the table to inspect later years."
        )
        st.dataframe(main_df, use_container_width=True, height=420)

        columns = list(main_df.columns)
        month_candidates = [c for c in columns if "month" in c.lower() or "date" in c.lower()]
        year_candidates = [c for c in columns if c.lower() == "year" or "year" in c.lower()]

        # Year coverage is calculated after the user selects the actual date/year columns.
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

        try:
            selected_periods = parse_months(main_df, month_col, year_col)
            year_counts = pd.Series(selected_periods.year).value_counts().sort_index()
            coverage = pd.DataFrame({
                "Year": year_counts.index.astype(int),
                "Monthly records": year_counts.values,
            })
            st.markdown("#### Main dataset year coverage")
            st.dataframe(coverage, hide_index=True, use_container_width=True)
            st.caption(
                f"Detected coverage: {selected_periods.min()} to {selected_periods.max()} "
                f"({len(selected_periods):,} records across {year_counts.size} year(s))."
            )
        except Exception as date_error:
            st.warning(f"Year coverage cannot be confirmed yet: {date_error}")

        extra_df = st.session_state.get("extra_df")
        extra_month_col = None
        extra_year_col = None
        if extra_df is not None:
            st.markdown("### Preview additional variables")
            st.caption(
                f"Showing all {len(extra_df):,} uploaded rows and {len(extra_df.columns)} columns."
            )
            st.dataframe(extra_df, use_container_width=True, height=420)
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
                try:
                    extra_periods = parse_months(extra_df, extra_month_col, extra_year_col)
                    extra_year_counts = pd.Series(extra_periods.year).value_counts().sort_index()
                    extra_coverage = pd.DataFrame({
                        "Year": extra_year_counts.index.astype(int),
                        "Monthly records": extra_year_counts.values,
                    })
                    st.markdown("#### Additional variables year coverage")
                    st.dataframe(extra_coverage, hide_index=True, use_container_width=True)
                    st.caption(
                        f"Detected additional-data coverage: {extra_periods.min()} to {extra_periods.max()} "
                        f"({len(extra_periods):,} records across {extra_year_counts.size} year(s))."
                    )
                except Exception as date_error:
                    st.warning(f"Additional-data year coverage cannot be confirmed yet: {date_error}")
            else:
                st.warning(
                    "No Month/date column found in Additional Variables. "
                    "Rows will be matched by row order, so confirm that both files are aligned exactly."
                )

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
        if c not in ["Month", "Consumption_kWh", "_Source_File", "_Source_Sheet"]
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
