import streamlit as st
import sys
import subprocess

st.write("Python version:", sys.version)
st.write("Installed packages:")
result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
st.code(result.stdout)

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt

st.title('Time Series Forecasting with Prophet')

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Ensure 'ds' is a datetime type and 'y' is float
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)

    # Add day of week feature
    df['day_of_week'] = df['ds'].dt.dayofweek

    # Define holidays and special events
    holidays = pd.DataFrame({
        'holiday': ['prime_day', 'prime_day', 'prime_day', 'prime_day', 'prime_day', 'prime_day', 'prime_day', 'prime_day',
                    'black_friday', 'black_friday', 'black_friday',
                    'cyber_monday', 'cyber_monday', 'cyber_monday',
                    'new_year', 'new_year', 'new_year', 'new_year',
                    'valentine', 'valentine', 'valentine',
                    'christmas', 'christmas', 'christmas'],
        'ds': pd.to_datetime(['2022-07-12', '2022-07-13', '2023-07-11', '2023-07-12', '2024-07-09', '2024-07-10', '2023-07-16', '2023-07-17',
                              '2022-11-25', '2023-11-24', '2024-11-29',
                              '2022-11-28', '2023-11-27', '2024-12-02',
                              '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01',
                              '2022-02-14', '2023-02-14', '2024-02-14',
                              '2022-12-25', '2023-12-25', '2024-12-25']),
        'lower_window': 0,
        'upper_window': 1,
    })

    # Streamlit inputs for model parameters
    st.sidebar.header('Model Parameters')
    changepoint_prior_scale = st.sidebar.slider(
        'Changepoint Prior Scale', 0.001, 0.5, 0.05, 0.001)
    seasonality_prior_scale = st.sidebar.slider(
        'Seasonality Prior Scale', 0.01, 10.0, 10.0, 0.01)
    holidays_prior_scale = st.sidebar.slider(
        'Holidays Prior Scale', 0.01, 10.0, 10.0, 0.01)

    # Initialize the Prophet model with additional settings
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
    )

    # Add monthly seasonality
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # Add the day of week as a regressor
    m.add_regressor('day_of_week')

    # Fit the model
    m.fit(df)

    # Create future dataframe for forecasting
    future_periods = st.sidebar.slider('Forecast Periods (days)', 30, 365, 365)
    future = m.make_future_dataframe(periods=future_periods, freq='d')
    future['day_of_week'] = future['ds'].dt.dayofweek

    # Make predictions
    forecast = m.predict(future)

    # Combine the original data with forecast
    combined_df = pd.merge(
        df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='outer')

    # Plot forecast
    st.subheader('Forecast Plot')
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    st.subheader('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    # Perform cross-validation
    data_size = len(df)
    if data_size > 730:  # Only perform cross-validation if we have more than 2 years of data
        df_cv = cross_validation(m, initial='365 days',
                                 period='90 days', horizon='180 days')
        df_p = performance_metrics(df_cv)
        st.subheader('Cross-validation performance metrics:')
        st.write(df_p)
    else:
        st.write("Not enough data for meaningful cross-validation")

    # Calculate and print MAPE for the entire forecast
    actual = df['y']
    predicted = forecast['yhat'][:len(actual)]
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Display combined DataFrame
    st.subheader('Combined Data and Forecast')
    st.write(combined_df)

    # Display data size and date range
    st.write(f"Total number of data points: {data_size}")
    st.write(f"Date range: from {df['ds'].min()} to {df['ds'].max()}")

    # Download link for the combined DataFrame
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download full data and forecast as CSV",
        data=csv,
        file_name="full_data_and_forecast.csv",
        mime="text/csv",
    )

else:
    st.write("Please upload a CSV file to begin.")
