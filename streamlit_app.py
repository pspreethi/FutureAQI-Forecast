import streamlit as st
import zipfile
from io import StringIO
from io import BytesIO
import requests
import pandas as pd
import datetime
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = st.secrets["openaq"]["api_key"]
HEADERS = {"X-API-Key": API_KEY}
BASE_LOCATION_URL = "https://api.openaq.org/v3/locations"
BASE_SENSOR_URL = "https://api.openaq.org/v3/sensors"



def generate_download_link(data, filename):
    csv = StringIO()
    data.to_csv(csv, index=False)
    csv.seek(0)
    return csv


def generate_zip_file():
    # Create an in-memory bytes buffer
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add raw data to ZIP if available
        if 'raw_data' in st.session_state:
            raw_csv = generate_download_link(st.session_state['raw_data'], "raw_data.csv")
            zip_file.writestr("raw_data.csv", raw_csv.getvalue())

        # Add cleaned data to ZIP if available
        if 'cleaned_data' in st.session_state:
            cleaned_csv = generate_download_link(st.session_state['cleaned_data'], "cleaned_data.csv")
            zip_file.writestr("cleaned_data.csv", cleaned_csv.getvalue())

        # Add station data to ZIP if available
        if 'station_dfs' in st.session_state:
            for station, df_station in st.session_state['station_dfs'].items():
                station_csv = generate_download_link(df_station, f"{station}_data.csv")
                zip_file.writestr(f"{station.replace(' ', '_')}_data.csv", station_csv.getvalue())

    buffer.seek(0)
    return buffer


# Fetch paginated data
def fetch_paginated_data(url, params=None):
    page = 1
    limit = 1000
    all_results = []
    while True:
        try:
            full_url = f"{url}?page={page}&limit={limit}"
            response = requests.get(full_url, headers=HEADERS, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if not results:
                    break
                all_results.extend(results)
                page += 1
            else:
                break
        except Exception as e:
            logger.error(f"Error fetching data from {url} - {e}")
            break
    return all_results

# Normalize sensor data
def normalize_sensor_data(sensor_ids):
    all_sensor_data = []
    params = {"datetime_from": "2020-01-01", "datetime_to": "2025-02-25", "limit": 1000}
    for s_id in sensor_ids:
        page = 1
        while True:
            params["page"] = page
            url = f"{BASE_SENSOR_URL}/{s_id}/measurements/daily"
            try:
                response = requests.get(url, headers=HEADERS, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data.get("results"):
                        break
                    for record in data["results"]:
                        record["sensor_id"] = s_id
                        all_sensor_data.append(record)
                    page += 1
                else:
                    break
            except Exception as e:
                logger.error(f"Error fetching sensor {s_id}: {e}")
    return pd.json_normalize(all_sensor_data)

# Data cleaning functions
def parse_coverage_datetimes(df):
    datetime_cols = [
        "coverage.datetimeFrom.utc", "coverage.datetimeFrom.local",
        "coverage.datetimeTo.utc", "coverage.datetimeTo.local"
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["from_utc_date"] = df["coverage.datetimeFrom.utc"].dt.date
    df["from_local_date"] = df["coverage.datetimeFrom.local"].dt.date
    df["to_utc_date"] = df["coverage.datetimeTo.utc"].dt.date
    df["to_local_date"] = df["coverage.datetimeTo.local"].dt.date
    return df

# Interpolation
def interpolate_openaq_data(df):
    aqi_columns = [
        'value', 'summary.min', 'summary.q02', 'summary.q25', 'summary.median',
        'summary.q75', 'summary.q98', 'summary.max', 'summary.avg', 'summary.sd'
    ]
    df["from_local_date"] = pd.to_datetime(df["from_local_date"])
    df.set_index("from_local_date", inplace=True)
    for col in aqi_columns:
        df[col] = df[col].where(df[col] >= 0, np.nan)
    df[aqi_columns] = df[aqi_columns].interpolate(method='time')
    return df

# Clean OpenAQ data
def clean_openaq_data(df_raw, df_locations):
    df = pd.merge(df_raw, df_locations, left_on='sensor_id', right_on='s_id', how='left')
    df = parse_coverage_datetimes(df)
    df["parameter"] = df["parameter.name"].astype(str) + " " + df["parameter.units"].astype(str)
    columns_to_drop = [
        'coordinates', 'flagInfo.hasFlags', 'parameter.id', 'parameter.name', 'parameter.units', 'parameter.displayName',
        'period.label', 'period.interval', 'period.datetimeFrom.utc', 'period.datetimeFrom.local',
        'period.datetimeTo.utc', 'period.datetimeTo.local', 'coverage.expectedCount', 'coverage.expectedInterval',
        'coverage.observedCount', 'coverage.observedInterval', 'coverage.percentComplete', 'coverage.percentCoverage',
        'coverage.datetimeFrom.utc', 'coverage.datetimeFrom.local', 'coverage.datetimeTo.utc', 'coverage.datetimeTo.local',
        'datetimeFirst.utc', 'datetimeFirst.local', 'datetimeLast.utc', 'datetimeLast.local', 'distance', 's_id',
        's_parameter.id', 's_parameter.name', 's_parameter.units', 's_parameter.displayName'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    final_cols = ['value', 'sensor_id', 'summary.min', 'summary.q02', 'summary.q25', 'summary.median',
                 'summary.q75', 'summary.q98', 'summary.max', 'summary.avg', 'summary.sd', 'from_utc_date', 'from_local_date',
                 'to_utc_date', 'to_local_date', 'parameter', 'provider.name', 'id', 'name', 'locality']
    df_analysis = df[final_cols]
    df_cleaned = interpolate_openaq_data(df_analysis)
    return df_cleaned

# Split cleaned data by station
def split_by_station(df, date_col="to_local_date", station_col="name"):
    try:
        unique_stations = df[station_col].dropna().unique()
        station_dfs = {}

        for station in unique_stations:
            df_station = df[df[station_col] == station].copy()
            # Ensure the date column is present and in datetime format
            if date_col not in df_station.columns:
                st.error(f"Date column '{date_col}' not found in data for station: {station}")
                continue

            df_station[date_col] = pd.to_datetime(df_station[date_col], errors="coerce")
            df_station.set_index(date_col, inplace=True)

            # Group by date to remove duplicates and set daily frequency
            df_station = df_station.groupby(df_station.index).mean(numeric_only=True).asfreq("D")

            # Interpolate missing values
            aqi_cols = [
                'value', 'summary.min', 'summary.q02', 'summary.q25', 'summary.median',
                'summary.q75', 'summary.q98', 'summary.max', 'summary.avg', 'summary.sd'
            ]
            for col in aqi_cols:
                if col in df_station.columns:
                    df_station[col] = df_station[col].interpolate(method="time")

            # Save each station's DataFrame to a dictionary for use in Streamlit
            station_dfs[station] = df_station
        
        st.success("Data fetched, cleaned and split by location successfully!")
        return station_dfs

    except Exception as e:
        st.error(f"Error splitting data by station: {e}")


def generate_station_eda(df, station_name):
    try:
        #st.subheader(f"Exploratory Data Analysis (EDA) - {station_name}")

        # Add time components
        df["weekday"] = df.index.day_name()
        df["month"] = df.index.month
        df["year"] = df.index.year
        month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                     7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        df["month_name"] = df["month"].map(month_map)

        # Weekday Boxplot
        st.write("### AQI by Day of Week")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="weekday", y="summary.avg", data=df,
                    order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Monthly Boxplot
        st.write("### Monthly AQI Seasonality")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="month_name", y="summary.avg", data=df,
                    order=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ax=ax)
        st.pyplot(fig)

        # Yearly Boxplot
        st.write("### Year-over-Year AQI")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="year", y="summary.avg", data=df, ax=ax)
        st.pyplot(fig)

        # Weekly Avg Line Plot
        st.write("### Weekly Average AQI")
        weekly = df["summary.avg"].resample("W").mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(weekly.index, weekly, marker='o')
        plt.xlabel("Date")
        plt.ylabel("Average AQI")
        st.pyplot(fig)

        # Monthly Avg Line Plot
        st.write("### Monthly Average AQI")
        monthly = df["summary.avg"].resample("M").mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly.index, monthly, marker='o')
        plt.xlabel("Date")
        plt.ylabel("Average AQI")
        st.pyplot(fig)

        # 90-Day Rolling Trend
        st.write("### Long-Term AQI Trend (90-Day Moving Average)")
        df["rolling_90"] = df["summary.avg"].rolling(window=90).mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["summary.avg"], alpha=0.4, label="Daily AQI")
        ax.plot(df.index, df["rolling_90"], color="red", label="90-Day MA")
        plt.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating EDA for {station_name}: {e}")


def forecast_station_prophet(df, station_name):
    try:
        st.subheader(f"90-Day AQI Forecast - {station_name}")

        df = df[["summary.avg"]].dropna().reset_index()
        df.columns = ["ds", "y"]

        # Initialize and fit the model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.1
        )
        model.add_country_holidays(country_name='IN')
        model.fit(df)

        # Create future dataframe and predict
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        # Merge forecast with actual values
        forecast = pd.merge(forecast, df[['ds', 'y']], on='ds', how='left')

        # Store forecast in session state
        forecast_key = f"forecast_{station_name.replace(' ', '_')}"
        st.session_state[forecast_key] = forecast
        st.success(f"Forecast data stored in session for {station_name}")

        # Plot forecast
        fig1 = model.plot(forecast)
        plt.title(f"90-Day AQI Forecast - {station_name}")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        st.pyplot(fig1)

        # Plot forecast components
        fig2 = model.plot_components(forecast)
        plt.title(f"Trend and Seasonality Components - {station_name}")
        st.pyplot(fig2)

        # Display the forecast DataFrame
        st.write("Forecast Data:")
        st.dataframe(forecast)

        # Convert forecast to CSV for download
        csv = StringIO()
        forecast.to_csv(csv, index=False)
        csv.seek(0)

        # Download button for forecast data
        st.download_button(
            label="Download Forecast as CSV",
            data=csv.getvalue(),
            file_name=f"{station_name.replace(' ', '_')}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error generating forecast for {station_name}: {e}")


# Improved download buttons layout
def display_download_buttons():
    with st.expander("Download Data", expanded=True):
        if 'raw_data' in st.session_state:
            raw_csv = generate_download_link(st.session_state['raw_data'], "raw_data.csv")
            st.download_button(
                label="Download Raw Data",
                data=raw_csv.getvalue(),
                file_name="raw_data.csv",
                mime="text/csv"
            )
        if 'cleaned_data' in st.session_state:
            cleaned_csv = generate_download_link(st.session_state['cleaned_data'], "cleaned_data.csv")
            st.download_button(
                label="Download Cleaned Data",
                data=cleaned_csv.getvalue(),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        if 'station_dfs' in st.session_state:
            station_list = list(st.session_state['station_dfs'].keys())
            for station in station_list:
                df_station = st.session_state['station_dfs'][station]
                station_csv = generate_download_link(df_station, f"{station}_data.csv")
                st.download_button(
                    label=f"Download {station} Data",
                    data=station_csv.getvalue(),
                    file_name=f"{station.replace(' ', '_')}_data.csv",
                    mime="text/csv"
                )


def evaluate_forecast_station(df, station_name):
    try:
        # Ensure forecast data exists
        if df.empty:
            st.warning(f"No forecast data available for {station_name}.")
            return

        # Extract actual and predicted values
        actual = df['y'].dropna()
        predicted = df['yhat'].dropna()

        if len(actual) == 0 or len(predicted) == 0:
            st.warning(f"No valid actual or predicted values for {station_name}.")
            return

        # Calculate MAE, RMSE, MAPE
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100

        # Display Metrics
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

        # Cross-Validation with Prophet
        df = df[["ds", "y"]].dropna()

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.add_country_holidays(country_name='IN')
        model.fit(df)

        # Cross-validation metrics
        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='90 days')
        df_performance = performance_metrics(df_cv)

        # Display Cross-Validation Metrics
        st.write("### Cross-Validation Performance Metrics")
        st.dataframe(df_performance)

        # Plot RMSE over Forecast Horizon
        st.write("### RMSE over Forecast Horizon")
        fig = plot_cross_validation_metric(df_cv, metric='rmse')
        st.pyplot(fig)

        # Save metrics to CSV for download
        csv = StringIO()
        df_performance.to_csv(csv, index=False)
        csv.seek(0)
        st.download_button(
            label="Download Evaluation Metrics as CSV",
            data=csv.getvalue(),
            file_name=f"{station_name.replace(' ', '_')}_evaluation_metrics.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error evaluating forecast for {station_name}: {e}")


# Streamlit App
def clean_and_display_data():
    all_locations = fetch_paginated_data(BASE_LOCATION_URL)
    in_locations = [loc for loc in all_locations if loc.get("country", {}).get("code") == "IN"]
    df_all = pd.json_normalize(
        in_locations,
        record_path=["sensors"],
        meta=["id", "name", "locality", "timezone", "isMobile", "isMonitor", "licenses", "instruments", ["provider", "name"]],
        record_prefix="s_",
        errors="ignore"
    )
    df_pm25 = df_all[df_all["s_name"] == "pm25 µg/m³"]
    df_filtered = df_pm25[(df_pm25["provider.name"] == "AirNow") & (df_pm25["locality"].notna()) & (df_pm25["licenses"].notna())]
    sensor_ids = list(df_filtered["s_id"])
    df_sensor_data = normalize_sensor_data(sensor_ids)
    df_cleaned = clean_openaq_data(df_sensor_data, df_all)
    station_dfs = split_by_station(df_cleaned)
    for station, data in station_dfs.items():
        st.write(f"Station: {station}")
        st.dataframe(data)

    # Store raw and cleaned data in session state for download
    st.session_state['raw_data'] = df_all
    st.session_state['cleaned_data'] = df_cleaned
    st.session_state['station_dfs'] = station_dfs

    if station_dfs:
        st.session_state['station_dfs'] = station_dfs
        st.success("Data has been cleaned and split by station!")
    else:
        st.error("No station data available after cleaning.")



# === Streamlit Configuration ===
st.set_page_config(page_title="Air Quality Data Forecaster", layout="wide")
st.title("Air Quality Data Forecaster")

# === Navigation Sidebar ===
st.sidebar.title("Navigation")

# Check if page is already set in session state, else default to "About"
if "page" not in st.session_state:
    st.session_state.page = "About"
    
if st.sidebar.button("About"):
    st.session_state.page = "About"
    st.header("About This Project")

    st.markdown("""
    ### Project Overview
    This project aims to forecast Air Quality Index (AQI) for major Indian cities using historical data. The predictions are based on consistent AQI data collected from various monitoring stations.

    ### Data Source:
    We use data from [OpenAQ API](https://docs.openaq.org/) to fetch real-time AQI readings from licensed sensors.
    
    ### More Information:
    - Visit the official [OpenAQ website](https://openaq.org/) for more details on global air quality data.
    - Check the [OpenAQ API Documentation](https://docs.openaq.org/) for technical specifications and available endpoints.
    
    ### Assumptions
    - All sensor locations are stationary in India.
    - Only 6 sensor locations have a valid license.
    - All selected sensors are actively monitoring (isMonitor is true).
    - No sensors are duplicated, ensuring that each measurement belongs to a unique location.
    - For simplicity, only considering data from the **AirNow** provider.
    - Only sensors measuring **PM2.5** (Particulate Matter with diameter < 2.5 µm) are considered.

    ### Why AirNow Provider?
    Why AirNow Provider?
    - AirNow data is more consistent and reliable for AQI forecasting.
    - Although it is limited to major cities, it offers accurate and continuous measurements.
    - This consistency makes it suitable for forecasting and data analysis.

    ### Data Pipeline
    1. **Data Fetching:** Fetches data from OpenAQ using their API.
    2. **Data Cleaning:** Handles missing values, negative values, and aggregates data by date.
    3. **Data Interpolation:** Missing values are interpolated using time-based methods.
    4. **Forecasting:** Uses the **Prophet** model to generate a 90-day forecast for each location.
    5. **Evaluation:** The forecast accuracy is assessed using MAE, RMSE, and MAPE metrics.

    ### Model Used: Prophet
    - Prophet was chosen for its ability to handle seasonality and holidays.
    - It is well-suited for time-series data with daily and yearly patterns.
    - Handles missing data and can incorporate holiday effects.

    ### Key Fields as per OpenAQ
    - **id:** Unique identifier for each location.
    - **name:** Name of the station.
    - **country:** Country where the station operates, including its ISO code.
    - **provider:** The organization sharing the data.
    - **owner:** Organization or individual responsible for the station.
    - **coordinates:** Latitude and longitude of the station.
    - **sensors:** List of sensors measuring pollutants.
    - **licenses:** License details for data sharing.
    - **timezone:** Local time zone of the station.

    This project serves as a comprehensive AQI forecasting tool, leveraging consistent data and robust modeling.
    """)


# Button to fetch and clean data
if st.sidebar.button("Fetch and Clean Data"):
    st.success("Data fetching initiated!")
    try:
        clean_and_display_data()
    except Exception as e:
        st.error(f"Failed fetching and cleaning data: {e}")


# Download ZIP Button
if 'station_dfs' in st.session_state:
    st.sidebar.markdown("## Download All Data")
    zip_buffer = generate_zip_file()
    st.sidebar.download_button(
        label="Download All Data as ZIP",
        data=zip_buffer,
        file_name="air_quality_data.zip",
        mime="application/zip"
    )

# Check if cleaned data is available
if 'station_dfs' in st.session_state and st.session_state['station_dfs']:
    # Sidebar: Station selection and analysis type
    station_list = list(st.session_state['station_dfs'].keys())
    selected_station = st.sidebar.selectbox("Select Station", station_list, key="station_selector")

    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["View Data", "EDA", "Forecasting", "Evaluation"],
        key="analysis_selector"
    )

    # Display the selected station data and analysis
    if selected_station:
        st.markdown(f"### Station: **{selected_station}**")
        df_station = st.session_state['station_dfs'].get(selected_station)

        if df_station is not None:
            # Convert date column to datetime and set as index if not already
            if "to_local_date" in df_station.columns:
                try:
                    df_station["to_local_date"] = pd.to_datetime(df_station["to_local_date"], errors="coerce")
                    df_station.set_index("to_local_date", inplace=True)
                    st.success(f"Date index set for station: {selected_station}")
                except Exception as e:
                    st.error(f"Error setting date index for station {selected_station}: {e}")

            # Conditional display based on selected analysis type
            st.markdown(f"## {analysis_type} for {selected_station}")
            if analysis_type == "EDA":
                generate_station_eda(df_station, selected_station)
            elif analysis_type == "View Data":
                st.write(f"Station: {selected_station}")
                st.dataframe(df_station)
            elif analysis_type == "Forecasting":
                forecast_station_prophet(df_station, selected_station)
            elif analysis_type == "Evaluation":
                forecast_key = f"forecast_{selected_station.replace(' ', '_')}"
                if forecast_key in st.session_state:
                    forecast_df = st.session_state[forecast_key]
                    st.write(f"Evaluating forecast for station: {selected_station}")
                    try:
                        evaluate_forecast_station(forecast_df, selected_station)
                        st.success("Forecast evaluation completed!")
                    except Exception as e:
                        st.error(f"Error during evaluation: {e}")
                else:
                    st.warning(f"Forecast not available for {selected_station}. Please generate the forecast first.")
else:
    st.warning("No station data available. Please click 'Fetch and Clean Data' first.")
