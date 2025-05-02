# Air Quality Data Forecaster

A comprehensive pipeline and forecasting tool for analyzing air quality (PM2.5) data from major cities in India using the OpenAQ API. The project fetches licensed sensor data, performs exploratory data analysis (EDA), and applies time series forecasting using Facebook Prophet, with an interactive dashboard built on Streamlit.

[Launch the App here](https://futureaqi-forecast.streamlit.app/)  
[Behind the build](https://medium.com/@pokasaipreethi/data-dust-and-a-podcast-detour-into-air-quality-forecasting-3bc29a90ed3b)

---

## Features

- Integration with the OpenAQ API to retrieve licensed sensor data.
- Filtering of valid PM2.5 sensors from reliable sources (e.g., AirNow).
- Time-indexed data cleaning and interpolation.
- Exploratory Data Analysis (EDA) with seasonal and trend insights.
- 90-day forecasting using the Prophet time series model.
- Accuracy evaluation with MAE, RMSE, and MAPE.
- Interactive Streamlit interface for non-technical users.

---
## Assumptions

- Only stationary sensors in India are used.
- Only sensors with active licenses and sufficient historical data are considered.
- PM2.5 parameter is prioritized due to relevance and availability.
- Sensor readings are location-specific and non-duplicated.
- Provider "AirNow" was chosen for consistency in recent data.

---

## Technologies Used

- Python 3.10+
- Pandas, NumPy, Matplotlib, Seaborn
- Prophet (time series forecasting)
- Streamlit (web app interface)
- Requests, dotenv, joblib

---
## Forecasting Details
- Model: Prophet (additive seasonality)
- Frequency: Daily
- Seasonality: Weekly + Yearly
- Horizon: 90 Days
- Metrics: MAE, RMSE, MAPE
- Cross-validation: 2 years initial, 90 days horizon, sliding every 180 days

---
## Data Source

- Provider: OpenAQ API
- Country: India (IN)
- Parameter: PM2.5 only
- Sensor Provider: AirNow

---
## Getting Started

Follow these steps to run the application locally:

```bash
# 1. Clone the repository
git clone https://github.com/pspreethi/FutureAQI-Forecast.git
cd air-quality-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAQ API key in the secrets file
mkdir -p .streamlit
echo "[openaq]\napi_key = \"your_openaq_api_key\"" > .streamlit/secrets.toml

# 4. Run the Streamlit application
streamlit run streamlit_app.py
```
---
## License

MIT License


