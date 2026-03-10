import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle
import os

# --- Helper Functions ---
def load_artifacts():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        cols = pickle.load(open('columns.pkl', 'rb'))
        return model, encoder, cols
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def fetch_flight_data(flight_number, api_key):
    url = f"https://aerodatabox.p.rapidapi.com/flights/number/{flight_number}"
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()[0]
    except Exception:
        return None

def fetch_weather_data_nws(lat, lon):
    try:
        point_res = requests.get(f"https://api.weather.gov/points/{lat},{lon}").json()
        forecast_res = requests.get(point_res['properties']['forecastHourly']).json()
        current = forecast_res['properties']['periods'][0]
        return {
            "wind_speed": float(current['windSpeed'].split(' ')[0]),
            "probability_of_precipitation": current.get('probabilityOfPrecipitation', {}).get('value', 0) or 0,
            "description": current['shortForecast']
        }
    except Exception:
        return {"wind_speed": 10.0, "probability_of_precipitation": 0.0, "description": "Clear"}

def map_api_to_features(flight_info, weather_info, encoder, train_cols):
    # Robust timestamp extraction
    departure = flight_info.get('departure', {})
    sched_time = departure.get('scheduledTimeUtc') or departure.get('scheduledTimeLocal') or datetime.now().isoformat()
    
    dt_obj = datetime.fromisoformat(sched_time.replace('Z', '+00:00'))
    month, day_of_week = dt_obj.month, dt_obj.isoweekday()
    dep_time = int(dt_obj.strftime('%H%M'))
    season = 'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Fall'
    
    raw_input = pd.DataFrame([{
        'Month': month, 'DayOfWeek': day_of_week, 'DepTime': dep_time,
        'Carrier': flight_info.get('airline', {}).get('iata', 'AA'), 
        'OriginAirport': departure.get('airport', {}).get('name', 'JFK'),
        'Continent': 'North America'
    }])
    
    encoded_input = encoder.transform(raw_input)
    features = {
        'Month': month, 'DayOfWeek': day_of_week, 'DepTime': dep_time,
        'Carrier': encoded_input['Carrier'].iloc[0], 'OriginAirport': encoded_input['OriginAirport'].iloc[0],
        'Precipitation': weather_info['probability_of_precipitation'] / 10.0, 'WindSpeed': weather_info['wind_speed'],
        'Climate_Arctic Winds': 1 if season == 'Winter' else 0, 'Climate_Arid': 0, 'Climate_Monsoon': 0,
        'Climate_Temperate': 1 if season in ['Spring', 'Fall'] else 0, 'Climate_Tropical Storm': 1 if season == 'Summer' else 0,
        'Continent_Africa': 0, 'Continent_Asia': 0, 'Continent_Europe': 0, 'Continent_North America': 1,
        'Continent_Oceania': 0, 'Continent_South America': 0
    }
    return pd.DataFrame([features])[train_cols]

# --- Streamlit UI ---
st.set_page_config(page_title="Global Flight Predictor", layout="wide")
st.title("✈️ Global Flight Delay Predictor & Explainer")

model, encoder, train_cols = load_artifacts()

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("AeroDataBox API Key", type="password")

flight_num = st.text_input("Flight Number", value="AA500")

if st.button("Run Analysis"):
    if not api_key:
        st.warning("Please enter an API key in the sidebar.")
    else:
        with st.spinner("Analyzing... "):
            f_data = fetch_flight_data(flight_num, api_key)
            if f_data:
                dep_info = f_data.get('departure', {})
                loc = dep_info.get('airport', {}).get('location', {})
                lat, lon = loc.get('lat', 40.64), loc.get('lon', -73.77)
                
                w_data = fetch_weather_data_nws(lat, lon)
                input_df = map_api_to_features(f_data, w_data, encoder, train_cols)
                prob = model.predict_proba(input_df)[0][1]
                status = "DELAYED" if prob > 0.5 else "ON TIME"
                
                st.metric("Delay Probability", f"{prob:.2%}")
                st.write(f"**Status:** {status} | **Weather:** {w_data['description']}")
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                ev = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(shap.Explanation(values=sv, base_values=ev, data=input_df.iloc[0], feature_names=train_cols), show=False)
                st.pyplot(plt.gcf())
            else:
                st.error("Flight not found. Try a common number like AA500 or DL123.")
