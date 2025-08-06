import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import joblib

# --- Paths ---
DATA_PATH = "../data/Smart_Hospital_Sensor_Preprocessed.csv"
ENERGY_MODEL_PATH = "../models/rnn_energy_forecasting_model.keras"
ANOMALY_MODEL_PATH = "../models/lstm_anomaly_model.keras"
MAINTENANCE_MODEL_PATH = "../models/maintenance_classifier.pkl"
MAINTENANCE_SCALER_PATH = "../models/maintenance_scaler.pkl"
SEQ_LEN = 24

# --- Load Data ---
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor('s').dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

# --- Forecasting Function ---
def forecast_energy(df, building_id, room_id):
    df_filtered = df[(df["building_id"] == building_id) & (df["room_id"] == room_id)].copy()
    df_filtered = df_filtered.dropna()
    features = ["temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count", "light_level", "hour", "weekday", "month"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[features])

    if len(X_scaled) < SEQ_LEN:
        st.warning("Not enough data to forecast.")
        return pd.DataFrame()

    X_input = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(features))
    model = load_model(ENERGY_MODEL_PATH)

    forecast = []
    for _ in range(24):
        pred = model.predict(X_input, verbose=0)[0][0]
        forecast.append(pred)
        new_input = np.append(X_input[0][1:], [X_input[0][-1]], axis=0)
        X_input = new_input.reshape(1, SEQ_LEN, len(features))

    last_time = pd.to_datetime(df_filtered["timestamp"]).max()
    forecast_timestamps = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=24, freq="h").floor('s').strftime('%Y-%m-%d %H:%M:%S')
    return pd.DataFrame({"timestamp": forecast_timestamps, "predicted_energy_kwh": forecast})

# --- Anomaly Detection Function ---
def detect_anomalies(df, building_id, room_id):
    df_filtered = df[(df["building_id"] == building_id) & (df["room_id"] == room_id)].copy()
    features = ["temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count"]
    df_filtered = df_filtered.dropna(subset=features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[features])

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:i + seq_length])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)

    X_seq, y_true = create_sequences(X_scaled, SEQ_LEN)
    model = load_model(ANOMALY_MODEL_PATH)
    y_pred = model.predict(X_seq, verbose=0)

    mse = np.mean(np.power(y_true - y_pred, 2), axis=1)
    mse_full = np.concatenate([np.zeros(SEQ_LEN), mse])
    df_filtered["reconstruction_error_lstm"] = mse_full
    threshold = np.percentile(mse, 99)
    df_filtered["is_anomaly_lstm"] = (df_filtered["reconstruction_error_lstm"] > threshold).astype(int)

    df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"]).dt.floor('s').dt.strftime('%Y-%m-%d %H:%M:%S')
    return df_filtered[["timestamp", "reconstruction_error_lstm", "is_anomaly_lstm"]]

# --- Predictive Maintenance Function ---
def run_predictive_maintenance(df, building_id, room_id):
    df_filtered = df[(df["building_id"] == building_id) & (df["room_id"] == room_id)].copy()
    features = ["temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count", "light_level"]
    df_filtered = df_filtered.dropna(subset=features)
    scaler = joblib.load(MAINTENANCE_SCALER_PATH)
    clf = joblib.load(MAINTENANCE_MODEL_PATH)
    X_scaled = scaler.transform(df_filtered[features])
    predictions = clf.predict(X_scaled)
    df_filtered["predicted_needs_maintenance"] = predictions
    return df_filtered[["timestamp", "predicted_needs_maintenance"]]

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Hospital Dashboard", layout="wide")
st.title("🏥 NHS Smart Hospital Dashboard")
st.markdown("Visualising **Energy Forecasting**, **Anomaly Detection**, **EDA**, and **Maintenance Intelligence**.")

df = load_data()
buildings = df["building_id"].unique()
building_id = st.sidebar.selectbox("🏢 Select Building", buildings)
room_id = st.sidebar.selectbox("🚪 Select Room", df[df["building_id"] == building_id]["room_id"].unique())

# --- EDA Section ---
st.header("📊 Exploratory Data Analysis (EDA)")
eda_df = df[(df["building_id"] == building_id) & (df["room_id"] == room_id)].copy()
sub_summary = eda_df.describe().to_html()
st.subheader("🔍 Statistical Summary")
st.markdown(sub_summary, unsafe_allow_html=True)
sensor_cols = ["temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count"]
for col in sensor_cols:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eda_df["timestamp"], y=eda_df[col], mode="lines", name=col))
    fig.update_layout(title=f"{col} over Time", xaxis_title="Timestamp", yaxis_title=col)
    st.plotly_chart(fig, use_container_width=True)
corr = eda_df[sensor_cols].corr()
fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="Correlation")))
fig_corr.update_layout(title="Correlation Heatmap")
st.plotly_chart(fig_corr, use_container_width=True)

# --- Forecasting ---
st.header("⚡ Energy Forecasting")
forecast_df = forecast_energy(df, building_id, room_id)
if not forecast_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["predicted_energy_kwh"], mode="lines+markers", name="Forecast"))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), file_name="energy_forecast.csv")

# --- Anomaly Detection ---
st.header("🚨 LSTM-Based Anomaly Detection")
anomaly_df = detect_anomalies(df, building_id, room_id)
fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_df["timestamp"], y=anomaly_df["reconstruction_error_lstm"], mode="lines", name="Reconstruction Error"))
fig.add_trace(go.Scatter(x=anomaly_df[anomaly_df["is_anomaly_lstm"] == 1]["timestamp"], y=anomaly_df[anomaly_df["is_anomaly_lstm"] == 1]["reconstruction_error_lstm"], mode="markers", name="Anomalies", marker=dict(color="red", size=8)))
st.plotly_chart(fig, use_container_width=True)
st.download_button("📥 Download Anomaly CSV", anomaly_df.to_csv(index=False), file_name="anomaly_results.csv")

# --- Predictive Maintenance ---
st.header("🛠️ Predictive Maintenance Alerts")
maintenance_df = run_predictive_maintenance(df, building_id, room_id)
fig = go.Figure()
fig.add_trace(go.Scatter(x=maintenance_df["timestamp"], y=maintenance_df["predicted_needs_maintenance"], mode="markers", marker=dict(color="orange"), name="Needs Maintenance"))
fig.update_layout(title="Predicted Maintenance Alerts", xaxis_title="Timestamp", yaxis_title="Alert (1=Yes)")
st.plotly_chart(fig, use_container_width=True)
st.download_button("📥 Download Maintenance Predictions", maintenance_df.to_csv(index=False), file_name="maintenance_alerts.csv")

# --- Occupancy-Driven Control Simulator ---
st.header("🏠 Occupancy-Driven Control Simulator")
occu_df = df[(df["building_id"] == building_id) & (df["room_id"] == room_id)].copy()
occu_df["timestamp"] = pd.to_datetime(occu_df["timestamp"])
occu_df = occu_df.sort_values("timestamp")

occu_df["HVAC_status"] = occu_df["occupancy_count"].apply(lambda x: "ON" if x > 0 else "OFF")
st.subheader("💨 HVAC Control Based on Occupancy")
fig = go.Figure()
fig.add_trace(go.Scatter(x=occu_df["timestamp"], y=occu_df["occupancy_count"], mode="lines+markers", name="Occupancy Count"))
fig.update_layout(title="Room Occupancy", xaxis_title="Timestamp", yaxis_title="Occupancy Count")
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=occu_df["timestamp"], y=occu_df["HVAC_status"].apply(lambda x: 1 if x == "ON" else 0), mode="lines+markers", name="HVAC Status"))
fig2.update_layout(title="Simulated HVAC Status", xaxis_title="Timestamp", yaxis_title="HVAC (1=ON, 0=OFF)")
st.plotly_chart(fig2, use_container_width=True)

# --- Simulated Energy Savings ---
st.subheader("💡 Simulated Energy Savings")
energy_saved = occu_df.apply(lambda row: row["energy_kwh"] if row["HVAC_status"] == "OFF" else 0, axis=1).sum()
st.success(f"Estimated energy saved by switching HVAC off when unoccupied: {energy_saved:.2f} kWh")