import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def run_occupancy_forecast(input_csv: str, model_dir: str, output_dir: str):
    # Load data
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.dropna(subset=["occupancy_count"])

    # Feature Engineering
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month

    features = ["temperature", "humidity", "light_level", "co2_level", "energy_kwh", "hour", "weekday", "month"]
    target = "occupancy_count"

    X = df[features]
    y = df[target]

    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model trained — MSE: {mse:.2f}, R²: {r2:.2f}")

    # Save outputs
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "occupancy_forecast_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "occupancy_scaler.pkl"))

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:100], label="Actual", marker="o")
    plt.plot(y_pred[:100], label="Predicted", marker="x")
    plt.title("Occupancy Forecasting (First 100 Samples)")
    plt.xlabel("Time Index")
    plt.ylabel("Occupancy Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occupancy_forecast_plot.png"))
    print(f"📊 Saved: {os.path.join(output_dir, 'occupancy_forecast_plot.png')}")

if __name__ == "__main__":
    run_occupancy_forecast(
        input_csv="data/Smart_Hospital_Sensor_Preprocessed.csv",
        model_dir="models",
        output_dir="outputs"
    )