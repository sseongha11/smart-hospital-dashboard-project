# src/preprocess.py

import pandas as pd
import os


def run_preprocessing(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Sort by time
    df = df.sort_values("timestamp")

    # Feature: Hour, Day, Weekday, Month
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday  # 0=Monday
    df["month"] = df["timestamp"].dt.month

    # Feature: Is weekend
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Feature: Rolling means (3-hour, 6-hour, 24-hour)
    for col in ["temperature", "co2_level", "energy_kwh", "humidity", "occupancy_count"]:
        df[f"{col}_roll3"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_roll6"] = df[col].rolling(window=6, min_periods=1).mean()
        df[f"{col}_roll24"] = df[col].rolling(window=24, min_periods=1).mean()

    # Drop NAs (only if large rolling windows used)
    df.dropna(inplace=True)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to: {output_path}")


if __name__ == "__main__":
    input_csv = os.path.join("data", "Smart_Hospital_Sensor_Data.csv")
    output_csv = os.path.join("data", "Smart_Hospital_Sensor_Preprocessed.csv")
    run_preprocessing(input_csv, output_csv)