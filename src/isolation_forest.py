import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def run_isolation_forest(input_csv: str, output_csv: str, output_dir: str):
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])

    # Select numeric features for modeling
    features = [
        "temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count",
        "temperature_roll3", "co2_level_roll3", "energy_kwh_roll3", "humidity_roll3", "occupancy_count_roll3"
    ]
    features = [col for col in features if col in df.columns]

    # Drop NAs for selected features
    df = df.dropna(subset=features)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Fit Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    df["anomaly_score"] = model.fit_predict(X_scaled)
    df["is_anomaly"] = (df["anomaly_score"] == -1).astype(int)

    # Save with anomaly label
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Isolation Forest results saved to: {output_csv}")

    # ---- Visualizations ---- #

    # # Single plot: Temperature
    # plt.figure(figsize=(15, 6))
    # sns.lineplot(data=df, x="timestamp", y="temperature", label="Temperature")
    # sns.scatterplot(data=df[df["is_anomaly"] == 1], x="timestamp", y="temperature", color="red", label="Anomaly", s=25)
    # plt.title("Temperature Anomaly Detection (Isolation Forest)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "anomaly_temperature_isoforest.png"))
    # print("📊 Saved: outputs/anomaly_temperature_isoforest.png")
    #
    # Multi-panel anomaly visualiation
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)

    # Temperature
    sns.lineplot(data=df, x="timestamp", y="temperature", ax=axes[0], label="Temperature")
    sns.scatterplot(data=df[df["is_anomaly"] == 1], x="timestamp", y="temperature", ax=axes[0], color="red", s=15, label="Anomaly")
    axes[0].set_title("Temperature Anomalies")

    # Humidity
    sns.lineplot(data=df, x="timestamp", y="humidity", ax=axes[1], label="Humidity", color="purple")
    sns.scatterplot(data=df[df["is_anomaly"] == 1], x="timestamp", y="humidity", ax=axes[1], color="red", s=15, label="Anomaly")
    axes[1].set_title("Humidity Anomalies")

    # CO₂
    sns.lineplot(data=df, x="timestamp", y="co2_level", ax=axes[2], label="CO₂ Level", color="green")
    sns.scatterplot(data=df[df["is_anomaly"] == 1], x="timestamp", y="co2_level", ax=axes[2], color="red", s=15, label="Anomaly")
    axes[2].set_title("CO₂ Level Anomalies")

    # Energy
    sns.lineplot(data=df, x="timestamp", y="energy_kwh", ax=axes[3], label="Energy Usage", color="blue")
    sns.scatterplot(data=df[df["is_anomaly"] == 1], x="timestamp", y="energy_kwh", ax=axes[3], color="red", s=15, label="Anomaly")
    axes[3].set_title("Energy Usage Anomalies")

    for ax in axes:
        ax.legend()
        ax.set_xlabel("")

    plt.tight_layout()
    multi_plot_path = os.path.join(output_dir, "anomaly_multisensor_overview.png")
    plt.savefig(multi_plot_path)
    print(f"📊 Saved: {multi_plot_path}")

if __name__ == "__main__":
    input_path = os.path.join("data", "Smart_Hospital_Sensor_Preprocessed.csv")
    output_path = os.path.join("outputs", "isolation_forest_results.csv")
    run_isolation_forest(input_path, output_path, "outputs")