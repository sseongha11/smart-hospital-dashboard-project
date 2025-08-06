# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Optional: use plotly for interactivity if available
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

def run_eda(csv_path: str):
    # Load the data
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    print("✅ Data loaded successfully.\n")

    # Create outputs folder if not exists
    output_dir = "../outputs/eda"
    os.makedirs(output_dir, exist_ok=True)

    # Basic statistics table
    print("🔍 Data Summary (Numerical Features):")
    print(df.describe().T)

    print("\n🗂️ Columns:")
    print(df.columns.tolist())

    # Expanded time series plots
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True)

    sns.lineplot(data=df, x=df.index, y="temperature", ax=axes[0, 0], color='orange')
    axes[0, 0].set_title("🌡️ Temperature Over Time")

    sns.lineplot(data=df, x=df.index, y="humidity", ax=axes[0, 1], color='purple')
    axes[0, 1].set_title("💧 Humidity Over Time")

    sns.lineplot(data=df, x=df.index, y="co2_level", ax=axes[1, 0], color='green')
    axes[1, 0].set_title("🫁 CO₂ Level Over Time")

    sns.lineplot(data=df, x=df.index, y="light_level", ax=axes[1, 1], color='gold')
    axes[1, 1].set_title("💡 Light Level Over Time")

    sns.lineplot(data=df, x=df.index, y="energy_kwh", ax=axes[2, 0], color='blue')
    axes[2, 0].set_title("⚡ Energy Usage Over Time")

    sns.lineplot(data=df, x=df.index, y="occupancy_count", ax=axes[2, 1], color='brown')
    axes[2, 1].set_title("👥 Occupancy Count Over Time")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_timeseries_expanded.png"))
    print("📊 Saved: outputs/eda_timeseries_expanded.png")

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("🔗 Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation.png"))
    print("📊 Saved: outputs/eda_correlation.png")

    # Distribution plots
    plt.figure(figsize=(16, 10))
    for i, col in enumerate(["temperature", "co2_level", "humidity", "energy_kwh"], 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"📊 Distribution: {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_distributions.png"))
    print("📊 Saved: outputs/eda_distributions.png")

    # Room-wise average table
    avg_by_room = df.groupby("room_id")[["temperature", "co2_level", "energy_kwh", "humidity"]].mean().round(2)
    print("\n🏠 Room-wise Sensor Averages:")
    print(avg_by_room)

    # Save room-wise averages as CSV
    avg_by_room.to_csv(os.path.join(output_dir, "room_averages.csv"))
    print("📄 Saved: outputs/room_averages.csv")

    # Compare CO2 across rooms
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df.reset_index(), x="timestamp", y="co2_level", hue="room_id", linewidth=1)
    plt.title("🫁 CO₂ Levels by Room")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_co2_by_room.png"))
    print("📊 Saved: outputs/eda_co2_by_room.png")

    # Compare energy usage by building
    if df["building_id"].nunique() > 1:
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df.reset_index(), x="timestamp", y="energy_kwh", hue="building_id", linewidth=1.2)
        plt.title("⚡ Energy Usage by Building")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_energy_by_building.png"))
        print("📊 Saved: outputs/eda_energy_by_building.png")

    # Optional interactive Plotly chart
    if PLOTLY_AVAILABLE:
        fig = px.line(df.reset_index(), x="timestamp", y="co2_level", color="room_id", title="🫁 CO₂ Levels by Room (Interactive)")
        fig.write_html(os.path.join(output_dir, "co2_by_room.html"))
        print("📊 Saved: outputs/co2_by_room.html (interactive)")

if __name__ == "__main__":
    data_path = os.path.join("data", "Smart_Hospital_Sensor_Data.csv")
    run_eda(data_path)