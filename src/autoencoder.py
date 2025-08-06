# src/autoencoder.py

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

def run_autoencoder(input_csv: str, output_csv: str, output_dir: str):
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])

    # Select features
    features = [
        "temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count",
        "temperature_roll6", "humidity_roll6", "co2_level_roll6", "energy_kwh_roll6"
    ]
    features = [col for col in features if col in df.columns]
    df = df.dropna(subset=features)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Train-test split
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

    # Build autoencoder
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = autoencoder.fit(X_train, X_train,
                              validation_data=(X_test, X_test),
                              epochs=100,
                              batch_size=64,
                              callbacks=[early_stop],
                              verbose=0)

    # Reconstruction error
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    df["reconstruction_error"] = mse
    threshold = np.percentile(mse, 99)
    df["is_anomaly_autoencoder"] = (mse > threshold).astype(int)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Autoencoder results saved to: {output_csv}")

    # Plot reconstruction error
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df, x="timestamp", y="reconstruction_error", label="Reconstruction Error")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Anomaly Threshold")
    plt.title("Reconstruction Error (Autoencoder)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "autoencoder_reconstruction_error.png"))
    print("📊 Saved: autoencoder_reconstruction_error.png")

if __name__ == "__main__":
    input_path = os.path.join("data", "Smart_Hospital_Sensor_Preprocessed.csv")
    output_path = os.path.join("outputs", "autoencoder_results.csv")
    run_autoencoder(input_path, output_path, "outputs")