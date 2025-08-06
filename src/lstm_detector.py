import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_lstm_detector(input_csv: str, output_csv: str, output_dir: str):
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # 1️⃣ Select and scale 5 core features
    features = ["temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count"]
    df = df.dropna(subset=features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # 2️⃣ Create sequences
    seq_length = 24  # 24-hour window
    X_seq, y_seq = create_sequences(X_scaled, seq_length)

    # 3️⃣ Split data
    train_size = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # 4️⃣ Build model
    model = Sequential([
        Input(shape=(seq_length, X_scaled.shape[1])),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(X_scaled.shape[1])
    ])
    model.compile(optimizer="adam", loss="mse")

    # 5️⃣ Train
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=50,
              batch_size=64,
              callbacks=[early_stop],
              verbose=0)

    # 6️⃣ Predict
    y_pred = model.predict(X_seq, verbose=0)
    mse = np.mean(np.power(y_seq - y_pred, 2), axis=1)
    mse_full = np.concatenate([np.zeros(seq_length), mse])
    df["reconstruction_error_lstm"] = mse_full
    threshold = np.percentile(mse, 99)
    df["is_anomaly_lstm"] = (df["reconstruction_error_lstm"] > threshold).astype(int)

    # 7️⃣ Save results
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ LSTM results saved to: {output_csv}")

    # 8️⃣ Save model
    model_path = os.path.join(output_dir, "lstm_anomaly_model.keras")
    save_model(model, model_path)
    print(f"💾 LSTM model saved to: {model_path}")

    # 9️⃣ Plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df, x="timestamp", y="reconstruction_error_lstm", label="Reconstruction Error")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Anomaly Threshold")
    plt.title("LSTM-Based Anomaly Detection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_reconstruction_error.png"))
    print("📊 Saved: lstm_reconstruction_error.png")

if __name__ == "__main__":
    input_path = os.path.join("data", "Smart_Hospital_Sensor_Preprocessed.csv")
    output_path = os.path.join("outputs", "lstm_results.csv")
    run_lstm_detector(input_path, output_path, "outputs")