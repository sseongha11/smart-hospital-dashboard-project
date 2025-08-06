import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import GRU, Dense, Dropout
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

def run_rnn_energy_forecasting(input_csv: str, output_dir: str):
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    features = [
        "temperature", "humidity", "co2_level", "occupancy_count", "light_level",
        "hour", "weekday", "month", "is_weekend"
    ]
    target = "energy_kwh"
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    SEQ_LEN = 24
    X_seq, y_seq = create_sequences(X_scaled, SEQ_LEN)
    _, y_target = create_sequences(y_scaled, SEQ_LEN)

    # Split
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_target[:train_size], y_target[train_size:]

    model = Sequential([
        GRU(64, return_sequences=False, input_shape=(SEQ_LEN, X_seq.shape[2])),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=50,
              batch_size=64,
              callbacks=[early_stop],
              verbose=0)

    # Save the model with `.keras` extension
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "rnn_energy_forecasting_model.keras")
    save_model(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Evaluate
    y_pred = model.predict(X_test).flatten()
    y_test_flat = y_test.flatten()
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred))
    r2 = r2_score(y_test_flat, y_pred)
    print(f"🔢 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.4f}")

    df_results = pd.DataFrame({"actual": y_test_flat, "predicted": y_pred})
    df_results.to_csv(os.path.join(output_dir, "energy_forecasting_rnn_results.csv"), index=False)

    # Plot results
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_results.reset_index(drop=True))
    plt.title("⚡ RNN (GRU) Energy Forecasting")
    plt.xlabel("Samples")
    plt.ylabel("Energy (kWh)")
    plt.legend(["Actual", "Predicted"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_forecasting_rnn_plot.png"))
    print("📊 Saved: energy_forecasting_rnn_plot.png")

    # Forecast next 24 hours using last known sequence
    forecast_input = X_scaled[-SEQ_LEN:]
    forecast_seq = forecast_input.reshape(1, SEQ_LEN, X_seq.shape[2])

    future_predictions = []
    for _ in range(24):
        pred = model.predict(forecast_seq)[0][0]
        future_predictions.append(pred)
        next_input = np.append(forecast_seq[0][1:], [forecast_input[-1]], axis=0)
        forecast_seq = next_input.reshape(1, SEQ_LEN, X_seq.shape[2])

    forecast_df = pd.DataFrame({
        "step_ahead": list(range(1, 25)),
        "predicted_energy_kwh": future_predictions
    })
    forecast_df.to_csv(os.path.join(output_dir, "rnn_energy_forecast_next_24h.csv"), index=False)
    print("📈 Forecast saved: rnn_energy_forecast_next_24h.csv")

if __name__ == "__main__":
    input_path = os.path.join("data", "Smart_Hospital_Sensor_Preprocessed.csv")
    run_rnn_energy_forecasting(input_path, "outputs")