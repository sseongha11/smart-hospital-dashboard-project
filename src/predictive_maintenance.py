# src/predictive_maintenance.py

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def generate_synthetic_labels(df):
    # Create label based on synthetic thresholds (simulating maintenance need)
    df["needs_maintenance"] = (
        (df["temperature"] > 30) &
        (df["co2_level"] > 1200) &
        (df["humidity"] > 70) &
        (df["occupancy_count"] > 20)
    ).astype(int)
    return df

def run_binary_maintenance_classifier(input_csv: str, output_dir: str):
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    df = df.dropna()
    df = generate_synthetic_labels(df)

    features = ["temperature", "humidity", "co2_level", "energy_kwh", "occupancy_count", "light_level"]
    target = "needs_maintenance"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    os.makedirs(output_dir, exist_ok=True)

    # Save model and scaler
    joblib.dump(clf, os.path.join(output_dir, "maintenance_classifier.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "maintenance_scaler.pkl"))

    # Add predictions to original dataset
    df["predicted_needs_maintenance"] = clf.predict(X_scaled)
    df.to_csv(os.path.join(output_dir, "maintenance_predictions.csv"), index=False)

    # Print evaluation metrics
    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print("📊 Saved confusion matrix to outputs.")

if __name__ == "__main__":
    input_path = os.path.join("data", "Smart_Hospital_Sensor_Preprocessed.csv")
    run_binary_maintenance_classifier(input_path, "outputs")