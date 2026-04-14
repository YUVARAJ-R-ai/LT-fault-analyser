import numpy as np
import pandas as pd
import os

NUM_SAMPLES_PER_CLASS = 200
SAMPLE_RATE = 1000
DURATION = 1.0
NOMINAL_FREQ = 50.0
NOMINAL_CURRENT_PEAK = 0.25 

def generate_waveform(condition, t):
    base_wave = np.sin(2 * np.pi * NOMINAL_FREQ * t)
    noise = np.random.normal(0, 0.01, len(t))
    if condition == "Normal":
        return (NOMINAL_CURRENT_PEAK * base_wave) + noise
    elif condition == "Line_Cut":
        return np.random.normal(0, 0.002, len(t))
    elif condition == "Overload":
        multiplier = np.random.uniform(2.0, 3.5)
        return (NOMINAL_CURRENT_PEAK * multiplier * base_wave) + noise
    elif condition == "Short_Circuit":
        multiplier = np.random.uniform(10.0, 20.0)
        return (NOMINAL_CURRENT_PEAK * multiplier * base_wave) + noise

def calculate_features(waveform):
    rms = np.sqrt(np.mean(waveform**2))
    peak = np.max(np.abs(waveform))
    mean_val = np.mean(waveform)
    var = np.var(waveform)
    return [rms, peak, mean_val, var]

def main():
    print("Generating 1-Phase Synthetic Data for Hardware POC...")
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    features_list = []
    labels = []
    conditions = ["Normal", "Line_Cut", "Overload", "Short_Circuit"]
    for condition in conditions:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            wave = generate_waveform(condition, t)
            feats = calculate_features(wave)
            features_list.append(feats)
            labels.append(condition)
    df = pd.DataFrame(features_list, columns=["RMS", "Peak", "Mean", "Variance"])
    df["Label"] = labels
    os.makedirs("data", exist_ok=True)
    csv_path = "data/1_phase_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data generated to {csv_path}")

if __name__ == "__main__":
    main()
