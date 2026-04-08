import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import random
import os
from tqdm import tqdm

# --- Configuration Parameters ---
SAMPLING_RATE = 1000  # Hz (1 kHz for faster processing)
DURATION = 2          # seconds (2 seconds per sample)
NUM_SAMPLES_PER_SCENARIO = 100 # Number of samples for each scenario
BASE_FREQUENCY = 50   # Hz
NOMINAL_CURRENT_RMS = 100 # Amps (RMS)
FAULT_CURRENT_MULTIPLIER = 3 # Reduced for more realistic simulation

# Time vector for a single sample
t = np.linspace(0, DURATION, int(SAMPLING_RATE * DURATION), endpoint=False)

# --- Helper Functions ---

def add_noise(signal, snr_db=30):
    """Adds Gaussian noise to a signal."""
    signal_power = np.mean(signal**2)
    if signal_power == 0:
        return signal
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def create_base_waveform(amplitude, phase_shift_deg=0, harmonic_distortion=0):
    """Creates a 50Hz sinusoidal waveform with optional harmonic distortion."""
    phase_shift_rad = np.deg2rad(phase_shift_deg)
    waveform = amplitude * np.sin(2 * np.pi * BASE_FREQUENCY * t + phase_shift_rad)
    
    if harmonic_distortion > 0:
        # Add 3rd and 5th harmonics for realistic distortion
        waveform += amplitude * 0.1 * harmonic_distortion * np.sin(2 * np.pi * 3 * BASE_FREQUENCY * t + phase_shift_rad)
        waveform += amplitude * 0.05 * harmonic_distortion * np.sin(2 * np.pi * 5 * BASE_FREQUENCY * t + phase_shift_rad)
    return waveform

# --- Scenario Generators ---

def generate_normal_operation():
    """Generates a normal operation waveform."""
    amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * np.random.uniform(0.8, 1.1)
    phase_A = create_base_waveform(amplitude, 0)
    phase_B = create_base_waveform(amplitude, -120)
    phase_C = create_base_waveform(amplitude, -240)
    
    # Add small random noise
    phase_A = add_noise(phase_A, snr_db=np.random.uniform(35, 45))
    phase_B = add_noise(phase_B, snr_db=np.random.uniform(35, 45))
    phase_C = add_noise(phase_C, snr_db=np.random.uniform(35, 45))
    
    return np.column_stack([phase_A, phase_B, phase_C]), "Normal"

def generate_lt_line_cut():
    """Generates an LT line cut (open conductor) waveform."""
    amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * np.random.uniform(0.7, 1.0)
    phase_A = create_base_waveform(amplitude, 0)
    phase_B = create_base_waveform(amplitude, -120)
    phase_C = create_base_waveform(amplitude, -240)
    
    # Choose which phase to cut and when
    cut_phase_idx = np.random.randint(0, 3)
    cut_time_idx = np.random.randint(int(SAMPLING_RATE * 0.3), int(SAMPLING_RATE * (DURATION - 0.3)))
    
    # Apply line cut
    phases = [phase_A, phase_B, phase_C]
    phases[cut_phase_idx][cut_time_idx:] = 0
    
    # Add noise
    phase_A = add_noise(phases[0], snr_db=np.random.uniform(30, 40))
    phase_B = add_noise(phases[1], snr_db=np.random.uniform(30, 40))
    phase_C = add_noise(phases[2], snr_db=np.random.uniform(30, 40))
    
    return np.column_stack([phase_A, phase_B, phase_C]), "LT_Line_Cut"

def generate_ground_fault():
    """Generates a ground fault waveform."""
    amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * np.random.uniform(0.6, 0.9)
    phase_A = create_base_waveform(amplitude, 0)
    phase_B = create_base_waveform(amplitude, -120)
    phase_C = create_base_waveform(amplitude, -240)
    
    # Choose fault phase and timing
    fault_phase_idx = np.random.randint(0, 3)
    fault_time_idx = np.random.randint(int(SAMPLING_RATE * 0.3), int(SAMPLING_RATE * (DURATION - 0.5)))
    
    # Create ground fault surge
    surge_amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * np.random.uniform(2.0, 4.0)
    surge_duration = np.random.randint(int(SAMPLING_RATE * 0.05), int(SAMPLING_RATE * 0.15))
    
    phases = [phase_A, phase_B, phase_C]
    
    # Apply surge and decay
    if fault_time_idx + surge_duration < len(phases[fault_phase_idx]):
        surge_envelope = np.exp(-np.linspace(0, 5, surge_duration))
        surge_signal = surge_amplitude * np.sin(2 * np.pi * BASE_FREQUENCY * t[fault_time_idx:fault_time_idx + surge_duration])
        phases[fault_phase_idx][fault_time_idx:fault_time_idx + surge_duration] += surge_signal * surge_envelope
        
        # Gradual decay after surge
        decay_length = min(int(SAMPLING_RATE * 0.2), len(phases[fault_phase_idx]) - (fault_time_idx + surge_duration))
        if decay_length > 0:
            decay_envelope = np.exp(-np.linspace(0, 3, decay_length))
            original_signal = phases[fault_phase_idx][fault_time_idx + surge_duration:fault_time_idx + surge_duration + decay_length]
            phases[fault_phase_idx][fault_time_idx + surge_duration:fault_time_idx + surge_duration + decay_length] = original_signal * decay_envelope
    
    # Add noise
    phase_A = add_noise(phases[0], snr_db=np.random.uniform(20, 30))
    phase_B = add_noise(phases[1], snr_db=np.random.uniform(20, 30))
    phase_C = add_noise(phases[2], snr_db=np.random.uniform(20, 30))
    
    return np.column_stack([phase_A, phase_B, phase_C]), "Ground_Fault"

def generate_short_circuit():
    """Generates a short circuit waveform."""
    amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * np.random.uniform(0.5, 0.8)
    fault_types = ["single_phase", "phase_to_phase", "three_phase"]
    fault_type = np.random.choice(fault_types)
    
    phase_A = create_base_waveform(amplitude, 0)
    phase_B = create_base_waveform(amplitude, -120)
    phase_C = create_base_waveform(amplitude, -240)
    
    fault_time_idx = np.random.randint(int(SAMPLING_RATE * 0.3), int(SAMPLING_RATE * (DURATION - 0.3)))
    fault_amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * FAULT_CURRENT_MULTIPLIER
    
    if fault_type == "single_phase":
        fault_phase = np.random.randint(0, 3)
        phases = [phase_A, phase_B, phase_C]
        fault_signal = create_base_waveform(fault_amplitude, [0, -120, -240][fault_phase], 
                                           harmonic_distortion=np.random.uniform(0.3, 0.8))
        phases[fault_phase][fault_time_idx:] = fault_signal[fault_time_idx:]
        
    elif fault_type == "phase_to_phase":
        fault_phases = np.random.choice([0, 1, 2], size=2, replace=False)
        phases = [phase_A, phase_B, phase_C]
        for i, phase_idx in enumerate(fault_phases):
            polarity = 1 if i == 0 else -1
            fault_signal = create_base_waveform(fault_amplitude * polarity, [0, -120, -240][phase_idx],
                                               harmonic_distortion=np.random.uniform(0.3, 0.8))
            phases[phase_idx][fault_time_idx:] = fault_signal[fault_time_idx:]
            
    else:  # three_phase
        fault_amplitude *= 1.5
        fault_A = create_base_waveform(fault_amplitude, 0, harmonic_distortion=np.random.uniform(0.5, 1.0))
        fault_B = create_base_waveform(fault_amplitude, -120, harmonic_distortion=np.random.uniform(0.5, 1.0))
        fault_C = create_base_waveform(fault_amplitude, -240, harmonic_distortion=np.random.uniform(0.5, 1.0))
        
        phase_A[fault_time_idx:] = fault_A[fault_time_idx:]
        phase_B[fault_time_idx:] = fault_B[fault_time_idx:]
        phase_C[fault_time_idx:] = fault_C[fault_time_idx:]
    
    # Add noise
    phase_A = add_noise(phase_A, snr_db=np.random.uniform(15, 25))
    phase_B = add_noise(phase_B, snr_db=np.random.uniform(15, 25))
    phase_C = add_noise(phase_C, snr_db=np.random.uniform(15, 25))
    
    return np.column_stack([phase_A, phase_B, phase_C]), "Short_Circuit"

def generate_overload():
    """Generates an overload waveform."""
    initial_amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * np.random.uniform(0.9, 1.1)
    final_amplitude = NOMINAL_CURRENT_RMS * np.sqrt(2) * np.random.uniform(1.3, 2.0)
    
    # Create amplitude ramp
    ramp_start = np.random.randint(int(SAMPLING_RATE * 0.2), int(SAMPLING_RATE * 0.8))
    ramp_end = np.random.randint(ramp_start + int(SAMPLING_RATE * 0.3), len(t))
    
    amplitude_profile = np.full_like(t, initial_amplitude)
    ramp_indices = np.arange(ramp_start, min(ramp_end, len(t)))
    if len(ramp_indices) > 0:
        amplitude_profile[ramp_indices] = np.linspace(initial_amplitude, final_amplitude, len(ramp_indices))
        amplitude_profile[ramp_end:] = final_amplitude
    
    # Generate waveforms with varying amplitude
    phase_A = np.zeros_like(t)
    phase_B = np.zeros_like(t)
    phase_C = np.zeros_like(t)
    
    for i in range(len(t)):
        amp = amplitude_profile[i]
        distortion = (amp - initial_amplitude) / (final_amplitude - initial_amplitude) * 0.3 if final_amplitude != initial_amplitude else 0
        phase_A[i] = amp * np.sin(2 * np.pi * BASE_FREQUENCY * t[i]) * (1 + distortion * np.sin(6 * np.pi * BASE_FREQUENCY * t[i]))
        phase_B[i] = amp * np.sin(2 * np.pi * BASE_FREQUENCY * t[i] - 2*np.pi/3) * (1 + distortion * np.sin(6 * np.pi * BASE_FREQUENCY * t[i]))
        phase_C[i] = amp * np.sin(2 * np.pi * BASE_FREQUENCY * t[i] - 4*np.pi/3) * (1 + distortion * np.sin(6 * np.pi * BASE_FREQUENCY * t[i]))
    
    # Add noise
    phase_A = add_noise(phase_A, snr_db=np.random.uniform(25, 35))
    phase_B = add_noise(phase_B, snr_db=np.random.uniform(25, 35))
    phase_C = add_noise(phase_C, snr_db=np.random.uniform(25, 35))
    
    return np.column_stack([phase_A, phase_B, phase_C]), "Overload"

# --- Feature Extraction Functions ---

def extract_features(waveform_data):
    """Extract features from waveform data for machine learning."""
    features = []
    
    for phase_idx in range(3):  # For each phase
        phase_data = waveform_data[:, phase_idx]
        
        # Time domain features
        rms = np.sqrt(np.mean(phase_data**2))
        peak = np.max(np.abs(phase_data))
        std_dev = np.std(phase_data)
        mean_val = np.mean(phase_data)
        
        # Frequency domain features
        fft_data = np.fft.fft(phase_data)
        freq_spectrum = np.abs(fft_data)
        
        # Fundamental frequency component
        fund_freq_idx = int(BASE_FREQUENCY * DURATION)
        if fund_freq_idx < len(freq_spectrum):
            fundamental_mag = freq_spectrum[fund_freq_idx]
        else:
            fundamental_mag = 0
            
        # Total Harmonic Distortion (THD)
        harmonics = []
        for h in range(2, 6):  # 2nd to 5th harmonics
            harm_idx = int(h * BASE_FREQUENCY * DURATION)
            if harm_idx < len(freq_spectrum):
                harmonics.append(freq_spectrum[harm_idx])
            else:
                harmonics.append(0)
        
        thd = np.sqrt(sum(h**2 for h in harmonics)) / (fundamental_mag + 1e-10)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(phase_data)) != 0)
        
        features.extend([rms, peak, std_dev, mean_val, fundamental_mag, thd, zero_crossings])
    
    # Three-phase features
    # Phase imbalance
    phase_rms = [np.sqrt(np.mean(waveform_data[:, i]**2)) for i in range(3)]
    avg_rms = np.mean(phase_rms)
    phase_imbalance = np.max(phase_rms) - np.min(phase_rms)
    
    # Zero sequence current (approximation)
    zero_seq = np.mean(np.sum(waveform_data, axis=1))
    
    features.extend([phase_imbalance, zero_seq, avg_rms])
    
    return np.array(features)

# --- Dataset Generation ---

def generate_full_dataset(num_samples_per_scenario):
    """Generate complete dataset with features and labels."""
    scenario_generators = {
        "Normal": generate_normal_operation,
        "LT_Line_Cut": generate_lt_line_cut,
        "Ground_Fault": generate_ground_fault,
        "Short_Circuit": generate_short_circuit,
        "Overload": generate_overload,
    }
    
    all_features = []
    all_labels = []
    all_raw_data = []
    example_data = {}
    
    print("Generating dataset...")
    
    for scenario_name, generator_func in scenario_generators.items():
        print(f"Generating {num_samples_per_scenario} samples for: {scenario_name}")
        
        for i in tqdm(range(num_samples_per_scenario), desc=f"{scenario_name}"):
            waveform, label = generator_func()
            
            # Store first example for plotting
            if label not in example_data:
                example_data[label] = waveform
            
            # Extract features
            features = extract_features(waveform)
            
            all_features.append(features)
            all_labels.append(label)
            all_raw_data.append(waveform.flatten())  # Flatten for storage
    
    return np.array(all_features), np.array(all_labels), np.array(all_raw_data), example_data

# --- Plotting Function ---

def plot_examples(example_data):
    """Plot example waveforms for each fault type."""
    fig, axes = plt.subplots(len(example_data), 1, figsize=(12, 3 * len(example_data)))
    
    if len(example_data) == 1:
        axes = [axes]
    
    for i, (label, data) in enumerate(example_data.items()):
        ax = axes[i]
        ax.plot(t, data[:, 0], label='Phase A', color='red', linewidth=1)
        ax.plot(t, data[:, 1], label='Phase B', color='green', linewidth=1)
        ax.plot(t, data[:, 2], label='Phase C', color='blue', linewidth=1)
        ax.set_title(f'{label} Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('example_waveforms.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # Generate dataset
    features, labels, raw_data, examples = generate_full_dataset(NUM_SAMPLES_PER_SCENARIO)
    
    print(f"\nDataset generated successfully!")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Raw data shape: {raw_data.shape}")
    
    # Print label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} samples")
    
    # Create feature names
    feature_names = []
    for phase in ['A', 'B', 'C']:
        feature_names.extend([
            f'RMS_Phase_{phase}', f'Peak_Phase_{phase}', f'Std_Phase_{phase}',
            f'Mean_Phase_{phase}', f'Fundamental_Phase_{phase}', f'THD_Phase_{phase}',
            f'ZeroCrossings_Phase_{phase}'
        ])
    feature_names.extend(['Phase_Imbalance', 'Zero_Sequence', 'Avg_RMS'])
    
    # Save datasets
    # Features dataset
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['label'] = labels
    features_df.to_csv('power_system_features.csv', index=False)
    
    # Raw waveform dataset (flattened)
    raw_columns = []
    for phase in ['A', 'B', 'C']:
        for i in range(len(t)):
            raw_columns.append(f'Phase_{phase}_t_{i}')
    
    raw_df = pd.DataFrame(raw_data, columns=raw_columns)
    raw_df['label'] = labels
    raw_df.to_csv('power_system_raw_waveforms.csv', index=False)
    
    print(f"\nDatasets saved:")
    print(f"  Features: power_system_features.csv ({features_df.shape[0]} rows, {features_df.shape[1]} columns)")
    print(f"  Raw waveforms: power_system_raw_waveforms.csv ({raw_df.shape[0]} rows, {raw_df.shape[1]} columns)")
    
    # Plot examples
    plot_examples(examples)
    
    print("\nDataset generation completed successfully!")