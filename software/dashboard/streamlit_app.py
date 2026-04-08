import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import json
import time
from datetime import datetime
import random

# --- Page Configuration (Best to put this at the top) ---
st.set_page_config(
    page_title="Power System Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-container {
    background: #f0f2f6; /* Simplified background for normal state */
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 0.5rem 0;
}

/* --- NEW AND IMPROVED FAULT STYLES START HERE --- */
.fault-detected {
    background-color: #d32f2f; /* A strong, solid red background */
    color: white; /* Make all default text inside WHITE */
    border-left: 5px solid #b71c1c; /* A darker red for the border */
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(211, 47, 47, 0); }
    100% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0); }
}

.status-normal {
    color: #4caf50;
    font-weight: bold;
}

.status-fault {
    color: white; /* Ensure fault name is also white */
    font-weight: 900; /* Make it extra bold to stand out */
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5); /* Add a subtle shadow for depth */
    animation: blink 1s infinite;
}

@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
/* --- END OF NEW STYLES --- */

</style>
""", unsafe_allow_html=True)


# --- Core Application Class ---
class PowerSystemMonitor:
    def __init__(self):
        self.load_model()
        self.setup_parameters()

    def load_model(self):
        """Load the trained model and preprocessors."""
        try:
            self.model = joblib.load('power_system_fault_model.pkl')
            self.scaler = joblib.load('power_system_fault_model_scaler.pkl')
            self.label_encoder = joblib.load('power_system_fault_model_label_encoder.pkl')
            with open('power_system_fault_model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
            st.error("⚠️ Model files not found! Please run the dataset generator and model trainer first.")

    def setup_parameters(self):
        """Setup simulation parameters."""
        self.sampling_rate = 1000
        self.duration = 2
        self.base_frequency = 50
        self.nominal_current_rms = 100
        self.t = np.linspace(0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)

    def generate_waveform(self, fault_type="Normal", noise_level=0.1):
        """Generate waveform based on fault type."""
        amplitude = self.nominal_current_rms * np.sqrt(2) * (1 + random.uniform(-0.1, 0.1))
        
        if fault_type == "Normal":
            phase_A = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t)
            phase_B = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 2*np.pi/3)
            phase_C = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 4*np.pi/3)
        elif fault_type == "LT_Line_Cut":
            cut_phase = random.randint(0, 2)
            cut_time = random.uniform(0.3, 1.7)
            cut_idx = int(cut_time * self.sampling_rate)
            phase_A = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t)
            phase_B = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 2*np.pi/3)
            phase_C = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 4*np.pi/3)
            phases = [phase_A, phase_B, phase_C]
            phases[cut_phase][cut_idx:] = 0
            phase_A, phase_B, phase_C = phases
        elif fault_type == "Ground_Fault":
            fault_phase = random.randint(0, 2)
            fault_time = random.uniform(0.3, 1.0)
            fault_idx = int(fault_time * self.sampling_rate)
            phase_A = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t)
            phase_B = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 2*np.pi/3)
            phase_C = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 4*np.pi/3)
            surge_amplitude = amplitude * 3
            surge_duration = int(0.1 * self.sampling_rate)
            surge_end = min(fault_idx + surge_duration, len(self.t))
            phases = [phase_A, phase_B, phase_C]
            surge_envelope = np.exp(-np.linspace(0, 5, surge_end - fault_idx))
            phases[fault_phase][fault_idx:surge_end] += surge_amplitude * surge_envelope * np.sin(2 * np.pi * self.base_frequency * self.t[fault_idx:surge_end])
            phase_A, phase_B, phase_C = phases
        elif fault_type == "Short_Circuit":
            fault_time = random.uniform(0.3, 1.0)
            fault_idx = int(fault_time * self.sampling_rate)
            fault_amplitude = amplitude * 3
            phase_A = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t)
            phase_B = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 2*np.pi/3)
            phase_C = amplitude * np.sin(2 * np.pi * self.base_frequency * self.t - 4*np.pi/3)
            phase_A[fault_idx:] = fault_amplitude * np.sin(2 * np.pi * self.base_frequency * self.t[fault_idx:])
            phase_B[fault_idx:] = fault_amplitude * np.sin(2 * np.pi * self.base_frequency * self.t[fault_idx:] - 2*np.pi/3)
            phase_C[fault_idx:] = fault_amplitude * np.sin(2 * np.pi * self.base_frequency * self.t[fault_idx:] - 4*np.pi/3)
        elif fault_type == "Overload":
            ramp_start, ramp_end = int(0.3 * self.sampling_rate), int(1.2 * self.sampling_rate)
            final_amplitude = amplitude * 1.5
            amplitude_profile = np.ones_like(self.t) * amplitude
            amplitude_profile[ramp_start:ramp_end] = np.linspace(amplitude, final_amplitude, ramp_end - ramp_start)
            amplitude_profile[ramp_end:] = final_amplitude
            phase_A = amplitude_profile * np.sin(2 * np.pi * self.base_frequency * self.t)
            phase_B = amplitude_profile * np.sin(2 * np.pi * self.base_frequency * self.t - 2*np.pi/3)
            phase_C = amplitude_profile * np.sin(2 * np.pi * self.base_frequency * self.t - 4*np.pi/3)
        
        noise_factor = noise_level * amplitude * 0.5 # Reduced noise factor for clarity
        phase_A += np.random.normal(0, noise_factor, len(self.t))
        phase_B += np.random.normal(0, noise_factor, len(self.t))
        phase_C += np.random.normal(0, noise_factor, len(self.t))
        
        return np.column_stack([phase_A, phase_B, phase_C])

    def extract_features(self, waveform_data):
        """Extract features from waveform data. (Verified to produce 24 features)"""
        features = []
        
        # Per-phase features (7 features x 3 phases = 21 features)
        for phase_idx in range(3):
            phase_data = waveform_data[:, phase_idx]
            rms = np.sqrt(np.mean(phase_data**2))
            peak = np.max(np.abs(phase_data))
            std_dev = np.std(phase_data)
            mean_val = np.mean(phase_data)
            fft_data = np.fft.fft(phase_data)
            freq_spectrum = np.abs(fft_data)
            fund_freq_idx = int(self.base_frequency * self.duration)
            fundamental_mag = freq_spectrum[fund_freq_idx] if fund_freq_idx < len(freq_spectrum) else 0
            harmonics = []
            for h in range(2, 6):
                harm_idx = int(h * self.base_frequency * self.duration)
                harmonics.append(freq_spectrum[harm_idx] if harm_idx < len(freq_spectrum) else 0)
            thd = np.sqrt(sum(h**2 for h in harmonics)) / (fundamental_mag + 1e-10)
            zero_crossings = np.sum(np.diff(np.sign(phase_data)) != 0)
            features.extend([rms, peak, std_dev, mean_val, fundamental_mag, thd, zero_crossings])
        
        # Three-phase features (3 features)
        phase_rms = [np.sqrt(np.mean(waveform_data[:, i]**2)) for i in range(3)]
        avg_rms = np.mean(phase_rms)
        phase_imbalance = np.max(phase_rms) - np.min(phase_rms)
        zero_seq = np.mean(np.sum(waveform_data, axis=1))
        features.extend([phase_imbalance, zero_seq, avg_rms])
        
        return np.array(features)

    def predict_fault(self, waveform):
        """Predict fault type from waveform."""
        if not self.model_loaded:
            return "Model Not Loaded", {}

        # 1. Extract features as a NumPy array
        features_np = self.extract_features(waveform)
        
        # 2. Convert to a DataFrame with named columns to resolve UserWarning
        features_df = pd.DataFrame(features_np.reshape(1, -1), columns=self.metadata['feature_names'])

        # 3. Predict using the named DataFrame
        if self.metadata['scaler_needed']:
            features_processed = self.scaler.transform(features_df)
            prediction = self.model.predict(features_processed)[0]
            probabilities = self.model.predict_proba(features_processed)[0]
        else:
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]

        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        prob_dict = {label: prob for i, (label, prob) in enumerate(zip(self.label_encoder.classes_, probabilities))}
        
        return predicted_label, prob_dict

# --- UI Functions for Each Mode ---
def show_real_time_monitoring(monitor):
    """Real-time monitoring dashboard."""
    st.markdown("## 📈 Real-time Power System Monitoring")
    col1, col2, col3, col4 = st.columns(4)
    auto_run = col1.checkbox("🔄 Auto Refresh", value=True)
    refresh_rate = col2.selectbox("⏱️ Refresh Rate (s)", [1, 2, 5], index=1)
    noise_level = col3.slider("🔊 Noise Level", 0.0, 0.3, 0.1, 0.05)
    fault_probability = col4.slider("⚠️ Fault Probability", 0.0, 0.5, 0.1, 0.05)

    status_container, charts_container, metrics_container = st.container(), st.container(), st.container()

    if 'run_once' not in st.session_state:
        st.session_state.run_once = True
    
    if auto_run or st.session_state.run_once:
        st.session_state.run_once = False
        actual_fault = "Normal"
        if random.random() < fault_probability:
            fault_types = list(monitor.label_encoder.classes_)
            fault_types.remove("Normal")
            actual_fault = random.choice(fault_types)
        
        waveform = monitor.generate_waveform(actual_fault, noise_level)
        predicted_fault, probabilities = monitor.predict_fault(waveform)

        with status_container:
            if predicted_fault != "Normal":
                st.markdown(f'<div class="metric-container fault-detected"><h2>🚨 FAULT DETECTED: <span class="status-fault">{predicted_fault}</span></h2><p>Confidence: {max(probabilities.values()):.2%}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-container"><h2>✅ System Status: <span class="status-normal">NORMAL</span></h2><p>All parameters within acceptable range</p></div>', unsafe_allow_html=True)
        
        with charts_container:
            col1, col2 = st.columns(2)
            with col1:
                fig_wave = go.Figure()
                for i, (phase, color) in enumerate(zip(['A', 'B', 'C'], ['red', 'green', 'blue'])):
                    fig_wave.add_trace(go.Scatter(x=monitor.t, y=waveform[:, i], mode='lines', name=f'Phase {phase}', line=dict(color=color, width=2)))
                fig_wave.update_layout(title="Real-time Current Waveforms", xaxis_title="Time (s)", yaxis_title="Current (A)", height=400, template="plotly_white")
                st.plotly_chart(fig_wave, use_container_width=True)
            with col2:
                fig_prob = px.bar(x=list(probabilities.keys()), y=list(probabilities.values()), title="Fault Classification Probabilities", labels={'x':'Fault Type', 'y':'Probability'}, height=400, template="plotly_white")
                fig_prob.update_traces(marker_color=['#f44336' if k == predicted_fault else '#1f77b4' for k in probabilities.keys()])
                st.plotly_chart(fig_prob, use_container_width=True)

        with metrics_container:
            st.markdown("### 📊 System Metrics")
            phase_rms = [np.sqrt(np.mean(waveform[:, i]**2)) for i in range(3)]
            imbalance = max(phase_rms) - min(phase_rms)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Phase A RMS", f"{phase_rms[0]:.1f} A", f"{phase_rms[0] - 100:.1f} A")
            col2.metric("Phase B RMS", f"{phase_rms[1]:.1f} A", f"{phase_rms[1] - 100:.1f} A")
            col3.metric("Phase C RMS", f"{phase_rms[2]:.1f} A", f"{phase_rms[2] - 100:.1f} A")
            col4.metric("Frequency", f"50.0 Hz", "0.0 Hz")
            col5.metric("Phase Imbalance", f"{imbalance:.1f} A", delta_color="inverse")

        if auto_run:
            time.sleep(refresh_rate)
            st.rerun()

def show_manual_testing(monitor):
    st.markdown("## 🧪 Manual Fault Testing")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Test Parameters")
        fault_type = st.selectbox("Select Fault Type to Simulate", list(monitor.label_encoder.classes_))
        noise_level = st.slider("Adjust Noise Level", 0.0, 0.5, 0.1, 0.05)
        if st.button("🎯 Generate & Test Case", type="primary", use_container_width=True):
            st.session_state.test_waveform = monitor.generate_waveform(fault_type, noise_level)
            st.session_state.test_fault_type = fault_type

    with col2:
        if 'test_waveform' in st.session_state:
            waveform, actual_fault = st.session_state.test_waveform, st.session_state.test_fault_type
            predicted_fault, probabilities = monitor.predict_fault(waveform)
            
            st.markdown("### Test Results")
            res_col1, res_col2 = st.columns(2)
            res_col1.markdown(f"**Actual Fault:** `{actual_fault}`")
            res_col1.markdown(f"**Predicted Fault:** `{predicted_fault}`")
            if actual_fault == predicted_fault: res_col1.success("✅ Correct Classification")
            else: res_col1.error("❌ Incorrect Classification")
            
            res_col2.write("**Classification Confidence:**")
            for fault, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
                res_col2.progress(float(prob), text=f"{fault}: {prob:.2%}")

            fig = go.Figure()
            for i, (phase, color) in enumerate(zip(['A', 'B', 'C'], ['red', 'green', 'blue'])):
                fig.add_trace(go.Scatter(x=monitor.t, y=waveform[:, i], mode='lines', name=f'Phase {phase}', line=dict(color=color)))
            fig.update_layout(title=f"Test Waveform: {actual_fault}", height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

def show_historical_analysis(monitor):
    st.markdown("## 📜 Historical Analysis")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Historical Data Settings")
    num_days = st.sidebar.slider("Days of History", 1, 30, 7)
    fault_occurrence_rate = st.sidebar.slider("Fault Rate (per hour)", 0.0, 0.5, 0.05, 0.01)

    if st.button("📈 Generate Historical Data", use_container_width=True):
        progress_bar = st.progress(0, text="Generating historical data...")
        timestamps = pd.date_range(end=datetime.now(), periods=num_days * 24, freq='H')
        data = []
        for i, ts in enumerate(timestamps):
            actual_fault = "Normal"
            if random.random() < fault_occurrence_rate:
                fault_types = list(monitor.label_encoder.classes_); fault_types.remove("Normal")
                actual_fault = random.choice(fault_types)
            waveform = monitor.generate_waveform(actual_fault, random.uniform(0.05, 0.2))
            predicted_fault, probabilities = monitor.predict_fault(waveform)
            rms_vals = [np.sqrt(np.mean(waveform[:, i]**2)) for i in range(3)]
            data.append({'timestamp': ts, 'actual_fault': actual_fault, 'predicted_fault': predicted_fault, 'confidence': max(probabilities.values()), 'rms_a': rms_vals[0], 'rms_b': rms_vals[1], 'rms_c': rms_vals[2]})
            progress_bar.progress((i + 1) / len(timestamps), text=f"Simulating hour {i+1}/{len(timestamps)}")
        st.session_state.historical_df = pd.DataFrame(data)
        progress_bar.empty()

    if 'historical_df' in st.session_state:
        df = st.session_state.historical_df
        st.markdown("### Overview of Historical Faults")
        fault_counts = df['predicted_fault'].value_counts()
        fig_pie = px.pie(fault_counts, values=fault_counts.values, names=fault_counts.index, title='Distribution of Predicted Fault Types')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("### Timeline of Predicted Faults and RMS Current")
        fig_timeline = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Phase RMS Over Time", "Predicted Fault Events"))
        for phase, color in zip(['a', 'b', 'c'], ['red', 'green', 'blue']):
            fig_timeline.add_trace(go.Scatter(x=df['timestamp'], y=df[f'rms_{phase}'], mode='lines', name=f'RMS Phase {phase.upper()}', line=dict(color=color, width=1)), row=1, col=1)
        fault_events = df[df['predicted_fault'] != 'Normal']
        fig_timeline.add_trace(go.Scatter(x=fault_events['timestamp'], y=[1]*len(fault_events), mode='markers', name='Fault Event', marker=dict(symbol='x', size=8, color='black'), text=fault_events['predicted_fault'], hoverinfo='text'), row=2, col=1)
        fig_timeline.update_layout(height=600, hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("### Detailed Historical Records")
        st.dataframe(df)

# --- Main App Logic ---
def main():
    st.markdown('<h1 class="main-header">⚡ Power System Fault Detection & Monitoring</h1>', unsafe_allow_html=True)

    if 'monitor' not in st.session_state:
        st.session_state.monitor = PowerSystemMonitor()
    monitor = st.session_state.monitor

    if not monitor.model_loaded:
        st.warning("Application is running in a limited mode. Please train a model to enable full functionality.")
        return

    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.success(f"✅ Model Loaded: **{monitor.metadata['model_name']}**")
    mode = st.sidebar.radio(
        "Select Monitoring Mode",
        ["Real-time Simulation", "Manual Fault Injection", "Historical Analysis"],
        captions=["Live feed simulation", "Test specific faults", "Analyze past data"]
    )

    if mode == "Real-time Simulation":
        show_real_time_monitoring(monitor)
    elif mode == "Manual Fault Injection":
        show_manual_testing(monitor)
    else:
        show_historical_analysis(monitor)

if __name__ == "__main__":
    main()