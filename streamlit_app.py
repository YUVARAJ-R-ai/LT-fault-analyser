import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import json
import time
from datetime import datetime, timedelta
import random
from scipy.signal import butter, lfilter

# Configure Streamlit page
st.set_page_config(
    page_title="Power System Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .fault-detected {
        background: linear-gradient(90deg, #ffebee 0%, #ffffff 100%);
        border-left: 5px solid #f44336;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .status-normal {
        color: #4caf50;
        font-weight: bold;
    }
    .status-fault {
        color: #f44336;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
            st.error("⚠️ Model files not found! Please train the model first.")
    
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
            
            # Add surge
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
            
            # Apply fault to all phases
            phase_A[fault_idx:] = fault_amplitude * np.sin(2 * np.pi * self.base_frequency * self.t[fault_idx:])
            phase_B[fault_idx:] = fault_amplitude * np.sin(2 * np.pi * self.base_frequency * self.t[fault_idx:] - 2*np.pi/3)
            phase_C[fault_idx:] = fault_amplitude * np.sin(2 * np.pi * self.base_frequency * self.t[fault_idx:] - 4*np.pi/3)
        
        elif fault_type == "Overload":
            ramp_start = int(0.3 * self.sampling_rate)
            ramp_end = int(1.2 * self.sampling_rate)
            final_amplitude = amplitude * 1.5
            
            amplitude_profile = np.ones_like(self.t) * amplitude
            amplitude_profile[ramp_start:ramp_end] = np.linspace(amplitude, final_amplitude, ramp_end - ramp_start)
            amplitude_profile[ramp_end:] = final_amplitude
            
            phase_A = amplitude_profile * np.sin(2 * np.pi * self.base_frequency * self.t)
            phase_B = amplitude_profile * np.sin(2 * np.pi * self.base_frequency * self.t - 2*np.pi/3)
            phase_C = amplitude_profile * np.sin(2 * np.pi * self.base_frequency * self.t - 4*np.pi/3)
        
        # Add noise
        noise_factor = noise_level * amplitude
        phase_A += np.random.normal(0, noise_factor, len(self.t))
        phase_B += np.random.normal(0, noise_factor, len(self.t))
        phase_C += np.random.normal(0, noise_factor, len(self.t))
        
        return np.column_stack([phase_A, phase_B, phase_C])
    
    def extract_features(self, waveform_data):
        """Extract features from waveform data."""
        features = []
        
        for phase_idx in range(3):
            phase_data = waveform_data[:, phase_idx]
            
            # Time domain features
            rms = np.sqrt(np.mean(phase_data**2))
            peak = np.max(np.abs(phase_data))
            std_dev = np.std(phase_data)
            mean_val = np.mean(phase_data)
            
            # Frequency domain features
            fft_data = np.fft.fft(phase_data)
            freq_spectrum = np.abs(fft_data)
            
            fund_freq_idx = int(self.base_frequency * self.duration)
            if fund_freq_idx < len(freq_spectrum):
                fundamental_mag = freq_spectrum[fund_freq_idx]
            else:
                fundamental_mag = 0
            
            # THD calculation
            harmonics = []
            for h in range(2, 6):
                harm_idx = int(h * self.base_frequency * self.duration)
                if harm_idx < len(freq_spectrum):
                    harmonics.append(freq_spectrum[harm_idx])
                else:
                    harmonics.append(0)
            
            thd = np.sqrt(sum(h**2 for h in harmonics)) / (fundamental_mag + 1e-10)
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(phase_data)) != 0)
            
            features.extend([rms, peak, std_dev, mean_val, fundamental_mag, thd, zero_crossings])
        
        # Three-phase features
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
        
        features = self.extract_features(waveform)
        
        if self.metadata['scaler_needed']:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
        else:
            prediction = self.model.predict(features.reshape(1, -1))[0]
            probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            label = self.label_encoder.inverse_transform([i])[0]
            prob_dict[label] = prob
        
        return predicted_label, prob_dict

def main():
    # Title
    st.markdown('<h1 class="main-header">⚡ Power System Fault Detection & Monitoring</h1>', unsafe_allow_html=True)
    
    # Initialize monitor
    if 'monitor' not in st.session_state:
        st.session_state.monitor = PowerSystemMonitor()
    
    monitor = st.session_state.monitor
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Model status
    if monitor.model_loaded:
        st.sidebar.success("✅ Model Loaded Successfully")
        st.sidebar.markdown(f"**Model Type:** {monitor.metadata['model_name']}")
    else:
        st.sidebar.error("❌ Model Not Loaded")
        st.sidebar.markdown("Please ensure model files are available.")
    
    st.sidebar.markdown("---")
    
    # Monitoring mode selection
    mode = st.sidebar.selectbox(
        "📊 Monitoring Mode",
        ["Real-time Simulation", "Manual Fault Injection", "Historical Analysis"]
    )
    
    if mode == "Real-time Simulation":
        show_real_time_monitoring(monitor)
    elif mode == "Manual Fault Injection":
        show_manual_testing(monitor)
    else:
        show_historical_analysis(monitor)

def show_real_time_monitoring(monitor):
    """Real-time monitoring dashboard."""
    st.markdown("## 📈 Real-time Power System Monitoring")
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        auto_run = st.checkbox("🔄 Auto Refresh", value=True)
    with col2:
        refresh_rate = st.selectbox("⏱️ Refresh Rate", [1, 2, 5], index=1)
    with col3:
        noise_level = st.slider("🔊 Noise Level", 0.0, 0.3, 0.1, 0.05)
    with col4:
        fault_probability = st.slider("⚠️ Fault Probability", 0.0, 0.5, 0.1, 0.05)
    
    # Status containers
    status_container = st.container()
    
    # Charts container
    charts_container = st.container()
    
    # Metrics container
    metrics_container = st.container()
    
    if auto_run or st.button("🔄 Refresh Data"):
        # Simulate fault occurrence
        if random.random() < fault_probability:
            fault_types = ["LT_Line_Cut", "Ground_Fault", "Short_Circuit", "Overload"]
            actual_fault = random.choice(fault_types)
        else:
            actual_fault = "Normal"
        
        # Generate waveform
        waveform = monitor.generate_waveform(actual_fault, noise_level)
        
        # Predict fault
        predicted_fault, probabilities = monitor.predict_fault(waveform)
        
        # Calculate metrics
        phase_rms = [np.sqrt(np.mean(waveform[:, i]**2)) for i in range(3)]
        phase_peaks = [np.max(np.abs(waveform[:, i])) for i in range(3)]
        frequency = 50.0  # Assumed
        
        # Display status
        with status_container:
            if predicted_fault != "Normal":
                st.markdown(
                    f'<div class="metric-container fault-detected">'
                    f'<h2>🚨 FAULT DETECTED: <span class="status-fault">{predicted_fault}</span></h2>'
                    f'<p>Confidence: {max(probabilities.values()):.2%}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<h2>✅ System Status: <span class="status-normal">NORMAL</span></h2>'
                    f'<p>All parameters within acceptable range</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        
        # Display charts
        with charts_container:
            col1, col2 = st.columns(2)
            
            with col1:
                # Waveform plot
                fig_wave = go.Figure()
                
                colors = ['red', 'green', 'blue']
                phases = ['Phase A', 'Phase B', 'Phase C']
                
                for i, (phase, color) in enumerate(zip(phases, colors)):
                    fig_wave.add_trace(go.Scatter(
                        x=monitor.t,
                        y=waveform[:, i],
                        mode='lines',
                        name=phase,
                        line=dict(color=color, width=2)
                    ))
                
                fig_wave.update_layout(
                    title="Real-time Current Waveforms",
                    xaxis_title="Time (s)",
                    yaxis_title="Current (A)",
                    height=400,
                    showlegend=True,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_wave, use_container_width=True)
            
            with col2:
                # Probability bar chart
                fig_prob = go.Figure(data=[
                    go.Bar(
                        x=list(probabilities.keys()),
                        y=list(probabilities.values()),
                        marker_color=['red' if k == predicted_fault else 'lightblue' for k in probabilities.keys()]
                    )
                ])
                
                fig_prob.update_layout(
                    title="Fault Classification Probabilities",
                    xaxis_title="Fault Type",
                    yaxis_title="Probability",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
        
        # Display metrics
        with metrics_container:
            st.markdown("### 📊 System Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Phase A RMS", f"{phase_rms[0]:.1f} A", 
                         delta=f"{phase_rms[0] - 100:.1f}")
            with col2:
                st.metric("Phase B RMS", f"{phase_rms[1]:.1f} A", 
                         delta=f"{phase_rms[1] - 100:.1f}")
            with col3:
                st.metric("Phase C RMS", f"{phase_rms[2]:.1f} A", 
                         delta=f"{phase_rms[2] - 100:.1f}")
            with col4:
                st.metric("Frequency", f"{frequency:.1f} Hz", 
                         delta=f"{frequency - 50:.1f}")
            with col5:
                imbalance = max(phase_rms) - min(phase_rms)
                st.metric("Phase Imbalance", f"{imbalance:.1f} A", 
                         delta=None if imbalance < 5 else f"+{imbalance:.1f}")
        
        # Auto refresh
        if auto_run:
            time.sleep(refresh_rate)
            st.rerun()

def show_manual_testing(monitor):
    """Manual fault injection interface."""
    st.markdown("## 🧪 Manual Fault Testing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Test Parameters")
        
        fault_type = st.selectbox(
            "Fault Type",
            ["Normal", "LT_Line_Cut", "Ground_Fault", "Short_Circuit", "Overload"]
        )
        
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1, 0.05)
        
        if st.button("🎯 Generate Test Case", type="primary"):
            st.session_state.test_waveform = monitor.generate_waveform(fault_type, noise_level)
            st.session_state.test_fault_type = fault_type
    
    with col2:
        if 'test_waveform' in st.session_state:
            waveform = st.session_state.test_waveform
            actual_fault = st.session_state.test_fault_type
            
            # Predict fault
            predicted_fault, probabilities = monitor.predict_fault(waveform)
            
            # Show results
            st.markdown("### Test Results")
            
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.markdown(f"**Actual Fault:** {actual_fault}")
                st.markdown(f"**Predicted Fault:** {predicted_fault}")
                
                if actual_fault == predicted_fault:
                    st.success("✅ Correct Classification")
                else:
                    st.error("❌ Incorrect Classification")
            
            with col2b:
                st.markdown("**Classification Confidence:**")
                for fault, prob in probabilities.items():
                    confidence = f"{prob:.2%}"
                    if fault == predicted_fault:
                        st.markdown(f"**{fault}:** {confidence} ⭐")
                    else:
                        st.markdown(f"{fault}: {confidence}")
            
            # Plot waveform
            fig = go.Figure()
            colors = ['red', 'green', 'blue']
            phases = ['Phase A', 'Phase B', 'Phase C']
            
            for i, (phase, color) in enumerate(zip(phases, colors)):
                fig.add_trace(go.Scatter(
                    x=monitor.t,
                    y=waveform[:, i],
                    mode='lines',
                    name=phase,
                    line=dict(color=color, width=2)
                ))
            
            fig.update_layout(
                title=f"Test Waveform: {actual_fault}",
                xaxis_title="Time (s)",
                yaxis_title="Current (A)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_historical_analysis(monitor):
    """Historical data analysis."""
    st.markdown("## 📊 Historical Analysis")
    
    if monitor.model_loaded:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Historical Data Settings")
        num_days = st.sidebar.slider("Days of History", 1, 60, 7)
        fault_occurrence_rate = st.sidebar.slider("Fault Rate (per hour)", 0.0, 0.5, 0.05, 0.01)
        
        if st.button("📈 Generate Sample Historical Data"):
            
            progress_text = st.empty()
            progress_bar = st.progress(0)

            dates = pd.date_range(end=datetime.now(), periods=num_days * 24, freq='H') # Hourly data points
            
            historical_data = []
            fault_types = ["Normal", "LT_Line_Cut", "Ground_Fault", "Short_Circuit", "Overload"]

            for i, timestamp in enumerate(dates):
                
                # Simulate fault occurrence for historical data
                current_fault_type = "Normal"
                if random.random() < fault_occurrence_rate:
                    current_fault_type = random.choice([f for f in fault_types if f != "Normal"])
                
                # Generate waveform for this historical point
                waveform = monitor.generate_waveform(current_fault_type, noise_level=random.uniform(0.05, 0.2))
                
                # Predict fault using the trained model
                predicted_fault, probabilities = monitor.predict_fault(waveform)

                # Store key metrics and predictions
                rms_a = np.sqrt(np.mean(waveform[:, 0]**2))
                rms_b = np.sqrt(np.mean(waveform[:, 1]**2))
                rms_c = np.sqrt(np.mean(waveform[:, 2]**2))
                
                historical_data.append({
                    'timestamp': timestamp,
                    'actual_fault': current_fault_type,
                    'predicted_fault': predicted_fault,
                    'rms_a': rms_a,
                    'rms_b': rms_b,
                    'rms_c': rms_c,
                    'confidence': max(probabilities.values()) if probabilities else 0,
                    'waveform_data': waveform # Store raw waveform for detailed view if needed
                })
                progress_bar.progress((i + 1) / len(dates))
                progress_text.text(f"Generating data for {i+1}/{len(dates)} hours...")

            st.session_state.historical_df = pd.DataFrame(historical_data)
            progress_text.text("Historical data generated!")
            progress_bar.empty() # Clear the progress bar after completion

        if 'historical_df' in st.session_state:
            df_hist = st.session_state.historical_df
            
            st.markdown("### Overview of Historical Faults")
            
            # Fault distribution
            fault_counts = df_hist['predicted_fault'].value_counts().reset_index()
            fault_counts.columns = ['Fault Type', 'Count']
            
            fig_pie = px.pie(fault_counts, values='Count', names='Fault Type', 
                             title='Distribution of Predicted Fault Types',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("### Timeline of Predicted Faults")
            
            # Timeline of RMS and Faults
            fig_timeline = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                         vertical_spacing=0.1,
                                         subplot_titles=("Phase RMS Over Time", "Predicted Fault Events"))
            
            fig_timeline.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['rms_a'], mode='lines', name='RMS Phase A', line=dict(color='red', width=1)), row=1, col=1)
            fig_timeline.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['rms_b'], mode='lines', name='RMS Phase B', line=dict(color='green', width=1)), row=1, col=1)
            fig_timeline.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['rms_c'], mode='lines', name='RMS Phase C', line=dict(color='blue', width=1)), row=1, col=1)
            fig_timeline.update_yaxes(title_text="RMS Current (A)", row=1, col=1)

            # Mark fault events on the timeline
            fault_events_df = df_hist[df_hist['predicted_fault'] != 'Normal']
            if not fault_events_df.empty:
                fig_timeline.add_trace(go.Scatter(
                    x=fault_events_df['timestamp'],
                    y=[1]*len(fault_events_df), # Dummy y-value for marker placement
                    mode='markers',
                    name='Fault Event',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    text=fault_events_df['predicted_fault'] + ' (' + (fault_events_df['confidence']*100).round(2).astype(str) + '%)',
                    hoverinfo='text'
                ), row=2, col=1)
            
            fig_timeline.update_yaxes(title_text="Fault Event", showticklabels=False, row=2, col=1)
            fig_timeline.update_xaxes(title_text="Time", row=2, col=1)
            fig_timeline.update_layout(height=600, title_text="Historical RMS and Fault Events", hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_timeline, use_container_width=True)

            st.markdown("### Detailed Historical Records")
            st.dataframe(df_hist[['timestamp', 'actual_fault', 'predicted_fault', 'confidence', 'rms_a', 'rms_b', 'rms_c']].set_index('timestamp').sort_index(ascending=False))
            
            # Optional: Allow user to select a specific fault event for waveform visualization
            st.markdown("### Inspect Specific Fault Waveform")
            fault_dates = df_hist[df_hist['predicted_fault'] != 'Normal']['timestamp'].tolist()
            if fault_dates:
                selected_fault_date = st.selectbox("Select a Fault Event to Inspect", sorted(fault_dates, reverse=True))
                
                if selected_fault_date:
                    fault_record = df_hist[df_hist['timestamp'] == selected_fault_date].iloc[0]
                    waveform = fault_record['waveform_data']
                    actual_fault = fault_record['actual_fault']
                    predicted_fault = fault_record['predicted_fault']

                    fig_detail = go.Figure()
                    colors = ['red', 'green', 'blue']
                    phases = ['Phase A', 'Phase B', 'Phase C']
                    
                    for i, (phase, color) in enumerate(zip(phases, colors)):
                        fig_detail.add_trace(go.Scatter(
                            x=monitor.t,
                            y=waveform[:, i],
                            mode='lines',
                            name=phase,
                            line=dict(color=color, width=2)
                        ))
                    
                    fig_detail.update_layout(
                        title=f"Waveform for {selected_fault_date} (Predicted: {predicted_fault}, Actual: {actual_fault})",
                        xaxis_title="Time (s)",
                        yaxis_title="Current (A)",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_detail, use_container_width=True)
            else:
                st.info("No fault events found in the generated historical data to inspect.")

    else:
        st.warning("Model not loaded. Please train the model first to use historical analysis.")


if __name__ == "__main__":
    main()