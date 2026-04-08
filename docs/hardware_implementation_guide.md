# Hardware Implementation Guide: Transformer-Mounted LT Fault Detection System

## 1. System Overview

This document outlines the hardware architecture and implementation strategy for a **Transformer-Mounted Intelligent LT Fault Detection System**. The system is designed to detect Low-Tension (LT) line breaks (open conductors) and other faults (Short Circuit, Overload) from the distribution transformer side, eliminating the need for expensive distributed sensors.

### Key Features
*   **Centralized Detection**: Single unit per transformer protects the entire feeder.
*   **Non-Invasive Sensing**: Uses Split-Core Current Transformers (SCT-013) clamped around the output cables.
*   **Machine Learning Core**: Uses an **ESP8266 (NodeMCU/Wemos)** microcontroller to run a lightweight ML model.
*   **GSM Alerting**: Instant SMS notification to utility engineers upon fault detection.
*   **External ADC Requirement**: Since ESP8266 has only 1 analog pin, an **ADS1115** is required to read 3-phase currents.

---

## 2. Hardware Architecture

### 2.1 Core Components
| Component | Function | Specification | Cost Est. (INR) |
| :--- | :--- | :--- | :--- |
| **ESP8266 (NodeMCU)** | Main Processor & ML Inference | 80MHz, Wi-Fi | ₹300 |
| **ADS1115 (REQUIRED)** | 4-Channel ADC | 16-bit resolution, I2C interface | ₹150 |
| **SCT-013-000** (x3) | Current Sensors | 100A/50mA (Split Core) | ₹350 each |
| **SIM800L / A7670C** | GSM Module | 2G/4G Connectivity | ₹300 - ₹1200 |
| **Relay Module** | Trip Actuation | 5V, 1-Channel, Opto-isolated | ₹80 |
| **Power Supply** | Main Power | 5V 2A Adapter / Buck Converter | ₹150 |

### 2.2 Circuit Diagram Concept

#### A. Current Sensing Layer (3-Phase)
**Crucial Note**: The ESP8266 cannot read 3 sensors directly. We connect them to the **ADS1115**.

**Signal Path (Per Phase):**
1.  **SCT-013 Output**: Connects to a **Burden Resistor** (e.g., $33\Omega$) -> Voltage.
2.  **DC Bias Circuit**:
    *   Since ADS1115 can read differential or single-ended, we have two options.
    *   *Option A (Simpler)*: Use the same 1.65V bias circuit as before. Connect biased signal to ADS1115 A0, A1, A2.
    *   *Option B (Precision)*: Connect CT wires differentially to A0-A1 (Phase A) - *Requires multiple ADCs or multiplexing*.
    *   *Decision*: We will use **Option A** (Single-ended with Bias) to measure 3 phases with 1 ADS1115 module (leaving A3 for spare/neutral).
3.  **Interface**: Connect ADS1115 via I2C (D1/SCL, D2/SDA) to ESP8266.

#### B. GSM & Relay Interface
*   **GSM**: Use SoftwareSerial.
    *   RX (GSM) -> D5 (GPIO 14)
    *   TX (GSM) -> D6 (GPIO 12)
*   **Relay**: D7 (GPIO 13).
*   **I2C (ADS1115)**: D1 (SCL), D2 (SDA).

---

## 3. Software & Machine Learning Implementation

### 3.1 The Challenge: Line Break vs. No Load
The critical challenge is distinguishing between:
1.  **Line Break**: One phase drops to 0A (Fault).
2.  **No Load**: Consumers switch off appliances, current drops near 0A.

### 3.2 Detection Strategy (ML Features)
Instead of simple thresholds, we extract **Features** from the raw waveform (10-20ms window):

1.  **RMS Current ($I_{rms}$)**: Standard magnitude.
2.  **Zero Sequence Current ($I_0$)**:
    *   $I_0 = I_a + I_b + I_c$ (Vector Sum).
    *   In a balanced system (even with no load), $I_0 \approx 0$.
    *   In a **Line Break**, $I_0$ spikes significantly because the vector balance is destroyed.
3.  **Phase Imbalance Ratio**:
    *   $Ratio = \frac{|I_{max} - I_{min}|}{I_{avg}}$
    *   A line break causes this ratio to maximize (close to 1 or infinity).
4.  **Crest Factor**: Peak / RMS. Helps detect arcing or intermittent contact.

### 3.3 Implementation Steps

#### Phase 1: Data Collection (Calibration)
Before training, we must capture *real* signatures from the specific hardware.
*   **Action**: Write simple firmware to print raw ADC values to Serial Monitor.
*   **Experiment**:
    *   Record 1 minute of "Normal" load (balanced/unbalanced).
    *   Record "Simulated Faults" (safely disconnect one phase sensing wire to simulate a cut).
    *   Save this as `training_data.csv`.

#### Phase 2: Model Training (Python)
*   **Input**: `training_data.csv`
*   **Algorithm**: **Decision Tree** or **Random Forest** (max depth 5).
    *   *Why?* Low computational cost, easily verifiable logic.
*   **Tools**: `scikit-learn` for training, `micromlgen` to port to C code.

#### Phase 3: Embedded Deployment
*   **Firmware Loop**:
    1.  Read ADC (collect 1 cycle buffer).
    2.  Compute Features ($I_{rms}, I_0, Imbalance$).
    3.  Run `predict()` (Generated C function).
    4.  Debounce: If `FAULT` is predicted for 5 consecutive cycles -> **TRIP**.

---

## 4. Reliability & Safety Enhancements

To address reliability concerns:

1.  **Software Debouncing**: Do not trip on a single glitch. Require the fault to persist for ~100ms (5 cycles).
2.  **Hard Limits (Safety Net)**:
    *   If $I_{rms} > 200\%$, TRIP immediately (Short Circuit fallback).
    *   Bypasses ML for catastrophic faults.
3.  **Keep-Alive Heartbeat**: GSM module sends a "System OK" SMS once a day to confirm the sensor is alive.

## 5. Cost Analysis (Per Unit)

| Item | Estimated Cost |
| :--- | :--- |
| Electronics (ESP8266 + ADS1115 + Sensors) | ₹1,550 |
| Enclosure & PCB | ₹500 |
| Power Supply | ₹200 |
| **Total BOM Cost** | **₹2,250** |

*Comparable market solutions (e.g., auto-reclosers) cost ₹50,000+.*

---

## 6. Next Steps for Implementation

1.  **Procure Hardware**: Buy ESP32, SCT-013, and resistors.
2.  **bench Test**: Build the bias circuit on a breadboard.
3.  **Log Data**: Use a customized `logger.py` script to save serial data from the ESP32 to your PC.
4.  **Train**: Run the training script on this real-world data.
