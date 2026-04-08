# Testing vs. Deployment: Understanding the Setup

## The Confusion
You are testing in a **Lab / Home** environment with a standard wall socket.
*   **Standard Wall Socket**: Single Phase (220V) + Neutral.
*   **Your Project Goal**: Monitor a 3-Phase Distribution Transformer.

## How to Test "3-Phase Logic" with 1 Bulb?

You cannot create 3 discrete phases from a single socket easily. However, you can **Simulate** the conditions to test your code.

### 1. The Lab Setup (What you build NOW)
You will use **ONE bulb** and **ONE CT Sensor**.
*   **Physical**: Connect CT to `ADS1115 A0`.
*   **Software Simulation**:
    *   In your code, you will *pretend* that `Input A1` and `Input A2` are also reading data.
    *   **Normal High Load**: You turn ON the bulb. Code reads High Amps on A0. You *copy* this value to A1 and A2 in software to simulate a "Balanced 3-Phase Load".
    *   **Line Break Simulation**: You disconnect the bulb. Current drops to 0. Code sees A0=0, but you keep A1, A2 high (in software). Use this to test if your ML model detects "Unbalance".

### 2. The Field Setup (What goes on the Pole)
When you install this for real:
*   The Transformer has **4 wires output** (R, Y, B, Neutral).
*   You clamp **CT1** around R-Phase wire.
*   You clamp **CT2** around Y-Phase wire.
*   You clamp **CT3** around B-Phase wire.
*   **Result**: The ESP8266 receives 3 *independent* real-world signals.

### Summary
*   **For the Prototype**: Build the hardware with 1 CT sensor. Use it to train the model to recognize "Current ON" vs "Current OFF".
*   **For the Code**: Write the code to handle 3 inputs, but for testing, just feed the same sensor data into all 3 generic slots (or physically move the sensor to test different ports).

Does this clear up the doubt? You are physically limited to 1 phase in your room, but the *hardware design* supports 3.
