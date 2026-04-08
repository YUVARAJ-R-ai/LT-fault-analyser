"""
Real-time Current Waveform Plotter
Reads from ESP8266 serial and plots RMS current live.
Usage: python software/waveform_plotter.py
"""
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import sys
import glob

# --- Config ---
BAUD_RATE = 115200
WINDOW_SIZE = 100  # Number of points to show on screen at a time

def find_serial_port():
    """Auto-detect the ESP8266 serial port."""
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    if not ports:
        print("ERROR: No USB serial device found.")
        print("  - Is the ESP8266 plugged in?")
        print("  - Did you run: sudo systemctl stop ModemManager.service")
        sys.exit(1)
    print(f"Found device: {ports[0]}")
    return ports[0]

# --- Data buffers ---
time_data = deque(maxlen=WINDOW_SIZE)
current_data = deque(maxlen=WINDOW_SIZE)
raw_data = deque(maxlen=WINDOW_SIZE)
counter = [0]

# --- Setup plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle("LT Line Fault Detection - Live Current Monitor", fontsize=14, fontweight='bold')

line1, = ax1.plot([], [], 'r-', linewidth=1.5, label='RMS Current (A)')
line2, = ax2.plot([], [], 'b-', linewidth=1, label='Raw ADC Peak Value', alpha=0.7)

ax1.set_ylabel("Current (A)")
ax1.set_xlabel("Sample #")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1, 20)  # Adjust based on your load

ax2.set_ylabel("Raw ADC Value")
ax2.set_xlabel("Sample #")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# --- Connect to serial ---
port = find_serial_port()
try:
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    print(f"Connected to {port} at {BAUD_RATE} baud.")
    print("Close the plot window to stop.")
except serial.SerialException as e:
    print(f"ERROR: Could not open port {port}: {e}")
    sys.exit(1)

def animate(frame):
    """Called on each animation frame to update the plot."""
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        # Skip comment lines from firmware
        if line.startswith('#') or not line:
            return line1, line2

        parts = line.split(',')
        if len(parts) == 2:
            raw = int(parts[0])
            current = float(parts[1])

            counter[0] += 1
            time_data.append(counter[0])
            raw_data.append(raw)
            current_data.append(current)

            line1.set_data(list(time_data), list(current_data))
            line2.set_data(list(time_data), list(raw_data))

            ax1.set_xlim(max(0, counter[0] - WINDOW_SIZE), counter[0] + 5)
            ax2.set_xlim(max(0, counter[0] - WINDOW_SIZE), counter[0] + 5)
            
            ax2.set_ylim(min(raw_data) - 100, max(raw_data) + 100)

    except (ValueError, UnicodeDecodeError):
        pass  # Ignore malformed lines during startup

    return line1, line2

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()

ser.close()
print("Serial port closed.")
