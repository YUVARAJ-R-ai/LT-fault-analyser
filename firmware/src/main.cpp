#include <Adafruit_ADS1X15.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino.h>
#include <Wire.h>

// ---- Hardware ----
Adafruit_ADS1115 ads;
Adafruit_SSD1306 display(128, 64, &Wire, -1);

// ---- Calibration ----
// SCT-013-000: 100A / 50mA, Turns Ratio = 2000:1
// Burden Resistor: 30 Ohms (3 x 10 Ohm in series)
const double ICAL = 66.67;
const double ADS_MULTIPLIER = 0.000031250; // GAIN_FOUR

// Reduced samples to avoid WDT reset on ESP8266
const int NUM_SAMPLES = 50;

// ---- Waveform Buffer for OLED ----
const int WAVE_WIDTH = 128;
int waveform[WAVE_WIDTH];
int waveIdx = 0;

void drawOLED(double current_rms, int16_t peak_raw) {
  display.clearDisplay();

  // Big current value
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print(current_rms, 3);
  display.println(" A");

  // Raw value
  display.setTextSize(1);
  display.setCursor(0, 20);
  display.print("RAW:");
  display.print(peak_raw);

  // Waveform graph (y=32 to y=63)
  int graphTop = 32;
  int graphH = 31;

  display.drawLine(0, graphTop + graphH / 2, 127, graphTop + graphH / 2,
                   SSD1306_WHITE);

  for (int i = 0; i < WAVE_WIDTH - 1; i++) {
    int idx1 = (waveIdx + i) % WAVE_WIDTH;
    int idx2 = (waveIdx + i + 1) % WAVE_WIDTH;
    int y1 = constrain(graphTop + graphH / 2 - waveform[idx1], graphTop,
                       graphTop + graphH);
    int y2 = constrain(graphTop + graphH / 2 - waveform[idx2], graphTop,
                       graphTop + graphH);
    display.drawLine(i, y1, i + 1, y2, SSD1306_WHITE);
  }

  display.display();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!ads.begin()) {
    Serial.println("# ERROR: ADS1115 not found!");
    while (1) {
      delay(1000);
    }
  }
  ads.setGain(GAIN_FOUR);
  Serial.println("# ADS1115 OK");

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("# WARNING: SSD1306 not found");
  } else {
    Serial.println("# OLED OK");
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("LT Fault Detector");
    display.println("v1.0");
    display.display();
    delay(1000);
  }

  memset(waveform, 0, sizeof(waveform));
  Serial.println("# READY");
  Serial.println("peak_raw,current_rms");
}

void loop() {
  // Two-pass RMS: First collect samples, then subtract DC offset

  int16_t samples[NUM_SAMPLES];
  int16_t peak_raw = 0;

  // Pass 1: Collect all samples and calculate mean (DC bias)
  double sum = 0.0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    samples[i] = ads.readADC_SingleEnded(0);
    sum += samples[i];
    if (abs(samples[i]) > abs(peak_raw))
      peak_raw = samples[i];
    yield();
  }
  double dc_offset = sum / NUM_SAMPLES;

  // Pass 2: Subtract DC offset, then calculate RMS of AC component only
  double sum_sq = 0.0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    double ac = (double)samples[i] - dc_offset; // Remove DC bias
    sum_sq += ac * ac;

    // Store for waveform display
    if (i % (NUM_SAMPLES / 4) == 0) {
      int scaled = (int)(ac / 50.0); // Scale for display
      waveform[waveIdx] = constrain(scaled, -15, 15);
      waveIdx = (waveIdx + 1) % WAVE_WIDTH;
    }
  }

  // Calculate true AC RMS current
  double rms_raw = sqrt(sum_sq / NUM_SAMPLES);
  double voltage_rms = rms_raw * ADS_MULTIPLIER;
  double current_rms = voltage_rms * ICAL;

  // Noise floor: readings below 0.01A are just ADC noise
  if (current_rms < 0.01)
    current_rms = 0.0;

  // Serial output
  Serial.print(peak_raw);
  Serial.print(",");
  Serial.print(current_rms, 4);
  Serial.print(",");
  Serial.println((int)dc_offset); // Show DC offset for debugging

  // OLED output
  drawOLED(current_rms, peak_raw);

  yield();
}
