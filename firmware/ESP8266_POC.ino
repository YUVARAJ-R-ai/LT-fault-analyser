#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>
#include "model.h"  // The C++ model we exported using micromlgen

#define CT_PIN A0        // Analog input pin for CT sensor
#define SAMPLES 100      // Number of samples per reading
#define SAMPLE_INTERVAL 10 // Microseconds between samples

// Initialize OLED (Change if you are using a different OLED type)
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

Eloquent::ML::Port::DecisionTree clf;

void setup() {
    Serial.begin(115200);
    u8g2.begin();
    pinMode(CT_PIN, INPUT);
    
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(0, 10, "1-Phase Monitor");
    u8g2.sendBuffer();
    delay(2000);
}

void loop() {
    float sum_sq = 0;
    float sum = 0;
    float max_val = 0;
    float values[SAMPLES];
    
    // 1. Gather Data array
    for (int i = 0; i < SAMPLES; i++) {
        int raw = analogRead(CT_PIN);
        // Voltage translation for ESP8266 ADC (0-1.0V typical, though NodeMCU scales to 3.3V)
        // Assume CT sensor is biased at mid-point. Adjust math for your specific CT circuit!
        float current = ((raw / 1023.0) * 3.3 - 1.65) * 30.0; // Example conversion, TUNE THIS
        
        values[i] = current;
        sum += current;
        sum_sq += (current * current);
        if (abs(current) > max_val) max_val = abs(current);
        
        delayMicroseconds(SAMPLE_INTERVAL);
    }
    
    // 2. Compute the 4 Features our model expects: [RMS, Peak, Mean, Variance]
    float mean_val = sum / SAMPLES;
    float rms = sqrt(sum_sq / SAMPLES);
    float peak = max_val;
    
    float variance = 0;
    for (int i = 0; i < SAMPLES; i++) {
        variance += pow(values[i] - mean_val, 2);
    }
    variance /= SAMPLES;
    
    float features[] = {rms, peak, mean_val, variance};
    
    // 3. Run Inference on ESP8266
    const char* predicted_label = clf.predictLabel(features);
    
    // 4. Output to OLED
    u8g2.clearBuffer();
    u8g2.setCursor(0, 15);
    u8g2.print("Status: ");
    u8g2.print(predicted_label);
    
    u8g2.setCursor(0, 35);
    u8g2.print("RMS: "); u8g2.print(rms, 3); u8g2.print(" A");
    
    u8g2.setCursor(0, 55);
    u8g2.print("Peak: "); u8g2.print(peak, 3); u8g2.print(" A");
    
    u8g2.sendBuffer();
    
    // Output to Serial Monitor for debugging
    Serial.print("RMS:"); Serial.print(rms);
    Serial.print(" / Peak:"); Serial.print(peak);
    Serial.print(" => Pred:"); Serial.println(predicted_label);
    
    delay(500); // Small delay before next read loop
}
