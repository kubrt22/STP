/*
  ESP32 – 3x AD8232 EMG Data Stream
  Sends: emg1,emg2,emg3 over Serial
  Baud rate: 115200
*/

#define EMG1_PIN 34
#define EMG2_PIN 35
#define EMG3_PIN 32

// Sampling rate control
const int SAMPLE_DELAY_MS = 5; // ~200 Hz

void setup() {
  Serial.begin(115200);
  delay(2000); // give Python time to connect

  // Configure ADC
  analogReadResolution(12);          // 0–4095
  analogSetAttenuation(ADC_11db);    // Full range (~3.3V)

  pinMode(EMG1_PIN, INPUT);
  pinMode(EMG2_PIN, INPUT);
  pinMode(EMG3_PIN, INPUT);

  Serial.println("ESP32 EMG READY");
}

void loop() {
  int emg1 = analogRead(EMG1_PIN);
  int emg2 = analogRead(EMG2_PIN);
  int emg3 = analogRead(EMG3_PIN);

  // Send comma-separated values
  Serial.print(emg1);
  Serial.print(",");
  Serial.print(emg2);
  Serial.print(",");
  Serial.println(emg3);

  delay(SAMPLE_DELAY_MS);
}
