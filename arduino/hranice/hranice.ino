#define ECG_PIN 15

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);
}

void loop() {
  int ecgValue = analogRead(ECG_PIN);

  Serial.println(ecgValue);

  delayMicroseconds(4000);
}
