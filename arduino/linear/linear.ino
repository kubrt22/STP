#include <ESP32Servo.h>

#define EMG_PIN    15
#define SERVO_PIN  18

Servo myServo;

const int sampleCount = 20;
int samples[sampleCount];
int sampleIndex = 0;
long sampleSum = 0;

// ladeni
int emgMin = 80;
int emgMax = 1200;

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);

  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(0);

  for (int i = 0; i < sampleCount; i++) {
    samples[i] = 0;
  }
}

void loop() {
  int raw = analogRead(EMG_PIN);

  int emgValue = abs(raw - 2048);

  sampleSum -= samples[sampleIndex];
  samples[sampleIndex] = emgValue;
  sampleSum += emgValue;

  sampleIndex++;
  if (sampleIndex >= sampleCount) sampleIndex = 0;

  int emgAvg = sampleSum / sampleCount;

  int servoAngle = map(emgAvg, emgMin, emgMax, 0, 180);
  servoAngle = constrain(servoAngle, 0, 180);

  myServo.write(servoAngle);

  Serial.print(emgAvg);
  Serial.print(" ");
  Serial.println(servoAngle);

  delay(5); // cca 200 Hz
}
