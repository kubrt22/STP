#include <ESP32Servo.h>

#define EMG_PIN    15
#define SERVO_PIN  18

Servo myServo;

double mean = 0;
double inputs = 0;
double y = 0;
double last50[50];
int i = 0;

void setup()
{
  Serial.begin(115200);

  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db); 

  // Servo
  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(0);

  for (i = 0; i < 50; i++) {
    last50[i] = 0;
  }
}

void loop()
{
  int raw = analogRead(EMG_PIN);

  inputs++;

  mean = (mean * (inputs - 1) + raw) / inputs;

  double rectified = abs(raw - mean);

  if (inputs - 1 < 50)
  {
    last50[(int)inputs - 1] = rectified;
  }
  else
  {
    for (i = 0; i < 49; i++)
    {
      last50[i] = last50[i + 1];
    }
    last50[49] = rectified;
  }

  y = 0;
  for (i = 0; i < 50; i++)
  {
    y += last50[i];
  }

  y = y * 4 / 12000;

  int servoAngle;

  if (y < 1)
    servoAngle = 0;
  else if (y < 1.75)
    servoAngle = 45;
  else if (y < 2.25)
    servoAngle = 90;
  else if (y < 3)
    servoAngle = 135;
  else
    servoAngle = 180;

  myServo.write(servoAngle);

  Serial.print(y);
  Serial.print(" ");
  Serial.println(servoAngle);

  delay(5);
}
