#include <esp_now.h>
#include <WiFi.h>
#include <ESP32Servo.h>

#define SERVO_PIN 4

Servo myServo;

typedef struct struct_message {
  bool buttonPressed;
} struct_message;

struct_message incomingMessage;

void onDataRecv(const esp_now_recv_info *info, const uint8_t *data, int len) {
  memcpy(&incomingMessage, data, sizeof(incomingMessage));

  if (incomingMessage.buttonPressed) {
    Serial.println("Button pressed -> Servo 180°");
    myServo.write(180);
  } else {
    Serial.println("Button released -> Servo 0°");
    myServo.write(0);
  }
}

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);

  myServo.setPeriodHertz(50);    // standard 50 Hz servo
  myServo.attach(SERVO_PIN, 500, 2400); // min/max pulse width

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  esp_now_register_recv_cb(onDataRecv);
}

void loop() {}