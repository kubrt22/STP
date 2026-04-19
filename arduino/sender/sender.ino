#include <esp_now.h>
#include <WiFi.h>

#define BUTTON_PIN 4  // change if needed

uint8_t receiverMAC[] = {0x68, 0xfe, 0x71, 0x8b, 0x3c, 0xd0}; // <-- replace with your MAC

typedef struct struct_message {
  bool buttonPressed;
} struct_message;

struct_message message;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }
}

void loop() {
  bool currentState = digitalRead(BUTTON_PIN) == LOW;

  message.buttonPressed = currentState;

  esp_now_send(receiverMAC, (uint8_t *)&message, sizeof(message));

  delay(50); // small delay for stability
}