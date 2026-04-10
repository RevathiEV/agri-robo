#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "Rekha";
const char* password = "12345678";

WebServer server(80);

// ESP32 -> L298N mapping
const int IN1 = 25;
const int IN2 = 26;
const int IN3 = 27;
const int IN4 = 14;
const int ENA = 33;
const int ENB = 32;

String currentDirection = "stop";
unsigned long lastCommandAt = 0;
const unsigned long COMMAND_TIMEOUT_MS = 700;

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  currentDirection = "stop";
}

void moveFront() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  currentDirection = "front";
}

void moveBack() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  currentDirection = "back";
}

void turnLeft() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  currentDirection = "left";
}

void turnRight() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  currentDirection = "right";
}

void sendJson(int statusCode, String body) {
  server.send(statusCode, "application/json", body);
}

void handleHealth() {
  String body = "{\"status\":\"ok\",\"ip\":\"" + WiFi.localIP().toString() +
                "\",\"direction\":\"" + currentDirection + "\"}";
  sendJson(200, body);
}

void handleMove() {
  if (!server.hasArg("direction")) {
    sendJson(400, "{\"success\":false,\"message\":\"Missing direction\"}");
    return;
  }

  String direction = server.arg("direction");

  if (direction == "front") {
    moveFront();
  } else if (direction == "back") {
    moveBack();
  } else if (direction == "left") {
    turnLeft();
  } else if (direction == "right") {
    turnRight();
  } else if (direction == "stop") {
    stopMotors();
  } else {
    sendJson(400, "{\"success\":false,\"message\":\"Invalid direction\"}");
    return;
  }

  lastCommandAt = millis();

  String body = "{\"success\":true,\"direction\":\"" + currentDirection + "\"}";
  sendJson(200, body);
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  digitalWrite(ENA, HIGH);
  digitalWrite(ENB, HIGH);
  stopMotors();

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.on("/health", HTTP_GET, handleHealth);
  server.on("/move", HTTP_GET, handleMove);
  server.on("/move", HTTP_POST, handleMove);
  server.begin();
}

void loop() {
  server.handleClient();

  if (currentDirection != "stop" &&
      millis() - lastCommandAt > COMMAND_TIMEOUT_MS) {
    stopMotors();
  }
}
