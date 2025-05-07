#include <ESP32Servo.h>

Servo lidarServo;

#define TRIG_PIN 4    // GPIO4
#define ECHO_PIN 5    // GPIO5
#define SERVO_PIN 13  // GPIO13 (PWM válido)
#define UMBRAL 15     // Distancia mínima en cm

int angulo = 0;
int direccion = 1;  // 1 = aumentando, -1 = disminuyendo

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  lidarServo.attach(SERVO_PIN);
}

long medirDistancia() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duracion = pulseIn(ECHO_PIN, HIGH, 30000); // Timeout 30 ms
  if (duracion == 0) return 999; // No eco recibido

  long distancia = duracion * 0.034 / 2;
  return distancia;
}

void loop() {
  lidarServo.write(angulo);
  delay(100);  // Espera a que el servo se mueva

  long distancia = medirDistancia();

  Serial.print("Ángulo: ");
  Serial.print(angulo);
  Serial.print("°, Distancia: ");
  Serial.print(distancia);
  Serial.println(" cm");

  if (distancia < UMBRAL) {
    direccion = -direccion;
    Serial.println("⚠ Objeto detectado cerca. Cambiando dirección.");
  }

  angulo += direccion * 2;

  if (angulo >= 180) {
    angulo = 180;
    direccion = -1;
  } else if (angulo <= 0) {
    angulo = 0;
    direccion = 1;
  }

  delay(50);
}
