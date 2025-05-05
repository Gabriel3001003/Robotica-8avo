#include <Servo.h>

#define TRIG_PIN 9
#define ECHO_PIN 10
#define SERVO_PIN 6
#define UMBRAL 15  // Distancia mínima en cm

Servo lidarServo;
int angulo = 0;
int direccion = 1;  // 1 = aumentando ángulo, -1 = disminuyendo

void setup() {
  Serial.begin(9600);
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

  long duracion = pulseIn(ECHO_PIN, HIGH, 30000);  // timeout de 30 ms
  if (duracion == 0) return 999; // Sin respuesta, retorno alto
  long distancia = duracion * 0.034 / 2;
  return distancia;
}

void loop() {
  lidarServo.write(angulo);              // Mover el servo al ángulo actual
  delay(100);                            // Pequeña pausa para estabilizar el servo
  long distancia = medirDistancia();     // Medir distancia

  Serial.print("Ángulo: ");
  Serial.print(angulo);
  Serial.print("°, Distancia: ");
  Serial.print(distancia);
  Serial.println(" cm");

  // Si un objeto está cerca, invertir dirección
  if (distancia < UMBRAL) {
    direccion = -direccion;
    Serial.println("⚠ Objeto detectado cerca. Cambiando dirección.");
  }

  // Ajustar el ángulo
  angulo += direccion * 2;  // Paso suave de 2 grados

  // Evitar que el ángulo se salga de los límites
  if (angulo >= 180) {
    angulo = 180;
    direccion = -1;
  } else if (angulo <= 0) {
    angulo = 0;
    direccion = 1;
  }

  delay(50); // Delay total por ciclo: 100 + 50 = 150 ms aprox
}
