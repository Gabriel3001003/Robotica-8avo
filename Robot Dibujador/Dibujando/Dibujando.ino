#include <ESP32Servo.h>
#include <math.h>

Servo servo1;  // hombro
Servo servo2;  // codo

const int SERVO1_PIN = 13;
const int SERVO2_PIN = 12;

const float L1 = 13.8;
const float L2 = 13.8;

void setup() {
  Serial.begin(115200);
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);

  // Posición inicial
  servo1.write(90);
  servo2.write(90);
  delay(1000);
  Serial.println("Listo para recibir coordenadas x,y");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int sepIndex = input.indexOf(',');
    if (sepIndex > 0) {
      float x = input.substring(0, sepIndex).toFloat();
      float y = input.substring(sepIndex + 1).toFloat();

      float dist = sqrt(x * x + y * y);
      if (dist > (L1 + L2) || dist < fabs(L1 - L2)) {
        Serial.println("Posición fuera de alcance");
        return;
      }

      float a = acos((x*x + y*y - L1*L1 - L2*L2) / (2 * L1 * L2));
      float b = atan2(y, x) - atan2(L2 * sin(a), L1 + L2 * cos(a));

      int angle1 = degrees(b);
      int angle2 = degrees(a);

      angle1 = constrain(angle1 + 90, 0, 180);
      angle2 = constrain(angle2, 0, 180);

      Serial.printf("Moviendo a ángulos: servo1=%d, servo2=%d\n", angle1, angle2);

      servo1.write(angle1);
      servo2.write(angle2);

      delay(100);  // Tiempo para que los servos se muevan
    } else {
      Serial.println("Formato inválido, usar x,y");
    }
  }
}
