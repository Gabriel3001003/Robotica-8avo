import cv2
import numpy as np
import serial
import time

# Abrir el puerto serial (ajusta si es necesario)
ser = serial.Serial('COM9', 115200)
time.sleep(2)

# Cargar imagen en blanco y negro
image = cv2.imread("cc.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Encontrar contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Escalado: cada píxel equivale a 0.07 cm
escala_cm = 0.07

# Desplazamiento para centrar (ajustar al rango del brazo)
offset_x = 5
offset_y = 5

# Recorrer cada punto de cada contorno (sin simplificar)
for cnt in contours:
    for point in cnt:
        x_px, y_px = point[0]
        x_cm = x_px * escala_cm + offset_x
        y_cm = y_px * escala_cm + offset_y

        # Limitar el rango al área de trabajo
        if 0 < x_cm < 20 and 0 < y_cm < 20:
            comando = f"{x_cm:.2f},{y_cm:.2f}\n"
            print("Enviando:", comando.strip())
            ser.write(comando.encode())
            time.sleep(0.03)

ser.close()
print("Dibujo terminado.")
