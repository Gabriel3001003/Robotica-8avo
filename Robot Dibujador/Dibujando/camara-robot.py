import cv2
import numpy as np
import serial
import time

# Parámetros de configuración
OFFSET_X = 5 # Centro del área de trabajo (en cm)
OFFSET_Y = 5
ESCALA_REAL_CM = 10 # Tamaño real del área (10x10 cm)
SERIAL_PORT = 'COM9'  
BAUD_RATE = 115200

# 1. Inicializar cámara
cap = cv2.VideoCapture(1)  
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona 'c' para capturar la imagen...")

imagen = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar imagen.")
        break

    cv2.imshow("Vista en Vivo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        imagen = frame.copy()
        print("Imagen capturada.")
        break

cap.release()
cv2.destroyAllWindows()

if imagen is None:
    print("No se capturó ninguna imagen.")
    exit()

# 2. Procesar imagen
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Encontrar contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if not contours:
    print("No se detectaron contornos.")
    exit()

# 4. Elegir el contorno más grande
contour = max(contours, key=cv2.contourArea)
epsilon = 0.01 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

# 5. Centrar y escalar figura
x_min, y_min, w, h = cv2.boundingRect(approx)
cx = x_min + w / 2
cy = y_min + h / 2
ESCALA_CM = ESCALA_REAL_CM / max(w, h)

puntos_cm = []
for pt in approx:
    x_px, y_px = pt[0]
    x_cm = (x_px - cx) * ESCALA_CM + OFFSET_X
    y_cm = (y_px - cy) * ESCALA_CM + OFFSET_Y
    y_cm = ESCALA_REAL_CM - y_cm  # Invertir eje Y

    if 0 <= x_cm <= 10 and 0 <= y_cm <= 10:
        puntos_cm.append((x_cm, y_cm))

if len(puntos_cm) < 2:
    print("No hay suficientes puntos válidos para dibujar.")
    exit()

# 6. Mostrar la figura escalada
canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255
for i in range(1, len(puntos_cm)):
    x1, y1 = puntos_cm[i - 1]
    x2, y2 = puntos_cm[i]
    pt1 = (int(x1 * 40), int(y1 * 40))
    pt2 = (int(x2 * 40), int(y2 * 40))
    cv2.line(canvas, pt1, pt2, (0, 0, 0), 2)

cv2.imshow("Figura a Dibujar (10x10 cm)", canvas)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# 7. Enviar coordenadas por serial
try:
    print("Conectando al puerto serial...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)

    for x, y in puntos_cm:
        comando = f"{x:.2f},{y:.2f}\n"
        print("Enviando:", comando.strip())
        ser.write(comando.encode())
        time.sleep(0.05)

    ser.write(b"END\n")
    print("Dibujo enviado correctamente.")
    ser.close()
except Exception as e:
    print(f"Error al enviar por serial: {e}")
