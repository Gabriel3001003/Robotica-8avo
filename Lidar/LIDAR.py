import cv2
import torch
import matplotlib.pyplot as plt
import serial
import time
from collections import deque

# Conectar con Arduino
arduino = serial.Serial('COM9', 115200, timeout=1)  # Cambia 'COM3' si es necesario
time.sleep(2)

# Cargar MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Transformaciones de entrada
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Activar modo interactivo
plt.ion()
fig, (ax_depth, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title("MiDaS + Sensor Ultrasónico")

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Buffers para gráficas
distancias = deque(maxlen=50)
angulos = deque(maxlen=50)
tiempos = deque(maxlen=50)
tiempo_actual = 0

# Variables
distancia_cm = None
angulo = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Leer datos desde Arduino
    try:
        while arduino.in_waiting:
            line = arduino.readline().decode('utf-8').strip()
            if "Distancia" in line and "Ángulo" in line:
                print(f"[ARDUINO] {line}")
                partes = line.split(",")
                for p in partes:
                    if "Ángulo" in p:
                        angulo = int(p.split(":")[1].replace("°", "").strip())
                    elif "Distancia" in p:
                        distancia_cm = int(p.split(":")[1].replace("cm", "").strip())
    except Exception as e:
        print(f"Error leyendo serial: {e}")

    # Procesar MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    output_display = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_display = cv2.cvtColor(output_display, cv2.COLOR_GRAY2BGR)

    # Mostrar datos sobre la cámara
    if distancia_cm is not None:
        cv2.putText(frame, f"Distancia: {distancia_cm} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if angulo is not None:
        cv2.putText(frame, f"Ángulo: {angulo}°", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar profundidad con matplotlib
    ax_depth.clear()
    ax_depth.imshow(depth_map)
    ax_depth.set_title("Mapa de Profundidad")
    ax_depth.axis("off")

    # Agregar datos a gráficas
    if distancia_cm is not None and angulo is not None:
        tiempo_actual += 1
        tiempos.append(tiempo_actual)
        distancias.append(distancia_cm)
        angulos.append(angulo)

        # Dibujar gráfica
        ax_plot.clear()
        ax_plot.plot(tiempos, distancias, label="Distancia (cm)", color='green')
        ax_plot.plot(tiempos, angulos, label="Ángulo (°)", color='orange')
        ax_plot.set_title("Sensor Ultrasónico y Servo")
        ax_plot.set_xlabel("Tiempo")
        ax_plot.set_ylabel("Valor")
        ax_plot.legend()
        ax_plot.grid(True)

    plt.draw()
    plt.pause(0.001)

    # Mostrar video
    cv2.imshow('Video', frame)
    cv2.imshow('Mapa de profundidad (OpenCV)', output_display)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Finalizar
cap.release()
arduino.close()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
