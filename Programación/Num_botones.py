import cv2
import numpy as np

# Cargar la imagen
image_path = "Botones.jpg"  # Cambia esto por la ruta de tu imagen
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un desenfoque para reducir ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detección de bordes con Canny
edges = cv2.Canny(blurred, 50, 150)

# Encontrar contornos
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar los contornos por área para contar solo los botones
button_contours = [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 1000]

# Contar los botones detectados
num_buttons = len(button_contours)
print(f"Número de botones detectados: {num_buttons}")

# Dibujar los contornos detectados en la imagen original
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, button_contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos
title = "Botones Detectados"
cv2.imshow(title, image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opcional: Guardar la imagen procesada
cv2.imwrite("Botones_contados.jpg", image_with_contours)
