import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filters(image_path, grayscale=False):
    # Cargar la imagen
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: No se pudo cargar la imagen. Verifica la ruta.")
        return
    
    # Opción para convertir a escala de grises
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(image)  # Asegurarse de que sea uint8 para el histograma
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para Matplotlib

    # Definir el kernel para los filtros morfológicos
    kernel = np.ones((5, 5), np.uint8)

    # Aplicar filtros morfológicos
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    # Nueva opción: Imagen Original - Apertura
    original_menos_opening = cv2.subtract(image, opening)

    # Función para binarizar la imagen usando el máximo valor del histograma
    def binarize_image(image, grayscale):
        # Calcular el histograma de la imagen
        hist_values = cv2.calcHist([np.uint8(image)], [0], None, [256], [0, 256])
        # Obtener el valor máximo del histograma (umbral)
        max_hist_value = np.argmax(hist_values)
        print(f"Valor máximo del histograma (umbral): {max_hist_value}")

        # Aplicar binarización: px < threshold = 0, px >= threshold = 1
        _, binary_image = cv2.threshold(np.uint8(image), max_hist_value, 255, cv2.THRESH_BINARY)

        return binary_image

    # Binarizar las imágenes procesadas
    binarized_images = [binarize_image(img, grayscale) for img in [image, erosion, dilation, opening, closing, gradient, tophat, blackhat, original_menos_opening]]

    # Lista de imágenes y títulos
    images = [image, erosion, dilation, opening, closing, gradient, tophat, blackhat, original_menos_opening]
    
    titles = ["Original", "Erosión", "Dilatación", "Apertura", "Cierre", "Gradiente Morfológico", "Top-Hat", "Black-Hat", "Original - Apertura"]

    # Ventana para mostrar las imágenes procesadas
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(3, 5, i+1)  # 3 filas, 5 columnas
        plt.imshow(images[i], cmap="gray" if grayscale else None)
        plt.title(titles[i])
        plt.axis("off")  # Ocultar los ejes

    plt.tight_layout()
    plt.show()

    # Ventana para mostrar los histogramas
    plt.figure(figsize=(15, 6))
    for i in range(len(images)):
        plt.subplot(3, 5, i+1)  # 3 filas, 5 columnas
        if grayscale:
            hist_values = cv2.calcHist([np.uint8(images[i])], [0], None, [256], [0, 256])  # Asegurarse de que la imagen esté en uint8
            plt.plot(hist_values, color='black')
        else:
            # Histograma por cada canal de color
            colors = ('r', 'g', 'b')
            for j, col in enumerate(colors):
                hist_values = cv2.calcHist([images[i]], [j], None, [256], [0, 256])
                plt.plot(hist_values, color=col)

        plt.title(f"Histograma - {titles[i]}")
        plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

    # Ventana para mostrar las imágenes binarizadas
    plt.figure(figsize=(15, 10))
    for i in range(len(binarized_images)):
        plt.subplot(3, 5, i+1)  # 3 filas, 5 columnas
        plt.imshow(binarized_images[i], cmap="gray")
        plt.title(f"Binarizada - {titles[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Ruta de la imagen (cambia esto con la ruta de tu imagen)
image_path = "Botones.jpg"

# Llamar a la función con opción de escala de grises
apply_filters(image_path, grayscale=True)  # Cambia a False para ver en color
