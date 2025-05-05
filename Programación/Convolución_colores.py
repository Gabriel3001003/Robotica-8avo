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

    # Aplicar filtros
    suavizado = cv2.GaussianBlur(image, (5, 5), 0)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    prewitt_x = cv2.filter2D(image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt = prewitt_x + prewitt_y  # Suma de Prewitt en X y Y
    laplace = cv2.Laplacian(image, cv2.CV_64F)
    
    # Filtro High-Pass (detección de bordes)
    kernel_highpass = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    high_pass = cv2.filter2D(image, -1, kernel_highpass)

    median = cv2.medianBlur(image, 5)
    min_filter = cv2.erode(image, np.ones((3,3), np.uint8))
    max_filter = cv2.dilate(image, np.ones((3,3), np.uint8))
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)

    # Nueva opción: Imagen Original - Prewitt
    original_menos_prewitt = cv2.subtract(image, prewitt)

    # Aplicar CLAHE (Equalización adaptativa de histograma con limitación de contraste)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(np.uint8(image))  # Aplicar CLAHE a la imagen (si es en escala de grises)

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
    binarized_images = [binarize_image(img, grayscale) for img in [image, suavizado, sobel_x, sobel_y, prewitt_x, prewitt_y, prewitt, laplace, high_pass, median, min_filter, max_filter, bilateral, original_menos_prewitt, clahe_image]]

    # Lista de imágenes y títulos
    images = [image, suavizado, sobel_x, sobel_y, prewitt_x, prewitt_y, prewitt, laplace, high_pass,
              median, min_filter, max_filter, bilateral, original_menos_prewitt, clahe_image]
    
    titles = ["Original", "Suavizado", "Sobel X", "Sobel Y", "Prewitt X", "Prewitt Y", "Prewitt Total",
              "Laplace", "High Pass", "Mediana", "Filtro Min", "Filtro Max", "Bilateral", "Original - Prewitt", "CLAHE"]

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
image_path = "Tripulacion.jpg"

# Llamar a la función con opción de escala de grises
apply_filters(image_path, grayscale=True)  # Cambia a False para ver en color