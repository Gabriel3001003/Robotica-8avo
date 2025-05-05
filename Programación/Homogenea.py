import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def homogeneous_matrix(rotation_angle, translation):
    """Crea una matriz homogénea 4x4 con una rotación en Z y una traslación."""
    theta = np.radians(rotation_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    H = np.array([
        [cos_t, -sin_t, 0, translation[0]],
        [sin_t, cos_t,  0, translation[1]],
        [0,     0,      1, translation[2]],
        [0,     0,      0, 1]
    ])
    return H

def transform_points(points, H):
    """Aplica la matriz homogénea a un conjunto de puntos."""
    homogeneous_points = np.vstack((points, np.ones((1, points.shape[1]))))
    transformed_points = H @ homogeneous_points
    return transformed_points[:3]

# Definimos los enlaces como puntos en 3D (cada columna es un punto)
links = np.array([
    [0, 2, 4],  # X
    [0, 1, 3],  # Y
    [0, 0, 0]   # Z
])

# Aplicamos una transformación (rotación de 45° y traslación de (2,2,0))
H = homogeneous_matrix(45, [2, 2, 0])
transformed_links = transform_points(links, H)

# Graficamos los enlaces originales y transformados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Función para etiquetar los puntos en la gráfica
def label_points(ax, points, label_prefix, color):
    for i in range(points.shape[1]):
        x, y, z = points[:, i]
        ax.text(x, y, z, f'{label_prefix}({x:.1f},{y:.1f},{z:.1f})', color=color)

# Enlaces originales
ax.plot(links[0], links[1], links[2], 'bo-', label="Original")
label_points(ax, links, "O", "blue")

# Enlaces transformados
ax.plot(transformed_links[0], transformed_links[1], transformed_links[2], 'ro-', label="Transformado")
label_points(ax, transformed_links, "T", "red")

# Configuración de ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title("Transformación con Matriz Homogénea")

plt.show()

#La imagen que el codigo muestra es una rotación de 45 grados, los valores están etiquetados en la misma imagen.
#La línea azul representa los enlaces antes de la transformación.
#La línea roja es el mismo conjunto de puntos después de la transformación homogénea
