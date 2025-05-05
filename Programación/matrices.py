import numpy as np  #NumPy para trabajar con matrices

# Función para ingresar una matriz
def ingresar_matriz(filas, columnas):
    matriz = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            valor = float(input(f"Ingrese el valor de la posición ({i+1},{j+1}): "))
            fila.append(valor)
        matriz.append(fila)
    return np.array(matriz)  # Convertimos la lista en una matriz de NumPy

# Pedimos dimensiones de ambas matrices
filas_A = int(input("Ingrese el número de filas de la primera matriz: "))
columnas_A = int(input("Ingrese el número de columnas de la primera matriz: "))

filas_B = int(input("Ingrese el número de filas de la segunda matriz: "))
columnas_B = int(input("Ingrese el número de columnas de la segunda matriz: "))

# Ingresamos los valores de las matrices
print("\nIngrese los valores de la primera matriz:")
A = ingresar_matriz(filas_A, columnas_A)

print("\nIngrese los valores de la segunda matriz:")
B = ingresar_matriz(filas_B, columnas_B)

# Verificamos si las matrices tienen el mismo tamaño para poder sumarlas/restarlas
if A.shape == B.shape:
    suma = A + B
    resta = A - B
else:
    suma = "No se puede sumar/restar (dimensiones diferentes)"
    resta = "No se puede sumar/restar (dimensiones diferentes)"

# Verificamos si las matrices se pueden multiplicar 
if A.shape[1] == B.shape[0]:
    multiplicacion = np.dot(A, B)
else:
    multiplicacion = "No se puede multiplicar (columnas de A ≠ filas de B)"

# Mostramos resultados
print("\nMatriz A:\n", A)
print("Matriz B:\n", B)
print("\nSuma:\n", suma)
print("\nResta:\n", resta)
print("\nMultiplicación:\n", multiplicacion)
