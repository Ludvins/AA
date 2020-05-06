#!/usr/bin/env python3

########################################################################
############################## IMPORTS #################################
########################################################################

import numpy as np
import matplotlib.pyplot as plt
import math


# Fijamos la semilla
np.random.seed(1)

########################################################################
####################### FUNCIONES AUXILIARES  ##########################
########################################################################

# Declaramos la función uilizada en el ejercicio anterior.
def f(x, y, a, b):
    return 1 if y - a * x - b >= 0 else -1


def simula_recta( intervalo= [-50, 50]):
    """
    Genera dos puntos aleatorios y devuelve los coeficientes de la recta que los une
    """
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]
    if x2 == x1:
        print("Se han generado dos puntos alineados verticalmente -> No se puede construir una recta como funcion que los una. (División por 0)")
    # y = a*x + b
    a = (y2 - y1) / (x2 - x1)  # Calculo de la pendiente.
    b = y1 - a * x1  # Calculo del termino independiente.

    return a, b


def simula_unif(N, dim, rango):
    """
    Devuelve un conjunto de N puntos de R^dim, dentro del rango indicado
    utilizando una districubión uniforme.
    """
    return np.random.uniform(rango[0], rango[1], (N, dim))



def scatter(x, y=None, f=None, label="Function"):
    """
    Funcion scatter, nos permite pintar un conjunto de puntos en un plano 2D.
    Argumentos:
    - x: Conjunto de puntos
    Argumentos opcionales:
    - y: Conjunto de etiquetas de dichos puntos.
    - f: Función a representar
    - label: Etiqueta de la funcion f
    """
    _, ax = plt.subplots()
    ax.set_ylim(1.1 * np.min(x[:, 1]), 1.1 * np.max(x[:, 1]))
    if y is None:
        ax.scatter(x[:, 0], x[:, 1])
    else:
        colors = {-1: "red", 1: "blue"}
        # Para cada clase:
        for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
            # Obten los miembros de la clase
            class_members = x[np.where(y == cls)[0]]

            # Representa en scatter plot
            ax.scatter(
                class_members[:, 0], class_members[:, 1], c=colors[cls], label=name
            )
        if f is not None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            X, Y = np.meshgrid(
                np.linspace(xmin - 0.2, xmax + 0.2, 100),
                np.linspace(ymin - 0.2, ymax + 0.2, 100),
            )
            positions = np.vstack([X.ravel(), Y.ravel()])

            Z = f(positions.T).reshape(X.shape[0], X.shape[0])
            plt.contourf(X, Y, Z, levels=0, alpha=0.2)
            plt.contour(X, Y, Z, levels=[0], linewidths=2).collections[0].set_label(
                label
            )

    if y is not None or f is not None:
        ax.legend()
    plt.show()


########################################################################
############################# ALGORITMOS  ##############################
########################################################################


def ajusta_PLA(datos, labels, max_iter, vini):
    """Calcula el hiperplano solución al problema de clasificación binaria.
    Argumentos:
    - datos: matriz de datos,
    - labels: Etiquetas,
    - max_iter: Número máximo de iteraciones
    - vini: Valor inicial
    Devuelve:
    - w: El vector de pesos
    - it: el número de iteraciones."""

    w = vini.copy()

    for i in range(1, max_iter+1):
        w_old = w.copy()

        for dato, label in zip(datos, labels):
            if w.dot(dato) * label <= 0:
                w += label * dato

        if (w == w_old).all():  # No hay cambios
            return w, i 

    return w, i

# Numero de puntos a utilziar
N = 100
a, b = simula_recta()

# Generamos el conjunto de puntos
x = simula_unif(N, 2, [-N / 2, N / 2])

# Creamos el conjunto de etiquetas
y = np.array([f (x[i, 0], x[i, 1], a, b) for i in range(N)])

# Generamos el conjunto de datos homogeneizados
x_hom = np.hstack((np.ones((N, 1)), x))

# Guardamos el conjunto de etiquetas con ruido en un array distinto
y_noise = y.copy()

# Modifica un 10% aleatorio de cada etiqueta
for label in {-1, 1}:
    labels = np.nonzero(y == label)[0]
    rand_labels = np.random.choice(labels, math.ceil(0.1 * len(labels)), replace=False)
    y_noise[rand_labels] = -y_noise[rand_labels]


def get_results(data, labels, classifier):
    """Obtiene el porcentaje de puntos correctamente clasificados
    por un clasificador dado.
    Argumentos:
    - data: los datos a utilizar,
    - labels: conjunto de etiquetas correctas,
    - classifier: clasificador, acepta los datos como parámetros."""

    # Si la etiqueta y la clasificación tienen signo distinto, entonces está mal clasificado.
    # en caso contrario esta bien clasificado.
    answers = labels * classifier(data)
    return 100 * len(answers[answers >= 0]) / len(labels)


def PLA_experiment(x, y, max_iters=1000):
    """
    Ejecuta el algoritmo PLA sobre el conjunto de datos dado
     - Utilizando un vector inicial nulo.
     - Utilizando un vector inicial aleatorio x10. Devuelve la media
       y la desviación típica de los resultados.

    Muestra en cada caso:
     - El porcentaje de acierto.
     - El número de iteraciones.
    """
    
    print("\tVector inicial de zeros.")
    w, its = ajusta_PLA(x, y, max_iters, np.zeros(3))
    
    print("\t\tIteraciones: {} épocas".format(its))
    print("\t\t% correctos: {}%".format(get_results(x, y, lambda x: x.dot(w))))
    print("\t\tGráfica")
    scatter(x[:, [1, 2]], y, lambda x: w[0] + x[:, 0] * w[1] + x[:, 1] * w[2])

    print("\tVector inicial aleatorio (media de 10 ejecuciones)")
    n_tests = 10
    iterations = np.empty((n_tests,))
    percentages = np.empty((n_tests,))

    for i in range(n_tests):
        w, iterations[i] = ajusta_PLA(x, y, max_iters, np.random.rand(3))
        percentages[i] = get_results(x, y, lambda x: x.dot(w))

    print(
        "\t\tIteraciones: \u03BC = {} épocas, \u03C3 = {:.02f}".format(
            np.mean(iterations), np.std(iterations)
        )
    )
    print(
        "\t\t% correctos: \u03BC = {}%, \u03C3 = {:.02f}".format(
            np.mean(percentages), np.std(percentages)
        )
    )


print("Apartado a.1 (sin ruido)")
PLA_experiment(x_hom, y)

# Ahora con los datos del ejercicio 1.2.b
print("Apartado a.2 (con ruido)")
PLA_experiment(x_hom, y_noise)


############################################################
##################### Apartado b ###########################
############################################################


def logistic_error(x, y, w):
    """
    Definicion del error logístico vectorial
    """
    return np.mean(np.log(1 + np.exp(-y * x.dot(w))))


def d_logistic_error(x, y, w):
    """
    Definicion de la derivada del error logístico (no vectorial)
    """
    return -y * x / (1 + np.exp(y * w.dot(x)))


def rl_sgd(data, labels, eta=0.01, w=None):
    """
    Implementación de Regresión logística con Gradiente
    Descendente Estocástico utilizando un tamaño del batch
    de 1.
    Argumentos:
    - data: Conjunto de datos con coordenadas homogéneas
    - labels: Conjunto de etiquetas de los datos
    Opcionales:
    - eta: Tasa de aprendizaje
    - w: Vector inicial de pesos. Vector nulo por defecto.

    Devuelve: el vector de pesos w.
    """

    # Inicializamos el vector de pesos
    if w is None:
        w = np.zeros(data.shape[1])
    # Inicializamos el vector de indices que vamos a utilizar.
    indexes = np.arange(data.shape[0])

    while True:
        w_old = w.copy()
        # Generamos una permutación de los índices
        indexes = np.random.permutation(indexes)
        # Para cada uno de ellos actualizamos el vector de pesos
        for index in indexes:
            w += -eta * d_logistic_error(data[index], labels[index], w)

        # Criterio de parada, distancia menor a 0.01
        if np.linalg.norm(w - w_old) <= 0.01:
            break

    return w


# Genera conjunto de datos 
a, b = simula_recta([0,2])
N = 100
datos = simula_unif(N, 2, [0, 2])
datos_hom = np.hstack((np.ones((N, 1)), datos))
labels = np.array([f (datos[i, 0], datos[i, 1], a, b) for i in range(N)])

# Ejecutamos el algoritmo sobre el conjunto de datos
w = rl_sgd(datos_hom, labels)

# Mostramos una gráfica
print("Apartado b.2")
print("\tGrafico de conjunto de entrenamiento.")
scatter(
    datos, labels, lambda x: w[0] + x[:, 0] * w[1] + x[:, 1] * w[2],
)

# Generamos un conjunto de test de tamaño 1000.
N = 1000
test = simula_unif(N, 2, [0, 2])
test_hom = np.hstack((np.ones((N, 1)), test))
labels_test = np.array([f (test[i, 0], test[i, 1], a, b) for i in range(N)])

print("\tGrafico de conjunto de test.")
scatter(
    test, labels_test, lambda x: w[0] + x[:, 0] * w[1] + x[:, 1] * w[2],
)

print(
    "\t% correctos en test RL: {}%".format(
        get_results(test_hom, labels_test, lambda x: x.dot(w))
    )
)
print("\tError: {}".format(logistic_error(test_hom, labels_test, w)))
