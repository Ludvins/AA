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


def f(x, y, a, b):
    return 1 if y - a * x - b >= 0 else -1


def simula_recta():
    points = np.random.uniform(-50, 50, size=(2, 2))
    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]
    # y = a*x + b
    a = (y2 - y1) / (x2 - x1)  # Calculo de la pendiente.
    b = y1 - a * x1  # Calculo del termino independiente.

    return a, b


def simula_unif(N, dim, rango):
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
    ws = [w]

    for it in range(max_iter):
        w_old = w.copy()

        for dato, label in zip(datos, labels):
            if w.dot(dato) * label <= 0:
                w += label * dato

        ws.append(w)

        if np.all(w == w_old):  # No hay cambios
            return w, ws, len(ws)

    return w, ws, len(ws)


N = 100
a, b = simula_recta()

x = simula_unif(N, 2, [-N / 2, N / 2])

y = np.empty((N))
for i in range(N):
    y[i] = f(x[i, 0], x[i, 1], a, b)

x_hom = np.hstack((np.ones((N, 1)), x))
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
    """Prueba el algoritmo de Perceptron para un conjunto de x dado."""

    print("\tVector inicial de zeros.")
    w, _, its = ajusta_PLA(x, y, max_iters, np.zeros(3))
    print("\t\tIteraciones: {} épocas".format(its))
    print("\t\t% correctos: {}%".format(get_results(x, y, lambda x: x.dot(w))))
    scatter(x[:, [1, 2]], y, lambda x: w[0] + x[:, 0] * w[1] + x[:, 1] * w[2])
    print("\tVector inicial aleatorio (media de 10 ejecuciones)")

    iterations = np.empty((10,))
    percentages = np.empty((10,))

    for i in range(10):
        w, _, iterations[i] = ajusta_PLA(x, y, max_iters, np.random.rand(3))
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


print("Apartado 2.1.a (sin ruido)")
PLA_experiment(x_hom, y)

# Ahora con los datos del ejercicio 1.2.b
print("Apartado 2.1.b (con ruido)")
PLA_experiment(x_hom, y_noise)
