#!/usr/bin/env python3


#######################################################################
############################## IMPORTS ################################
#######################################################################

import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(1)

#######################################################################
################### TRATAMIENTO DE DATOS INICIAL ######################
#######################################################################

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
    # Leemos los ficheros
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []
    # Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
    for i in range(0, datay.size):
        if datay[i] == digits[0] or datay[i] == digits[1]:
            if datay[i] == digits[0]:
                y.append(labels[0])
            else:
                y.append(labels[1])
            x.append(np.array([1, datax[i][0], datax[i][1]]))

    x = np.array(x, np.float64)
    y = np.array(y, np.float64)

    return x, y

print("Leyendo los datos... ", end="", flush=True)
# Lectura de los datos de entrenamiento
x_train, y_train = readData("datos/X_train.npy", "datos/y_train.npy", [4, 8], [-1, 1])
# Lectura de los datos para el test
x_test, y_test = readData("datos/X_test.npy", "datos/y_test.npy", [4, 8], [-1, 1])
print("Hecho")

#########################################################
################# FUNCIONES AUXILIARES ##################
#########################################################

def scatter(x, y=None, w=None, labels=None):
    """
    Funcion scatter, nos permite pintar un conjunto de puntos en un plano 2D.
    Argumentos:
    - x: Conjunto de puntos
    Argumentos opcionales:
    - y: Conjunto de etiquetas de dichos puntos.
    - w: Vector con modelos de regresión.
    - labels: Etiquetas para los modelos en w.

    Admite pasar modelos de regresión no lineal (como se nos pide en algún ejercicio), en este caso esta
    limitado a el caso que se nos pide con 6 pesos.
    """
    _, ax = plt.subplots()
    ax.set_xlabel("Intensidad Promedio")
    ax.set_ylabel("Simetría")
    ax.set_ylim(1.1 * np.min(x[:, 2]), 1.1 * np.max(x[:, 2]))
    if y is None:
        ax.scatter(x[:, 1], x[:, 2])
    else:
        colors = {-1: "red", 1: "blue"}
        # Para cada clase:
        for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
            # Obten los miembros de la clase
            class_members = x[np.where(y == cls)[0]]

            # Representa en scatter plot
            ax.scatter(
                class_members[:, 1], class_members[:, 2], c=colors[cls], label=name
            )
        if w is not None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            v = np.array([xmin, xmax])
            for a, n in zip(w, labels):
                ax.plot(v, (-a[0] - v * a[1]) / a[2], label=n)

    if y is not None or w is not None:
        ax.legend()
    plt.show()


def Err(data, labels, classifier):
    """Obtiene el tanto por 1 de puntos incorrectamente clasificados
    por un clasificador dado.
    Argumentos:
    - data: los datos a utilizar,
    - labels: conjunto de etiquetas correctas,
    - classifier: clasificador."""

    # Si la etiqueta y la clasificación tienen signo distinto, entonces está mal clasificado.
    # en caso contrario esta bien clasificado.
    answers = labels * classifier(data)
    return len(answers[answers < 0]) / len(labels)

#############################################################
###################### PLA-Pocket ###########################
#############################################################

# Pseudo-inversa
def pseudoinverse(x, y):
    """Calcula el vector w a partir del método de la pseudo-inversa."""
    u, s, v = np.linalg.svd(x)
    d = np.diag([0 if np.allclose(p, 0) else 1 / p for p in s])
    return v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)


def PLAPocket(datos, labels, max_iter, vini):
    """Calcula el hiperplano solución al problema de clasificación binaria.
    Argumentos posicionales:
    - datos: matriz de datos,
    - labels: Etiquetas,
    - max_iter: Número máximo de iteraciones
    - vini: Valor inicial
    Devuelve:
    - w_best: El mejor vector de pesos
    - it: El número de iteraciones."""

    w = vini.copy()
    w_best = w.copy()
    err_best = Err(datos, labels, lambda x: x.dot(w))

    for it in range(max_iter):
        w_old = w.copy()

        for dato, label in zip(datos, labels):
            if label * w.dot(dato) <= 0:
                w += label * dato

        err = Err(datos, labels, lambda x: x.dot(w))
        if err <= err_best:
            w_best = w.copy()
            err_best = err

        if np.all(w == w_old):  # No hay cambios
            break

    return w_best, it


########################################################
##################### APARTADO A #######################
########################################################

# Calculamos los pesos de la pseudoinversa
w_lin = pseudoinverse(x_train, y_train)

print("Aplicando PLAPocket sobre RL... ", end="", flush=True)
w_pla, _ = PLAPocket(x_train, y_train, 1000, w_lin)
print("Hecho")

print("PLAPocket sobre vector nulo... ", end="", flush=True)
w_pla_cero, _ = PLAPocket(x_train, y_train, 1000, np.random.rand(3))
print("Hecho")

ws = [w_lin, w_pla, w_pla_cero]
names = ["Regresión lineal   ", "PLA-Pocket (RL)    ", "PLA-Pocket (random)"]

print("Apartado b.1 (Gráficas)")

scatter(
    x_train, y_train, w=ws, labels=names,
)

scatter(
    x_test, y_test, w=ws, labels=names,
)

print("Apartado b.2")
for w, n in zip(ws, names):
    print(
        "\tEin   ", n, " -> ", Err(x_train, y_train, lambda x: x.dot(w)),
    )
    print(
        "\tEtest ", n, " -> ", Err(x_test, y_test, lambda x: x.dot(w)),
    )


def cota(err, N, delta, cardinality):
    """Calcula cota superior de Eout.
    Argumentos posicionales:
    - err: El error estimado.
    - N: El tamaño de la muestra.
    - delta: La tolerancia a error.
    Devuelve:
    - Cota superior de Eout."""
    return err + np.sqrt((1 / (2 * N)) * (np.log(2*cardinality / delta)))


Ein =  Err(x_train, y_train, lambda x: x.dot(w_pla))
Etest = Err(x_test, y_test, lambda x: x.dot(w_pla))
delta = 0.05

print("Apartado b.3")
print("\tCota superior de Eout (con Ein)   -> ", cota(Ein, len(x_train), delta, 2**192))
print("\tCota superior de Eout (con Etest) -> ", cota(Etest, len(x_test), delta, 1))
