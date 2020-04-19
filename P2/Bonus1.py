#!/usr/bin/env python3


#######################################################################
############################## IMPORTS ################################
#######################################################################

import matplotlib.pyplot as plt
from sympy import *
import numpy as np
import math
import inspect
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys

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


# Lectura de los datos de entrenamiento
x_train, y_train = readData("datos/X_train.npy", "datos/y_train.npy", [4, 8], [-1, 1])
# Lectura de los datos para el test
x_test, y_test = readData("datos/X_test.npy", "datos/y_test.npy", [4, 8], [-1, 1])

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


def Err(x, y, w):
    """Calcula el error para un modelo de regresión lineal"""
    wN = np.linalg.norm(x.dot(w) - y) ** 2
    return wN / len(x)


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
  - w, El vector de pesos y
  - iterations el número de iteraciones."""

    w = vini.copy()
    w_best = w.copy()
    result_best = get_results(datos, labels, lambda x: x.dot(w))

    for it in range(max_iter):
        w_old = w.copy()

        for dato, label in zip(datos, labels):
            if label * w.dot(dato) <= 0:
                w += label * dato

         result = get_results(datos, labels, lambda x: x.dot(w))
        if result > result_best:
            w_best = w.copy()
            result_best = result

        if np.all(w == w_old):  # No hay cambios
            break

    return w_best, it


########################################################
##################### APARTADO A #######################
########################################################

w_lin = pseudoinverse(x_train, y_train)

print("Calculando pesos PLAPocket a partir de RL... ", end="", flush=True)
w_pla, _ = PLAPocket(x_train, y_train, 1000, w_lin)
print("Hecho")

print("Calculando pesos PLAPocket a partir de vector de ceros... ", end="", flush=True)
w_pla_cero, _ = PLAPocket(x_train, y_train, 1000, np.random.rand(3))
print("Hecho")

ws = [w_lin, w_pla, w_pla_cero]
names = ["Regresión lineal", "PLA-Pocket (RL)", "PLA-Pocket (random)"]

print("Apartado 3.b.1")

scatter(
    x_train, y_train, w = ws, labels = nombres,
)

scatter(
    x_test, y_test, w = ws, labels = nombres,
)


for w, nombre in zip(ws, nombres):
    print(
        "Ein   para ",
        nombre,
        ": ",
        100 - get_results(x_train, y_train, lambda x: x.dot(w)),
    )
    print(
        "Etest para ",
        nombre,
        ": ",
        100 - get_results(x_test, y_test, lambda x: x.dot(w)),
    )


def cota(err, N, delta):
    """Calcula cota superior de Eout.
  Argumentos posicionales:
  - err: El error estimado,
  - N: El tamaño de la muestra y
  - delta: La tolerancia a error.
  Devuelve:
  - Cota superior de Eout"""
    return err + np.sqrt(1 / (2 * N) * (np.log(2 / delta) + 3 * 64 * np.log(2)))


Ein = Err(x_train, y_train, w_pla)
Etest = Err(x_test, y_test, w_pla)
delta = 0.05

print("Apartado 3.2.c (en terminal)")
print("Cota superior de Eout   (con Ein): {}".format(cota(Ein, len(x_train), delta)))
print("Cota superior de Eout (con Etest): {}".format(cota(Etest, len(x_test), delta)))
