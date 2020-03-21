#!/usr/bin/env python3

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

#######################################################################
################### TRATAMIENTO DE DATOS INICIAL ######################
#######################################################################

# Load Data
x_train = np.load("datos/X_train.npy")
x_test = np.load("datos/X_test.npy")
y_train = np.load("datos/y_train.npy")
y_test = np.load("datos/y_test.npy")

# Create Pandas' Dataframe
df_train = pd.DataFrame({'Intensidad Promedio': x_train[:, 0], 'Simetria': x_train[:, 1], 'Y': y_train})
df_test = pd.DataFrame({'Intensidad Promedio': x_test[:, 0], 'Simetria': x_test[:, 1], 'Y': y_test})

# Get rows with Y value of 1 or 5
df_train = df_train[(df_train['Y'] == 1) | (df_train['Y'] == 5)]
# Change 1 to -1
df_train.loc[df_train['Y'] == 1, 'Y'] = -1
# Change 5 to 1
df_train.loc[df_train['Y'] == 5, 'Y'] = 1

df_test = df_test[(df_test['Y'] == 1) | (df_test['Y'] == 5)]
df_test.loc[df_test['Y'] == 1, 'Y'] = -1
df_test.loc[df_test['Y'] == 5, 'Y'] = 1

x_train = df_train[['Intensidad Promedio', 'Simetria']].to_numpy()
y_train = df_train[['Y']].to_numpy()
x_test = df_test[['Intensidad Promedio', 'Simetria']].to_numpy()
y_test = df_test[['Y']].to_numpy()

#######################################################################
########################## FUNCIONES AUXILIARES #######################
#######################################################################

def Err(x, y , w):
    """Función de error de un modelo de regresion lineal"""
    return (np.linalg.norm(x.dot(w) - y)**2)/len(x)

def dErr(x, y, w):
    """Derivada de la función de error"""
    return 2*(x.T.dot(x.dot(w) - y))/len(x)

def scatter(df, w, labels):

    _, ax = plt.subplots()
    class_colors = {-1: 'green', 1: 'blue'}
    
    # Para cada clase:
    for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
        # Obten los miembros de la clase
        class_members = df[df['Y'] == cls]
        # Representa en scatter plot
        ax.scatter(class_members['Intensidad Promedio'],
                   class_members['Simetria'],
                   c=class_colors[cls],
                   label=name)

    xmin, xmax = ax.get_xlim()
    x = np.array([xmin, xmax])
    for a, n in zip(w, labels):
        ax.plot(x, a[1] + x*a[0], label = n)
    ax.legend()
    plt.show()


#######################################################################
############################### ALGORITMOS ############################
#######################################################################

def sgd(x, y, r = 0.01, max_iterations = 20000, batch_size = 32, verbose = 0):
    """Implementa la función de gradiente descendiente estocástico
    para problemas de regresión lineal.
    Argumentos:
    - x: Datos en coordenadas homogéneas
    - y: Etiquetas asociadas (-1 o 1)
    Argumentos opcionales:
    - r: Tasa de aprendizaje
    - max_iterations: máximo número de iteraciones
    - batch_size: tamaño del batch"""

    
    w = np.zeros((2, 1))
    ws = [w]
    its = 0
    indexes = np.random.permutation(np.arange(len(x)))
    offset = 0

    for i in range(max_iterations) :

        if verbose:
            print("  ", i,": Ein -> ", Err(x, y, w))
        
        ids = indexes[offset : offset + batch_size]
        w = w - r*dErr(x[ids, :], y[ids], w)
        ws += [w]

        offset += batch_size
        if offset > len(x):
            offset = 0
            indexes = np.random.permutation(np.arange(len(x)))
    return w, ws

def pseudoinverse(x, y):
  """Calcula el vector w a partir del método de la pseudo-inversa."""
  u, s, v = np.linalg.svd(x)
  d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
  return v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)




w_sgd, ws_sgd = sgd(x_train, y_train)
print('Bondad del resultado para grad. descendente estocastico:')
print("  Ein:  ", Err(x_train, y_train, w_sgd))
print("  Eout: ", Err(x_test, y_test, w_sgd))


w_pinv = pseudoinverse(x_train, y_train)
print('\nBondad del resultado para pseudo-inversa:')
print("  Ein:  ", Err(x_train, y_train, w_pinv))
print("  Eout: ", Err(x_test, y_test, w_pinv))

scatter(df_train, [w_sgd, w_pinv], ["SGD", "PINV"])


