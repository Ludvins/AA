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

# Cargamos los datos
x_train = np.load("./datos/X_train.npy")
x_test = np.load("./datos/X_test.npy")
y_train = np.load("./datos/y_train.npy")
y_test = np.load("./datos/y_test.npy")

# Creamos dataFrames de Pandas, añadiendo una columna de 1s.
df_train = pd.DataFrame({'Hom':1, 'Intensidad Promedio': x_train[:, 0], 'Simetria': x_train[:, 1], 'Y': y_train})
df_test = pd.DataFrame({'Hom':1, 'Intensidad Promedio': x_test[:, 0], 'Simetria': x_test[:, 1], 'Y': y_test})

# Nos quedamos con aquellas filas cuya Y valga 1 o 5
df_train = df_train[(df_train['Y'] == 1) | (df_train['Y'] == 5)]
# Cambiamos 1 por -1
df_train.loc[df_train['Y'] == 1, 'Y'] = -1
# Cambiamos 5 por 1
df_train.loc[df_train['Y'] == 5, 'Y'] = 1

df_test = df_test[(df_test['Y'] == 1) | (df_test['Y'] == 5)]
df_test.loc[df_test['Y'] == 1, 'Y'] = -1
df_test.loc[df_test['Y'] == 5, 'Y'] = 1

# Volcamos los datafames en vectores
x_train = df_train[['Hom', 'Intensidad Promedio', 'Simetria']].to_numpy()
y_train = df_train[['Y']].to_numpy()
x_test = df_test[['Hom', 'Intensidad Promedio', 'Simetria']].to_numpy()
y_test = df_test[['Y']].to_numpy()

#######################################################################
########################## FUNCIONES AUXILIARES #######################
#######################################################################

def to_numpy(func):
  """
  Función decorador para convertir funciones a versión NumPy
  Esto nos permite llamar funciones que se declaran con 2 variables
  por comodidad utilizando una unica variable bi-dimensional
  """
  def numpy_func(w):
    return func(*w)
  return numpy_func

def Err(x, y , w):
    """Función de error de un modelo de regresion lineal"""
    return (np.linalg.norm(x.dot(w) - y)**2)/len(x)

def dErr(x, y, w):
    """Derivada de la función de error"""
    return 2*(x.T.dot(x.dot(w) - y))/len(x)

def scatter(x, y = None, w = None, labels = None):
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
  ax.set_xlabel('Intensidad Promedio')
  ax.set_ylabel('Simetría')
  ax.set_ylim(1.1*np.min(x[:, 2]),1.1*np.max(x[:, 2]))
  if y is None:
    ax.scatter(x[:, 1], x[:, 2])
  else:
    colors = {-1: 'red', 1: 'blue'}
    # Para cada clase:
    for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
      # Obten los miembros de la clase
      class_members = x[np.where(y == cls)[0]]
      
      # Representa en scatter plot
      ax.scatter(class_members[:, 1],
                 class_members[:, 2],
                 c = colors[cls],
                 label = name)
    if w is not None:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      v = np.array([xmin, xmax])
      if labels is not None:
        for a, n in zip(w, labels):
          if len(a) == 3:
            ax.plot(v, (-a[0] -v*a[1])/a[2], label = n)
          else:
            X, Y = np.meshgrid(np.linspace(xmin-0.2, xmax+0.2, 100), np.linspace(ymin-0.2, ymax+0.2, 100))
            F = a[0] + a[1]*X + a[2]*Y + a[3]*X*Y + a[4]*X*X + a[5]*Y*Y
            plt.contour(X, Y, F, [0]).collections[0].set_label(n)
      else:
        for a in w:
          if len(a) == 3:
            ax.plot(v, (-a[0] -v*a[1])/a[2])
          else:
            X, Y = np.meshgrid(np.linspace(xmin-0.2, xmax+0.2, 100), np.linspace(ymin-0.2, ymax+0.2, 100))
            F = a[0] + a[1]*X + a[2]*Y + a[3]*X*Y + a[4]*X*X + a[5]*Y*Y
            plt.contour(X, Y, F, [0])
                
  if y is not None or w is not None:
    ax.legend()
  plt.show() 


#######################################################################
############################### APARTADO 1 ############################
#######################################################################

def sgd(x, y, r = 0.01, max_iterations = 500, batch_size = 32,verbose = 0):
    """Implementa la función de gradiente descendiente estocástico.
    Argumentos:
    - x: Datos en coordenadas homogéneas
    - y: Etiquetas asociadas (-1 o 1)
    Argumentos opcionales:
    - r: Tasa de aprendizaje
    - max_iterations: Máximo número de iteraciones
    - batch_size: Tamaño del batch
    - verbose: Muestreo de Ein por iteración

    Devuelve:
    - w: coeficientes de la recta de regresión w[1] + w[0]x
    - ws: vector de coeficientes en cada iteracion. Con fines de muestreo. 
    """

    # Incializamos los coeficientes de la recta de regresión a (0,0)
    w = np.zeros((x.shape[1], 1))
    ws = [w]

    # Inicializamos el vector de indices
    indexes = np.random.permutation(np.arange(len(x)))
    # Marcador de posición en el vector de indices
    offset = 0

    for i in range(max_iterations) :

        if verbose:
            print("  ", i,": Ein -> ", Err(x, y, w))

        # Tomamos los indices que vamos a utilizar en esta iteración
        ids = indexes[offset : offset + batch_size]
        # Actualizamos w y ws
        w = w - r*dErr(x[ids, :], y[ids], w)
        ws += [w]
        # Actualizamos el offset
        offset += batch_size

        # En caso de pasarnos, reseteamos
        if offset > len(x):
            offset = 0
            indexes = np.random.permutation(np.arange(len(x)))
            
    return w, ws

def pseudoinverse(x, y):
  """Calcula el vector w a partir del método de la pseudo-inversa."""
  # Calculamos la descomposicion usv
  u, s, v = np.linalg.svd(x)
  # Calculamos la inversa de s, poniendo a 0 los valores cercanos debidos a errores de redondeo.
  d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
  
  return v.T.dot(d).dot(d).dot(v).dot(x.T).dot(y)

print("Apartado 1.")
w_sgd, ws_sgd = sgd(x_train, y_train)
print('\tBondad del resultado para gradiente descendente estocastico (500 iteraciones):')
print("\t\tEin:  ", Err(x_train, y_train, w_sgd))
print("\t\tEout: ", Err(x_test, y_test, w_sgd))
input()

w_pinv = pseudoinverse(x_train, y_train)
print('\tBondad del resultado para pseudo-inversa:')
print("\t\tEin:  ", Err(x_train, y_train, w_pinv))
print("\t\tEout: ", Err(x_test, y_test, w_pinv))
input()

print("\tMostrando gráfico con ambos modelos de regresión...")
scatter(x_train, y_train, [w_sgd, w_pinv], ["SGD", "PINV"])


#######################################################################
############################### APARTADO 2 ############################
#######################################################################

def simula_unif(n, d, size):
    return np.random.uniform(-size, size, (n, d))

@to_numpy
def f(x1, x2):
  """Función de apartado 2.2.b"""
  return np.sign((x1 - 0.2)**2 + x2**2 - 0.6)

def crea_datos(noise = True):
  """
  Crea los datos ya homogeneizados
  """
  x = simula_unif(1000, 2, 1)
  # Crea las etiquetas, asignando aleatorias a el 10% final
  if noise:
    y = np.hstack((
      f(x[:900, :].T),
      np.random.choice([-1, 1], 100)))[:, None]
  else:
    y = f(x.T)[:, None]
  # Añade columna de 1s
  x = np.hstack((np.ones((1000, 1)), x))
  return x, y
    
print("Apartado 2.")
x , y = crea_datos()
print("\tMostrando puntos generados...")
scatter(x)
print("\tMostrando puntos etiquetados...")
scatter(x,y)

w, ws = sgd(x,y)
print("\tBondad del resultado del experimento (1 ejecución)")
print("\t\tEin:  {}".format(Err(x, y, w)))
input()

print("\tMostrando puntos etiquetados junto con recta de regresión...")
scatter(x,y,[w], ["SDG lineal"])


def experimento1():
  """
  Función que se encarga de realizar el primer experimento que se propone.
  """
  Errs = 0

  for i in range(1000):
    sys.stdout.write('\r')
    sys.stdout.write("\t\tCalculando las 1000 iteraciones del experimento: [%-20s] %d%%" % ('='*(i//50), i//10))
    sys.stdout.flush()

    x_train, y_train = crea_datos()
    x_test, y_test = crea_datos()
    
    w, _ = sgd(x_train, y_train)
    Ein = Err(x_train, y_train, w)
    Eout = Err(x_test, y_test, w)
    
    Errs = Errs + np.array([Ein, Eout])
    
  Ei, Eo = Errs/1000

  print("\n\tBondad del resultado del experimento (1000 ejecuciones)")
  print("\t\tEin:  {}".format(Ei))
  print("\t\tEout: {}".format(Eo))
  input()

print("\tExperimento 1. Ajuste de los datos mediante un modelo de regresión lineal.")
experimento1()

        
def add_caracteristicas(x):
  """
  Esta función añade a x = [x1, x2] 3 nuevas columnas de caracteristicas
  - x1*x2
  - x1*x1
  - x2*x2
  """
  x1x2 =  np.multiply(x[:,1],x[:,2])[:, None]
  x1x1 =  np.multiply(x[:,1],x[:,1])[:, None]
  x2x2 =  np.multiply(x[:,2],x[:,2])[:, None]
  x = np.hstack((x, x1x2, x1x1, x2x2))
  return x


x, y = crea_datos()

x = add_caracteristicas(x)
    
w, _ = sgd(x, y)
Ein = Err(x, y, w)
  
print("\tBondad del resultado del experimento con nuevas características (1 ejecucion)")
print("\t\tEin:  {}".format(Ein))
input()

print("\tMostrando modelo de regresión no lineal.")
scatter(x,y,[w], ["SDG no lineal"])




def experimento2():
  """
  Función que se encarga de realiar el segundo experimento que se propone, conlas nuevas características
  """
  Errs = 0

  for i in range(1000):
    sys.stdout.write('\r')
    sys.stdout.write("\t\tCalculando las 1000 iteraciones del experimento 2: [%-20s] %d%%" % ('='*(i//50), i//10))
    sys.stdout.flush()
    x_train, y_train = crea_datos()
    x_test, y_test = crea_datos()
    
    x_train = add_caracteristicas(x_train)
    x_test = add_caracteristicas(x_test)
    
    w, _ = sgd(x_train, y_train)
    Ein = Err(x_train, y_train, w)
    Eout = Err(x_test, y_test, w)
    
    Errs = Errs + np.array([Ein, Eout])

  Ei, Eo = Errs/1000

  print("\n\tBondad del resultado del experimento con nuevas características (1000 ejecuciones)")
  print("\t\tEin:  {}".format(Ei))
  print("\t\tEout: {}".format(Eo))
  input()

print("\tExperimento 2. Ajuste de los datos mediante un modelo de regresión no lineal.")
experimento2()


x, y = crea_datos()
x = add_caracteristicas(x)

print("\tMostrando resultados de usar pesos perfectos en el modelo no lineal:")
# Vector de pesos perfectos extraido de la función.
w = np.array([-0.6+0.2*0.2, -2*0.2, 0.0, 0.0, 1, 1]).astype(np.float64)[:, None]
print("\t\tBondad del resultado del experimento con pesos perfectos:")
print("\t\t\tEin:  {}".format(Err(x, y, w)))
input()
print("\t\tMostrando modelo de regresión no lineal perfecto.")
scatter(x,y,[w], ["SDG no lineal, pesos perfectos"])

x, y = crea_datos(False)
x = add_caracteristicas(x)

print("\tMostrando resultados de usar pesos perfectos sin ruido en los datos:")
# Vector de pesos perfectos extraido de la función.
w = np.array([-0.6+0.2*0.2, -2*0.2, 0.0, 0.0, 1, 1]).astype(np.float64)[:, None]
print("\t\tBondad del resultado del experimento con pesos perfectos:")
print("\t\t\tEin:  {}".format(Err(x, y, w)))
input()
