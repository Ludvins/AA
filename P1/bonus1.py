#!/usr/bin/env python3


import matplotlib.pyplot as plt
from sympy import *
import numpy as np
import math
import inspect
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys
from terminaltables import AsciiTable


def display_figure(f, points):

  fig = plt.figure(figsize=plt.figaspect(2.))
  fig.suptitle('Evolución de los puntos en iteraciones y sobre superficie')

  # First subplot
  ax = fig.add_subplot(2, 2, 1)

  ax.plot([f(a) for a in points], linestyle='--', marker='o', color='b')
  ax.grid(True)
  ax.set_ylabel('f(x,y)')

  # Second subplot
  ax = fig.add_subplot(2, 2, 2, projection='3d')

    # Obtención de coordenadas de los puntos
  px = [a[0] for a in points] 
  py = [a[1] for a in points]

  # Imagenes de los puntos
  pz = [f(a) for a in points]

  # Seleccinamos un entorno que englobe todos los puntos que tenemos

  x_min = np.min(px)
  x_max = np.max(px)
  y_min = np.min(py)
  y_max = np.max(py)
  
  x = np.linspace(x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1)
  y = np.linspace(y_min - (y_max-y_min)*0.1, y_max + (y_max-y_min)*0.1)
  X, Y = np.meshgrid(x, y)
  Z = f([X, Y])
  
  # Pintamos la superficia de la función
  ax.plot_surface(X, Y, Z, alpha = 0.5)

  # Pintamos los puntos
  ax.scatter3D(px, py, pz, c='r',depthshade=False);
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('f(x,y)')

  
  plt.show()

def to_numpy(func):
  """
  Función decorador para convertir funciones a versión NumPy
  Esto nos permite llamar funciones que se declaran con 2 variables
  por comodidad utilizando una unica variable bi-dimensional
  """
  def numpy_func(w):
    return func(*w)
  return numpy_func


def newton(p, f, df, hf, r, max_iterations):
  """ Aproxima mínimos con el método de Newton.
    Argumentos:
    - p: Punto inicial
    - f: Función a minimizar
    - df: Gradiente de f
    - hf: Hessiana de f
    - r: Tasa de aprendizaje
    - max_iterations: Máximo número de iteraciones
    Devuelve:
    - Mínimo hallado
    """

  w = p
  ws = [p]
  
  for _ in range(max_iterations):
    w = w - r*np.linalg.inv(hf(w)).dot(df(w))
    ws.append(w)

  return w, ws

"""Declaración de la función del ejercicio 1.3"""
@to_numpy
def f(x,y):
  return (x-2)**2 + 2*(y + 2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
  
"""Derivada de f respecto de x"""
def f_x(x,y):
  return 2*(2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) + x - 2) 

"""Derivada de f respecto de y"""
def f_y(x,y):
  return 4*(np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+y+2)

"""Segunda derivada de f respecto de x"""
def f_xx(x,y):
  return 2 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

"""Segunda derivada de f respecto de x e y"""
def f_xy(x, y):
  return 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

"""Segunda derivada de f respecto de y"""
def f_yy(x,y):
  return 4 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

"""Gradiente de f"""
@to_numpy
def df(x,y):
    return np.array([f_x(x,y), f_y(x,y)])

"""Hessiana de f"""
@to_numpy
def hf(x, y):
  return np.array([
    f_xx(x,y), f_xy(x,y),
    f_xy(x,y), f_yy(x,y)
  ]).reshape((2, 2))

print("\nBonus 1.")
print("\tRepresentación de curva de decrecimiento del método de Newton. Punto inicial (1,-1). Tasa de aprendizaje 0.01")

w_001, ws_001 = newton( np.array([1., -1.]).astype(np.float64),
                        f,
                        df,
                        hf,
                        0.01,
                        50
)
display_figure(f, ws_001)

print("\tRepresentación de curva de decrecimiento del método de Newton. Punto inicial (1,-1). Tasa de aprendizaje 0.1")

w_01, ws_01 = newton( np.array([1., -1.]).astype(np.float64),
                      f,
                      df,
                      hf,
                      0.1,
                      50
)


display_figure(f, ws_01)


"""
Definimos un array 'data' donde almacenaremos la información de la tabla por filas.
Luego usaremos la libreria AsciiTable para mostrar dicha tabla por pantalla
"""
data = [
  ['x inicial', 'y inicial', 'x final', 'y final', 'valor mínimo'],
]

"""
Valores iniciales de las ejecuciones
"""
initial_points = [
  np.array([2.1, -2.1]).astype(np.float64),
  np.array([3, -3]).astype(np.float64),
  np.array([1.5, 1.5]).astype(np.float64),
  np.array([1.0, -1.0]).astype(np.float64)
]

"""
Para cada punto de los iniciales, realizamos el gradiente descendente y almacenamos los resultados en la tabla
"""
for p in initial_points:
  minima, _ = newton(p,
                     f,
                     df,
                     hf,
                     0.1,
                     50)
  data.append([p[0], p[1], minima[0], minima[1], f(minima)])

table = AsciiTable(data)
print('\tTabla con distintos puntos iniciales. Tasa de aprendizaje = 0.1')
print(table.table)

data = [
  ['x inicial', 'y inicial', 'x final', 'y final', 'valor mínimo'],
]

"""
Valores iniciales de las ejecuciones
"""
initial_points = [
  np.array([2.1, -2.1]).astype(np.float64),
  np.array([3, -3]).astype(np.float64),
  np.array([1.5, 1.5]).astype(np.float64),
  np.array([1.0, -1.0]).astype(np.float64)
]

"""
Para cada punto de los iniciales, realizamos el gradiente descendente y almacenamos los resultados en la tabla
"""
for p in initial_points:
  minima, _ = newton(p,
                     f,
                     df,
                     hf,
                     0.01,
                     50)
  data.append([p[0], p[1], minima[0], minima[1], f(minima)])

table = AsciiTable(data)
print('\tTabla con distintos puntos iniciales. Tasa de aprendizaje = 0.01')
print(table.table)


