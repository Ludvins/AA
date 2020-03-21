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
from terminaltables import AsciiTable

#######################################################################
###################### FUNCIONES AUXILIARES ###########################
#######################################################################
"""
Función decorador para convertir funciones a versión NumPy
Esto nos permite llamar funciones que se declaran con 2 variables
por comodidad utilizando una unicavariable bi-dimensional
"""
def to_numpy(func):
  def numpy_func(w):
    return func(*w)
  return numpy_func


"""
Muestra una figura 3D de la función f junto con la imagen de un conjunto de puntos
Argumentos:
- f: función sobre la que pintar un entorno
- points: Array de coordenadas x,y de los puntos.
"""
def display_figure(f, points):

  fig = plt.figure(figsize=plt.figaspect(2.))
  fig.suptitle('Evolución de los puntos en iteracoines y sobre superficie')

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


#######################################################################
######################## GRADIENTE DESCENDENTE ########################
#######################################################################
"""
Aproxima el mínimo de una función mediante el método de gradiente descendiente.
Argumentos posicionales:
- x_0: Punto inicial
- f: Función a minimizar. Debe ser diferenciable
- df: Gradiente de `f`
- r: Tasa de aprendizaje
- max_iter: Máximo número de iteraciones

Argumentos opcionales:
- target_value: Valor buscado, criterio de parada.
Si no se especifica se ignora este criterio de parada.

Devuelve:
- Mínimo hallado
- Número de iteraciones que se han necesitado
- Array de secuencia de puntos obtenidos
"""
def gradient_descent(f, df, x_0, r, max_iter, target_value=-math.inf):
    it = 0
    point = np.copy(x_0)
    x = [[*point]]
    
    while it < max_iter and f(point) > target_value:
        point -= r*df(point)
        it += 1
        x.append([*point])
    return point, it, x

#######################################################################
######################## FUNCIONES DEL EJERCICIO ######################
#######################################################################
  
"""Declaración de la función del ejercicio 1.2"""
@to_numpy
def E(u,v):
    return (u*np.exp(v) - 2*v*np.exp(-u))**2

"""Derivada parcial de E con respecto a u"""
def E_u(u, v):
  return 2 * (u*np.exp(v) - 2*v*np.exp(-u)) * (np.exp(v) + 2*v*np.exp(-u))

"""Derivada parcial de E con respecto a v"""
def E_v(u, v):
  return 2 * (u*np.exp(v) - 2*v*np.exp(-u)) * (u*np.exp(v) - 2*np.exp(-u))

"""Gradiente de E"""
@to_numpy
def dE(u, v):
  return np.array([E_u(u, v), E_v(u, v)])

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

"""Gradiente de f"""
@to_numpy
def df(x,y):
    return np.array([f_x(x,y), f_y(x,y)])

#######################################################################
############################# EJERCICIO 1.2 ###########################
#######################################################################
  
print("Ejercicio 1.2")
minima, it, _ = gradient_descent(E,
                                 dE,
                                 np.array([1.0, 1.0]).astype(np.float64),
                                 0.1,
                                 1000,
                                 1e-14)

print('\tFuncion que estamos tratando: (ue^v - 2ve^-u)^2')
print('\tDerivada respecto a u: 2(ue^v - 2ve^-u)(e^v + 2ve^-u)')
print('\tDerivada respecto a v: 2(ue^v - 2ve^-u)(ue^v - 2e^-u)')
print('\tGradiente de la funcion: [2(ue^v - 2ve^-u)(e^v + 2ve^-u),  2(ue^v - 2ve^-u)(ue^v - 2e^-u)]')

print('\tNumero de iteraciones necesarias: {}'.format(it))
print('\tPunto obtenido: {}'.format(minima))
print('\tValor de la función en el punto: {}'.format(E(minima)))
input()

#######################################################################
############################# EJERCICIO 1.3 ###########################
#######################################################################

print("Ejercicio 1.3")
minima, it, points = gradient_descent(f,
                                     df,
                                     np.array([1.0, -1.0]).astype(np.float64),
                                     0.01,
                                     50)

print('\tResultados utilizando tasa de aprendizaje 0.01') 
print('\tNumero de iteraciones: {}'.format(it))
print('\tPunto obtenido: {}'.format(minima))
print('\tValor de la función en el punto: {}'.format(E(minima)))
input()

display_figure(f, points)


minima, it, points = gradient_descent(f,
                                      df,
                                      np.array([1.0, -1.0]).astype(np.float64),
                                      0.1,
                                      50)

print('\tResultados utilizando tasa de aprendizaje 0.1') 
print('\tPunto obtenido: {}'.format(minima))
print('\tValor de la función en el punto: {}'.format(E(minima)))
input()

display_figure(f, points)

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
  minima, it, _ = gradient_descent(f,
                                   df,
                                   p,
                                   0.01,
                                   50)
  data.append([p[0], p[1], minima[0], minima[1], f(minima)])

table = AsciiTable(data)
print('\tTabla apartado 1.2.b')
print(table.table)
