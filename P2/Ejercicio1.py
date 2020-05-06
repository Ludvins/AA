########################################################################
############################## IMPORTS #################################
########################################################################

import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(1)

#######################################################################
######################## FUNCIONES AUXILIARES #########################
#######################################################################


def simula_unif(N, dim, rango):
    """
    Devuelve un conjunto de N puntos de R^dim, dentro del rango indicado
    utilizando una districubión uniforme.
    """
    return np.random.uniform(rango[0], rango[1], (N, dim))


def simula_gaus(N, dim, sigma):
    """
    Genera un conjunto de N puntos de R^dim utilizando una distribución normal donde
    en cadacolumna se utiliza un sigma determinado.
    """
    media = 0
    out = np.zeros((N, dim), np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i, :] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta():
    """
    Genera dos puntos aleatorios y devuelve los coeficientes de la recta que los une
    """
    points = np.random.uniform(-50, 50, size=(2, 2))
    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]
    if x2 == x1:
        print(
            "Se han generado dos puntos alineados verticalmente -> No se puede construir una recta como funcion que los una. (División por 0)"
        )
    # y = a*x + b
    a = (y2 - y1) / (x2 - x1)  # Calculo de la pendiente.
    b = y1 - a * x1  # Calculo del termino independiente.

    return a, b


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
            # Calculamos la malla a utilizar
            X, Y = np.meshgrid(
                np.linspace(xmin - 0.2, xmax + 0.2, 100),
                np.linspace(ymin - 0.2, ymax + 0.2, 100),
            )
            # Calculamos las imagenes por f de los puntos de la malla
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = f(positions.T).reshape(X.shape[0], X.shape[0])

            # Utilizamos contourf para pintar las regiones que separa la funcion
            plt.contourf(X, Y, Z, levels=0, alpha=0.2)
            # Pintamos la funcion con contour
            plt.contour(X, Y, Z, levels=[0], linewidths=2).collections[0].set_label(
                label
            )

    if y is not None or f is not None:
        ax.legend()
    plt.show()


#################################################################
#################################################################
########################### APARTADO 1 ##########################
#################################################################
#################################################################

print("Apartado 1.")
print("\tApartado 1.a (Gráfica)")
x = simula_unif(50, 2, [-50, 50])
scatter(x)

print("\tApartado 1.b (Gráfica)")
x = simula_gaus(50, 2, np.array([5, 7]))
scatter(x)


#################################################################
#################################################################
########################### APARTADO 2 ##########################
#################################################################
#################################################################

print("Apartado 2.")

# Definimos la función que vamos a utilizar para asignar etiquetas
#  equivale a signo(y -ax - b), pero como la función np.sign devuelve
#  0 al evaluarla en 0, utilizamos esta construcción en su lugar.
def f(x, y, a, b):
    return 1 if y - a * x - b >= 0 else -1


# Reestablecemos la semilla
np.random.seed(1)

# Numero de puntos a utilziar
N = 100
a, b = simula_recta()

# Generamos el conjunto de puntos
x = simula_unif(N, 2, [-N / 2, N / 2])
# Creamos el conjunto de etiquetas
y = np.array([f(x[i, 0], x[i, 1], a, b) for i in range(N)])

# Definimos una función lambda que corresponde con la recta
f_recta = lambda x: x[:, 1] - a * x[:, 0] - b

print("\tApartado 2.a (Gráfica)")
scatter(x, y, f_recta, "Recta")

y_noise = y.copy()
# Modifica un 10% aleatorio de cada etiqueta.
# Para cada etiqueta seleccionamos un 10% de los indices y los multiplicamos por -1.
for label in {-1, 1}:
    labels = np.nonzero(y == label)[0]
    rand_labels = np.random.choice(labels, math.ceil(0.1 * len(labels)), replace=False)
    y_noise[rand_labels] = -y[rand_labels]
y = y_noise

print("\tApartado 2.b (Gráfica)")
scatter(x, y, f_recta, "Recta")


#################################################################
#################################################################
########################### APARTADO 3 ##########################
#################################################################
#################################################################


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


# Lista de clasificadores con su nombre
classifiers = [
    (lambda x: x[:, 1] - a * x[:, 0] - b, "Recta"),
    (lambda x: (x[:, 0] - 10) ** 2 + (x[:, 1] - 20) ** 2 - 400, "Elipse 1"),
    (lambda x: 0.5 * (x[:, 0] + 10) ** 2 + (x[:, 1] - 20) ** 2 - 400, "Elipse 2"),
    (lambda x: 0.5 * (x[:, 0] - 10) ** 2 - (x[:, 1] + 20) ** 2 - 400, "Hiperbola"),
    (lambda x: x[:, 1] - 20 * x[:, 0] ** 2 - 5 * x[:, 0] + 3, "Parábola"),
]

print("Apartado 2.c")

for c, t in classifiers:
    print("\tGráfico de ", t)
    scatter(x, y, c, t)

    print("\tCorrectos para: " + str(t) + " " + str(get_results(x, y, c)) + "%")
