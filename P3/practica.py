#!/usr/bin/env python3

####################################################################
########################### IMPORTS ################################
####################################################################

import math
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegressionCV, SGDRegressor, Perceptron
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier, DummyRegressor

from rich import print
from rich.progress import track

from itertools import product

##################################################################
######################## LOAD DATA ###############################
##################################################################

def read_datafile(filename, delimiter = ","):
    """Carga datos desde un fichero de texto.
    Argumentos:
    - filename: Nombre del fichero
    - delimiter: El delimitador que separa los datos
    """
    data = np.loadtxt(filename, delimiter=delimiter)
    return data[:, :-1], data[:, -1]

digits_tra_x, digits_tra_y = read_datafile("datos/optdigits.tra")
digits_test_x, digits_test_y = read_datafile("datos/optdigits.tes")

#################################################################
####################### ERROR FUNCTIONS #########################
#################################################################

def estima_error_clasif(clasificador, X_tra, y_tra, X_test, y_test):
  print("Error {} en training: {:.3f}".format(
    clasificador.steps[-1][0], 1 - clasificador.score(X_tra, y_tra)))
  print("Error {} en test: {:.3f}".format(
    clasificador.steps[-1][0], 1 - clasificador.score(X_test, y_test)))

#################################################################
######################## DATA PLOTTING ##########################
#################################################################

def show_preprocess_correlation_matrix(data, preprocess, title=None):
  """Muestra matriz de correlación para datos antes y después del preprocesado."""
  print("Matriz de correlación pre y post procesado (dígitos)")

  fig, axs = plt.subplots(1, 2, figsize=[12.0, 5.8])

  corr_matrix = np.abs(np.corrcoef(data.T))
  im = axs[0].matshow(corr_matrix, cmap="cividis")
  axs[0].title.set_text("Sin preprocesado")

  prep_data = preprocess.fit_transform(data)
  corr_matrix_post = np.abs(np.corrcoef(prep_data.T))
  axs[1].matshow(corr_matrix_post, cmap="cividis")
  axs[1].title.set_text("Con preprocesado")

  if title is not None:
      fig.suptitle(title)
  fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
  plt.show()

def confusion_matrix(y_real, y_pred, tipo):
  """Muestra matriz de confusión."""
  mat = confusion_matrix(y_real, y_pred)
  mat = 100*mat.astype("float64")/mat.sum(axis=1)[:, np.newaxis]
  fig, ax = plt.subplots()
  ax.matshow(mat, cmap="Purples")
  ax.set(title="Matriz de confusión para predictor {}".format(tipo),
         xticks=np.arange(10),
         yticks=np.arange(10),
         xlabel="Etiqueta real",
         ylabel="Etiqueta predicha")

  for i in range(10):
    for j in range(10):
      ax.text(j,
              i,
              "{:.0f}%".format(mat[i, j]),
              ha="center",
              va="center",
              color="black" if mat[i, j] < 50 else "white")

  plt.show()

def scatter(x, y, title=None): 
    """Representa conjunto de puntos 2D clasificados.
    Argumentos posicionales:
    - x: Coordenadas 2D de los puntos
    - y: Etiquetas"""
    
    _, ax = plt.subplots()

    # Establece límites
    xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)
    
    # Pinta puntos
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.8)

    # Pinta etiquetas
    labels = np.unique(y)
    for label in labels:
        centroid = np.mean(x[y == label], axis=0)
        ax.annotate(int(label),
                    centroid,
                    size=14,
                    weight="bold",
                    color="white",
                    backgroundcolor="black")

    # Muestra título
    if title is not None:
        plt.title(title)
    plt.show()

def scatter_with_TSNE(data_preprocess):
    prep_data = data_preprocess.fit_transform(digits_tra_x)
    X_new = TSNE(n_components=2).fit_transform(prep_data)
    scatter(X_new, digits_tra_y)


#################################################################
#################### TESTING FUNCTIONS ##########################
#################################################################


def run_configurations(preprocessors, classifiers):
    for p in list(product(preprocessors, classifiers)):
        
        pipe = Pipeline(p[0] + p[1], verbose = True)
        print("Testeando Configuracion: ")
        print(pipe.steps)
        pipe.fit(digits_tra_x, digits_tra_y)

        
        y_pred_logistic = pipe.predict(digits_test_x)

        estima_error_clasif(pipe, digits_tra_x, digits_tra_y, digits_test_x,
                            digits_test_y)

################################################################
##################### MODELS and CALLS #########################
################################################################

logistic_clasiffier = [("Logistic",
                        LogisticRegressionCV(Cs=10,
                                             max_iter=10000,
                                             penalty="l2",
                                             cv=5,
                                             scoring='accuracy',
                                             fit_intercept=True,
                                             multi_class='multinomial'))]

perceptron_clasiffier = [("Perceptron", Perceptron(penalty="l2", eta0 = 0.1, max_iter = 10000))]

data_preprocess = Pipeline(
    [("Variance Threshold", VarianceThreshold(threshold=0.0)),
     ("Scaler", StandardScaler()),
     ("PCA", PCA(n_components=0.999))])


# run_configurations([data_preprocess.steps], [logistic_clasiffier, perceptron_clasiffier])
best_model = Pipeline(data_preprocess.steps + logistic_clasiffier)

    
# Preprocess data
# show_preprocess_correlation_matrix(digits_tra_x, data_preprocess)
# print("Calculando gráfico 2D utlizando TSNE")
# scatter_with_TSNE(data_preprocess)

# Fitting data
best_model.fit(digits_tra_x, digits_tra_y)
y_pred = best_model.predict(digits_test_x)
# muestra_confusion(digits_test_y, y_pred, "Logístico")
estima_error_clasif(best_model, digits_tra_x, digits_tra_y, digits_test_x, digits_test_y)




