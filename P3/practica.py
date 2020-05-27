import math
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    plot_roc_curve,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.linear_model import (
    LassoCV,
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
    Perceptron,
    SGDRegressor,
    LinearRegression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier, DummyRegressor

try:
    from rich import print
    from rich.progress import track
except ModuleNotFoundError:
    pass

from itertools import product

from pandas import read_csv


seed = 6

#################################################################
######################## DATA PLOTTING ##########################
#################################################################


def plot_digit(point):
    """ Plots a given datapoint from digits database."""
    b = point.reshape((8, 8))
    fig, ax = plt.subplots()
    ax.matshow(b, cmap=plt.cm.Blues)

    for i in range(8):
        for j in range(8):
            c = b[j, i]
            ax.text(i, j, str(c), va="center", ha="center")

    plt.show()


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


def show_confusion_matrix(y_real, y_pred):
    """Muestra matriz de confusión."""
    mat = confusion_matrix(y_real, y_pred)
    mat = 100 * mat.astype("float64") / mat.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap="Purples")
    ax.set(
        title="Matriz de confusión",
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Etiqueta real",
        ylabel="Etiqueta predicha",
    )

    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                "{:.0f}%".format(mat[i, j]),
                ha="center",
                va="center",
                color="black" if mat[i, j] < 50 else "white",
            )

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
        ax.annotate(
            int(label),
            centroid,
            size=14,
            weight="bold",
            color="white",
            backgroundcolor="black",
        )

    # Muestra título
    if title is not None:
        plt.title(title)
    plt.show()


def scatter_with_TSNE(X, y, data_preprocess):
    prep_data = data_preprocess.fit_transform(X)
    X_new = TSNE(n_components=2).fit_transform(prep_data)
    scatter(X_new, y)


################################################################
##################### MODELS and CALLS #########################
################################################################


def kfold_models(models, X, y, seed, stratified=True):

    best_score = 0

    print("\tLos modelos que se van a considerar son: ")
    for (name, _) in models:
        print("\t\t", name)
    print("\n")

    if stratified:
        kfold = StratifiedKFold(random_state=seed, shuffle=True)
    else:
        kfold = KFold(random_state=seed, shuffle=True)

    for (name, model) in models:

        print("\t--> {} <--".format(name))

        score = np.mean(cross_val_score(model, X, y, cv=kfold, n_jobs=-1))

        if best_score < score:
            best_score = score
            best_model = model

        # Mostramos los resultados
        print("\tScore en K-fold: {:.3f}".format(score))
        print("\n")

    print("\n\tMejor modelo: ", end="")
    print(best_model)
    return best_model


def main_clasificacion():

    print("CLASIFICACIÓN")
    print("\tCargando datos... ", end="")
    digits_tra = np.loadtxt("datos/optdigits.tra", delimiter=",")
    digits_tra_x = digits_tra[:, :-1]
    digits_tra_y = digits_tra[:, -1]
    print("Hecho")

    # Creamos el pipeline que se encargará del preprocesado
    data_preprocess = Pipeline(
        [
            ("PCA", PCA(n_components=0.95)),
            ("Polynomial", PolynomialFeatures(degree=2)),
            ("Scaler", StandardScaler()),
        ]
    )

    # Ajustamos el pipeline a los datos  y lo aplicamos a ambos conjuntos
    print("\tAplicando preprocesado a los datos... ", end="")
    X_train = data_preprocess.fit_transform(digits_tra_x, digits_tra_y)
    y_train = digits_tra_y
    print("Hecho")

    # show_preprocess_correlation_matrix(digits_tra_x, data_preprocess)
    # print("Calculando gráfico 2D utilizando TSNE...", end="")
    # scatter_with_TSNE(digits_tra_x, digits_tra_y, data_preprocess)
    # print("Hecho")

    # Vector de modelos a considerar
    models = []
    models += [
        (
            "Logistic regresion: C={}, Multi_class={}".format(c, mc),
            LogisticRegression(
                C=c, multi_class=mc, penalty="l2", max_iter=1000, random_state=seed
            ),
        )
        for c in np.logspace(-2, 2, 3)
        for mc in ["ovr", "multinomial"]
    ]
    models += [
        (
            "RidgeClassifier: \u03B1={}".format(a),
            RidgeClassifier(alpha=a, random_state=seed),
        )
        for a in np.logspace(-2, 2, 3)
    ]
    models += [
        (
            "Perceptron: \u03B1={}".format(a),
            Perceptron(alpha=a, penalty="l2", random_state=seed),
        )
        for a in np.logspace(-7, -3, 3)
    ]

    best_model = kfold_models(models, X_train, y_train, seed)

    best_model.fit(X_train, digits_tra_y)

    digits_test = np.loadtxt("datos/optdigits.tes", delimiter=",")
    digits_test_x = digits_test[:, :-1]
    digits_test_y = digits_test[:, -1]
    X_test = data_preprocess.transform(digits_test_x)

    print(
        "\tAccuracy en test: {:.3f}%".format(
            100 * best_model.score(X_test, digits_test_y)
        )
    )

    # show_confusion_matrix(digits_test_y, best_model.predict(X_test))


def communities_dataset():
    attrib = read_csv("datos/communities.atrib", delim_whitespace=True)
    data = read_csv("datos/communities.data", names=attrib["attributes"])

    # Remove non-predictive features

    # state: US state (by number) - not counted as predictive above, but if considered,
    #        should be considered nominal (nominal)
    # county: numeric code for county - not predictive, and many missing values (numeric)
    # community: numeric code for community - not predictive and many missing values (numeric)
    # communityname: community name - not predictive - for information only (string)
    # fold: fold number for non-random 10 fold cross validation, potentially useful for
    #       debugging, paired tests - not predictive (numeric)
    data = data.drop(
        columns=["state", "county", "community", "communityname", "fold"], axis=1
    )

    data = data.replace("?", np.nan)
    feat_miss = data.columns[data.isnull().any()]

    # OtherPerCap has only one missing value and will be filled by a mean value using Imputer from sklearn.preprocessing. The others features present many missing values and will be removed from the data set.
    data["OtherPerCap"].fillna(data["OtherPerCap"].notna().mean(), inplace=True)
    feat_miss = data.columns[data.isnull().any()]

    data.dropna(axis=1, inplace=True)

    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values

    return train_test_split(X, y, test_size=0.25, random_state=seed)


def main_regresion():

    X_train, X_test, y_train, y_test = communities_dataset()

    preprocesado = Pipeline(
        [
            ("PCA", PCA(n_components=0.95)),
            # ("Polynomial", PolynomialFeatures(degree=2)),
            ("Scaling", StandardScaler()),
        ]
    )

    X_train = preprocesado.fit_transform(X_train, y_train)

    # Vector de modelos a considerar
    models = [("LinearRegression", LinearRegression(fit_intercept=True))]

    models += [
        (
            "SGDRegressor: \u03B1={}".format(a),
            SGDRegressor(
                loss="squared_loss",
                penalty="l2",
                max_iter=1000,
                tol=1e-5,
                alpha=a,
                random_state=seed,
            ),
        )
        for a in np.logspace(-7, -3, 3)
    ]

    best_model = kfold_models(models, X_train, y_train, seed, stratified=False)

    best_model.fit(X_train, y_train)
    X_test = preprocesado.transform(X_test)
    y_pred = best_model.predict(X_test)

    print("R^2: {:.4f}".format(r2_score(y_test, y_pred)))
    print("MSE: {:.4f}".format(mean_squared_error(y_test, y_pred)))
    print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred)))


# main_clasificacion()
main_regresion()
