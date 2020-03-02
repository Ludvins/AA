#!/usr/bin/env python
# coding: utf-8

# Imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Define pandas DataFrame to work with.
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']

#####################################################
##################### Exercise 1 ####################
#####################################################

def ex1():

    scatter = plt.scatter(
        x = iris_df[iris.feature_names[2]],
        y = iris_df[iris.feature_names[3]],
        c = iris_df['species'],
        cmap = ListedColormap(['r', 'g', 'b'])
    )

    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    plt.legend(
        handles = scatter.legend_elements()[0],
        labels = [iris.target_names[0], iris.target_names[1], iris.target_names[2]],
        loc = 'lower right'
    )
    plt.show()

#####################################################
##################### Exercise 2 ####################
#####################################################

def ex2():
    
    # Use train_test_split from sklearn to split the set.
    # The stratify set is used to maintain diversity over the sets.
    # Shuffle makes the splits be random.
    train, test = train_test_split(iris_df, shuffle = True, stratify = iris_df['species'], test_size = 0.2)
    print(train.shape)
    print(test.shape)
    

    # Let's check if the proportion of each class maintains.
    print("Recuento original de cada clase:") 
    print(iris_df['species'].value_counts())
    print("Recuento del conjunto Train: ")
    print(train['species'].value_counts())
    print("Recuento del conjunto Test: ")
    print(test['species'].value_counts())

#####################################################
##################### Exercise 3 ####################
#####################################################

def ex3():
    
    # Define all needed arrays.
    values = [i/99*2*np.pi for i in range(0,100)]
    sin = np.sin(values)
    cos = np.cos(values)
    sin_plus_cos = sin + cos


    # Make a scatter for each array and show the plotting.
    plt.plot(values, sin, 'k--', label = "sin")
    plt.plot(values, cos, 'b--', label = "cos")
    plt.plot(values, sin_plus_cos, 'r--' ,label = "sin+cos")
    plt.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    print("Ejercicio 1")
    ex1()
    print("Ejercicio 2")
    ex2()
    print("Ejercicio 3")
    ex3()
