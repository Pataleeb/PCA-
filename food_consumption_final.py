import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as spio
import scipy.sparse.linalg as ll
from numpy.linalg import eigh

import sklearn.preprocessing as skpp
from mpmath.matrices.eigen_symmetric import eighe
from nltk.sem.chat80 import country


class FoodConsumptionPCA:
    def __init__(self, input_path = "data/food-consumption.csv"):
        self.data=pd.read_csv(input_path)
        print(f"Data loaded from {input_path}")
        self.explore_data()

    def explore_data(self):

        print(self.data.head())
        rows,cols=self.data.shape
        print(f"Rows: {rows}, Columns: {cols}")
        print("\nColumns:")
        print(self.data.columns.tolist())

    def country_pca(self):
        country=self.data.iloc[:,1:].values
        m,n=country.shape

        Inew=self.data.iloc[:,0].values

        stdcountry=np.std(country,axis=0)
        country=country @ np.diag(1/stdcountry)
        country=country.T

        ##PCA
        mu = np.mean(country,axis = 1)
        xc = country - mu[:,None]
        C = np.dot(xc, xc.T) / m
        
        K = 2
        S, W =eigh(C)
        S = S[-K:]
        W = W[:,-K:]

        dim1 = np.dot(W[:, 0].T, xc) / math.sqrt(S[0])
        dim2 = np.dot(W[:, 1].T, xc) / math.sqrt(S[1])

        plt.figure(figsize=(12,6))
        for i, label in enumerate(Inew):
            plt.scatter(dim1[i], dim2[i], marker='o', color='orange')
            plt.text(dim1[i] + 0.02, dim2[i], label, fontsize=9)



        plt.title("PCA of Food Consumption by Country")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True)
        plt.tight_layout()


        plt.savefig('country.png', bbox_inches="tight")
        plt.show()

    def food_consumption_pca(self):
        food_data=self.data.iloc[:,1:].values.T
        m,n=food_data.shape
        Inew=self.data.columns[1:].values
        stdfood=np.std(food_data,axis=0)
        food_data=food_data @ np.diag(1/stdfood)
        food_data=food_data.T

        mu = np.mean(food_data,axis = 1)
        xc = food_data - mu[:,None]
        C = np.dot(xc, xc.T) / m

        K = 2
        S, W =eigh(C)
        S = S[-K:]
        W = W[:,-K:]
        dim1 = np.dot(W[:, 0].T, xc) / math.sqrt(S[0])
        dim2 = np.dot(W[:, 1].T, xc) / math.sqrt(S[1])
        plt.figure(figsize=(12,6))
        for i, label in enumerate(Inew):
            plt.scatter(dim1[i], dim2[i], marker='o', color='violet')  # Plot points
            plt.text(dim1[i] + 0.02, dim2[i], label, fontsize=9)  # Add food item name labels

        plt.title("PCA of Food Items by Country")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('food_  item.png', bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    food_consumption = FoodConsumptionPCA(input_path="data/food-consumption.csv")
    food_consumption.country_pca()
    food_consumption.food_consumption_pca()
