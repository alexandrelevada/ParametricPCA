#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:47:15 2019

@author: Alexandre L. M. Levada

Python implementation of Parametric PCA under Gaussian hypothesis

For more details see the paper:

Levada, A. L. M. Parametric PCA for unsupervised metrc learning, Pattern Recognition Letters, 2020.
https://doi.org/10.1016/j.patrec.2020.05.011


"""

import warnings
import numpy as np
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Computes the KL-divergence between two Gaussian distributions
def divergenciaKL(mu1, sigma1, mu2, sigma2):
    if sigma1 == 0:     # variância não pode ser zero
        sigma1 = 0.001
    if sigma2 == 0:     # variância não pode ser zero
        sigma2 = 0.001
    dKL1 = np.log(np.sqrt(sigma2)/np.sqrt(sigma1)) + (sigma1 + (mu1 - mu2)**2)/(2*sigma2) - 0.5
    dKL2 = np.log(np.sqrt(sigma1)/np.sqrt(sigma2)) + (sigma2 + (mu2 - mu1)**2)/(2*sigma1) - 0.5
    return 0.5*(dKL1 + dKL2)  # retorna a divergência KL simetrizada

def Bhattacharyya(mu1, sigma1, mu2, sigma2):
    if sigma1 == 0:     # variância não pode ser zero
        sigma1 = 0.001
    if sigma2 == 0:     # variância não pode ser zero
        sigma2 = 0.001
    desvio1, desvio2 = np.sqrt(sigma1), np.sqrt(sigma2)
    BC = np.sqrt((2*desvio1*desvio2)/(sigma1 + sigma2))*np.exp((-1/4)*(((mu1 - mu2)**2)/(sigma1 + sigma2))) 
    BhatDist = -np.log(BC)
    return BhatDist
  
def Hellinger(mu1, sigma1, mu2, sigma2):
    if sigma1 == 0:     # variância não pode ser zero
        sigma1 = 0.001
    if sigma2 == 0:     # variância não pode ser zero
        sigma2 = 0.001
    desvio1, desvio2 = np.sqrt(sigma1), np.sqrt(sigma2)
    BC = np.sqrt((2*desvio1*desvio2)/(sigma1 + sigma2))*np.exp((-1/4)*(((mu1 - mu2)**2)/(sigma1 + sigma2))) 
    HellDist = (1 - BC)
    return HellDist

def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))

    # Sort the eigenvalues
    ordem = v.argsort()

    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]

    # Projection matrix
    Wpca = maiores_autovetores

    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados

def ParametricPCA(dados, k, d, dist):
    # Inicial matrices for storing the means and variances for each patch
    medias = np.zeros((dados.shape[0], dados.shape[1]))
    variancias = np.zeros((dados.shape[0], dados.shape[1]))

    # Creates a KNN graph from the dataset (the value of K affects the results )
    # The second parameter is the number of neighbors K
    esparsa = sknn.kneighbors_graph(dados, k, mode='connectivity', include_self=True)
    A = esparsa.toarray()
    
    # Computes the local means and variances for each patch
    for i in range(dados.shape[0]):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = dados[indices]
        medias[i, :] = amostras.mean(0)
        variancias[i, :] = amostras.std(0)**2
        
    # Compute the average parameters (for the average distribution)       
    mus = medias.mean(0)
    sigmas = variancias.mean(0)

    # Define the surrogate for the covariance matrix
    matriz_final = np.zeros((dados.shape[1], dados.shape[1]))

    # Define the vector of parametric distances
    vetor = np.zeros(dados.shape[1])

    # Computes the surrogate for the covariance matrix
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            if dist == 'KL':
                vetor[j] = divergenciaKL(medias[i,j], variancias[i,j], mus[j], sigmas[j])
            elif dist == 'BHAT':
                vetor[j] = Bhattacharyya(medias[i,j], variancias[i,j], mus[j], sigmas[j])
            else:
                vetor[j] = Hellinger(medias[i,j], variancias[i,j], mus[j], sigmas[j])
            matriz_final = matriz_final + np.outer(vetor, vetor)
            
    matriz_final = matriz_final/dados.shape[0]
    
    # Eigenvalues and eigenvectors of the surrogate matrix
    v, w = np.linalg.eig(matriz_final)

    # Sort the eigenvalues
    ordem = v.argsort()

    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]

    # Projection matrix
    Wpca = maiores_autovetores

    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


def main():

    X = skdata.load_iris()
    dados = X['data']
    target = X['target']
    dados = preprocessing.scale(dados)
    
    # Extract features
    novos_dados_ppca = ParametricPCA(dados, 20, 2, 'KL')
    novos_dados_pca = myPCA(dados, 2)

    # Parametric PCA
    X_train, X_test, y_train, y_test = train_test_split(novos_dados_ppca.real.T, target, test_size=.4, random_state=42)
    
    print()
    print('Results for Parametric PCA')
    print()

     # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    s = neigh.score(X_test, y_test)
    print('KNN accuracy: ', s)

    # SVM Classification
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    s = svm.score(X_test, y_test)
    print('SVM accuracy: ', s)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    s = nb.score(X_test, y_test)
    print('NB accuracy: ', s)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    s = dt.score(X_test, y_test)
    print('DT accuracy: ', s)

    # # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    s = qda.score(X_test, y_test)
    print('QDA accuracy: ', s)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    s = mpl.score(X_test, y_test)
    print('MPL accuracy: ', s)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    s = gpc.score(X_test, y_test)
    print('GPC accuracy: ', s)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    s = rfc.score(X_test, y_test)
    print('RFC accuracy: ', s)

    # Computes the Silhoutte coefficient
    print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_ppca.real.T, target, metric='euclidean'))
    print()

    print('Results for regular PCA')
    print()

    # Regular PCA
    X_train, X_test, y_train, y_test = train_test_split(novos_dados_pca.real.T, target, test_size=.4, random_state=42)

    # KNN Classification
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    print('KNN accuracy: ', neigh.score(X_test, y_test))

    # SVM Classification
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    s = svm.score(X_test, y_test)
    print('SVM accuracy: ', s)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    s = nb.score(X_test, y_test)
    print('NB accuracy: ', s)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    s = dt.score(X_test, y_test)
    print('DT accuracy: ', s)

    # # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    s = qda.score(X_test, y_test)
    print('QDA accuracy: ', s)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    s = mpl.score(X_test, y_test)
    print('MPL accuracy: ', s)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    s = gpc.score(X_test, y_test)
    print('GPC accuracy: ', s)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    s = rfc.score(X_test, y_test)
    print('RFC accuracy: ', s)

    # Computes the Silhoutte coefficient
    print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_pca.real.T, target, metric='euclidean'))
    print()


if __name__ == "__main__":
    main()    