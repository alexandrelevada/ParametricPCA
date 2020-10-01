#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:47:15 2019

@author: Alexandre Levada

Parametric PCA for metric learning
Source code for paper:

Levada, A. L. M. Parametric PCA for unsupervised metric learning, Pattern Recognition Letters, v.135, pp. 425-430, 2020.

"""

import warnings
import numpy as np
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import matplotlib.pyplot as plt
from scipy.special import erf
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
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
        sigma1 = 0.01
    if sigma2 == 0:     # variância não pode ser zero
        sigma2 = 0.01

    dKL1 = np.log(np.sqrt(sigma2)/np.sqrt(sigma1)) + (sigma1 + (mu1 - mu2)**2)/(2*sigma2) - 0.5
    dKL2 = np.log(np.sqrt(sigma1)/np.sqrt(sigma2)) + (sigma2 + (mu2 - mu1)**2)/(2*sigma1) - 0.5

    return 0.5*(dKL1 + dKL2)  # retorna a divergência KL simetrizada


def Bhattacharyya(mu1, sigma1, mu2, sigma2):
    if sigma1 == 0:     # variância não pode ser zero
        sigma1 = 0.01
    if sigma2 == 0:     # variância não pode ser zero
        sigma2 = 0.01

    desvio1, desvio2 = np.sqrt(sigma1), np.sqrt(sigma2)
    BC = np.sqrt((2*desvio1*desvio2)/(sigma1 + sigma2))*np.exp((-1/4)*(((mu1 - mu2)**2)/(sigma1 + sigma2))) 
    BhatDist = -np.log(BC)

    return BhatDist
  

def Hellinger(mu1, sigma1, mu2, sigma2):
    if sigma1 == 0:     # variância não pode ser zero
        sigma1 = 0.01
    if sigma2 == 0:     # variância não pode ser zero
        sigma2 = 0.01

    desvio1, desvio2 = np.sqrt(sigma1), np.sqrt(sigma2)
    BC = np.sqrt((2*desvio1*desvio2)/(sigma1 + sigma2))*np.exp((-1/4)*(((mu1 - mu2)**2)/(sigma1 + sigma2))) 
    HellDist = (1 - BC)
    
    return HellDist


def PCA(dados, d):
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
    
    # Compute the standard deviation for Fisher information based metrics    
    desvios = np.sqrt(variancias)
        
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
            elif dist == 'HELL':
                vetor[j] = Hellinger(medias[i,j], variancias[i,j], mus[j], sigmas[j])                
            else:
                vetor[j] = divergenciaKL(medias[i,j], variancias[i,j], mus[j], sigmas[j])
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


# Analysis of the supervised classification accuracy 
def batch_Parametric_PCA():

    # Datasets
    X = skdata.load_iris()     # K = 95
    #X = skdata.fetch_openml(name='Engine1', version=1) # K = 235
    #X = skdata.fetch_openml(name='prnn_crabs', version=1) # K = 10
    #X = skdata.fetch_openml(name='analcatdata_happiness', version=1) # K = 53
    #X = skdata.fetch_openml(name='mux6', version=1) # K = 105
    #X = skdata.fetch_openml(name='threeOf9', version=1) # K = 385
    #X = skdata.fetch_openml(name='parity5', version=1) # K = 25
    #X = skdata.fetch_openml(name='sa-heart', version=1) # K = 74
    #X = skdata.fetch_openml(name='vertebra-column', version=1) # K = 305
    #X = skdata.fetch_openml(name='breast-tissue', version=2) # K = 5
    #X = skdata.fetch_openml(name='transplant', version=2)  # K = 65
    #X = skdata.fetch_openml(name='hayes-roth', version=2)  # K = 5
    #X = skdata.fetch_openml(name='plasma_retinol', version=2)  # K = 145
    #X = skdata.fetch_openml(name='aids', version=1) # K = 42
    #X = skdata.fetch_openml(name='lupus', version=1) # K = 37
    #X = skdata.fetch_openml(name='pwLinear', version=2)  # K = 135
    #X = skdata.fetch_openml(name='fruitfly', version=2) # K = 120
    #X = skdata.fetch_openml(name='pm10', version=2) # K = 485
    #X = skdata.fetch_openml(name='visualizing_livestock', version=1) # K = 125
    #X = skdata.fetch_openml(name='strikes', version=2)  # K = 130   

    dados = X['data']
    target = X['target']
    
    # Normalize data
    dados = preprocessing.scale(dados)

    n = dados.shape[0]
    m = dados.shape[1]

    print('N = ', n)
    print('M = ', m)

    inicio = 5
    incremento = 5

    lista_k = list(range(inicio, n, incremento))

    acuracias = []
    kappas_medios = []

    for k in lista_k:

        print('K = ', k)
        novos_dados = ParametricPCA(dados, k, 2, 'KL') 

        #%%%%%%%%%%%%%%%%% Parametric PCA
        print('Parametric PCA for supervised classification')
    
        # Split training data
        X_train, X_test, y_train, y_test = train_test_split(novos_dados.real.T, target, test_size=.4, random_state=42)
        acc = []
    
        # KNN
        neigh = KNeighborsClassifier(n_neighbors=7)
        neigh.fit(X_train, y_train) 
        s = neigh.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(neigh.predict(X_test), y_test)
        acc.append(s)
        print('KNN accuracy: ', s)
    
        # SVM
        svm = SVC(gamma='auto')
        svm.fit(X_train, y_train)
        s = svm.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(svm.predict(X_test), y_test)
        acc.append(s)
        print('SVM accuracy: ', s)

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        s = nb.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(nb.predict(X_test), y_test)
        acc.append(s)
        print('NB accuracy: ', s)

        # Decision Tree
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X_train, y_train)
        s = dt.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(dt.predict(X_test), y_test)
        acc.append(s)
        print('DT accuracy: ', s)

        # Quadratic Discriminant 
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        s = qda.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(qda.predict(X_test), y_test)
        acc.append(s)
        print('QDA accuracy: ', s)

        # MPL classifier
        mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
        mpl.fit(X_train, y_train)
        s = mpl.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(mpl.predict(X_test), y_test)
        acc.append(s)
        print('MPL accuracy: ', s)

        # Gaussian Process
        gpc = GaussianProcessClassifier()
        gpc.fit(X_train, y_train)
        s = gpc.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(gpc.predict(X_test), y_test)
        acc.append(s)
        print('GPC accuracy: ', s)

        # Random Forest Classifier
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        s = rfc.score(X_test, y_test)
        kap = metrics.cohen_kappa_score(rfc.predict(X_test), y_test)
        acc.append(s)
        print('RFC accuracy: ', s)
        
        acuracia = sum(acc)/len(acc)
        
        # Computes the Silhoutte coefficient
        print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados.real.T, target, metric='euclidean'))
        print('Average accuracy: ', acuracia)
        print()

        acuracias.append(acuracia)

    print('List of values for K: ', lista_k)
    print('Supervised classification accuracies: ', acuracias)
    acuracias = np.array(acuracias)
    print('Max Acc: ', acuracias.max())
    print('K* = ', lista_k[acuracias.argmax()])
    print()

    plt.figure(1)
    plt.plot(lista_k, acuracias)
    plt.title('Mean accuracies for different values of K (neighborhood)')
    plt.show()

     #%%%%%%%%%%%%% Dimensionality reduction methods
    # PCA
    novos_dados_pca = PCA(dados, 2)     
    # Kernel PCA
    model = KernelPCA(n_components=2, kernel='rbf')   
    novos_dados_kpca = model.fit_transform(dados)
    novos_dados_kpca = novos_dados_kpca.T
    # ISOMAP
    model = Isomap(n_neighbors=20, n_components=2)
    novos_dados_isomap = model.fit_transform(dados)
    novos_dados_isomap = novos_dados_isomap.T
    # LLE
    model = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
    novos_dados_LLE = model.fit_transform(dados)
    novos_dados_LLE = novos_dados_LLE.T
    # Lap. Eig.
    model = SpectralEmbedding(n_neighbors=20, n_components=2)
    novos_dados_Lap = model.fit_transform(dados)
    novos_dados_Lap = novos_dados_Lap.T
    
    #%%%%%%%%%%%%%%%%% PCA 
    print('Results for PCA')
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(novos_dados_pca.real.T, target, test_size=.4, random_state=42)
    acc = []
       
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    s = neigh.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(neigh.predict(X_test), y_test)
    acc.append(s)
    print('KNN accuracy: ', s)
        
    # SVM
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    s = svm.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(svm.predict(X_test), y_test)
    acc.append(s)
    print('SVM accuracy: ', s)
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    s = nb.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(nb.predict(X_test), y_test)
    acc.append(s)
    print('NB accuracy: ', s)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    s = dt.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(dt.predict(X_test), y_test)
    acc.append(s)
    print('DT accuracy: ', s)
    
    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    s = qda.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(qda.predict(X_test), y_test)
    acc.append(s)
    print('QDA accuracy: ', s)
    
    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    s = mpl.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(mpl.predict(X_test), y_test)
    acc.append(s)
    print('MPL accuracy: ', s)
    
    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    s = gpc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(gpc.predict(X_test), y_test)
    acc.append(s)
    print('GPC accuracy: ', s)
    
    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    s = rfc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(rfc.predict(X_test), y_test)
    acc.append(s)
    print('RFC accuracy: ', s)
    
    # Computes the Silhoutte coefficient
    print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_pca.real.T, target, metric='euclidean'))
    print('Average accuracy: ', sum(acc)/len(acc))
    print()
        
    #%%%%%%%%%%%%%%%%% KPCA
    print('Results for KPCA')
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(novos_dados_kpca.real.T, target, test_size=.4, random_state=42)
    acc = []
        
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    s = neigh.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(neigh.predict(X_test), y_test)
    acc.append(s)
    print('KNN accuracy: ', s)
    
    # SVM
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    s = svm.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(svm.predict(X_test), y_test)
    acc.append(s)
    print('SVM accuracy: ', s)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    s = nb.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(nb.predict(X_test), y_test)
    acc.append(s)
    print('NB accuracy: ', s)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    s = dt.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(dt.predict(X_test), y_test)
    acc.append(s)
    print('DT accuracy: ', s)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    s = qda.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(qda.predict(X_test), y_test)
    acc.append(s)
    print('QDA accuracy: ', s)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    s = mpl.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(mpl.predict(X_test), y_test)
    acc.append(s)
    print('MPL accuracy: ', s)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    s = gpc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(gpc.predict(X_test), y_test)
    acc.append(s)
    print('GPC accuracy: ', s)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    s = rfc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(rfc.predict(X_test), y_test)
    acc.append(s)
    print('RFC accuracy: ', s)

    # Computes the Silhoutte coefficient
    print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_kpca.real.T, target, metric='euclidean'))
    print('Average accuracy: ', sum(acc)/len(acc))
    print()
    
    #%%%%%%%%%%%%%%%%% ISOMAP
    print('Results for ISOMAP')
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(novos_dados_isomap.real.T, target, test_size=.4, random_state=42)
    acc = []
    
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    s = neigh.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(neigh.predict(X_test), y_test)
    acc.append(s)
    print('KNN accuracy: ', s)
    
    # SVM
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    s = svm.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(svm.predict(X_test), y_test)
    acc.append(s)
    print('SVM accuracy: ', s)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    s = nb.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(nb.predict(X_test), y_test)
    acc.append(s)
    print('NB accuracy: ', s)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    s = dt.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(dt.predict(X_test), y_test)
    acc.append(s)
    print('DT accuracy: ', s)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    s = qda.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(qda.predict(X_test), y_test)
    acc.append(s)
    print('QDA accuracy: ', s)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    s = mpl.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(mpl.predict(X_test), y_test)
    acc.append(s)
    print('MPL accuracy: ', s)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    s = gpc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(gpc.predict(X_test), y_test)
    acc.append(s)
    print('GPC accuracy: ', s)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    s = rfc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(rfc.predict(X_test), y_test)
    acc.append(s)
    print('RFC accuracy: ', s)

    # Computes the Silhoutte coefficient
    print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_isomap.real.T, target, metric='euclidean'))
    print('Average accuracy: ', sum(acc)/len(acc))
    print()
    
    #%%%%%%%%%%%%%%%%% LLE
    print('Results for LLE')
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(novos_dados_LLE.real.T, target, test_size=.4, random_state=42)
    acc = []
    
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    s = neigh.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(neigh.predict(X_test), y_test)
    acc.append(s)
    print('KNN accuracy: ', s)
    
    # SVM
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    s = svm.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(svm.predict(X_test), y_test)
    acc.append(s)
    print('SVM accuracy: ', s)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    s = nb.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(nb.predict(X_test), y_test)
    acc.append(s)
    print('NB accuracy: ', s)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    s = dt.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(dt.predict(X_test), y_test)
    acc.append(s)
    print('DT accuracy: ', s)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    s = qda.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(qda.predict(X_test), y_test)
    acc.append(s)
    print('QDA accuracy: ', s)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    s = mpl.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(mpl.predict(X_test), y_test)
    acc.append(s)
    print('MPL accuracy: ', s)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    s = gpc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(gpc.predict(X_test), y_test)
    acc.append(s)
    print('GPC accuracy: ', s)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    s = rfc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(rfc.predict(X_test), y_test)
    acc.append(s)
    print('RFC accuracy: ', s)

    # Computes the Silhoutte coefficient
    print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_LLE.real.T, target, metric='euclidean'))
    print('Average accuracy: ', sum(acc)/len(acc))
    print()
    
    #%%%%%%%%%%%%%%%%% Laplacian Eigenmaps
    print('Results for Laplacian Eigenmaps')
    
    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(novos_dados_Lap.real.T, target, test_size=.4, random_state=42)
    acc = []
    
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    s = neigh.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(neigh.predict(X_test), y_test)
    acc.append(s)
    print('KNN accuracy: ', s)
    
    # SVM
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)
    s = svm.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(svm.predict(X_test), y_test)
    acc.append(s)
    print('SVM accuracy: ', s)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    s = nb.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(nb.predict(X_test), y_test)
    acc.append(s)
    print('NB accuracy: ', s)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    s = dt.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(dt.predict(X_test), y_test)
    acc.append(s)
    print('DT accuracy: ', s)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    s = qda.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(qda.predict(X_test), y_test)
    acc.append(s)
    print('QDA accuracy: ', s)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    s = mpl.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(mpl.predict(X_test), y_test)
    acc.append(s)
    print('MPL accuracy: ', s)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    s = gpc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(gpc.predict(X_test), y_test)
    acc.append(s)
    print('GPC accuracy: ', s)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    s = rfc.score(X_test, y_test)
    kap = metrics.cohen_kappa_score(rfc.predict(X_test), y_test)
    acc.append(s)
    print('RFC accuracy: ', s)

    # Computes the Silhoutte coefficient
    print('Silhouette coefficient: ', metrics.silhouette_score(novos_dados_Lap.real.T, target, metric='euclidean'))
    print('Average accuracy: ', sum(acc)/len(acc))
    print()
    
    batch_Parametric_PCA_cluster(X)


# Analisys of the clusters with Silhouette Coefficient
def batch_Parametric_PCA_cluster(X):
    
    dados = X['data']
    target = X['target']

    # Normalize data
    dados = preprocessing.scale(dados)

    n = dados.shape[0]
    m = dados.shape[1]

    print('N = ', n)
    print('M = ', m)

    inicio = 5
    incremento = 5

    lista_k = list(range(inicio, n, incremento))

    clusters = []

    for k in lista_k:

        print('K = ', k)
        novos_dados = ParametricPCA(dados, k, 2, 'KL') 

        #%%%%%%%%%%%%%%%%% PCA-R
        print('Parametric PCA for cluster analysis (Silhouette coefficient)')
    
        # Split training data
        X_train, X_test, y_train, y_test = train_test_split(novos_dados.real.T, target, test_size=.4, random_state=42)

        # Computes the Silhoutte coefficient
        s = metrics.silhouette_score(novos_dados.real.T, target, metric='euclidean')
        print('Silhouette coefficient: ', s)
        print()

        clusters.append(s)

    print('List of values for K: ', lista_k)
    print('Silhouette coefficients: ', clusters)
    clusters = np.array(clusters)
    print('Max SC: ', clusters.max())
    print('K* = ', lista_k[clusters.argmax()])

    plt.figure(1)
    plt.plot(lista_k, clusters)
    plt.title('Silhouette Coefficients for different values of K (neighborhood)')
    plt.show()


#%%%%%%%%%% Main function
    
if __name__ == "__main__":
    batch_Parametric_PCA()
    
