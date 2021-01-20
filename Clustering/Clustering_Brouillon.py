#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:36:41 2021

@author: marielb
"""


# Importation des librairies

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
import pandas as pd
import time


# Ouverture de l'image

im = Image.open(
    "/Users/marielb/Documents/COURS_3A/IODAA/PFR/DetectionRangs/images/rang_OTSU_NL_S.jpg")


# Conversion de l'image

img_array = np.array(im)
X = []
Y = []
mat_coord = np.argwhere(img_array[:, :, 0] == 255)
# Coordonnées des pixels des plantes
X, Y = np.where(img_array[:, :, 0] == 255)


# DBSCAN : obtention des clusters de rangs

mat_clustered = DBSCAN(eps=70, min_samples=100).fit(mat_coord)
# Représentation graphique du résultat de clustering par DBSCAN
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(
    Y, X, c=mat_clustered.labels_, s=1, cmap='Paired', marker='+')
plt.show()
fig.show()

# Passage au format dataframe de la librairie Panda
dataframe_coord = pd.DataFrame(mat_coord)  # Données de l'image
dataframe_coord = dataframe_coord.rename(columns={0: 'X', 1: 'Y'})
label = pd.DataFrame(mat_clustered.labels_)  # Résultats du clustering
label = label.rename(columns={0: 'label'})
dataframe_coord = pd.concat([dataframe_coord, label], axis=1)
# Labellisation de chaque cluster de l'image dans les clusters de rangs


# Fuzzy clustering : obtention des clusters de plantes

labels_rangs = np.unique(dataframe_coord[['label']].to_numpy())
# Récupère les étiquettes des rangs sur les pixels
rangs_coord = []
# Liste de np.array contenant tous les pixels appartenant au ième rang
rangs_clustered = []
fpcs = []  # Mesure de qualité du clustering
e = 0.005  # Pas, pour la différence de partitionnement entre deux itérations
m_p = 2   # Paramètre de flou, puissance appliquée à la matrice d'appartenance
historique_clusters = [[], []]  # Nombre de cluster initialement et finalement
clusters_plantes = []

start = time.time()
# Pour chaque rang détecté par le DBSCAN
for i in labels_rangs:
    rang = dataframe_coord[
        dataframe_coord["label"] == i][['X', 'Y']].to_numpy().T
    # Transformation du dataframe en un np.array à 2 lignes et un nombre de
    # colonnes correspondant au nombre de pixels du rang
    rangs_coord.append(rang)

    # Le nombre de clusters d'un rang est environ proportionnel à son nombre de
    # pixels. Ce coefficient est une moyenne calculée empiriquement.
    nb_clusters = int(len(rang[0])/390)
    historique_clusters[0].append(nb_clusters)
    if nb_clusters == 0:  # S'il y a trop peu de pixels
        nb_clusters = 1

    # Fonction c-means de la librairie scikit fuzzy
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(rang, c=nb_clusters, m=m_p,
                                             error=e, maxiter=2000)
    fpcs.append(fpc)  # Mesure de qualité du clustering flou

    # Coloration de chaque point des données selon son cluster le plus probable
    cluster = np.argmax(u, axis=0)  # Equivalent au label du DBSCAN
    clusters_plantes.append(cluster)
    historique_clusters[1].append(len(u))

    fig2 = plt.figure()
    plt.scatter(rang[0], rang[1], c=cluster, cmap='Paired', s=1)
    ax = fig2.add_subplot(111)
    plt.title(
        "nombre de pixel = {}, nombre de cluster initial = {}, final = {}"
        .format(len(rang[0]), nb_clusters, len(u)))

    # Centre de chaque cluster signalé par une croix
    Xcluster = []
    Ycluster = []
    for pt in cntr:
        Xcluster.append(pt[0])
        Ycluster.append(pt[1])

    plt.scatter(Xcluster, Ycluster, s=40, marker='+', color='red')
    plt.title(
        'nb_clusters = {}; FPC = {}; erreur = {}; puissance fuzzy = {}'.format(
            nb_clusters, fpc, e, m_p))
    plt.show()
    fig2.show()
end = time.time()
print(end - start)