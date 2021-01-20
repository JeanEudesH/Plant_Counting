# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:30:47 2021

@author: Le groupe tournesol
"""

# Libraries importation
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
import skfuzzy as fuzz

# Image Opening
path = ""  # path of the OTSU image
img = Image.open(path)


def DBSCAN_clustering(img, eps_, min_samples_):
    """
    - Extract white pixel (intensity of 255) from
    the image matrix.
    - Clustering coordinates using DBSCAN algorithm
    - return a dataframe with coordinante of each plants
    pixel and its cluster label
    """
    img_array = np.array(img)
    mat_coord = np.argwhere(
        img_array[:, :, 0] == 255
    )  # extraction of white pixels coordinates
    mat_clustered = DBSCAN(eps=eps_, min_samples=min_samples_).fit(
        mat_coord
    )  # clustering using DBSCAN
    dataframe_coord = pd.DataFrame(mat_coord)
    dataframe_coord = dataframe_coord.rename(columns={0: "X", 1: "Y"})
    label = pd.DataFrame(mat_clustered.labels_)
    label = label.rename(columns={0: "label"})
    dataframe_coord = pd.concat(
        [dataframe_coord, label], axis=1
    )  # Dataframe gathering each plants pixel its label
    return dataframe_coord


def Fuzzy_clustering(dataframe_coord):
    """"""
    labels_rangs = np.unique(
        dataframe_coord[["label"]].to_numpy()
    )  # Get labels from plants pixels
    rangs_coord = []  # empty list of np array containing plants pixels from one rang
    # rangs_clustered = []
    fpcs = (
        []
    )  # Mesure de qualité du clustering, variable qui permet d'avoir une idée de la qualité du clustering , porhce de 1 c'est bien
    e = 0.005  # Pas, pour la différence de partitionnement entre deux itérations, a voir si on le met en parametre
    m_p = 2  # Paramètre de flou, puissance appliquée à la matrice d'appartenance, souvent autour de 2, pour le moment en dur
    historique_clusters = [
        [],
        [],
    ]  # Nombre de cluster initialement et finalement, permet de comparer la différence de cluster entre ce que l'on a mis et ce qui est finalement retourné.
    clusters_plantes = []  # s
    return
