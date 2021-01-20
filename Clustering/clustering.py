# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:30:47 2021

@author: Le groupe tournesol (the best)

A FAIRE :

    - Finir les fonctions (duh)
    - Se mettre d'accord sur un seuil minimal de plantes dans un rang
    (==> nombre minimal de pixels à trouver.)
    - Voir la robustesse des fonctions AutomaticNbClusters et
    Threshold_PixelsRang dans types d'images
    (dépend de la taille de l'image, du nombre total de rangs et le nombre
    total de pixels "plantes" )
    - Déterminer quelles variables nous allons utiliser pour vérifier la
    qualité du dénombrement des plantes de l'image.


    - Uniformiser les noms des variables (tout mettre en anglais)
    - Commenter proprement en anglais
    - Docstrings en anglais

"""

# Import of librairies
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
# pip install -U scikit-fuzzy
import skfuzzy as fuzz

# Open the binarized image
path = ""  # path of the OTSU image
img = Image.open('PATH')


def DBSCAN_clustering(img, eps_, min_samples_):
    """
    The objective of this function is to differenciate rows of a field.
    As input, it needs a binarized picture (cf function/package...) taken by an
    unmanned aerial vehicle (uav). The white pixels representing the plants in
    are labelled with the DBSCAN algorithm according to the row they belong to.
    The package Scikit-learn is used to do so.
    The intensity value of the pixels is 255.

    Extraction of the white pixels of an Image object
    - Extract white pixel (intensity of 255) from
    the image matrix.
    - Clustering coordinates using DBSCAN algorithm


    Parameters
    ----------
    img : PIL Image
        Image opened with the PIL package. It is an image of plants cultivated
        in a field.
        The picture can be taken by an unmanned aerial vehicle (uav).

    eps_ : INTEGER
        Density distance for .

    min_samples_ : TYPE
        DESCRIPTION.


    Returns
    -------
    dataframe_coord : Panda Dataframe
        Dataframe with the X and Y coordinates of the plants' pixels
        and their cluster's label

    """

    img_array = np.array(img)
    mat_coord = np.argwhere(
        img_array[:, :, 0] == 255
    )  # extraction of white pixels coordinates
    mat_clustered = DBSCAN(eps=eps_, min_samples=min_samples_).fit(
        mat_coord
    )  # clustering using DBSCAN
    # Panda dataframe
    dataframe_coord = pd.DataFrame(mat_coord)
    dataframe_coord = dataframe_coord.rename(columns={0: "X", 1: "Y"})
    label = pd.DataFrame(mat_clustered.labels_)
    label = label.rename(columns={0: "label"})
    dataframe_coord = pd.concat(
        [dataframe_coord, label], axis=1
    )  # Dataframe gathering each plants pixel its label
    return dataframe_coord


def PlantsDetection(dataframe_coord):
    """
    The objective of this function is to differenciate the plants in each row
    of the binarized picture. It is based on predefined rows, the corresponding
    pixels labelled.
    It calls for the Fuzzy_Clustering function.
    The final result cen be used to initialize a grid for a multiple agents
    system to count more precisely the number of plants in the picture.


    Parameters
    ----------
    dataframe_coord : Panda dataframe
        Dataframe with the X and Y coordinates and as a third column, the row
        the pixel belongs to.
        It is obtained with the function DBSCAN_clustering.


    Returns
    -------
    JSON_final : JSON file that can be used as a grid to initialize the agents
        of a multiple agents system. Its size is the number of rows and the
        size of a row is the number of plants (centroïd coordinates).
    """

    labels_rangs = np.unique(
        dataframe_coord[["label"]].to_numpy()
    )  # Number of rows and their label
    rangs_coord = []
    # empty list of np.array containing plants pixels coordinates from one rang
    # rangs_clustered = []
    fpcs = []
    # Mesure de qualité du clustering, variable qui permet d'avoir une idée de
    # la qualité du clustering , comprise entre 0 et 1, bien quand proche de 1

    # Paramètres de la fonction c-means à passer en paramètre par défaut ici
    e = 0.005  # Pas, pour la différence de partitionnement entre deux
    # itérations, a voir si on le met en parametre
    m_i = 2000  # Nombre maximal d'itération
    m_p = 2
    # Paramètre de flou, puissance appliquée à la matrice d'appartenance,
    # souvent autour de 2, pour le moment en dur

    historique_clusters = [[], []]
    # Nombre de cluster initialement et finalement, permet de comparer
    # la différence de cluster entre ce que l'on a mis et ce qui est finalement
    # retourné.

    JSON_final = []
    # liste de liste (la première correspond à un rang et à l'intérieur
    # chaque couple aux coordonnées du centroïde d'un cluster ie plante)

    for i in labels_rangs:
        rang = dataframe_coord[
            dataframe_coord["label"] == i][['X', 'Y']].to_numpy().T

        if Threshold_PixelsRang() is True:
            nb_clusters = AutomaticNbClusters(rang)
            historique_clusters[0].append(nb_clusters)
            resultat_fuzzyClustering, nbFinalClusters = Fuzzy_Clustering(rang,
                                                                         nb_clusters, e, m_p, m_i)
            historique_clusters[1].append(nbFinalClusters)

            JSON_final.append([resultat_fuzzyClustering])

    return JSON_final


def Threshold_PixelsRang(rang, seuil):
    """
    Determine if there are enough pixels in the result of the DBSCAN_clustering
    function. It can be considered as a row according to a threshold pixels.

    Parameters
    ----------
    rang : List
        List of 2 lists, the values of X and Y, respectively, coordinates of
        the pixels representing the plants in the row.

    Return
    -------
    isRow : Boolean
        Boolean indicating if there are enough pixel in a row to count at least
        a plant. A VERIFIER A PARTIR DE COMBIEN DE PLANTES DANS LE RANG ON
        CONSIDERE OK POUR INITIALISATION D'UN AGENT PLANTE

    """

    if len(rang[0]) > seuil:
        return True
    else:
        return False


def AutomaticNbClusters(rang):
    """
    Nombre de pixels dans chaque rangs en entrée
    (liste de X et Y, les coordonnées de chaque pixel du rang)
    Returns
    -------
    Nombre de clusters dans le rang

    """

    nb_clusters = int(len(rang[0])/390)

    if nb_clusters == 0:  # If too few pixels, supplementary security
        nb_clusters = 1

    return nb_clusters


def Fuzzy_Clustering(rang, nb_clusters, e, m_p, m_i):
    """
    En entrée, rang est la liste des coordonnées des pixels du rang
    (np.array et 2 listes de longueur du nombre de pixels)

    Fuzzy clustering pour un rang défini par DBSCAN (ici)
    Returns
    -------
    None.

    """
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(rang, c=nb_clusters, m=m_p,
                                             error=e, maxiter=m_i)
    # Attribution d'une étiquette à chaque point des données selon son cluster
    # le plus probable (cf matrice u)

    nbClusterFinal = len(u)

    PosCentCcluster = []
    for pt in cntr:
        PosCentCcluster.append([pt[0], pt[1]])

    # Ajouter d'autres paramètres pour vérifier
    # la qualité du clustering comme fpc ou jm
    return PosCentCcluster, nbClusterFinal
