# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:30:47 2021

@author: Le groupe tournesol (the best)

"""

# Import of librairies
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
import sys
import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# pip install -U scikit-fuzzy
import skfuzzy as fuzz

# If import does work, use the following lines
# os.chdir("../Utility/")
# import general_IO as gIO

# else
if "/home/fort/Documents/APT 3A/Cours/Ekinocs/Plant_Counting" not in sys.path:
    sys.path.append("/home/fort/Documents/APT 3A/Cours/Ekinocs/Plant_Counting")

os.chdir("/home/fort/Documents/APT 3A/Cours/Ekinocs/Plant_Counting/Utility")
import Utility.general_IO as gIO


def DBSCAN_clustering(img, epsilon, min_point):
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

    epsilon : INTEGER
        Two points are considered neighbors if the distance between the two
        points is below the threshold epsilon.

    min_point : INTEGER
        The minimum number of neighbors a given point should have in order to
        be classified as a core point.


    Returns
    -------
    dataframe_coord : PANDAS DATAFRAME
        Dataframe with the X and Y coordinates of the plants' pixels
        and their cluster's label.

    """

    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array[:, :, 0] == 255)

    # clustering using DBSCAN
    mat_clustered = DBSCAN(eps=epsilon, min_samples=min_point).fit(mat_coord)

    # Panda dataframe
    dataframe_coord = pd.DataFrame(mat_coord)
    dataframe_coord = dataframe_coord.rename(columns={0: "X", 1: "Y"})
    label = pd.DataFrame(mat_clustered.labels_)
    label = label.rename(columns={0: "label"})

    # Dataframe gathering each plants pixel its label
    dataframe_coord = pd.concat([dataframe_coord, label], axis=1)
    return dataframe_coord


def Plants_Detection(dataframe_coord, e, max_iter, m_p, threshold, image):
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

    e : FLOAT
        Stopping criterion for the partitionning between two iterations,
        set at 0.005

    max_iter : INTEGER
        maximum iteration number, set at 100

    m_i : INTEGER
        Fuzzy parameters, power apply to U (d'appartenance) matrix. It is
        often set at 2.

    Threshold : INTEGER
        Threshod in order to determine if there are enough pixels in the result
        of the DBSCAN_clustering to considere a cluster as a row


    Returns
    -------
    JSON_final : JSON file that can be used as a grid to initialize the agents
        of a multiple agents system. Its size is the number of rows and the
        size of a row is the number of plants (centroïd coordinates).
    """
    # Number of rows and their label
    label_row = np.unique(dataframe_coord[["label"]].to_numpy())

    # Quality metrics of the Fuzzy clustering method, include in the interval
    # [0,1], 1 representing a good result.
    # fpcs = []

    # Historic_cluster save for each row the number of estimate clusters
    # and the final number of clusters
    historic_cluster = [[], []]

    # JSON_final contain final plants positions to initialize the MAS.
    # Each list is a row and contain couple of coordinates representing plants.
    JSON_final = []

    # Centers of all plants clusters coordinates
    XY = [[], []]

    # For each rows, do the plants detection.
    for row in label_row:
        # row_pixels is a matrix with pixels coordinates belonging to a row.
        print(row)
        row_pixels = (
            dataframe_coord[dataframe_coord["label"] == row][["X", "Y"]].to_numpy().T
        )

        # If the row is really a row.
        if Threshold_Pixels_Row(row_pixels, threshold) is False:
            dataframe_coord.drop(
                dataframe_coord[dataframe_coord["label"] == row].index, inplace=True
            )
        else:
            # Determination of the initial estimating number of clusters,
            # and adding to historic_cluster
            estimate_nb_clusters = Automatic_Cluster_Number(row_pixels)
            historic_cluster[0].append(estimate_nb_clusters)

            # Clustering using Fuzzy_Clustering, and adding to historic_cluster
            results_fuzzy_clustering, final_nb_clusters = Fuzzy_Clustering(
                row_pixels, estimate_nb_clusters, e, m_p, max_iter
            )
            historic_cluster[1].append(final_nb_clusters)

            for pt in results_fuzzy_clustering:
                XY[0].append(pt[0])
                XY[1].append(pt[1])

            # Append to the final JSON positions of plants
            # for the considered row.
            JSON_final.append(results_fuzzy_clustering)

    # Plot
    Plot(dataframe_coord, XY, image)
    return JSON_final


def Threshold_Pixels_Row(row_pixels, threshold):
    """
    Determine if there are enough pixels in the result of the
    DBSCAN_clustering function. Define if a cluster can be considered
    as a row according to a threshold pixels.

    Parameters
    ----------
    row_pixels : List
        List of 2 lists, the values of X and Y, respectively, coordinates of
        pixels representing plants in the row.

    Return
    -------
    isRow : Boolean
        Boolean indicating if there are enough pixel in a row to count at least
        a plant. A VERIFIER A PARTIR DE COMBIEN DE PLANTES DANS LE RANG ON
        CONSIDERE OK POUR INITIALISATION D'UN AGENT PLANTE

    """
    # There are enough pixels.
    if len(row_pixels[0]) > threshold:
        return True
    else:
        return False


def Automatic_Cluster_Number(row_pixels, _RAs_group_size):
    """
    Objective : to estimate the number of clusters to initialize the fuzzy
    clustering function. To do so we divide the number of pixels in a row by
    an estimation of the surface of one plant : 40% of a lenght of a plant agent squared.

    Parameters
    ----------
    row_pixels : list of np.array
        Values of X and Y, the coordinates of the pixels in a row.

    Returns
    -------
    Estimated_nb_clusters : Integer
        An estimation of the number of clusters (plants) in a row.
    """

    estimated_nb_clusters = int(
        len(row_pixels[0]) / (_RAs_group_size * _RAs_group_size * 0.4)
    )

    # If too few pixels, supplementary security
    if estimated_nb_clusters == 0:
        estimated_nb_clusters = 1

    return estimated_nb_clusters


def Fuzzy_Clustering(row_pixels, estimate_nb_clusters, e, m_p, max_i):
    """
    Apply the Fuzzy clustering algorithm in order to determine the number
    of clusters, ie the number of plants in a row. Take as an input pixels
    coordinate for on row : row_pixels, the initial number of cluster :
    estimate_nb_clusters, and parameters e, m_p, max_i required for
    cmeans algorithm.

    Returns
    -------
    Return information about plants positions for one row. Its also
    possible to return other parameters like fpc quality score.

    position_cluster_center : LIST
        List of coordinate couple representing center of clusters
        position ie the position of each plant.

    final_nb_clusters : INTEGER
        The final number of cluster, ie of plants, in a row.


    """
    position_cluster_center = []

    centres, u, u0, d, jm, p, fpc = fuzz.cmeans(
        row_pixels, c=estimate_nb_clusters, m=m_p, error=e, maxiter=max_i, seed=0
    )

    final_nb_clusters = len(u)

    for position in centres:
        position_cluster_center.append([int(position[0]), int(position[1])])

    return position_cluster_center, final_nb_clusters


def Plot(mat_coord, centresCoordinates, image):
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111)
    label_cluster = np.unique(mat_coord[["label"]].to_numpy())
    txts = []
    for i in label_cluster:
        xtext = np.median(mat_coord[mat_coord["label"] == i]["X"])
        ytext = np.median(mat_coord[mat_coord["label"] == i]["Y"])
        txt = ax.text(ytext, xtext, str(i))
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
        txts.append(txt)
    scatter_row = ax.scatter(
        mat_coord["Y"].tolist(),
        mat_coord["X"].tolist(),
        c=mat_coord["label"].tolist(),
        s=0.5,
        cmap="Paired",
    )
    scatter_plant = plt.scatter(
        centresCoordinates[1], centresCoordinates[0], s=10, marker="+", color="k"
    )
    # plt.show()
    fig.savefig("/home/fort/Bureau/results/" + image.split(".")[0] + "cluster.png")
    return


def Total_Plant_Position(
    _path_output_root,
    session,
    epsilon,
    min_point,
    e,
    max_iter,
    m_p,
    threshold,
    _RAs_group_size,
):
    # Open the binarized image
    list_image = listdir(
        _path_output_root + "/Output/session_" + str(session) + "/Otsu_R"
    )
    print(list_image)
    for image in list_image:
        print("--- start --- ", image)
        img = Image.open(_path_output_root + "/" + image)
        print("--- DBSCAN ---")
        dataframe_coord = DBSCAN_clustering(img, epsilon, min_point)
        dataframe_coord.drop(
            dataframe_coord[dataframe_coord["label"] == -1].index, inplace=True
        )
        print("--- Plant_detection ---")
        JSON_final = Plants_Detection(
            dataframe_coord, e, max_iter, m_p, threshold, image
        )
        print("--- write_json ---")
        path_JSON_output = (
            _path_output_root
            + "/outputCL/session"
            + str(session)
            + "/Plant_CL_Predictions"
        )
        gIO.check_make_directory(path_JSON_output)
        gIO.WriteJson(
            path_JSON_output,
            "Predicting_initial_plant_" + image.split(".")[0],
            JSON_final,
        )

    return


if __name__ == "__main__":
    Total_Plant_Position(
        _path_output_root="/home/fort/Documents/APT 3A/Cours/Ekinocs/Output_General",
        session=1,
        epsilon=10,
        min_point=30,
        e=0.05,
        max_iter=100,
        m_p=2,
        threshold=2000,
        _RAs_group_size=20,
    )
